"""Self-contained SQLite reports and queries; no plotting or model dependencies."""

from contextlib import AbstractContextManager
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE run (id INTEGER PRIMARY KEY CHECK(id=1), metadata TEXT NOT NULL,
                  status TEXT NOT NULL, error TEXT);
CREATE TABLE pair (
    id INTEGER PRIMARY KEY, source_path TEXT NOT NULL, source INTEGER NOT NULL,
    target INTEGER NOT NULL, sample INTEGER NOT NULL, reference_path TEXT,
    original BLOB NOT NULL, generated BLOB NOT NULL,
    original_logits BLOB NOT NULL, original_probabilities BLOB NOT NULL,
    generated_logits BLOB NOT NULL, generated_probabilities BLOB NOT NULL);
CREATE TABLE explanation (
    id INTEGER PRIMARY KEY, pair_id INTEGER NOT NULL REFERENCES pair(id),
    method TEXT NOT NULL, attribution BLOB NOT NULL, optimal_index INTEGER NOT NULL,
    optimal_delta REAL NOT NULL, auc REAL NOT NULL, UNIQUE(pair_id, method));
CREATE TABLE threshold (
    explanation_id INTEGER NOT NULL REFERENCES explanation(id),
    idx INTEGER NOT NULL, area REAL NOT NULL, cutoff REAL NOT NULL,
    mask BLOB NOT NULL, image BLOB NOT NULL, logits BLOB NOT NULL,
    probabilities BLOB NOT NULL, delta REAL NOT NULL,
    PRIMARY KEY(explanation_id, idx));
CREATE INDEX ranking ON explanation(method, optimal_delta DESC);
CREATE INDEX class_pair ON pair(source, target);
"""


def pack(array) -> bytes:
    """Encode a typed NumPy array without pickle."""
    stream = BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return zlib.compress(stream.getvalue())


def unpack(blob: bytes) -> np.ndarray:
    """Decode a compressed array without executing serialized objects."""
    return np.load(BytesIO(zlib.decompress(blob)), allow_pickle=False)


class Report(AbstractContextManager):
    """Query a report read-only, or create it explicitly through Report.create."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.connection = sqlite3.connect(
            self.path.resolve().as_uri() + "?mode=ro", uri=True
        )
        self.connection.row_factory = sqlite3.Row
        if (
            self.connection.execute("PRAGMA user_version").fetchone()[0]
            != SCHEMA_VERSION
        ):
            self.close()
            raise ValueError("Unsupported report schema version")

    @classmethod
    def create(cls, path, metadata, overwrite=False):
        """Create an empty report, refusing to overwrite unless explicitly requested."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Report already exists: {path}; set output.overwrite: true to replace it"
                )
            path.unlink()
        report = cls.__new__(cls)
        report.path = path
        report.connection = sqlite3.connect(path)
        report.connection.row_factory = sqlite3.Row
        report.connection.execute("PRAGMA foreign_keys=ON")
        report.connection.executescript(SCHEMA)
        report.connection.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
        with report.connection:
            report.connection.execute(
                "INSERT INTO run VALUES (1, ?, 'running', NULL)",
                (json.dumps(metadata),),
            )
        return report

    @property
    def metadata(self):
        """Resolved configuration, class labels, model identity, and inferred dimensions."""
        return json.loads(
            self.connection.execute("SELECT metadata FROM run").fetchone()[0]
        )

    @property
    def status(self):
        """Return run completion state and failure details."""
        return dict(self.connection.execute("SELECT status, error FROM run").fetchone())

    def finish(self, error: str | None = None):
        """Record completion or failure without discarding committed explanations."""
        with self.connection:
            self.connection.execute(
                "UPDATE run SET status=?, error=?",
                ("failed" if error is not None else "complete", error),
            )

    def add_pair(
        self,
        source_path,
        source,
        target,
        sample,
        original,
        generated,
        original_logits,
        original_probabilities,
        generated_logits,
        generated_probabilities,
        reference_path=None,
    ) -> int:
        """Store images and predictions once for all methods evaluating the pair."""
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO pair VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(source_path),
                    source,
                    target,
                    sample,
                    str(reference_path) if reference_path else None,
                    pack(original),
                    pack(generated),
                    pack(original_logits),
                    pack(original_probabilities),
                    pack(generated_logits),
                    pack(generated_probabilities),
                ),
            )
        return cursor.lastrowid

    def add_explanation(self, pair_id, method, attribution, result) -> int:
        """Atomically store a complete method sweep, including every image and prediction."""
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO explanation VALUES (NULL,?,?,?,?,?,?)",
                (
                    pair_id,
                    method,
                    pack(attribution),
                    result.optimal_index,
                    float(result.deltas[result.optimal_index]),
                    result.auc,
                ),
            )
            explanation_id = cursor.lastrowid
            self.connection.executemany(
                "INSERT INTO threshold VALUES (?,?,?,?,?,?,?,?,?)",
                [
                    (
                        explanation_id,
                        index,
                        float(area),
                        float(result.cutoffs[index]),
                        pack(result.masks[index]),
                        pack(result.images[index]),
                        pack(result.logits[index]),
                        pack(result.probabilities[index]),
                        float(result.deltas[index]),
                    )
                    for index, area in enumerate(result.areas)
                ],
            )
        return explanation_id

    def methods(self) -> list[str]:
        """List stored attribution methods."""
        return [
            row[0]
            for row in self.connection.execute(
                "SELECT DISTINCT method FROM explanation ORDER BY method"
            )
        ]

    def top(self, method: str, k: int = 5, source=None, target=None) -> list[dict]:
        """Query exemplars by optimal delta, optionally restricting source and target."""
        if k < 1:
            raise ValueError("k must be positive")
        where, parameters = ["e.method=?"], [method]
        for field, value in (("source", source), ("target", target)):
            if value is not None:
                where.append(f"p.{field}=?")
                parameters.append(value)
        query = (
            "SELECT e.*, p.source_path, p.source, p.target, p.sample FROM explanation e "
            "JOIN pair p ON p.id=e.pair_id WHERE "
            + " AND ".join(where)
            + " ORDER BY e.optimal_delta DESC, e.id LIMIT ?"
        )
        return [dict(row) for row in self.connection.execute(query, [*parameters, k])]

    def pair(self, pair_id: int) -> dict:
        """Read stored source/generated images and exact predictions."""
        row = self.connection.execute(
            "SELECT * FROM pair WHERE id=?", (pair_id,)
        ).fetchone()
        if row is None:
            raise KeyError(pair_id)
        result = dict(row)
        for key in (
            "original",
            "generated",
            "original_logits",
            "original_probabilities",
            "generated_logits",
            "generated_probabilities",
        ):
            result[key] = unpack(result[key])
        return result

    def thresholds(self, explanation_id: int) -> list[dict]:
        """Read the complete ordered sweep for an interactive exemplar."""
        result = []
        for row in self.connection.execute(
            "SELECT * FROM threshold WHERE explanation_id=? ORDER BY idx",
            (explanation_id,),
        ):
            record = dict(row)
            for key in ("mask", "image", "logits", "probabilities"):
                record[key] = unpack(record[key])
            result.append(record)
        return result

    def curve(self, method: str):
        """Return area grid, median, and 25th/75th percentiles across stored curves."""
        rows = self.connection.execute(
            "SELECT t.explanation_id, t.area, t.delta FROM threshold t "
            "JOIN explanation e ON e.id=t.explanation_id WHERE e.method=? ORDER BY t.explanation_id, t.idx",
            (method,),
        )
        curves = {}
        for row in rows:
            curves.setdefault(row[0], []).append((row[1], row[2]))
        if not curves:
            raise ValueError(f"No explanations for method {method}")
        grid = np.linspace(0, 1, 101)
        interpolated = [
            np.interp(grid, *np.array(points).T) for points in curves.values()
        ]
        p25, median, p75 = np.quantile(interpolated, [0.25, 0.5, 0.75], axis=0)
        return grid, median, p25, p75

    def close(self):
        """Release the database connection."""
        self.connection.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
