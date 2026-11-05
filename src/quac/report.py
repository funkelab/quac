from contextlib import AbstractContextManager
from io import BytesIO
import json
from pathlib import Path
import sqlite3
import zlib

import numpy as np

from .artifacts import ImageStore

SCHEMA_VERSION = 3
SCHEMA = """
CREATE TABLE run (id INTEGER PRIMARY KEY CHECK(id=1), metadata TEXT NOT NULL,
                  status TEXT NOT NULL, error TEXT);
CREATE TABLE pair (
    id INTEGER PRIMARY KEY, source_input_path TEXT NOT NULL, source_class INTEGER NOT NULL,
    target_class INTEGER NOT NULL, sample INTEGER NOT NULL, reference_path TEXT,
    source_path TEXT NOT NULL, generated_path TEXT NOT NULL,
    source_logits BLOB NOT NULL, source_probabilities BLOB NOT NULL,
    generated_logits BLOB NOT NULL, generated_probabilities BLOB NOT NULL);
CREATE TABLE explanation (
    id INTEGER PRIMARY KEY, pair_id INTEGER NOT NULL REFERENCES pair(id),
    method TEXT NOT NULL, attribution_map_path TEXT NOT NULL, optimal_index INTEGER NOT NULL,
    optimal_delta REAL NOT NULL, auc REAL NOT NULL, UNIQUE(pair_id, method));
CREATE TABLE threshold (
    explanation_id INTEGER NOT NULL REFERENCES explanation(id),
    idx INTEGER NOT NULL, area REAL NOT NULL, cutoff REAL NOT NULL,
    mask_path TEXT NOT NULL, counterfactual_path TEXT NOT NULL, logits BLOB NOT NULL,
    probabilities BLOB NOT NULL, delta REAL NOT NULL,
    PRIMARY KEY(explanation_id, idx));
CREATE INDEX ranking ON explanation(method, optimal_delta DESC);
CREATE INDEX class_pair ON pair(source_class, target_class);
"""


def pack(array) -> bytes:
    """Encode a typed NumPy array without pickle."""
    stream = BytesIO()
    np.save(stream, np.asarray(array), allow_pickle=False)
    return zlib.compress(stream.getvalue())


def unpack(blob: bytes) -> np.ndarray:
    """Decode a compressed array without executing serialized objects."""
    return np.load(BytesIO(zlib.decompress(blob)), allow_pickle=False)


def _selection(method, source_class, target_class):
    conditions, parameters = ["e.method=?"], [method]
    for field, value in (
        ("source_class", source_class),
        ("target_class", target_class),
    ):
        if value is not None:
            conditions.append(f"p.{field}=?")
            parameters.append(value)
    return " AND ".join(conditions), parameters


class Report(AbstractContextManager):
    """Query a report read-only, or create it explicitly through Report.create."""

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.images = ImageStore(self.path)
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
        self._curve_table = (
            "curve_point"
            if self.connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='curve_point'"
            ).fetchone()
            else "threshold"
        )

    @classmethod
    def create(cls, path, metadata, overwrite=False, *, image_directory=None):
        """Create an empty report, refusing to overwrite unless explicitly requested."""
        path = Path(path).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"Report already exists: {path}; set output.overwrite: true to replace it"
                )
            path.unlink()
        report = cls.__new__(cls)
        report._curve_table = "threshold"
        report.path = path
        report.images = ImageStore(
            path,
            Path(image_directory).resolve() if image_directory is not None else None,
        )
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
        source_input_path,
        source_class,
        target_class,
        sample,
        source_image,
        generated_image,
        source_logits,
        source_probabilities,
        generated_logits,
        generated_probabilities,
        reference_path=None,
    ) -> int:
        """Write source and generated images and record their paths and predictions."""
        with self.images.batch() as directory, self.connection:
            cursor = self.connection.execute(
                "INSERT INTO pair VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    str(source_input_path),
                    source_class,
                    target_class,
                    sample,
                    str(reference_path) if reference_path else None,
                    self.images.write(directory / "source.tif", source_image),
                    self.images.write(directory / "generated.tif", generated_image),
                    pack(source_logits),
                    pack(source_probabilities),
                    pack(generated_logits),
                    pack(generated_probabilities),
                ),
            )
        return cursor.lastrowid

    def add_explanation(self, pair_id, method, attribution_map, result) -> int:
        """Commit a counterfactual sweep and its artifact paths; discard artifacts on failure."""
        with self.images.batch() as directory, self.connection:
            cursor = self.connection.execute(
                "INSERT INTO explanation VALUES (NULL,?,?,?,?,?,?)",
                (
                    pair_id,
                    method,
                    self.images.write(
                        directory / "attribution-map.tif", attribution_map
                    ),
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
                        self.images.write(
                            directory / f"{index:04d}-mask.tif", result.masks[index]
                        ),
                        self.images.write(
                            directory / f"{index:04d}-counterfactual.tif",
                            result.counterfactuals[index],
                        ),
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

    def directions(self, method: str) -> list[tuple[int, int]]:
        """List source/target class directions with completed explanations."""
        rows = self.connection.execute(
            "SELECT DISTINCT p.source_class, p.target_class FROM pair p "
            "JOIN explanation e ON e.pair_id=p.id WHERE e.method=? "
            "AND p.source_class != p.target_class ORDER BY p.source_class, p.target_class",
            (method,),
        )
        return [tuple(row) for row in rows]

    def top(
        self, method: str, k: int = 5, source_class=None, target_class=None
    ) -> list[dict]:
        """Query exemplars by optimal delta, optionally filtering by source and target class."""
        if k < 1:
            raise ValueError("k must be positive")
        where, parameters = _selection(method, source_class, target_class)
        query = (
            "SELECT e.*, p.source_input_path, p.source_class, p.target_class, p.sample FROM explanation e "
            "JOIN pair p ON p.id=e.pair_id WHERE "
            + where
            + " ORDER BY e.optimal_delta DESC, e.id LIMIT ?"
        )
        return [dict(row) for row in self.connection.execute(query, [*parameters, k])]

    def pair(self, pair_id: int) -> dict:
        """Load source and generated images and decode their predictions."""
        row = self.connection.execute(
            "SELECT * FROM pair WHERE id=?", (pair_id,)
        ).fetchone()
        if row is None:
            raise KeyError(pair_id)
        result = dict(row)
        for role in ("source", "generated"):
            result[f"{role}_image"] = self.images.read(result[f"{role}_path"])
        for key in (
            "source_logits",
            "source_probabilities",
            "generated_logits",
            "generated_probabilities",
        ):
            result[key] = unpack(result[key])
        return result

    def thresholds(self, explanation_id: int) -> list[dict]:
        """Read the ordered sweep, loading masks and counterfactuals from disk."""
        result = []
        for row in self.connection.execute(
            "SELECT * FROM threshold WHERE explanation_id=? ORDER BY idx",
            (explanation_id,),
        ):
            record = dict(row)
            record["mask"] = self.images.read(record["mask_path"], spatial=True)
            record["counterfactual"] = self.images.read(record["counterfactual_path"])
            for key in ("logits", "probabilities"):
                record[key] = unpack(record[key])
            result.append(record)
        return result

    def curve(self, method: str, source_class=None, target_class=None):
        """Return median and 25th/75th percentile curves for the selected classes."""
        where, parameters = _selection(method, source_class, target_class)
        rows = self.connection.execute(
            f"SELECT t.explanation_id, t.area, t.delta FROM {self._curve_table} t "
            "JOIN explanation e ON e.id=t.explanation_id JOIN pair p ON p.id=e.pair_id "
            "WHERE " + where + " ORDER BY t.explanation_id, t.idx",
            parameters,
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

    def example_curves(self, pair_id: int) -> dict[str, dict]:
        """Return each method's mask areas, classifier changes, and QuAC score for a pair."""
        rows = self.connection.execute(
            "SELECT e.method, e.auc, t.area, t.delta FROM explanation e "
            f"JOIN {self._curve_table} t ON t.explanation_id=e.id WHERE e.pair_id=? "
            "ORDER BY e.method, t.idx",
            (pair_id,),
        )
        curves = {}
        for row in rows:
            curve = curves.setdefault(
                row["method"], {"score": row["auc"], "areas": [], "deltas": []}
            )
            curve["areas"].append(row["area"])
            curve["deltas"].append(row["delta"])
        return curves

    def close(self):
        """Release the database connection."""
        self.connection.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
