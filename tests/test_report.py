import sqlite3

import numpy as np
import pytest

from quac.artifacts import ImageStore
from quac.report import Report, pack, unpack


def test_array_codec_preserves_dtype_and_values():
    array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    decoded = unpack(pack(array))
    assert decoded.dtype == array.dtype
    np.testing.assert_array_equal(decoded, array)


def test_report_queries_and_complete_sweeps(report_path):
    with Report(report_path) as report:
        assert report.status == {"status": "complete", "error": None}
        assert report.methods() == ["random", "residual"]
        assert report.metadata["classes"] == ["a", "b"]
        assert report.metadata["shape"] == [1, 4, 4]
        top = report.top("residual", k=10)
        assert len(top) == 2
        assert top[0]["optimal_delta"] >= top[1]["optimal_delta"]
        assert len(report.top("residual", source=0, target=1)) == 1
        for row in top:
            pair = report.pair(row["pair_id"])
            records = report.thresholds(row["id"])
            assert len(records) == 5
            assert records[row["optimal_index"]]["delta"] == row["optimal_delta"]
            np.testing.assert_array_equal(records[0]["image"], pair["original"])
            np.testing.assert_array_equal(records[-1]["image"], pair["generated"])
            for record in records:
                assert record["logits"].shape == record["probabilities"].shape == (2,)
                assert np.isclose(
                    record["delta"],
                    record["probabilities"][pair["target"]]
                    - pair["original_probabilities"][pair["target"]],
                )
        grid, median, p25, p75 = report.curve("residual")
        assert grid.shape == median.shape == (101,)
        assert np.all(p25 <= median) and np.all(median <= p75)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            report.connection.execute("DELETE FROM threshold")


def test_schema_and_overwrite_validation(tmp_path):
    path = tmp_path / "report.sqlite"
    with Report.create(path, {"classes": ["a", "b"]}) as report:
        report.finish("test failure")
    with pytest.raises(FileExistsError):
        Report.create(path, {})
    with Report(path) as report:
        assert report.status["status"] == "failed"
        assert report.methods() == []
        with pytest.raises(ValueError, match="No explanations"):
            report.curve("unknown")
    with Report.create(path, {}, overwrite=True) as report:
        report.finish()
    database = sqlite3.connect(path)
    database.execute("PRAGMA user_version=999")
    database.close()
    with pytest.raises(ValueError, match="schema"):
        Report(path)


def test_failed_explanation_rolls_back_sweep(report_path):
    from types import SimpleNamespace

    with Report(report_path) as report:
        pair_id = report.top("residual")[0]["pair_id"]
    report = Report.__new__(Report)
    report.images = ImageStore(report_path)
    report.connection = sqlite3.connect(report_path)
    report.connection.execute("PRAGMA foreign_keys=ON")
    broken = SimpleNamespace(
        optimal_index=0,
        deltas=[1],
        auc=1,
        areas=[0, 1],
        cutoffs=[0],
        masks=[np.zeros((4, 4))],
        images=[np.zeros((1, 4, 4))],
        logits=[np.zeros(2)],
        probabilities=[np.zeros(2)],
    )
    before = set(report_path.parent.rglob("*.tif"))
    with pytest.raises(IndexError):
        report.add_explanation(pair_id, "broken", np.zeros((4, 4)), broken)
    assert (
        report.connection.execute(
            "SELECT count(*) FROM explanation WHERE method='broken'"
        ).fetchone()[0]
        == 0
    )
    assert set(report_path.parent.rglob("*.tif")) == before
    report.close()


def test_database_stores_image_paths_instead_of_pixels(report_path):
    from pathlib import Path

    with Report(report_path) as report:
        for table, columns in (
            ("pair", ["original_path", "generated_path"]),
            ("explanation", ["attribution_path"]),
            ("threshold", ["mask_path", "image_path"]),
        ):
            types = {
                row[1]: row[2]
                for row in report.connection.execute(f"PRAGMA table_info({table})")
            }
            for column in columns:
                assert types[column] == "TEXT"
                paths = [
                    row[0]
                    for row in report.connection.execute(
                        f"SELECT {column} FROM {table}"
                    )
                ]
                assert paths
                for value in paths:
                    assert not Path(value).is_absolute()
                    assert (report_path.parent / value).is_file()
                    assert (report_path.parent / value).is_relative_to(
                        report_path.parent / "images"
                    )
        assert "original" not in {
            row[1] for row in report.connection.execute("PRAGMA table_info(pair)")
        }
        assert "image" not in {
            row[1] for row in report.connection.execute("PRAGMA table_info(threshold)")
        }


def test_report_and_images_can_be_moved_together(report_path, tmp_path, monkeypatch):
    import shutil

    relocated = tmp_path / "moved-output"
    shutil.move(str(report_path.parent), relocated)
    monkeypatch.chdir(tmp_path.parent)
    with Report(relocated / report_path.name) as report:
        row = report.top("residual")[0]
        pair = report.pair(row["pair_id"])
        records = report.thresholds(row["id"])
        np.testing.assert_array_equal(records[0]["image"], pair["original"])
        np.testing.assert_array_equal(records[-1]["image"], pair["generated"])


def test_failed_image_write_leaves_no_rows_or_files(tmp_path, monkeypatch):
    with Report.create(tmp_path / "report.sqlite", {}) as report:
        write = report.images.write

        def fail_generated(path, array):
            if path.name == "generated.tif":
                raise OSError("disk full")
            return write(path, array)

        monkeypatch.setattr(report.images, "write", fail_generated)
        with pytest.raises(OSError, match="disk full"):
            report.add_pair(
                "source.tif",
                0,
                1,
                0,
                np.zeros((1, 4, 4), dtype=np.float32),
                np.ones((1, 4, 4), dtype=np.float32),
                [1, 0],
                [0.7, 0.3],
                [0, 1],
                [0.3, 0.7],
            )
        assert report.connection.execute("SELECT count(*) FROM pair").fetchone()[0] == 0
        assert not list((tmp_path / "images").iterdir())


@pytest.mark.parametrize("shape", [(1, 5, 7), (3, 5, 7), (5, 7)])
def test_artifact_files_preserve_values_and_shape(tmp_path, shape):
    image = np.random.default_rng(0).random(shape, dtype=np.float32)
    store = ImageStore(tmp_path / "report.sqlite")
    with store.batch() as directory:
        path = store.write(directory / "image.tif", image)
    loaded = store.read(path, spatial=len(shape) == 2)
    assert loaded.dtype == image.dtype
    np.testing.assert_array_equal(loaded, image)


def test_missing_artifact_reports_its_path(report_path):
    with Report(report_path) as report:
        row = report.top("residual")[0]
        pair = report.pair(row["pair_id"])
        path = report_path.parent / pair["generated_path"]
        path.unlink()
        with pytest.raises(FileNotFoundError, match="generated.tif"):
            report.pair(row["pair_id"])
