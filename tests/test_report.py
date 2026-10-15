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
    with pytest.raises(IndexError):
        report.add_explanation(pair_id, "broken", np.zeros((4, 4)), broken)
    assert (
        report.connection.execute(
            "SELECT count(*) FROM explanation WHERE method='broken'"
        ).fetchone()[0]
        == 0
    )
    report.close()
