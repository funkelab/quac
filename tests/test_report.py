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
        assert len(report.top("residual", source_class=0, target_class=1)) == 1
        for row in top:
            pair = report.pair(row["pair_id"])
            records = report.thresholds(row["id"])
            assert len(records) == 5
            assert records[row["optimal_index"]]["delta"] == row["optimal_delta"]
            np.testing.assert_array_equal(
                records[0]["counterfactual"], pair["source_image"]
            )
            np.testing.assert_array_equal(
                records[-1]["counterfactual"], pair["generated_image"]
            )
            for record in records:
                assert record["logits"].shape == record["probabilities"].shape == (2,)
                assert np.isclose(
                    record["delta"],
                    record["probabilities"][pair["target_class"]]
                    - pair["source_probabilities"][pair["target_class"]],
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
        counterfactuals=[np.zeros((1, 4, 4))],
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


def test_curves_are_filtered_by_direction(multiclass_report_path):
    with Report(multiclass_report_path) as report:
        assert report.directions("residual") == [
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
        ]
        curves = []
        for source, target in report.directions("residual"):
            rows = report.top("residual", 10, source, target)
            assert len(rows) == 2
            assert {(row["source_class"], row["target_class"]) for row in rows} == {
                (source, target)
            }
            deltas = [
                [record["delta"] for record in report.thresholds(row["id"])]
                for row in rows
            ]
            grid, median, q25, q75 = report.curve("residual", source, target)
            expected_q25, expected_median, expected_q75 = np.quantile(
                deltas, [0.25, 0.5, 0.75], axis=0
            )
            np.testing.assert_allclose(median[::25], expected_median)
            np.testing.assert_allclose(q25[::25], expected_q25)
            np.testing.assert_allclose(q75[::25], expected_q75)
            curves.append(median)
        assert not np.allclose(curves[0], curves[2])
