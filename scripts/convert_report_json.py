import argparse
import json
import sqlite3
from pathlib import Path
import tempfile
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np

from quac.report import Report


def _resolve(value, base):
    if not value or value == "None":
        raise ValueError("A required image or array path is missing")
    path = Path(value).expanduser()
    path = path if path.is_absolute() else base / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def _image(path):
    array = np.asarray(iio.imread(path))
    original_dtype = array.dtype
    array = array.astype(np.float32)
    if original_dtype == np.uint8:
        array /= 255
    elif array.min() < 0 or array.max() > 1:
        low = array.min(axis=(0, 1), keepdims=True)
        span = array.max(axis=(0, 1), keepdims=True) - low
        array = np.divide(array - low, span, out=np.zeros_like(array), where=span > 0)
    if array.ndim == 2:
        array = array[None]
    elif array.ndim == 3:
        array = np.moveaxis(array, -1, 0)
    if array.ndim != 3 or array.shape[0] not in (1, 3) or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite grayscale or RGB image: {path}")
    return array


def _spatial(path, shape, *, mask=False):
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        if mask and not np.allclose(array, array[0], atol=1e-6):
            raise ValueError(
                f"Channel-specific masks cannot form a single spatial contour: {path}"
            )
        array = array.mean(axis=0)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite array of shape {shape}: {path}")
    if mask and (array.min() < 0 or array.max() > 1):
        raise ValueError(f"Mask values must be in [0,1]: {path}")
    return array


def _probabilities(value, count):
    array = np.asarray(value, dtype=float)
    if (
        array.shape != (count,)
        or not np.isfinite(array).all()
        or np.any(array < 0)
        or np.any(array > 1)
        or not np.isclose(array.sum(), 1, atol=1e-5)
    ):
        raise ValueError(f"Expected {count} class probabilities")
    return array


def _sweep(record):
    areas = np.asarray(record.get("normalized_mask_sizes"), dtype=float)
    deltas = np.asarray(record.get("score_changes"), dtype=float)
    if (
        areas.ndim != 1
        or not areas.size
        or areas.shape != deltas.shape
        or not np.isfinite(areas).all()
        or not np.isfinite(deltas).all()
        or np.any(areas < 0)
        or np.any(areas > 1)
    ):
        raise ValueError(
            "Expected matching finite normalized_mask_sizes and score_changes"
        )
    optimum = int(np.argmin(np.abs(areas) - deltas))
    return areas, deltas, optimum


def convert_report(
    input_path, output_path=None, *, base_dir=None, method=None, classes=None
):
    """Copy recorded images and predictions, retaining complete curves and supplied scores."""
    input_path = Path(input_path).resolve()
    data = json.loads(input_path.read_text())
    records = data["results"]
    if not records:
        raise ValueError("The JSON report has no results")
    base = Path(base_dir).resolve() if base_dir else input_path.parent
    output = (
        Path(output_path).resolve()
        if output_path
        else input_path.parent / "quac-output" / "report.sqlite"
    )
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    count = len(records[0]["query_prediction"])
    labels = (
        classes
        or data.get("metadata", {}).get("classes")
        or [str(i) for i in range(count)]
    )
    if len(labels) != count or not all(isinstance(label, str) for label in labels):
        raise ValueError("Class labels must match the prediction vector length")
    shape = _image(_resolve(records[0]["query_path"], base)).shape
    metadata = {
        "classes": labels,
        "shape": list(shape),
        "source_report": str(input_path),
        "source_metadata": data.get("metadata", {}),
        "conversion": {
            "images": "selected counterfactual only",
            "logits": "unavailable",
            "attribution_channels": "mean",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    image_root = (
        output.parent
        if output.parent.name == "quac-output"
        else output.parent / "quac-output"
    )
    with tempfile.TemporaryDirectory(
        prefix=".convert-", dir=output.parent
    ) as temporary:
        temporary = Path(temporary)
        with Report.create(temporary / "report.sqlite", metadata) as report:
            report.connection.execute(
                "CREATE TABLE curve_point (explanation_id INTEGER NOT NULL REFERENCES explanation(id), "
                "idx INTEGER NOT NULL, area REAL NOT NULL, delta REAL NOT NULL, "
                "PRIMARY KEY (explanation_id, idx))"
            )
            pairs = {}
            for index, record in enumerate(records):
                try:
                    source, target = record["source_class"], record["target_class"]
                    if not all(
                        isinstance(value, int) and 0 <= value < count
                        for value in (source, target)
                    ):
                        raise ValueError(
                            "Class indices are outside the prediction vector"
                        )
                    source_path = _resolve(record["query_path"], base)
                    generated_path = _resolve(record.get("generated_path"), base)
                    source_image, generated_image = _image(source_path), _image(
                        generated_path
                    )
                    counterfactual = _image(
                        _resolve(record["counterfactual_path"], base)
                    )
                    if any(
                        image.shape != shape
                        for image in (source_image, generated_image, counterfactual)
                    ):
                        raise ValueError(f"All image shapes must match {shape}")
                    mask = _spatial(
                        _resolve(record["mask_path"], base), shape[-2:], mask=True
                    )
                    attribution = _spatial(
                        _resolve(record.get("attribution_path"), base), shape[-2:]
                    )
                    source_prob = _probabilities(record["query_prediction"], count)
                    generated_prob = _probabilities(
                        record.get("generated_prediction"), count
                    )
                    probability = _probabilities(
                        record["counterfactual_prediction"], count
                    )
                    areas, deltas, optimum = _sweep(record)
                    if not np.isclose(
                        probability[target] - source_prob[target],
                        deltas[optimum],
                        atol=1e-5,
                    ):
                        raise ValueError(
                            "Saved counterfactual prediction does not match the optimal curve point"
                        )
                    key = (str(source_path), str(generated_path), source, target)
                    if key not in pairs:
                        pair_id = report.add_pair(
                            source_path,
                            source,
                            target,
                            0,
                            source_image,
                            generated_image,
                            np.full(count, np.nan),
                            source_prob,
                            np.full(count, np.nan),
                            generated_prob,
                        )
                    if key in pairs:
                        pair_id, known_source, known_generated = pairs[key]
                        if not (
                            np.allclose(source_prob, known_source)
                            and np.allclose(generated_prob, known_generated)
                        ):
                            raise ValueError(
                                "Conflicting predictions for the same image pair"
                            )
                    else:
                        pairs[key] = (pair_id, source_prob, generated_prob)
                    score = record.get("score")
                    if score is None:
                        order = np.argsort(areas, kind="stable")
                        score = float(np.trapezoid(deltas[order], areas[order]))
                    if not np.isfinite(score):
                        raise ValueError("QuAC score must be finite")
                    cutoff = record.get("optimal_threshold")
                    if cutoff is None or not np.isfinite(cutoff):
                        raise ValueError("A finite optimal_threshold is required")
                    result = SimpleNamespace(
                        optimal_index=0,
                        areas=[areas[optimum]],
                        deltas=[deltas[optimum]],
                        auc=float(score),
                        cutoffs=[float(cutoff)],
                        masks=[mask],
                        counterfactuals=[counterfactual],
                        logits=[np.full(count, np.nan)],
                        probabilities=[probability],
                    )
                    name = (
                        record.get("method")
                        or method
                        or input_path.parent.name
                        or data.get("name")
                    )
                    explanation = report.add_explanation(
                        pair_id, name, attribution, result
                    )
                    with report.connection:
                        report.connection.executemany(
                            "INSERT INTO curve_point VALUES (?,?,?,?)",
                            [
                                (explanation, i, float(areas[j]), float(deltas[j]))
                                for i, j in enumerate(np.argsort(areas, kind="stable"))
                            ],
                        )
                except (ValueError, KeyError, OSError, sqlite3.Error) as error:
                    raise ValueError(f"Result {index}: {error}") from error
            report.finish()
            destination = image_root / "images" / temporary.name.removeprefix(".")
            for table, columns in (
                ("pair", ("source_path", "generated_path")),
                ("explanation", ("attribution_map_path",)),
                ("threshold", ("mask_path", "counterfactual_path")),
            ):
                for column in columns:
                    prefix = destination.relative_to(output.parent).as_posix() + "/"
                    report.connection.execute(
                        f"UPDATE {table} SET {column}=? || substr({column}, 8)",
                        (prefix,),
                    )
            report.connection.commit()
        destination.parent.mkdir(parents=True, exist_ok=True)
        (temporary / "images").rename(destination)
        (temporary / "report.sqlite").rename(output)
    return output


def main():
    """Convert a JSON report and copy its referenced artifacts into quac-output."""
    parser = argparse.ArgumentParser(
        description="Convert a report JSON to SQLite, copying images under quac-output.",
        epilog="Keeps full curves and scores, but only the saved counterfactual is selectable. "
        "Logits are unavailable. RGB masks must be identical across channels; attribution channels "
        "are averaged. Images use the JSON report reader's normalization. Install this checkout "
        "with `pip install -e .` before running.",
    )
    parser.add_argument("report", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Default: JSON directory/quac-output/report.sqlite",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Base for relative artifact paths (default: JSON directory)",
    )
    parser.add_argument(
        "--method",
        help="Method for records without one (default: JSON parent directory name)",
    )
    parser.add_argument(
        "--classes", nargs="+", help="Class names in prediction-vector order"
    )
    args = parser.parse_args()
    try:
        output = convert_report(
            args.report,
            args.output,
            base_dir=args.base_dir,
            method=args.method,
            classes=args.classes,
        )
    except (ValueError, KeyError, OSError, sqlite3.Error) as error:
        parser.exit(1, f"Conversion failed: {error}\n")
    print(output)


if __name__ == "__main__":
    main()
