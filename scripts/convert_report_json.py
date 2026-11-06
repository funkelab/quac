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


def _spatial(path, shape, *, mask=False, shared=False):
    array = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if array.ndim == 3 and array.shape[0] in (1, 3):
        if (mask or shared) and not np.allclose(array, array[0], atol=1e-6):
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


def _recorded_result(record, base, shape, count, areas, deltas, optimum):
    counterfactual = _image(_resolve(record["counterfactual_path"], base))
    if counterfactual.shape != shape:
        raise ValueError(f"All image shapes must match {shape}")
    mask = _spatial(_resolve(record["mask_path"], base), shape[-2:], mask=True)
    probability = _probabilities(record["counterfactual_prediction"], count)
    source_prob = _probabilities(record["query_prediction"], count)
    target = record["target_class"]
    if not np.isclose(
        probability[target] - source_prob[target], deltas[optimum], atol=1e-5
    ):
        raise ValueError(
            "Saved counterfactual prediction does not match the optimal curve point"
        )
    cutoff = record.get("optimal_threshold")
    if cutoff is None or not np.isfinite(cutoff):
        raise ValueError("A finite optimal_threshold is required")
    return SimpleNamespace(
        optimal_index=0,
        areas=[areas[optimum]],
        deltas=[deltas[optimum]],
        auc=_score(record, areas, deltas),
        cutoffs=[float(cutoff)],
        masks=[mask],
        counterfactuals=[counterfactual],
        logits=[np.full(count, np.nan)],
        probabilities=[probability],
    )


def _score(record, areas, deltas):
    score = record.get("score")
    if score is None:
        order = np.argsort(areas, kind="stable")
        score = float(np.trapezoid(deltas[order], areas[order]))
    if not np.isfinite(score):
        raise ValueError("QuAC score must be finite")
    return float(score)


class Reconstructor:
    """Rebuild the threshold sweep with a TorchScript classifier and explicit processing settings."""

    def __init__(
        self,
        checkpoint,
        *,
        device="cpu",
        batch_size=16,
        scale=1.0,
        shift=0.0,
        threshold_min=-1.0,
        threshold_max=1.0,
        closing_size=10,
        blur_size=11,
    ):
        import torch
        from quac.classifier import load_classifier

        try:
            import cv2
        except ImportError as error:
            raise ValueError(
                "Reconstruction requires opencv-python-headless; install it with pip"
            ) from error
        if (
            batch_size < 1
            or closing_size < 1
            or blur_size < 1
            or blur_size % 2 != 1
            or not np.isfinite([scale, shift, threshold_min, threshold_max]).all()
            or threshold_min >= threshold_max
        ):
            raise ValueError(
                "Require a positive batch/closing size, positive odd blur size, and finite increasing threshold bounds"
            )
        self.cv2 = cv2
        self.device = torch.device(device)
        self.classifier = load_classifier(Path(checkpoint), self.device)
        self.batch_size, self.scale, self.shift = batch_size, scale, shift
        self.threshold_min, self.threshold_max = threshold_min, threshold_max
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_size, closing_size)
        )
        self.blur_size = blur_size
        self.metadata = dict(
            images="complete threshold sweep",
            logits="classifier",
            checkpoint=str(Path(checkpoint).resolve()),
            device=device,
            batch_size=batch_size,
            scale=scale,
            shift=shift,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            closing_size=closing_size,
            blur_size=blur_size,
        )

    def _predict(self, images, count):
        import torch
        from quac.classifier import predict

        values = torch.from_numpy(np.asarray(images, dtype=np.float32)).to(self.device)
        logits, probabilities = predict(
            self.classifier, values * self.scale + self.shift, count, self.batch_size
        )
        return logits.numpy(), probabilities.numpy()

    def evaluate(
        self, record, source, generated, attribution, count, areas, deltas, optimum
    ):
        """Return a complete evaluation after checking agreement with the recorded predictions and areas."""
        from quac.evaluation import Evaluation

        pair_logits, pair_probabilities = self._predict(
            np.stack([source, generated]), count
        )
        for key, actual in zip(
            ("query_prediction", "generated_prediction"), pair_probabilities
        ):
            if record.get(key) is not None and not np.allclose(
                actual, _probabilities(record[key], count), atol=1e-4, rtol=1e-4
            ):
                raise ValueError(
                    f"Classifier disagrees with {key}; check checkpoint, scale, shift, and image preprocessing"
                )
        cutoffs = np.arange(
            self.threshold_min,
            self.threshold_max,
            (self.threshold_max - self.threshold_min) / len(areas),
        )
        if len(cutoffs) != len(areas):
            raise ValueError(
                "Threshold bounds do not reproduce the recorded sweep length"
            )
        masks, measured_areas = [], []
        for cutoff in cutoffs:
            binary = self.cv2.morphologyEx(
                (attribution > cutoff).astype(np.uint8),
                self.cv2.MORPH_CLOSE,
                self.kernel,
            )
            measured_areas.append(float(binary.mean()))
            masks.append(
                self.cv2.GaussianBlur(
                    binary.astype(np.float32), (self.blur_size, self.blur_size), 0
                )
            )
        if not np.allclose(measured_areas, areas, atol=1e-6, rtol=0):
            raise ValueError(
                "Reconstructed mask areas disagree with the report; check threshold bounds and closing size"
            )
        masks = np.stack(masks)
        images, outputs, predictions = [], [], []
        for start in range(0, len(masks), self.batch_size):
            mask = masks[start : start + self.batch_size, None]
            batch = source[None] * (1 - mask) + generated[None] * mask
            logits, probabilities = self._predict(batch, count)
            images.append(batch)
            outputs.append(logits)
            predictions.append(probabilities)
        outputs, predictions = np.concatenate(outputs), np.concatenate(predictions)
        measured_deltas = (
            predictions[:, record["target_class"]]
            - pair_probabilities[0, record["target_class"]]
        )
        if not np.allclose(measured_deltas, deltas, atol=1e-4, rtol=1e-4):
            raise ValueError(
                "Reconstructed classifier changes disagree with the report; check blur size and preprocessing"
            )
        if record.get("counterfactual_prediction") is not None and not np.allclose(
            predictions[optimum],
            _probabilities(record["counterfactual_prediction"], count),
            atol=1e-4,
            rtol=1e-4,
        ):
            raise ValueError(
                "Reconstructed counterfactual prediction disagrees with the report"
            )
        order = np.argsort(areas, kind="stable")
        return (
            Evaluation(
                np.asarray(measured_areas)[order],
                cutoffs[order],
                masks[order],
                np.concatenate(images)[order],
                outputs[order],
                predictions[order],
                measured_deltas[order],
                int(np.flatnonzero(order == optimum)[0]),
                _score(record, areas, deltas),
            ),
            pair_logits,
            pair_probabilities,
        )


def convert_report(
    input_path,
    output_path=None,
    *,
    base_dir=None,
    method=None,
    classes=None,
    reconstructor=None,
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
    if reconstructor is not None:
        metadata["conversion"] = reconstructor.metadata
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
            if reconstructor is None:
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
                    if source_image.shape != shape or generated_image.shape != shape:
                        raise ValueError(f"All image shapes must match {shape}")
                    attribution = _spatial(
                        _resolve(record.get("attribution_path"), base),
                        shape[-2:],
                        shared=reconstructor is not None,
                    )
                    areas, deltas, optimum = _sweep(record)
                    if reconstructor is None:
                        result = _recorded_result(
                            record, base, shape, count, areas, deltas, optimum
                        )
                        source_prob = _probabilities(record["query_prediction"], count)
                        generated_prob = _probabilities(
                            record.get("generated_prediction"), count
                        )
                        source_logits = generated_logits = np.full(count, np.nan)
                    else:
                        result, pair_logits, pair_probabilities = (
                            reconstructor.evaluate(
                                record,
                                source_image,
                                generated_image,
                                attribution,
                                count,
                                areas,
                                deltas,
                                optimum,
                            )
                        )
                        source_logits, generated_logits = pair_logits
                        source_prob, generated_prob = pair_probabilities
                    key = (str(source_path), str(generated_path), source, target)
                    if key not in pairs:
                        pair_id = report.add_pair(
                            source_path,
                            source,
                            target,
                            0,
                            source_image,
                            generated_image,
                            source_logits,
                            source_prob,
                            generated_logits,
                            generated_prob,
                        )
                        pairs[key] = (pair_id, source_prob, generated_prob)
                    else:
                        pair_id, known_source, known_generated = pairs[key]
                        if not (
                            np.allclose(source_prob, known_source)
                            and np.allclose(generated_prob, known_generated)
                        ):
                            raise ValueError(
                                "Conflicting predictions for the same image pair"
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
                    if reconstructor is None:
                        with report.connection:
                            report.connection.executemany(
                                "INSERT INTO curve_point VALUES (?,?,?,?)",
                                [
                                    (explanation, i, float(areas[j]), float(deltas[j]))
                                    for i, j in enumerate(
                                        np.argsort(areas, kind="stable")
                                    )
                                ],
                            )
                except (
                    ValueError,
                    KeyError,
                    OSError,
                    sqlite3.Error,
                    RuntimeError,
                ) as error:
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
        epilog="Without a classifier checkpoint, keeps full curves and scores, but only the saved "
        "counterfactual is selectable and logits are unavailable. RGB masks must be identical across channels; attribution channels "
        "are averaged. Images use the JSON report reader's normalization. Install this checkout "
        "with `pip install -e .` before running. Reconstruction additionally requires "
        "opencv-python-headless. Threshold count comes from the recorded curve; processing defaults "
        "match main's Processor. Reconstruction verifies recorded probabilities and mask areas; "
        "use the original settings and input preprocessing. Reconstructed RGB attribution channels "
        "must agree to define one spatial blending mask.",
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
    parser.add_argument(
        "--classifier-checkpoint",
        type=Path,
        help="Reconstruct the complete sweep using a TorchScript classifier",
    )
    parser.add_argument(
        "--device", default="cpu", help="Classifier device (default: cpu)"
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Multiply classifier inputs by this value",
    )
    parser.add_argument(
        "--shift",
        type=float,
        default=0.0,
        help="Add to classifier inputs after scaling",
    )
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=1.0,
        help="Exclusive upper threshold bound",
    )
    parser.add_argument(
        "--closing-size",
        type=int,
        default=10,
        help="Elliptical morphological closing kernel size",
    )
    parser.add_argument(
        "--blur-size",
        type=int,
        default=11,
        help="Odd Gaussian kernel size; 1 disables blurring",
    )
    args = parser.parse_args()
    try:
        reconstructor = None
        if args.classifier_checkpoint:
            reconstructor = Reconstructor(
                args.classifier_checkpoint,
                device=args.device,
                batch_size=args.batch_size,
                scale=args.scale,
                shift=args.shift,
                threshold_min=args.threshold_min,
                threshold_max=args.threshold_max,
                closing_size=args.closing_size,
                blur_size=args.blur_size,
            )
        output = convert_report(
            args.report,
            args.output,
            base_dir=args.base_dir,
            method=args.method,
            classes=args.classes,
            reconstructor=reconstructor,
        )
    except (ValueError, KeyError, OSError, sqlite3.Error, RuntimeError) as error:
        parser.exit(1, f"Conversion failed: {error}\n")
    print(output)


if __name__ == "__main__":
    main()
