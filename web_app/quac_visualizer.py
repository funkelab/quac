#!/usr/bin/env python3
"""
QuAC Visualizer - Web-based GUI for exploring Explanation and Report objects.

Features:
- Toggle between query/counterfactual images to spot differences
- Mask overlay with opacity based on mask values (0-1)
- Filtering by source/target class, score thresholds
- Download functionality for results and images

Usage:
    # Use configuration file (reads report directory from config)
    uv run web_app/quac_visualizer.py --config config.yaml

    # Use configuration file with report override
    uv run web_app/quac_visualizer.py --config config.yaml --report /path/to/report.json
    uv run web_app/quac_visualizer.py --config config.yaml --report /path/to/reports/
"""

import argparse
import json
import logging
import zipfile
import yaml
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image

from quac.explanation import Explanation, explanation_encoder
from quac.report import Report
from quac.config import ExperimentConfig, get_data_config
from quac.data import create_transform

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Create module-level logger
logger = logging.getLogger(__name__)

# Global variable to store the current report
current_report: Optional[Report] = None
report_base_path: Optional[Path] = None
blinding_enabled: bool = False
image_transform = None  # Will store the transform for images


def load_report_from_path(path: Union[str, Path]) -> Report:
    """Load a report from a file path or directory."""
    path = Path(path)
    if path.is_file():
        report = Report()
        report.load(path)
        return report
    elif path.is_dir():
        return Report.from_directory(str(path))
    else:
        raise FileNotFoundError(f"Report path not found: {path}")


def find_explanation_by_id(exp_id: str) -> Optional[Explanation]:
    """Find an explanation by its string ID."""
    if current_report is None:
        return None

    for exp in current_report.explanations:
        if str(hash(exp)) == exp_id:
            return exp
    return None


def tensor_to_image_response(tensor: torch.Tensor) -> bytes:
    """Convert a CHW tensor to a PNG image response."""
    # Apply transform if available
    if image_transform is not None:
        tensor = image_transform(tensor)

    # Convert from CHW to HWC
    if tensor.dim() == 3:
        tensor = tensor.permute(1, 2, 0)

    # Convert to numpy and ensure values are in [0, 1]
    img_array = tensor.cpu().numpy()

    # Normalize to [0, 1] if needed
    if img_array.max() > 1.0 or img_array.min() < 0.0:
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())

    # Convert to [0, 255] uint8
    img_array = (img_array * 255).astype(np.uint8)

    # Handle grayscale (single channel)
    if img_array.shape[2] == 1:
        img_array = img_array.squeeze(2)
        pil_image = Image.fromarray(img_array, mode="L")
    else:
        pil_image = Image.fromarray(img_array, mode="RGB")

    # Save to bytes buffer
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def serialize_explanation(explanation: Explanation) -> Dict:
    """Convert an Explanation object to a JSON-serializable dictionary."""
    # Use the standard explanation_encoder from the package
    data = explanation_encoder(explanation)

    # Add web-app specific fields
    data["id"] = str(
        hash(explanation)
    )  # Unique identifier as string to avoid JS precision loss

    # Add prediction confidence scores for web display
    data["source_confidence"] = (
        max(explanation.query_prediction) if explanation.query_prediction else 0
    )
    data["target_confidence"] = (
        max(explanation.counterfactual_prediction)
        if explanation.counterfactual_prediction
        else 0
    )

    return data


@app.route("/")
def index():
    """Serve the main dashboard page."""
    return render_template("index.html")


@app.route("/api/report/info")
def report_info():
    """Get basic information about the loaded report."""
    if current_report is None:
        return jsonify({"error": "No report loaded"}), 400

    # Get unique classes from explanations
    source_classes = set()
    target_classes = set()
    for exp in current_report.explanations:
        source_classes.add(exp.source_class)
        target_classes.add(exp.target_class)

    return jsonify(
        {
            "name": current_report.name,
            "metadata": current_report.metadata,
            "num_explanations": len(current_report),
            "source_classes": (
                sorted(list(source_classes)) if not blinding_enabled else []
            ),
            "target_classes": (
                sorted(list(target_classes)) if not blinding_enabled else []
            ),
            "score_range": {
                "min": 0.0,  # QuAC scores theoretically range from 0 to 1
                "max": 1.0,
                "data_min": (
                    min(exp.score for exp in current_report.explanations)
                    if len(current_report) > 0
                    else 0
                ),
                "data_max": (
                    max(exp.score for exp in current_report.explanations)
                    if len(current_report) > 0
                    else 1
                ),
            },
            "blinding_enabled": blinding_enabled,
        }
    )


@app.route("/api/explanations")
def get_explanations():
    """Get explanations with optional filtering."""
    if current_report is None:
        return jsonify({"error": "No report loaded"}), 400

    # Get query parameters with better handling
    source_class_str = request.args.get("source_class")
    target_class_str = request.args.get("target_class")
    min_score = request.args.get("min_score", type=float)
    max_score = request.args.get("max_score", type=float)
    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", default=0, type=int)

    # Convert class parameters properly
    source_class = (
        None
        if source_class_str is None or source_class_str == ""
        else int(source_class_str)
    )
    target_class = (
        None
        if target_class_str is None or target_class_str == ""
        else int(target_class_str)
    )

    logger.debug(
        f"Filter params: source={source_class}, target={target_class}, min_score={min_score}, max_score={max_score}, offset={offset}, limit={limit}"
    )

    # Start with the full report
    filtered_report = current_report

    # Apply filters
    if source_class is not None:
        logger.debug(f"Filtering by source_class: {source_class}")
        filtered_report = filtered_report.from_source(source_class)
        logger.debug(
            f"After source filter: {len(filtered_report.explanations)} explanations"
        )

    if target_class is not None:
        logger.debug(f"Filtering by target_class: {target_class}")
        filtered_report = filtered_report.to_target(target_class)
        logger.debug(
            f"After target filter: {len(filtered_report.explanations)} explanations"
        )

    if min_score is not None:
        logger.debug(f"Filtering by min_score: {min_score}")
        filtered_report = filtered_report.score_threshold(min_score)
        logger.debug(
            f"After min_score filter: {len(filtered_report.explanations)} explanations"
        )

    # Apply max score filter (custom logic since Report doesn't have this built-in)
    if max_score is not None:
        logger.debug(f"Filtering by max_score: {max_score}")
        explanations = [
            exp for exp in filtered_report.explanations if exp.score <= max_score
        ]
        new_report = Report(name=f"{filtered_report.name}_filtered")
        new_report.explanations = explanations
        filtered_report = new_report
        logger.debug(
            f"After max_score filter: {len(filtered_report.explanations)} explanations"
        )

    # Sort by score (highest first)
    explanations = sorted(
        filtered_report.explanations, key=lambda x: x.score, reverse=True
    )

    logger.debug(f"After filtering: {len(explanations)} explanations found")

    # Apply offset and limit
    if limit is not None:
        explanations = explanations[offset : offset + limit]
    else:
        explanations = explanations[offset:]

    # Serialize explanations
    serialized = [serialize_explanation(exp) for exp in explanations]

    return jsonify(
        {
            "explanations": serialized,
            "total": len(filtered_report.explanations),
            "offset": offset,
            "returned": len(serialized),
        }
    )


@app.route("/api/explanation/<int:explanation_id>")
def get_explanation_details(explanation_id: int):
    """Get detailed information for a specific explanation."""
    if current_report is None:
        return jsonify({"error": "No report loaded"}), 400

    # Find explanation by hash ID
    explanation = None
    for exp in current_report.explanations:
        if hash(exp) == explanation_id:
            explanation = exp
            break

    if explanation is None:
        return jsonify({"error": "Explanation not found"}), 404

    return jsonify(serialize_explanation(explanation))


@app.route("/api/explanation/<exp_id>/annotation", methods=["POST"])
def save_annotation(exp_id: str):
    """Save annotation for a specific explanation."""
    if current_report is None:
        return jsonify({"error": "No report loaded"}), 400

    try:
        data = request.get_json()
        annotation = data.get("annotation", "")

        # Find the explanation by ID
        explanation = None

        print(f"Looking for explanation with ID: {exp_id}")

        for exp in current_report.explanations:
            exp_hash = str(hash(exp))
            print(f"Checking explanation hash: {exp_hash}")
            if exp_hash == exp_id:
                explanation = exp
                break

        if explanation is None:
            print(f"No explanation found with ID {exp_id}")
            return jsonify({"error": "Explanation not found"}), 404

        # Update the annotation
        explanation.annotation = annotation

        return jsonify({"success": True, "message": "Annotation saved"})

    except Exception as e:
        return jsonify({"error": f"Failed to save annotation: {str(e)}"}), 500


@app.route("/api/curve")
def get_quac_curve():
    """Get QuAC curve data for the current report or filtered subset."""
    if current_report is None:
        return jsonify({"error": "No report loaded"}), 400

    # Apply same filtering as get_explanations for consistency
    source_class = request.args.get("source_class", type=int)
    target_class = request.args.get("target_class", type=int)
    min_score = request.args.get("min_score", type=float)
    max_score = request.args.get("max_score", type=float)

    filtered_report = current_report

    if source_class is not None:
        filtered_report = filtered_report.from_source(source_class)
    if target_class is not None:
        filtered_report = filtered_report.to_target(target_class)
    if min_score is not None:
        filtered_report = filtered_report.score_threshold(min_score)
    if max_score is not None:
        explanations = [
            exp for exp in filtered_report.explanations if exp.score <= max_score
        ]
        new_report = Report(name=f"{filtered_report.name}_filtered")
        new_report.explanations = explanations
        filtered_report = new_report

    try:
        if len(filtered_report.explanations) == 0:
            return jsonify({"error": "No explanations match the current filters"}), 400

        median, p25, p75 = filtered_report.get_curve()
        return jsonify(
            {
                "x_values": filtered_report.interp_mask_values.tolist(),
                "median": median.tolist(),
                "p25": p25.tolist(),
                "p75": p75.tolist(),
                "num_samples": len(filtered_report.explanations),
            }
        )
    except Exception as e:
        return jsonify({"error": f"Could not generate curve: {str(e)}"}), 500


@app.route("/api/image/<exp_id>/<image_type>")
def serve_image(exp_id: str, image_type: str):
    """Serve image data from explanation object."""
    try:
        # Find the explanation
        explanation = find_explanation_by_id(exp_id)
        if explanation is None:
            return jsonify({"error": "Explanation not found"}), 404

        # Get the appropriate tensor
        if image_type == "query":
            tensor = explanation.query
        elif image_type == "counterfactual":
            tensor = explanation.counterfactual
        else:
            return jsonify({"error": f"Invalid image type: {image_type}"}), 400

        # Convert tensor to image
        image_data = tensor_to_image_response(tensor)

        # Create response
        buffer = BytesIO(image_data)
        buffer.seek(0)

        return send_file(buffer, mimetype="image/png")

    except Exception as e:
        return jsonify({"error": f"Error serving image: {str(e)}"}), 500


@app.route("/api/mask/<exp_id>")
def serve_mask(exp_id: str):
    """Serve mask data from explanation object as JSON array."""
    try:
        # Find the explanation
        explanation = find_explanation_by_id(exp_id)
        if explanation is None:
            return jsonify({"error": "Explanation not found"}), 404

        # Get mask tensor
        mask_tensor = explanation.mask

        # Convert to numpy
        mask_data = mask_tensor.cpu().numpy()

        # Normalize to channels-last format (height, width, channels)
        if mask_data.ndim == 3:
            # If channels-first (channels, height, width), transpose to channels-last
            if mask_data.shape[0] <= 3 and mask_data.shape[0] < mask_data.shape[1]:
                mask_data = np.transpose(mask_data, (1, 2, 0))
            # Squeeze singleton channel dimensions
            if mask_data.shape[2] == 1:
                mask_data = mask_data.squeeze(2)
        elif mask_data.ndim > 3:
            # Squeeze extra dimensions
            mask_data = mask_data.squeeze()
            if (
                mask_data.ndim == 3
                and mask_data.shape[0] <= 3
                and mask_data.shape[0] < mask_data.shape[1]
            ):
                mask_data = np.transpose(mask_data, (1, 2, 0))

        # Normalize values to 0-255 range for image display
        if mask_data.max() > mask_data.min():
            mask_data = (
                (mask_data - mask_data.min())
                / (mask_data.max() - mask_data.min())
                * 255
            ).astype(np.uint8)
        else:
            mask_data = np.zeros_like(mask_data, dtype=np.uint8)

        # Ensure mask is always 3-channel RGB
        if mask_data.ndim == 2:
            # Convert grayscale to RGB by putting mask in red channel, zeros in green/blue
            height, width = mask_data.shape
            rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_mask[:, :, 0] = mask_data  # Red channel gets the mask
            # Green and blue channels remain 0
            mask_data = rgb_mask

        return jsonify(
            {
                "mask": mask_data.tolist(),
                "shape": list(mask_data.shape),
            }
        )

    except Exception as e:
        return jsonify({"error": f"Error serving mask: {str(e)}"}), 500


@app.route("/api/download/explanations")
def download_explanations():
    """Download filtered explanations as JSON."""
    if current_report is None:
        return jsonify({"error": "No report loaded"}), 400

    # Apply same filtering as get_explanations
    source_class = request.args.get("source_class", type=int)
    target_class = request.args.get("target_class", type=int)
    min_score = request.args.get("min_score", type=float)
    max_score = request.args.get("max_score", type=float)

    filtered_report = current_report

    if source_class is not None:
        filtered_report = filtered_report.from_source(source_class)
    if target_class is not None:
        filtered_report = filtered_report.to_target(target_class)
    if min_score is not None:
        filtered_report = filtered_report.score_threshold(min_score)
    if max_score is not None:
        explanations = [
            exp for exp in filtered_report.explanations if exp.score <= max_score
        ]
        new_report = Report(name=f"{filtered_report.name}_filtered")
        new_report.explanations = explanations
        filtered_report = new_report

    # Create JSON data with filters added to metadata
    metadata_with_filters = (
        dict(filtered_report.metadata) if filtered_report.metadata else {}
    )
    metadata_with_filters["filters"] = {
        "source_class": source_class,
        "target_class": target_class,
        "min_score": min_score,
        "max_score": max_score,
    }

    data = {
        "report_name": filtered_report.name,
        "metadata": metadata_with_filters,
        "explanations": [
            serialize_explanation(exp) for exp in filtered_report.explanations
        ],
    }

    # Create in-memory file
    json_data = json.dumps(data, indent=2)
    buffer = BytesIO(json_data.encode("utf-8"))
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype="application/json",
        as_attachment=True,
        download_name=f"{filtered_report.name}_filtered_explanations.json",
    )


@app.route("/api/download/images")
def download_images():
    """Download images for filtered explanations as a ZIP file."""
    if current_report is None:
        return jsonify({"error": "No report loaded"}), 400

    # Apply same filtering as get_explanations
    source_class = request.args.get("source_class", type=int)
    target_class = request.args.get("target_class", type=int)
    min_score = request.args.get("min_score", type=float)
    max_score = request.args.get("max_score", type=float)

    filtered_report = current_report

    if source_class is not None:
        filtered_report = filtered_report.from_source(source_class)
    if target_class is not None:
        filtered_report = filtered_report.to_target(target_class)
    if min_score is not None:
        filtered_report = filtered_report.score_threshold(min_score)
    if max_score is not None:
        explanations = [
            exp for exp in filtered_report.explanations if exp.score <= max_score
        ]
        new_report = Report(name=f"{filtered_report.name}_filtered")
        new_report.explanations = explanations
        filtered_report = new_report

    # Create ZIP file in memory
    zip_buffer = BytesIO()

    try:
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for i, exp in enumerate(filtered_report.explanations):
                score_str = f"score_{exp.score:.4f}"
                folder_name = f"explanation_{i:04d}_{score_str}_{exp.source_class}to{exp.target_class}"

                # Add query image
                if exp._query_path and Path(exp._query_path).exists():
                    zip_file.write(
                        exp._query_path,
                        f"{folder_name}/query.{Path(exp._query_path).suffix}",
                    )

                # Add counterfactual image
                if exp._counterfactual_path and Path(exp._counterfactual_path).exists():
                    zip_file.write(
                        exp._counterfactual_path,
                        f"{folder_name}/counterfactual.{Path(exp._counterfactual_path).suffix}",
                    )

                # Add mask
                if exp._mask_path and Path(exp._mask_path).exists():
                    zip_file.write(exp._mask_path, f"{folder_name}/mask.npy")

                # Add explanation metadata
                metadata = serialize_explanation(exp)
                metadata_json = json.dumps(metadata, indent=2)
                zip_file.writestr(f"{folder_name}/metadata.json", metadata_json)

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{filtered_report.name}_filtered_images.zip",
        )

    except Exception as e:
        return jsonify({"error": f"Error creating ZIP file: {str(e)}"}), 500


def main():
    parser = argparse.ArgumentParser(description="QuAC Visualizer")
    parser.add_argument(
        "--config", "-c", type=str, required=True, help="Path to configuration file"
    )
    parser.add_argument(
        "--report", type=str, help="Override report path (file or directory)"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to run the server on"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    parser.add_argument(
        "--blind",
        action="store_true",
        help="Enable blinding mode (hide class information)",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load configuration
    with open(args.config, "r") as file:
        config_data = yaml.safe_load(file)
    experiment_config = ExperimentConfig(**config_data)
    logger.info(f"Loaded configuration from: {args.config}")

    # Create image transform from data config
    data_config = get_data_config(experiment_config, "test")
    global image_transform
    image_transform = create_transform(
        img_size=data_config.img_size,
        grayscale=data_config.grayscale,
        rgb=data_config.rgb,
    )
    logger.info(
        f"Created image transform: img_size={data_config.img_size}, grayscale={data_config.grayscale}, rgb={data_config.rgb}"
    )

    # Set report path from config, then override if provided
    report_path = str(Path(experiment_config.solver.root_dir) / "reports")
    if args.report:
        report_path = args.report
    logger.info(f"Using report: {report_path}")

    global current_report, report_base_path, blinding_enabled
    blinding_enabled = args.blind

    current_report = load_report_from_path(report_path)
    report_base_path = Path(report_path).resolve()
    if report_base_path.is_file():
        report_base_path = report_base_path.parent

    logger.info(f"Loaded report: {current_report.name}")
    logger.info(f"Number of explanations: {len(current_report)}")
    logger.info(f"Report base path: {report_base_path}")
    logger.info(f"Blinding mode: {'Enabled' if blinding_enabled else 'Disabled'}")

    logger.info(f"Starting QuAC Visualizer on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
