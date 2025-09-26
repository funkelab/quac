#!/usr/bin/env python3
"""
QuAC Visualizer - Web-based GUI for exploring Explanation and Report objects.

Features:
- Toggle between query/counterfactual images to spot differences
- Mask overlay with opacity based on mask values (0-1)
- Filtering by source/target class, score thresholds
- Download functionality for results and images

Usage:
    uv run web_app/quac_visualizer.py --report-path /path/to/report.json
    uv run web_app/quac_visualizer.py --report-dir /path/to/reports/directory
"""

import argparse
import json
import mimetypes
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Optional, Union
from urllib.parse import unquote

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from quac.explanation import Explanation, serialize
from quac.report import Report

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

# Global variable to store the current report
current_report: Optional[Report] = None
report_base_path: Optional[Path] = None


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


def serialize_explanation(explanation: Explanation) -> Dict:
    """Convert an Explanation object to a JSON-serializable dictionary."""
    return {
        "id": hash(explanation),  # Unique identifier
        "query_path": explanation._query_path,
        "counterfactual_path": explanation._counterfactual_path,
        "mask_path": explanation._mask_path,
        "query_prediction": serialize(explanation.query_prediction),
        "counterfactual_prediction": serialize(explanation.counterfactual_prediction),
        "source_class": explanation.source_class,
        "target_class": explanation.target_class,
        "score": explanation.score,
        "normalized_mask_sizes": serialize(explanation._normalized_mask_sizes),
        "score_changes": serialize(explanation._score_changes),
        "optimal_threshold": explanation._optimal_threshold,
        "method": getattr(explanation, "method", None),
        # Add prediction confidence scores
        "source_confidence": (
            max(explanation.query_prediction) if explanation.query_prediction else 0
        ),
        "target_confidence": (
            max(explanation.counterfactual_prediction)
            if explanation.counterfactual_prediction
            else 0
        ),
    }


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
            "source_classes": sorted(list(source_classes)),
            "target_classes": sorted(list(target_classes)),
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

    print(
        f"Filter params: source={source_class}, target={target_class}, min_score={min_score}, max_score={max_score}, offset={offset}, limit={limit}"
    )

    # Start with the full report
    filtered_report = current_report

    # Apply filters
    if source_class is not None:
        print(f"Filtering by source_class: {source_class}")
        filtered_report = filtered_report.from_source(source_class)
        print(f"After source filter: {len(filtered_report.explanations)} explanations")

    if target_class is not None:
        print(f"Filtering by target_class: {target_class}")
        filtered_report = filtered_report.to_target(target_class)
        print(f"After target filter: {len(filtered_report.explanations)} explanations")

    if min_score is not None:
        print(f"Filtering by min_score: {min_score}")
        filtered_report = filtered_report.score_threshold(min_score)
        print(
            f"After min_score filter: {len(filtered_report.explanations)} explanations"
        )

    # Apply max score filter (custom logic since Report doesn't have this built-in)
    if max_score is not None:
        print(f"Filtering by max_score: {max_score}")
        explanations = [
            exp for exp in filtered_report.explanations if exp.score <= max_score
        ]
        new_report = Report(name=f"{filtered_report.name}_filtered")
        new_report.explanations = explanations
        filtered_report = new_report
        print(
            f"After max_score filter: {len(filtered_report.explanations)} explanations"
        )

    # Sort by score (highest first)
    explanations = sorted(
        filtered_report.explanations, key=lambda x: x.score, reverse=True
    )

    print(f"After filtering: {len(explanations)} explanations found")

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


@app.route("/api/image/<path:image_path>")
def serve_image(image_path: str):
    """Serve image files."""
    try:
        # URL decode the image path
        decoded_path = unquote(image_path)

        # Flask's <path:> route strips the leading /, so add it back for absolute paths
        if not decoded_path.startswith("/"):
            decoded_path = "/" + decoded_path

        # Use the decoded path as absolute path
        full_path = Path(decoded_path)

        if not full_path.exists():
            return jsonify({"error": f"Image not found: {full_path}"}), 404

        # Determine MIME type
        mime_type, _ = mimetypes.guess_type(str(full_path))
        if mime_type is None:
            mime_type = "application/octet-stream"

        return send_file(full_path, mimetype=mime_type)

    except Exception as e:
        return jsonify({"error": f"Error serving image: {str(e)}"}), 500


@app.route("/api/mask/<path:mask_path>")
def serve_mask(mask_path: str):
    """Serve mask files as JSON array."""
    try:
        # URL decode the mask path
        decoded_path = unquote(mask_path)

        # Flask's <path:> route strips the leading /, so add it back for absolute paths
        if not decoded_path.startswith("/"):
            decoded_path = "/" + decoded_path

        # Use the decoded path as absolute path
        full_path = Path(decoded_path)

        if not full_path.exists():
            return jsonify({"error": f"Mask not found: {full_path}"}), 404

        # Load numpy array and convert to JSON
        mask_data = np.load(full_path)

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

    # Create JSON data
    data = {
        "report_name": filtered_report.name,
        "metadata": filtered_report.metadata,
        "filters": {
            "source_class": source_class,
            "target_class": target_class,
            "min_score": min_score,
            "max_score": max_score,
        },
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
    parser.add_argument("--report-path", type=str, help="Path to report JSON file")
    parser.add_argument(
        "--report-dir", type=str, help="Path to directory containing reports"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to run the server on"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to run the server on"
    )
    parser.add_argument("--debug", action="store_true", help="Run in debug mode")

    args = parser.parse_args()

    if not args.report_path and not args.report_dir:
        parser.error("Either --report-path or --report-dir must be provided")

    global current_report, report_base_path

    try:
        if args.report_path:
            current_report = load_report_from_path(args.report_path)
            report_base_path = Path(args.report_path).parent.resolve()
        else:
            current_report = load_report_from_path(args.report_dir)
            report_base_path = Path(args.report_dir).resolve()

        print(f"Loaded report: {current_report.name}")
        print(f"Number of explanations: {len(current_report)}")
        print(f"Report base path: {report_base_path}")

    except Exception as e:
        print(f"Error loading report: {e}")
        return

    print(f"Starting QuAC Visualizer on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
