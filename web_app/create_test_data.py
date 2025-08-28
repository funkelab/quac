#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=2.0",
#     "pillow>=9.0.0",
# ]
# ///
"""
Create dummy test data for the QuAC Visualizer.
This generates sample images and masks to demonstrate functionality.
"""

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


def create_test_images():
    """Create dummy test images and masks."""

    # Create directories
    test_dir = Path("test_example")
    test_dir.mkdir(exist_ok=True)
    (test_dir / "images").mkdir(exist_ok=True)
    (test_dir / "masks").mkdir(exist_ok=True)

    # Image dimensions
    img_size = (256, 256)

    # Sample classes
    classes = ["dog", "cat", "bird"]
    colors = {
        "dog": (139, 69, 19),  # Brown
        "cat": (128, 128, 128),  # Gray
        "bird": (65, 105, 225),  # Blue
    }

    explanations = []

    # Generate dummy explanations
    for i in range(15):
        source_class = classes[i % 3]
        target_class = classes[(i + 1) % 3]

        # Create query image (simple colored rectangle with text)
        query_img = Image.new("RGB", img_size, color=(240, 240, 240))
        draw = ImageDraw.Draw(query_img)

        # Draw main shape
        shape_color = colors[source_class]
        if i % 3 == 0:  # Rectangle
            draw.rectangle(
                [50, 50, 200, 200], fill=shape_color, outline=(0, 0, 0), width=2
            )
        elif i % 3 == 1:  # Circle
            draw.ellipse(
                [50, 50, 200, 200], fill=shape_color, outline=(0, 0, 0), width=2
            )
        else:  # Triangle
            draw.polygon(
                [(125, 50), (50, 200), (200, 200)],
                fill=shape_color,
                outline=(0, 0, 0),
                width=2,
            )

        # Add some distinguishing features
        if i % 4 == 0:  # Add dots
            draw.ellipse([80, 80, 100, 100], fill=(255, 255, 255))
            draw.ellipse([150, 150, 170, 170], fill=(255, 255, 255))
        elif i % 4 == 1:  # Add lines
            draw.line([70, 70, 180, 180], fill=(255, 255, 255), width=3)
            draw.line([70, 180, 180, 70], fill=(255, 255, 255), width=3)

        try:
            # Try to add text (may fail if no font available)
            font = ImageFont.load_default()
            draw.text((10, 10), source_class, fill=(0, 0, 0), font=font)
        except (OSError, ImportError):
            # Fallback without font
            draw.text((10, 10), source_class, fill=(0, 0, 0))

        query_path = test_dir / "images" / f"query_{i:03d}.png"
        query_img.save(query_path)

        # Create counterfactual image (similar but with target class color and slight modification)
        cf_img = query_img.copy()
        draw_cf = ImageDraw.Draw(cf_img)

        # Change main shape color
        target_color = colors[target_class]
        if i % 3 == 0:  # Rectangle
            draw_cf.rectangle(
                [50, 50, 200, 200], fill=target_color, outline=(0, 0, 0), width=2
            )
        elif i % 3 == 1:  # Circle
            draw_cf.ellipse(
                [50, 50, 200, 200], fill=target_color, outline=(0, 0, 0), width=2
            )
        else:  # Triangle
            draw_cf.polygon(
                [(125, 50), (50, 200), (200, 200)],
                fill=target_color,
                outline=(0, 0, 0),
                width=2,
            )

        # Add small difference (this is what the user should spot when toggling)
        if i % 2 == 0:
            # Add a small square in corner
            draw_cf.rectangle([210, 210, 230, 230], fill=(255, 0, 0))
        else:
            # Add a small circle
            draw_cf.ellipse([20, 210, 40, 230], fill=(0, 255, 0))

        try:
            draw_cf.text((10, 230), target_class, fill=(0, 0, 0), font=font)
        except (OSError, ImportError):
            draw_cf.text((10, 230), target_class, fill=(0, 0, 0))

        cf_path = test_dir / "images" / f"counterfactual_{i:03d}.png"
        cf_img.save(cf_path)

        # Create mask (importance map)
        # Higher values (closer to 1) where the difference is
        mask = np.ones((256, 256), dtype=np.float32) * 0.2  # Base low importance

        if i % 2 == 0:
            # High importance around the small red square
            mask[200:240, 200:240] = 0.9
        else:
            # High importance around the small green circle
            mask[200:240, 10:50] = 0.9

        # Medium importance around the main shape
        mask[40:210, 40:210] = 0.6

        # Add some noise
        noise = np.random.normal(0, 0.1, mask.shape)
        mask = np.clip(mask + noise, 0, 1)

        mask_path = test_dir / "masks" / f"mask_{i:03d}.npy"
        np.save(mask_path, mask)

        # Generate QuAC curve data
        mask_sizes = np.linspace(0, 1, 21)
        # Score changes should decrease as mask size increases (more masking = lower score)
        base_score = 0.8 + 0.2 * np.random.random()  # Random base between 0.8-1.0
        score_changes = []
        for size in mask_sizes:
            # Simulate decreasing performance as more is masked
            change = base_score * (1 - size * 0.7) + np.random.normal(0, 0.05)
            score_changes.append(max(0, change))

        # Overall QuAC score (area under curve)
        quac_score = np.trapz(score_changes, mask_sizes)

        # Create explanation entry
        explanation = {
            "query_path": f"images/query_{i:03d}.png",
            "counterfactual_path": f"images/counterfactual_{i:03d}.png",
            "mask_path": f"masks/mask_{i:03d}.npy",
            "query_prediction": (
                [0.1, 0.1, 0.8]
                if source_class == "dog"
                else ([0.8, 0.1, 0.1] if source_class == "cat" else [0.1, 0.8, 0.1])
            ),
            "counterfactual_prediction": (
                [0.1, 0.1, 0.8]
                if target_class == "dog"
                else ([0.8, 0.1, 0.1] if target_class == "cat" else [0.1, 0.8, 0.1])
            ),
            "source_class": classes.index(source_class),
            "target_class": classes.index(target_class),
            "score": float(quac_score),
            "normalized_mask_sizes": mask_sizes.tolist(),
            "score_changes": score_changes,
            "optimal_threshold": 0.5,
            "method": ["gradcam", "integrated_gradients", "lime"][i % 3],
        }

        explanations.append(explanation)

    # Create report JSON
    report = {
        "name": "test_report",
        "metadata": {
            "created_by": "test_data_generator",
            "description": "Dummy data for QuAC Visualizer testing",
            "num_classes": len(classes),
            "class_names": classes,
        },
        "results": explanations,
    }

    report_path = test_dir / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Created test data in {test_dir}/")
    print(f"Generated {len(explanations)} explanations")
    print(f"Report saved to: {report_path}")
    print("\nTo test the visualizer, run:")
    print(f"uv run web_app/quac_visualizer.py --report-path {report_path}")


if __name__ == "__main__":
    create_test_images()
