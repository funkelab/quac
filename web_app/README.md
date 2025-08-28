# QuAC Visualizer

A web-based GUI for exploring Explanation and Report objects from the QuAC (Quantitative Attribution with Counterfactuals) framework.

## Features

- **Toggle-based Image Viewer**: Switch between query and counterfactual images to easily spot subtle differences
- **Mask Overlay System**: View importance masks as opacity overlays on images
- **Color-blind Friendly Design**: Purple/green theme optimized for accessibility
- **Advanced Filtering**: Filter by source class, target class, and QuAC score ranges
- **Interactive Visualizations**: QuAC curves for both aggregate and individual explanations
- **Download Functionality**: Export filtered results as JSON or organized image packages

## Installation

### Using uv (Recommended)

The visualizer uses uv's script dependencies feature, so no separate installation is needed:

```bash
# Run directly with uv (will automatically install dependencies)
uv run web_app/quac_visualizer.py --report-path /path/to/your/report.json
```

### Traditional Installation

Alternatively, you can install the web dependencies manually:

```bash
# Install the main QuAC package with web dependencies
pip install -e .[web]

# Or install QuAC package separately and add web dependencies
pip install flask werkzeug
```

## Usage

### Using uv (Recommended)

```bash
# Load a specific report file
uv run web_app/quac_visualizer.py --report-path /path/to/your/report.json

# Load from a directory with multiple reports
uv run web_app/quac_visualizer.py --report-dir /path/to/reports/directory

# Additional options
uv run web_app/quac_visualizer.py --report-path /path/to/report.json \
    --host 0.0.0.0 \
    --port 8000 \
    --debug
```

### Traditional Usage

```bash
cd web_app
python quac_visualizer.py --report-path /path/to/your/report.json
```

### Options

- `--report-path`: Path to a specific report JSON file
- `--report-dir`: Path to directory containing report subdirectories
- `--host`: Host to bind to (default: 127.0.0.1)
- `--port`: Port to run on (default: 5000)
- `--debug`: Enable Flask debug mode

## Testing with Dummy Data

To test the functionality, you can use the provided test example:

```bash
uv run web_app/quac_visualizer.py --report-path web_app/test_example/report.json
```

This will load a dummy report with sample images and masks to demonstrate all features.

## Interface Guide

### Main Components

1. **Sidebar (Left Panel)**:
   - **Report Info**: Basic statistics about the loaded report
   - **Filters**: Source class, target class, and score range filters
   - **QuAC Curve**: Aggregate curve visualization for filtered data
   - **Downloads**: Export filtered results

2. **Explanation List (Middle Panel)**:
   - Scrollable list of explanations sorted by QuAC score
   - Color-coded score badges (green=high, yellow=medium, red=low)
   - Click to select an explanation for detailed viewing

3. **Image Viewer (Right Panel)**:
   - Toggle between Query and Counterfactual images
   - Optional mask overlay with opacity-based visualization
   - Individual QuAC curve for the selected explanation
   - Explanation metadata display

### Controls

#### Image Navigation
- **Query/Counterfactual Toggle**: Radio buttons to switch between images
- **Mask Overlay**: Checkbox to show/hide importance mask
- **Keyboard Shortcuts**:
  - `Q`: Show query image
  - `C`: Show counterfactual image
  - `M`: Toggle mask overlay
  - `Escape`: Close image viewer

#### Filtering
- **Source Class**: Filter by the original class of images
- **Target Class**: Filter by the target class for transformations
- **Score Ranges**: Use sliders to filter by QuAC score thresholds
- **Apply Filters**: Updates both the explanation list and QuAC curve

#### Downloads
- **Download Filtered JSON**: Export current filtered explanations as JSON
- **Download Filtered Images**: Export images and masks as organized ZIP file

### Mask Overlay Logic

The mask overlay uses an opacity-based visualization:
- **Higher mask values (→ 1.0)**: Lower opacity overlay → **more visible** image regions
- **Lower mask values (→ 0.0)**: Higher opacity overlay → **more hidden** image regions

This means bright/unmasked areas in the overlay indicate regions the model found **less important**, while transparent areas show **important regions**.

### Understanding the Data

#### Explanation Objects
Each explanation contains:
- **Images**: Query (original), counterfactual (transformed), and importance mask
- **Predictions**: Model confidence scores for each image
- **Classes**: Source and target class information
- **QuAC Score**: Overall quality metric (higher = better explanation)
- **Curve Data**: Mask size vs. score change for detailed analysis

#### Report Structure
Reports aggregate multiple explanations and provide:
- **Filtering methods**: Select subsets by class or score
- **Curve analysis**: Statistical summaries across explanations
- **Metadata**: Information about the evaluation process

## Expected Data Format

### Report JSON Structure
```json
{
  "name": "report_name",
  "metadata": {},
  "results": [
    {
      "query_path": "path/to/query_image.jpg",
      "counterfactual_path": "path/to/counterfactual_image.jpg", 
      "mask_path": "path/to/mask.npy",
      "query_prediction": [0.1, 0.9, 0.0],
      "counterfactual_prediction": [0.0, 0.1, 0.9],
      "source_class": 1,
      "target_class": 2,
      "score": 0.85,
      "normalized_mask_sizes": [0.0, 0.1, 0.2, ...],
      "score_changes": [0.0, 0.1, 0.3, ...],
      "method": "attribution_method_name"
    }
  ]
}
```

### File Organization
```
your_data/
├── report.json
├── images/
│   ├── query_001.jpg
│   ├── counterfactual_001.jpg
│   └── ...
└── masks/
    ├── mask_001.npy
    └── ...
```

## Troubleshooting

### Common Issues

1. **Images not loading**: Check that image paths in the report JSON are correct relative to the report file location

2. **Mask overlay not showing**: Ensure mask files are valid NumPy arrays with values between 0-1

3. **QuAC curves not displaying**: Verify that `normalized_mask_sizes` and `score_changes` arrays are present and have matching lengths

4. **Memory issues with large datasets**: Use the pagination (load more) feature for reports with many explanations

### Performance Tips

- For large reports, consider filtering to reduce the number of explanations loaded
- The web interface loads images on-demand to minimize memory usage
- Use the download functionality to work with subsets of data offline

## Development

The visualizer is built with:
- **Backend**: Flask with REST API endpoints
- **Frontend**: Vanilla JavaScript with Bootstrap 5 and Chart.js
- **Styling**: Color-blind friendly purple/green theme

To modify or extend the visualizer, see the source files:
- `quac_visualizer.py`: Flask backend and API endpoints
- `templates/index.html`: Main interface template
- `static/css/style.css`: Styling and color theme
- `static/js/app.js`: JavaScript application logic