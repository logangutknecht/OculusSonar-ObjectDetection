# Sonar Data Annotation Guide

## Overview

The updated `data_annotator.py` tool now supports direct annotation of frames extracted from `.oculus` sonar files. This streamlines the workflow for creating training data for YOLOv8 object detection models.

## Training Data Requirements

For effective YOLOv8 model training on sonar imagery:

| Performance Level | Images per Class | Description |
|------------------|------------------|-------------|
| **Minimum** | 100-200 | Basic detection, prone to false positives/negatives |
| **Good** | 500-1000 | Decent accuracy, suitable for testing |
| **Recommended** | 1000-5000 | Robust detection, production-ready |
| **Optimal** | 5000+ | Best performance, handles edge cases well |

**Important:** Taking every 10th frame from multiple videos to get thousands of training images is **NOT overkill** - it's actually ideal for robust model training, especially for challenging sonar imagery.

## Quick Start

### Method 1: Command Line (Recommended)

```bash
# List available .oculus files
python src/ml_detector/data_annotator.py --list-files

# Annotate specific file with default settings (every 10th frame)
python src/ml_detector/data_annotator.py --oculus-file raw_assets/Oculus_20250805_164829.oculus

# Custom frame interval (e.g., every 5th frame for more data)
python src/ml_detector/data_annotator.py --oculus-file raw_assets/Oculus_20250805_164829.oculus --frame-interval 5

# Keep extracted frames after annotation
python src/ml_detector/data_annotator.py --oculus-file raw_assets/Oculus_20250805_164829.oculus --keep-frames

# Specify custom output directory
python src/ml_detector/data_annotator.py --oculus-file raw_assets/Oculus_20250805_164829.oculus --output-dir training_data/my_annotations
```

### Method 2: Interactive Menu

```bash
python src/ml_detector/data_annotator.py
```

Then select option 1 to annotate frames from an .oculus file. The tool will:
1. Show you all available .oculus files in `raw_assets/`
2. Let you select which file to annotate
3. Ask for frame interval (default: every 10th frame)
4. Extract frames and guide you through annotation

## Annotation Controls

During annotation:

| Key | Action |
|-----|--------|
| **Click & Drag** | Draw bounding box |
| **1-6** | Select object class |
| **u** | Undo last annotation |
| **c** | Clear all annotations |
| **s** | Save and continue to next image |
| **Space** | Skip current image |
| **q** | Quit (without saving current image) |

## Object Classes

Default classes for sonar objects:
1. **object** - General/unknown objects
2. **sphere** - Spherical objects (balls, buoys)
3. **barrel** - Cylindrical objects
4. **cube** - Box-shaped objects
5. **fish** - Marine life
6. **debris** - Unidentified debris

## Output Format

The tool generates YOLO-format training data:

```
output_dir/
├── images/           # Annotated images
│   ├── frame_00000.png
│   ├── frame_00010.png
│   └── ...
├── labels/           # YOLO format annotations
│   ├── frame_00000.txt
│   ├── frame_00010.txt
│   └── ...
└── classes.txt       # Class definitions
```

Each label file contains:
```
<class_id> <center_x> <center_y> <width> <height>
```
All values normalized to [0, 1].

## Workflow for Multiple Videos

To build a comprehensive training dataset:

```bash
# Annotate multiple videos with consistent settings
for file in raw_assets/*.oculus; do
    echo "Processing $file..."
    python src/ml_detector/data_annotator.py \
        --oculus-file "$file" \
        --frame-interval 10 \
        --output-dir "training_data/annotations/$(basename $file .oculus)"
done
```

## Tips for Quality Annotations

1. **Be Consistent**: Use the same criteria for what constitutes an object across all frames
2. **Include Edge Cases**: Annotate partially visible objects and difficult examples
3. **Vary Frame Intervals**: Use different intervals for different videos to increase diversity
4. **Review Annotations**: Periodically review your annotations for consistency
5. **Include Negative Examples**: Some frames without objects help reduce false positives

## Frame Extraction Settings

- **Frame Interval**: Default is 10 (every 10th frame)
  - Smaller values (e.g., 5) = more frames, more similar data
  - Larger values (e.g., 20) = fewer frames, more diverse data
  
- **Processing**: Frames are minimally processed for better visibility:
  - 1st and 99th percentiles used for normalization
  - Preserves original sonar characteristics
  - No heavy filtering that might remove important features

## Training the Model

After annotation, train your YOLOv8 model:

```python
from src.ml_detector.train_sonar_detector import SonarModelTrainer

trainer = SonarModelTrainer(
    data_directory="training_data/annotations",
    model_size='m',  # nano, small, medium, large, xlarge
    epochs=100,
    batch_size=16
)

trainer.train()
model_path = trainer.export_model()
```

## Troubleshooting

- **"No .oculus files found"**: Check that files are in `raw_assets/` directory
- **"Module not found"**: Run from project root directory
- **Window not responding**: Press key only when the OpenCV window is in focus
- **Annotations not saving**: Ensure you press 's' to save before moving to next image

## Next Steps

1. Annotate frames from multiple .oculus files for dataset diversity
2. Aim for at least 1000 annotated images for good performance
3. Split data into train/validation/test sets (typical: 70/20/10)
4. Train YOLOv8 model using the annotated data
5. Validate model performance on test data
6. Deploy using hybrid detection mode for best results
