# Machine Learning Based Sonar Object Detection Guide

## Overview

This guide explains how to use the new machine learning (ML) based detection system for sonar object detection. The system uses YOLOv8, a state-of-the-art object detection model, optimized specifically for sonar imagery.

## Table of Contents
1. [Data Preparation](#data-preparation)
2. [Training Your Model](#training-your-model)
3. [Using the Trained Model](#using-the-trained-model)
4. [Validation and Testing](#validation-and-testing)
5. [Tips for Better Results](#tips-for-better-results)

## Data Preparation

### Step 1: Organize Your Sonar Images

First, extract frames from your .oculus files and save them as images:

```python
from src.sonar_processor import OculusFileReader
import cv2
from pathlib import Path

# Extract frames from .oculus file
reader = OculusFileReader("raw_assets/your_sonar_file.oculus")
frames = reader.parse_all_frames()

# Save frames as images
output_dir = Path("training_data/raw_images")
output_dir.mkdir(exist_ok=True, parents=True)

for i, frame in enumerate(frames):
    img = frame.intensity_data
    # Normalize to 0-255 range
    img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(str(output_dir / f"frame_{i:04d}.png"), img_normalized)
```

### Step 2: Annotate Your Data

Use the provided annotation tool to mark objects in your sonar images:

```bash
# Run the annotation tool
python src/ml_detector/data_annotator.py
```

The annotation tool provides an interactive interface where you can:
- Draw bounding boxes around detected objects
- Assign class labels (sphere, barrel, cube, fish, debris, or custom)
- Save annotations in YOLO format automatically

**Annotation Tips:**
- Draw tight bounding boxes around the bright sonar returns
- Include the shadow region if visible (helps the model learn object height)
- Aim for at least 100-200 annotated objects per class
- Include diverse examples (different ranges, angles, intensities)

### Step 3: Prepare Training Data Structure

Your annotated data will be automatically organized as:
```
training_data/
├── images/        # Sonar images
├── labels/        # YOLO format annotations
└── classes.txt    # Class names
```

## Training Your Model

### Basic Training

Train a model with default settings optimized for sonar:

```bash
# Train with default settings
python src/ml_detector/train_sonar_detector.py --data training_data
```

### Advanced Training Options

Customize training parameters for your specific needs:

```python
from src.ml_detector.train_sonar_detector import SonarModelTrainer

# Initialize trainer with custom settings
trainer = SonarModelTrainer(
    data_dir="training_data",
    project_name="my_sonar_detector"
)

# Customize training parameters
trainer.config.update({
    'epochs': 150,              # More epochs for better accuracy
    'batch_size': 8,            # Smaller batch if limited GPU memory
    'imgsz': 640,               # Image size (640 is good balance)
    'learning_rate': 0.01,      # Initial learning rate
    'patience': 30,             # Early stopping patience
    'conf_threshold': 0.25,     # Lower for sonar (harder detection)
})

# Train the model
trainer.train()

# Evaluate performance
metrics = trainer.evaluate()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"Precision: {metrics.box.mp:.3f}")
print(f"Recall: {metrics.box.mr:.3f}")

# Export for deployment
model_path = trainer.export_model('onnx')  # or 'torchscript'
```

### Training Progress

Monitor training progress in the `training_runs/` directory:
- `train/weights/best.pt` - Best model weights
- `train/weights/last.pt` - Latest checkpoint
- Training curves and metrics visualizations

## Using the Trained Model

### Method 1: Update Configuration File

Edit `config.yaml` to use your trained model:

```yaml
detection:
  use_ml: true                    # Enable ML detection
  use_hybrid: false                # Or true for ML+Classical fusion
  ml_model_path: "models/sonar_yolo_best_20240101_120000.pt"
  ml_confidence_threshold: 0.25   # Adjust based on your needs
  ml_iou_threshold: 0.45          # For overlapping detections
  fusion_method: 'weighted'       # For hybrid mode: 'weighted', 'voting', or 'nms'
```

Then run normally:
```bash
python main.py raw_assets/your_sonar_file.oculus --save-outputs
```

### Method 2: Direct Python Usage

```python
from src.ml_detector.ml_sonar_detector import MLSonarDetector
import cv2

# Initialize detector
detector = MLSonarDetector(
    model_path="models/your_trained_model.pt",
    confidence_threshold=0.25,
    device='cuda'  # or 'cpu'
)

# Load and process image
img = cv2.imread("sonar_frame.png", cv2.IMREAD_GRAYSCALE)
detections = detector.detect(img, preprocess=True)

# Process detections
for det in detections:
    print(f"Found {det.class_name} at {det.bbox} with confidence {det.confidence:.2f}")
```

### Method 3: Hybrid Detection (ML + Classical)

Combine ML with physics-based detection for best results:

```python
from src.ml_detector.ml_sonar_detector import HybridSonarDetector

# Initialize hybrid detector
detector = HybridSonarDetector(
    ml_model_path="models/your_model.pt",
    use_classical=True,
    use_specialized=True,
    fusion_method='weighted'
)

# Detect objects
detections = detector.detect(sonar_image)
```

## Validation and Testing

### Validate on Test Dataset

```bash
# Validate model performance
python src/ml_detector/validate_model.py \
    --model models/your_model.pt \
    --dataset validation_data
```

### Test on Live Sonar Data

```bash
# Test on actual sonar recordings
python src/ml_detector/validate_model.py \
    --model models/your_model.pt \
    --sonar-file raw_assets/test_recording.oculus
```

### Compare Detection Methods

```bash
# Compare ML vs Classical vs Hybrid
python src/ml_detector/validate_model.py \
    --model models/your_model.pt \
    --compare test_image.png
```

## Tips for Better Results

### 1. Data Quality
- **Annotation Consistency**: Be consistent in how you draw bounding boxes
- **Balance Classes**: Try to have similar numbers of examples for each class
- **Include Hard Examples**: Add difficult cases (low contrast, partial occlusions)
- **Shadow Information**: Include shadow regions when visible - they indicate 3D structure

### 2. Data Augmentation
The training pipeline automatically applies sonar-specific augmentations:
- Intensity variations (simulates different sonar gains)
- Noise addition (realistic sonar noise)
- Motion blur (platform movement)
- Horizontal flipping (valid for most sonar scenarios)

### 3. Model Selection
- Start with `yolov8n.pt` (nano) for quick training and testing
- Use `yolov8s.pt` (small) or `yolov8m.pt` (medium) for better accuracy
- Larger models need more training data and time

### 4. Hyperparameter Tuning
Key parameters for sonar detection:
```python
# Sonar-optimized settings
config = {
    'conf_threshold': 0.2,    # Lower than typical CV (sonar is harder)
    'iou_threshold': 0.4,     # Slightly lower for fuzzy sonar boundaries
    'imgsz': 640,             # Good balance of speed/accuracy
    'batch_size': 16,         # Adjust based on GPU memory
    'mosaic': 0.3,            # Reduced mosaic for sonar
    'hsv_v': 0.3,             # Brightness augmentation only (grayscale)
}
```

### 5. Performance Optimization
- **GPU Acceleration**: Use CUDA if available (10-50x faster)
- **Batch Processing**: Process multiple frames together
- **Model Warmup**: Call `detector.warmup()` before processing
- **ONNX Export**: Use ONNX format for production deployment

### 6. Troubleshooting

**Low Detection Rate:**
- Lower confidence threshold (try 0.15-0.25)
- Check if preprocessing is appropriate
- Ensure training data matches test conditions

**Too Many False Positives:**
- Increase confidence threshold
- Use NMS with lower IOU threshold
- Consider hybrid detector with voting

**Model Not Learning:**
- Check annotations are correct
- Ensure sufficient training data (>100 examples)
- Verify images are properly normalized

## Example Workflow

Here's a complete workflow from raw sonar data to deployed detector:

```python
# 1. Extract frames from sonar files
from src.sonar_processor import OculusFileReader
import cv2

reader = OculusFileReader("raw_assets/training_data.oculus")
frames = reader.parse_all_frames()

for i, frame in enumerate(frames[::10]):  # Every 10th frame
    img = cv2.normalize(frame.intensity_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    cv2.imwrite(f"training_data/raw_images/frame_{i:04d}.png", img)

# 2. Annotate images (run annotation tool)
# python src/ml_detector/data_annotator.py

# 3. Train model
from src.ml_detector.train_sonar_detector import train_from_scratch

model_path = train_from_scratch(
    "training_data",
    epochs=100,
    batch_size=16
)

# 4. Update configuration
import yaml

with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

config['detection']['use_ml'] = True
config['detection']['ml_model_path'] = model_path

with open("config.yaml", 'w') as f:
    yaml.dump(config, f)

# 5. Run detection pipeline
import subprocess

subprocess.run([
    "python", "main.py", 
    "raw_assets/test_file.oculus",
    "--save-outputs"
])
```

## Support and Resources

- **YOLOv8 Documentation**: https://docs.ultralytics.com/
- **Computer Vision for Sonar**: Research papers on sonar object detection
- **Data Augmentation**: Experiment with sonar-specific augmentations

Remember: Machine learning models improve with more and better data. Start small, iterate, and gradually build up your dataset for best results!
