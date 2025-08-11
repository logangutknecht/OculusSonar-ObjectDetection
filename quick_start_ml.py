"""
Quick Start Script for ML-based Sonar Object Detection
Run this to get started with machine learning detection quickly!
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import logging

# Add src to path
sys.path.append('src')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required dependencies are installed"""
    required = {
        'cv2': 'opencv-python',
        'torch': 'torch',
        'ultralytics': 'ultralytics',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("Missing dependencies. Please install:")
        print(f"pip install {' '.join(missing)}")
        return False
    return True

def extract_sample_frames():
    """Extract sample frames from existing sonar files"""
    from sonar_processor import OculusFileReader
    
    # Create directories
    Path("training_data/raw_images").mkdir(parents=True, exist_ok=True)
    
    # Find a sonar file
    sonar_files = list(Path("raw_assets").glob("*.oculus"))
    
    if not sonar_files:
        logger.error("No .oculus files found in raw_assets/")
        return False
    
    # Use first file
    sonar_file = sonar_files[0]
    logger.info(f"Extracting frames from {sonar_file}")
    
    # Extract frames
    reader = OculusFileReader(str(sonar_file))
    frames = reader.parse_all_frames()
    
    # Save every 10th frame (to get variety)
    saved = 0
    for i in range(0, len(frames), 10):
        frame = frames[i]
        img = frame.intensity_data
        
        # Normalize and save
        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        output_path = Path("training_data/raw_images") / f"frame_{i:04d}.png"
        cv2.imwrite(str(output_path), img_norm)
        saved += 1
    
    logger.info(f"Extracted {saved} sample frames to training_data/raw_images/")
    return True

def create_sample_annotations():
    """Create sample annotations for demonstration"""
    import json
    
    # Create sample annotations (you should replace with real annotations)
    Path("training_data/images").mkdir(exist_ok=True)
    Path("training_data/labels").mkdir(exist_ok=True)
    
    # Create classes file
    classes = ["object", "sphere", "debris"]
    with open("training_data/classes.txt", "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    logger.info("Created sample annotation structure in training_data/")
    logger.info("Please use the annotation tool to create real annotations:")
    logger.info("  python src/ml_detector/data_annotator.py")
    
    return True

def train_simple_model():
    """Train a simple model for demonstration"""
    from ml_detector.train_sonar_detector import train_from_scratch
    
    # Check if we have annotated data
    if not Path("training_data/images").exists() or \
       len(list(Path("training_data/images").glob("*.png"))) == 0:
        logger.warning("No annotated images found. Please annotate data first.")
        logger.info("Run: python src/ml_detector/data_annotator.py")
        return None
    
    # Train with minimal epochs for quick demo
    logger.info("Training model (this may take a while)...")
    model_path = train_from_scratch(
        "training_data",
        epochs=10,  # Very few epochs for quick demo
        batch_size=4
    )
    
    return model_path

def test_detection():
    """Test detection with a trained model or classical detector"""
    from ml_detector.ml_sonar_detector import HybridSonarDetector
    from specialized_detector import SpecializedSonarDetector
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Find a test image
    test_images = list(Path("training_data/raw_images").glob("*.png"))
    
    if not test_images:
        logger.error("No test images found")
        return
    
    # Load test image
    test_img_path = test_images[0]
    img = cv2.imread(str(test_img_path), cv2.IMREAD_GRAYSCALE)
    
    # Try to find a trained model
    model_files = list(Path("models").glob("*.pt")) if Path("models").exists() else []
    
    if model_files:
        # Use ML detector
        logger.info(f"Using ML model: {model_files[0]}")
        detector = HybridSonarDetector(
            ml_model_path=str(model_files[0]),
            use_classical=True,
            use_specialized=True
        )
    else:
        # Use classical detector
        logger.info("No ML model found, using classical detector")
        detector = SpecializedSonarDetector(shadow_analysis=True)
    
    # Detect objects
    detections = detector.detect(img)
    
    # Visualize results
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Detection Test - Found {len(detections)} objects")
    
    for det in detections:
        x, y, w, h = det.bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, f'{det.class_name} ({det.confidence:.2f})',
               color='yellow', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    logger.info(f"Detected {len(detections)} objects")

def main():
    """Main quick start workflow"""
    print("=" * 60)
    print("SONAR ML DETECTION - QUICK START")
    print("=" * 60)
    
    # Step 1: Check dependencies
    print("\n1. Checking dependencies...")
    if not check_dependencies():
        return
    print("   ✓ All dependencies installed")
    
    # Step 2: Extract sample frames
    print("\n2. Extracting sample frames...")
    if extract_sample_frames():
        print("   ✓ Sample frames extracted")
    
    # Step 3: Create annotation structure
    print("\n3. Setting up annotation structure...")
    create_sample_annotations()
    
    # Step 4: Provide instructions
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("\n1. ANNOTATE YOUR DATA:")
    print("   Run: python src/ml_detector/data_annotator.py")
    print("   - Draw bounding boxes around objects")
    print("   - Save annotations (press 's')")
    print("   - Aim for at least 50-100 annotations")
    
    print("\n2. TRAIN YOUR MODEL:")
    print("   Run: python src/ml_detector/train_sonar_detector.py --data training_data")
    print("   - This will train a YOLOv8 model on your data")
    print("   - Training may take 30-60 minutes depending on data size")
    
    print("\n3. USE YOUR MODEL:")
    print("   Update config.yaml:")
    print("     detection:")
    print("       use_ml: true")
    print("       ml_model_path: 'models/your_model.pt'")
    print("\n   Then run: python main.py raw_assets/your_file.oculus")
    
    print("\n4. TEST RIGHT NOW (with classical detector):")
    response = input("   Would you like to test detection now? (y/n): ")
    
    if response.lower() == 'y':
        test_detection()
    
    print("\n" + "=" * 60)
    print("For detailed instructions, see ML_DETECTION_GUIDE.md")
    print("=" * 60)

if __name__ == "__main__":
    main()
