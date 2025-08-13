"""
Simple annotation script that directly uses training_data folder
No file dialogs - just direct annotation!
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from ml_detector.data_annotator import SonarAnnotator

def main():
    print("="*60)
    print("SIMPLE SONAR IMAGE ANNOTATOR")
    print("="*60)
    
    # Use default directories
    input_dir = Path("training_data/raw_images")
    output_dir = Path("training_data")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"\nError: Input directory '{input_dir}' does not exist!")
        print("\nPlease ensure you have images in 'training_data/raw_images/'")
        print("You can:")
        print("1. Copy your sonar images (PNG/JPG) to training_data/raw_images/")
        print("2. Or run: python quick_start_ml.py to extract sample frames")
        return
    
    # Count images
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    
    if not image_files:
        print(f"\nNo images found in {input_dir}")
        print("Please add PNG or JPG images to this directory.")
        return
    
    print(f"\nFound {len(image_files)} images in {input_dir}")
    print(f"Annotations will be saved to {output_dir}")
    
    print("\n" + "="*60)
    print("ANNOTATION CONTROLS:")
    print("="*60)
    print("  Click & Drag: Draw bounding box")
    print("  1-6: Select class (1=object, 2=sphere, 3=barrel, 4=cube, 5=fish, 6=debris)")
    print("  S: Save and next image")
    print("  U: Undo last box")
    print("  C: Clear all boxes")
    print("  Space: Skip image")
    print("  Q: Quit")
    print("="*60)
    
    input("\nPress ENTER to start annotating...")
    
    # Create annotator and start
    annotator = SonarAnnotator()
    annotator.annotate_directory(str(input_dir), str(output_dir))
    
    print("\nAnnotation complete!")
    
    # Show summary if annotations were created
    labels_dir = output_dir / "labels"
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        if label_files:
            print(f"\nCreated {len(label_files)} annotation files")
            print(f"Annotations saved in: {labels_dir}")
            print("\nYou can now train a model with:")
            print("  python src/ml_detector/train_sonar_detector.py --data training_data")

if __name__ == "__main__":
    main()

