"""
Annotation script for RAW unfiltered sonar images
Uses the raw frames for better object visibility
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append('src')

from ml_detector.data_annotator import SonarAnnotator

def main():
    print("="*60)
    print("RAW SONAR IMAGE ANNOTATOR")
    print("Using unfiltered images for better object visibility")
    print("="*60)
    
    # Use raw unfiltered images
    input_dir = Path("training_data/raw_unfiltered")
    output_dir = Path("training_data")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"\nError: Raw image directory '{input_dir}' does not exist!")
        print("Run: py extract_raw_frames.py")
        return
    
    # Count images
    image_files = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"\nNo images found in {input_dir}")
        return
    
    print(f"\nFound {len(image_files)} RAW unfiltered images")
    print("These images preserve original sonar intensity without filtering")
    print(f"Annotations will be saved to {output_dir}")
    
    print("\n" + "="*60)
    print("WHAT TO LOOK FOR IN RAW SONAR:")
    print("="*60)
    print("• BRIGHT SPOTS: Strong sonar returns from objects")
    print("• DARK SHADOWS: Acoustic shadows behind objects (important!)")
    print("• CRESCENTS: Spherical objects appear as bright crescents")
    print("• RECTANGLES: Boxes/cubes show as rectangular bright areas")
    print("• Include both the bright object AND its shadow in the box")
    print("="*60)
    
    print("\nANNOTATION CONTROLS:")
    print("="*60)
    print("  Click & Drag: Draw bounding box")
    print("  1: object (general)")
    print("  2: sphere")
    print("  3: barrel")
    print("  4: cube")
    print("  5: fish")
    print("  6: debris")
    print("  S: Save and next")
    print("  U: Undo last box")
    print("  C: Clear all")
    print("  Space: Skip image")
    print("  Q: Quit")
    print("="*60)
    
    print("\nTIPS FOR RAW SONAR ANNOTATION:")
    print("• Objects appear as bright areas against darker background")
    print("• Shadows (dark areas) directly behind bright spots confirm 3D objects")
    print("• Box should include both object and shadow")
    print("• If unsure, skip the image (Space key)")
    
    input("\nPress ENTER to start annotating raw sonar images...")
    
    # Create annotator and start
    annotator = SonarAnnotator()
    annotator.annotate_directory(str(input_dir), str(output_dir))
    
    print("\nAnnotation complete!")
    
    # Show summary
    labels_dir = output_dir / "labels"
    if labels_dir.exists():
        label_files = list(labels_dir.glob("*.txt"))
        if label_files:
            print(f"\nCreated {len(label_files)} annotation files")
            print(f"Annotations saved in: {labels_dir}")
            print("\nNext steps:")
            print("1. Continue annotating more images if needed")
            print("2. Train your model:")
            print("   py src/ml_detector/train_sonar_detector.py --data training_data --epochs 50")

if __name__ == "__main__":
    main()

