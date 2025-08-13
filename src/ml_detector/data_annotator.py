"""
Sonar Image Annotation Tool for Machine Learning
Helps create training data for ML-based object detection
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import argparse
import tempfile
import shutil

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.sonar_processor import OculusFileReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BoundingBox:
    """Represents a bounding box annotation"""
    
    def __init__(self, x1: int, y1: int, x2: int, y2: int, class_name: str = "object"):
        self.x1 = min(x1, x2)
        self.y1 = min(y1, y2)
        self.x2 = max(x1, x2)
        self.y2 = max(y1, y2)
        self.class_name = class_name
    
    def to_yolo_format(self, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert to YOLO format (center_x, center_y, width, height) normalized to [0, 1]"""
        center_x = (self.x1 + self.x2) / 2 / img_width
        center_y = (self.y1 + self.y2) / 2 / img_height
        width = (self.x2 - self.x1) / img_width
        height = (self.y2 - self.y1) / img_height
        return center_x, center_y, width, height
    
    def to_dict(self) -> Dict:
        """Convert to dictionary format"""
        return {
            'x1': self.x1,
            'y1': self.y1,
            'x2': self.x2,
            'y2': self.y2,
            'class_name': self.class_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BoundingBox':
        """Create from dictionary"""
        return cls(data['x1'], data['y1'], data['x2'], data['y2'], data.get('class_name', 'object'))


class SonarAnnotator:
    """Interactive annotation tool for sonar images"""
    
    def __init__(self, window_name: str = "Sonar Annotator"):
        self.window_name = window_name
        self.current_image = None
        self.display_image = None
        self.annotations = []
        self.current_box = None
        self.drawing = False
        self.start_point = None
        self.current_class = "object"
        self.class_names = ["object", "sphere", "barrel", "cube", "fish", "debris"]
        self.class_colors = {
            "object": (0, 255, 0),      # Green
            "sphere": (255, 0, 0),       # Blue
            "barrel": (0, 0, 255),       # Red
            "cube": (255, 255, 0),       # Cyan
            "fish": (255, 0, 255),       # Magenta
            "debris": (0, 255, 255)      # Yellow
        }
        self.scale_factor = 1.0
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes"""
        # Adjust coordinates for scaling
        x = int(x / self.scale_factor)
        y = int(y / self.scale_factor)
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                # Update display with current box being drawn
                self.display_image = self.current_image.copy()
                self._draw_annotations()
                cv2.rectangle(self.display_image, 
                            (int(self.start_point[0] * self.scale_factor), 
                             int(self.start_point[1] * self.scale_factor)),
                            (int(x * self.scale_factor), int(y * self.scale_factor)),
                            self.class_colors[self.current_class], 2)
                cv2.imshow(self.window_name, self.display_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if self.drawing:
                self.drawing = False
                # Create bounding box
                if self.start_point[0] != x and self.start_point[1] != y:
                    box = BoundingBox(self.start_point[0], self.start_point[1], 
                                    x, y, self.current_class)
                    self.annotations.append(box)
                    self._refresh_display()
    
    def _draw_annotations(self):
        """Draw all annotations on the display image"""
        for box in self.annotations:
            color = self.class_colors.get(box.class_name, (0, 255, 0))
            cv2.rectangle(self.display_image,
                        (int(box.x1 * self.scale_factor), int(box.y1 * self.scale_factor)),
                        (int(box.x2 * self.scale_factor), int(box.y2 * self.scale_factor)),
                        color, 2)
            # Add label
            label = f"{box.class_name}"
            cv2.putText(self.display_image, label,
                       (int(box.x1 * self.scale_factor), int(box.y1 * self.scale_factor - 5)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _refresh_display(self):
        """Refresh the display with current annotations"""
        self.display_image = self.current_image.copy()
        self._draw_annotations()
        cv2.imshow(self.window_name, self.display_image)
    
    def annotate_image(self, image_path: str) -> List[BoundingBox]:
        """
        Annotate a single image
        
        Returns:
            List of bounding box annotations
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"Could not load image: {image_path}")
            return []
        
        # Calculate scale factor to fit window
        max_width = 1200
        max_height = 800
        h, w = img.shape[:2]
        
        scale_w = min(1.0, max_width / w)
        scale_h = min(1.0, max_height / h)
        self.scale_factor = min(scale_w, scale_h)
        
        # Resize for display
        display_h = int(h * self.scale_factor)
        display_w = int(w * self.scale_factor)
        self.current_image = cv2.resize(img, (display_w, display_h))
        self.display_image = self.current_image.copy()
        self.annotations = []
        
        # Create window and set mouse callback
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Display instructions
        print("\n" + "="*50)
        print(f"Annotating: {Path(image_path).name}")
        print("="*50)
        print("Instructions:")
        print("  - Click and drag to draw bounding boxes")
        print("  - Press 1-6 to select class:")
        for i, class_name in enumerate(self.class_names, 1):
            print(f"    {i}: {class_name}")
        print("  - Press 'u' to undo last annotation")
        print("  - Press 'c' to clear all annotations")
        print("  - Press 's' to save and continue")
        print("  - Press 'q' to quit without saving")
        print("  - Press SPACE to skip image")
        print(f"Current class: {self.current_class}")
        print("="*50)
        
        cv2.imshow(self.window_name, self.display_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            # Class selection
            if ord('1') <= key <= ord('6'):
                idx = key - ord('1')
                if idx < len(self.class_names):
                    self.current_class = self.class_names[idx]
                    print(f"Selected class: {self.current_class}")
            
            # Undo last annotation
            elif key == ord('u'):
                if self.annotations:
                    self.annotations.pop()
                    self._refresh_display()
                    print("Undid last annotation")
            
            # Clear all annotations
            elif key == ord('c'):
                self.annotations = []
                self._refresh_display()
                print("Cleared all annotations")
            
            # Save and continue
            elif key == ord('s'):
                print(f"Saved {len(self.annotations)} annotations")
                break
            
            # Skip image
            elif key == ord(' '):
                print("Skipping image")
                self.annotations = []
                break
            
            # Quit
            elif key == ord('q'):
                cv2.destroyWindow(self.window_name)
                return None
        
        cv2.destroyWindow(self.window_name)
        return self.annotations
    
    def extract_frames_from_oculus(self, oculus_file: str, output_dir: str,
                                  frame_interval: int = 10,
                                  apply_minimal_processing: bool = True) -> List[str]:
        """
        Extract frames from .oculus file for annotation
        
        Args:
            oculus_file: Path to .oculus file
            output_dir: Directory to save extracted frames
            frame_interval: Extract every Nth frame (default: 10)
            apply_minimal_processing: Apply minimal contrast enhancement
            
        Returns:
            List of extracted frame paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Extracting frames from {oculus_file}")
        
        # Load sonar data
        reader = OculusFileReader(oculus_file)
        frames = reader.parse_all_frames()
        
        logger.info(f"Found {len(frames)} total frames in file")
        
        # Select frames to extract
        frame_indices = range(0, len(frames), frame_interval)
        logger.info(f"Will extract {len(list(frame_indices))} frames (every {frame_interval}th frame)")
        
        extracted_files = []
        
        for idx in frame_indices:
            frame = frames[idx]
            
            # Get raw intensity data
            raw_data = frame.intensity_data
            
            if apply_minimal_processing:
                # Apply minimal processing for better visibility during annotation
                non_zero = raw_data[raw_data > 0]
                if len(non_zero) > 0:
                    # Use percentiles to avoid outliers
                    p_low = np.percentile(non_zero, 1)
                    p_high = np.percentile(non_zero, 99)
                    
                    # Normalize using percentile range
                    normalized = np.clip(raw_data, p_low, p_high)
                    normalized = ((normalized - p_low) / (p_high - p_low) * 255).astype(np.uint8)
                else:
                    normalized = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            else:
                # Completely raw
                if raw_data.max() > 0:
                    normalized = (raw_data * 255.0 / raw_data.max()).astype(np.uint8)
                else:
                    normalized = raw_data.astype(np.uint8)
            
            # Save frame
            filename = f"{Path(oculus_file).stem}_frame_{idx:05d}.png"
            output_file = output_path / filename
            cv2.imwrite(str(output_file), normalized)
            extracted_files.append(str(output_file))
        
        logger.info(f"Extracted {len(extracted_files)} frames to {output_path}")
        return extracted_files
    
    def annotate_oculus_file(self, oculus_file: str, output_dir: str,
                            frame_interval: int = 10,
                            temp_dir: Optional[str] = None,
                            keep_extracted_frames: bool = False):
        """
        Extract and annotate frames from an .oculus file
        
        Args:
            oculus_file: Path to .oculus file
            output_dir: Directory to save annotations
            frame_interval: Extract every Nth frame
            temp_dir: Temporary directory for extracted frames (auto-created if None)
            keep_extracted_frames: Whether to keep extracted frames after annotation
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for YOLO format
        (output_path / 'images').mkdir(exist_ok=True)
        (output_path / 'labels').mkdir(exist_ok=True)
        
        # Extract frames to temporary directory
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp(prefix="sonar_frames_")
            cleanup_temp = True
        else:
            temp_dir = str(Path(temp_dir))
            Path(temp_dir).mkdir(exist_ok=True, parents=True)
            cleanup_temp = False
        
        print(f"\n{'='*60}")
        print(f"Extracting frames from: {Path(oculus_file).name}")
        print(f"Frame interval: every {frame_interval}th frame")
        print(f"Temporary frames directory: {temp_dir}")
        print(f"{'='*60}\n")
        
        # Extract frames
        extracted_files = self.extract_frames_from_oculus(
            oculus_file, temp_dir, frame_interval, apply_minimal_processing=True
        )
        
        if not extracted_files:
            logger.error("No frames extracted")
            return
        
        print(f"\nReady to annotate {len(extracted_files)} frames")
        print("Press any key to start annotation...")
        input()
        
        # Now annotate the extracted frames
        self._annotate_frame_list(extracted_files, output_path)
        
        # Cleanup temporary directory if requested
        if cleanup_temp and not keep_extracted_frames:
            print(f"\nCleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
        elif keep_extracted_frames:
            print(f"\nExtracted frames kept in: {temp_dir}")
    
    def _annotate_frame_list(self, image_files: List[str], output_path: Path):
        """Helper method to annotate a list of image files"""
        # Annotation statistics
        annotated_count = 0
        skipped_count = 0
        total_annotations = 0
        
        # Class mapping for YOLO
        class_to_id = {name: i for i, name in enumerate(self.class_names)}
        
        # Save class names file
        with open(output_path / 'classes.txt', 'w') as f:
            for class_name in self.class_names:
                f.write(f"{class_name}\n")
        
        # Process each image
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}]")
            
            # Annotate image
            annotations = self.annotate_image(str(img_path))
            
            if annotations is None:  # User quit
                break
            
            if annotations:  # Has annotations
                annotated_count += 1
                total_annotations += len(annotations)
                
                # Copy image to output
                img = cv2.imread(str(img_path))
                img_name = Path(img_path).name
                output_img_path = output_path / 'images' / img_name
                cv2.imwrite(str(output_img_path), img)
                
                # Save YOLO format annotations
                h, w = img.shape[:2]
                label_path = output_path / 'labels' / f"{Path(img_path).stem}.txt"
                
                with open(label_path, 'w') as f:
                    for box in annotations:
                        class_id = class_to_id[box.class_name]
                        cx, cy, bw, bh = box.to_yolo_format(w, h)
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                
                # Also save JSON format for reference
                json_annotations = [box.to_dict() for box in annotations]
                json_path = output_path / 'labels' / f"{Path(img_path).stem}.json"
                with open(json_path, 'w') as f:
                    json.dump({
                        'image': img_name,
                        'width': w,
                        'height': h,
                        'annotations': json_annotations
                    }, f, indent=2)
            else:
                skipped_count += 1
        
        # Print summary
        print("\n" + "="*50)
        print("Annotation Summary:")
        print(f"  Total images processed: {annotated_count + skipped_count}")
        print(f"  Images with annotations: {annotated_count}")
        print(f"  Images skipped: {skipped_count}")
        print(f"  Total bounding boxes: {total_annotations}")
        if annotated_count > 0:
            print(f"  Average boxes per image: {total_annotations/annotated_count:.1f}")
        print(f"  Output directory: {output_path}")
        print("="*50)
    
    def annotate_directory(self, input_dir: str, output_dir: str, 
                          image_extensions: List[str] = ['.png', '.jpg', '.jpeg']):
        """
        Annotate all images in a directory
        
        Args:
            input_dir: Directory containing images
            output_dir: Directory to save annotations
            image_extensions: Valid image file extensions
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for YOLO format
        (output_path / 'images').mkdir(exist_ok=True)
        (output_path / 'labels').mkdir(exist_ok=True)
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'**/*{ext}'))
        
        if not image_files:
            logger.error(f"No images found in {input_dir}")
            return
        
        print(f"\nFound {len(image_files)} images to annotate")
        
        # Use the helper method
        self._annotate_frame_list([str(f) for f in image_files], output_path)


def convert_existing_annotations(input_file: str, output_dir: str, image_dir: str):
    """
    Convert existing annotations from other formats to YOLO format
    
    Args:
        input_file: Path to annotation file (JSON, CSV, etc.)
        output_dir: Directory to save YOLO format annotations
        image_dir: Directory containing the corresponding images
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # This is a template function - implement based on your existing annotation format
    logger.info(f"Converting annotations from {input_file}")
    
    # Example for JSON format
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Process and convert to YOLO format
    # ... implementation depends on your existing format


if __name__ == "__main__":
    import time
    
    parser = argparse.ArgumentParser(description="Sonar Image Annotation Tool")
    parser.add_argument('--oculus-file', type=str, help='Path to specific .oculus file to annotate')
    parser.add_argument('--output-dir', type=str, default='training_data/annotations',
                       help='Output directory for annotations')
    parser.add_argument('--frame-interval', type=int, default=10,
                       help='Extract every Nth frame (default: 10)')
    parser.add_argument('--keep-frames', action='store_true',
                       help='Keep extracted frames after annotation')
    parser.add_argument('--temp-dir', type=str, help='Directory for temporary frame extraction')
    parser.add_argument('--list-files', action='store_true',
                       help='List available .oculus files in raw_assets')
    
    args = parser.parse_args()
    
    # If --list-files is specified, show available files and exit
    if args.list_files:
        raw_assets = Path("raw_assets")
        if raw_assets.exists():
            oculus_files = list(raw_assets.glob("*.oculus"))
            if oculus_files:
                print("\nAvailable .oculus files in raw_assets/:")
                print("="*50)
                for i, file in enumerate(oculus_files, 1):
                    print(f"{i}. {file.name}")
                print("="*50)
                print("\nTo annotate a file, run:")
                print("python src/ml_detector/data_annotator.py --oculus-file raw_assets/<filename>")
            else:
                print("No .oculus files found in raw_assets/")
        else:
            print("raw_assets/ directory not found")
        exit()
    
    # If oculus file is specified via command line
    if args.oculus_file:
        oculus_path = Path(args.oculus_file)
        if not oculus_path.exists():
            print(f"Error: File not found: {args.oculus_file}")
            exit(1)
        
        if not oculus_path.suffix == '.oculus':
            print(f"Error: File must be an .oculus file")
            exit(1)
        
        print(f"\nAnnotating frames from: {oculus_path.name}")
        annotator = SonarAnnotator()
        annotator.annotate_oculus_file(
            str(oculus_path),
            args.output_dir,
            frame_interval=args.frame_interval,
            temp_dir=args.temp_dir,
            keep_extracted_frames=args.keep_frames
        )
    else:
        # Interactive menu
        print("\nSonar Image Annotation Tool")
        print("="*50)
        print("1. Annotate frames from .oculus file")
        print("2. Annotate existing images from directory")
        print("3. Convert existing annotations")
        print("4. Exit")
        
        choice = input("\nEnter choice (1-4): ")
        
        if choice == "1":
            # List available .oculus files
            raw_assets = Path("raw_assets")
            oculus_files = []
            
            if raw_assets.exists():
                oculus_files = list(raw_assets.glob("*.oculus"))
            
            if not oculus_files:
                # Browse for file
                print("\nNo .oculus files found in raw_assets/")
                print("Opening file dialog to select .oculus file...")
                
                root = tk.Tk()
                root.withdraw()
                root.lift()
                root.attributes('-topmost', True)
                root.update()
                time.sleep(0.5)
                
                oculus_file = filedialog.askopenfilename(
                    title="Select .oculus file",
                    filetypes=[("Oculus files", "*.oculus"), ("All files", "*.*")],
                    initialdir=Path.cwd()
                )
                root.destroy()
                
                if not oculus_file:
                    print("No file selected. Exiting.")
                    exit()
            else:
                # Show menu of available files
                print("\nAvailable .oculus files:")
                print("-"*50)
                for i, file in enumerate(oculus_files, 1):
                    # Get file info
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"{i}. {file.name} ({size_mb:.1f} MB)")
                
                print(f"{len(oculus_files) + 1}. Browse for other file...")
                
                while True:
                    try:
                        selection = input(f"\nSelect file (1-{len(oculus_files) + 1}): ")
                        selection = int(selection)
                        if 1 <= selection <= len(oculus_files):
                            oculus_file = str(oculus_files[selection - 1])
                            break
                        elif selection == len(oculus_files) + 1:
                            # Browse for file
                            root = tk.Tk()
                            root.withdraw()
                            root.lift()
                            root.attributes('-topmost', True)
                            root.update()
                            time.sleep(0.5)
                            
                            oculus_file = filedialog.askopenfilename(
                                title="Select .oculus file",
                                filetypes=[("Oculus files", "*.oculus"), ("All files", "*.*")],
                                initialdir=Path.cwd()
                            )
                            root.destroy()
                            
                            if not oculus_file:
                                print("No file selected. Exiting.")
                                exit()
                            break
                        else:
                            print("Invalid selection. Please try again.")
                    except ValueError:
                        print("Invalid input. Please enter a number.")
            
            # Get frame interval
            print(f"\nSelected: {Path(oculus_file).name}")
            interval_input = input("Enter frame interval (default 10, press Enter to use default): ").strip()
            if interval_input:
                try:
                    frame_interval = int(interval_input)
                except ValueError:
                    print("Invalid input. Using default interval of 10.")
                    frame_interval = 10
            else:
                frame_interval = 10
            
            # Get output directory
            default_output = Path("training_data") / "annotations" / Path(oculus_file).stem
            print(f"\nDefault output directory: {default_output}")
            custom_output = input("Enter custom output directory (press Enter for default): ").strip()
            
            if custom_output:
                output_dir = custom_output
            else:
                output_dir = str(default_output)
            
            # Ask about keeping extracted frames
            keep_frames = input("\nKeep extracted frames after annotation? (y/n, default: n): ").lower() == 'y'
            
            # Create annotator and process
            print("\nStarting annotation tool...")
            annotator = SonarAnnotator()
            annotator.annotate_oculus_file(
                oculus_file,
                output_dir,
                frame_interval=frame_interval,
                keep_extracted_frames=keep_frames
            )
            
        elif choice == "2":
            print("\nInitializing file dialogs...")
            print("NOTE: File dialog windows may open behind other windows!")
            print("Check your taskbar if you don't see them.\n")
            
            # Initialize tkinter and bring to front
            root = tk.Tk()
            root.withdraw()
            root.lift()
            root.attributes('-topmost', True)
            root.update()
            time.sleep(0.5)  # Give it time to initialize
            
            # Select input directory
            print("Opening dialog to select directory with sonar images...")
            print("(If you don't see it, check behind this window or in the taskbar)")
            input_dir = filedialog.askdirectory(
                title="Select directory with sonar images",
                initialdir=Path.cwd() / "training_data" / "raw_images" if (Path.cwd() / "training_data" / "raw_images").exists() else Path.cwd()
            )
            
            if not input_dir:
                print("No directory selected. Exiting.")
                root.destroy()
                exit()
            
            print(f"Selected input directory: {input_dir}")
            
            # Select output directory
            print("\nOpening dialog to select output directory for annotations...")
            print("(If you don't see it, check behind this window or in the taskbar)")
            output_dir = filedialog.askdirectory(
                title="Select output directory for annotations",
                initialdir=Path(input_dir).parent
            )
            
            if not output_dir:
                output_dir = str(Path(input_dir).parent / "annotations")
                print(f"No output directory selected. Using default: {output_dir}")
            else:
                print(f"Selected output directory: {output_dir}")
            
            # Destroy the tkinter root
            root.destroy()
            
            # Create annotator and process
            print("\nStarting annotation tool...")
            annotator = SonarAnnotator()
            annotator.annotate_directory(input_dir, output_dir)
            
        elif choice == "3":
            print("Convert existing annotations - to be implemented based on your format")
            
        else:
            print("Exiting...")
