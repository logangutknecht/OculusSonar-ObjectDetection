"""
Sonar Image Annotation Tool for Machine Learning
Helps create training data for ML-based object detection
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tkinter as tk
from tkinter import filedialog, messagebox
import logging

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
                output_img_path = output_path / 'images' / img_path.name
                cv2.imwrite(str(output_img_path), img)
                
                # Save YOLO format annotations
                h, w = img.shape[:2]
                label_path = output_path / 'labels' / f"{img_path.stem}.txt"
                
                with open(label_path, 'w') as f:
                    for box in annotations:
                        class_id = class_to_id[box.class_name]
                        cx, cy, bw, bh = box.to_yolo_format(w, h)
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                
                # Also save JSON format for reference
                json_annotations = [box.to_dict() for box in annotations]
                json_path = output_path / 'labels' / f"{img_path.stem}.json"
                with open(json_path, 'w') as f:
                    json.dump({
                        'image': img_path.name,
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
    # Simple CLI interface
    root = tk.Tk()
    root.withdraw()
    
    print("Sonar Image Annotation Tool")
    print("="*50)
    print("1. Annotate new images")
    print("2. Convert existing annotations")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == "1":
        # Select input directory
        input_dir = filedialog.askdirectory(title="Select directory with sonar images")
        if not input_dir:
            print("No directory selected")
            exit()
        
        # Select output directory
        output_dir = filedialog.askdirectory(title="Select output directory for annotations")
        if not output_dir:
            output_dir = str(Path(input_dir).parent / "annotations")
            print(f"Using default output directory: {output_dir}")
        
        # Create annotator and process
        annotator = SonarAnnotator()
        annotator.annotate_directory(input_dir, output_dir)
        
    elif choice == "2":
        print("Convert existing annotations - to be implemented based on your format")
        
    else:
        print("Exiting...")
