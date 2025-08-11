"""
Validation and Testing Utilities for ML Sonar Detector
Helps evaluate model performance and visualize results
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns
import pandas as pd
import logging

# Add parent directory for imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ml_detector.ml_sonar_detector import MLSonarDetector, HybridSonarDetector
from object_detector import DetectedObject

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelValidator:
    """Validates ML model performance on sonar data"""
    
    def __init__(self, model_path: str):
        """
        Initialize validator
        
        Args:
            model_path: Path to trained model
        """
        self.model_path = Path(model_path)
        self.detector = MLSonarDetector(str(model_path))
        self.results = {}
        
    def validate_on_dataset(self, dataset_path: str, 
                           visualize: bool = True,
                           save_results: bool = True) -> Dict:
        """
        Validate model on a labeled dataset
        
        Args:
            dataset_path: Path to dataset with images and labels
            visualize: Whether to create visualizations
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary with validation metrics
        """
        dataset_path = Path(dataset_path)
        images_dir = dataset_path / 'images'
        labels_dir = dataset_path / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            raise ValueError(f"Invalid dataset structure at {dataset_path}")
        
        # Load class names
        classes_file = dataset_path / 'classes.txt'
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        else:
            class_names = ['object']
        
        # Process all images
        all_predictions = []
        all_ground_truths = []
        image_results = []
        
        image_files = list(images_dir.glob('*.png')) + list(images_dir.glob('*.jpg'))
        logger.info(f"Validating on {len(image_files)} images...")
        
        for img_path in image_files:
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Get predictions
            predictions = self.detector.detect(img, preprocess=True)
            
            # Load ground truth
            label_path = labels_dir / f"{img_path.stem}.txt"
            ground_truths = self._load_yolo_labels(label_path, img.shape[:2], class_names)
            
            # Store results
            image_results.append({
                'image': img_path.name,
                'predictions': predictions,
                'ground_truths': ground_truths,
                'num_predictions': len(predictions),
                'num_ground_truths': len(ground_truths)
            })
            
            all_predictions.extend(predictions)
            all_ground_truths.extend(ground_truths)
        
        # Calculate metrics
        metrics = self._calculate_metrics(image_results)
        
        # Add summary statistics
        metrics['total_images'] = len(image_files)
        metrics['total_predictions'] = len(all_predictions)
        metrics['total_ground_truths'] = len(all_ground_truths)
        
        # Visualize results
        if visualize:
            self._visualize_results(image_results[:5], metrics)  # Show first 5 images
        
        # Save results
        if save_results:
            output_dir = Path('validation_results')
            output_dir.mkdir(exist_ok=True)
            
            # Save metrics
            with open(output_dir / 'metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save detailed results
            detailed_results = []
            for result in image_results:
                detailed_results.append({
                    'image': result['image'],
                    'num_predictions': result['num_predictions'],
                    'num_ground_truths': result['num_ground_truths'],
                    'predictions': [self._detection_to_dict(d) for d in result['predictions']],
                    'ground_truths': [self._detection_to_dict(d) for d in result['ground_truths']]
                })
            
            with open(output_dir / 'detailed_results.json', 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            logger.info(f"Results saved to {output_dir}")
        
        self.results = metrics
        return metrics
    
    def _load_yolo_labels(self, label_path: Path, img_shape: Tuple[int, int], 
                         class_names: List[str]) -> List[DetectedObject]:
        """Load YOLO format labels as DetectedObject instances"""
        detections = []
        
        if not label_path.exists():
            return detections
        
        h, w = img_shape
        
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:5])
                    
                    # Convert from normalized to pixel coordinates
                    x = int((cx - bw/2) * w)
                    y = int((cy - bh/2) * h)
                    width = int(bw * w)
                    height = int(bh * h)
                    
                    detection = DetectedObject(
                        bbox=(x, y, width, height),
                        confidence=1.0,  # Ground truth
                        class_name=class_names[class_id] if class_id < len(class_names) else f"class_{class_id}",
                        centroid=(cx * w, cy * h),
                        area=width * height,
                        intensity_mean=0,
                        intensity_std=0,
                        frame_index=0
                    )
                    detections.append(detection)
        
        return detections
    
    def _calculate_metrics(self, image_results: List[Dict]) -> Dict:
        """Calculate validation metrics"""
        # Calculate IoU-based metrics
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        iou_threshold = 0.5
        
        for result in image_results:
            preds = result['predictions']
            gts = result['ground_truths']
            
            # Match predictions to ground truths
            matched_gts = set()
            
            for pred in preds:
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt in enumerate(gts):
                    if gt_idx not in matched_gts:
                        iou = self._calculate_iou(pred.bbox, gt.bbox)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                
                if best_iou >= iou_threshold:
                    true_positives += 1
                    matched_gts.add(best_gt_idx)
                else:
                    false_positives += 1
            
            false_negatives += len(gts) - len(matched_gts)
        
        # Calculate precision, recall, F1
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'iou_threshold': iou_threshold
        }
        
        return metrics
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 < xi1 or yi2 < yi1:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        
        # Calculate union
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _detection_to_dict(self, detection: DetectedObject) -> Dict:
        """Convert DetectedObject to dictionary"""
        return {
            'bbox': detection.bbox,
            'confidence': detection.confidence,
            'class_name': detection.class_name,
            'centroid': detection.centroid,
            'area': detection.area
        }
    
    def _visualize_results(self, image_results: List[Dict], metrics: Dict):
        """Create visualizations of validation results"""
        # Create figure with subplots
        n_images = min(len(image_results), 5)
        fig, axes = plt.subplots(2, n_images, figsize=(4*n_images, 8))
        
        if n_images == 1:
            axes = axes.reshape(2, 1)
        
        for idx, result in enumerate(image_results[:n_images]):
            # Load image
            img_path = Path('validation_results') / 'images' / result['image']
            if not img_path.exists():
                img_path = Path('training_data') / 'images' / result['image']
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Plot predictions
                axes[0, idx].imshow(img_rgb)
                axes[0, idx].set_title(f"Predictions ({len(result['predictions'])})")
                axes[0, idx].axis('off')
                
                for det in result['predictions']:
                    x, y, w, h = det.bbox
                    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                            edgecolor='red', facecolor='none')
                    axes[0, idx].add_patch(rect)
                
                # Plot ground truth
                axes[1, idx].imshow(img_rgb)
                axes[1, idx].set_title(f"Ground Truth ({len(result['ground_truths'])})")
                axes[1, idx].axis('off')
                
                for det in result['ground_truths']:
                    x, y, w, h = det.bbox
                    rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                            edgecolor='green', facecolor='none')
                    axes[1, idx].add_patch(rect)
        
        # Add metrics text
        fig.suptitle(f"Validation Results - Precision: {metrics['precision']:.3f}, "
                    f"Recall: {metrics['recall']:.3f}, F1: {metrics['f1_score']:.3f}",
                    fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    def test_on_live_data(self, sonar_file_path: str, 
                         output_dir: str = "test_results",
                         save_frames: bool = True) -> Dict:
        """
        Test model on live sonar data
        
        Args:
            sonar_file_path: Path to .oculus sonar file
            output_dir: Directory to save results
            save_frames: Whether to save annotated frames
            
        Returns:
            Test results dictionary
        """
        # Import sonar processor
        from sonar_processor import OculusFileReader
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Load sonar file
        logger.info(f"Loading sonar file: {sonar_file_path}")
        reader = OculusFileReader(sonar_file_path)
        frames = reader.parse_all_frames()
        
        logger.info(f"Processing {len(frames)} frames...")
        
        # Process frames
        all_detections = []
        detection_counts = []
        
        for i, frame in enumerate(frames):
            # Get frame data
            img = frame.intensity_data
            
            # Detect objects
            detections = self.detector.detect(img, frame_index=i, preprocess=True)
            
            all_detections.extend(detections)
            detection_counts.append(len(detections))
            
            # Save annotated frame
            if save_frames and (i % 10 == 0 or len(detections) > 0):  # Save every 10th frame or frames with detections
                self._save_annotated_frame(img, detections, output_path / f"frame_{i:04d}.png")
        
        # Calculate statistics
        results = {
            'total_frames': len(frames),
            'total_detections': len(all_detections),
            'frames_with_detections': sum(1 for c in detection_counts if c > 0),
            'avg_detections_per_frame': np.mean(detection_counts),
            'max_detections_in_frame': max(detection_counts),
            'detection_counts': detection_counts
        }
        
        # Save results
        with open(output_path / 'test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test complete. Results saved to {output_path}")
        logger.info(f"Total detections: {results['total_detections']}")
        logger.info(f"Frames with detections: {results['frames_with_detections']}/{results['total_frames']}")
        
        return results
    
    def _save_annotated_frame(self, img: np.ndarray, detections: List[DetectedObject], 
                             output_path: Path):
        """Save frame with detection annotations"""
        # Normalize image for display
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
        
        # Convert to color for annotations
        if len(img.shape) == 2:
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img.copy()
        
        # Draw detections
        for det in detections:
            x, y, w, h = det.bbox
            
            # Draw bounding box
            cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add label
            label = f"{det.class_name}: {det.confidence:.2f}"
            cv2.putText(img_color, label, (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save image
        cv2.imwrite(str(output_path), img_color)


class PerformanceAnalyzer:
    """Analyzes and compares performance of different detection methods"""
    
    @staticmethod
    def compare_detectors(image_path: str, 
                         ml_model_path: Optional[str] = None,
                         use_classical: bool = True,
                         use_specialized: bool = True) -> Dict:
        """
        Compare different detection methods on the same image
        
        Args:
            image_path: Path to test image
            ml_model_path: Path to ML model
            use_classical: Whether to test classical detector
            use_specialized: Whether to test specialized detector
            
        Returns:
            Comparison results
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        results = {}
        
        # Test ML detector
        if ml_model_path and Path(ml_model_path).exists():
            ml_detector = MLSonarDetector(ml_model_path)
            ml_detections = ml_detector.detect(img, preprocess=True)
            results['ml'] = {
                'count': len(ml_detections),
                'detections': ml_detections,
                'avg_confidence': np.mean([d.confidence for d in ml_detections]) if ml_detections else 0
            }
        
        # Test classical detector
        if use_classical:
            from object_detector import ClassicalDetector
            classical = ClassicalDetector()
            classical_detections = classical.detect(img)
            results['classical'] = {
                'count': len(classical_detections),
                'detections': classical_detections,
                'avg_confidence': np.mean([d.confidence for d in classical_detections]) if classical_detections else 0
            }
        
        # Test specialized detector
        if use_specialized:
            from specialized_detector import SpecializedSonarDetector
            specialized = SpecializedSonarDetector()
            specialized_detections = specialized.detect(img)
            results['specialized'] = {
                'count': len(specialized_detections),
                'detections': specialized_detections,
                'avg_confidence': np.mean([d.confidence for d in specialized_detections]) if specialized_detections else 0
            }
        
        # Test hybrid detector
        if ml_model_path and Path(ml_model_path).exists():
            hybrid = HybridSonarDetector(
                ml_model_path=ml_model_path,
                use_classical=use_classical,
                use_specialized=use_specialized
            )
            hybrid_detections = hybrid.detect(img)
            results['hybrid'] = {
                'count': len(hybrid_detections),
                'detections': hybrid_detections,
                'avg_confidence': np.mean([d.confidence for d in hybrid_detections]) if hybrid_detections else 0
            }
        
        # Visualize comparison
        PerformanceAnalyzer._visualize_comparison(img, results)
        
        return results
    
    @staticmethod
    def _visualize_comparison(img: np.ndarray, results: Dict):
        """Visualize detection comparison"""
        n_methods = len(results)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 5))
        
        if n_methods == 1:
            axes = [axes]
        
        # Convert image to RGB for display
        if len(img.shape) == 2:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        colors = {'ml': 'red', 'classical': 'blue', 'specialized': 'green', 'hybrid': 'purple'}
        
        for idx, (method, data) in enumerate(results.items()):
            axes[idx].imshow(img_rgb)
            axes[idx].set_title(f"{method.upper()}\n{data['count']} detections\nAvg conf: {data['avg_confidence']:.2f}")
            axes[idx].axis('off')
            
            # Draw detections
            color = colors.get(method, 'yellow')
            for det in data['detections']:
                x, y, w, h = det.bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                        edgecolor=color, facecolor='none')
                axes[idx].add_patch(rect)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate ML sonar detector")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--dataset', type=str,
                       help='Path to validation dataset')
    parser.add_argument('--sonar-file', type=str,
                       help='Path to .oculus file for testing')
    parser.add_argument('--compare', type=str,
                       help='Image path for detector comparison')
    
    args = parser.parse_args()
    
    if args.dataset:
        # Validate on dataset
        validator = ModelValidator(args.model)
        metrics = validator.validate_on_dataset(args.dataset)
        
        print("\nValidation Results:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1 Score: {metrics['f1_score']:.3f}")
    
    elif args.sonar_file:
        # Test on sonar file
        validator = ModelValidator(args.model)
        results = validator.test_on_live_data(args.sonar_file)
    
    elif args.compare:
        # Compare detectors
        results = PerformanceAnalyzer.compare_detectors(
            args.compare,
            ml_model_path=args.model
        )
        
        print("\nComparison Results:")
        for method, data in results.items():
            print(f"  {method}: {data['count']} detections, avg confidence: {data['avg_confidence']:.3f}")
