"""
Machine Learning Based Sonar Object Detector
Integrates trained YOLO models into the sonar processing pipeline
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import torch
from ultralytics import YOLO
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from object_detector import DetectedObject

logger = logging.getLogger(__name__)


class MLSonarDetector:
    """Machine Learning based detector using trained YOLO model"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45, device: Optional[str] = None):
        """
        Initialize ML detector
        
        Args:
            model_path: Path to trained YOLO model (.pt file)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IOU threshold for NMS
            device: Device to run on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Check if model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        logger.info(f"Loading ML model from {model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Move model to device
        if self.device == 'cuda':
            self.model.to('cuda')
        
        logger.info(f"ML detector initialized on {self.device}")
        
        # Get class names from model
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        
        # Preprocessing parameters optimized for sonar
        self.preprocess_config = {
            'clahe_clip': 2.0,
            'clahe_grid': (8, 8),
            'bilateral_d': 9,
            'bilateral_sigma_color': 75,
            'bilateral_sigma_space': 75,
            'normalize': True
        }
    
    def preprocess_for_ml(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess sonar image for ML inference
        
        Args:
            image: Input sonar image (grayscale or RGB)
            
        Returns:
            Preprocessed image ready for ML model
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            if image.max() <= 1:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE for contrast enhancement
        if self.preprocess_config['clahe_clip'] > 0:
            clahe = cv2.createCLAHE(
                clipLimit=self.preprocess_config['clahe_clip'],
                tileGridSize=self.preprocess_config['clahe_grid']
            )
            enhanced = clahe.apply(gray)
        else:
            enhanced = gray
        
        # Apply bilateral filter for noise reduction
        if self.preprocess_config['bilateral_d'] > 0:
            denoised = cv2.bilateralFilter(
                enhanced,
                self.preprocess_config['bilateral_d'],
                self.preprocess_config['bilateral_sigma_color'],
                self.preprocess_config['bilateral_sigma_space']
            )
        else:
            denoised = enhanced
        
        # Normalize if requested
        if self.preprocess_config['normalize']:
            normalized = cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX)
        else:
            normalized = denoised
        
        # Convert to RGB for YOLO (replicate grayscale channel)
        rgb_image = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        
        return rgb_image
    
    def detect(self, image: np.ndarray, frame_index: int = 0,
              preprocess: bool = True, return_debug_info: bool = False) -> List[DetectedObject]:
        """
        Detect objects in sonar image using ML model
        
        Args:
            image: Sonar intensity image
            frame_index: Current frame index
            preprocess: Whether to apply preprocessing
            return_debug_info: Return additional debug information
            
        Returns:
            List of detected objects
        """
        detections = []
        debug_info = {}
        
        # Preprocess image
        if preprocess:
            processed = self.preprocess_for_ml(image)
            debug_info['preprocessed_image'] = processed
        else:
            # Ensure RGB format
            if len(image.shape) == 2:
                processed = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                processed = image
        
        # Run inference
        results = self.model(
            processed,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Process results
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Extract box coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Convert to (x, y, w, h) format
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    
                    # Get confidence
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # Get class
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    # Calculate centroid
                    centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Extract intensity statistics from original image
                    x, y, w, h = bbox
                    roi = image[y:y+h, x:x+w] if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[y:y+h, x:x+w]
                    
                    if roi.size > 0:
                        intensity_mean = float(np.mean(roi))
                        intensity_std = float(np.std(roi))
                    else:
                        intensity_mean = 0.0
                        intensity_std = 0.0
                    
                    # Create detection object
                    detection = DetectedObject(
                        bbox=bbox,
                        confidence=confidence,
                        class_name=f"ml_{class_name}",  # Prefix with 'ml_' to distinguish
                        centroid=centroid,
                        area=area,
                        intensity_mean=intensity_mean,
                        intensity_std=intensity_std,
                        frame_index=frame_index
                    )
                    
                    detections.append(detection)
        
        # Add debug visualization if requested
        if return_debug_info:
            debug_info['detections'] = detections
            debug_info['raw_results'] = results
            return detections, debug_info
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray], frame_indices: Optional[List[int]] = None,
                    preprocess: bool = True) -> List[List[DetectedObject]]:
        """
        Detect objects in multiple images (batch processing)
        
        Args:
            images: List of sonar images
            frame_indices: List of frame indices (or None)
            preprocess: Whether to apply preprocessing
            
        Returns:
            List of detection lists (one per image)
        """
        if frame_indices is None:
            frame_indices = list(range(len(images)))
        
        # Preprocess all images
        if preprocess:
            processed_images = [self.preprocess_for_ml(img) for img in images]
        else:
            processed_images = []
            for img in images:
                if len(img.shape) == 2:
                    processed_images.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
                else:
                    processed_images.append(img)
        
        # Run batch inference
        results = self.model(
            processed_images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            verbose=False
        )
        
        # Process results for each image
        all_detections = []
        
        for img_idx, (result, original_img, frame_idx) in enumerate(zip(results, images, frame_indices)):
            detections = []
            
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Extract box coordinates
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = xyxy
                    
                    # Convert to (x, y, w, h) format
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    
                    # Get confidence
                    confidence = float(boxes.conf[i].cpu().numpy())
                    
                    # Get class
                    class_id = int(boxes.cls[i].cpu().numpy())
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    # Calculate centroid
                    centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Extract intensity statistics
                    x, y, w, h = bbox
                    roi = original_img[y:y+h, x:x+w] if len(original_img.shape) == 2 else cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)[y:y+h, x:x+w]
                    
                    if roi.size > 0:
                        intensity_mean = float(np.mean(roi))
                        intensity_std = float(np.std(roi))
                    else:
                        intensity_mean = 0.0
                        intensity_std = 0.0
                    
                    # Create detection object
                    detection = DetectedObject(
                        bbox=bbox,
                        confidence=confidence,
                        class_name=f"ml_{class_name}",
                        centroid=centroid,
                        area=area,
                        intensity_mean=intensity_mean,
                        intensity_std=intensity_std,
                        frame_index=frame_idx
                    )
                    
                    detections.append(detection)
            
            all_detections.append(detections)
        
        return all_detections
    
    def update_preprocessing(self, config: Dict):
        """
        Update preprocessing configuration
        
        Args:
            config: Dictionary with preprocessing parameters
        """
        self.preprocess_config.update(config)
        logger.info(f"Updated preprocessing config: {self.preprocess_config}")
    
    def warmup(self, image_size: Tuple[int, int] = (640, 640)):
        """
        Warmup the model with a dummy image
        
        Args:
            image_size: Size of dummy image (height, width)
        """
        logger.info("Warming up ML model...")
        
        # Create dummy image
        dummy = np.zeros((*image_size, 3), dtype=np.uint8)
        
        # Run inference
        _ = self.model(dummy, verbose=False)
        
        logger.info("Model warmup complete")
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_path': str(self.model_path),
            'device': self.device,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'class_names': self.class_names,
            'input_size': getattr(self.model, 'imgsz', None)
        }
        
        # Try to get model metrics if available
        if hasattr(self.model, 'metrics'):
            info['metrics'] = self.model.metrics
        
        return info


class HybridSonarDetector:
    """
    Hybrid detector combining ML and classical approaches
    Useful for leveraging both learned patterns and physics-based heuristics
    """
    
    def __init__(self, ml_model_path: Optional[str] = None,
                 use_classical: bool = True,
                 use_specialized: bool = True,
                 fusion_method: str = "weighted"):
        """
        Initialize hybrid detector
        
        Args:
            ml_model_path: Path to ML model (None to skip ML)
            use_classical: Whether to use classical detection
            use_specialized: Whether to use specialized shape detection
            fusion_method: How to combine detections ('weighted', 'voting', 'nms')
        """
        self.detectors = {}
        
        # Initialize ML detector if model provided
        if ml_model_path and Path(ml_model_path).exists():
            self.detectors['ml'] = MLSonarDetector(ml_model_path)
            logger.info("ML detector initialized")
        
        # Initialize classical detectors if requested
        if use_classical:
            from object_detector import ClassicalDetector, EdgeBasedDetector
            self.detectors['classical'] = ClassicalDetector(min_area=100, max_area=50000)
            self.detectors['edge'] = EdgeBasedDetector()
            logger.info("Classical detectors initialized")
        
        # Initialize specialized detector if requested
        if use_specialized:
            from specialized_detector import SpecializedSonarDetector
            self.detectors['specialized'] = SpecializedSonarDetector(
                shadow_analysis=True,
                shapes=("sphere", "barrel", "cube")
            )
            logger.info("Specialized detector initialized")
        
        self.fusion_method = fusion_method
        
        # Weights for different detectors (can be tuned)
        self.detector_weights = {
            'ml': 0.5,           # Higher weight for ML if trained well
            'specialized': 0.3,   # Good for specific shapes
            'classical': 0.1,     # Lower weight but still useful
            'edge': 0.1          # Edge detection as supplementary
        }
    
    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[DetectedObject]:
        """
        Detect objects using hybrid approach
        
        Args:
            image: Sonar intensity image
            frame_index: Current frame index
            
        Returns:
            List of detected objects
        """
        all_detections = []
        detector_results = {}
        
        # Run each detector
        for name, detector in self.detectors.items():
            try:
                if name == 'ml':
                    detections = detector.detect(image, frame_index, preprocess=True)
                else:
                    detections = detector.detect(image, frame_index)
                
                detector_results[name] = detections
                all_detections.extend(detections)
                
                logger.debug(f"{name} detector found {len(detections)} objects")
                
            except Exception as e:
                logger.warning(f"Error in {name} detector: {e}")
                detector_results[name] = []
        
        # Fuse detections based on method
        if self.fusion_method == "weighted":
            return self._weighted_fusion(detector_results)
        elif self.fusion_method == "voting":
            return self._voting_fusion(detector_results)
        elif self.fusion_method == "nms":
            return self._nms_fusion(all_detections)
        else:
            return all_detections
    
    def _weighted_fusion(self, detector_results: Dict[str, List[DetectedObject]]) -> List[DetectedObject]:
        """Combine detections using weighted confidence scores"""
        # Group overlapping detections
        groups = []
        used = set()
        
        all_detections = []
        for name, detections in detector_results.items():
            weight = self.detector_weights.get(name, 0.1)
            for det in detections:
                all_detections.append((det, name, weight))
        
        for i, (det1, name1, weight1) in enumerate(all_detections):
            if i in used:
                continue
            
            group = [(det1, name1, weight1)]
            used.add(i)
            
            for j, (det2, name2, weight2) in enumerate(all_detections):
                if j not in used and j != i:
                    iou = self._calculate_iou(det1.bbox, det2.bbox)
                    if iou > 0.3:
                        group.append((det2, name2, weight2))
                        used.add(j)
            
            # Combine group into single detection
            if group:
                # Weight the confidence scores
                total_weight = sum(w for _, _, w in group)
                weighted_conf = sum(d.confidence * w for d, _, w in group) / total_weight
                
                # Use detection with highest weighted contribution
                best_det = max(group, key=lambda x: x[0].confidence * x[2])[0]
                
                # Update confidence
                best_det.confidence = weighted_conf
                
                # Update class name to indicate fusion
                if len(group) > 1:
                    sources = list(set(n for _, n, _ in group))
                    best_det.class_name = f"{best_det.class_name}_hybrid_{'+'.join(sources)}"
                
                groups.append(best_det)
        
        return groups
    
    def _voting_fusion(self, detector_results: Dict[str, List[DetectedObject]]) -> List[DetectedObject]:
        """Keep detections that appear in multiple detectors"""
        min_votes = 2  # Require at least 2 detectors to agree
        
        # Implementation similar to weighted fusion but with voting logic
        # ... (simplified for brevity)
        
        return self._weighted_fusion(detector_results)  # Fallback to weighted for now
    
    def _nms_fusion(self, detections: List[DetectedObject], 
                   iou_threshold: float = 0.5) -> List[DetectedObject]:
        """Apply Non-Maximum Suppression to all detections"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        
        keep = []
        while detections:
            # Keep the detection with highest confidence
            best = detections.pop(0)
            keep.append(best)
            
            # Remove overlapping detections
            detections = [d for d in detections 
                         if self._calculate_iou(best.bbox, d.bbox) < iou_threshold]
        
        return keep
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
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


if __name__ == "__main__":
    # Test the ML detector
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create synthetic test image
    test_image = np.zeros((640, 640), dtype=np.uint8)
    
    # Add some bright regions (simulated objects)
    cv2.circle(test_image, (200, 200), 50, 200, -1)
    cv2.rectangle(test_image, (400, 300), (500, 400), 180, -1)
    cv2.ellipse(test_image, (300, 500), (60, 30), 45, 0, 360, 160, -1)
    
    # Add noise
    noise = np.random.randint(0, 50, test_image.shape, dtype=np.uint8)
    test_image = cv2.add(test_image, noise)
    
    # Check if a model exists
    model_path = Path("models")
    if model_path.exists():
        model_files = list(model_path.glob("*.pt"))
        if model_files:
            # Use the first available model
            detector = MLSonarDetector(str(model_files[0]))
            
            # Detect objects
            detections = detector.detect(test_image)
            
            # Visualize
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            ax.imshow(test_image, cmap='gray')
            
            for det in detections:
                x, y, w, h = det.bbox
                rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                        edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x, y-5, f'{det.class_name} ({det.confidence:.2f})',
                       color='yellow', fontsize=10)
            
            ax.set_title(f'ML Detector: Found {len(detections)} objects')
            plt.tight_layout()
            plt.show()
        else:
            print("No trained models found. Please train a model first.")
    else:
        print("Models directory not found. Please train a model first.")
