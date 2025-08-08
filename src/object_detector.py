"""
Object Detection Module for Sonar Data
Implements various object detection algorithms for sonar imagery
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging
from scipy import ndimage

# Optional imports
try:
    from sklearn.cluster import DBSCAN
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    
try:
    import torch
    import torchvision
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


@dataclass
class DetectedObject:
    """Container for a detected object in sonar data"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height) in image coords
    confidence: float
    class_name: str
    centroid: Tuple[float, float]
    area: float
    intensity_mean: float
    intensity_std: float
    frame_index: int
    # Optional associated shadow bounding box (x, y, w, h) in image coords
    shadow_bbox: Optional[Tuple[int, int, int, int]] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center of bounding box"""
        x, y, w, h = self.bbox
        return (x + w/2, y + h/2)


class ClassicalDetector:
    """Classical computer vision based object detector"""
    
    def __init__(self, min_area: int = 100, max_area: int = 10000):
        """
        Initialize classical detector
        
        Args:
            min_area: Minimum object area in pixels
            max_area: Maximum object area in pixels
        """
        self.min_area = min_area
        self.max_area = max_area
    
    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[DetectedObject]:
        """
        Detect objects using classical CV techniques
        
        Args:
            image: Sonar intensity image
            frame_index: Index of current frame
            
        Returns:
            List of detected objects
        """
        detections = []
        
        # Preprocessing
        processed = self._preprocess(image)
        
        # Find contours
        contours = self._find_contours(processed)
        
        # Filter and create detections
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if self.min_area <= area <= self.max_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate properties
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = x + w/2, y + h/2
                
                # Extract region statistics
                roi = image[y:y+h, x:x+w]
                intensity_mean = np.mean(roi)
                intensity_std = np.std(roi)
                
                # Create detection
                detection = DetectedObject(
                    bbox=(x, y, w, h),
                    confidence=min(intensity_mean / 255.0, 1.0),  # Simple confidence based on intensity
                    class_name="object",
                    centroid=(cx, cy),
                    area=area,
                    intensity_mean=intensity_mean,
                    intensity_std=intensity_std,
                    frame_index=frame_index
                )
                
                detections.append(detection)
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for detection"""
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            blurred, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _find_contours(self, binary_image: np.ndarray) -> List[np.ndarray]:
        """Find contours in binary image"""
        contours, _ = cv2.findContours(
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        return contours


class EdgeBasedDetector:
    """Edge detection based object detector"""
    
    def __init__(self, low_threshold: int = 50, high_threshold: int = 150):
        """
        Initialize edge-based detector
        
        Args:
            low_threshold: Lower threshold for Canny edge detection
            high_threshold: Upper threshold for Canny edge detection
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[DetectedObject]:
        """
        Detect objects using edge detection
        
        Args:
            image: Sonar intensity image
            frame_index: Index of current frame
            
        Returns:
            List of detected objects
        """
        detections = []
        
        # Ensure uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        # Apply Canny edge detection
        edges = cv2.Canny(image, self.low_threshold, self.high_threshold)
        
        # Find contours from edges
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process contours
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 50:  # Minimum area threshold
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate centroid
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = M["m10"] / M["m00"]
                    cy = M["m01"] / M["m00"]
                else:
                    cx, cy = x + w/2, y + h/2
                
                # Extract region statistics
                roi = image[y:y+h, x:x+w]
                intensity_mean = np.mean(roi)
                intensity_std = np.std(roi)
                
                # Create detection
                detection = DetectedObject(
                    bbox=(x, y, w, h),
                    confidence=0.5,  # Fixed confidence for edge-based
                    class_name="edge_object",
                    centroid=(cx, cy),
                    area=area,
                    intensity_mean=intensity_mean,
                    intensity_std=intensity_std,
                    frame_index=frame_index
                )
                
                detections.append(detection)
        
        return detections


class ClusterBasedDetector:
    """DBSCAN clustering based object detector"""
    
    def __init__(self, eps: float = 5.0, min_samples: int = 10, intensity_threshold: int = 100):
        """
        Initialize cluster-based detector
        
        Args:
            eps: Maximum distance between samples in a cluster
            min_samples: Minimum samples in a cluster
            intensity_threshold: Minimum intensity to consider a pixel
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not installed. Cluster detection unavailable.")
            self.available = False
        else:
            self.available = True
        self.eps = eps
        self.min_samples = min_samples
        self.intensity_threshold = intensity_threshold
    
    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[DetectedObject]:
        """
        Detect objects using DBSCAN clustering
        
        Args:
            image: Sonar intensity image
            frame_index: Index of current frame
            
        Returns:
            List of detected objects
        """
        detections = []
        
        if not self.available:
            return detections
        
        # Find high-intensity pixels
        high_intensity_mask = image > self.intensity_threshold
        y_coords, x_coords = np.where(high_intensity_mask)
        
        if len(x_coords) == 0:
            return detections
        
        # Create feature matrix for clustering
        features = np.column_stack((x_coords, y_coords))
        
        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        labels = clustering.fit_predict(features)
        
        # Process each cluster
        unique_labels = set(labels)
        unique_labels.discard(-1)  # Remove noise label
        
        for label in unique_labels:
            # Get cluster points
            cluster_mask = labels == label
            cluster_x = x_coords[cluster_mask]
            cluster_y = y_coords[cluster_mask]
            
            # Calculate bounding box
            x_min, x_max = cluster_x.min(), cluster_x.max()
            y_min, y_max = cluster_y.min(), cluster_y.max()
            
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            
            # Calculate properties
            centroid = (cluster_x.mean(), cluster_y.mean())
            area = len(cluster_x)
            
            # Extract intensity statistics
            roi = image[y_min:y_max+1, x_min:x_max+1]
            intensity_mean = np.mean(roi)
            intensity_std = np.std(roi)
            
            # Create detection
            detection = DetectedObject(
                bbox=bbox,
                confidence=min(area / 1000.0, 1.0),  # Confidence based on cluster size
                class_name="cluster",
                centroid=centroid,
                area=area,
                intensity_mean=intensity_mean,
                intensity_std=intensity_std,
                frame_index=frame_index
            )
            
            detections.append(detection)
        
        return detections


class YOLODetector:
    """YOLO-based deep learning object detector"""
    
    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.5):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to YOLO model weights (uses pretrained if None)
            confidence_threshold: Minimum confidence for detections
        """
        self.confidence_threshold = confidence_threshold
        self.model = None
        
        if not HAS_TORCH:
            logger.warning("PyTorch not installed. YOLO detection unavailable.")
            return
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            from ultralytics import YOLO
            
            if model_path:
                self.model = YOLO(model_path)
            else:
                # Use a pretrained model (can be fine-tuned for sonar data)
                self.model = YOLO('yolov8n.pt')  # Nano model for speed
            
            logger.info(f"YOLO model loaded on {self.device}")
        except ImportError:
            logger.warning("Ultralytics not installed. YOLO detection unavailable.")
    
    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[DetectedObject]:
        """
        Detect objects using YOLO
        
        Args:
            image: Sonar intensity image
            frame_index: Index of current frame
            
        Returns:
            List of detected objects
        """
        if self.model is None:
            logger.warning("YOLO model not available")
            return []
        
        detections = []
        
        # Ensure image is in correct format
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        # Convert grayscale to RGB for YOLO
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
        
        # Run inference
        results = self.model(image_rgb, conf=self.confidence_threshold)
        
        # Process results
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                    
                    # Extract properties
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    # Calculate additional properties
                    centroid = ((x1 + x2) / 2, (y1 + y2) / 2)
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Extract intensity statistics
                    roi = image[int(y1):int(y2), int(x1):int(x2)]
                    intensity_mean = np.mean(roi) if roi.size > 0 else 0
                    intensity_std = np.std(roi) if roi.size > 0 else 0
                    
                    # Create detection
                    detection = DetectedObject(
                        bbox=bbox,
                        confidence=confidence,
                        class_name=class_name,
                        centroid=centroid,
                        area=area,
                        intensity_mean=intensity_mean,
                        intensity_std=intensity_std,
                        frame_index=frame_index
                    )
                    
                    detections.append(detection)
        
        return detections


class SonarObjectDetector:
    """Main object detector combining multiple detection methods"""
    
    def __init__(self, methods: List[str] = ['classical', 'edge', 'cluster']):
        """
        Initialize combined detector
        
        Args:
            methods: List of detection methods to use
        """
        self.detectors = {}
        
        if 'classical' in methods:
            self.detectors['classical'] = ClassicalDetector()
        if 'edge' in methods:
            self.detectors['edge'] = EdgeBasedDetector()
        if 'cluster' in methods:
            if HAS_SKLEARN:
                self.detectors['cluster'] = ClusterBasedDetector()
            else:
                logger.warning("Skipping cluster detector - sklearn not installed")
        if 'yolo' in methods:
            if HAS_TORCH:
                self.detectors['yolo'] = YOLODetector()
            else:
                logger.warning("Skipping YOLO detector - PyTorch not installed")
    
    def detect(self, image: np.ndarray, frame_index: int = 0, 
               combine_method: str = 'union') -> List[DetectedObject]:
        """
        Detect objects using multiple methods
        
        Args:
            image: Sonar intensity image
            frame_index: Index of current frame
            combine_method: How to combine detections ('union', 'intersection', 'vote')
            
        Returns:
            List of detected objects
        """
        all_detections = []
        
        # Run all detectors
        for name, detector in self.detectors.items():
            try:
                detections = detector.detect(image, frame_index)
                all_detections.extend(detections)
                logger.info(f"{name} detector found {len(detections)} objects")
            except Exception as e:
                logger.error(f"Error in {name} detector: {e}")
        
        # Combine detections based on method
        if combine_method == 'union':
            # Return all detections
            return all_detections
        elif combine_method == 'intersection':
            # Return only overlapping detections
            return self._filter_overlapping(all_detections, min_overlap=0.5)
        elif combine_method == 'vote':
            # Return detections that appear in multiple methods
            return self._filter_by_voting(all_detections, min_votes=2)
        else:
            return all_detections
    
    def _filter_overlapping(self, detections: List[DetectedObject], 
                          min_overlap: float = 0.5) -> List[DetectedObject]:
        """Filter detections to keep only overlapping ones"""
        filtered = []
        
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections):
                if i != j:
                    iou = self._calculate_iou(det1.bbox, det2.bbox)
                    if iou > min_overlap:
                        filtered.append(det1)
                        break
        
        return filtered
    
    def _filter_by_voting(self, detections: List[DetectedObject], 
                         min_votes: int = 2) -> List[DetectedObject]:
        """Filter detections by voting from multiple detectors"""
        # Group overlapping detections
        groups = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
            
            group = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections):
                if j not in used and j != i:
                    iou = self._calculate_iou(det1.bbox, det2.bbox)
                    if iou > 0.3:  # Lower threshold for voting
                        group.append(det2)
                        used.add(j)
            
            if len(group) >= min_votes:
                # Use detection with highest confidence from group
                best_det = max(group, key=lambda x: x.confidence)
                groups.append(best_det)
        
        return groups
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                      box2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union for two boxes"""
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
    # Test detectors with synthetic data
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    # Create synthetic sonar image with objects
    image = np.zeros((200, 256), dtype=np.uint8)
    
    # Add some objects
    cv2.rectangle(image, (50, 50), (100, 100), 200, -1)  # Bright rectangle
    cv2.circle(image, (150, 100), 30, 180, -1)  # Bright circle
    cv2.ellipse(image, (200, 150), (40, 20), 45, 0, 360, 160, -1)  # Ellipse
    
    # Add noise
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    image = cv2.add(image, noise)
    
    # Initialize detector
    detector = SonarObjectDetector(methods=['classical', 'edge', 'cluster'])
    
    # Detect objects
    detections = detector.detect(image)
    
    # Visualize
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(image, cmap='gray')
    
    # Draw detections
    for det in detections:
        x, y, w, h = det.bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y-5, f'{det.class_name} ({det.confidence:.2f})', 
               color='yellow', fontsize=8)
    
    ax.set_title(f'Detected {len(detections)} objects')
    plt.tight_layout()
    plt.show()
