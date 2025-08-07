"""
Specialized Object Detector for Sonar Data
Detects specific underwater objects: spheres, barrels, and cubes
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging
from scipy import ndimage
from scipy.signal import find_peaks

from object_detector import DetectedObject

logger = logging.getLogger(__name__)


class ShapeAnalyzer:
    """Analyzes shapes and patterns in sonar data"""
    
    @staticmethod
    def detect_crescent_shapes(image: np.ndarray, min_radius: int = 15, max_radius: int = 80) -> List[Tuple[int, int, int]]:
        """
        Detect crescent/partial circular shapes (indicative of spheres)
        
        Args:
            image: Binary or grayscale image
            min_radius: Minimum radius to detect
            max_radius: Maximum radius to detect
            
        Returns:
            List of (x, y, radius) for detected crescents
        """
        crescents = []
        
        # Apply Hough Circle Transform
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(image, (9, 9), 2)
        
        # Detect circles (including partial ones) with stricter parameters
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.5,  # Increased for better accuracy
            minDist=min_radius * 3,  # Increased to reduce overlapping detections
            param1=100,  # Increased for stricter edge detection
            param2=50,   # Increased for stricter center detection
            minRadius=min_radius,
            maxRadius=max_radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                
                # Verify it's a crescent by checking intensity pattern
                if ShapeAnalyzer._is_crescent_pattern(image, x, y, r):
                    crescents.append((int(x), int(y), int(r)))
        
        return crescents
    
    @staticmethod
    def _is_crescent_pattern(image: np.ndarray, cx: int, cy: int, radius: int) -> bool:
        """
        Check if the circular region has a crescent intensity pattern
        
        Args:
            image: Input image
            cx, cy: Circle center
            radius: Circle radius
            
        Returns:
            True if crescent pattern detected
        """
        h, w = image.shape[:2]
        
        # Sample points around the circle
        angles = np.linspace(0, 2 * np.pi, 36)
        intensities = []
        
        for angle in angles:
            x = int(cx + radius * np.cos(angle))
            y = int(cy + radius * np.sin(angle))
            
            if 0 <= x < w and 0 <= y < h:
                intensities.append(image[y, x])
            else:
                intensities.append(0)
        
        intensities = np.array(intensities)
        
        # Crescent pattern: high intensity on one side, low on the other
        # Check if there's a significant gradient
        max_intensity = intensities.max()
        min_intensity = intensities.min()
        
        if max_intensity - min_intensity > 80:  # Increased threshold for more significant difference
            # Check for continuous high region (not scattered)
            high_region = intensities > (max_intensity * 0.7)
            
            # Count transitions between high and low
            transitions = np.sum(np.abs(np.diff(high_region.astype(int))))
            
            # Crescent should have 2 transitions (enter and exit high region)
            if transitions <= 4:  # Allow some noise
                return True
        
        return False
    
    @staticmethod
    def detect_elliptical_shapes(image: np.ndarray, min_area: int = 500) -> List[Dict]:
        """
        Detect elliptical/oval shapes (indicative of barrels)
        
        Args:
            image: Binary or grayscale image
            min_area: Minimum area for ellipse detection
            
        Returns:
            List of ellipse parameters
        """
        ellipses = []
        
        # Threshold if grayscale
        if len(image.shape) == 2:
            _, binary = cv2.threshold(image.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
        else:
            binary = image
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > min_area and len(contour) >= 5:
                # Fit ellipse
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (cx, cy), (width, height), angle = ellipse
                    
                    # Check if it's elliptical (not too circular)
                    aspect_ratio = max(width, height) / min(width, height)
                    
                    if 1.5 <= aspect_ratio <= 4.0:  # Barrel-like aspect ratio
                        ellipses.append({
                            'center': (cx, cy),
                            'axes': (width, height),
                            'angle': angle,
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
                except:
                    continue
        
        return ellipses
    
    @staticmethod
    def detect_straight_edges(image: np.ndarray, min_length: int = 30) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect straight edges (indicative of cubes/rectangular objects)
        
        Args:
            image: Binary or grayscale image
            min_length: Minimum edge length
            
        Returns:
            List of line segments as ((x1, y1), (x2, y2))
        """
        edges = []
        
        # Apply Canny edge detection
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        canny = cv2.Canny(image, 50, 150, apertureSize=3)
        
        # Detect lines using Hough Line Transform
        lines = cv2.HoughLinesP(
            canny,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=min_length,
            maxLineGap=10
        )
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                if length >= min_length:
                    edges.append(((x1, y1), (x2, y2)))
        
        return edges
    
    @staticmethod
    def detect_corner_patterns(edges: List[Tuple[Tuple[int, int], Tuple[int, int]]]) -> List[Dict]:
        """
        Detect corner patterns from edges (indicative of cubes)
        
        Args:
            edges: List of edge segments
            
        Returns:
            List of corner patterns with their properties
        """
        corners = []
        
        # Find edge intersections/corners
        for i, edge1 in enumerate(edges):
            (x1a, y1a), (x1b, y1b) = edge1
            
            for j, edge2 in enumerate(edges[i+1:], i+1):
                (x2a, y2a), (x2b, y2b) = edge2
                
                # Check if edges are connected or nearly connected
                dist_threshold = 10  # pixels
                
                # Check all possible connections
                connections = [
                    ((x1a, y1a), (x2a, y2a)),
                    ((x1a, y1a), (x2b, y2b)),
                    ((x1b, y1b), (x2a, y2a)),
                    ((x1b, y1b), (x2b, y2b))
                ]
                
                for (px1, py1), (px2, py2) in connections:
                    dist = np.sqrt((px2 - px1)**2 + (py2 - py1)**2)
                    
                    if dist < dist_threshold:
                        # Calculate angle between edges
                        vec1 = np.array([x1b - x1a, y1b - y1a])
                        vec2 = np.array([x2b - x2a, y2b - y2a])
                        
                        # Normalize vectors
                        vec1 = vec1 / np.linalg.norm(vec1)
                        vec2 = vec2 / np.linalg.norm(vec2)
                        
                        # Calculate angle
                        cos_angle = np.dot(vec1, vec2)
                        angle = np.arccos(np.clip(cos_angle, -1, 1))
                        angle_deg = np.degrees(angle)
                        
                        # Check for right angles (cube corners)
                        if 75 <= angle_deg <= 105:  # Near 90 degrees
                            corners.append({
                                'point': ((px1 + px2) // 2, (py1 + py2) // 2),
                                'edge1': edge1,
                                'edge2': edge2,
                                'angle': angle_deg
                            })
        
        return corners


class ShadowDetector:
    """Detects and analyzes shadows to identify object volume"""
    
    @staticmethod
    def detect_object_shadows(image: np.ndarray, object_bbox: Tuple[int, int, int, int],
                             shadow_threshold: int = 50) -> Optional[Dict]:
        """
        Detect shadow cast by an object
        
        Args:
            image: Sonar intensity image
            object_bbox: Object bounding box (x, y, w, h)
            shadow_threshold: Intensity threshold for shadow
            
        Returns:
            Shadow properties or None if no shadow detected
        """
        x, y, w, h = object_bbox
        
        # Look for shadow behind the object (in sonar, typically below/to the right)
        # Shadow search region
        shadow_search_height = h * 2
        shadow_search_width = w
        
        # Define shadow search area
        shadow_y_start = y + h
        shadow_y_end = min(shadow_y_start + shadow_search_height, image.shape[0])
        shadow_x_start = x
        shadow_x_end = min(x + shadow_search_width, image.shape[1])
        
        if shadow_y_end <= shadow_y_start or shadow_x_end <= shadow_x_start:
            return None
        
        # Extract shadow region
        shadow_region = image[shadow_y_start:shadow_y_end, shadow_x_start:shadow_x_end]
        
        # Check if region is darker than surroundings
        mean_intensity = np.mean(shadow_region)
        
        if mean_intensity < shadow_threshold:
            # Calculate shadow properties
            shadow_mask = shadow_region < shadow_threshold
            shadow_pixels = np.sum(shadow_mask)
            
            if shadow_pixels > 0:
                # Find shadow extent
                shadow_coords = np.where(shadow_mask)
                shadow_length = shadow_coords[0].max() - shadow_coords[0].min() if len(shadow_coords[0]) > 0 else 0
                
                return {
                    'bbox': (shadow_x_start, shadow_y_start, 
                            shadow_x_end - shadow_x_start, 
                            shadow_y_end - shadow_y_start),
                    'mean_intensity': mean_intensity,
                    'pixel_count': shadow_pixels,
                    'length': shadow_length,
                    'volume_indicator': shadow_length / h  # Ratio indicates object height
                }
        
        return None


class SpecializedSonarDetector:
    """Specialized detector for spheres, barrels, and cubes in sonar data"""
    
    def __init__(self, shadow_analysis: bool = True):
        """
        Initialize specialized detector
        
        Args:
            shadow_analysis: Enable shadow-based volume detection
        """
        self.shadow_analysis = shadow_analysis
        self.shape_analyzer = ShapeAnalyzer()
        self.shadow_detector = ShadowDetector()
    
    def detect(self, image: np.ndarray, frame_index: int = 0) -> List[DetectedObject]:
        """
        Detect specific objects in sonar image
        
        Args:
            image: Sonar intensity image
            frame_index: Current frame index
            
        Returns:
            List of detected objects classified as sphere, barrel, or cube
        """
        detections = []
        
        # Preprocess image
        processed = self._preprocess(image)
        
        # Detect spheres (crescents)
        spheres = self._detect_spheres(processed, image, frame_index)
        detections.extend(spheres)
        
        # Detect barrels (ellipses)
        barrels = self._detect_barrels(processed, image, frame_index)
        detections.extend(barrels)
        
        # Detect cubes (straight edges and corners)
        cubes = self._detect_cubes(processed, image, frame_index)
        detections.extend(cubes)
        
        # Analyze shadows for volume confirmation
        if self.shadow_analysis:
            detections = self._analyze_shadows(detections, image)
        
        return detections
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for shape detection"""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        
        # Apply bilateral filter to preserve edges
        filtered = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(filtered)
        
        return enhanced
    
    def _detect_spheres(self, processed: np.ndarray, original: np.ndarray, 
                       frame_index: int) -> List[DetectedObject]:
        """Detect spherical objects"""
        detections = []
        
        # Find crescent shapes
        crescents = self.shape_analyzer.detect_crescent_shapes(processed)
        
        for cx, cy, radius in crescents:
            # Create bounding box
            x = max(0, cx - radius)
            y = max(0, cy - radius)
            w = min(radius * 2, original.shape[1] - x)
            h = min(radius * 2, original.shape[0] - y)
            
            # Calculate properties
            roi = original[y:y+h, x:x+w]
            intensity_mean = np.mean(roi)
            intensity_std = np.std(roi)
            
            # Only create detection if intensity is significant
            if intensity_mean > 120:  # Threshold for valid object
                detection = DetectedObject(
                    bbox=(x, y, w, h),
                    confidence=0.75,  # Base confidence for sphere
                    class_name="sphere",
                    centroid=(cx, cy),
                    area=np.pi * radius * radius,
                    intensity_mean=intensity_mean,
                    intensity_std=intensity_std,
                    frame_index=frame_index
                )
                detections.append(detection)
        
        return detections
    
    def _detect_barrels(self, processed: np.ndarray, original: np.ndarray,
                       frame_index: int) -> List[DetectedObject]:
        """Detect barrel-shaped objects"""
        detections = []
        
        # Find elliptical shapes
        ellipses = self.shape_analyzer.detect_elliptical_shapes(processed)
        
        for ellipse in ellipses:
            cx, cy = ellipse['center']
            width, height = ellipse['axes']
            
            # Create bounding box
            x = int(cx - width/2)
            y = int(cy - height/2)
            w = int(width)
            h = int(height)
            
            # Ensure bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, original.shape[1] - x)
            h = min(h, original.shape[0] - y)
            
            # Calculate properties
            roi = original[y:y+h, x:x+w]
            intensity_mean = np.mean(roi) if roi.size > 0 else 0
            intensity_std = np.std(roi) if roi.size > 0 else 0
            
            # Only create detection if intensity is significant
            if intensity_mean > 120:  # Threshold for valid object
                detection = DetectedObject(
                    bbox=(x, y, w, h),
                    confidence=0.7,  # Base confidence for barrel
                    class_name="barrel",
                    centroid=(cx, cy),
                    area=ellipse['area'],
                    intensity_mean=intensity_mean,
                    intensity_std=intensity_std,
                    frame_index=frame_index
                )
                detections.append(detection)
        
        return detections
    
    def _detect_cubes(self, processed: np.ndarray, original: np.ndarray,
                     frame_index: int) -> List[DetectedObject]:
        """Detect cube-shaped objects"""
        detections = []
        
        # Find straight edges
        edges = self.shape_analyzer.detect_straight_edges(processed)
        
        # Find corner patterns
        corners = self.shape_analyzer.detect_corner_patterns(edges)
        
        # Group corners into potential cubes
        if corners:
            # Simple clustering of corners
            corner_points = [c['point'] for c in corners]
            
            # Find bounding box of corner cluster
            if corner_points:
                xs = [p[0] for p in corner_points]
                ys = [p[1] for p in corner_points]
                
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)
                
                # Check if it forms a reasonable cube shape
                width = x_max - x_min
                height = y_max - y_min
                
                if width > 30 and height > 30:  # Increased minimum size
                    # Calculate properties
                    roi = original[y_min:y_max, x_min:x_max]
                    intensity_mean = np.mean(roi) if roi.size > 0 else 0
                    intensity_std = np.std(roi) if roi.size > 0 else 0
                    
                    # Only create detection if intensity is significant
                    if intensity_mean > 120:  # Threshold for valid object
                        detection = DetectedObject(
                            bbox=(x_min, y_min, width, height),
                            confidence=0.65,  # Base confidence for cube
                            class_name="cube",
                            centroid=((x_min + x_max) / 2, (y_min + y_max) / 2),
                            area=width * height,
                            intensity_mean=intensity_mean,
                            intensity_std=intensity_std,
                            frame_index=frame_index
                        )
                        detections.append(detection)
        
        return detections
    
    def _analyze_shadows(self, detections: List[DetectedObject], 
                        image: np.ndarray) -> List[DetectedObject]:
        """Analyze shadows to confirm object volume and adjust confidence"""
        for detection in detections:
            shadow = self.shadow_detector.detect_object_shadows(image, detection.bbox)
            
            if shadow:
                # Increase confidence if shadow indicates volume
                volume_indicator = shadow['volume_indicator']
                
                if volume_indicator > 0.5:  # Significant shadow length
                    detection.confidence = min(detection.confidence * 1.3, 1.0)
                    
                    # Add shadow info to class name
                    detection.class_name = f"{detection.class_name}_with_shadow"
        
        return detections


if __name__ == "__main__":
    # Test the specialized detector
    import sys
    sys.path.append('..')
    from sonar_processor import OculusFileReader
    
    # Create test detector
    detector = SpecializedSonarDetector(shadow_analysis=True)
    
    # Test with synthetic data
    test_image = np.zeros((200, 256), dtype=np.uint8)
    
    # Add a crescent shape (sphere)
    center = (100, 50)
    radius = 20
    for angle in np.linspace(0, np.pi, 50):
        x = int(center[0] + radius * np.cos(angle))
        y = int(center[1] + radius * np.sin(angle))
        if 0 <= x < 256 and 0 <= y < 200:
            test_image[y, x] = 200
    
    # Add shadow below
    test_image[70:90, 90:110] = 30
    
    # Detect objects
    detections = detector.detect(test_image)
    
    print(f"Detected {len(detections)} objects:")
    for det in detections:
        print(f"  - {det.class_name} at {det.centroid} with confidence {det.confidence:.2f}")
