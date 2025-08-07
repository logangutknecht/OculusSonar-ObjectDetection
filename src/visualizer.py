"""
Visualization Module for Sonar Data and Detections
Provides real-time and static visualization capabilities
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

# Optional imports for 3D visualization
try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

logger = logging.getLogger(__name__)


class SonarVisualizer:
    """Main visualization class for sonar data"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualizer
        
        Args:
            figsize: Figure size for matplotlib plots
        """
        self.figsize = figsize
        self.colormap = 'viridis'  # Default colormap
    
    def plot_sonar_frame(self, frame_data: np.ndarray, 
                        detections: Optional[List] = None,
                        title: str = "Sonar Frame",
                        save_path: Optional[str] = None) -> None:
        """
        Plot a single sonar frame with optional detections
        
        Args:
            frame_data: 2D array of sonar intensities
            detections: List of DetectedObject instances
            title: Plot title
            save_path: Path to save figure (optional)
        """
        fig, ax = plt.subplots(1, 1, figsize=self.figsize)
        
        # Display sonar data
        im = ax.imshow(frame_data, cmap=self.colormap, aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Intensity')
        
        # Add detections if provided
        if detections:
            for det in detections:
                self._draw_detection(ax, det)
        
        ax.set_title(title)
        ax.set_xlabel('Bearing Index')
        ax.set_ylabel('Range Index')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        plt.tight_layout()
        plt.show()
    
    def plot_polar_sonar(self, frame_data: np.ndarray,
                        bearings_deg: np.ndarray,
                        range_resolution: float,
                        detections: Optional[List] = None,
                        title: str = "Polar Sonar View") -> None:
        """
        Plot sonar data in polar coordinates
        
        Args:
            frame_data: 2D array of sonar intensities [range, bearing]
            bearings_deg: Array of bearing angles in degrees
            range_resolution: Range resolution in meters
            detections: List of DetectedObject instances
            title: Plot title
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # Create meshgrid for polar plot
        ranges = np.arange(frame_data.shape[0]) * range_resolution
        bearings_rad = np.radians(bearings_deg)
        
        # Create meshgrid
        R, Theta = np.meshgrid(ranges, bearings_rad, indexing='ij')
        
        # Plot sonar data
        c = ax.pcolormesh(Theta, R, frame_data, cmap=self.colormap, shading='auto')
        
        # Add colorbar
        plt.colorbar(c, ax=ax, label='Intensity')
        
        # Configure polar plot
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)  # Clockwise
        
        ax.set_title(title)
        
        plt.tight_layout()
        plt.show()
    
    def plot_detections_overlay(self, image: np.ndarray,
                               detections: List,
                               confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Overlay detections on image
        
        Args:
            image: Input image
            detections: List of DetectedObject instances
            confidence_threshold: Minimum confidence to display
            
        Returns:
            Image with overlaid detections
        """
        # Convert to color if grayscale
        if len(image.shape) == 2:
            output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            output = image.copy()
        
        for det in detections:
            if det.confidence >= confidence_threshold:
                # Draw bounding box
                x, y, w, h = det.bbox
                color = self._get_color_for_confidence(det.confidence)
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = f"{det.class_name}: {det.confidence:.2f}"
                cv2.putText(output, label, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw centroid
                cx, cy = int(det.centroid[0]), int(det.centroid[1])
                cv2.circle(output, (cx, cy), 3, (0, 255, 0), -1)
        
        return output
    
    def create_3d_plot(self, x: np.ndarray, y: np.ndarray, 
                      intensities: np.ndarray,
                      title: str = "3D Sonar View"):
        """
        Create interactive 3D plot of sonar data
        
        Args:
            x: X coordinates (Cartesian)
            y: Y coordinates (Cartesian)
            intensities: Intensity values
            title: Plot title
            
        Returns:
            Plotly figure object (if plotly installed) or None
        """
        if not HAS_PLOTLY:
            logger.warning("Plotly not installed. 3D visualization unavailable.")
            return None
            
        # Flatten arrays for scatter plot
        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = np.zeros_like(x_flat)  # Assume flat seafloor
        intensity_flat = intensities.flatten()
        
        # Filter out low intensity points for clarity
        mask = intensity_flat > np.percentile(intensity_flat, 25)
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x_flat[mask],
            y=y_flat[mask],
            z=z_flat[mask],
            mode='markers',
            marker=dict(
                size=2,
                color=intensity_flat[mask],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Intensity")
            ),
            text=[f"Intensity: {i:.1f}" for i in intensity_flat[mask]],
            hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>%{text}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data'
            ),
            height=700
        )
        
        return fig
    
    def _draw_detection(self, ax, detection):
        """Draw a single detection on matplotlib axis"""
        x, y, w, h = detection.bbox
        
        # Choose color based on confidence
        color = self._get_color_for_confidence(detection.confidence)
        
        # Draw bounding box
        rect = patches.Rectangle((x, y), w, h, linewidth=2,
                                edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        
        # Add label
        label = f"{detection.class_name}\n{detection.confidence:.2f}"
        ax.text(x, y - 5, label, color='white', fontsize=8,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.5))
        
        # Mark centroid
        cx, cy = detection.centroid
        ax.plot(cx, cy, 'g+', markersize=10, markeredgewidth=2)
    
    def _get_color_for_confidence(self, confidence: float) -> Tuple[int, int, int]:
        """Get color based on confidence level"""
        if confidence > 0.8:
            return (0, 255, 0)  # Green for high confidence
        elif confidence > 0.5:
            return (255, 255, 0)  # Yellow for medium confidence
        else:
            return (255, 0, 0)  # Red for low confidence


class RealTimeVisualizer:
    """Real-time visualization for sonar data streams"""
    
    def __init__(self, window_name: str = "Sonar Detection"):
        """
        Initialize real-time visualizer
        
        Args:
            window_name: OpenCV window name
        """
        self.window_name = window_name
        self.is_running = False
        self.frame_buffer = []
        self.detection_buffer = []
    
    def start(self):
        """Start visualization window"""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        self.is_running = True
        logger.info("Real-time visualization started")
    
    def stop(self):
        """Stop visualization and close window"""
        self.is_running = False
        cv2.destroyWindow(self.window_name)
        logger.info("Real-time visualization stopped")
    
    def update(self, frame: np.ndarray, detections: Optional[List] = None):
        """
        Update visualization with new frame
        
        Args:
            frame: Sonar intensity frame
            detections: List of detected objects
        """
        if not self.is_running:
            self.start()
        
        # Normalize frame for display
        display_frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Convert to color
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
        
        # Apply colormap
        display_frame = cv2.applyColorMap(display_frame, cv2.COLORMAP_VIRIDIS)
        
        # Add detections
        if detections:
            for det in detections:
                self._draw_detection_cv2(display_frame, det)
        
        # Add frame info
        info_text = f"Frame: {len(self.frame_buffer)}"
        if detections:
            info_text += f" | Objects: {len(detections)}"
        cv2.putText(display_frame, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display
        cv2.imshow(self.window_name, display_frame)
        
        # Store in buffer
        self.frame_buffer.append(frame)
        self.detection_buffer.append(detections)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop()
            return False
        elif key == ord('s'):
            # Save current frame
            cv2.imwrite(f"frame_{len(self.frame_buffer)}.png", display_frame)
            logger.info(f"Saved frame {len(self.frame_buffer)}")
        
        return True
    
    def _draw_detection_cv2(self, image: np.ndarray, detection):
        """Draw detection using OpenCV"""
        x, y, w, h = detection.bbox
        
        # Choose color based on confidence
        if detection.confidence > 0.8:
            color = (0, 255, 0)  # Green
        elif detection.confidence > 0.5:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        # Draw bounding box
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        # Add label
        label = f"{detection.class_name}: {detection.confidence:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw label background
        cv2.rectangle(image, (x, y - label_size[1] - 10),
                     (x + label_size[0], y), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Draw centroid
        cx, cy = int(detection.centroid[0]), int(detection.centroid[1])
        cv2.circle(image, (cx, cy), 3, (0, 255, 0), -1)


class DetectionAnalyzer:
    """Analyze and visualize detection statistics"""
    
    @staticmethod
    def plot_detection_statistics(detections_per_frame: List[List],
                                 save_path: Optional[str] = None):
        """
        Plot detection statistics over time
        
        Args:
            detections_per_frame: List of detection lists for each frame
            save_path: Path to save figure
        """
        # Calculate statistics
        num_detections = [len(dets) for dets in detections_per_frame]
        avg_confidence = []
        
        for dets in detections_per_frame:
            if dets:
                avg_conf = np.mean([d.confidence for d in dets])
                avg_confidence.append(avg_conf)
            else:
                avg_confidence.append(0)
        
        # Create subplots
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot number of detections
        axes[0].plot(num_detections, 'b-', linewidth=2)
        axes[0].fill_between(range(len(num_detections)), num_detections, alpha=0.3)
        axes[0].set_ylabel('Number of Detections')
        axes[0].set_title('Detection Count Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Plot average confidence
        axes[1].plot(avg_confidence, 'r-', linewidth=2)
        axes[1].fill_between(range(len(avg_confidence)), avg_confidence, alpha=0.3)
        axes[1].set_ylabel('Average Confidence')
        axes[1].set_xlabel('Frame Number')
        axes[1].set_title('Detection Confidence Over Time')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved statistics to {save_path}")
        
        plt.show()
    
    @staticmethod
    def create_heatmap(detections_per_frame: List[List],
                      image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create heatmap of detection locations
        
        Args:
            detections_per_frame: List of detection lists for each frame
            image_shape: Shape of the sonar images
            
        Returns:
            Heatmap array
        """
        heatmap = np.zeros(image_shape, dtype=np.float32)
        
        for detections in detections_per_frame:
            for det in detections:
                x, y, w, h = det.bbox
                # Add Gaussian blob at detection location
                cy, cx = int(y + h/2), int(x + w/2)
                
                # Create Gaussian kernel
                size = max(w, h)
                kernel = cv2.getGaussianKernel(size, size/3)
                kernel = kernel * kernel.T
                kernel = kernel / kernel.max() * det.confidence
                
                # Add to heatmap (with bounds checking)
                y1 = max(0, cy - size//2)
                y2 = min(image_shape[0], cy + size//2)
                x1 = max(0, cx - size//2)
                x2 = min(image_shape[1], cx + size//2)
                
                kernel_y1 = 0 if cy >= size//2 else size//2 - cy
                kernel_y2 = size if cy + size//2 <= image_shape[0] else size - (cy + size//2 - image_shape[0])
                kernel_x1 = 0 if cx >= size//2 else size//2 - cx
                kernel_x2 = size if cx + size//2 <= image_shape[1] else size - (cx + size//2 - image_shape[1])
                
                if y2 > y1 and x2 > x1:
                    heatmap[y1:y2, x1:x2] += kernel[kernel_y1:kernel_y2, kernel_x1:kernel_x2]
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap


if __name__ == "__main__":
    # Test visualization with synthetic data
    from object_detector import DetectedObject
    
    # Create synthetic sonar frame
    frame = np.random.randint(0, 100, (200, 256), dtype=np.uint8)
    frame[50:100, 100:150] = 200  # Add bright region
    
    # Create synthetic detections
    detections = [
        DetectedObject(
            bbox=(100, 50, 50, 50),
            confidence=0.85,
            class_name="object1",
            centroid=(125, 75),
            area=2500,
            intensity_mean=200,
            intensity_std=10,
            frame_index=0
        ),
        DetectedObject(
            bbox=(30, 120, 40, 40),
            confidence=0.65,
            class_name="object2",
            centroid=(50, 140),
            area=1600,
            intensity_mean=150,
            intensity_std=15,
            frame_index=0
        )
    ]
    
    # Test static visualization
    viz = SonarVisualizer()
    viz.plot_sonar_frame(frame, detections, title="Test Sonar Frame")
    
    # Test real-time visualization
    rt_viz = RealTimeVisualizer()
    rt_viz.start()
    
    for i in range(10):
        # Simulate frame updates
        frame_copy = frame.copy()
        frame_copy += np.random.randint(-10, 10, frame.shape, dtype=np.int16)
        frame_copy = np.clip(frame_copy, 0, 255).astype(np.uint8)
        
        if not rt_viz.update(frame_copy, detections):
            break
        
        cv2.waitKey(100)  # Simulate frame rate
    
    rt_viz.stop()
