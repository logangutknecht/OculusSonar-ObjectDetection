"""
Oculus Sonar Object Detection System
Main application for processing and analyzing Oculus sonar data
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import cv2
from typing import List, Optional
import logging
from tqdm import tqdm
import yaml
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sonar_processor import OculusFileReader, SonarFrame
from sonar_filters import SonarEnhancer, MedianFilter, BilateralFilter, AdaptiveThresholdFilter
from object_detector import SonarObjectDetector, DetectedObject
from specialized_detector import SpecializedSonarDetector
from visualizer import SonarVisualizer, RealTimeVisualizer, DetectionAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SonarObjectDetectionPipeline:
    """Main pipeline for sonar object detection"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.reader = None
        self.enhancer = self._setup_enhancer()
        self.detector = self._setup_detector()
        # Only detect spheres for now. Enable optional brightest-front mode? set detect_brightest_front_only=True to use.
        self.specialized_detector = SpecializedSonarDetector(
            shadow_analysis=True,
            shapes=("sphere",),
            detect_brightest_front_only=False
        )
        self.visualizer = SonarVisualizer()
        self.rt_visualizer = RealTimeVisualizer()
        
        # Storage for results
        self.frames = []
        self.detections_per_frame = []
        
        logger.info("Pipeline initialized")
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'filters': {
                'median_kernel': 3,
                'bilateral_d': 5,
                'bilateral_sigma_color': 50,
                'bilateral_sigma_space': 50,
                'normalize_percentiles': [2, 98],
                'remove_water_column': 10,
                'range_correction_alpha': 0.1
            },
            'detection': {
                # Only use specialized detector (spheres) for now
                'methods': [],
                'combine_method': 'union',
                'min_area': 40000,  # Minimum area threshold
                # 'max_area': None,  # Optional upper bound; leave unset to allow large objects
                'confidence_threshold': 0.5,
                'min_intensity_mean': 0,
                'use_specialized': True  # Use specialized detector for spheres, barrels, cubes
            },
            'visualization': {
                'colormap': 'viridis',
                'show_realtime': True,
                'save_outputs': False,
                'output_dir': 'outputs'
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
                # Merge with defaults
                for key in default_config:
                    if key in loaded_config:
                        default_config[key].update(loaded_config[key])
        
        return default_config
    
    def _setup_enhancer(self) -> SonarEnhancer:
        """Setup sonar data enhancer with filters"""
        enhancer = SonarEnhancer()
        
        filter_config = self.config['filters']
        
        # Add filters based on configuration
        enhancer.add_filter(MedianFilter(kernel_size=filter_config['median_kernel']))
        enhancer.add_filter(BilateralFilter(
            d=filter_config['bilateral_d'],
            sigma_color=filter_config['bilateral_sigma_color'],
            sigma_space=filter_config['bilateral_sigma_space']
        ))
        
        return enhancer
    
    def _setup_detector(self) -> SonarObjectDetector:
        """Setup object detector"""
        det_config = self.config['detection']
        return SonarObjectDetector(methods=det_config['methods'])
    
    def load_file(self, file_path: str) -> int:
        """
        Load an Oculus sonar file
        
        Args:
            file_path: Path to .oculus file
            
        Returns:
            Number of frames loaded
        """
        logger.info(f"Loading file: {file_path}")
        
        self.reader = OculusFileReader(file_path)
        self.frames = self.reader.parse_all_frames()
        
        logger.info(f"Loaded {len(self.frames)} frames")
        return len(self.frames)
    
    def process_frame(self, frame: SonarFrame) -> tuple:
        """
        Process a single sonar frame
        
        Args:
            frame: SonarFrame object
            
        Returns:
            Tuple of (enhanced_image, detections)
        """
        # Get raw intensity data
        raw_image = frame.intensity_data
        
        # Apply preprocessing
        filter_config = self.config['filters']
        
        # Normalize intensity
        enhanced = SonarEnhancer.normalize_intensity(
            raw_image,
            filter_config['normalize_percentiles'][0],
            filter_config['normalize_percentiles'][1]
        )
        
        # Remove water column
        enhanced = SonarEnhancer.remove_water_column(
            enhanced,
            filter_config['remove_water_column']
        )
        
        # Apply range correction
        enhanced = SonarEnhancer.apply_range_correction(
            enhanced,
            filter_config['range_correction_alpha']
        )
        
        # Apply enhancement filters
        enhanced = self.enhancer.process(enhanced)
        
        # Detect objects
        if self.config['detection'].get('use_specialized', False):
            # Use specialized detector for specific shapes
            detections = self.specialized_detector.detect(enhanced, frame.frame_index)
            
            # Also run general detectors if configured
            if self.config['detection']['methods']:
                general_detections = self.detector.detect(
                    enhanced,
                    frame.frame_index,
                    combine_method=self.config['detection']['combine_method']
                )
                # Combine detections, prioritizing specialized ones
                detections.extend([d for d in general_detections 
                                 if not any(self._overlaps(d, sd) for sd in detections)])
        else:
            detections = self.detector.detect(
                enhanced,
                frame.frame_index,
                combine_method=self.config['detection']['combine_method']
            )
        
        # Filter by confidence
        min_conf = self.config['detection']['confidence_threshold']
        detections = [d for d in detections if d.confidence >= min_conf]
        
        # Apply minimum area threshold (no upper bound for now)
        min_area = self.config['detection'].get('min_area', 40000)
        detections = [d for d in detections if d.area >= min_area]
        
        # Filter by intensity
        min_intensity = self.config['detection'].get('min_intensity_mean', 100)
        detections = [d for d in detections if d.intensity_mean >= min_intensity]
        
        # Sort by confidence and limit number of detections
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        max_detections = self.config['detection'].get('max_detections_per_frame', 5)
        detections = detections[:max_detections]
        
        return enhanced, detections
    
    def _overlaps(self, det1: DetectedObject, det2: DetectedObject, threshold: float = 0.5) -> bool:
        """Check if two detections overlap significantly"""
        x1, y1, w1, h1 = det1.bbox
        x2, y2, w2, h2 = det2.bbox
        
        # Calculate intersection
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 < xi1 or yi2 < yi1:
            return False
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou > threshold
    
    def process_all_frames(self, show_progress: bool = True) -> None:
        """
        Process all loaded frames
        
        Args:
            show_progress: Show progress bar
        """
        if not self.frames:
            logger.error("No frames loaded")
            return
        
        logger.info(f"Processing {len(self.frames)} frames...")
        
        self.detections_per_frame = []
        
        # Setup progress bar
        iterator = tqdm(self.frames) if show_progress else self.frames
        
        for frame in iterator:
            enhanced, detections = self.process_frame(frame)
            self.detections_per_frame.append(detections)
            
            # Update progress bar description
            if show_progress:
                iterator.set_description(f"Frame {frame.frame_index}: {len(detections)} objects")
        
        logger.info(f"Processing complete. Total detections: {sum(len(d) for d in self.detections_per_frame)}")
    
    def visualize_results(self, frame_indices: Optional[List[int]] = None) -> None:
        """
        Visualize processing results
        
        Args:
            frame_indices: Specific frames to visualize (None for all)
        """
        if not self.frames or not self.detections_per_frame:
            logger.error("No processed data to visualize")
            return
        
        if frame_indices is None:
            # Show first, middle, and last frames
            n_frames = len(self.frames)
            frame_indices = [0, n_frames // 2, n_frames - 1] if n_frames > 2 else list(range(n_frames))
        
        for idx in frame_indices:
            if 0 <= idx < len(self.frames):
                frame = self.frames[idx]
                detections = self.detections_per_frame[idx]
                
                # Process frame for visualization
                enhanced, _ = self.process_frame(frame)
                
                # Create visualization
                self.visualizer.plot_sonar_frame(
                    enhanced,
                    detections,
                    title=f"Frame {idx}: {len(detections)} detections"
                )
    
    def run_realtime_visualization(self) -> None:
        """Run real-time visualization of all frames"""
        if not self.frames:
            logger.error("No frames loaded")
            return
        
        logger.info("Starting real-time visualization (press 'q' to quit, 's' to save frame)")
        
        self.rt_visualizer.start()
        
        for i, frame in enumerate(self.frames):
            # Process frame
            enhanced, detections = self.process_frame(frame)
            
            # Update visualization
            if not self.rt_visualizer.update(enhanced, detections):
                break
            
            # Control frame rate
            cv2.waitKey(100)  # 10 FPS
        
        self.rt_visualizer.stop()
    
    def analyze_detections(self) -> None:
        """Analyze and visualize detection statistics"""
        if not self.detections_per_frame:
            logger.error("No detections to analyze")
            return
        
        # Plot statistics
        DetectionAnalyzer.plot_detection_statistics(self.detections_per_frame)
        
        # Create and show heatmap
        if self.frames:
            shape = self.frames[0].intensity_data.shape
            heatmap = DetectionAnalyzer.create_heatmap(self.detections_per_frame, shape)
            
            self.visualizer.plot_sonar_frame(
                heatmap * 255,
                title="Detection Heatmap (All Frames)"
            )
    
    def save_results(self, output_dir: str = "outputs", save_all_frames: bool = False) -> None:
        """
        Save processing results to disk
        
        Args:
            output_dir: Directory to save outputs
            save_all_frames: If True, save all frames with detections, otherwise just samples
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get base filename from input file
        if self.reader and hasattr(self.reader, 'file_path'):
            base_name = Path(self.reader.file_path).stem
        else:
            base_name = "oculus_output"
        
        # Save detection data as JSON
        detections_data = []
        for frame_idx, detections in enumerate(self.detections_per_frame):
            frame_data = {
                'frame_index': frame_idx,
                'num_detections': len(detections),
                'detections': [
                    {
                        'bbox': [int(x) for x in det.bbox],
                        'shadow_bbox': ([int(x) for x in det.shadow_bbox]
                                        if getattr(det, 'shadow_bbox', None) is not None else None),
                        'confidence': float(det.confidence),
                        'class_name': det.class_name,
                        'centroid': [float(x) for x in det.centroid],
                        'area': float(det.area),
                        'intensity_mean': float(det.intensity_mean),
                        'intensity_std': float(det.intensity_std)
                    }
                    for det in detections
                ]
            }
            detections_data.append(frame_data)
        
        json_path = output_path / f'{base_name}_detections.json'
        with open(json_path, 'w') as f:
            json.dump(detections_data, f, indent=2)
        
        logger.info(f"Saved detection data to {json_path}")
        
        # Save visualizations
        if self.frames:
            # Determine which frames to save
            # By default, save ONLY frames where an object is clearly detected
            # (sphere with a shadow and above confidence threshold). If none, save samples.
            def is_clear_detection(dets: List[DetectedObject]) -> bool:
                min_conf = self.config['detection'].get('confidence_threshold', 0.5)
                for d in dets:
                    if d.class_name.startswith('sphere') and getattr(d, 'shadow_bbox', None) is not None and d.confidence >= min_conf:
                        return True
                return False

            detected_indices = [i for i, dets in enumerate(self.detections_per_frame) if is_clear_detection(dets)]

            if save_all_frames:
                frames_to_save = detected_indices or [0, len(self.frames) // 2, len(self.frames) - 1]
            else:
                # Save only detected frames (fallback to a few samples if none)
                frames_to_save = detected_indices or [0, len(self.frames) // 2, len(self.frames) - 1]
            
            for i in frames_to_save:
                if i < len(self.frames):
                    frame = self.frames[i]
                    detections = self.detections_per_frame[i] if i < len(self.detections_per_frame) else []
                    
                    # Get raw frame data
                    raw_image = frame.intensity_data
                    
                    # Normalize raw image for saving
                    raw_normalized = cv2.normalize(raw_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                    
                    # Apply colormap to raw image for better visualization
                    raw_colored = cv2.applyColorMap(raw_normalized, cv2.COLORMAP_VIRIDIS)
                    
                    # Save raw frame
                    raw_path = output_path / f'{base_name}_frame_{i:04d}_raw.png'
                    cv2.imwrite(str(raw_path), raw_colored)
                    logger.info(f"Saved raw frame {i} to {raw_path}")
                    
                    # Process and save detection overlay in rectangular view
                    # Draw boxes on the VIRIDIS-colored image to make green/red visible
                    overlay_base = raw_colored.copy()
                    output_img = self.visualizer.plot_detections_overlay(overlay_base, detections)
                    
                    detection_path = output_path / f'{base_name}_frame_{i:04d}_detections.png'
                    cv2.imwrite(str(detection_path), output_img)
                    logger.info(f"Saved detection frame {i} to {detection_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Oculus Sonar Object Detection System')
    
    parser.add_argument('input_file', type=str, nargs='?',
                       default='raw_assets/Oculus_20250805_164829.oculus',
                       help='Path to Oculus sonar file')
    
    parser.add_argument('--config', type=str, default=None,
                       help='Path to configuration file')
    
    parser.add_argument('--mode', type=str, default='all',
                       choices=['process', 'visualize', 'realtime', 'analyze', 'all'],
                       help='Processing mode')
    
    parser.add_argument('--save-outputs', action='store_true',
                       help='Save outputs to disk')
    
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory')
    
    parser.add_argument('--save-all-frames', action='store_true',
                       help='Save all frames with detections (not just samples)')
    
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                       help='Specific frame indices to visualize')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not Path(args.input_file).exists():
        logger.error(f"Input file not found: {args.input_file}")
        
        # List available files in raw_assets
        raw_assets_dir = Path('raw_assets')
        if raw_assets_dir.exists():
            oculus_files = list(raw_assets_dir.glob('*.oculus'))
            if oculus_files:
                logger.info("Available files in raw_assets:")
                for f in oculus_files:
                    logger.info(f"  - {f.name}")
                    # Show file size
                    size_mb = f.stat().st_size / (1024 * 1024)
                    logger.info(f"    Size: {size_mb:.1f} MB")
        return 1
    
    # Initialize pipeline
    pipeline = SonarObjectDetectionPipeline(args.config)
    
    # Load file
    n_frames = pipeline.load_file(args.input_file)
    if n_frames == 0:
        logger.error("No frames loaded from file")
        return 1
    
    # Process based on mode
    if args.mode in ['process', 'all']:
        pipeline.process_all_frames()
    
    if args.mode in ['visualize', 'all']:
        pipeline.visualize_results(args.frames)
    
    if args.mode == 'realtime':
        pipeline.run_realtime_visualization()
    
    if args.mode in ['analyze', 'all']:
        if args.mode == 'all' or pipeline.detections_per_frame:
            pipeline.analyze_detections()
    
    # Save outputs if requested
    if args.save_outputs:
        pipeline.save_results(args.output_dir, save_all_frames=args.save_all_frames)
    
    logger.info("Processing complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
