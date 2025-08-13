"""
Extract RAW unfiltered frames from Oculus sonar files for annotation
Preserves original sonar data without any enhancement or filtering
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import logging

# Add src to path
sys.path.append('src')

from sonar_processor import OculusFileReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_raw_frames(sonar_file: str, output_dir: str, 
                       frame_interval: int = 10,
                       max_frames: int = None,
                       apply_minimal_processing: bool = False):
    """
    Extract raw frames from sonar file with minimal or no processing
    
    Args:
        sonar_file: Path to .oculus file
        output_dir: Directory to save frames
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames to extract
        apply_minimal_processing: If True, apply minimal contrast adjustment
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Extracting raw frames from {sonar_file}")
    
    # Load sonar data
    reader = OculusFileReader(sonar_file)
    frames = reader.parse_all_frames()
    
    logger.info(f"Found {len(frames)} total frames")
    
    # Select frames to extract
    frame_indices = range(0, len(frames), frame_interval)
    if max_frames:
        frame_indices = list(frame_indices)[:max_frames]
    
    extracted = 0
    for idx in frame_indices:
        frame = frames[idx]
        
        # Get RAW intensity data - no filtering!
        raw_data = frame.intensity_data
        
        if apply_minimal_processing:
            # Only apply minimal processing for visibility
            # Just normalize to 0-255 range, no other filters
            
            # Find actual data range (ignore zero padding)
            non_zero = raw_data[raw_data > 0]
            if len(non_zero) > 0:
                # Use percentiles to avoid outliers affecting normalization
                p_low = np.percentile(non_zero, 1)  # 1st percentile
                p_high = np.percentile(non_zero, 99)  # 99th percentile
                
                # Normalize using percentile range
                normalized = np.clip(raw_data, p_low, p_high)
                normalized = ((normalized - p_low) / (p_high - p_low) * 255).astype(np.uint8)
            else:
                # Fallback to simple normalization
                normalized = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        else:
            # Completely raw - just scale to 0-255 range
            # This preserves the original intensity relationships
            if raw_data.max() > 0:
                normalized = (raw_data * 255.0 / raw_data.max()).astype(np.uint8)
            else:
                normalized = raw_data.astype(np.uint8)
        
        # Save frame
        filename = f"frame_{idx:05d}_raw.png"
        output_file = output_path / filename
        cv2.imwrite(str(output_file), normalized)
        
        extracted += 1
        
        if extracted % 10 == 0:
            logger.info(f"Extracted {extracted} frames...")
    
    logger.info(f"Successfully extracted {extracted} raw frames to {output_path}")
    return extracted


def extract_from_all_files(raw_assets_dir: str = "raw_assets",
                          output_dir: str = "training_data/raw_images",
                          frames_per_file: int = 10):
    """
    Extract raw frames from all .oculus files in directory
    
    Args:
        raw_assets_dir: Directory containing .oculus files
        output_dir: Output directory for frames
        frames_per_file: Max frames to extract per file
    """
    raw_path = Path(raw_assets_dir)
    oculus_files = list(raw_path.glob("*.oculus"))
    
    if not oculus_files:
        logger.error(f"No .oculus files found in {raw_assets_dir}")
        return
    
    logger.info(f"Found {len(oculus_files)} .oculus files")
    
    # Clear output directory
    output_path = Path(output_dir)
    if output_path.exists():
        response = input(f"\nOutput directory '{output_dir}' exists. Clear it? (y/n): ")
        if response.lower() == 'y':
            import shutil
            shutil.rmtree(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.mkdir(parents=True, exist_ok=True)
    
    total_extracted = 0
    
    for sonar_file in oculus_files:
        logger.info(f"\nProcessing {sonar_file.name}...")
        
        # Calculate frame interval to get desired number of frames
        try:
            reader = OculusFileReader(str(sonar_file))
            total_frames = len(reader.frames)
            
            # Calculate interval to get approximately frames_per_file frames
            interval = max(1, total_frames // frames_per_file)
            
            extracted = extract_raw_frames(
                str(sonar_file),
                output_dir,
                frame_interval=interval,
                max_frames=frames_per_file,
                apply_minimal_processing=False  # Keep completely raw
            )
            total_extracted += extracted
            
        except Exception as e:
            logger.error(f"Error processing {sonar_file}: {e}")
            continue
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Extraction complete!")
    logger.info(f"Total frames extracted: {total_extracted}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"{'='*60}")


def compare_raw_vs_filtered(sonar_file: str, frame_index: int = 0):
    """
    Show comparison between raw and filtered image
    """
    import matplotlib.pyplot as plt
    sys.path.append('src')
    from sonar_filters import SonarEnhancer, MedianFilter, BilateralFilter
    
    # Load frame
    reader = OculusFileReader(sonar_file)
    frames = reader.parse_all_frames()
    
    if frame_index >= len(frames):
        logger.error(f"Frame index {frame_index} out of range (max: {len(frames)-1})")
        return
    
    frame = frames[frame_index]
    raw_data = frame.intensity_data
    
    # Create different versions
    # 1. Completely raw
    raw_normalized = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 2. With CLAHE only (contrast enhancement)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_enhanced = clahe.apply(raw_normalized)
    
    # 3. With full filtering
    enhancer = SonarEnhancer()
    enhancer.add_filter(MedianFilter(kernel_size=3))
    enhancer.add_filter(BilateralFilter(d=5, sigma_color=50, sigma_space=50))
    filtered = enhancer.process(raw_normalized)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(raw_normalized, cmap='gray')
    axes[0].set_title('RAW (Best for Annotation)')
    axes[0].axis('off')
    
    axes[1].imshow(clahe_enhanced, cmap='gray')
    axes[1].set_title('CLAHE Enhanced')
    axes[1].axis('off')
    
    axes[2].imshow(filtered, cmap='gray')
    axes[2].set_title('Fully Filtered')
    axes[2].axis('off')
    
    plt.suptitle(f'Frame {frame_index} - Raw vs Filtered Comparison')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract raw sonar frames for annotation")
    parser.add_argument('--input', type=str, default='raw_assets',
                       help='Input directory with .oculus files or single file')
    parser.add_argument('--output', type=str, default='training_data/raw_images',
                       help='Output directory for frames')
    parser.add_argument('--frames-per-file', type=int, default=15,
                       help='Number of frames to extract per file')
    parser.add_argument('--compare', action='store_true',
                       help='Show comparison of raw vs filtered')
    parser.add_argument('--frame-index', type=int, default=0,
                       help='Frame index for comparison')
    
    args = parser.parse_args()
    
    if args.compare:
        # Show comparison
        oculus_files = list(Path(args.input).glob("*.oculus"))
        if oculus_files:
            print(f"Comparing raw vs filtered for {oculus_files[0]}")
            compare_raw_vs_filtered(str(oculus_files[0]), args.frame_index)
        else:
            print("No .oculus files found")
    else:
        # Check if input is a file or directory
        input_path = Path(args.input)
        
        if input_path.is_file() and input_path.suffix == '.oculus':
            # Single file
            extract_raw_frames(
                str(input_path),
                args.output,
                frame_interval=10,
                apply_minimal_processing=False
            )
        else:
            # Directory of files
            extract_from_all_files(
                args.input,
                args.output,
                args.frames_per_file
            )
    
    print("\nYou can now annotate the raw images using:")
    print("  python annotate_simple.py")
    print("or")
    print("  python src/ml_detector/data_annotator.py")
