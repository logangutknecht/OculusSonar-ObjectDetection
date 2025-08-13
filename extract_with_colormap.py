"""
Extract sonar frames with optional colormap for easier visualization
"""

import sys
import os
from pathlib import Path
import cv2
import numpy as np
import logging

sys.path.append('src')
from sonar_processor import OculusFileReader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_with_colormap(sonar_file: str, output_dir: str, 
                          colormap: str = 'viridis',
                          frame_interval: int = 10,
                          max_frames: int = None):
    """
    Extract frames with optional colormap applied
    
    Args:
        sonar_file: Path to .oculus file
        output_dir: Directory to save frames
        colormap: Colormap to apply ('viridis', 'jet', 'hot', 'bone', 'gray', None)
        frame_interval: Extract every Nth frame
        max_frames: Maximum number of frames
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Map string names to OpenCV colormaps
    colormap_dict = {
        'viridis': cv2.COLORMAP_VIRIDIS,
        'jet': cv2.COLORMAP_JET,
        'hot': cv2.COLORMAP_HOT,
        'bone': cv2.COLORMAP_BONE,
        'turbo': cv2.COLORMAP_TURBO,
        'rainbow': cv2.COLORMAP_RAINBOW,
        'ocean': cv2.COLORMAP_OCEAN,
        'gray': None,  # No colormap
        None: None
    }
    
    cmap = colormap_dict.get(colormap, None)
    
    logger.info(f"Extracting frames with colormap: {colormap}")
    
    # Load sonar data
    reader = OculusFileReader(sonar_file)
    frames = reader.parse_all_frames()
    
    # Select frames
    frame_indices = range(0, len(frames), frame_interval)
    if max_frames:
        frame_indices = list(frame_indices)[:max_frames]
    
    for idx in frame_indices:
        frame = frames[idx]
        raw_data = frame.intensity_data
        
        # Normalize to 0-255
        if raw_data.max() > 0:
            # Use percentiles for better contrast
            p_low = np.percentile(raw_data[raw_data > 0], 2)
            p_high = np.percentile(raw_data[raw_data > 0], 98)
            normalized = np.clip(raw_data, p_low, p_high)
            normalized = ((normalized - p_low) / (p_high - p_low) * 255).astype(np.uint8)
        else:
            normalized = raw_data.astype(np.uint8)
        
        # Apply colormap if requested
        if cmap is not None:
            colored = cv2.applyColorMap(normalized, cmap)
        else:
            # Keep as grayscale (but save as 3-channel for consistency)
            colored = cv2.cvtColor(normalized, cv2.COLOR_GRAY2BGR)
        
        # Save
        filename = f"frame_{idx:05d}_{colormap if colormap else 'gray'}.png"
        cv2.imwrite(str(output_path / filename), colored)
    
    logger.info(f"Extracted {len(list(frame_indices))} frames to {output_path}")


def create_comparison():
    """Create a comparison of different colormaps"""
    import matplotlib.pyplot as plt
    
    # Load a sample frame
    sonar_files = list(Path("raw_assets").glob("*.oculus"))
    if not sonar_files:
        return
    
    reader = OculusFileReader(str(sonar_files[0]))
    frames = reader.parse_all_frames()
    
    if len(frames) > 100:
        frame = frames[100]  # Get frame 100
    else:
        frame = frames[0]
    
    raw_data = frame.intensity_data
    
    # Normalize
    normalized = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Create different versions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Grayscale (best for ML)
    axes[0, 0].imshow(normalized, cmap='gray')
    axes[0, 0].set_title('Grayscale (Best for ML)')
    axes[0, 0].axis('off')
    
    # Viridis
    viridis = cv2.applyColorMap(normalized, cv2.COLORMAP_VIRIDIS)
    axes[0, 1].imshow(cv2.cvtColor(viridis, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Viridis Colormap')
    axes[0, 1].axis('off')
    
    # Jet
    jet = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    axes[0, 2].imshow(cv2.cvtColor(jet, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('Jet Colormap')
    axes[0, 2].axis('off')
    
    # Hot
    hot = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
    axes[1, 0].imshow(cv2.cvtColor(hot, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Hot Colormap')
    axes[1, 0].axis('off')
    
    # Turbo
    turbo = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    axes[1, 1].imshow(cv2.cvtColor(turbo, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Turbo Colormap')
    axes[1, 1].axis('off')
    
    # Bone
    bone = cv2.applyColorMap(normalized, cv2.COLORMAP_BONE)
    axes[1, 2].imshow(cv2.cvtColor(bone, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('Bone Colormap')
    axes[1, 2].axis('off')
    
    plt.suptitle('Sonar Image Colormap Comparison\n(Grayscale is best for ML training)')
    plt.tight_layout()
    plt.savefig('colormap_comparison.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    print("\nColormap comparison saved to colormap_comparison.png")
    print("\nRecommendation: Use grayscale for ML training")
    print("Colormaps are just for human visualization!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract sonar frames with colormap")
    parser.add_argument('--colormap', type=str, default='gray',
                       choices=['gray', 'viridis', 'jet', 'hot', 'bone', 'turbo', 'rainbow', 'ocean'],
                       help='Colormap to apply')
    parser.add_argument('--compare', action='store_true',
                       help='Show colormap comparison')
    parser.add_argument('--output', type=str, default='training_data/colormap_images',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.compare:
        create_comparison()
    else:
        # Extract with specified colormap
        sonar_files = list(Path("raw_assets").glob("*.oculus"))
        if sonar_files:
            print(f"Extracting frames with {args.colormap} colormap...")
            for sonar_file in sonar_files[:3]:  # Do first 3 files as example
                extract_with_colormap(
                    str(sonar_file),
                    args.output,
                    colormap=args.colormap,
                    frame_interval=50,
                    max_frames=5
                )
            print(f"\nFrames saved to {args.output}")
            print("\nNote: Grayscale is recommended for ML training!")
            print("Colormaps are just visual aids for humans.")

