"""
Sonar Data Filtering and Enhancement Module
Implements various filters for improving sonar data quality
"""

import numpy as np
import cv2
from scipy import ndimage, signal
from scipy.ndimage import morphology
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SonarFilter:
    """Base class for sonar data filters"""
    
    def __init__(self, name: str):
        self.name = name
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply filter to sonar data"""
        raise NotImplementedError


class MedianFilter(SonarFilter):
    """Median filter for noise reduction"""
    
    def __init__(self, kernel_size: int = 3):
        super().__init__("Median Filter")
        self.kernel_size = kernel_size
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply median filter to reduce speckle noise"""
        return cv2.medianBlur(data.astype(np.uint8), self.kernel_size)


class BilateralFilter(SonarFilter):
    """Bilateral filter for edge-preserving smoothing"""
    
    def __init__(self, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
        super().__init__("Bilateral Filter")
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply bilateral filter for edge-preserving smoothing"""
        return cv2.bilateralFilter(
            data.astype(np.uint8), 
            self.d, 
            self.sigma_color, 
            self.sigma_space
        )


class AdaptiveThresholdFilter(SonarFilter):
    """Adaptive threshold for object segmentation"""
    
    def __init__(self, block_size: int = 11, C: float = 2):
        super().__init__("Adaptive Threshold")
        self.block_size = block_size
        self.C = C
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding"""
        return cv2.adaptiveThreshold(
            data.astype(np.uint8),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.block_size,
            self.C
        )


class MorphologicalFilter(SonarFilter):
    """Morphological operations for shape refinement"""
    
    def __init__(self, operation: str = 'close', kernel_size: int = 3):
        super().__init__(f"Morphological {operation}")
        self.operation = operation
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply morphological operation"""
        if self.operation == 'close':
            return cv2.morphologyEx(data.astype(np.uint8), cv2.MORPH_CLOSE, self.kernel)
        elif self.operation == 'open':
            return cv2.morphologyEx(data.astype(np.uint8), cv2.MORPH_OPEN, self.kernel)
        elif self.operation == 'gradient':
            return cv2.morphologyEx(data.astype(np.uint8), cv2.MORPH_GRADIENT, self.kernel)
        else:
            return data


class SonarEnhancer:
    """Advanced sonar data enhancement pipeline"""
    
    def __init__(self):
        self.filters = []
    
    def add_filter(self, filter_obj: SonarFilter):
        """Add a filter to the pipeline"""
        self.filters.append(filter_obj)
        logger.info(f"Added {filter_obj.name} to pipeline")
    
    def process(self, data: np.ndarray) -> np.ndarray:
        """Process data through all filters"""
        result = data.copy()
        for filter_obj in self.filters:
            result = filter_obj.apply(result)
        return result
    
    @staticmethod
    def normalize_intensity(data: np.ndarray, 
                          percentile_low: float = 2, 
                          percentile_high: float = 98) -> np.ndarray:
        """
        Normalize intensity values using percentile clipping
        
        Args:
            data: Input intensity data
            percentile_low: Lower percentile for clipping
            percentile_high: Upper percentile for clipping
            
        Returns:
            Normalized data in range [0, 255]
        """
        # Calculate percentiles
        p_low = np.percentile(data, percentile_low)
        p_high = np.percentile(data, percentile_high)
        
        # Clip and normalize
        clipped = np.clip(data, p_low, p_high)
        normalized = ((clipped - p_low) / (p_high - p_low) * 255).astype(np.uint8)
        
        return normalized
    
    @staticmethod
    def enhance_contrast(data: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        
        Args:
            data: Input intensity data
            clip_limit: Threshold for contrast limiting
            
        Returns:
            Contrast enhanced data
        """
        # Create CLAHE object
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        
        # Apply CLAHE
        enhanced = clahe.apply(data.astype(np.uint8))
        
        return enhanced
    
    @staticmethod
    def remove_water_column(data: np.ndarray, 
                          min_range_bins: int = 10) -> np.ndarray:
        """
        Remove water column noise near the sonar
        
        Args:
            data: Input intensity data [range, bearing]
            min_range_bins: Number of range bins to zero out
            
        Returns:
            Data with water column removed
        """
        result = data.copy()
        result[:min_range_bins, :] = 0
        return result
    
    @staticmethod
    def apply_range_correction(data: np.ndarray, 
                             alpha: float = 0.1) -> np.ndarray:
        """
        Apply Time-Varying Gain (TVG) correction for range-dependent attenuation
        
        Args:
            data: Input intensity data [range, bearing]
            alpha: Attenuation coefficient
            
        Returns:
            Range-corrected data
        """
        range_count, bearing_count = data.shape
        
        # Create range correction factor
        ranges = np.arange(1, range_count + 1)
        tvg = 20 * np.log10(ranges) + 2 * alpha * ranges
        tvg = tvg[:, np.newaxis]  # Make it broadcastable
        
        # Normalize TVG to prevent overflow
        tvg = tvg / tvg.max() * 2.0
        
        # Apply correction
        corrected = data.astype(np.float32) * tvg
        
        # Clip to valid range
        corrected = np.clip(corrected, 0, 255).astype(np.uint8)
        
        return corrected
    
    @staticmethod
    def detect_shadows(data: np.ndarray, 
                      threshold: int = 30,
                      min_shadow_length: int = 5) -> np.ndarray:
        """
        Detect acoustic shadows (dark regions behind objects)
        
        Args:
            data: Input intensity data [range, bearing]
            threshold: Intensity threshold for shadow detection
            min_shadow_length: Minimum consecutive dark pixels to be considered shadow
            
        Returns:
            Binary mask of detected shadows
        """
        shadows = np.zeros_like(data, dtype=np.uint8)
        
        # Process each bearing
        for b in range(data.shape[1]):
            beam = data[:, b]
            
            # Find dark regions
            dark_mask = beam < threshold
            
            # Find consecutive dark regions
            labeled, num_features = ndimage.label(dark_mask)
            
            for i in range(1, num_features + 1):
                shadow_region = labeled == i
                if np.sum(shadow_region) >= min_shadow_length:
                    shadows[:, b][shadow_region] = 255
        
        return shadows
    
    @staticmethod
    def beam_pattern_correction(data: np.ndarray, 
                               beam_width_deg: float = 1.0) -> np.ndarray:
        """
        Correct for beam pattern effects
        
        Args:
            data: Input intensity data [range, bearing]
            beam_width_deg: Beam width in degrees
            
        Returns:
            Beam-pattern corrected data
        """
        range_count, bearing_count = data.shape
        
        # Create beam pattern correction
        # Typically stronger returns in the center of the beam
        bearing_correction = np.ones(bearing_count)
        center = bearing_count // 2
        
        for i in range(bearing_count):
            angle_offset = abs(i - center) / bearing_count * 90  # Assuming ±45° total FOV
            # Simple cosine correction
            bearing_correction[i] = np.cos(np.radians(angle_offset * 0.5))
        
        # Apply correction
        corrected = data.astype(np.float32) * bearing_correction[np.newaxis, :]
        
        return np.clip(corrected, 0, 255).astype(np.uint8)


class SonarDenoiser:
    """Advanced denoising techniques for sonar data"""
    
    @staticmethod
    def wavelet_denoise(data: np.ndarray, wavelet: str = 'db4', level: int = 3) -> np.ndarray:
        """
        Apply wavelet denoising
        
        Args:
            data: Input intensity data
            wavelet: Wavelet type
            level: Decomposition level
            
        Returns:
            Denoised data
        """
        try:
            import pywt
        except ImportError:
            # If pywt not installed, return original data
            return data
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec2(data.astype(np.float32), wavelet, level=level)
        
        # Estimate noise level (using MAD of finest detail coefficients)
        sigma = np.median(np.abs(coeffs[-1][-1])) / 0.6745
        
        # Threshold coefficients
        threshold = sigma * np.sqrt(2 * np.log(data.size))
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [
            tuple([pywt.threshold(c, threshold, 'soft') for c in level_coeffs])
            for level_coeffs in coeffs_thresh[1:]
        ]
        
        # Reconstruct
        denoised = pywt.waverec2(coeffs_thresh, wavelet)
        
        # Ensure same shape and type
        denoised = denoised[:data.shape[0], :data.shape[1]]
        
        return np.clip(denoised, 0, 255).astype(np.uint8)
    
    @staticmethod
    def non_local_means(data: np.ndarray, h: float = 10, template_window: int = 7, search_window: int = 21) -> np.ndarray:
        """
        Apply Non-Local Means denoising
        
        Args:
            data: Input intensity data
            h: Filter strength
            template_window: Size of template patch
            search_window: Size of search area
            
        Returns:
            Denoised data
        """
        return cv2.fastNlMeansDenoising(
            data.astype(np.uint8),
            None,
            h,
            template_window,
            search_window
        )


def create_default_pipeline() -> SonarEnhancer:
    """Create a default sonar enhancement pipeline"""
    enhancer = SonarEnhancer()
    
    # Add filters in order
    enhancer.add_filter(MedianFilter(kernel_size=3))
    enhancer.add_filter(BilateralFilter(d=5, sigma_color=50, sigma_space=50))
    
    return enhancer


if __name__ == "__main__":
    # Test filters with synthetic data
    import matplotlib.pyplot as plt
    
    # Create synthetic sonar data
    range_count, bearing_count = 200, 256
    synthetic_data = np.random.randint(0, 100, (range_count, bearing_count), dtype=np.uint8)
    
    # Add some structure
    synthetic_data[50:100, 100:150] = 200  # Bright object
    synthetic_data[100:150, 100:150] = 20  # Shadow
    
    # Create enhancement pipeline
    enhancer = create_default_pipeline()
    
    # Process data
    enhanced = enhancer.process(synthetic_data)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(synthetic_data, cmap='gray')
    axes[0].set_title('Original')
    axes[1].imshow(enhanced, cmap='gray')
    axes[1].set_title('Enhanced')
    plt.tight_layout()
    plt.show()
