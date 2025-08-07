"""
Oculus Sonar Data Processor
Main module for loading, processing, and analyzing Oculus sonar data
"""

import struct
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SonarFrame:
    """Container for a single sonar frame with metadata"""
    timestamp: float
    range_resolution: float  # meters per range bin
    range_count: int  # number of range bins
    bearing_count: int  # number of beams
    bearings: np.ndarray  # beam angles in decidegrees
    intensity_data: np.ndarray  # 2D array [range, bearing]
    frame_index: int
    
    @property
    def bearings_rad(self) -> np.ndarray:
        """Get bearing angles in radians"""
        return self.bearings * np.pi / 1800.0
    
    @property
    def bearings_deg(self) -> np.ndarray:
        """Get bearing angles in degrees"""
        return self.bearings / 10.0
    
    @property
    def max_range(self) -> float:
        """Get maximum range in meters"""
        return self.range_resolution * self.range_count


class OculusFileReader:
    """Reader for Oculus .oculus binary files"""
    
    # Oculus message constants
    OCULUS_ID = 0x4F53
    SIMPLEFIRE_V2_MSG_ID = 35
    HEADER_SIZE = 16
    
    def __init__(self, file_path: str):
        """
        Initialize the Oculus file reader
        
        Args:
            file_path: Path to the .oculus file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.messages = []
        self.frames = []
        logger.info(f"Initialized reader for: {self.file_path}")
    
    def load_file(self) -> List[bytes]:
        """
        Load and parse the .oculus file to extract messages
        
        Returns:
            List of raw message bytes
        """
        with open(self.file_path, "rb") as f:
            data = f.read()
        
        messages = []
        i = 0
        
        while i < len(data) - self.HEADER_SIZE:
            # Check for Oculus message header
            oculus_id = struct.unpack_from('<H', data, i)[0]
            message_id = struct.unpack_from('<H', data, i + 6)[0]
            
            if oculus_id == self.OCULUS_ID and message_id == self.SIMPLEFIRE_V2_MSG_ID:
                # Extract payload size and full message
                payload_size = struct.unpack_from('<I', data, i + 10)[0]
                full_size = self.HEADER_SIZE + payload_size
                
                if i + full_size > len(data):
                    break
                    
                messages.append(data[i:i+full_size])
                i += full_size
            else:
                i += 1
        
        self.messages = messages
        logger.info(f"Loaded {len(messages)} sonar frames from {self.file_path}")
        return messages
    
    def parse_frame(self, msg_bytes: bytes, frame_index: int) -> SonarFrame:
        """
        Parse a SimpleFire V2 message into a SonarFrame
        
        Args:
            msg_bytes: Raw message bytes
            frame_index: Index of this frame in the sequence
            
        Returns:
            Parsed SonarFrame object
        """
        # Skip header and fixed fields to get to variable data
        offset = 162  # Range resolution field offset from documentation
        
        # Parse metadata
        range_res = struct.unpack_from('<d', msg_bytes, offset)[0]
        offset += 8
        range_count = struct.unpack_from('<H', msg_bytes, offset)[0]
        offset += 2
        bearing_count = struct.unpack_from('<H', msg_bytes, offset)[0]
        offset += 2
        offset += 16  # Skip reserved fields
        
        image_offset = struct.unpack_from('<I', msg_bytes, offset)[0]
        offset += 4
        image_size = struct.unpack_from('<I', msg_bytes, offset)[0]
        offset += 4
        message_size = struct.unpack_from('<I', msg_bytes, offset)[0]
        offset += 4
        
        # Extract bearing angles (in decidegrees)
        bearings = np.array(struct.unpack_from(
            '<' + 'h' * bearing_count, msg_bytes, offset
        ))
        
        # Extract intensity data
        image_data = msg_bytes[image_offset:image_offset + image_size]
        intensity = np.frombuffer(image_data, dtype=np.uint8).reshape(
            (range_count, bearing_count)
        )
        
        # Create timestamp (could be enhanced with actual time from message)
        timestamp = frame_index * 1.0  # Simple frame-based timestamp
        
        return SonarFrame(
            timestamp=timestamp,
            range_resolution=range_res,
            range_count=range_count,
            bearing_count=bearing_count,
            bearings=bearings,
            intensity_data=intensity,
            frame_index=frame_index
        )
    
    def parse_all_frames(self) -> List[SonarFrame]:
        """
        Parse all messages into SonarFrame objects
        
        Returns:
            List of parsed SonarFrame objects
        """
        if not self.messages:
            self.load_file()
        
        frames = []
        for i, msg in enumerate(self.messages):
            try:
                frame = self.parse_frame(msg, i)
                frames.append(frame)
            except Exception as e:
                logger.warning(f"Failed to parse frame {i}: {e}")
        
        self.frames = frames
        logger.info(f"Successfully parsed {len(frames)} frames")
        return frames
    
    def get_frame(self, index: int) -> Optional[SonarFrame]:
        """
        Get a specific frame by index
        
        Args:
            index: Frame index
            
        Returns:
            SonarFrame or None if index is invalid
        """
        if not self.frames:
            self.parse_all_frames()
        
        if 0 <= index < len(self.frames):
            return self.frames[index]
        return None
    
    def to_cartesian(self, frame: SonarFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert sonar polar data to Cartesian coordinates
        
        Args:
            frame: SonarFrame to convert
            
        Returns:
            Tuple of (x_coords, y_coords, intensities) arrays
        """
        # Create meshgrid of range and bearing indices
        range_bins = np.arange(frame.range_count)
        bearing_indices = np.arange(frame.bearing_count)
        
        # Convert to actual values
        ranges_m = range_bins[:, np.newaxis] * frame.range_resolution
        bearings_rad = frame.bearings_rad[np.newaxis, :]
        
        # Convert to Cartesian (sonar coordinate system)
        x = ranges_m * np.sin(bearings_rad)  # Across-track (starboard positive)
        y = ranges_m * np.cos(bearings_rad)  # Along-track (forward positive)
        
        return x, y, frame.intensity_data


if __name__ == "__main__":
    # Test the reader with a sample file
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Use first file from raw_assets as default
        file_path = "raw_assets/Oculus_20250805_164829.oculus"
    
    try:
        reader = OculusFileReader(file_path)
        frames = reader.parse_all_frames()
        
        if frames:
            frame = frames[0]
            print(f"\nFirst frame info:")
            print(f"  Range count: {frame.range_count}")
            print(f"  Bearing count: {frame.bearing_count}")
            print(f"  Range resolution: {frame.range_resolution:.3f} m")
            print(f"  Max range: {frame.max_range:.1f} m")
            print(f"  Bearing range: {frame.bearings_deg.min():.1f}° to {frame.bearings_deg.max():.1f}°")
            print(f"  Intensity range: {frame.intensity_data.min()} to {frame.intensity_data.max()}")
    except Exception as e:
        print(f"Error: {e}")
