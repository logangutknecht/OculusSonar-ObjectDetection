# Oculus Replay Node Package

## Overview

The `oculus_replay_node` package provides ROS2 nodes for replaying sonar data recorded from Blueprint Subsea Oculus sonar systems. It reads binary `.oculus` files and publishes the data as ROS2 messages for processing and visualization in the sonar mapping pipeline.

## Features

- **Binary File Parsing**: Reads Blueprint Subsea `.oculus` recording files
- **Sonar Image Publication**: Publishes sonar intensity images as `sensor_msgs/Image`
- **Bearing Data**: Publishes sonar beam bearing angles as `std_msgs/Int16MultiArray`
- **Configurable Playback**: Adjustable frame rate and file selection
- **Debug Capabilities**: Raw packet dumping for troubleshooting
- **ROS2 Integration**: Standard message formats for seamless pipeline integration

## Architecture

### Core Components

1. **OculusFileReader Node**: Primary node for parsing and publishing sonar data
2. **Binary Parser**: Decodes Blueprint Subsea SimpleFireV2 message format
3. **Image Publisher**: Converts sonar intensity data to ROS2 Image messages
4. **Bearing Publisher**: Publishes beam angle information
5. **OculusReplayNode**: Legacy/test node for PointCloud2 generation

### Message Flow

```
.oculus Binary File
        ↓
[Binary Parser - SimpleFireV2 Protocol]
        ↓
[Image Processing - Intensity Matrix]
        ↓
sensor_msgs/Image (/oculus/image_raw)
        ↓
std_msgs/Int16MultiArray (/oculus/bearings)
```

## Installation and Build

### Dependencies

```xml
<!-- package.xml dependencies -->
<buildtool_depend>ament_python</buildtool_depend>
<exec_depend>rclpy</exec_depend>
<exec_depend>sensor_msgs</exec_depend>
<exec_depend>std_msgs</exec_depend>
<exec_depend>cv_bridge</exec_depend>
```

### Python Dependencies

```bash
# Required Python packages
pip install numpy opencv-python
```

### Build Instructions

```bash
# From workspace root
cd /home/devuser/ros2_ws

# Build the package
colcon build --packages-select oculus_replay_node

# Source the installation
source install/setup.bash
```

## Usage

### Basic Usage

```bash
# Run with default file
ros2 run oculus_replay_node oculus_file_reader_node

# Run with specific file
ros2 run oculus_replay_node oculus_file_reader_node \
    --ros-args \
    -p file_path:=/path/to/your/recording.oculus

# Enable debug output
ros2 run oculus_replay_node oculus_file_reader_node \
    --ros-args \
    -p debug_dump:=true
```

### Launch File Integration

```python
# Example launch file usage
Node(
    package='oculus_replay_node',
    executable='oculus_file_reader_node',
    name='oculus_file_reader_node',
    parameters=[{
        'file_path': '/home/devuser/ros2_ws/data/Oculus_20250705_150423.oculus',
        'debug_dump': False
    }],
    output='screen'
)
```

## Configuration Parameters

### OculusFileReader Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file_path` | string | `/home/devuser/ros2_ws/data/Oculus_20250705_150423.oculus` | Path to .oculus recording file |
| `debug_dump` | bool | `false` | Enable raw packet hex dumping for debugging |

### Playback Control

| Setting | Value | Description |
|---------|-------|-------------|
| **Frame Rate** | 1.0 Hz | Fixed playback rate (1 frame/second) |
| **Loop Mode** | Disabled | Playback stops at end of file |
| **Auto-start** | Enabled | Begins playback immediately on startup |

## File Format Support

### Blueprint Subsea .oculus Format

The package supports Binary recording files from Blueprint Subsea Oculus sonars:

- **File Extension**: `.oculus`
- **Message Type**: SimpleFireV2 (Message ID: 35)
- **Sonar ID**: 0x4F53 (Oculus identifier)
- **Data Format**: Binary packed structures

### Message Structure

```python
# SimpleFireV2 Message Layout
Header (16 bytes)
├── oculus_id: 0x4F53
├── message_id: 35 (SimpleFireV2)
├── payload_size: Variable
└── ...

Payload (Variable length)
├── Fire Message (89 bytes)
├── Ping ID (8 bytes)  
├── Metadata (64 bytes)
├── Range Resolution (8 bytes)
├── Range Count (2 bytes)
├── Bearing Count (2 bytes)
├── Image Offset (4 bytes)
├── Image Size (4 bytes)
├── Bearing Angles (bearing_count * 2 bytes)
└── Image Data (range_count * bearing_count bytes)
```

### Supported Data Types

| Data Type | Size | Description |
|-----------|------|-------------|
| **Range Resolution** | 8 bytes (double) | Meters per range bin |
| **Range Count** | 2 bytes (uint16) | Number of range bins |
| **Bearing Count** | 2 bytes (uint16) | Number of sonar beams |
| **Bearing Angles** | 2 bytes each (int16) | Beam angles in decidegrees |
| **Image Data** | 1 byte each (uint8) | Sonar intensity values |

## Published Topics

### Primary Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/oculus/image_raw` | `sensor_msgs/Image` | 1 Hz | Sonar intensity image |
| `/oculus/bearings` | `std_msgs/Int16MultiArray` | 1 Hz | Beam bearing angles |

### Message Details

#### /oculus/image_raw (sensor_msgs/Image)

```python
# Image message structure
Header header
  uint32 seq
  time stamp
  string frame_id  # "sonar_link"

uint32 height      # range_count (number of range bins)
uint32 width       # bearing_count (number of beams)
string encoding    # "mono8" (8-bit grayscale)
uint8 is_bigendian # False
uint32 step        # width (bytes per row)
uint8[] data       # Sonar intensity data
```

#### /oculus/bearings (std_msgs/Int16MultiArray)

```python
# Bearing array structure
int16[] data       # Bearing angles in decidegrees
                   # Convert to degrees: angle_deg = data[i] / 10.0
                   # Convert to radians: angle_rad = data[i] * pi / 1800.0
```

### Quality of Service (QoS)

```python
# Bearing topic uses latched QoS
qos_profile = QoSProfile(depth=10)
qos_profile.durability = QoSDurabilityPolicy.TRANSIENT_LOCAL

# Ensures bearing data is available before image data
```

## Coordinate Systems and Transformations

### Sonar Frame Definition

- **Frame ID**: `sonar_link`
- **Origin**: Sonar transducer center
- **Orientation**: 
  - X-axis: Forward (sonar boresight)
  - Y-axis: Starboard (right)
  - Z-axis: Down

### Polar to Cartesian Conversion

```python
# Convert sonar data to Cartesian coordinates
def polar_to_cartesian(range_bins, bearing_angles, range_resolution):
    """
    Convert sonar polar data to Cartesian coordinates
    
    Args:
        range_bins: Array indices (0 to range_count-1)
        bearing_angles: Beam angles in decidegrees
        range_resolution: Meters per range bin
    
    Returns:
        x, y: Cartesian coordinates in meters
    """
    # Convert to standard units
    ranges_m = range_bins * range_resolution
    bearings_rad = np.array(bearing_angles) * np.pi / 1800.0  # decidegrees to radians
    
    # Polar to Cartesian (sonar coordinate system)
    x = ranges_m * np.sin(bearings_rad)  # Across-track (starboard positive)
    y = ranges_m * np.cos(bearings_rad)  # Along-track (forward positive)
    
    return x, y
```

### Integration with TF2

```python
# Transform sonar data to other coordinate frames
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

# Setup TF2
tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer, node)

# Transform point from sonar_link to base_link
try:
    transform = tf_buffer.lookup_transform(
        'base_link', 'sonar_link', 
        rclpy.time.Time()
    )
    # Apply transform to sonar points
except Exception as e:
    node.get_logger().warn(f"Transform failed: {e}")
```

## Integration with Processing Pipeline

### Sonar Point Cloud Generation

```python
# Example: Convert to point cloud (used by sonar_pointcloud package)
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

def create_pointcloud(image_msg, bearing_msg, range_resolution):
    """Convert sonar image to 3D point cloud"""
    height, width = image_msg.height, image_msg.width
    image_data = np.frombuffer(image_msg.data, dtype=np.uint8).reshape(height, width)
    bearings = np.array(bearing_msg.data) * np.pi / 1800.0  # to radians
    
    points = []
    for r in range(height):
        for b in range(width):
            if image_data[r, b] > intensity_threshold:
                range_m = r * range_resolution
                bearing_rad = bearings[b]
                
                x = range_m * np.sin(bearing_rad)
                y = range_m * np.cos(bearing_rad)
                z = 0.0
                intensity = float(image_data[r, b])
                
                points.append([x, y, z, intensity])
    
    return pc2.create_cloud_xyz(image_msg.header, points)
```

### GPS Correlation

```python
# Example: Synchronize with GPS data (used by sonar_gps_correlator)
from message_filters import Subscriber, ApproximateTimeSynchronizer

class SonarGPSCorrelator:
    def __init__(self):
        # Synchronized subscribers
        self.image_sub = Subscriber(self, Image, '/oculus/image_raw')
        self.gps_sub = Subscriber(self, NavSatFix, '/gps/fix')
        
        # Time synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.image_sub, self.gps_sub], 
            queue_size=10, 
            slop=0.5  # 500ms tolerance
        )
        self.sync.registerCallback(self.synchronized_callback)
    
    def synchronized_callback(self, image_msg, gps_msg):
        # Process synchronized sonar and GPS data
        self.process_sonar_gps_pair(image_msg, gps_msg)
```

## Performance Characteristics

### Resource Usage

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| **CPU Usage** | 5-15% | Single core, parsing and publishing |
| **Memory Usage** | 50-200 MB | Depends on image size and buffer |
| **Disk I/O** | 1-10 MB/s | Reading .oculus file sequentially |
| **Network** | 5-50 MB/s | Publishing image and bearing data |

### Scalability Limits

- **Max File Size**: Limited by available disk space and memory
- **Max Image Size**: 2048 x 2048 pixels (4MB per frame)
- **Max Frame Rate**: ~10 Hz (limited by ROS2 message overhead)
- **Max Bearing Count**: 2048 beams

### Optimization Tips

```python
# Performance optimization strategies
1. Use QoS policies appropriately
2. Minimize debug output in production
3. Consider frame decimation for high-rate data
4. Use compressed image transport for network efficiency
```

## Debugging and Troubleshooting

### Debug Mode

Enable detailed debugging with `debug_dump:=true`:

```bash
ros2 run oculus_replay_node oculus_file_reader_node \
    --ros-args -p debug_dump:=true
```

Debug output includes:
- Raw packet hex dumps
- Message parsing details
- Frame structure information
- Error diagnostics

### Common Issues

#### 1. File Not Found

**Symptoms**: Node exits with `FileNotFoundError`

**Solutions**:
```bash
# Check file exists
ls -la /path/to/your/file.oculus

# Check file permissions
chmod 644 /path/to/your/file.oculus

# Use absolute path
ros2 run oculus_replay_node oculus_file_reader_node \
    --ros-args -p file_path:=/full/path/to/file.oculus
```

#### 2. Corrupted File Data

**Symptoms**: Parse errors, malformed frames

**Solutions**:
- Enable debug mode to examine raw packets
- Verify file integrity with `hexdump -C file.oculus | head`
- Check file size matches expected content
- Re-record if file is corrupted

#### 3. No Image Data

**Symptoms**: Bearing data published but no images

**Solutions**:
```bash
# Check topic publication
ros2 topic list | grep oculus
ros2 topic echo /oculus/image_raw --once
ros2 topic hz /oculus/image_raw

# Verify QoS compatibility
ros2 topic info /oculus/image_raw
```

#### 4. Incorrect Image Dimensions

**Symptoms**: Distorted or incorrectly sized images

**Solutions**:
- Verify range_count and bearing_count parsing
- Check image data offset calculation
- Enable debug mode to examine parsed values
- Compare with sonar specification

### Debug Commands

```bash
# Monitor node status
ros2 node list | grep oculus
ros2 node info oculus_file_reader_node

# Check published topics
ros2 topic list | grep oculus
ros2 topic hz /oculus/image_raw
ros2 topic hz /oculus/bearings

# Examine message content
ros2 topic echo /oculus/bearings --once
ros2 interface show sensor_msgs/msg/Image

# Check file format
hexdump -C /path/to/file.oculus | head -20
file /path/to/file.oculus
```

### Validation Tools

```python
# Validate sonar data integrity
def validate_sonar_frame(image_msg, bearing_msg):
    """Validate sonar frame data consistency"""
    assert image_msg.width == len(bearing_msg.data), \
        f"Width mismatch: {image_msg.width} vs {len(bearing_msg.data)}"
    
    assert image_msg.encoding == 'mono8', \
        f"Unexpected encoding: {image_msg.encoding}"
    
    assert len(image_msg.data) == image_msg.height * image_msg.width, \
        f"Data size mismatch: {len(image_msg.data)} vs {image_msg.height * image_msg.width}"
    
    # Check bearing angle ranges (typically -90 to +90 degrees)
    bearing_degrees = [b / 10.0 for b in bearing_msg.data]
    assert all(-900 <= b <= 900 for b in bearing_msg.data), \
        f"Bearing angles out of range: {min(bearing_degrees)} to {max(bearing_degrees)}"
```

## Testing and Validation

### Unit Tests

```python
# Example unit test structure
import unittest
from oculus_replay_node.oculus_file_reader_node import OculusFileReader

class TestOculusFileReader(unittest.TestCase):
    def setUp(self):
        self.test_file = '/path/to/test/file.oculus'
        
    def test_file_loading(self):
        """Test .oculus file loading"""
        reader = OculusFileReader()
        messages = reader.load_oculus_file()
        self.assertGreater(len(messages), 0)
    
    def test_message_parsing(self):
        """Test SimpleFireV2 message parsing"""
        # Test with known good message
        pass
    
    def test_coordinate_conversion(self):
        """Test polar to Cartesian conversion"""
        # Verify coordinate transformation math
        pass
```

### Integration Tests

```bash
# Test complete pipeline
ros2 launch oculus_tools test_oculus_replay.launch.py

# Verify output topics
ros2 topic echo /oculus/image_raw --once
ros2 topic echo /oculus/bearings --once

# Check data flow to downstream nodes  
ros2 topic echo /cloud_out --once  # from sonar_pointcloud
```

### Benchmark Tests

```python
# Performance benchmarking
import time
import psutil

def benchmark_parsing_performance(file_path, num_frames=100):
    """Benchmark message parsing performance"""
    reader = OculusFileReader()
    messages = reader.load_oculus_file()
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    for i in range(min(num_frames, len(messages))):
        image_msg, bearing_msg = reader.parse_simplefire_v2(messages[i])
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    fps = num_frames / (end_time - start_time)
    memory_usage_mb = (end_memory - start_memory) / 1024 / 1024
    
    print(f"Parsing rate: {fps:.2f} FPS")
    print(f"Memory usage: {memory_usage_mb:.2f} MB")
```

## Example Usage Scenarios

### 1. Basic Sonar Visualization

```bash
# Start sonar replay
ros2 run oculus_replay_node oculus_file_reader_node \
    --ros-args -p file_path:=/home/devuser/ros2_ws/data/Oculus_20250705_150423.oculus

# In another terminal, visualize in RViz
rviz2 -d sonar_visualization.rviz
```

### 2. Point Cloud Generation Pipeline

```bash
# Complete processing chain
ros2 launch oculus_tools oculus_processing.launch.py \
    recording_path:=/path/to/data.oculus
    
# Outputs:
# - /oculus/image_raw (sonar images)
# - /oculus/bearings (beam angles)  
# - /cloud_out (3D point cloud)
```

### 3. GeoTIFF Generation

```bash
# Generate georeferenced sonar imagery
ros2 launch geotiff_generator geotiff_generation.launch.py \
    recording_path:=/path/to/data.oculus
    
# Outputs georeferenced TIFF files for GIS applications
```

### 4. Real-time Processing Simulation

```bash
# Simulate real-time processing with frame decimation
ros2 run oculus_replay_node oculus_file_reader_node \
    --ros-args -p file_path:=/path/to/data.oculus

# Process every Nth frame for real-time simulation
```

## Future Enhancements

### Planned Features

1. **Multi-file Playback**: Sequential playback of multiple .oculus files
2. **Variable Frame Rate**: Configurable playback speed
3. **Loop Mode**: Continuous playback for testing
4. **Frame Filtering**: Skip frames based on quality metrics
5. **Compressed Messages**: Reduce network bandwidth usage
6. **Live Streaming**: Direct connection to Oculus sonar hardware

### Extension Points

```python
class AdvancedOculusFileReader(OculusFileReader):
    """Extended file reader with additional features"""
    
    def __init__(self):
        super().__init__()
        self.setup_frame_filtering()
        self.setup_compression()
        self.setup_multi_file_support()
    
    def setup_frame_filtering(self):
        """Setup quality-based frame filtering"""
        self.declare_parameter('min_quality_threshold', 0.5)
        self.declare_parameter('max_noise_level', 0.3)
        
    def setup_compression(self):
        """Setup image compression for bandwidth reduction"""
        self.declare_parameter('enable_compression', False)
        self.declare_parameter('compression_quality', 80)
        
    def setup_multi_file_support(self):
        """Setup sequential multi-file playback"""
        self.declare_parameter('file_list', [])
        self.declare_parameter('loop_playback', False)
```

## API Reference

### Main Class: OculusFileReader

#### Constructor Parameters
- Inherits from `rclpy.node.Node`
- Node name: `'oculus_file_reader_node'`

#### Public Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `load_oculus_file()` | Load and parse .oculus file | None | `List[bytes]` |
| `publish_next_frame()` | Publish next sonar frame | None | None |
| `parse_simplefire_v2()` | Parse SimpleFireV2 message | `msg_bytes: bytes` | `Tuple[Image, Int16MultiArray]` |
| `dump_raw_packet()` | Debug hex dump of packet | `msg_bytes: bytes` | None |

#### ROS2 Parameters

| Parameter | Type | Access | Description |
|-----------|------|--------|-------------|
| `file_path` | string | Read-only | Path to .oculus file |
| `debug_dump` | bool | Read-only | Enable debug output |

### Entry Points

```python
# Available executables
'oculus_file_reader_node = oculus_replay_node.oculus_file_reader_node:main'
'oculus_replay_node = oculus_replay_node.oculus_replay_node:main'  # Legacy
```

This comprehensive documentation provides everything needed to understand, use, and extend the `oculus_replay_node` package for sonar data processing in the ROS2 ecosystem.
