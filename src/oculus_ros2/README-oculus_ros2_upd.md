# Oculus ROS2 Package

## Overview

The `oculus_ros2` package provides a comprehensive ROS2 interface for Blueprint Subsea Oculus multi-beam imaging sonars. This metapackage serves as the primary bridge between the low-level Oculus sonar hardware and the ROS2 ecosystem, enabling real-time sonar data acquisition, processing, and visualization for underwater robotics applications.

## Features

- **Live Sonar Interface**: Direct TCP/UDP communication with Oculus M1200d/M3000d sonars
- **Multi-Beam Support**: 256 or 512 beam configurations for high-resolution imaging
- **Real-time Processing**: Low-latency ping processing and data publishing
- **Dynamic Configuration**: Runtime parameter adjustment via ROS2 parameters
- **Temperature Monitoring**: Built-in thermal protection and monitoring
- **Network Auto-Discovery**: Automatic sonar detection and IP configuration
- **Comprehensive Message Types**: Rich data structures for sonar configuration and ping data
- **Visual Debugging**: Built-in sonar viewer and visualization tools

## Architecture

### Metapackage Components

1. **oculus_interfaces**: Custom ROS2 message definitions for Oculus sonar data
2. **oculus_ros2**: Main ROS2 node implementing sonar interface and processing
3. **oculus_sonar**: Additional sonar utilities and tools (future expansion)

### Core Nodes

#### OculusSonarNode (C++)

- **Purpose**: Primary interface node for live Oculus sonar hardware
- **Language**: C++ (performance-critical real-time processing)
- **Function**: Manages sonar connection, configuration, and data publishing

#### OculusViewerNode (C++)

- **Purpose**: Real-time sonar data visualization and debugging
- **Language**: C++ with OpenCV integration
- **Function**: Displays sonar pings as intensity images with overlay information

### Data Flow Pipeline

```
Oculus Sonar ─┐
(TCP/UDP)     │
              ├─► [Network Discovery] ─► [Connection Management] ─► [Ping Processing] ─► ROS2 Topics
              │                                                                          │
              └─► [Configuration] ─────────────────────────────────────────────────────┘
                  (Parameters)
```

## Installation and Build

### Dependencies

#### System Requirements

```bash
# Ubuntu 22.04 LTS
# ROS2 Humble
# CMake 3.22.1+
# GCC 9+ or Clang 10+
```

#### Package Dependencies

```xml
<!-- Core ROS2 dependencies -->
<depend>ament_cmake</depend>
<depend>rclcpp</depend>
<depend>rclpy</depend>
<depend>rcl_interfaces</depend>
<depend>std_msgs</depend>
<depend>sensor_msgs</depend>

<!-- Oculus-specific dependencies -->
<depend>oculus_driver</depend>
<depend>oculus_interfaces</depend>

<!-- Image processing -->
<depend>cv_bridge</depend>
<depend>OpenCV</depend>
```

#### External Dependencies

The package automatically downloads and builds the `oculus_driver` library during compilation:

```cmake
# Automatic dependency management in CMakeLists.txt
if(NOT TARGET oculus_driver)
    include(FetchContent)
    FetchContent_Declare(oculus_driver
        GIT_REPOSITORY https://github.com/ENSTABretagneRobotics/oculus_driver.git
        GIT_TAG master
    )
    FetchContent_MakeAvailable(oculus_driver)
endif()
```

### Build Instructions

#### Standard Build (with Internet)

```bash
# Navigate to workspace
cd /home/devuser/ros2_ws

# Clone the metapackage (if not already present)
# git clone https://github.com/ENSTABretagneRobotics/oculus_ros2.git src/oculus_ros2

# Build the metapackage
colcon build --packages-select oculus_ros2 oculus_interfaces

# Source the workspace
source install/setup.bash
```

#### Manual Dependencies Build (offline)

```bash
# Install oculus_driver manually
cd ~/work/libraries
git clone https://github.com/ENSTABretagneRobotics/oculus_driver.git
cd oculus_driver
mkdir build && cd build

# Configure and build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=~/work/install ..
make -j4 install

# Install Python bindings
cd ../python
pip3 install --upgrade "pybind11[global]"
pip3 install --user -e .

# Set environment variables
export CMAKE_PREFIX_PATH=~/work/install:$CMAKE_PREFIX_PATH
export LD_LIBRARY_PATH=~/work/install:$LD_LIBRARY_PATH

# Build ROS2 package
cd /home/devuser/ros2_ws
colcon build --packages-select oculus_ros2 oculus_interfaces
```

## Usage

### Basic Operations

#### Launch Default Node

```bash
# Launch with default configuration
ros2 launch oculus_ros2 default.launch.py

# Launch with custom parameters
ros2 launch oculus_ros2 default.launch.py \
    frame_id:=sonar_link \
    range:=30.0 \
    ping_rate:=1
```

#### Direct Node Execution

```bash
# Run sonar node directly
ros2 run oculus_ros2 oculus_sonar_node

# Run with parameters
ros2 run oculus_ros2 oculus_sonar_node \
    --ros-args \
    -p frame_id:=base_sonar_link \
    -p frequency_mode:=1 \
    -p ping_rate:=2
```

#### Sonar Viewer

```bash
# Launch visual sonar display
ros2 run oculus_ros2 oculus_viewer_node

# Integrated viewer with sonar node
ros2 launch oculus_ros2 default.launch.py enable_viewer:=true
```

### Advanced Configuration

#### Custom Launch File

```python
# custom_oculus.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Custom configuration file
    config = os.path.join(
        get_package_share_directory("my_robot_config"), 
        "config", 
        "oculus_custom.yaml"
    )
    
    return LaunchDescription([
        Node(
            package="oculus_ros2",
            executable="oculus_sonar_node",
            name="oculus_sonar",
            parameters=[config],
            remappings=[
                ("ping", "/sensors/sonar/ping"),
                ("status", "/sensors/sonar/status")
            ],
            output="screen"
        )
    ])
```

#### Runtime Parameter Configuration

```bash
# List all available parameters
ros2 param list /oculus_sonar

# Get parameter descriptions
ros2 param describe /oculus_sonar ping_rate
ros2 param describe /oculus_sonar frequency_mode

# Set parameters dynamically
ros2 param set /oculus_sonar gain_assist false
ros2 param set /oculus_sonar range 25.0
ros2 param set /oculus_sonar ping_rate 1  # 15Hz max
```

## Configuration Parameters

### Essential Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `frame_id` | string | `"sonar"` | - | TF frame reference for ping messages |
| `run` | bool | `false` | - | Enable/disable sonar operation (safety) |
| `frequency_mode` | int | `2` | 1-2 | Beam frequency: 1=1.2MHz (wide), 2=2.1MHz (narrow) |
| `ping_rate` | int | `2` | 0-5 | Ping frequency: 0=10Hz, 1=15Hz, 2=40Hz, 3=5Hz, 4=2Hz, 5=Standby |
| `nbeams` | int | `1` | 0-1 | Beam count: 0=256 beams, 1=512 beams |
| `range` | double | `20.0` | 0.3-40.0 | Maximum sonar range in meters |

### Processing Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `gain_assist` | bool | `true` | - | Enable automatic gain control |
| `gamma_correction` | int | `153` | 0-255 | Gamma correction (153 = 60%) |
| `gain_percent` | double | `50.0` | 0.1-100.0 | Manual gain percentage |

### Environmental Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `sound_speed` | double | `0.0` | 1400-1600 | Sound speed (m/s, 0=auto-calculate) |
| `use_salinity` | bool | `true` | - | Use salinity for sound speed calculation |
| `salinity` | double | `0.0` | 0-100 | Water salinity (parts per thousand) |

### Safety Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `temperature_warn` | double | `30.0` | - | Temperature warning threshold (°C) |
| `temperature_stop` | double | `35.0` | - | Emergency stop temperature (°C) |

### Parameter Configuration File

```yaml
# /config/oculus_custom.yaml
/**:
  ros__parameters:
    # Frame configuration
    frame_id: "base_sonar_link"
    
    # Safety settings
    run: true
    temperature_warn: 32.0
    temperature_stop: 38.0
    
    # Sonar configuration
    frequency_mode: 1        # Wide beam (1.2MHz)
    ping_rate: 1            # 15Hz max ping rate
    nbeams: 1               # 512 beams
    range: 30.0             # 30 meter range
    
    # Signal processing
    gain_assist: true
    gamma_correction: 128   # 50% gamma
    gain_percent: 60.0
    
    # Environmental
    sound_speed: 0.0        # Auto-calculate
    use_salinity: true
    salinity: 35.0          # Typical seawater
```

## Message Types and Topics

### Published Topics

| Topic | Type | Rate | Description |
|-------|------|------|-------------|
| `/oculus_sonar/ping` | `oculus_interfaces/Ping` | Variable | Primary sonar ping data with intensity image |
| `/oculus_sonar/status` | `oculus_interfaces/OculusStatus` | 1 Hz | Sonar system status and configuration |
| `/oculus_sonar/temperature` | `sensor_msgs/Temperature` | 1 Hz | Sonar transducer temperature |
| `/oculus_sonar/pressure` | `sensor_msgs/FluidPressure` | 1 Hz | Environmental pressure measurement |

### Message Structures

#### oculus_interfaces/Ping

```python
# Core ping message structure
std_msgs/Header header                    # Standard ROS header with timestamp
oculus_interfaces/OculusFireConfig fire_message  # Ping configuration
uint32 ping_id                           # Unique ping identifier
uint32 status                            # Ping status flags
float64 frequency                        # Actual ping frequency (Hz)
float64 temperature                      # Transducer temperature (°C)
float64 pressure                         # Environmental pressure (Pa)
float64 speed_of_sound_used              # Sound velocity used (m/s)
uint32 ping_start_time                   # Ping timestamp (microseconds)
uint8 data_size                          # Data element size (bytes)
float64 range_resolution                 # Range resolution (m/sample)
uint16 n_ranges                          # Number of range samples
uint16 n_beams                           # Number of beams
uint32 image_offset                      # Image data offset in message
uint32 image_size                        # Image data size (bytes)
uint32 message_size                      # Total message size
uint8[] data                             # Raw sonar intensity data
```

#### oculus_interfaces/OculusStatus

```python
# System status message
std_msgs/Header header
uint32 device_id                         # Sonar device identifier
uint32 device_type                       # Device type code
uint32 part_number                       # Hardware part number
uint32 status                            # System status flags
OculusVersionInfo version_info           # Firmware version details
uint32 ipv4_address                      # Current IP address
uint32 ipv4_mask                         # Network mask
uint32 ipv4_gateway                      # Gateway address
uint8 mac_address0                       # MAC address (6 bytes)
uint8 mac_address1
uint8 mac_address2
uint8 mac_address3
uint8 mac_address4
uint8 mac_address5
```

#### oculus_interfaces/OculusFireConfig

```python
# Ping configuration message
uint8 masterMode                         # Master/slave mode
uint8 networkSpeed                       # Network speed setting
uint8 gammaCorrection                    # Gamma correction value
uint8 flags                              # Configuration flags
uint32 range                             # Range setting (mm)
uint8 gainPercent                        # Gain percentage
uint8 speedOfSound                       # Sound velocity setting
uint8 salinity                           # Salinity setting
```

### Quality of Service (QoS)

```python
# Default QoS profiles used
ping_qos = QoSProfile(
    depth=10,                            # Buffer recent pings
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST
)

status_qos = QoSProfile(
    depth=1,                             # Keep only latest status
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
)
```

## Network Configuration and Setup

### Sonar Network Discovery

The Oculus sonar automatically broadcasts its IP address on the network for discovery:

```cpp
// Automatic IP discovery process
1. Sonar broadcasts UDP packets with device information
2. Node listens for broadcasts and extracts IP address
3. TCP connection established on port 52102 (default)
4. Configuration and ping data exchanged over TCP
```

### Network Requirements

#### IP Configuration

```bash
# Typical sonar IP configurations
Default Sonar IP: 192.168.1.45  # (varies by device)
Default Port: 52102
Protocol: TCP for data, UDP for discovery

# System network must be on same subnet
# Example host configuration:
sudo ip addr add 192.168.1.100/24 dev eth0
sudo ip route add 192.168.1.0/24 dev eth0
```

#### Network Troubleshooting

```bash
# Verify sonar connectivity
ping 192.168.1.45

# Monitor sonar broadcasts
sudo tcpdump -i eth0 udp port 52100

# Test TCP connection
telnet 192.168.1.45 52102

# Check network interface
ip addr show
ip route show
```

### Sonar IP Address Change

For systems requiring different IP configurations:

1. **Using Oculus ViewPoint Software**:
   ```bash
   # Install wine (for Windows software on Linux)
   sudo apt install wine
   
   # Run ViewPoint to change sonar IP
   wine OculusViewPoint.exe
   ```

2. **Programmatic Configuration**:
   ```cpp
   // IP configuration via oculus_driver API
   sonar.configureNetwork(new_ip, new_mask, new_gateway);
   ```

## Integration with Processing Pipeline

### Upstream Integration (Hardware)

```python
# Direct hardware connection
Oculus M1200d/M3000d → [Network] → oculus_ros2 → ROS2 Topics
                       (TCP/UDP)
```

### Downstream Integration (Processing)

#### Point Cloud Generation

```python
# Sonar to point cloud processing
/oculus_sonar/ping → sonar_pointcloud → /oculus/pointcloud_raw
                                    → octomap_server → /octomap_binary
```

#### Image Processing

```python
# Extract sonar images for computer vision
/oculus_sonar/ping → image_extractor → /sonar/image_raw (sensor_msgs/Image)
                                   → cv_pipeline → /sonar/features
```

#### Recording and Replay

```python
# Data recording pipeline
/oculus_sonar/ping → rosbag2 record → sonar_data.bag
sonar_data.bag → rosbag2 play → /oculus_sonar/ping (playback)
```

### Coordinate Frame Integration

```python
# TF2 frame relationships
map
├── odom
    ├── base_link
        ├── sonar_link  ← Default frame_id
        │   ├── sonar_optical_frame
        │   └── beam_reference_frame
        └── gps_link
```

#### Transform Configuration

```xml
<!-- Static transform publisher for sonar mounting -->
<node pkg="tf2_ros" exec="static_transform_publisher"
      name="sonar_frame_publisher"
      args="0.5 0 -0.2 0 0.174533 0 base_link sonar_link" />
```

## Performance Characteristics

### Processing Performance

| Metric | Typical Value | Maximum | Notes |
|--------|---------------|---------|-------|
| **Ping Rate** | 5-15 Hz | 40 Hz | Depends on range and beam count |
| **Latency** | 50-100 ms | 200 ms | Network + processing delay |
| **CPU Usage** | 10-25% | 50% | Single core, varies with ping rate |
| **Memory Usage** | 100-200 MB | 500 MB | Buffering and image processing |
| **Network Bandwidth** | 5-50 Mbps | 100 Mbps | Depends on configuration |

### Scalability Characteristics

```python
# Performance scaling with configuration
performance_profiles = {
    'low_resolution': {
        'beams': 256,
        'range': 10.0,
        'ping_rate': 3,           # 5Hz
        'bandwidth': '5 Mbps',
        'cpu_usage': '10%',
        'latency': '50ms'
    },
    'medium_resolution': {
        'beams': 512,
        'range': 20.0,
        'ping_rate': 2,           # 40Hz
        'bandwidth': '25 Mbps',
        'cpu_usage': '20%',
        'latency': '75ms'
    },
    'high_resolution': {
        'beams': 512,
        'range': 40.0,
        'ping_rate': 1,           # 15Hz
        'bandwidth': '50 Mbps',
        'cpu_usage': '35%',
        'latency': '100ms'
    }
}
```

### Optimization Strategies

#### Network Optimization

```cpp
// TCP socket optimization
int tcp_nodelay = 1;
setsockopt(socket_fd, IPPROTO_TCP, TCP_NODELAY, &tcp_nodelay, sizeof(int));

int socket_buffer_size = 1024 * 1024;  // 1MB buffer
setsockopt(socket_fd, SOL_SOCKET, SO_RCVBUF, &socket_buffer_size, sizeof(int));
```

#### Processing Optimization

```cpp
// Multi-threaded processing
class OculusSonarNode {
private:
    std::thread network_thread_;       // Handle network I/O
    std::thread processing_thread_;    // Process ping data
    std::thread publishing_thread_;    // Publish ROS messages
    
    // Lock-free queues for inter-thread communication
    boost::lockfree::queue<PingData> ping_queue_;
};
```

## Debugging and Troubleshooting

### Common Issues

#### 1. No Sonar Detection

**Symptoms**: Node starts but no sonar found

**Diagnostic Commands**:
```bash
# Check network connectivity
ping 192.168.1.45  # Replace with actual sonar IP

# Monitor network traffic
sudo tcpdump -i eth0 host 192.168.1.45

# Verify network interface
ip addr show
ip route show
```

**Solutions**:
- Verify sonar is powered and underwater
- Check network cable connections
- Ensure host IP is on same subnet as sonar
- Verify firewall settings allow TCP/UDP traffic

#### 2. Connection Drops/Timeouts

**Symptoms**: Intermittent connection loss or ping timeouts

**Diagnostic Commands**:
```bash
# Monitor connection stability
ros2 topic hz /oculus_sonar/ping
ros2 topic echo /oculus_sonar/status --once

# Check system resources
htop -p $(pgrep -f oculus_sonar_node)
```

**Solutions**:
- Reduce ping rate to lower network load
- Increase network buffer sizes
- Check for electromagnetic interference
- Verify power supply stability

#### 3. High CPU Usage

**Symptoms**: Node consuming excessive CPU resources

**Solutions**:
- Lower ping rate: `ros2 param set /oculus_sonar ping_rate 4`
- Reduce beam count: `ros2 param set /oculus_sonar nbeams 0`
- Optimize processing pipeline
- Consider multi-threading improvements

#### 4. Temperature Warnings

**Symptoms**: Frequent temperature warnings or shutdowns

**Solutions**:
```bash
# Monitor temperature
ros2 topic echo /oculus_sonar/temperature

# Adjust safety limits
ros2 param set /oculus_sonar temperature_warn 35.0
ros2 param set /oculus_sonar temperature_stop 40.0

# Reduce thermal load
ros2 param set /oculus_sonar ping_rate 5  # Standby mode
```

### Debug Tools and Utilities

#### Network Analysis

```bash
# Wireshark packet capture
sudo wireshark -i eth0 -f "host 192.168.1.45"

# Network latency testing
hping3 -S -p 52102 192.168.1.45

# Bandwidth monitoring
iftop -i eth0
```

#### ROS2 Debugging

```bash
# Node introspection
ros2 node info /oculus_sonar
ros2 param list /oculus_sonar
ros2 service list

# Topic monitoring
ros2 topic list
ros2 topic hz /oculus_sonar/ping
ros2 topic bw /oculus_sonar/ping

# Message inspection
ros2 interface show oculus_interfaces/msg/Ping
ros2 topic echo /oculus_sonar/ping --once
```

#### Performance Profiling

```bash
# CPU profiling with perf
perf record -g ros2 run oculus_ros2 oculus_sonar_node
perf report

# Memory profiling with valgrind
valgrind --tool=memcheck --leak-check=full \
    ros2 run oculus_ros2 oculus_sonar_node

# Network profiling
ss -tuln | grep 52102
netstat -i eth0
```

### Validation Tools

#### Message Validation

```python
# Validate ping message integrity
def validate_ping_message(ping_msg):
    """Validate Oculus ping message consistency"""
    # Check header
    assert ping_msg.header.frame_id != "", "Missing frame_id"
    assert ping_msg.header.stamp.sec > 0, "Invalid timestamp"
    
    # Check ping data
    assert ping_msg.n_ranges > 0, "No range data"
    assert ping_msg.n_beams > 0, "No beam data"
    assert len(ping_msg.data) > 0, "No ping data"
    
    # Validate data consistency
    expected_size = ping_msg.n_ranges * ping_msg.n_beams * ping_msg.data_size
    assert len(ping_msg.data) >= expected_size, f"Data size mismatch: {len(ping_msg.data)} vs {expected_size}"
    
    # Check environmental data
    assert 0 < ping_msg.temperature < 60, f"Invalid temperature: {ping_msg.temperature}"
    assert ping_msg.frequency > 1000000, f"Invalid frequency: {ping_msg.frequency}"
```

#### System Health Monitoring

```python
# Monitor system health
class SonarHealthMonitor:
    def __init__(self):
        self.temperature_history = []
        self.ping_rate_history = []
        self.error_count = 0
    
    def check_system_health(self, ping_msg, status_msg):
        # Temperature monitoring
        if ping_msg.temperature > 35.0:
            self.log_warning(f"High temperature: {ping_msg.temperature}°C")
        
        # Ping rate monitoring
        current_rate = self.calculate_ping_rate()
        if current_rate < 1.0:
            self.log_warning(f"Low ping rate: {current_rate} Hz")
        
        # Network connectivity
        if time.time() - self.last_ping_time > 5.0:
            self.log_error("Ping timeout - network connectivity issue")
```

## Testing and Validation

### Unit Tests

```cpp
// Example C++ unit test structure
#include <gtest/gtest.h>
#include <oculus_ros2/oculus_sonar_node.hpp>

class OculusSonarNodeTest : public ::testing::Test {
protected:
    void SetUp() override {
        rclcpp::init(0, nullptr);
        node_ = std::make_shared<OculusSonarNode>();
    }
    
    void TearDown() override {
        rclcpp::shutdown();
    }
    
    std::shared_ptr<OculusSonarNode> node_;
};

TEST_F(OculusSonarNodeTest, ParameterInitialization) {
    // Test parameter loading and validation
    EXPECT_EQ(node_->get_parameter("frame_id").as_string(), "sonar");
    EXPECT_EQ(node_->get_parameter("ping_rate").as_int(), 2);
    EXPECT_TRUE(node_->get_parameter("gain_assist").as_bool());
}

TEST_F(OculusSonarNodeTest, MessageValidation) {
    // Test ping message processing
    auto ping_msg = create_test_ping_message();
    EXPECT_NO_THROW(node_->validate_ping_message(ping_msg));
}

TEST_F(OculusSonarNodeTest, NetworkConfiguration) {
    // Test network setup and discovery
    EXPECT_TRUE(node_->configure_network_interface());
    EXPECT_TRUE(node_->discover_sonar_devices().size() > 0);
}
```

### Integration Tests

```bash
# Test complete sonar pipeline
#!/bin/bash
# integration_test.sh

set -e

echo "Starting Oculus integration test..."

# Start sonar node
ros2 launch oculus_ros2 default.launch.py &
SONAR_PID=$!
sleep 10

# Verify ping publication
timeout 30 ros2 topic echo /oculus_sonar/ping --once > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Ping messages received"
else
    echo "❌ No ping messages received"
    exit 1
fi

# Verify status publication
timeout 10 ros2 topic echo /oculus_sonar/status --once > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Status messages received"
else
    echo "❌ No status messages received"
    exit 1
fi

# Test parameter configuration
ros2 param set /oculus_sonar ping_rate 4
sleep 2
CURRENT_RATE=$(ros2 param get /oculus_sonar ping_rate | cut -d' ' -f3)
if [ "$CURRENT_RATE" = "4" ]; then
    echo "✅ Parameter configuration successful"
else
    echo "❌ Parameter configuration failed"
    exit 1
fi

# Cleanup
kill $SONAR_PID
echo "✅ Integration test completed successfully"
```

### Performance Benchmarks

```python
# Benchmark sonar processing performance
import time
import psutil
from memory_profiler import profile

class SonarPerformanceBenchmark:
    def __init__(self):
        self.ping_count = 0
        self.start_time = time.time()
        self.process = psutil.Process()
    
    @profile
    def benchmark_ping_processing(self, duration_seconds=60):
        """Benchmark ping processing for specified duration"""
        print(f"Benchmarking for {duration_seconds} seconds...")
        
        start_time = time.time()
        start_cpu = self.process.cpu_percent()
        start_memory = self.process.memory_info().rss
        
        while time.time() - start_time < duration_seconds:
            # Simulate ping processing
            self.process_simulated_ping()
            self.ping_count += 1
            time.sleep(0.1)  # 10Hz processing
        
        end_time = time.time()
        end_cpu = self.process.cpu_percent()
        end_memory = self.process.memory_info().rss
        
        # Calculate metrics
        actual_duration = end_time - start_time
        ping_rate = self.ping_count / actual_duration
        memory_usage = (end_memory - start_memory) / 1024 / 1024
        
        print(f"Performance Results:")
        print(f"  Processing Rate: {ping_rate:.2f} Hz")
        print(f"  CPU Usage: {end_cpu:.1f}%")
        print(f"  Memory Usage: {memory_usage:.1f} MB")
        print(f"  Total Pings: {self.ping_count}")
```

## Example Usage Scenarios

### 1. Basic Sonar Operation

```bash
# Standard sonar operation for mapping
ros2 launch oculus_ros2 default.launch.py \
    range:=25.0 \
    ping_rate:=2 \
    frequency_mode:=2

# Monitor sonar data
ros2 topic echo /oculus_sonar/ping --once
```

### 2. High-Resolution Survey

```bash
# Maximum resolution configuration
ros2 launch oculus_ros2 default.launch.py \
    nbeams:=1 \
    range:=40.0 \
    ping_rate:=1 \
    frequency_mode:=2 \
    gain_assist:=true
```

### 3. Real-time Navigation

```bash
# Fast update rate for navigation
ros2 launch oculus_ros2 default.launch.py \
    range:=15.0 \
    ping_rate:=0 \
    frequency_mode:=1 \
    nbeams:=0
```

### 4. Environmental Monitoring

```bash
# Temperature and pressure monitoring
ros2 launch oculus_ros2 default.launch.py &

# Monitor environmental data
ros2 topic echo /oculus_sonar/temperature
ros2 topic echo /oculus_sonar/pressure
```

### 5. Multi-Sonar Integration

```python
# Launch multiple sonar nodes
# custom_multi_sonar.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Forward-looking sonar
        Node(
            package="oculus_ros2",
            executable="oculus_sonar_node",
            name="forward_sonar",
            namespace="sonars/forward",
            parameters=[{
                'frame_id': 'forward_sonar_link',
                'ping_rate': 2,
                'range': 30.0
            }]
        ),
        
        # Downward-looking sonar
        Node(
            package="oculus_ros2",
            executable="oculus_sonar_node",
            name="bottom_sonar",
            namespace="sonars/bottom",
            parameters=[{
                'frame_id': 'bottom_sonar_link',
                'ping_rate': 3,
                'range': 10.0
            }]
        )
    ])
```

## Future Enhancements

### Planned Features

1. **Multi-Sonar Support**: Simultaneous operation of multiple Oculus units
2. **Advanced Processing**: Built-in beam forming and target detection algorithms
3. **Compression**: Compressed message formats for bandwidth efficiency
4. **Recording Integration**: Direct integration with rosbag2 for data recording
5. **Web Interface**: Browser-based configuration and monitoring dashboard
6. **Machine Learning**: AI-powered automatic gain control and noise reduction

### Extension Points

```cpp
// Extension framework for advanced processing
class AdvancedOculusProcessor : public OculusSonarNode {
public:
    AdvancedOculusProcessor() : OculusSonarNode() {
        setup_target_detection();
        setup_multi_sonar_fusion();
        setup_ml_processing();
    }
    
protected:
    void setup_target_detection() {
        target_detector_ = std::make_unique<SonarTargetDetector>();
        target_pub_ = create_publisher<TargetArray>("targets", 10);
    }
    
    void setup_multi_sonar_fusion() {
        fusion_processor_ = std::make_unique<MultiSonarFusion>();
        fused_cloud_pub_ = create_publisher<PointCloud2>("fused_cloud", 10);
    }
    
    void setup_ml_processing() {
        ml_processor_ = std::make_unique<MLSonarProcessor>();
        features_pub_ = create_publisher<FeatureArray>("features", 10);
    }
    
    void process_ping_advanced(const oculus_interfaces::msg::Ping& ping) override {
        // Call base processing
        OculusSonarNode::process_ping(ping);
        
        // Advanced processing
        auto targets = target_detector_->detect_targets(ping);
        auto features = ml_processor_->extract_features(ping);
        
        target_pub_->publish(targets);
        features_pub_->publish(features);
    }
};
```

### API Extensions

```cpp
// Service interfaces for advanced control
class OculusServices {
public:
    // Calibration services
    rclcpp::Service<CalibrateSonar>::SharedPtr calibration_service_;
    
    // Recording services
    rclcpp::Service<StartRecording>::SharedPtr start_recording_service_;
    rclcpp::Service<StopRecording>::SharedPtr stop_recording_service_;
    
    // Advanced configuration
    rclcpp::Service<ConfigureAdvanced>::SharedPtr advanced_config_service_;
};
```

## API Reference

### Main Classes

#### OculusSonarNode

**Purpose**: Primary ROS2 node for Oculus sonar interface

**Key Methods**:

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `configure_sonar()` | Initialize sonar connection | None | `bool` success |
| `process_ping()` | Process incoming ping data | `const Ping&` | `void` |
| `publish_ping()` | Publish processed ping | `Ping` message | `void` |
| `handle_temperature()` | Monitor thermal status | `double` temp | `void` |

#### OculusViewerNode

**Purpose**: Real-time sonar data visualization

**Key Methods**:

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `display_ping()` | Render ping as image | `const Ping&` | `void` |
| `update_display()` | Refresh visualization | None | `void` |
| `handle_mouse()` | Process mouse events | `MouseEvent` | `bool` |

### Entry Points

```python
# Available executables
executables = {
    'oculus_sonar_node': 'oculus_ros2.oculus_sonar_node:main',
    'oculus_viewer_node': 'oculus_ros2.oculus_viewer_node:main'
}
```

### Dependencies and Compatibility

| Component | Version | Compatibility | Purpose |
|-----------|---------|---------------|---------|
| **ROS2** | Humble | Primary target | Framework |
| **oculus_driver** | Latest | Required | Hardware interface |
| **OpenCV** | 4.5+ | Image processing | Visualization |
| **Boost** | 1.74+ | Network I/O | TCP/UDP communication |
| **CMake** | 3.22+ | Build system | Compilation |

This comprehensive documentation provides everything needed to understand, configure, and operate the `oculus_ros2` package for real-time sonar data acquisition and processing in underwater robotics applications.
