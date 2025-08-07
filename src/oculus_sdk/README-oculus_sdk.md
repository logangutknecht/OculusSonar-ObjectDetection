# Oculus SDK Package - Summary Documentation

## Overview

The `oculus_sdk` package contains the official Blueprint Subsea C++ SDK for interfacing with Oculus multi-beam sonar systems, specifically the M3000d model used in the Naviator platform. This package provides the foundational libraries, headers, and reference implementations for sonar communication and data processing.

## Purpose and Role

### Primary Functions
- **Hardware Interface Layer**: Provides TCP/UDP communication protocols for Oculus sonar devices
- **Data Structure Definitions**: Defines standard message formats and sonar data structures
- **Reference Implementation**: Contains complete Qt-based viewer application demonstrating SDK usage
- **Protocol Documentation**: Includes official Blueprint Subsea specifications and data structure definitions

### Integration with ROS2 Pipeline
```
Blueprint Subsea Oculus M3000d Hardware
    ↓
Oculus SDK (TCP/UDP Communication)
    ↓
oculus_driver (ROS2 Wrapper) 
    ↓
oculus_replay_node (Message Publishing)
    ↓
sonar_pointcloud → geotiff_generator
```

## Package Structure

### Core Components

```
oculus_sdk/
├── Code/                           # Main source code directory
│   ├── Oculus/                     # Core SDK libraries
│   │   ├── Oculus.h               # Primary header with data structures
│   │   ├── OsClientCtrl.cpp/.h    # TCP client controller
│   │   ├── OsStatusRx.cpp/.h      # UDP status receiver
│   │   └── DataWrapper.h          # Configuration structures
│   ├── Displays/                  # Visualization components
│   │   └── SonarSurface.cpp/.h    # OpenGL sonar fan display
│   ├── OculusSonar/              # Qt-based viewer application
│   ├── Controls/                  # UI control widgets
│   ├── RmGl/                     # OpenGL rendering utilities
│   ├── RmUtil/                   # Utility functions
│   └── Media/                    # UI resources and themes
├── Docs/                         # Official documentation
│   └── NT-148-D00527-06 Oculus Data Structure Definitions.pdf
├── README-SDK-CP.md             # Comprehensive SDK documentation
└── README-SDK-RC.md             # Release candidate notes
```

## Key Features and Capabilities

### Sonar Communication Protocol

#### TCP Control Interface
- **SimpleFire V2 Commands**: Configure sonar operation parameters
- **Real-time Control**: Adjust range, frequency, ping rate, gain settings
- **Device Discovery**: Automatic detection of Oculus devices on network
- **Connection Management**: Robust connection handling with error recovery

#### UDP Status Monitoring
- **Device Health**: Temperature, pressure, voltage monitoring
- **Operational Status**: Ping statistics, error reporting, system state
- **Network Diagnostics**: Connection quality and data throughput metrics

### Supported Oculus Models

| Model | Frequency | Max Range | Beam Count | Ping Rate |
|-------|-----------|-----------|------------|-----------|
| **M370s** | 1.2/2.1 MHz | 200m | 256 beams | 40 Hz |
| **M750d** | 0.75/1.2 MHz | 120m | 256 beams | 15 Hz |
| **M1200d** | 0.75/1.2 MHz | 40m | 512 beams | 15 Hz |
| **M3000d** | 1.2/2.1 MHz | 10m | 512 beams | 15 Hz |

### Data Structures and Message Types

#### Core Message Headers
```cpp
// Universal message header for all Oculus communications
struct OculusMessageHeader {
    uint16_t oculusId;        // 0x4f53 - Oculus identifier
    uint16_t srcDeviceId;     // Source device ID
    uint16_t dstDeviceId;     // Destination device ID
    uint16_t msgId;           // Message type identifier
    uint16_t msgVersion;      // Message version
    uint32_t payloadSize;     // Size of message payload
    uint16_t spare2;          // Reserved
};
```

#### SimpleFire V2 Configuration
```cpp
// Primary command message for sonar configuration
struct OculusSimpleFireMessage {
    OculusMessageHeader head;  // Standard header
    uint8_t masterMode;        // Master/slave operation mode
    uint8_t pingRate;          // Ping frequency (Hz)
    uint8_t networkSpeed;      // Network data rate
    uint8_t gammaCorrection;   // Display gamma correction
    uint8_t flags;             // Operation flags
    double rangePercent;       // Range as percentage of max
    double speedOfSound;       // Sound velocity (m/s)
    double salinity;           // Water salinity (ppt)
    // ... additional configuration parameters
};
```

#### Ping Result Data
```cpp
// Sonar return data structure
struct OculusSimplePingResult {
    OculusMessageHeader messageHeader;
    uint32_t pingId;           // Unique ping identifier
    uint32_t status;           // Ping status flags
    double frequency;          // Operating frequency (Hz)
    double temperature;        // Water temperature (°C)
    double pressure;           // Water pressure (bar)
    double speedOfSoundUsed;   // Applied sound velocity
    uint32_t pingStartTime;    // Ping transmission time
    uint32_t dataSize;         // Size of acoustic data
    double rangeResolution;    // Range resolution (m)
    uint16_t nRanges;          // Number of range bins
    uint16_t nBeams;           // Number of beam angles
    uint32_t imageOffset;      // Offset to image data
    uint32_t imageSize;        // Size of image data
    uint32_t messageSize;      // Total message size
    // Followed by bearing angles and image data
};
```

### Device Configuration Parameters

#### Operating Modes
- **Navigation Mode**: Long range, lower resolution for obstacle avoidance
- **Inspection Mode**: Short range, high resolution for detailed imaging
- **Dual Frequency**: Simultaneous high/low frequency operation

#### Environmental Parameters
- **Speed of Sound**: 1400-1600 m/s (temperature and salinity dependent)
- **Water Temperature**: -2°C to +35°C operating range
- **Salinity Compensation**: 0-40 ppt salinity range
- **Pressure Rating**: Up to 6000m depth (model dependent)

## Integration with ROS2 Components

### Data Flow Pipeline
```
Oculus M3000d Hardware
    ↓ (TCP/UDP via Oculus SDK)
oculus_driver (C++ wrapper)
    ↓ (Custom messages)
oculus_ros2 (ROS2 publisher)
    ↓ (sensor_msgs/Image + bearings)
oculus_replay_node (File playback)
    ↓ (Standard ROS2 messages)
sonar_pointcloud (3D conversion)
    ↓ (sensor_msgs/PointCloud2)
Mapping & Navigation Systems
```

### Message Conversion
```cpp
// SDK data → ROS2 messages
OculusSimplePingResult sdk_data;
// ↓
sensor_msgs::Image ros_image;
std_msgs::Int16MultiArray ros_bearings;
```

### Coordinate System Mapping
```cpp
// Oculus SDK coordinate system → ROS2 frames
SDK Frame: {X: Starboard, Y: Forward, Z: Down}
    ↓
ROS2 Frame: {X: Forward, Y: Port, Z: Up} (sonar_link)
```

## Licensing and Usage

### License Information
- **License**: GNU General Public License v3.0
- **Copyright**: © 2017 Blueprint Subsea
- **Version**: 1.15.168 (Reference implementation)

### Usage Restrictions
- **GPL v3 Compliance**: Derivative works must be open source
- **Commercial Use**: Contact Blueprint Subsea for commercial licensing
- **Distribution**: Source code must be provided with binary distributions

## Development and Integration

### Building the SDK
```bash
# Using Qt qmake
qmake OculusSDK.pro
make

# Or using Qt Creator
# Open OculusSDK.pro in Qt Creator and build
```

### Integration Patterns

#### Direct SDK Usage (C++)
```cpp
#include "Oculus/Oculus.h"
#include "Oculus/OsClientCtrl.h"

OsClientCtrl sonarClient;
sonarClient.connectToSonar("192.168.1.45", 52100);
sonarClient.sendSimpleFireMessage(config);
```

#### ROS2 Wrapper Integration
```cpp
// In oculus_driver package
class OculusDriver {
    OsClientCtrl* m_client;
    void processIncomingData(OculusSimplePingResult* data) {
        // Convert SDK data to ROS2 messages
        publishSonarImage(data);
        publishBearingData(data);
    }
};
```

### Key APIs and Functions

#### Connection Management
```cpp
// TCP connection to sonar
bool connectToSonar(const std::string& ipAddress, uint16_t port);
void disconnect();
bool isConnected() const;
```

#### Configuration Control
```cpp
// Send configuration commands
void sendSimpleFireMessage(const OculusSimpleFireMessage& msg);
void setRange(double rangePercent);
void setPingRate(uint8_t rateHz);
void setFrequency(bool highFreq);
```

#### Data Reception
```cpp
// Receive sonar data
void onPingResult(const OculusSimplePingResult& result);
void onStatusMessage(const OculusStatusMsg& status);
```

## Reference Application Features

### Qt-based Sonar Viewer
- **Real-time Visualization**: Live sonar fan display with OpenGL rendering
- **Recording Capability**: Save sonar sessions to compressed `.log` files
- **Playback Controls**: Frame-by-frame review with speed control
- **Parameter Adjustment**: Real-time sonar configuration via GUI
- **Measurement Tools**: Distance and angle measurement capabilities

### Display Components
- **Fan Display**: Traditional sonar sector display
- **B-Scope**: Range vs. time waterfall display
- **Data Overlays**: Range rings, bearing lines, measurement cursors
- **Color Palettes**: Multiple visualization modes (grayscale, false color)

## Performance Characteristics

### Network Performance
- **TCP Data Rate**: Up to 10-50 Mbps depending on configuration
- **UDP Status Rate**: 1-10 Hz status updates
- **Latency**: 10-50ms from ping to data reception
- **Buffer Management**: Configurable buffering for smooth data flow

### Resource Requirements
- **CPU Usage**: 10-30% for real-time processing
- **Memory**: 100-500 MB depending on buffer sizes
- **Network**: Dedicated Gigabit Ethernet recommended
- **Storage**: 1-10 GB/hour for recorded sessions

## Documentation and Support

### Official Documentation
- **Data Structure Definitions**: Complete protocol specification (PDF)
- **API Documentation**: Header file comments and examples
- **User Manual**: Operation and configuration guide
- **Technical Notes**: Performance optimization and troubleshooting

### Support Resources
- **Blueprint Subsea**: Official manufacturer support
- **Community Forums**: User discussion and troubleshooting
- **GitHub Issues**: Bug reports and feature requests
- **Integration Examples**: Reference implementations in various languages

## Future Considerations

### SDK Evolution
- **ROS2 Native Support**: Direct ROS2 message publishing
- **Protocol Updates**: Support for newer Oculus firmware versions
- **Performance Optimization**: Multi-threading and GPU acceleration
- **Extended Hardware Support**: Additional Oculus model variants

### Integration Improvements
- **Container Support**: Docker-based development environment
- **Cross-platform**: Linux ARM support for embedded systems
- **Cloud Integration**: Remote sonar operation capabilities
- **AI Integration**: Real-time object detection and classification

## Usage in Naviator Pipeline

### Current Integration
1. **SDK Libraries**: Used by `oculus_driver` for hardware communication
2. **Data Structures**: Referenced by ROS2 message converters
3. **Protocol Implementation**: Ensures compatibility with Blueprint Subsea standards
4. **Development Reference**: Provides working examples for custom implementations

### Relationship to Other Packages
- **oculus_driver**: Uses SDK for direct hardware communication
- **oculus_ros2**: Wraps SDK functionality in ROS2 nodes
- **oculus_replay_node**: Processes data originally captured via SDK
- **sonar_pointcloud**: Converts SDK-derived data to 3D representations

This SDK package serves as the foundational layer for all Oculus sonar operations in the Naviator autonomous underwater vehicle system, providing the essential interface between Blueprint Subsea hardware and the ROS2-based processing pipeline.
