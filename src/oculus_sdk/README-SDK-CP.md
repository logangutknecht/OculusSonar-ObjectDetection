# Oculus M3000d Multi-beam Sonar SDK

A C++ SDK and viewer application for handling sonar data from the Blueprint Subsea Oculus M3000d multi-beam sonar system.

## Overview

This project provides a complete software development kit (SDK) for interfacing with Blueprint Subsea Oculus multi-beam sonar devices, specifically the M3000d model. The SDK includes both the core communication libraries and a full-featured Qt-based viewer application for real-time sonar data visualization and recording.

## Project Structure

### `/Code` - Main Source Code

#### Core Oculus SDK (`/Code/Oculus/`)
- **`Oculus.h`** - Primary header defining Oculus data structures and communication protocols
  - Message types (SimpleFire, PingResult, UserConfig)
  - Device part numbers for various Oculus models (M370s, M750d, M1200d, M3000d, etc.)
  - Network communication structures
  - Sonar configuration parameters (ping rates, data sizes, frequencies)
- **`OsClientCtrl.h/.cpp`** - TCP client controller for sonar communication
- **`OsStatusRx.h/.cpp`** - UDP status message receiver
- **`DataWrapper.h`** - User configuration data structures
- **`OssDataWrapper.h`** - Additional data wrapper utilities

#### Display Components (`/Code/Displays/`)
- **`SonarSurface.h/.cpp`** - OpenGL-based fan display for sonar data visualization

#### User Interface (`/Code/OculusSonar/`)
- **`MainView.h/.cpp`** - Primary application window and controller
- **Control Widgets:**
  - `OnlineCtrls` - Live sonar operation controls
  - `ReviewCtrls` - Playback and file review controls
  - `SettingsCtrls` - Sonar configuration settings
  - `ToolsCtrls` - Measurement and analysis tools
  - `ModeCtrls` - Application mode switching
  - `ConnectForm` - Sonar connection management
  - `DeviceForm` - Device configuration interface

#### Graphics Engine (`/Code/RmGl/`)
- **`RmGlWidget.h/.cpp`** - OpenGL widget base class
- **`RmGlSurface.h/.cpp`** - OpenGL surface rendering
- **`RmGlOrtho.h/.cpp`** - Orthographic projection utilities
- **`PalWidget.h/.cpp`** - Color palette management

#### Utilities (`/Code/RmUtil/`)
- **`RmLogger.h/.cpp`** - Data logging system for recording sonar sessions
- **`RmPlayer.h/.cpp`** - Playback system for recorded sonar data
- **`RmUtil.h/.cpp`** - General utility functions

#### Custom Controls (`/Code/Controls/`)
- **`RangeSlider.h/.cpp`** - Custom range selection slider widget

#### Media Resources (`/Code/Media/`)
- Application icons, images, and CSS stylesheets
- Dark and light themes
- Control button graphics

### `/Docs` - Documentation
- **`NT-148-D00527-06 Oculus Data Structure Definitions.pdf`** - Official Blueprint Subsea documentation defining the Oculus data structures and communication protocols

## Key Features

### Multi-beam Sonar Support
- Support for multiple Oculus sonar models (M370s, M750d, M1200d, M3000d)
- Dual frequency operation (high/low frequency modes)
- Configurable ping rates (2Hz to 40Hz depending on model)
- Variable range settings for navigation and inspection modes

### Real-time Operation
- Live TCP/UDP communication with sonar devices
- Real-time sonar data visualization in fan display format
- Adjustable gain, range, and frequency settings
- Environmental parameter configuration (speed of sound, salinity)

### Data Recording and Playback
- Complete session recording capability
- Frame-by-frame playback with speed control
- Data export functionality
- Compressed data storage format

### Visualization Features
- OpenGL-accelerated fan display
- Multiple color palettes
- Grid overlay and measurement tools
- Zoom and pan capabilities
- Orientation controls (flip, mirror, rotate)

### User Interface
- Modern Qt-based interface with dark/light themes
- Modular control panels for different operational modes
- Connection management and device discovery
- Settings persistence and configuration management

## Build Requirements

- **Qt Framework** (5.x or 6.x) with the following modules:
  - Core, GUI, Network, Widgets
  - WinExtras (Windows-specific features)
  - Multimedia
- **OpenGL** support
- **C++20** compiler support
- **Windows SDK** (for Windows builds)

## Building the Project

The project uses Qt Creator and qmake:

```bash
qmake OculusSDK.pro
make
```

Build configurations:
- **Release**: `oculus-sdk.exe` in `./build/release/`
- **Debug**: `oculus-sdk-debug.exe` in `./build/debug/`

## Usage

### Connecting to a Sonar
1. Launch the application
2. Use the Connect form to discover and connect to available Oculus sonars
3. Configure sonar parameters (range, frequency, ping rate) via the Online controls
4. View real-time sonar data in the fan display

### Recording Data
1. Configure recording settings in the application
2. Start recording using the record controls
3. Data is saved in the proprietary `.log` format with compression

### Reviewing Recorded Data
1. Open recorded files using the File menu
2. Use Review controls for playback navigation
3. Utilize measurement tools for analysis

## Data Structures

The SDK defines comprehensive data structures for sonar communication:

- **OculusMessageHeader** - Common message header for all communications
- **OculusSimpleFireMessage** - Command messages to configure sonar operation
- **OculusSimplePingResult** - Sonar return data with acoustic measurements
- **OculusStatusMsg** - Device status and health information
- **PingConfig** - Detailed ping configuration parameters

## License

This software is licensed under the GNU General Public License v3.0. See the license headers in individual source files for details.

## Copyright

© 2017 Blueprint Subsea

## Version Information

Current version: 1.15.168 (as defined in `main.cpp`)

## Support

For technical documentation, refer to the official Blueprint Subsea Oculus Data Structure Definitions document included in the `/Docs` folder.
