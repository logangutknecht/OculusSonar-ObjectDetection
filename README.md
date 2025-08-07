# Oculus Sonar Object Detection System

A comprehensive Python-based system for processing, filtering, and detecting objects in Oculus sonar data. This system provides real-time visualization, multiple detection algorithms, and advanced filtering capabilities for underwater sonar imagery.

## Features

- **Data Processing**
  - Native support for Oculus `.oculus` binary files
  - SimpleFire V2 message parsing
  - Polar to Cartesian coordinate conversion
  - Frame-by-frame processing

- **Filtering & Enhancement**
  - Median filtering for speckle noise reduction
  - Bilateral filtering for edge-preserving smoothing
  - Adaptive thresholding for segmentation
  - Range-dependent attenuation correction (TVG)
  - Water column noise removal
  - Contrast enhancement (CLAHE)
  - Beam pattern correction

- **Object Detection Methods**
  - **Classical CV**: Contour-based detection with morphological operations
  - **Edge-based**: Canny edge detection with contour analysis
  - **Clustering**: DBSCAN-based clustering of high-intensity regions
  - **Deep Learning**: YOLOv8 integration (optional, requires training)
  - **Multi-method fusion**: Combine multiple detection methods

- **Visualization**
  - Real-time display with OpenCV
  - Static frame visualization with matplotlib
  - Polar coordinate plotting
  - 3D point cloud visualization (Plotly)
  - Detection overlay and bounding boxes
  - Statistical analysis and heatmaps

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd OculusSonar-ObjectDetection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For YOLO-based detection, install PyTorch:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

### Basic Usage

Process a sonar file with default settings:
```bash
python main.py raw_assets/Oculus_20250805_164829.oculus
```

### Processing Modes

1. **Process only** - Run detection without visualization:
```bash
python main.py <file.oculus> --mode process
```

2. **Visualize specific frames**:
```bash
python main.py <file.oculus> --mode visualize --frames 0 10 20
```

3. **Real-time visualization**:
```bash
python main.py <file.oculus> --mode realtime
```
Press 'q' to quit, 's' to save current frame

4. **Analyze detections**:
```bash
python main.py <file.oculus> --mode analyze
```

5. **Full pipeline** (default):
```bash
python main.py <file.oculus> --mode all --save-outputs
```

### Using Configuration Files

Create a custom configuration:
```bash
python main.py <file.oculus> --config my_config.yaml
```

See `config.yaml` for all available options.

## Project Structure

```
OculusSonar-ObjectDetection/
├── main.py                 # Main application entry point
├── config.yaml            # Default configuration
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── raw_assets/           # Oculus sonar data files (.oculus)
├── outputs/              # Generated outputs (detections, visualizations)
└── src/
    ├── sonar_processor.py    # Oculus file reader and data structures
    ├── sonar_filters.py      # Filtering and enhancement algorithms
    ├── object_detector.py    # Object detection algorithms
    └── visualizer.py         # Visualization utilities
```

## API Usage

### Loading and Processing Data

```python
from src.sonar_processor import OculusFileReader
from src.sonar_filters import SonarEnhancer, create_default_pipeline
from src.object_detector import SonarObjectDetector

# Load sonar data
reader = OculusFileReader("path/to/file.oculus")
frames = reader.parse_all_frames()

# Setup enhancement pipeline
enhancer = create_default_pipeline()

# Setup detector
detector = SonarObjectDetector(methods=['classical', 'edge'])

# Process a frame
frame = frames[0]
enhanced = enhancer.process(frame.intensity_data)
detections = detector.detect(enhanced, frame.frame_index)

# Print results
for det in detections:
    print(f"Object at {det.centroid} with confidence {det.confidence:.2f}")
```

### Custom Filtering Pipeline

```python
from src.sonar_filters import SonarEnhancer, MedianFilter, BilateralFilter

# Create custom pipeline
enhancer = SonarEnhancer()
enhancer.add_filter(MedianFilter(kernel_size=5))
enhancer.add_filter(BilateralFilter(d=9, sigma_color=75, sigma_space=75))

# Apply to data
filtered = enhancer.process(sonar_data)
```

### Real-time Processing

```python
from src.visualizer import RealTimeVisualizer

# Setup real-time visualization
viz = RealTimeVisualizer()
viz.start()

# Process frames in real-time
for frame in frames:
    enhanced, detections = process_frame(frame)
    if not viz.update(enhanced, detections):
        break  # User pressed 'q'

viz.stop()
```

## Detection Algorithms

### Classical Computer Vision
- Uses adaptive thresholding and contour detection
- Best for high-contrast objects with clear boundaries
- Fast and reliable for simple scenes

### Edge-based Detection
- Applies Canny edge detection
- Good for objects with strong edges
- Works well in noisy environments

### Clustering (DBSCAN)
- Groups high-intensity pixels into clusters
- Effective for irregular shaped objects
- Handles overlapping objects well

### Deep Learning (YOLO)
- Requires training on sonar-specific data
- Most accurate for complex scenes
- Can classify different object types

## Configuration Options

Key parameters in `config.yaml`:

- **Filtering**
  - `median_kernel`: Size of median filter (reduce speckle)
  - `bilateral_*`: Edge-preserving smoothing parameters
  - `normalize_percentiles`: Intensity normalization range
  - `remove_water_column`: Bins to remove near sonar

- **Detection**
  - `methods`: List of detection algorithms to use
  - `combine_method`: How to merge multiple detections
  - `confidence_threshold`: Minimum detection confidence

- **Visualization**
  - `colormap`: Color scheme for sonar display
  - `show_realtime`: Enable live visualization
  - `save_outputs`: Save results to disk

## Output Format

Detection results are saved as JSON:
```json
{
  "frame_index": 0,
  "num_detections": 2,
  "detections": [
    {
      "bbox": [100, 50, 50, 50],
      "confidence": 0.85,
      "class_name": "object",
      "centroid": [125.0, 75.0],
      "area": 2500.0,
      "intensity_mean": 200.0,
      "intensity_std": 10.0
    }
  ]
}
```

## Performance Tips

1. **For large files**: Process in batches using frame indices
2. **For real-time**: Use fewer detection methods (e.g., only 'classical')
3. **For accuracy**: Use all methods with 'vote' combination
4. **For speed**: Reduce filter kernel sizes and disable bilateral filter

## Troubleshooting

### Common Issues

1. **"File not found"**: Check file path and ensure .oculus extension
2. **Low detection rate**: Adjust `confidence_threshold` and filter parameters
3. **Too many false positives**: Increase `min_area` and use 'intersection' combination
4. **Slow processing**: Reduce number of detection methods or filter complexity

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- [ ] Train custom YOLO model on sonar data
- [ ] Add tracking across frames
- [ ] Implement SLAM for mapping
- [ ] Real-time streaming from sonar hardware
- [ ] GPU acceleration for filters
- [ ] Export to GeoTIFF format
- [ ] Multi-beam sonar support

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Acknowledgments

- Blueprint Subsea for Oculus SDK documentation
- ROS2 oculus_replay_node for file format insights
- OpenCV and scikit-image communities

## Contact

For questions or support, please open an issue on GitHub.