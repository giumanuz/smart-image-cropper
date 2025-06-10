# Smart Image Cropper

[![PyPI version](https://badge.fury.io/py/smart-image-cropper.svg)](https://badge.fury.io/py/smart-image-cropper)
[![Python Support](https://img.shields.io/pypi/pyversions/smart-image-cropper.svg)](https://pypi.org/project/smart-image-cropper/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An intelligent image cropping library that automatically detects objects in
images and creates optimized crops or collages. The library uses AI-powered
bounding box detection to identify the most important regions in your images and
intelligently crops them to standard aspect ratios.

## Features

- 🎯 **Smart Object Detection**: Automatically detects important objects in
  images using AI
- 🖼️ **Intelligent Cropping**: Crops images to optimal aspect ratios (4:5, 3:4,
  1:1, 4:3)
- 🎨 **Automatic Collages**: Creates beautiful collages when multiple objects
  are detected
- 📐 **Aspect Ratio Optimization**: Automatically expands crops to reach target
  aspect ratios
- 🔧 **Flexible Input**: Supports URLs, bytes, and PIL Images as input
- ⚡ **Fast Processing**: Efficient image processing with OpenCV
- 🐍 **Pure Python**: Easy to integrate into any Python project

## Installation

```bash
pip install smart-image-cropper
```

## Quick Start

```python
from smart_image_cropper import SmartImageCropper
from PIL import Image

# Initialize the cropper with your API credentials
cropper = SmartImageCropper(
    api_url="your-api-endpoint",
    api_key="your-api-key"
)

# Method 1: Process from URL
result_bytes = cropper.process_image("https://example.com/image.jpg")

# Method 2: Process from bytes
with open("image.jpg", "rb") as f:
    image_bytes = f.read()
result_bytes = cropper.process_image(image_bytes)

# Method 3: Process from PIL Image
pil_image = Image.open("image.jpg")
result_bytes = cropper.process_image(pil_image)

# Save the result
with open("cropped_result.jpg", "wb") as f:
    f.write(result_bytes)
```

## How It Works

1. **Object Detection**: The library sends your image to an AI-powered bounding
   box detection API
2. **Smart Selection**: Identifies the most important objects based on size and
   relevance
3. **Intelligent Processing**:
   - **Single Object**: Crops and expands to the nearest standard aspect ratio
   - **Multiple Objects**: Creates a collage with optimal layout
     (vertical/horizontal)
4. **Aspect Ratio Optimization**: Ensures the final result matches standard
   social media formats

## Supported Aspect Ratios

- **Portrait 4:5** (0.8) - Instagram posts
- **Portrait 3:4** (0.75) - Traditional photo format
- **Square 1:1** (1.0) - Instagram square posts
- **Landscape 4:3** (1.33) - Traditional landscape format

## API Requirements

This library requires access to a bounding box detection API. The API should:

- Accept POST requests with JSON payload containing base64-encoded images
- Return a job ID for asynchronous processing
- Provide a status endpoint to check job completion
- Return bounding box coordinates in the format:
  `{"x1": int, "y1": int, "x2": int, "y2": int}`

### Example API Integration

```python
# Your API should accept this format:
{
    "input": {
        "image": "base64-encoded-image-string"
    }
}

# And return:
{
    "id": "job-id-string"
}

# Status check should return:
{
    "status": "COMPLETED",  # or "FAILED"
    "output": [
        {"x1": 100, "y1": 50, "x2": 300, "y2": 250},
        {"x1": 400, "y1": 100, "x2": 600, "y2": 300}
    ]
}
```

## Advanced Usage

### Error Handling

```python
from smart_image_cropper import SmartImageCropper, SmartCropperError, APIError

try:
    cropper = SmartImageCropper(api_url="...", api_key="...")
    result = cropper.process_image("image.jpg")
except APIError as e:
    print(f"API Error: {e}")
except SmartCropperError as e:
    print(f"Processing Error: {e}")
```

### Logging

The library uses Python's logging module. Enable debug logging to see detailed
processing information:

```python
import logging

logging.basicConfig(level=logging.INFO)
# Now you'll see detailed processing logs
```

## Dependencies

- **OpenCV** (`opencv-python>=4.5.0`) - Image processing
- **NumPy** (`numpy>=1.20.0`) - Numerical operations
- **Pillow** (`Pillow>=8.0.0`) - PIL Image support
- **Requests** (`requests>=2.25.0`) - HTTP API calls

## Development

### Setting Up Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/smart-image-cropper.git
cd smart-image-cropper

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black .

# Type checking
mypy smart_image_cropper/
```

### Running Tests

```bash
pytest tests/ -v --cov=smart_image_cropper
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major
changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

## Changelog

### v1.0.0

- Initial release
- Support for URL, bytes, and PIL Image inputs
- Automatic object detection and smart cropping
- Collage creation for multiple objects
- Aspect ratio optimization

## Support

If you encounter any issues or have questions, please file an issue on the
[GitHub issue tracker](https://github.com/yourusername/smart-image-cropper/issues).

## Acknowledgments

- OpenCV community for excellent image processing tools
- PIL/Pillow developers for image handling capabilities
- The Python packaging community for excellent tools and documentation
