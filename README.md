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
- 🔄 **Multiple request modes**: Polling, webhook, single

## Installation

```bash
pip install smart-image-cropper
```

## Quick Start

```python
from smart_image_cropper import SmartImageCropper

# Initialize the cropper with your API credentials
cropper = SmartImageCropper(
    api_url="your-api-endpoint",
    api_key="your-api-key"
)
```

### Getting Bounding Boxes

The library supports three modes for getting bounding boxes:

1. **Polling** (default mode):

```python
# Automatically waits for job completion
bboxes = cropper.get_bounding_boxes(image_input, mode="polling")
```

2. **Webhook**:

```python
# Sends request and returns job ID
job_id = cropper.get_bounding_boxes(
    image_input,
    mode="webhook",
    webhook_url="https://your-webhook.com"
)

# The webhook will receive results when the job is completed
# The webhook payload will contain bounding boxes in the format:
# {
#     "delayTime": 1000,
#     "executionTime": 1000,
#     "input": {
#         "image": "image_bytes",
#     },
#     "id": "job_123",
#      "input": {
#         "image": "image_bytes",
#     },
#     "status": "COMPLETED",
#     "output": [
#         {"x1": 0, "y1": 0, "x2": 100, "y2": 100},
#         ...
#     ],
#     "webhook": "https://your-webhook.com"
# }
```

3. **Single Request**:

```python
# Only sends the request without waiting for results
cropper.get_bounding_boxes(image_input, mode="single")
```

### Creating a Collage

```python
# After getting bounding boxes
collage = cropper.create_collage(image_input, bboxes)
```

## Examples

### Complete Example with Polling

```python
from smart_image_cropper import SmartImageCropper
from PIL import Image

# Initialize the cropper
cropper = SmartImageCropper("https://api.example.com/detect", "your_api_key")

# Load the image
image = Image.open("example.jpg")

# Get bounding boxes (polling mode)
bboxes = cropper.get_bounding_boxes(image)

# Create the collage
result = cropper.create_collage(image, bboxes)

# Save the result
with open("result.jpg", "wb") as f:
    f.write(result)
```

### Webhook Example with Flask

```python
from smart_image_cropper import SmartImageCropper, BoundingBox
from flask import Flask, request

app = Flask(__name__)
cropper = SmartImageCropper("https://api.example.com/detect", "your_api_key")

@app.route("/process", methods=["POST"])
def process_image():
    image = request.files["image"]
    job_id = cropper.get_bounding_boxes(
        image.read(),
        mode="webhook",
        webhook_url="https://your-server.com/webhook"
    )
    return {"job_id": job_id}

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    job_id = data.get('id')
    if job_id and data.get('status') == 'COMPLETED':
        # Parse bounding boxes from the webhook data
        bboxes = [
            BoundingBox(
                x1=bbox['x1'],
                y1=bbox['y1'],
                x2=bbox['x2'],
                y2=bbox['y2']
            )
            for bbox in data.get('output', [])
        ]
        webhook_results[job_id] = bboxes
    return jsonify({'status': 'ok'})
```

## Notes

- Polling mode is the simplest to use but may block execution for a while
- Webhook mode is ideal for asynchronous applications or web servers
- Single mode is useful when you just want to send the request without waiting
  for results

## How It Works

1. The library sends the image to an AI-powered API for object detection
2. The API returns bounding boxes for detected objects
3. The library selects the best bounding boxes based on size and position
4. The image is cropped or a collage is created based on the selected boxes
5. The result is returned as image bytes

## Installation

```bash
pip install smart-image-cropper
```

## License

MIT

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
