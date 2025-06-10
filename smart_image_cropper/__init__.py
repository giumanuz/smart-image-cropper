"""Smart Image Cropper - An intelligent image cropping library."""

from .cropper import SmartImageCropper
from .exceptions import SmartCropperError, APIError, ImageProcessingError

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = ["SmartImageCropper", "SmartCropperError",
           "APIError", "ImageProcessingError"]
