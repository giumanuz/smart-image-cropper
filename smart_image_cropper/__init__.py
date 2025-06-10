"""Smart Image Cropper - An intelligent image cropping library."""

from .cropper import SmartImageCropper
from .exceptions import SmartCropperError, APIError, ImageProcessingError

__version__ = "1.0.0"
__author__ = "Giulio Manuzzi"
__email__ = "giuliomanuzzi@gmail.com"

__all__ = ["SmartImageCropper", "SmartCropperError",
           "APIError", "ImageProcessingError"]
