"""Smart Image Cropper - An intelligent image cropping library."""

from .cropper import (
    PLATFORM_HORIZONTAL,
    PLATFORM_VERTICAL,
    PlatformTarget,
    SmartImageCropper,
)
from .exceptions import SmartCropperError, APIError, ImageProcessingError

__version__ = "1.3.0"
__author__ = "Giulio Manuzzi"
__email__ = "giuliomanuzzi@gmail.com"

__all__ = [
    "SmartImageCropper",
    "SmartCropperError",
    "APIError",
    "ImageProcessingError",
    "PlatformTarget",
    "PLATFORM_HORIZONTAL",
    "PLATFORM_VERTICAL",
]
