"""Smart Image Cropper - An intelligent image cropping library."""

from .cropper import SmartImageCropper
from .exceptions import APIError, ImageProcessingError, SmartCropperError
from .platform import (
    PLATFORM_HORIZONTAL,
    PLATFORM_VERTICAL,
    PhotoTarget,
    PlatformFitInfo,
    PlatformTarget,
    Size,
    resolve_photo_target,
    select_platform_target,
)

__version__ = "1.4.6"
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
    "Size",
    "PhotoTarget",
    "PlatformFitInfo",
    "select_platform_target",
    "resolve_photo_target",
]
