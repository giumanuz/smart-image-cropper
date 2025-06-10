"""Custom exceptions for Smart Image Cropper."""


class SmartCropperError(Exception):
    """Base exception for Smart Image Cropper."""
    pass


class APIError(SmartCropperError):
    """Exception raised when API requests fail."""
    pass


class ImageProcessingError(SmartCropperError):
    """Exception raised when image processing fails."""
    pass


class InvalidInputError(SmartCropperError):
    """Exception raised when input is invalid."""
    pass
