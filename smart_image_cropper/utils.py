"""Utility functions for Smart Image Cropper."""

import io
import logging
from typing import Union

import cv2
import numpy as np
import requests
from PIL import Image

from .exceptions import ImageProcessingError, InvalidInputError

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utility functions for image handling."""

    @staticmethod
    def download_image_bytes(url: str, timeout: int = 30) -> bytes:
        """Download an image from URL and return as bytes."""
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            logger.error(f"Failed to download image from {url}: {str(e)}")
            raise ImageProcessingError(f"Failed to download image: {str(e)}")

    @staticmethod
    def pil_to_bytes(pil_image: Image.Image, format: str = "JPEG") -> bytes:
        """Convert PIL Image to bytes."""
        try:
            img_buffer = io.BytesIO()
            # Convert to RGB if necessary
            if pil_image.mode in ("RGBA", "P"):
                pil_image = pil_image.convert("RGB")
            pil_image.save(img_buffer, format=format)
            return img_buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to convert PIL image to bytes: {str(e)}")
            raise ImageProcessingError(
                f"Failed to convert PIL image: {str(e)}")

    @staticmethod
    def validate_image_bytes(image_bytes: bytes) -> bool:
        """Validate that bytes represent a valid image."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return img is not None
        except Exception:
            return False

    @staticmethod
    def normalize_input(image_input: Union[str, bytes, Image.Image]) -> bytes:
        """Normalize different input types to bytes."""
        if isinstance(image_input, str):
            # Assume it's a URL
            return ImageUtils.download_image_bytes(image_input)
        elif isinstance(image_input, bytes):
            # Already bytes, validate
            if not ImageUtils.validate_image_bytes(image_input):
                raise InvalidInputError("Invalid image bytes provided")
            return image_input
        elif isinstance(image_input, Image.Image):
            # PIL Image
            return ImageUtils.pil_to_bytes(image_input)
        else:
            raise InvalidInputError(
                f"Unsupported input type: {type(image_input)}. "
                "Supported types: str (URL), bytes, PIL.Image.Image"
            )
