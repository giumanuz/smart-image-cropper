"""Tests for utility functions."""

import pytest
from PIL import Image
import numpy as np
import io

from smart_image_cropper.utils import ImageUtils
from smart_image_cropper.exceptions import InvalidInputError, ImageProcessingError


class TestImageUtils:
    """Test ImageUtils class."""

    def test_validate_image_bytes_valid(self):
        """Test validation of valid image bytes."""
        # Create a simple test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [255, 0, 0]  # Red image

        import cv2
        _, buffer = cv2.imencode('.jpg', img)
        image_bytes = buffer.tobytes()

        assert ImageUtils.validate_image_bytes(image_bytes) is True

    def test_validate_image_bytes_invalid(self):
        """Test validation of invalid image bytes."""
        invalid_bytes = b"not an image"
        assert ImageUtils.validate_image_bytes(invalid_bytes) is False

    def test_pil_to_bytes_rgb(self):
        """Test conversion of RGB PIL image to bytes."""
        # Create a test PIL image
        pil_image = Image.new('RGB', (100, 100), color='red')

        result_bytes = ImageUtils.pil_to_bytes(pil_image)

        assert isinstance(result_bytes, bytes)
        assert len(result_bytes) > 0
        assert ImageUtils.validate_image_bytes(result_bytes)

    def test_pil_to_bytes_rgba(self):
        """Test conversion of RGBA PIL image to bytes."""
        # Create a test PIL image with alpha channel
        pil_image = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))

        result_bytes = ImageUtils.pil_to_bytes(pil_image)

        assert isinstance(result_bytes, bytes)
        assert len(result_bytes) > 0
        assert ImageUtils.validate_image_bytes(result_bytes)

    def test_normalize_input_bytes(self):
        """Test normalize_input with bytes input."""
        # Create test image bytes
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        import cv2
        _, buffer = cv2.imencode('.jpg', img)
        image_bytes = buffer.tobytes()

        result = ImageUtils.normalize_input(image_bytes)
        assert result == image_bytes

    def test_normalize_input_pil(self):
        """Test normalize_input with PIL Image input."""
        pil_image = Image.new('RGB', (100, 100), color='blue')

        result = ImageUtils.normalize_input(pil_image)

        assert isinstance(result, bytes)
        assert len(result) > 0
        assert ImageUtils.validate_image_bytes(result)

    def test_normalize_input_invalid_type(self):
        """Test normalize_input with invalid input type."""
        with pytest.raises(InvalidInputError, match="Unsupported input type"):
            ImageUtils.normalize_input(123)

    def test_normalize_input_invalid_bytes(self):
        """Test normalize_input with invalid bytes."""
        invalid_bytes = b"not an image"

        with pytest.raises(InvalidInputError, match="Invalid image bytes"):
            ImageUtils.normalize_input(invalid_bytes)
