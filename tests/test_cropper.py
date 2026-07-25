"""Tests for cropper functionality."""

import pytest
from unittest.mock import Mock, patch
from PIL import Image
from smart_image_cropper.cropper import (
    BoundingBox, AspectRatio, AspectRatioCalculator,
    BoundingBoxExpander,
    CollageDirection, CollageDirectionDecider, SmartImageCropper,
    BoundingBoxAPIClient,
    PLATFORM_HORIZONTAL, PLATFORM_VERTICAL,
)


class TestBoundingBox:
    """Test BoundingBox class."""

    def test_bbox_properties(self):
        """Test BoundingBox basic properties."""
        bbox = BoundingBox(10, 20, 110, 170)

        assert bbox.width == 100
        assert bbox.height == 150
        assert bbox.area == 15000
        assert abs(bbox.aspect_ratio - (100/150)) < 0.001

    def test_get_shape_type_square(self):
        """Test shape type detection for square."""
        bbox = BoundingBox(0, 0, 100, 100)
        assert bbox.get_shape_type() == "square"

        # Nearly square (within 5% tolerance)
        bbox = BoundingBox(0, 0, 100, 103)
        assert bbox.get_shape_type() == "square"

    def test_get_shape_type_horizontal(self):
        """Test shape type detection for horizontal rectangle."""
        bbox = BoundingBox(0, 0, 200, 100)
        assert bbox.get_shape_type() == "horizontal"

    def test_get_shape_type_vertical(self):
        """Test shape type detection for vertical rectangle."""
        bbox = BoundingBox(0, 0, 100, 200)
        assert bbox.get_shape_type() == "vertical"


class TestAspectRatio:
    """Test AspectRatio enum."""

    def test_aspect_ratio_values(self):
        """Test aspect ratio calculations."""
        assert abs(AspectRatio.PORTRAIT_4_5.ratio - 0.8) < 0.001
        assert abs(AspectRatio.PORTRAIT_3_4.ratio - 0.75) < 0.001
        assert abs(AspectRatio.SQUARE_1_1.ratio - 1.0) < 0.001
        assert abs(AspectRatio.LANDSCAPE_4_3.ratio - (4/3)) < 0.001


class TestAspectRatioCalculator:
    """Test AspectRatioCalculator class."""

    def test_find_closest_target_ratio_square(self):
        """Test finding closest ratio for square-ish images."""
        # Should return SQUARE_1_1 for ratios close to 1.0
        closest = AspectRatioCalculator.find_closest_target_ratio(1.0)
        assert closest == AspectRatio.SQUARE_1_1

        closest = AspectRatioCalculator.find_closest_target_ratio(0.95)
        assert closest == AspectRatio.SQUARE_1_1

    def test_find_closest_target_ratio_portrait(self):
        """Test finding closest ratio for portrait images."""
        # Should return PORTRAIT_4_5 for ratios close to 0.8
        closest = AspectRatioCalculator.find_closest_target_ratio(0.8)
        assert closest == AspectRatio.PORTRAIT_4_5

        # Should return PORTRAIT_3_4 for ratios close to 0.75
        closest = AspectRatioCalculator.find_closest_target_ratio(0.75)
        assert closest == AspectRatio.PORTRAIT_3_4

    def test_find_closest_target_ratio_landscape(self):
        """Test finding closest ratio for landscape images."""
        # Should return LANDSCAPE_4_3 for ratios close to 1.33
        closest = AspectRatioCalculator.find_closest_target_ratio(1.33)
        assert closest == AspectRatio.LANDSCAPE_4_3

    def test_needs_expansion(self):
        """Test needs_expansion method."""
        # Within tolerance - no expansion needed
        assert not AspectRatioCalculator.needs_expansion(
            1.0, 1.0, tolerance=0.05)
        assert not AspectRatioCalculator.needs_expansion(
            1.0, 1.03, tolerance=0.05)

        # Outside tolerance - expansion needed
        assert AspectRatioCalculator.needs_expansion(1.0, 1.1, tolerance=0.05)
        assert AspectRatioCalculator.needs_expansion(0.8, 1.0, tolerance=0.05)


class TestCollageDirectionDecider:
    """Test CollageDirectionDecider class."""

    def test_decide_direction_vertical(self):
        """Test vertical collage decision."""
        # Two squares should be vertical
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(0, 0, 100, 100)
        direction = CollageDirectionDecider.decide_direction(bbox1, bbox2)
        assert direction == CollageDirection.VERTICAL

        # Two horizontal rectangles should be vertical
        bbox1 = BoundingBox(0, 0, 200, 100)
        bbox2 = BoundingBox(0, 0, 200, 100)
        direction = CollageDirectionDecider.decide_direction(bbox1, bbox2)
        assert direction == CollageDirection.VERTICAL

        # Square + horizontal should be vertical
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(0, 0, 200, 100)
        direction = CollageDirectionDecider.decide_direction(bbox1, bbox2)
        assert direction == CollageDirection.VERTICAL

    def test_decide_direction_horizontal(self):
        """Test horizontal collage decision."""
        # Two vertical rectangles should be horizontal
        bbox1 = BoundingBox(0, 0, 100, 200)
        bbox2 = BoundingBox(0, 0, 100, 200)
        direction = CollageDirectionDecider.decide_direction(bbox1, bbox2)
        assert direction == CollageDirection.HORIZONTAL

        # Square + vertical should be horizontal
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(0, 0, 100, 200)
        direction = CollageDirectionDecider.decide_direction(bbox1, bbox2)
        assert direction == CollageDirection.HORIZONTAL


@pytest.fixture
def mock_api_client():
    with patch('smart_image_cropper.cropper.BoundingBoxAPIClient') as mock:
        api_client_instance = Mock()
        mock.return_value = api_client_instance

        # Store the return values for different modes
        return_values = {
            "polling": [BoundingBox(0, 0, 100, 100), BoundingBox(200, 200, 300, 300)],
            "webhook": "job_123",
            "single": None
        }

        # Mock the get_bounding_boxes method to handle validation and return values
        def mock_get_bounding_boxes(image_bytes, mode, webhook_url=None):
            if mode not in ["polling", "webhook", "single"]:
                raise ValueError(
                    "Mode must be one of: polling, webhook, single")
            if mode == "webhook" and not webhook_url:
                raise ValueError(
                    "webhook_url is required when mode is 'webhook'")
            return return_values.get(mode)

        api_client_instance.get_bounding_boxes.side_effect = mock_get_bounding_boxes
        yield api_client_instance


@pytest.fixture
def mock_image_utils():
    with patch('smart_image_cropper.cropper.ImageUtils') as mock:
        # Mock normalize_input to return the input bytes directly
        mock.normalize_input.return_value = b"test_image"
        # Mock validate_image_bytes to return True
        mock.validate_image_bytes.return_value = True
        yield mock


@pytest.fixture
def sample_image():
    return Image.new('RGB', (100, 100))


def test_get_bounding_boxes_polling_mode(mock_api_client, mock_image_utils):
    """Test get_bounding_boxes in polling mode."""
    cropper = SmartImageCropper("http://test.com", "test_key")
    image_bytes = b"test_image"

    # Execute
    result = cropper.get_bounding_boxes(image_bytes, mode="polling")

    # Assert
    assert len(result) == 2
    assert isinstance(result[0], BoundingBox)
    mock_api_client.get_bounding_boxes.assert_called_once_with(
        image_bytes, "polling", None
    )
    mock_image_utils.normalize_input.assert_called_once_with(image_bytes)


def test_get_bounding_boxes_webhook_mode(mock_api_client, mock_image_utils):
    """Test get_bounding_boxes in webhook mode."""
    # Setup
    mock_api_client.get_bounding_boxes.return_value = "job_123"

    cropper = SmartImageCropper("http://test.com", "test_key")
    image_bytes = b"test_image"
    webhook_url = "https://test.com/webhook"

    # Execute
    result = cropper.get_bounding_boxes(
        image_bytes,
        mode="webhook",
        webhook_url=webhook_url
    )

    # Assert
    assert result == "job_123"
    mock_api_client.get_bounding_boxes.assert_called_once_with(
        image_bytes, "webhook", webhook_url
    )
    mock_image_utils.normalize_input.assert_called_once_with(image_bytes)


def test_get_bounding_boxes_single_mode(mock_api_client, mock_image_utils):
    """Test get_bounding_boxes in single mode."""
    # Setup
    mock_api_client.get_bounding_boxes.return_value = None

    cropper = SmartImageCropper("http://test.com", "test_key")
    image_bytes = b"test_image"

    # Execute
    result = cropper.get_bounding_boxes(image_bytes, mode="single")

    # Assert
    assert result is None
    mock_api_client.get_bounding_boxes.assert_called_once_with(
        image_bytes, "single", None
    )
    mock_image_utils.normalize_input.assert_called_once_with(image_bytes)


def test_get_bounding_boxes_invalid_mode(mock_api_client, mock_image_utils):
    """Test get_bounding_boxes with invalid mode."""
    cropper = SmartImageCropper("http://test.com", "test_key")
    image_bytes = b"test_image"

    with pytest.raises(ValueError, match="Mode must be one of: polling, webhook, single"):
        cropper.get_bounding_boxes(image_bytes, mode="invalid")

    mock_image_utils.normalize_input.assert_called_once_with(image_bytes)


def test_get_bounding_boxes_webhook_missing_url(mock_api_client, mock_image_utils):
    """Test get_bounding_boxes in webhook mode without URL."""
    cropper = SmartImageCropper("http://test.com", "test_key")
    image_bytes = b"test_image"

    with pytest.raises(ValueError, match="webhook_url is required when mode is 'webhook'"):
        cropper.get_bounding_boxes(image_bytes, mode="webhook")

    mock_image_utils.normalize_input.assert_called_once_with(image_bytes)


def test_api_client_polling_mode():
    """Test BoundingBoxAPIClient in polling mode."""
    with patch('requests.post') as mock_post, \
            patch('requests.get') as mock_get:

        # Setup mock responses
        mock_post.return_value.json.return_value = {"id": "job_123"}
        mock_post.return_value.raise_for_status = Mock()

        mock_get.return_value.json.side_effect = [
            {"status": "IN_PROGRESS"},
            {"status": "COMPLETED", "output": [
                {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
            ]}
        ]
        mock_get.return_value.raise_for_status = Mock()

        client = BoundingBoxAPIClient("http://test.com", "test_key")
        result = client.get_bounding_boxes(b"test_image", mode="polling")

        assert len(result) == 1
        assert isinstance(result[0], BoundingBox)
        assert mock_post.call_count == 1
        assert mock_get.call_count == 2


def test_api_client_webhook_mode():
    """Test BoundingBoxAPIClient in webhook mode."""
    with patch('requests.post') as mock_post:
        # Setup mock response
        mock_post.return_value.json.return_value = {"id": "job_123"}
        mock_post.return_value.raise_for_status = Mock()

        client = BoundingBoxAPIClient("http://test.com", "test_key")
        result = client.get_bounding_boxes(
            b"test_image",
            mode="webhook",
            webhook_url="https://test.com/webhook"
        )

        assert result == "job_123"
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]['json']
        assert "webhook" in payload
        assert payload["webhook"] == "https://test.com/webhook"


def test_api_client_single_mode():
    """Test BoundingBoxAPIClient in single mode."""
    with patch('requests.post') as mock_post:
        # Setup mock response
        mock_post.return_value.json.return_value = {"id": "job_123"}
        mock_post.return_value.raise_for_status = Mock()

        client = BoundingBoxAPIClient("http://test.com", "test_key")
        result = client.get_bounding_boxes(b"test_image", mode="single")

        assert result is None
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]['json']
        assert "webhook" not in payload


def test_api_client_error_handling():
    """Test BoundingBoxAPIClient error handling."""
    with patch('requests.post') as mock_post:
        # Setup mock to raise an exception
        mock_post.side_effect = Exception("API Error")

        client = BoundingBoxAPIClient("http://test.com", "test_key")

        with pytest.raises(Exception, match="API Error"):
            client.get_bounding_boxes(b"test_image")


class TestPlatformTargets:
    """Tests for 1200x630 / 1080x1350 platform sizing."""

    def test_select_platform_vertical(self):
        selected = AspectRatioCalculator.select_platform_target(0.8)
        assert selected == PLATFORM_VERTICAL
        assert selected.width == 1080
        assert selected.height == 1350

    def test_select_platform_horizontal(self):
        selected = AspectRatioCalculator.select_platform_target(1.9)
        assert selected == PLATFORM_HORIZONTAL
        assert selected.width == 1200
        assert selected.height == 630

    def test_photo_size_with_bar_reserve(self):
        width, height = PLATFORM_VERTICAL.photo_size(20)
        assert width == 1080
        assert height == 1080

    def test_expand_to_ratio_widens_tall_bbox(self):
        import numpy as np

        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        bbox = BoundingBox(400, 100, 600, 900)  # 200x800, ratio 0.25
        expanded = BoundingBoxExpander.expand_to_ratio(img, bbox, 0.8)
        assert expanded.width > bbox.width
        assert abs(expanded.aspect_ratio - 0.8) < 0.05

    def test_clamp_to_ratio_when_expand_blocked(self):
        """When borders block expand, take max rect with desired proportions."""
        import numpy as np

        # Subject already touches left/right — cannot widen toward square
        img = np.zeros((800, 400, 3), dtype=np.uint8)
        bbox = BoundingBox(0, 100, 400, 700)  # 400x600, ratio ~0.67
        result = BoundingBoxExpander.expand_to_ratio(img, bbox, 1.0)
        assert abs(result.aspect_ratio - 1.0) < 0.02
        assert result.width == result.height
        assert result.width == 400  # max possible square inside expanded area

    def test_platform_resize_even_when_expand_incomplete(self, mock_api_client):
        import cv2
        import numpy as np

        # Narrow image: cannot expand a tall subject to 4:5 fully by width alone
        img = np.zeros((1200, 500, 3), dtype=np.uint8)
        img[:] = (10, 20, 30)
        _, buffer = cv2.imencode(".jpg", img)
        image_bytes = buffer.tobytes()

        with patch(
            "smart_image_cropper.cropper.ImageUtils.normalize_input",
            return_value=image_bytes,
        ):
            cropper = SmartImageCropper("http://test.com", "test_key")
            result = cropper.create_collage(
                image_bytes,
                [BoundingBox(50, 50, 450, 1150)],  # tall, near edges
                use_platform_targets=True,
                bar_reserve_pct=0,
            )

        out = cv2.imdecode(
            np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        assert out is not None
        assert out.shape[1] == 1080
        assert out.shape[0] == 1350

    def test_platform_single_bbox_resizes_when_reachable(self, mock_api_client):
        import cv2
        import numpy as np

        img = np.zeros((2000, 1600, 3), dtype=np.uint8)
        img[:] = (40, 80, 120)
        _, buffer = cv2.imencode(".jpg", img)
        image_bytes = buffer.tobytes()

        with patch("smart_image_cropper.cropper.ImageUtils.normalize_input", return_value=image_bytes):
            cropper = SmartImageCropper("http://test.com", "test_key")
            # Near 4:5 subject with room to expand in a large canvas
            result = cropper.create_collage(
                image_bytes,
                [BoundingBox(200, 100, 1400, 1600)],
                use_platform_targets=True,
                bar_reserve_pct=0,
            )

        out = cv2.imdecode(np.frombuffer(result.image_bytes, np.uint8), cv2.IMREAD_COLOR)
        assert out is not None
        assert out.shape[1] == 1080
        assert out.shape[0] == 1350
