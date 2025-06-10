"""Main Smart Image Cropper implementation."""

import base64
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
from PIL import Image

from .exceptions import APIError, ImageProcessingError
from .utils import ImageUtils

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AspectRatio(Enum):
    """Supported target aspect ratios."""

    PORTRAIT_4_5 = (4, 5)  # 0.8
    PORTRAIT_3_4 = (3, 4)  # 0.75
    SQUARE_1_1 = (1, 1)  # 1.0
    LANDSCAPE_4_3 = (4, 3)  # 1.33

    @property
    def ratio(self) -> float:
        return self.value[0] / self.value[1]


class CollageDirection(Enum):
    """Directions for the collage."""

    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def get_shape_type(self) -> str:
        """Determine if the bbox is square, horizontal rectangle, or vertical rectangle."""
        width_height_diff = abs(self.width - self.height) / \
            max(self.width, self.height)

        if width_height_diff <= 0.05:
            return "square"
        elif self.width > self.height:
            return "horizontal"
        else:
            return "vertical"


class AspectRatioCalculator:
    """Handles aspect ratio calculations."""

    @staticmethod
    def find_closest_target_ratio(current_ratio: float) -> AspectRatio:
        """Find the target format closest to the current ratio."""
        min_diff = float("inf")
        closest_ratio = AspectRatio.SQUARE_1_1

        for ratio in AspectRatio:
            diff = abs(current_ratio - ratio.ratio)
            if diff < min_diff:
                min_diff = diff
                closest_ratio = ratio

        logger.info(
            f"Current ratio: {current_ratio:.2f}, selected target: {closest_ratio.value[0]}:{closest_ratio.value[1]}"
        )
        return closest_ratio

    @staticmethod
    def needs_expansion(
        current_ratio: float, target_ratio: float, tolerance: float = 0.05
    ) -> bool:
        """Check if expansion is needed to reach the target ratio."""
        return abs(current_ratio - target_ratio) > tolerance


class BoundingBoxExpander:
    """Handles the expansion of bounding boxes."""

    @staticmethod
    def expand_to_width(
        img: np.ndarray, bbox: BoundingBox, target_width: int
    ) -> BoundingBox:
        """Expand a bbox to reach the target width."""
        img_height, img_width = img.shape[:2]
        current_width = bbox.width

        if current_width >= target_width:
            return bbox

        extra_width = target_width - current_width
        left_expand = extra_width // 2
        right_expand = extra_width - left_expand

        new_x1 = max(0, bbox.x1 - left_expand)
        new_x2 = min(img_width, bbox.x2 + right_expand)

        # Adjust if we couldn't expand enough
        if new_x2 - new_x1 < target_width:
            if new_x1 > 0:
                new_x1 = max(0, new_x2 - target_width)
            else:
                new_x2 = min(img_width, new_x1 + target_width)

        return BoundingBox(new_x1, bbox.y1, new_x2, bbox.y2)

    @staticmethod
    def expand_to_height(
        img: np.ndarray, bbox: BoundingBox, target_height: int
    ) -> BoundingBox:
        """Expand a bbox to reach the target height."""
        img_height, img_width = img.shape[:2]
        current_height = bbox.height

        if current_height >= target_height:
            return bbox

        extra_height = target_height - current_height
        top_expand = extra_height // 2
        bottom_expand = extra_height - top_expand

        new_y1 = max(0, bbox.y1 - top_expand)
        new_y2 = min(img_height, bbox.y2 + bottom_expand)

        # Adjust if we couldn't expand enough
        if new_y2 - new_y1 < target_height:
            if new_y1 > 0:
                new_y1 = max(0, new_y2 - target_height)
            else:
                new_y2 = min(img_height, new_y1 + target_height)

        return BoundingBox(bbox.x1, new_y1, bbox.x2, new_y2)

    @staticmethod
    def expand_to_aspect_ratio(
        img: np.ndarray, bbox: BoundingBox, target_ratio: AspectRatio
    ) -> BoundingBox:
        """Expand a bbox to reach the target format."""
        current_ratio = bbox.aspect_ratio
        target_ratio_value = target_ratio.ratio

        logger.info(
            f"Expanding bbox from {bbox.width}x{bbox.height} (ratio: {current_ratio:.3f}) to target ratio {target_ratio.value[0]}:{target_ratio.value[1]} ({target_ratio_value:.3f})"
        )

        if not AspectRatioCalculator.needs_expansion(current_ratio, target_ratio_value):
            logger.info(
                "Bbox already has correct aspect ratio, no expansion needed")
            return bbox

        expanded_bbox = bbox
        img_height, img_width = img.shape[:2]

        if current_ratio < target_ratio_value:
            # Too tall, expand width
            ideal_width = int(bbox.height * target_ratio_value)
            target_width = min(img_width, ideal_width)

            if target_width > bbox.width:
                expanded_bbox = BoundingBoxExpander.expand_to_width(
                    img, expanded_bbox, target_width
                )
                logger.info(f"Expanded width to {expanded_bbox.width}")
        else:
            # Too wide, expand height
            ideal_height = int(bbox.width / target_ratio_value)
            target_height = min(img_height, ideal_height)

            if target_height > bbox.height:
                expanded_bbox = BoundingBoxExpander.expand_to_height(
                    img, expanded_bbox, target_height
                )
                logger.info(f"Expanded height to {expanded_bbox.height}")

        final_ratio = expanded_bbox.aspect_ratio
        logger.info(
            f"Final bbox: {expanded_bbox.width}x{expanded_bbox.height}, ratio: {final_ratio:.3f}"
        )

        if AspectRatioCalculator.needs_expansion(
            final_ratio, target_ratio_value, tolerance=0.05
        ):
            logger.warning(
                f"Could not reach target ratio {target_ratio_value:.3f}, achieved {final_ratio:.3f}"
            )

        return expanded_bbox


class ImageCropper:
    """Handles image cropping and manipulation."""

    @staticmethod
    def crop_image(image_bytes: bytes, bbox: BoundingBox) -> bytes:
        """Crop image based on bounding box."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cropped = img[bbox.y1: bbox.y2, bbox.x1: bbox.x2]
            _, buffer = cv2.imencode(".jpg", cropped)
            return buffer.tobytes()
        except Exception as e:
            logger.error(f"Error cropping image: {str(e)}")
            raise ImageProcessingError(f"Error cropping image: {str(e)}")


class CollageCreator:
    """Handles the creation of collages from multiple bounding boxes."""

    @staticmethod
    def _equalize_dimensions(
        img: np.ndarray,
        bbox1: BoundingBox,
        bbox2: BoundingBox,
        direction: CollageDirection,
    ) -> Tuple[BoundingBox, BoundingBox]:
        """Equalize the dimensions of the bboxes for the collage."""
        if direction == CollageDirection.VERTICAL:
            target_width = max(bbox1.width, bbox2.width)
            bbox1 = (
                BoundingBoxExpander.expand_to_width(img, bbox1, target_width)
                if bbox1.width < target_width
                else bbox1
            )
            bbox2 = (
                BoundingBoxExpander.expand_to_width(img, bbox2, target_width)
                if bbox2.width < target_width
                else bbox2
            )
        else:
            target_height = max(bbox1.height, bbox2.height)
            bbox1 = (
                BoundingBoxExpander.expand_to_height(img, bbox1, target_height)
                if bbox1.height < target_height
                else bbox1
            )
            bbox2 = (
                BoundingBoxExpander.expand_to_height(img, bbox2, target_height)
                if bbox2.height < target_height
                else bbox2
            )

        return bbox1, bbox2

    @staticmethod
    def _adjust_for_aspect_ratio(
        img: np.ndarray,
        bbox1: BoundingBox,
        bbox2: BoundingBox,
        direction: CollageDirection,
        target_ratio: AspectRatio,
    ) -> Tuple[BoundingBox, BoundingBox]:
        """Adjust the dimensions to reach the target format."""
        if direction == CollageDirection.VERTICAL:
            collage_width = max(bbox1.width, bbox2.width)
            collage_height = bbox1.height + bbox2.height
        else:
            collage_width = bbox1.width + bbox2.width
            collage_height = max(bbox1.height, bbox2.height)

        current_ratio = collage_width / collage_height
        logger.info(
            f"Initial collage: {collage_width}x{collage_height} (ratio: {current_ratio:.3f})"
        )

        target_width = int(
            collage_height * target_ratio.value[0] / target_ratio.value[1]
        )
        target_height = int(
            collage_width * target_ratio.value[1] / target_ratio.value[0]
        )

        # Choose the adjustment that requires fewer changes
        width_diff = abs(target_width - collage_width)
        height_diff = abs(target_height - collage_height)

        logger.info(f"Target dimensions: {target_width}x{target_height}")
        logger.info(f"Width diff: {width_diff}, Height diff: {height_diff}")

        # Choose whether to expand width or height based on which requires fewer changes
        if width_diff <= height_diff:
            # Expand width (preferred when difference is less or equal)
            logger.info("Choosing to expand width")
            if target_width > collage_width:
                if direction == CollageDirection.VERTICAL:
                    # For vertical collages, expand both bboxes in width
                    bbox1 = BoundingBoxExpander.expand_to_width(
                        img, bbox1, target_width
                    )
                    bbox2 = BoundingBoxExpander.expand_to_width(
                        img, bbox2, target_width
                    )
                else:
                    # For horizontal collages, distribute expansion between the two bboxes
                    extra_width = target_width - collage_width
                    extra_per_bbox = extra_width // 2
                    new_width1 = bbox1.width + extra_per_bbox
                    new_width2 = bbox2.width + (extra_width - extra_per_bbox)
                    bbox1 = BoundingBoxExpander.expand_to_width(
                        img, bbox1, new_width1)
                    bbox2 = BoundingBoxExpander.expand_to_width(
                        img, bbox2, new_width2)
        else:
            # Expand height
            logger.info("Choosing to expand height")
            if target_height > collage_height:
                if direction == CollageDirection.HORIZONTAL:
                    # For horizontal collages, expand both bboxes in height
                    bbox1 = BoundingBoxExpander.expand_to_height(
                        img, bbox1, target_height
                    )
                    bbox2 = BoundingBoxExpander.expand_to_height(
                        img, bbox2, target_height
                    )
                else:
                    # For vertical collages, distribute expansion between the two bboxes
                    extra_height = target_height - collage_height
                    extra_per_bbox = extra_height // 2
                    new_height1 = bbox1.height + extra_per_bbox
                    new_height2 = bbox2.height + \
                        (extra_height - extra_per_bbox)
                    bbox1 = BoundingBoxExpander.expand_to_height(
                        img, bbox1, new_height1
                    )
                    bbox2 = BoundingBoxExpander.expand_to_height(
                        img, bbox2, new_height2
                    )

        return bbox1, bbox2

    @staticmethod
    def create_collage(
        image_bytes: bytes,
        bbox1: BoundingBox,
        bbox2: BoundingBox,
        direction: CollageDirection,
        target_ratio: AspectRatio,
    ) -> bytes:
        """Create a collage from two bounding boxes."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Equalize dimensions
            bbox1, bbox2 = CollageCreator._equalize_dimensions(
                img, bbox1, bbox2, direction
            )

            # Adjust for target format
            bbox1, bbox2 = CollageCreator._adjust_for_aspect_ratio(
                img, bbox1, bbox2, direction, target_ratio
            )

            # Crop regions
            crop1 = img[bbox1.y1: bbox1.y2, bbox1.x1: bbox1.x2]
            crop2 = img[bbox2.y1: bbox2.y2, bbox2.x1: bbox2.x2]

            # Create collage
            if direction == CollageDirection.VERTICAL:
                collage = np.vstack([crop1, crop2])
            else:
                collage = np.hstack([crop1, crop2])

            _, buffer = cv2.imencode(".jpg", collage)
            return buffer.tobytes()

        except Exception as e:
            logger.error(f"Error creating {direction.value} collage: {str(e)}")
            raise ImageProcessingError(f"Error creating collage: {str(e)}")


class CollageDirectionDecider:
    """Decide the direction of the collage based on the shapes of the bboxes."""

    @staticmethod
    def decide_direction(bbox1: BoundingBox, bbox2: BoundingBox) -> CollageDirection:
        """Decide if a vertical or horizontal collage should be created."""
        shape1 = bbox1.get_shape_type()
        shape2 = bbox2.get_shape_type()

        logger.info(f"Bbox shapes: {shape1} and {shape2}")

        if (
            (shape1 == "square" and shape2 == "square")
            or (shape1 == "horizontal" and shape2 == "horizontal")
            or (shape1 == "square" and shape2 == "horizontal")
            or (shape1 == "horizontal" and shape2 == "square")
        ):
            return CollageDirection.VERTICAL
        else:
            return CollageDirection.HORIZONTAL


class BoundingBoxAPIClient:
    """Client for the bounding box detection API."""

    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    def _wait_for_job_completion(self, job_id: str) -> Optional[List[BoundingBox]]:
        """Wait for the job to complete and return the bounding boxes."""
        while True:
            time.sleep(2)
            status_url = f"{self.api_url.rsplit('/', 1)[0]}/status/{job_id}"

            try:
                response = requests.get(status_url, headers=self.headers)
                response.raise_for_status()
                data = response.json()

                if data["status"] == "COMPLETED":
                    return self._parse_bboxes(data["output"])
                elif data["status"] == "FAILED":
                    logger.error(f"Job failed: {data}")
                    raise APIError(f"API job failed: {data}")

            except requests.RequestException as e:
                logger.error(f"Error checking status: {str(e)}")
                raise APIError(f"Error checking API status: {str(e)}")

    def _parse_bboxes(self, bboxes_data: List[dict]) -> List[BoundingBox]:
        """Convert the API data to BoundingBox objects."""
        return [
            BoundingBox(x1=bbox["x1"], y1=bbox["y1"],
                        x2=bbox["x2"], y2=bbox["y2"])
            for bbox in bboxes_data
        ]

    def get_bounding_boxes(self, image_bytes: bytes) -> List[BoundingBox]:
        """Get the bounding boxes for the given image."""
        logger.info("Requesting bboxes from API")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        try:
            payload = {"input": {"image": image_b64}}
            response = requests.post(
                self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()

            job_id = response.json()["id"]
            return self._wait_for_job_completion(job_id) or []

        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise APIError(f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error getting bounding boxes: {str(e)}")
            raise APIError(f"Unexpected API error: {str(e)}")


class SmartImageCropper:
    """Main Smart Image Cropper class."""

    def __init__(self, api_url: str, api_key: str):
        """
        Initialize the Smart Image Cropper.

        Args:
            api_url: URL of the bounding box detection API
            api_key: API key for authentication
        """
        self.api_client = BoundingBoxAPIClient(api_url, api_key)
        logger.info("Initialized SmartImageCropper")

    def _select_best_bboxes(self, bboxes: List[BoundingBox]) -> List[BoundingBox]:
        """Select the best bounding boxes for processing."""
        if not bboxes:
            return []

        sorted_bboxes = sorted(bboxes, key=lambda bb: bb.area, reverse=True)

        if len(sorted_bboxes) == 1:
            return sorted_bboxes

        bbox1, bbox2 = sorted_bboxes[0], sorted_bboxes[1]

        area_ratio = bbox1.area / \
            bbox2.area if bbox2.area > 0 else float("inf")
        if area_ratio >= 5:
            logger.info(
                f"Area difference too large ({area_ratio:.2f}x), using only largest bbox"
            )
            return [bbox1]

        return [bbox1, bbox2]

    def process_image(self, image_input: Union[str, bytes, Image.Image]) -> bytes:
        """
        Process an image and return the cropped result as bytes.

        Args:
            image_input: Can be a URL string, image bytes, or PIL Image

        Returns:
            bytes: The processed image as bytes

        Raises:
            SmartCropperError: If processing fails
        """
        # Normalize input to bytes
        image_bytes = ImageUtils.normalize_input(image_input)

        try:
            bboxes = self.api_client.get_bounding_boxes(image_bytes)
            if not bboxes:
                logger.warning(
                    "No bounding boxes found, returning original image")
                return image_bytes
            else:
                best_bboxes = self._select_best_bboxes(bboxes)
                return self._process_bboxes(image_bytes, best_bboxes)

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise

    def _process_bboxes(self, image_bytes: bytes, bboxes: List[BoundingBox]) -> bytes:
        """Process the bounding boxes to create the final result."""
        if len(bboxes) == 1:
            return self._process_single_bbox(image_bytes, bboxes[0])
        elif len(bboxes) == 2:
            return self._process_multiple_bboxes(image_bytes, bboxes[0], bboxes[1])
        else:
            logger.warning(
                "Unexpected number of bboxes, returning original image")
            return image_bytes

    def _process_single_bbox(self, image_bytes: bytes, bbox: BoundingBox) -> bytes:
        """Process a single bounding box."""
        logger.info("Single bbox detected, expanding to target aspect ratio")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        target_ratio = AspectRatioCalculator.find_closest_target_ratio(
            bbox.aspect_ratio
        )
        expanded_bbox = BoundingBoxExpander.expand_to_aspect_ratio(
            img, bbox, target_ratio
        )

        return ImageCropper.crop_image(image_bytes, expanded_bbox)

    def _process_multiple_bboxes(
        self, image_bytes: bytes, bbox1: BoundingBox, bbox2: BoundingBox
    ) -> bytes:
        """Process two bounding boxes to create a collage."""
        direction = CollageDirectionDecider.decide_direction(bbox1, bbox2)
        logger.info(f"Creating {direction.value} collage")

        # Calculate target format based on collage dimensions
        if direction == CollageDirection.VERTICAL:
            collage_width = max(bbox1.width, bbox2.width)
            collage_height = bbox1.height + bbox2.height
        else:
            collage_width = bbox1.width + bbox2.width
            collage_height = max(bbox1.height, bbox2.height)

        collage_ratio = collage_width / collage_height
        target_ratio = AspectRatioCalculator.find_closest_target_ratio(
            collage_ratio)

        return CollageCreator.create_collage(
            image_bytes, bbox1, bbox2, direction, target_ratio
        )
