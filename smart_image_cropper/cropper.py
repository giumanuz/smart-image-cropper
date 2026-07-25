"""Main Smart Image Cropper implementation."""

import base64
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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


@dataclass(frozen=True)
class PlatformTarget:
    """Final social post canvas size."""

    width: int
    height: int
    name: str = ""

    @property
    def ratio(self) -> float:
        return self.width / self.height

    def photo_size(self, bar_reserve_pct: float = 0.0) -> Tuple[int, int]:
        """Return photo area size after reserving space for a bottom bar."""
        reserve = max(0.0, min(float(bar_reserve_pct), 90.0)) / 100.0
        photo_height = max(1, int(round(self.height * (1.0 - reserve))))
        return self.width, photo_height


PLATFORM_HORIZONTAL = PlatformTarget(1200, 630, "horizontal")
PLATFORM_VERTICAL = PlatformTarget(1080, 1350, "vertical")


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


@dataclass
class ImageCoordinates:
    """Coordinate information for an image in the final result."""
    x: int
    y: int
    width: int
    height: int
    original_bbox: BoundingBox


@dataclass
class CropResult:
    """Result of a cropping operation containing image bytes and coordinate information."""
    image_bytes: bytes
    coordinates: List[ImageCoordinates]
    is_collage: bool = False


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
    def select_platform_target(content_ratio: float) -> PlatformTarget:
        """Pick horizontal (1200x630) or vertical (1080x1350) from content ratio."""
        horizontal_diff = abs(content_ratio - PLATFORM_HORIZONTAL.ratio)
        vertical_diff = abs(content_ratio - PLATFORM_VERTICAL.ratio)
        selected = (
            PLATFORM_HORIZONTAL
            if horizontal_diff <= vertical_diff
            else PLATFORM_VERTICAL
        )
        logger.info(
            "Content ratio %.3f → platform %s (%dx%d)",
            content_ratio,
            selected.name,
            selected.width,
            selected.height,
        )
        return selected

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
    def expand_to_ratio(
        img: np.ndarray, bbox: BoundingBox, target_ratio: float
    ) -> BoundingBox:
        """Expand a bbox proportionally to approach a target aspect ratio."""
        current_ratio = bbox.aspect_ratio

        logger.info(
            "Expanding bbox from %dx%d (ratio: %.3f) to target ratio %.3f",
            bbox.width,
            bbox.height,
            current_ratio,
            target_ratio,
        )

        if not AspectRatioCalculator.needs_expansion(current_ratio, target_ratio):
            logger.info("Bbox already has correct aspect ratio, no expansion needed")
            return bbox

        expanded_bbox = bbox
        img_height, img_width = img.shape[:2]

        if current_ratio < target_ratio:
            # Too tall, expand width
            ideal_width = int(bbox.height * target_ratio)
            target_width = min(img_width, ideal_width)

            if target_width > bbox.width:
                expanded_bbox = BoundingBoxExpander.expand_to_width(
                    img, expanded_bbox, target_width
                )
                logger.info(f"Expanded width to {expanded_bbox.width}")
        else:
            # Too wide, expand height
            ideal_height = int(bbox.width / target_ratio)
            target_height = min(img_height, ideal_height)

            if target_height > bbox.height:
                expanded_bbox = BoundingBoxExpander.expand_to_height(
                    img, expanded_bbox, target_height
                )
                logger.info(f"Expanded height to {expanded_bbox.height}")

        final_ratio = expanded_bbox.aspect_ratio
        logger.info(
            "Final bbox after expand: %dx%d, ratio: %.3f",
            expanded_bbox.width,
            expanded_bbox.height,
            final_ratio,
        )

        if AspectRatioCalculator.needs_expansion(
            final_ratio, target_ratio, tolerance=0.05
        ):
            logger.info(
                "Expand incomplete (%.3f vs %.3f) — clamping to largest rect with target ratio",
                final_ratio,
                target_ratio,
            )
            expanded_bbox = BoundingBoxExpander.clamp_to_ratio(
                expanded_bbox, target_ratio
            )

        return expanded_bbox

    @staticmethod
    def clamp_to_ratio(bbox: BoundingBox, target_ratio: float) -> BoundingBox:
        """Largest centered sub-rect of bbox with the exact target aspect ratio.

        Used when image borders block further expansion: keep max area, force ratio.
        """
        if bbox.width <= 0 or bbox.height <= 0:
            return bbox

        current_ratio = bbox.aspect_ratio
        if not AspectRatioCalculator.needs_expansion(
            current_ratio, target_ratio, tolerance=0.001
        ):
            return bbox

        if current_ratio > target_ratio:
            # Too wide → shrink width
            new_width = max(1, int(round(bbox.height * target_ratio)))
            new_height = bbox.height
            if new_width > bbox.width:
                new_width = bbox.width
                new_height = max(1, int(round(new_width / target_ratio)))
        else:
            # Too tall → shrink height
            new_height = max(1, int(round(bbox.width / target_ratio)))
            new_width = bbox.width
            if new_height > bbox.height:
                new_height = bbox.height
                new_width = max(1, int(round(new_height * target_ratio)))

        new_width = min(new_width, bbox.width)
        new_height = min(new_height, bbox.height)
        offset_x = (bbox.width - new_width) // 2
        offset_y = (bbox.height - new_height) // 2
        clamped = BoundingBox(
            bbox.x1 + offset_x,
            bbox.y1 + offset_y,
            bbox.x1 + offset_x + new_width,
            bbox.y1 + offset_y + new_height,
        )
        logger.info(
            "Clamped bbox to %dx%d (ratio: %.3f)",
            clamped.width,
            clamped.height,
            clamped.aspect_ratio,
        )
        return clamped

    @staticmethod
    def expand_to_aspect_ratio(
        img: np.ndarray, bbox: BoundingBox, target_ratio: AspectRatio
    ) -> BoundingBox:
        """Expand a bbox to reach the target format enum."""
        return BoundingBoxExpander.expand_to_ratio(img, bbox, target_ratio.ratio)


class ImageCropper:
    """Handles image cropping and manipulation."""

    @staticmethod
    def crop_image(image_bytes: bytes, bbox: BoundingBox) -> CropResult:
        """Crop image based on bounding box."""
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            cropped = img[bbox.y1: bbox.y2, bbox.x1: bbox.x2]
            _, buffer = cv2.imencode(".jpg", cropped)

            # Create coordinate information for the single cropped image
            coordinates = [ImageCoordinates(
                x=0,
                y=0,
                width=bbox.width,
                height=bbox.height,
                original_bbox=bbox
            )]

            return CropResult(
                image_bytes=buffer.tobytes(),
                coordinates=coordinates,
                is_collage=False
            )
        except Exception as e:
            logger.error(f"Error cropping image: {str(e)}")
            raise ImageProcessingError(f"Error cropping image: {str(e)}")

    @staticmethod
    def maybe_resize_to_size(
        result: CropResult,
        target_width: int,
        target_height: int,
        tolerance: float = 0.05,
    ) -> CropResult:
        """Center-crop to target ratio if needed, then always resize to platform size.

        If expansion could not fill the ratio, takes the largest centered crop that
        matches the desired proportions, then scales to the exact pixel size.
        """
        nparr = np.frombuffer(result.image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return result

        height, width = img.shape[:2]
        if width <= 0 or height <= 0:
            return result

        target_ratio = target_width / target_height
        current_ratio = width / height

        crop_x, crop_y = 0, 0
        crop_w, crop_h = width, height

        if AspectRatioCalculator.needs_expansion(
            current_ratio, target_ratio, tolerance=tolerance
        ):
            if current_ratio > target_ratio:
                crop_w = max(1, int(round(height * target_ratio)))
                crop_h = height
            else:
                crop_w = width
                crop_h = max(1, int(round(width / target_ratio)))

            crop_w = min(crop_w, width)
            crop_h = min(crop_h, height)
            crop_x = (width - crop_w) // 2
            crop_y = (height - crop_h) // 2
            img = img[crop_y: crop_y + crop_h, crop_x: crop_x + crop_w]
            width, height = crop_w, crop_h
            logger.info(
                "Took max centered crop %dx%d to match ratio %.3f",
                width,
                height,
                target_ratio,
            )

        if width == target_width and height == target_height:
            if crop_x == 0 and crop_y == 0:
                return result
            _, buffer = cv2.imencode(".jpg", img)
            adjusted_coordinates = [
                ImageCoordinates(
                    x=max(0, int(round(coord.x - crop_x))),
                    y=max(0, int(round(coord.y - crop_y))),
                    width=int(coord.width),
                    height=int(coord.height),
                    original_bbox=coord.original_bbox,
                )
                for coord in result.coordinates
            ]
            return CropResult(
                image_bytes=buffer.tobytes(),
                coordinates=adjusted_coordinates,
                is_collage=result.is_collage,
            )

        resized = cv2.resize(
            img, (target_width, target_height), interpolation=cv2.INTER_AREA
        )
        _, buffer = cv2.imencode(".jpg", resized)

        scale_x = target_width / width
        scale_y = target_height / height
        scaled_coordinates = [
            ImageCoordinates(
                x=int(round((coord.x - crop_x) * scale_x)),
                y=int(round((coord.y - crop_y) * scale_y)),
                width=int(round(coord.width * scale_x)),
                height=int(round(coord.height * scale_y)),
                original_bbox=coord.original_bbox,
            )
            for coord in result.coordinates
        ]

        logger.info(
            "Resized crop to platform photo size %dx%d", target_width, target_height
        )
        return CropResult(
            image_bytes=buffer.tobytes(),
            coordinates=scaled_coordinates,
            is_collage=result.is_collage,
        )


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
        target_ratio: float,
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

        target_width = int(collage_height * target_ratio)
        target_height = int(collage_width / target_ratio)

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
        target_ratio: float,
    ) -> CropResult:
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
                # Calculate coordinates in the final collage
                coordinates = [
                    ImageCoordinates(
                        x=0,
                        y=0,
                        width=bbox1.width,
                        height=bbox1.height,
                        original_bbox=bbox1
                    ),
                    ImageCoordinates(
                        x=0,
                        y=bbox1.height,
                        width=bbox2.width,
                        height=bbox2.height,
                        original_bbox=bbox2
                    )
                ]
            else:
                collage = np.hstack([crop1, crop2])
                # Calculate coordinates in the final collage
                coordinates = [
                    ImageCoordinates(
                        x=0,
                        y=0,
                        width=bbox1.width,
                        height=bbox1.height,
                        original_bbox=bbox1
                    ),
                    ImageCoordinates(
                        x=bbox1.width,
                        y=0,
                        width=bbox2.width,
                        height=bbox2.height,
                        original_bbox=bbox2
                    )
                ]

            _, buffer = cv2.imencode(".jpg", collage)

            return CropResult(
                image_bytes=buffer.tobytes(),
                coordinates=coordinates,
                is_collage=True
            )

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

    def get_bounding_boxes(self, image_bytes: bytes, mode: str = "polling", webhook_url: Optional[str] = None) -> Union[List[BoundingBox], str, None]:
        """
        Get the bounding boxes for the given image.

        Args:
            image_bytes: The image bytes to process
            mode: One of "polling", "webhook", or "single"
            webhook_url: Required if mode is "webhook"

        Returns:
            - If mode is "polling": List of BoundingBox objects
            - If mode is "webhook": Job ID string
            - If mode is "single": None (just sends the request)

        Raises:
            ValueError: If mode is invalid or webhook_url is missing when required
        """
        if mode not in ["polling", "webhook", "single"]:
            raise ValueError("Mode must be one of: polling, webhook, single")

        if mode == "webhook" and not webhook_url:
            raise ValueError("webhook_url is required when mode is 'webhook'")

        logger.info(f"Requesting bboxes from API with mode: {mode}")
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        try:
            payload: Dict[str, Any] = {"input": {"image": image_b64}}
            if mode == "webhook" and webhook_url:
                payload["webhook"] = webhook_url

            response = requests.post(
                self.api_url, json=payload, headers=self.headers)
            response.raise_for_status()

            job_id = response.json()["id"]

            if mode == "webhook":
                return job_id
            elif mode == "polling":
                return self._wait_for_job_completion(job_id) or []
            else:  # single mode
                return None

        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            raise APIError(f"API request failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error getting bounding boxes: {str(e)}")
            raise APIError(f"Unexpected API error: {str(e)}")

    def get_job_status(self, job_id: str) -> Optional[List[BoundingBox]]:
        """
        Get the status of a job and return bounding boxes if completed.

        Args:
            job_id: The ID of the job to check

        Returns:
            List of BoundingBox objects if job is completed, None otherwise
        """
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
            return None

        except requests.RequestException as e:
            logger.error(f"Error checking status: {str(e)}")
            raise APIError(f"Error checking API status: {str(e)}")


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

    def get_bounding_boxes(self, image_input: Union[str, bytes, Image.Image], mode: str = "polling", webhook_url: Optional[str] = None) -> Union[List[BoundingBox], str, None]:
        """
        Get bounding boxes for an image.

        Args:
            image_input: Can be a URL string, image bytes, or PIL Image
            mode: One of "polling", "webhook", or "single"
            webhook_url: Required if mode is "webhook"

        Returns:
            - If mode is "polling": List of BoundingBox objects
            - If mode is "webhook": Job ID string
            - If mode is "single": None (just sends the request)

        Raises:
            ValueError: If mode is invalid or webhook_url is missing when required
        """
        image_bytes = ImageUtils.normalize_input(image_input)
        return self.api_client.get_bounding_boxes(image_bytes, mode, webhook_url)

    def create_collage(
        self,
        image_input: Union[str, bytes, Image.Image],
        bboxes: List[BoundingBox],
        *,
        use_platform_targets: bool = False,
        bar_reserve_pct: float = 0.0,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> CropResult:
        """
        Create a collage from an image and its bounding boxes.

        Args:
            image_input: Can be a URL string, image bytes, or PIL Image
            bboxes: List of BoundingBox objects to process
            use_platform_targets: Expand toward 1200x630 / 1080x1350 instead of
                legacy 4:5/3:4/1:1/4:3 ratios
            bar_reserve_pct: Percent of final canvas height reserved for a bottom
                bar (photo area becomes shorter). Only used with platform targets.
            target_width / target_height: Optional explicit final canvas size.
                When set, overrides automatic horizontal/vertical selection.

        Returns:
            CropResult: The processed image with coordinate information
        """
        image_bytes = ImageUtils.normalize_input(image_input)
        best_bboxes = self._select_best_bboxes(bboxes)
        return self._process_bboxes(
            image_bytes,
            best_bboxes,
            use_platform_targets=use_platform_targets,
            bar_reserve_pct=bar_reserve_pct,
            target_width=target_width,
            target_height=target_height,
        )

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

    def _resolve_photo_target(
        self,
        content_ratio: float,
        *,
        use_platform_targets: bool,
        bar_reserve_pct: float,
        target_width: Optional[int],
        target_height: Optional[int],
    ) -> Tuple[float, Optional[Tuple[int, int]]]:
        """
        Resolve the photo-area aspect ratio and optional exact resize size.

        Returns:
            (target_ratio, exact_photo_size_or_None)
        """
        if target_width and target_height and target_width > 0 and target_height > 0:
            platform = PlatformTarget(target_width, target_height, "custom")
        elif use_platform_targets:
            platform = AspectRatioCalculator.select_platform_target(content_ratio)
        else:
            legacy = AspectRatioCalculator.find_closest_target_ratio(content_ratio)
            return legacy.ratio, None

        photo_width, photo_height = platform.photo_size(bar_reserve_pct)
        return photo_width / photo_height, (photo_width, photo_height)

    def _process_bboxes(
        self,
        image_bytes: bytes,
        bboxes: List[BoundingBox],
        *,
        use_platform_targets: bool = False,
        bar_reserve_pct: float = 0.0,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> CropResult:
        """Process the bounding boxes to create the final result."""
        if len(bboxes) == 1:
            return self._process_single_bbox(
                image_bytes,
                bboxes[0],
                use_platform_targets=use_platform_targets,
                bar_reserve_pct=bar_reserve_pct,
                target_width=target_width,
                target_height=target_height,
            )
        elif len(bboxes) == 2:
            return self._process_multiple_bboxes(
                image_bytes,
                bboxes[0],
                bboxes[1],
                use_platform_targets=use_platform_targets,
                bar_reserve_pct=bar_reserve_pct,
                target_width=target_width,
                target_height=target_height,
            )
        else:
            logger.warning(
                "Unexpected number of bboxes, returning original image")
            # Return original image with empty coordinates
            return CropResult(
                image_bytes=image_bytes,
                coordinates=[],
                is_collage=False
            )

    def _process_single_bbox(
        self,
        image_bytes: bytes,
        bbox: BoundingBox,
        *,
        use_platform_targets: bool = False,
        bar_reserve_pct: float = 0.0,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> CropResult:
        """Process a single bounding box."""
        logger.info("Single bbox detected, expanding to target aspect ratio")

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        target_ratio, photo_size = self._resolve_photo_target(
            bbox.aspect_ratio,
            use_platform_targets=use_platform_targets,
            bar_reserve_pct=bar_reserve_pct,
            target_width=target_width,
            target_height=target_height,
        )
        expanded_bbox = BoundingBoxExpander.expand_to_ratio(
            img, bbox, target_ratio
        )
        result = ImageCropper.crop_image(image_bytes, expanded_bbox)
        if photo_size:
            result = ImageCropper.maybe_resize_to_size(
                result, photo_size[0], photo_size[1]
            )
        return result

    def _process_multiple_bboxes(
        self,
        image_bytes: bytes,
        bbox1: BoundingBox,
        bbox2: BoundingBox,
        *,
        use_platform_targets: bool = False,
        bar_reserve_pct: float = 0.0,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
    ) -> CropResult:
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
        target_ratio, photo_size = self._resolve_photo_target(
            collage_ratio,
            use_platform_targets=use_platform_targets,
            bar_reserve_pct=bar_reserve_pct,
            target_width=target_width,
            target_height=target_height,
        )

        result = CollageCreator.create_collage(
            image_bytes, bbox1, bbox2, direction, target_ratio
        )
        if photo_size:
            result = ImageCropper.maybe_resize_to_size(
                result, photo_size[0], photo_size[1]
            )
        return result
