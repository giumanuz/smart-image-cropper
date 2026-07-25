"""Main Smart Image Cropper implementation."""

from __future__ import annotations

import base64
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
from PIL import Image

from .exceptions import APIError, ImageProcessingError
from .platform import (
    PLATFORM_HORIZONTAL,
    PLATFORM_VERTICAL,
    PhotoTarget,
    PlatformFitInfo,
    PlatformTarget,
    Size,
    resolve_photo_target,
)
from .utils import ImageUtils

logger = logging.getLogger(__name__)


class AspectRatio(Enum):
    """Supported legacy target aspect ratios."""

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
    fit: Optional[PlatformFitInfo] = None


class AspectRatioCalculator:
    """Handles aspect ratio calculations."""

    @staticmethod
    def find_closest_target_ratio(current_ratio: float) -> AspectRatio:
        """Find the legacy target format closest to the current ratio."""
        min_diff = float("inf")
        closest_ratio = AspectRatio.SQUARE_1_1

        for ratio in AspectRatio:
            diff = abs(current_ratio - ratio.ratio)
            if diff < min_diff:
                min_diff = diff
                closest_ratio = ratio

        logger.info(
            "legacy.ratio content_ratio=%.3f → %s:%s (%.3f)",
            current_ratio,
            closest_ratio.value[0],
            closest_ratio.value[1],
            closest_ratio.ratio,
        )
        return closest_ratio

    @staticmethod
    def select_platform_target(current_ratio: float) -> PlatformTarget:
        """Backward-compatible wrapper around platform.select_platform_target."""
        from .platform import select_platform_target

        return select_platform_target(current_ratio)

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
    def _max_size_for_ratio(
        img_width: int, img_height: int, target_ratio: float
    ) -> Tuple[int, int]:
        """Largest (w, h) with aspect target_ratio that fits in the image."""
        if target_ratio <= 0:
            return img_width, img_height
        if img_width / img_height >= target_ratio:
            height = img_height
            width = max(1, int(round(height * target_ratio)))
        else:
            width = img_width
            height = max(1, int(round(width / target_ratio)))
        return min(width, img_width), min(height, img_height)

    @staticmethod
    def _placement_range(
        img_width: int,
        img_height: int,
        subject: BoundingBox,
        crop_w: int,
        crop_h: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Valid top-left ranges so a crop contains subject and stays in-bounds."""
        if crop_w < subject.width or crop_h < subject.height:
            return None
        if crop_w > img_width or crop_h > img_height:
            return None
        x_lo = max(0, subject.x2 - crop_w)
        x_hi = min(subject.x1, img_width - crop_w)
        y_lo = max(0, subject.y2 - crop_h)
        y_hi = min(subject.y1, img_height - crop_h)
        if x_lo > x_hi or y_lo > y_hi:
            return None
        return x_lo, x_hi, y_lo, y_hi

    @staticmethod
    def expand_to_ratio(
        img: np.ndarray, bbox: BoundingBox, target_ratio: float
    ) -> Tuple[BoundingBox, bool]:
        """Grow bbox to the largest in-image crop at target_ratio that still contains it.

        Trusts the platform/photo target: expand as far as the source allows.
        If the subject cannot fit that ratio inside the image, clamp then expand.
        """
        img_height, img_width = img.shape[:2]
        subject = bbox
        clamped = False

        logger.info(
            "bbox.expand start=%dx%d@%.3f target_ratio=%.3f image=%dx%d",
            subject.width,
            subject.height,
            subject.aspect_ratio,
            target_ratio,
            img_width,
            img_height,
        )

        max_w, max_h = BoundingBoxExpander._max_size_for_ratio(
            img_width, img_height, target_ratio
        )

        if subject.width > max_w or subject.height > max_h:
            logger.info(
                "bbox.expand subject=%dx%d exceeds max=%dx%d — clamping to ratio",
                subject.width,
                subject.height,
                max_w,
                max_h,
            )
            subject = BoundingBoxExpander.clamp_to_ratio(subject, target_ratio)
            clamped = True
            if subject.width > max_w or subject.height > max_h:
                # Still too large after clamp: take the max platform rect on subject center.
                cx = (subject.x1 + subject.x2) // 2
                cy = (subject.y1 + subject.y2) // 2
                x1 = max(0, min(img_width - max_w, cx - max_w // 2))
                y1 = max(0, min(img_height - max_h, cy - max_h // 2))
                result = BoundingBox(x1, y1, x1 + max_w, y1 + max_h)
                logger.info(
                    "bbox.expand fallback max rect %dx%d@%.3f",
                    result.width,
                    result.height,
                    result.aspect_ratio,
                )
                return result, True

        # Minimum crop height that can cover the subject at target_ratio.
        if subject.width / max(subject.height, 1) > target_ratio:
            min_h = max(subject.height, int(math.ceil(subject.width / target_ratio)))
        else:
            min_h = subject.height
        if min_h > max_h or int(round(min_h * target_ratio)) > max_w:
            logger.info(
                "bbox.expand min crop for subject exceeds image max — clamping"
            )
            subject = BoundingBoxExpander.clamp_to_ratio(bbox, target_ratio)
            clamped = True
            min_h = subject.height

        # Binary search largest height at target_ratio that can contain subject.
        best: Optional[Tuple[int, int, int, int, int, int]] = None
        lo, hi = max(1, min_h), max_h
        while lo <= hi:
            mid_h = (lo + hi) // 2
            mid_w = max(1, int(round(mid_h * target_ratio)))
            if mid_w > max_w:
                mid_w = max_w
                mid_h = max(1, int(round(mid_w / target_ratio)))

            if mid_w < subject.width or mid_h < subject.height:
                # Crop still too small to cover the subject — grow.
                lo = max(lo + 1, mid_h + 1)
                continue

            ranges = BoundingBoxExpander._placement_range(
                img_width, img_height, subject, mid_w, mid_h
            )
            if ranges is not None:
                x_lo, x_hi, y_lo, y_hi = ranges
                best = (mid_w, mid_h, x_lo, x_hi, y_lo, y_hi)
                lo = mid_h + 1
            else:
                # Fits subject size but cannot be placed in-bounds — shrink.
                hi = mid_h - 1

        if best is None:
            # Should be rare after clamp; fall back to previous clamp-on-bbox behavior.
            logger.info(
                "bbox.expand no feasible max crop — clamping subject to ratio"
            )
            return BoundingBoxExpander.clamp_to_ratio(bbox, target_ratio), True

        crop_w, crop_h, x_lo, x_hi, y_lo, y_hi = best
        pref_x = subject.x1 + subject.width // 2 - crop_w // 2
        pref_y = subject.y1 + subject.height // 2 - crop_h // 2
        x1 = max(x_lo, min(x_hi, pref_x))
        y1 = max(y_lo, min(y_hi, pref_y))
        expanded = BoundingBox(x1, y1, x1 + crop_w, y1 + crop_h)

        logger.info(
            "bbox.expand done=%dx%d@%.3f (max_possible=%dx%d, clamped=%s)",
            expanded.width,
            expanded.height,
            expanded.aspect_ratio,
            max_w,
            max_h,
            clamped,
        )
        return expanded, clamped

    @staticmethod
    def clamp_to_ratio(bbox: BoundingBox, target_ratio: float) -> BoundingBox:
        """Largest centered sub-rect of bbox with the exact target aspect ratio."""
        if bbox.width <= 0 or bbox.height <= 0:
            return bbox

        current_ratio = bbox.aspect_ratio
        if not AspectRatioCalculator.needs_expansion(
            current_ratio, target_ratio, tolerance=0.001
        ):
            return bbox

        if current_ratio > target_ratio:
            new_width = max(1, int(round(bbox.height * target_ratio)))
            new_height = bbox.height
            if new_width > bbox.width:
                new_width = bbox.width
                new_height = max(1, int(round(new_width / target_ratio)))
        else:
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
            "bbox.clamp %dx%d@%.3f → %dx%d@%.3f",
            bbox.width,
            bbox.height,
            current_ratio,
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
        expanded, _clamped = BoundingBoxExpander.expand_to_ratio(
            img, bbox, target_ratio.ratio
        )
        return expanded


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
        *,
        photo_target: Optional[PhotoTarget] = None,
        clamped_bbox: bool = False,
    ) -> CropResult:
        """Center-crop to target ratio if needed, then always resize to platform size."""
        nparr = np.frombuffer(result.image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error("platform.fit decode failed — leaving crop unchanged")
            return result

        height, width = img.shape[:2]
        if width <= 0 or height <= 0:
            return result

        input_size = Size(width, height)
        target_ratio = target_width / target_height
        current_ratio = width / height

        crop_x, crop_y = 0, 0
        crop_w, crop_h = width, height
        center_cropped = False

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
            center_cropped = True
            logger.info(
                "platform.fit center_crop %s → %dx%d for ratio=%.3f",
                input_size,
                width,
                height,
                target_ratio,
            )

        resized = width != target_width or height != target_height
        if resized:
            img = cv2.resize(
                img, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
            logger.info(
                "platform.fit resize %dx%d → %dx%d",
                width,
                height,
                target_width,
                target_height,
            )

        _, buffer = cv2.imencode(".jpg", img)
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

        output_size = Size(target_width, target_height)
        fit: Optional[PlatformFitInfo] = None
        if photo_target is not None:
            fit = PlatformFitInfo(
                photo_target=photo_target,
                input_size=input_size,
                output_size=output_size,
                clamped_bbox=clamped_bbox,
                center_cropped=center_cropped,
                resized=resized,
            )
            logger.info("platform.fit done %s", fit.summary())

        return CropResult(
            image_bytes=buffer.tobytes(),
            coordinates=scaled_coordinates,
            is_collage=result.is_collage,
            fit=fit,
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
    ) -> Tuple[float, Optional[PhotoTarget]]:
        """Resolve photo-area ratio and optional typed PhotoTarget."""
        photo = resolve_photo_target(
            content_ratio,
            use_platform_targets=use_platform_targets,
            bar_reserve_pct=bar_reserve_pct,
            target_width=target_width,
            target_height=target_height,
        )
        if photo is not None:
            return photo.ratio, photo

        legacy = AspectRatioCalculator.find_closest_target_ratio(content_ratio)
        return legacy.ratio, None

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
        logger.info(
            "crop.process bboxes=%d platform=%s bar_reserve=%.1f",
            len(bboxes),
            use_platform_targets,
            bar_reserve_pct,
        )
        if len(bboxes) == 1:
            return self._process_single_bbox(
                image_bytes,
                bboxes[0],
                use_platform_targets=use_platform_targets,
                bar_reserve_pct=bar_reserve_pct,
                target_width=target_width,
                target_height=target_height,
            )
        if len(bboxes) == 2:
            return self._process_multiple_bboxes(
                image_bytes,
                bboxes[0],
                bboxes[1],
                use_platform_targets=use_platform_targets,
                bar_reserve_pct=bar_reserve_pct,
                target_width=target_width,
                target_height=target_height,
            )

        logger.warning("crop.process unexpected bbox count=%d — returning original", len(bboxes))
        return CropResult(image_bytes=image_bytes, coordinates=[], is_collage=False)

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
        logger.info(
            "crop.single subject=%dx%d@%.3f pos=(%d,%d)-(%d,%d)",
            bbox.width,
            bbox.height,
            bbox.aspect_ratio,
            bbox.x1,
            bbox.y1,
            bbox.x2,
            bbox.y2,
        )

        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ImageProcessingError("Failed to decode image for single-bbox crop")

        # Platform from the detector crop; then expand that crop toward photo size.
        target_ratio, photo = self._resolve_photo_target(
            bbox.aspect_ratio,
            use_platform_targets=use_platform_targets,
            bar_reserve_pct=bar_reserve_pct,
            target_width=target_width,
            target_height=target_height,
        )
        expanded_bbox, clamped = BoundingBoxExpander.expand_to_ratio(
            img, bbox, target_ratio
        )
        result = ImageCropper.crop_image(image_bytes, expanded_bbox)
        if photo is not None:
            result = ImageCropper.maybe_resize_to_size(
                result,
                photo.size.width,
                photo.size.height,
                photo_target=photo,
                clamped_bbox=clamped,
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
        if direction == CollageDirection.VERTICAL:
            collage_width = max(bbox1.width, bbox2.width)
            collage_height = bbox1.height + bbox2.height
        else:
            collage_width = bbox1.width + bbox2.width
            collage_height = max(bbox1.height, bbox2.height)

        collage_ratio = collage_width / collage_height
        logger.info(
            "crop.collage direction=%s rough=%dx%d@%.3f",
            direction.value,
            collage_width,
            collage_height,
            collage_ratio,
        )

        target_ratio, photo = self._resolve_photo_target(
            collage_ratio,
            use_platform_targets=use_platform_targets,
            bar_reserve_pct=bar_reserve_pct,
            target_width=target_width,
            target_height=target_height,
        )

        result = CollageCreator.create_collage(
            image_bytes, bbox1, bbox2, direction, target_ratio
        )
        if photo is not None:
            result = ImageCropper.maybe_resize_to_size(
                result,
                photo.size.width,
                photo.size.height,
                photo_target=photo,
                clamped_bbox=False,
            )
        return result
