"""Tests for platform sizing and fit pipeline."""

from __future__ import annotations

from typing import Tuple
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from smart_image_cropper.cropper import (
    BoundingBox,
    BoundingBoxExpander,
    SmartImageCropper,
)
from smart_image_cropper.platform import (
    PLATFORM_HORIZONTAL,
    PLATFORM_VERTICAL,
    Size,
    resolve_photo_target,
    select_platform_target,
)


def _jpeg_bytes(
    width: int, height: int, color: Tuple[int, int, int] = (40, 80, 120)
) -> bytes:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = color
    ok, buffer = cv2.imencode(".jpg", img)
    assert ok
    return buffer.tobytes()


def _decode_size(image_bytes: bytes) -> Tuple[int, int]:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    assert img is not None
    height, width = img.shape[:2]
    return width, height


class TestPlatformModule:
    def test_select_vertical_for_portrait(self) -> None:
        assert select_platform_target(0.8) == PLATFORM_VERTICAL

    def test_select_horizontal_for_wide(self) -> None:
        assert select_platform_target(1.9) == PLATFORM_HORIZONTAL

    def test_photo_size_with_bar_reserve(self) -> None:
        size = PLATFORM_VERTICAL.photo_size(20)
        assert size == Size(1080, 1080)

    def test_resolve_platform_photo_target(self) -> None:
        photo = resolve_photo_target(
            0.8, use_platform_targets=True, bar_reserve_pct=20
        )
        assert photo is not None
        assert photo.platform == PLATFORM_VERTICAL
        assert photo.size == Size(1080, 1080)

    def test_resolve_legacy_returns_none_photo(self) -> None:
        photo = resolve_photo_target(0.8, use_platform_targets=False)
        assert photo is None

    def test_size_rejects_non_positive(self) -> None:
        with pytest.raises(ValueError):
            Size(0, 10)


class TestBBoxExpandAndClamp:
    def test_expand_reaches_ratio_when_space_available(self) -> None:
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        bbox = BoundingBox(400, 100, 600, 900)  # 200x800
        expanded, clamped = BoundingBoxExpander.expand_to_ratio(img, bbox, 0.8)
        assert not clamped
        assert abs(expanded.aspect_ratio - 0.8) < 0.05
        assert expanded.width > bbox.width
        # Expand as far as the square source allows at 0.8
        assert expanded.height == 1000
        assert abs(expanded.width - 800) <= 1

    def test_expand_uses_full_source_when_already_near_ratio(self) -> None:
        img = np.zeros((630, 1200, 3), dtype=np.uint8)
        bbox = BoundingBox(500, 200, 700, 450)  # small subject
        expanded, clamped = BoundingBoxExpander.expand_to_ratio(
            img, bbox, 1200 / 630
        )
        assert not clamped
        assert expanded.width >= 1190
        assert expanded.height >= 620

    def test_clamp_when_subject_cannot_fit_ratio(self) -> None:
        img = np.zeros((800, 400, 3), dtype=np.uint8)
        bbox = BoundingBox(0, 100, 400, 700)  # full width, tall
        result, clamped = BoundingBoxExpander.expand_to_ratio(img, bbox, 1.0)
        assert clamped
        assert abs(result.aspect_ratio - 1.0) < 0.02
        assert result.width == result.height == 400


class TestPlatformCollagePipeline:
    @pytest.fixture
    def cropper(self) -> SmartImageCropper:
        with patch("smart_image_cropper.cropper.BoundingBoxAPIClient"):
            return SmartImageCropper("http://test", "key")

    def test_vertical_output_exact_pixels(self, cropper: SmartImageCropper) -> None:
        image_bytes = _jpeg_bytes(1600, 2000)
        with patch(
            "smart_image_cropper.cropper.ImageUtils.normalize_input",
            return_value=image_bytes,
        ):
            result = cropper.create_collage(
                image_bytes,
                [BoundingBox(200, 100, 1400, 1600)],
                use_platform_targets=True,
                bar_reserve_pct=0,
            )

        width, height = _decode_size(result.image_bytes)
        assert (width, height) == (1080, 1350)
        assert result.fit is not None
        assert result.fit.photo_target.platform == PLATFORM_VERTICAL
        assert result.fit.output_size == Size(1080, 1350)
        assert result.fit.resized

    def test_bar_reserve_shortens_photo_area(self, cropper: SmartImageCropper) -> None:
        image_bytes = _jpeg_bytes(1600, 2000)
        with patch(
            "smart_image_cropper.cropper.ImageUtils.normalize_input",
            return_value=image_bytes,
        ):
            result = cropper.create_collage(
                image_bytes,
                [BoundingBox(200, 100, 1400, 1600)],
                use_platform_targets=True,
                bar_reserve_pct=20,
            )

        width, height = _decode_size(result.image_bytes)
        assert (width, height) == (1080, 1080)
        assert result.fit is not None
        assert result.fit.photo_target.bar_reserve_pct == 20.0

    def test_incomplete_expand_still_hits_platform_size(
        self, cropper: SmartImageCropper
    ) -> None:
        image_bytes = _jpeg_bytes(500, 1200)
        with patch(
            "smart_image_cropper.cropper.ImageUtils.normalize_input",
            return_value=image_bytes,
        ):
            result = cropper.create_collage(
                image_bytes,
                [BoundingBox(50, 50, 450, 1150)],
                use_platform_targets=True,
                bar_reserve_pct=0,
            )

        width, height = _decode_size(result.image_bytes)
        assert (width, height) == (1080, 1350)
        assert result.fit is not None
        assert result.fit.clamped_bbox or result.fit.center_cropped

    def test_horizontal_platform_for_wide_subject(
        self, cropper: SmartImageCropper
    ) -> None:
        image_bytes = _jpeg_bytes(2000, 1000)
        with patch(
            "smart_image_cropper.cropper.ImageUtils.normalize_input",
            return_value=image_bytes,
        ):
            result = cropper.create_collage(
                image_bytes,
                [BoundingBox(100, 200, 1900, 800)],  # wide
                use_platform_targets=True,
                bar_reserve_pct=0,
            )

        width, height = _decode_size(result.image_bytes)
        assert (width, height) == (1200, 630)
        assert result.fit is not None
        assert result.fit.photo_target.platform == PLATFORM_HORIZONTAL

    def test_tall_crop_picks_vertical_and_expands_into_source(
        self, cropper: SmartImageCropper
    ) -> None:
        """Platform follows the detector crop; expand that crop as far as possible."""
        image_bytes = _jpeg_bytes(1600, 2000)
        with patch(
            "smart_image_cropper.cropper.ImageUtils.normalize_input",
            return_value=image_bytes,
        ):
            result = cropper.create_collage(
                image_bytes,
                [BoundingBox(700, 200, 900, 700)],  # tall crop ~0.4
                use_platform_targets=True,
                bar_reserve_pct=30,
            )

        width, height = _decode_size(result.image_bytes)
        assert result.fit is not None
        assert result.fit.photo_target.platform == PLATFORM_VERTICAL
        # 30% bar on 1080x1350 → photo 1080x945
        assert (width, height) == (1080, 945)
        # Expanded well beyond the tiny face before resize
        assert result.fit.input_size.height >= 500

    def test_legacy_mode_does_not_force_platform_pixels(
        self, cropper: SmartImageCropper
    ) -> None:
        image_bytes = _jpeg_bytes(1600, 2000)
        with patch(
            "smart_image_cropper.cropper.ImageUtils.normalize_input",
            return_value=image_bytes,
        ):
            result = cropper.create_collage(
                image_bytes,
                [BoundingBox(200, 100, 1400, 1600)],
                use_platform_targets=False,
            )

        width, height = _decode_size(result.image_bytes)
        assert (width, height) != (1080, 1350)
        assert result.fit is None
