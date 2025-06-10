"""Tests for cropper functionality."""

import pytest
import numpy as np
from smart_image_cropper.cropper import (
    BoundingBox, AspectRatio, AspectRatioCalculator,
    CollageDirection, CollageDirectionDecider
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
