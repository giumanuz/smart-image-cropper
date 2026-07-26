"""Platform social sizes for cropping.

Simple flow:
1. vertical 1080x1350, unless content ratio >= 1.4 → horizontal 1200x630
2. photo area = canvas minus bar_reserve_pct at the bottom
3. expand subject to that ratio, then resize to exact photo pixels
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Size:
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Size must be positive, got {self.width}x{self.height}")

    @property
    def ratio(self) -> float:
        return self.width / self.height

    def as_tuple(self) -> Tuple[int, int]:
        return self.width, self.height

    def __str__(self) -> str:
        return f"{self.width}x{self.height}"


@dataclass(frozen=True)
class PlatformTarget:
    width: int
    height: int
    name: str = ""

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"PlatformTarget must be positive, got {self.width}x{self.height}"
            )

    @property
    def ratio(self) -> float:
        return self.width / self.height

    @property
    def size(self) -> Size:
        return Size(self.width, self.height)

    def photo_size(self, bar_reserve_pct: float = 0.0) -> Size:
        reserve = max(0.0, min(float(bar_reserve_pct), 90.0)) / 100.0
        return Size(self.width, max(1, int(round(self.height * (1.0 - reserve)))))

    def __str__(self) -> str:
        return f"{self.name or 'custom'}({self.width}x{self.height})"


PLATFORM_HORIZONTAL = PlatformTarget(1200, 630, "horizontal")
PLATFORM_VERTICAL = PlatformTarget(1080, 1350, "vertical")

_DEFAULT_PLATFORMS: Tuple[PlatformTarget, ...] = (
    PLATFORM_HORIZONTAL,
    PLATFORM_VERTICAL,
)


@dataclass(frozen=True)
class PhotoTarget:
    platform: PlatformTarget
    size: Size
    bar_reserve_pct: float
    content_ratio: float

    @property
    def ratio(self) -> float:
        return self.size.ratio


@dataclass(frozen=True)
class PlatformFitInfo:
    photo_target: PhotoTarget
    input_size: Size
    output_size: Size
    clamped_bbox: bool
    center_cropped: bool
    resized: bool

    def summary(self) -> str:
        return (
            f"platform={self.photo_target.platform} "
            f"photo={self.photo_target.size} "
            f"bar_reserve={self.photo_target.bar_reserve_pct:.1f}% "
            f"in={self.input_size} out={self.output_size}"
        )


def select_platform_target(
    content_ratio: float,
    platforms: Tuple[PlatformTarget, ...] = _DEFAULT_PLATFORMS,
    *,
    horizontal_min_ratio: float = 1.4,
) -> PlatformTarget:
    """Vertical by default; horizontal only if clearly wide."""
    by_name = {p.name: p for p in platforms}
    if content_ratio >= horizontal_min_ratio and "horizontal" in by_name:
        selected = by_name["horizontal"]
    else:
        selected = by_name.get("vertical") or platforms[0]
    logger.info("platform.select %.3f → %s", content_ratio, selected)
    return selected


def resolve_photo_target(
    content_ratio: float,
    *,
    use_platform_targets: bool,
    bar_reserve_pct: float = 0.0,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    horizontal_min_ratio: float = 1.4,
) -> Optional[PhotoTarget]:
    if target_width and target_height and target_width > 0 and target_height > 0:
        platform = PlatformTarget(target_width, target_height, "custom")
    elif use_platform_targets:
        platform = select_platform_target(
            content_ratio, horizontal_min_ratio=horizontal_min_ratio
        )
    else:
        return None

    photo = PhotoTarget(
        platform=platform,
        size=platform.photo_size(bar_reserve_pct),
        bar_reserve_pct=float(bar_reserve_pct),
        content_ratio=float(content_ratio),
    )
    logger.info(
        "platform.resolve %s photo=%s bar=%.0f%%",
        platform,
        photo.size,
        photo.bar_reserve_pct,
    )
    return photo
