"""Platform social sizes and photo-area resolution for cropping."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Size:
    """Pixel dimensions."""

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
    """Final social post canvas size."""

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
        """Photo area after reserving vertical space for a bottom bar."""
        reserve = max(0.0, min(float(bar_reserve_pct), 90.0)) / 100.0
        photo_height = max(1, int(round(self.height * (1.0 - reserve))))
        return Size(self.width, photo_height)

    def __str__(self) -> str:
        label = self.name or "custom"
        return f"{label}({self.width}x{self.height})"


PLATFORM_HORIZONTAL = PlatformTarget(1200, 630, "horizontal")
PLATFORM_VERTICAL = PlatformTarget(1080, 1350, "vertical")

_DEFAULT_PLATFORMS: Tuple[PlatformTarget, ...] = (
    PLATFORM_HORIZONTAL,
    PLATFORM_VERTICAL,
)


@dataclass(frozen=True)
class PhotoTarget:
    """Resolved photo-area target used while cropping."""

    platform: PlatformTarget
    size: Size
    bar_reserve_pct: float
    content_ratio: float

    @property
    def ratio(self) -> float:
        return self.size.ratio


@dataclass(frozen=True)
class PlatformFitInfo:
    """Debug metadata describing how a crop was fitted to a platform size."""

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
            f"content_ratio={self.photo_target.content_ratio:.3f} "
            f"in={self.input_size} out={self.output_size} "
            f"clamped={self.clamped_bbox} "
            f"center_crop={self.center_cropped} "
            f"resized={self.resized}"
        )


def select_platform_target(
    content_ratio: float,
    platforms: Tuple[PlatformTarget, ...] = _DEFAULT_PLATFORMS,
) -> PlatformTarget:
    """Pick the platform whose aspect ratio is closest to the detector crop ratio."""
    if not platforms:
        raise ValueError("At least one platform target is required")

    selected = min(platforms, key=lambda p: abs(content_ratio - p.ratio))
    logger.info(
        "platform.select content_ratio=%.3f → %s (candidates=%s)",
        content_ratio,
        selected,
        ", ".join(f"{p.name}:{p.ratio:.3f}" for p in platforms),
    )
    return selected


def resolve_photo_target(
    content_ratio: float,
    *,
    use_platform_targets: bool,
    bar_reserve_pct: float = 0.0,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
) -> Optional[PhotoTarget]:
    """
    Resolve the photo area to crop into.

    Returns None when legacy aspect-ratio enums should be used instead.
    """
    if target_width and target_height and target_width > 0 and target_height > 0:
        platform = PlatformTarget(target_width, target_height, "custom")
    elif use_platform_targets:
        platform = select_platform_target(content_ratio)
    else:
        return None

    size = platform.photo_size(bar_reserve_pct)
    photo = PhotoTarget(
        platform=platform,
        size=size,
        bar_reserve_pct=float(bar_reserve_pct),
        content_ratio=float(content_ratio),
    )
    logger.info(
        "platform.resolve %s photo=%s bar_reserve=%.1f%% content_ratio=%.3f",
        platform,
        size,
        photo.bar_reserve_pct,
        photo.content_ratio,
    )
    return photo
