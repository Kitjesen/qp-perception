"""感知层数据类型：基础类型、检测、跟踪、目标观测。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrackState(str, Enum):
    """跟踪状态枚举，跨 decision/control/pipeline 共享。"""

    SEARCH = "search"
    TRACK = "track"
    LOCK = "lock"
    LOST = "lost"


@dataclass(frozen=True)
class BoundingBox:
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return max(self.w, 0.0) * max(self.h, 0.0)

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w * 0.5, self.y + self.h * 0.5


@dataclass(frozen=True)
class Detection:
    bbox: BoundingBox
    confidence: float
    class_id: str
    timestamp: float = 0.0


@dataclass
class Track:
    track_id: int
    bbox: BoundingBox
    confidence: float
    class_id: str
    first_seen_ts: float
    last_seen_ts: float
    age_frames: int = 1
    misses: int = 0
    velocity_px_per_s: tuple[float, float] = (0.0, 0.0)
    acceleration_px_per_s2: tuple[float, float] = (0.0, 0.0)
    mask_center: tuple[float, float] | None = None


@dataclass(frozen=True)
class TargetObservation:
    timestamp: float
    track_id: int
    bbox: BoundingBox
    confidence: float
    class_id: str
    velocity_px_per_s: tuple[float, float] = (0.0, 0.0)
    acceleration_px_per_s2: tuple[float, float] = (0.0, 0.0)
    mask_center: tuple[float, float] | None = None
