"""感知层配置：选择器、检测器。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelectorWeights:
    confidence: float = 0.35
    size: float = 0.20
    center_proximity: float = 0.20
    track_age: float = 0.15
    class_weight: float = 0.10
    switch_penalty: float = 0.30
    velocity_approach: float = 0.12  # bonus for targets moving toward frame center


@dataclass(frozen=True)
class SelectorConfig:
    weights: SelectorWeights = SelectorWeights()
    min_hold_time_s: float = 0.4
    delta_threshold: float = 0.12
    preferred_classes: dict[str, float] | None = None
    age_norm_frames: int = 60

    def class_weights(self) -> dict[str, float]:
        if self.preferred_classes is None:
            return {}
        return self.preferred_classes


@dataclass(frozen=True)
class DetectorConfig:
    model_path: str = "yolo11n.pt"
    confidence_threshold: float = 0.45
    nms_iou_threshold: float = 0.45
    img_size: int = 640
    device: str = ""
    tracker: str = "botsort.yaml"
    class_whitelist: tuple[str, ...] = ()
