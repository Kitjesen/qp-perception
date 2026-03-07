"""Tracking module: IoU, YOLO-Seg, Fusion MOT, and CMC."""

from qp_perception.tracking.cmc import CameraMotionCompensator
from qp_perception.tracking.fusion import FusionMOT, FusionMOTConfig, FusionSegTracker
from qp_perception.tracking.iou import SimpleIoUTracker
from qp_perception.tracking.yolo_seg import YoloSegTracker

__all__ = [
    "SimpleIoUTracker",
    "YoloSegTracker",
    "FusionMOT",
    "FusionMOTConfig",
    "FusionSegTracker",
    "CameraMotionCompensator",
]
