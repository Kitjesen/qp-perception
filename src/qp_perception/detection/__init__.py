"""Detection module: YOLO and passthrough detectors."""

from qp_perception.detection.passthrough import PassthroughDetector
from qp_perception.detection.yolo import YoloDetector

__all__ = ["YoloDetector", "PassthroughDetector"]
