"""qp-perception public API."""

__version__ = "0.1.0"

from qp_perception.config import DetectorConfig, SelectorConfig, SelectorWeights
from qp_perception.interfaces import Detector, TargetSelector, Tracker
from qp_perception.kalman import CentroidKalman2D, CentroidKalmanCA
from qp_perception.types import BoundingBox, Detection, TargetObservation, Track, TrackState

__all__ = [
    "BoundingBox",
    "TrackState",
    "Detection",
    "Track",
    "TargetObservation",
    "Detector",
    "Tracker",
    "TargetSelector",
    "DetectorConfig",
    "SelectorConfig",
    "SelectorWeights",
    "CentroidKalman2D",
    "CentroidKalmanCA",
]
