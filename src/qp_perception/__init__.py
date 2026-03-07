"""qp-perception — Modular visual perception library for detection, tracking, Re-ID, and target selection."""

__version__ = "0.1.0"

from qp_perception.config import DetectorConfig, SelectorConfig, SelectorWeights
from qp_perception.interfaces import Detector, TargetSelector, Tracker
from qp_perception.kalman import CentroidKalman2D, CentroidKalmanCA
from qp_perception.types import BoundingBox, Detection, TargetObservation, Track, TrackState

__all__ = [
    # types
    "BoundingBox",
    "TrackState",
    "Detection",
    "Track",
    "TargetObservation",
    # interfaces (Protocols)
    "Detector",
    "Tracker",
    "TargetSelector",
    # config
    "DetectorConfig",
    "SelectorConfig",
    "SelectorWeights",
    # kalman
    "CentroidKalman2D",
    "CentroidKalmanCA",
]

# NOTE: Submodule exports for concrete implementations will be added
# by their own __init__.py files:
#   qp_perception.detection   — YoloDetector, YoloSegTracker, PassthroughDetector, ...
#   qp_perception.tracking    — SimpleIoUTracker, FusionMOT, FusionSegTracker, ...
#   qp_perception.selection   — WeightedTargetSelector, WeightedMultiTargetSelector, ...
#   qp_perception.reid        — ReIDExtractor, AppearanceGallery (lazy-import, torch-dependent)
