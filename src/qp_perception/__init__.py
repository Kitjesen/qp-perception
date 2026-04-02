"""qp-perception — modular visual perception for robotics.

Detection, multi-object tracking (FusionMOT), person Re-ID (OSNet),
and target selection for autonomous robot applications.
"""

__version__ = "0.2.0"

# --- Core types ---
from qp_perception.types import BoundingBox, Detection, TargetObservation, Track, TrackState

# --- Interfaces (Protocol-based DI) ---
from qp_perception.interfaces import Detector, TargetSelector, Tracker

# --- Config ---
from qp_perception.config import DetectorConfig, SelectorConfig, SelectorWeights

# --- Kalman filters ---
from qp_perception.kalman import CentroidKalman2D, CentroidKalmanCA

# --- Tracking ---
from qp_perception.tracking.fusion import FusionMOT, FusionMOTConfig
from qp_perception.tracking.iou import SimpleIoUTracker
from qp_perception.tracking.cmc import CameraMotionCompensator

# --- Selection ---
from qp_perception.selection.weighted import WeightedTargetSelector
from qp_perception.selection.person_following import (
    FollowingConfig,
    PersonFollowingSelector,
)

__all__ = [
    # Types
    "BoundingBox",
    "Detection",
    "Track",
    "TrackState",
    "TargetObservation",
    # Interfaces
    "Detector",
    "Tracker",
    "TargetSelector",
    # Config
    "DetectorConfig",
    "SelectorConfig",
    "SelectorWeights",
    # Kalman
    "CentroidKalman2D",
    "CentroidKalmanCA",
    # Tracking
    "FusionMOT",
    "FusionMOTConfig",
    "SimpleIoUTracker",
    "CameraMotionCompensator",
    # Selection
    "WeightedTargetSelector",
    "PersonFollowingSelector",
    "FollowingConfig",
]

# Re-ID is torch-dependent — import lazily via qp_perception.reid
# from qp_perception.reid import ReIDExtractor, ReIDConfig, AppearanceGallery
