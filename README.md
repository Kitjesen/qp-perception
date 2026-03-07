# qp-perception

Modular visual perception library for detection, tracking, Re-ID, and target selection.

Extracted from [nova-rws](../../products/nova-rws) by 穹沛科技.

## Install

```bash
pip install -e .

# With Re-ID (torch) support:
pip install -e ".[reid]"

# Development:
pip install -e ".[dev]"
```

## Quick Start

```python
from qp_perception.config import DetectorConfig, SelectorConfig
from qp_perception.types import Detection, Track

# The library uses Protocol-based dependency injection.
# Concrete implementations live in submodules:
#   qp_perception.detection   — YoloDetector, YoloSegTracker, etc.
#   qp_perception.tracking    — SimpleIoUTracker, FusionMOT, etc.
#   qp_perception.selection   — WeightedTargetSelector
#   qp_perception.reid        — ReIDExtractor, AppearanceGallery

# Example with YoloSegTracker + WeightedTargetSelector:
from qp_perception.detection import YoloSegTracker
from qp_perception.selection import WeightedTargetSelector

detector = YoloSegTracker(DetectorConfig(model_path="yolo11n-seg.pt"))
selector = WeightedTargetSelector(SelectorConfig())

# frame = cv2.imread("image.jpg")
# detections = detector.detect(frame, timestamp=0.0)
# tracks = detector.update(detections, timestamp=0.0)
# target = selector.select(tracks, timestamp=0.0)
```

## Package Structure

```
src/qp_perception/
    __init__.py        # Public API re-exports
    types.py           # BoundingBox, Detection, Track, TargetObservation, TrackState
    interfaces.py      # Detector, Tracker, TargetSelector (Protocols)
    config.py          # DetectorConfig, SelectorConfig, SelectorWeights
    kalman.py          # CentroidKalman2D, CentroidKalmanCA
    detection/         # Detector implementations (YoloDetector, YoloSegTracker, ...)
    tracking/          # Tracker implementations (SimpleIoUTracker, FusionMOT, ...)
    selection/         # TargetSelector implementations (WeightedTargetSelector, ...)
    reid/              # Re-ID feature extraction and appearance gallery
```
