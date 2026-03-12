# qp-perception

Modular visual perception library for detection, tracking, Re-ID, and target selection.

## Install

```bash
pip install qp-perception

# With Re-ID support:
pip install "qp-perception[reid]"
```

For local development:

```bash
pip install -e .
pip install -e ".[dev]"
```

## Quick Start

```python
from qp_perception.types import BoundingBox, Detection
from qp_perception.tracking import SimpleIoUTracker
from qp_perception.selection import WeightedTargetSelector

tracker = SimpleIoUTracker()
selector = WeightedTargetSelector()

detections = [
    Detection(
        bbox=BoundingBox(x=100, y=120, w=60, h=90),
        confidence=0.95,
        class_id="person",
    )
]

tracks = tracker.update(detections, timestamp=0.0)
target = selector.select(tracks, timestamp=0.0)
```

## Package Structure

```text
src/qp_perception/
    __init__.py
    types.py
    interfaces.py
    config.py
    kalman.py
    detection/
    tracking/
    selection/
    reid/
```

## Release

This package is published from GitHub Actions to PyPI using Trusted Publishing.

Release a new version with:

```bash
git tag v0.1.0
git push origin v0.1.0
```
