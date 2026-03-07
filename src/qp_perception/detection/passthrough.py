"""Lightweight detector adapter for simulation / testing."""

from __future__ import annotations

from collections.abc import Iterable

from qp_perception.types import BoundingBox, Detection


class PassthroughDetector:
    """
    Accepts an iterable of detection-like dicts and normalizes output.
    Used for synthetic scenes and CI tests where YOLO is not needed.
    """

    def inject(self, detections: list) -> None:
        """Pre-load detections to be returned on the next detect() call."""
        self._injected = list(detections)

    def detect(self, frame: object, timestamp: float = 0.0) -> list[Detection]:
        # If pre-injected detections are available, consume them first.
        if hasattr(self, "_injected") and self._injected:
            out = self._injected
            self._injected = []
            return out

        if not isinstance(frame, Iterable):
            return []

        detections: list[Detection] = []
        for item in frame:
            if isinstance(item, Detection):
                # SyntheticScene returns Detection objects directly.
                detections.append(item)
                continue
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox", (0.0, 0.0, 0.0, 0.0))
            if len(bbox) != 4:
                continue
            detections.append(
                Detection(
                    bbox=BoundingBox(
                        float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                    ),
                    confidence=float(item.get("confidence", 0.0)),
                    class_id=str(item.get("class_id", "unknown")),
                    timestamp=timestamp,
                )
            )
        return detections
