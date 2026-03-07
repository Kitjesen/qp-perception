"""Perception layer interfaces -- implement these to swap detection/tracking/selection."""

from __future__ import annotations

from typing import Protocol

from qp_perception.types import Detection, TargetObservation, Track


class Detector(Protocol):
    """Takes a frame, returns a list of detections."""

    def detect(self, frame: object, timestamp: float) -> list[Detection]: ...


class Tracker(Protocol):
    """Takes detections, returns tracked objects with stable IDs."""

    def update(self, detections: list[Detection], timestamp: float) -> list[Track]: ...


class TargetSelector(Protocol):
    """Picks the best target from tracked objects."""

    def select(self, tracks: list[Track], timestamp: float) -> TargetObservation | None: ...
