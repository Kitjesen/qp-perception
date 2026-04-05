"""Unit tests for qp_perception.tracking.debug_export."""

from __future__ import annotations

import numpy as np
import pytest


class _Bbox:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h


class _Track:
    def __init__(self, tid, conf, bbox, cls="person", center=None):
        self.track_id = tid
        self.confidence = conf
        self.bbox = bbox
        self.class_id = cls
        self.mask_center = center


def _make_tracks(n=3):
    return [
        _Track(i, 0.9 - i * 0.1, _Bbox(i * 50, i * 30, 40, 60), center=(i * 50 + 20, i * 30 + 30))
        for i in range(n)
    ]


class TestToSvDetections:
    def test_empty(self):
        from qp_perception.tracking.debug_export import to_sv_detections

        det = to_sv_detections([])
        assert len(det.xyxy) == 0

    def test_xyxy_conversion(self):
        from qp_perception.tracking.debug_export import to_sv_detections

        tracks = [_Track(1, 0.95, _Bbox(10, 20, 50, 60))]
        det = to_sv_detections(tracks)
        np.testing.assert_array_almost_equal(det.xyxy[0], [10, 20, 60, 80])

    def test_confidence_and_id(self):
        from qp_perception.tracking.debug_export import to_sv_detections

        tracks = _make_tracks(2)
        det = to_sv_detections(tracks)
        assert det.confidence[0] == pytest.approx(0.9)
        assert det.tracker_id[1] == 1

    def test_data_fields(self):
        from qp_perception.tracking.debug_export import to_sv_detections

        tracks = [_Track(5, 0.8, _Bbox(0, 0, 10, 10), cls="chair", center=(5.0, 5.0))]
        det = to_sv_detections(tracks)
        assert det.data["class_name"] == ["chair"]
        assert det.data["mask_center"] == [(5.0, 5.0)]

    def test_none_mask_center(self):
        from qp_perception.tracking.debug_export import to_sv_detections

        tracks = [_Track(1, 0.9, _Bbox(0, 0, 10, 10), center=None)]
        det = to_sv_detections(tracks)
        assert det.data["mask_center"] == [None]

    def test_metadata_source(self):
        from qp_perception.tracking.debug_export import to_sv_detections

        det = to_sv_detections(_make_tracks(1))
        assert det.metadata["source"] == "qp_perception"


class TestAnnotateFrame:
    def test_returns_same_shape(self):
        from qp_perception.tracking.debug_export import annotate_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        out = annotate_frame(frame, _make_tracks(2))
        assert out.shape == frame.shape

    def test_empty_tracks_returns_original(self):
        from qp_perception.tracking.debug_export import annotate_frame

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        out = annotate_frame(frame, [])
        np.testing.assert_array_equal(out, frame)
