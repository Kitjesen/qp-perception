"""IoU跟踪器单元测试。"""

import pytest

from qp_perception.tracking.iou import SimpleIoUTracker, _iou
from qp_perception.types import BoundingBox, Detection


def _det(x=100, y=100, w=50, h=50, conf=0.9, cls="person"):
    return Detection(bbox=BoundingBox(x=x, y=y, w=w, h=h), confidence=conf, class_id=cls)


class TestIoU:
    def test_identical(self):
        a = BoundingBox(x=0, y=0, w=100, h=100)
        assert _iou(a, a) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = BoundingBox(x=0, y=0, w=50, h=50)
        b = BoundingBox(x=200, y=200, w=50, h=50)
        assert _iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = BoundingBox(x=0, y=0, w=100, h=100)
        b = BoundingBox(x=50, y=50, w=100, h=100)
        iou = _iou(a, b)
        assert 0.0 < iou < 1.0

    def test_zero_area(self):
        a = BoundingBox(x=0, y=0, w=0, h=0)
        b = BoundingBox(x=0, y=0, w=100, h=100)
        assert _iou(a, b) == 0.0


class TestSimpleIoUTracker:
    @pytest.fixture
    def tracker(self):
        return SimpleIoUTracker(iou_threshold=0.2, max_misses=3)

    def test_new_detection_creates_track(self, tracker):
        tracks = tracker.update([_det()], 0.0)
        assert len(tracks) == 1
        assert tracks[0].track_id == 1

    def test_matching_detection_updates_track(self, tracker):
        tracker.update([_det(x=100, y=100)], 0.0)
        tracks = tracker.update([_det(x=105, y=105)], 0.1)
        assert len(tracks) == 1
        # age_frames starts at 1 on creation and increments each matched frame
        assert tracks[0].age_frames == 2

    def test_no_match_creates_new_track(self, tracker):
        tracker.update([_det(x=0, y=0)], 0.0)
        tracks = tracker.update([_det(x=500, y=500)], 0.1)
        assert len(tracks) == 2

    def test_missing_detection_increments_misses(self, tracker):
        tracker.update([_det()], 0.0)
        tracks = tracker.update([], 0.1)
        assert len(tracks) == 1
        assert tracks[0].misses == 1

    def test_stale_track_removed(self, tracker):
        tracker.update([_det()], 0.0)
        for i in range(5):
            tracks = tracker.update([], 0.1 * (i + 1))
        assert len(tracks) == 0

    def test_velocity_computed(self, tracker):
        # Detections must overlap (IoU > threshold) to be matched and compute velocity.
        # Using a small step so both detections share significant IoU.
        tracker.update([_det(x=100, y=100, w=100, h=100)], 0.0)
        tracks = tracker.update([_det(x=110, y=100, w=100, h=100)], 1.0)
        # Both tracks matched; velocity = (110+50 - 100+50) / 1.0 = 10 px/s
        matched = [t for t in tracks if t.misses == 0 and t.age_frames > 1]
        assert len(matched) == 1
        vx, _ = matched[0].velocity_px_per_s
        assert abs(vx - 10.0) < 1.0

    def test_multiple_detections(self, tracker):
        dets = [_det(x=0, y=0), _det(x=300, y=300), _det(x=600, y=600)]
        tracks = tracker.update(dets, 0.0)
        assert len(tracks) == 3

    def test_empty_input(self, tracker):
        tracks = tracker.update([], 0.0)
        assert len(tracks) == 0

    def test_track_ids_unique(self, tracker):
        tracker.update([_det(x=0, y=0)], 0.0)
        tracker.update([_det(x=500, y=500)], 0.1)
        tracks = tracker.update([_det(x=0, y=0), _det(x=500, y=500)], 0.2)
        ids = [t.track_id for t in tracks]
        assert len(ids) == len(set(ids))

    def test_sorted_by_id(self, tracker):
        dets = [_det(x=500, y=500), _det(x=0, y=0)]
        tracks = tracker.update(dets, 0.0)
        assert tracks[0].track_id < tracks[1].track_id
