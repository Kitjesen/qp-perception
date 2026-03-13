"""旋转选择器单元测试。"""

import pytest

from qp_perception.selection.rotating import RotatingTargetSelector
from qp_perception.types import BoundingBox, Track


def _track(tid=1, x=600, y=340, w=80, h=150, conf=0.9, cls="person"):
    return Track(
        track_id=tid,
        bbox=BoundingBox(x=x, y=y, w=w, h=h),
        confidence=conf,
        class_id=cls,
        first_seen_ts=0.0,
        last_seen_ts=0.0,
        age_frames=10,
    )


class TestRotatingTargetSelector:
    @pytest.fixture
    def selector(self):
        return RotatingTargetSelector(
            frame_width=1280,
            frame_height=720,
            dwell_time_s=1.0,
        )

    def test_empty_tracks(self, selector):
        result = selector.select([], 0.0)
        assert result is None

    def test_single_track(self, selector):
        result = selector.select([_track(tid=1)], 0.0)
        assert result is not None
        assert result.track_id == 1

    def test_rotates_after_dwell(self, selector):
        tracks = [_track(tid=1), _track(tid=2), _track(tid=3)]
        r1 = selector.select(tracks, 0.0)
        r2 = selector.select(tracks, 0.5)  # within dwell
        assert r1.track_id == r2.track_id
        r3 = selector.select(tracks, 1.5)  # after dwell
        assert r3.track_id != r1.track_id

    def test_wraps_around(self, selector):
        tracks = [_track(tid=1), _track(tid=2)]
        ids = set()
        for i in range(5):
            r = selector.select(tracks, i * 1.5)
            if r:
                ids.add(r.track_id)
        assert len(ids) == 2  # both targets visited

    def test_track_disappears(self, selector):
        selector.select([_track(tid=1), _track(tid=2)], 0.0)
        # Track 1 disappears
        r = selector.select([_track(tid=2)], 1.5)
        assert r is not None
        assert r.track_id == 2
