"""PersonFollowingSelector unit tests."""

import time

import pytest

from qp_perception.selection.person_following import (
    FollowingConfig,
    PersonFollowingSelector,
)
from qp_perception.types import BoundingBox, Track


def _track(tid: int, cls: str = "person", conf: float = 0.9) -> Track:
    return Track(
        track_id=tid,
        bbox=BoundingBox(x=100, y=100, w=50, h=120),
        confidence=conf,
        class_id=cls,
        first_seen_ts=0.0,
        last_seen_ts=1.0,
    )


class TestLockLifecycle:
    def test_starts_unlocked(self):
        sel = PersonFollowingSelector()
        assert sel.state == "unlocked"
        assert sel.locked_track_id is None

    def test_lock_track(self):
        sel = PersonFollowingSelector()
        sel.lock_track(42, description="red shirt")
        assert sel.state == "locked"
        assert sel.locked_track_id == 42
        assert sel.description == "red shirt"
        assert sel.is_locked

    def test_unlock(self):
        sel = PersonFollowingSelector()
        sel.lock_track(42)
        sel.unlock()
        assert sel.state == "unlocked"
        assert sel.locked_track_id is None

    def test_select_returns_locked_track(self):
        sel = PersonFollowingSelector(FollowingConfig(auto_lock=False))
        sel.lock_track(5)
        tracks = [_track(3), _track(5), _track(7)]
        obs = sel.select(tracks, 1.0)
        assert obs is not None
        assert obs.track_id == 5

    def test_select_ignores_higher_confidence_track(self):
        """Locked track is returned even when another track scores higher."""
        sel = PersonFollowingSelector(FollowingConfig(auto_lock=False))
        sel.lock_track(5)
        tracks = [_track(3, conf=0.99), _track(5, conf=0.5)]
        obs = sel.select(tracks, 1.0)
        assert obs is not None
        assert obs.track_id == 5

    def test_select_filters_non_person(self):
        """Non-person tracks are ignored."""
        sel = PersonFollowingSelector(FollowingConfig(auto_lock=False))
        sel.lock_track(5)
        tracks = [_track(5, cls="car")]
        obs = sel.select(tracks, 1.0)
        assert obs is None


class TestAutoLock:
    def test_auto_lock_highest_confidence(self):
        sel = PersonFollowingSelector(FollowingConfig(auto_lock=True))
        tracks = [_track(1, conf=0.6), _track(2, conf=0.95)]
        obs = sel.select(tracks, 1.0)
        assert obs is not None
        assert obs.track_id == 2
        assert sel.is_locked

    def test_auto_lock_disabled(self):
        sel = PersonFollowingSelector(FollowingConfig(auto_lock=False))
        tracks = [_track(1)]
        obs = sel.select(tracks, 1.0)
        assert obs is None
        assert sel.state == "unlocked"

    def test_auto_lock_below_min_confidence(self):
        sel = PersonFollowingSelector(
            FollowingConfig(auto_lock=True, min_confidence=0.8)
        )
        tracks = [_track(1, conf=0.3)]
        obs = sel.select(tracks, 1.0)
        assert obs is None


class TestSearchAndLost:
    def test_missing_track_transitions_to_searching(self):
        sel = PersonFollowingSelector(FollowingConfig(auto_lock=False))
        sel.lock_track(5)
        # Track present
        sel.select([_track(5)], 1.0)
        assert sel.state == "locked"
        # Track gone
        sel.select([_track(99)], 2.0)
        assert sel.state == "searching"

    def test_track_reappears_returns_to_locked(self):
        sel = PersonFollowingSelector(FollowingConfig(auto_lock=False))
        sel.lock_track(5)
        sel.select([_track(5)], 1.0)
        sel.select([_track(99)], 2.0)  # searching
        obs = sel.select([_track(5)], 3.0)  # back!
        assert sel.state == "locked"
        assert obs is not None
        assert obs.track_id == 5

    def test_search_timeout_transitions_to_lost(self):
        cfg = FollowingConfig(auto_lock=False, search_timeout_s=0.05)
        sel = PersonFollowingSelector(cfg)
        sel.lock_track(5)
        sel.select([_track(5)], 1.0)
        sel.select([_track(99)], 2.0)  # searching
        time.sleep(0.1)
        sel.select([_track(99)], 2.2)  # timeout
        assert sel.state == "lost"
        assert sel.is_lost
        assert sel.needs_reselect

    def test_lost_returns_none(self):
        cfg = FollowingConfig(auto_lock=False, search_timeout_s=0.05)
        sel = PersonFollowingSelector(cfg)
        sel.lock_track(5)
        sel.select([_track(5)], 1.0)
        sel.select([], 2.0)
        time.sleep(0.1)
        sel.select([], 2.2)
        assert sel.state == "lost"
        obs = sel.select([_track(5)], 3.0)
        assert obs is None  # still lost, needs re-lock

    def test_relock_after_lost(self):
        cfg = FollowingConfig(auto_lock=False, search_timeout_s=0.05)
        sel = PersonFollowingSelector(cfg)
        sel.lock_track(5)
        sel.select([_track(5)], 1.0)
        sel.select([], 2.0)
        time.sleep(0.1)
        sel.select([], 2.2)
        assert sel.state == "lost"
        # Re-lock
        sel.lock_track(5)
        obs = sel.select([_track(5)], 3.0)
        assert obs is not None
        assert obs.track_id == 5
        assert sel.state == "locked"
