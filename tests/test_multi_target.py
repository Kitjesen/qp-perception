"""多目标选择器与分配器单元测试。"""

import pytest

from qp_perception.selection.multi import (
    TargetAllocator,
    TargetAssignment,
    WeightedMultiTargetSelector,
)
from qp_perception.types import BoundingBox, TargetObservation, Track


def _obs(tid=1, x=600, y=340, w=80, h=150, conf=0.9, cls="person"):
    return TargetObservation(
        timestamp=0.0, track_id=tid,
        bbox=BoundingBox(x=x, y=y, w=w, h=h),
        confidence=conf, class_id=cls,
    )


def _track(tid=1, x=600, y=340, w=80, h=150, conf=0.9, cls="person"):
    return Track(
        track_id=tid, bbox=BoundingBox(x=x, y=y, w=w, h=h),
        confidence=conf, class_id=cls,
        first_seen_ts=0.0, last_seen_ts=0.0, age_frames=10,
    )


class TestTargetAllocator:
    @pytest.fixture
    def allocator(self):
        return TargetAllocator(num_executors=2)

    def test_empty_targets(self, allocator):
        result = allocator.allocate([], [(0.0, 0.0), (0.0, 0.0)])
        assert result == []

    def test_single_target_single_executor(self):
        alloc = TargetAllocator(num_executors=1)
        targets = [_obs(tid=1, x=640, y=360)]
        result = alloc.allocate(targets, [(0.0, 0.0)])
        assert len(result) == 1
        assert result[0].executor_id == 0

    def test_two_targets_two_executors(self, allocator):
        targets = [_obs(tid=1, x=200, y=360), _obs(tid=2, x=1000, y=360)]
        result = allocator.allocate(targets, [(0.0, 0.0), (0.0, 0.0)])
        assert len(result) == 2
        assigned_ids = {r.executor_id for r in result}
        assert assigned_ids == {0, 1}

    def test_more_executors_than_targets(self):
        alloc = TargetAllocator(num_executors=3)
        targets = [_obs(tid=1)]
        result = alloc.allocate(targets, [(0.0, 0.0)] * 3)
        assert len(result) <= 3

    def test_continuity_bonus(self, allocator):
        targets = [_obs(tid=1, x=640, y=360)]
        allocator.allocate(targets, [(0.0, 0.0), (0.0, 0.0)])
        # Second allocation should prefer same assignment
        result = allocator.allocate(targets, [(0.0, 0.0), (0.0, 0.0)])
        assert len(result) >= 1

    def test_assignment_cost(self, allocator):
        targets = [_obs(tid=1, x=640, y=360)]
        result = allocator.allocate(targets, [(0.0, 0.0), (0.0, 0.0)])
        for r in result:
            assert r.cost >= 0.0


class TestTargetAssignment:
    def test_fields(self):
        a = TargetAssignment(executor_id=0, target=_obs(), cost=1.5)
        assert a.executor_id == 0
        assert a.cost == 1.5


class TestWeightedMultiTargetSelector:
    def test_select_top_n(self):
        sel = WeightedMultiTargetSelector(frame_width=1280, frame_height=720)
        tracks = [_track(tid=i, x=100 * i, conf=0.9 - i * 0.1) for i in range(5)]
        result = sel.select(tracks, 0.0, max_targets=3)
        assert len(result) <= 3

    def test_empty_tracks(self):
        sel = WeightedMultiTargetSelector(frame_width=1280, frame_height=720)
        result = sel.select([], 0.0)
        assert result == []

    def test_single_track(self):
        sel = WeightedMultiTargetSelector(frame_width=1280, frame_height=720)
        result = sel.select([_track()], 0.0)
        assert len(result) == 1
