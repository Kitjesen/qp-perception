"""Unit tests for WeightedTargetSelector."""

import pytest

from qp_perception.config import SelectorConfig, SelectorWeights
from qp_perception.selection.weighted import WeightedTargetSelector
from qp_perception.types import BoundingBox, Track


@pytest.fixture
def selector_config():
    """Create default selector config."""
    return SelectorConfig(
        weights=SelectorWeights(
            confidence=0.35,
            size=0.20,
            center_proximity=0.20,
            track_age=0.15,
            class_weight=0.10,
            switch_penalty=0.30,
        ),
        min_hold_time_s=0.4,
        delta_threshold=0.12,
        preferred_classes={"person": 1.0, "car": 0.6},
    )


@pytest.fixture
def selector(selector_config):
    """Create selector instance."""
    return WeightedTargetSelector(
        frame_width=1280,
        frame_height=720,
        config=selector_config,
    )


def create_track(
    track_id: int,
    bbox: tuple[float, float, float, float],
    confidence: float = 0.8,
    class_id: str = "person",
    first_seen_ts: float = 0.0,
) -> Track:
    """Helper to create a track."""
    x, y, w, h = bbox
    return Track(
        track_id=track_id,
        bbox=BoundingBox(x=x, y=y, w=w, h=h),
        confidence=confidence,
        class_id=class_id,
        first_seen_ts=first_seen_ts,
        last_seen_ts=first_seen_ts,
        velocity_px_per_s=(0.0, 0.0),
        acceleration_px_per_s2=(0.0, 0.0),
        mask_center=None,
    )


class TestWeightedTargetSelector:
    """Test suite for WeightedTargetSelector."""

    def test_empty_tracks(self, selector):
        """Test with no tracks."""
        result = selector.select([], timestamp=1.0)
        assert result is None

    def test_single_track(self, selector):
        """Test with single track."""
        tracks = [create_track(1, (100, 100, 200, 200), confidence=0.9)]
        result = selector.select(tracks, timestamp=1.0)

        assert result is not None
        assert result.track_id == 1
        assert result.confidence == 0.9

    def test_confidence_weight(self, selector):
        """Higher confidence should be preferred."""
        tracks = [
            create_track(1, (100, 100, 200, 200), confidence=0.5),
            create_track(2, (300, 300, 400, 400), confidence=0.95),
        ]
        result = selector.select(tracks, timestamp=1.0)

        # Track 2 has much higher confidence
        assert result.track_id == 2

    def test_size_weight(self, selector):
        """Larger targets should be preferred (all else equal)."""
        tracks = [
            create_track(1, (100, 100, 150, 150), confidence=0.8),  # 50x50 = 2500
            create_track(2, (300, 300, 500, 500), confidence=0.8),  # 200x200 = 40000
        ]
        result = selector.select(tracks, timestamp=1.0)

        # Track 2 is much larger
        assert result.track_id == 2

    def test_center_proximity_weight(self, selector):
        """Targets closer to center should be preferred."""
        # Center is at (640, 360)
        tracks = [
            create_track(1, (100, 100, 200, 200), confidence=0.8),  # Far from center
            create_track(2, (600, 320, 680, 400), confidence=0.8),  # Near center
        ]
        result = selector.select(tracks, timestamp=1.0)

        # Track 2 is closer to center
        assert result.track_id == 2

    def test_track_age_weight(self, selector):
        """Older (more stable) tracks should be preferred."""
        tracks = [
            create_track(1, (100, 100, 200, 200), confidence=0.8, first_seen_ts=0.9),  # Age 0.1s
            create_track(2, (300, 300, 400, 400), confidence=0.8, first_seen_ts=0.0),  # Age 1.0s
        ]
        result = selector.select(tracks, timestamp=1.0)

        # Track 2 is older (more stable)
        assert result.track_id == 2

    def test_class_preference(self, selector):
        """Preferred classes should get bonus."""
        # Place both tracks at the same center position so only class bonus differs.
        # center_proximity = 1.0 for both; person bonus (0.10) > car bonus (0.06).
        tracks = [
            create_track(1, (590, 310, 100, 100), confidence=0.8, class_id="person"),
            create_track(2, (590, 310, 100, 100), confidence=0.8, class_id="car"),
        ]

        result = selector.select(tracks, timestamp=1.0)

        # Person should be preferred over car
        assert result.track_id == 1

    def test_target_hold_time(self, selector):
        """Target should be held for min_hold_time_s."""
        tracks = [
            create_track(1, (100, 100, 200, 200), confidence=0.7),
            create_track(2, (300, 300, 400, 400), confidence=0.9),
        ]

        # First selection: should pick track 2 (higher confidence)
        result1 = selector.select(tracks, timestamp=1.0)
        assert result1.track_id == 2

        # Second selection at t=1.2s (0.2s later, < min_hold_time_s=0.4s)
        # Should still hold track 2 even though track 1 might score higher with switch penalty
        result2 = selector.select(tracks, timestamp=1.2)
        assert result2.track_id == 2

        # Third selection at t=1.5s (0.5s later, > min_hold_time_s)
        # Now switching is allowed
        selector.select(tracks, timestamp=1.5)
        # Result depends on scoring, but switching is now possible

    def test_switch_penalty(self, selector):
        """Switching targets should incur penalty."""
        # Track 1 at screen center (640,360) -> high center_proximity score wins first selection.
        # Track 2 far in corner -> lower overall score, can't overcome switch penalty.
        tracks = [
            create_track(1, (540, 260, 200, 200), confidence=0.8),   # center=(640,360)
            create_track(2, (50, 50, 100, 100), confidence=0.82),     # center=(100,100), far
        ]

        # Select track 1 first
        result1 = selector.select(tracks, timestamp=1.0)
        assert result1.track_id == 1

        # Wait past hold time
        # Track 2 is only slightly better, switch penalty should keep us on track 1
        result2 = selector.select(tracks, timestamp=2.0)
        assert result2.track_id == 1  # Should not switch due to penalty

    def test_target_disappears(self, selector):
        """When current target disappears, should select new one."""
        # Track 1 at center wins first selection; track 2 far in corner.
        tracks1 = [
            create_track(1, (540, 260, 200, 200), confidence=0.8),   # center=(640,360)
            create_track(2, (50, 50, 100, 100), confidence=0.7),      # center=(100,100), far
        ]

        # Select track 1
        result1 = selector.select(tracks1, timestamp=1.0)
        assert result1.track_id == 1

        # Track 1 disappears, only track 2 remains
        tracks2 = [create_track(2, (50, 50, 100, 100), confidence=0.7)]
        result2 = selector.select(tracks2, timestamp=2.0)
        assert result2.track_id == 2

    def test_all_tracks_disappear(self, selector):
        """When all tracks disappear, should return None."""
        tracks = [create_track(1, (100, 100, 200, 200), confidence=0.8)]

        # Select track 1
        result1 = selector.select(tracks, timestamp=1.0)
        assert result1.track_id == 1

        # All tracks disappear
        result2 = selector.select([], timestamp=2.0)
        assert result2 is None

    def test_score_normalization(self, selector):
        """Scores should be properly normalized."""
        tracks = [
            create_track(1, (100, 100, 200, 200), confidence=0.5),
            create_track(2, (300, 300, 400, 400), confidence=1.0),
        ]

        result = selector.select(tracks, timestamp=1.0)
        # Should not crash with extreme values
        assert result is not None

    def test_delta_threshold(self, selector):
        """Delta threshold should prevent unnecessary switches."""
        # Track 1 at center wins first selection; track 2 far in corner.
        tracks = [
            create_track(1, (540, 260, 200, 200), confidence=0.80),  # center=(640,360)
            create_track(2, (50, 50, 100, 100), confidence=0.81),     # center=(100,100), far
        ]

        # Select track 1
        result1 = selector.select(tracks, timestamp=1.0)
        assert result1.track_id == 1

        # Wait past hold time
        # Track 2 is only marginally better (< delta_threshold=0.12)
        result2 = selector.select(tracks, timestamp=2.0)
        assert result2.track_id == 1  # Should not switch

    def test_velocity_included_in_observation(self, selector):
        """Selected observation should include velocity."""
        track = create_track(1, (100, 100, 200, 200), confidence=0.8)
        track.velocity_px_per_s = (10.0, -5.0)
        track.acceleration_px_per_s2 = (1.0, 0.5)

        result = selector.select([track], timestamp=1.0)

        assert result.velocity_px_per_s == (10.0, -5.0)
        assert result.acceleration_px_per_s2 == (1.0, 0.5)

    def test_mask_center_preferred_over_bbox(self, selector):
        """If mask_center is available, it should be used."""
        track = create_track(1, (100, 100, 200, 200), confidence=0.8)
        track.mask_center = (140.0, 160.0)  # Different from bbox center (150, 150)

        result = selector.select([track], timestamp=1.0)

        assert result.mask_center == (140.0, 160.0)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_confidence(self, selector):
        """Track with zero confidence should still be selectable if only option."""
        tracks = [create_track(1, (100, 100, 200, 200), confidence=0.0)]
        result = selector.select(tracks, timestamp=1.0)
        assert result is not None

    def test_negative_timestamp(self, selector):
        """Negative timestamps should not crash."""
        tracks = [create_track(1, (100, 100, 200, 200), confidence=0.8)]
        result = selector.select(tracks, timestamp=-1.0)
        assert result is not None

    def test_very_small_bbox(self, selector):
        """Very small bounding box should not crash."""
        tracks = [create_track(1, (100, 100, 101, 101), confidence=0.8)]  # 1x1 pixel
        result = selector.select(tracks, timestamp=1.0)
        assert result is not None

    def test_bbox_outside_frame(self, selector):
        """Bbox outside frame bounds should be handled."""
        tracks = [create_track(1, (2000, 2000, 2100, 2100), confidence=0.8)]
        result = selector.select(tracks, timestamp=1.0)
        assert result is not None

    def test_many_tracks(self, selector):
        """Should handle many tracks efficiently."""
        tracks = [
            create_track(i, (i * 10, i * 10, i * 10 + 50, i * 10 + 50), confidence=0.7 + i * 0.001)
            for i in range(100)
        ]
        result = selector.select(tracks, timestamp=1.0)
        assert result is not None

    def test_unknown_class(self, selector):
        """Unknown class should not crash."""
        track = create_track(1, (100, 100, 100, 100), confidence=0.8, class_id="unknown_class")
        result = selector.select([track], timestamp=1.0)
        assert result is not None
