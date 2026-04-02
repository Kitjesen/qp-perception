"""Person Following Selector — lock-on tracking for robot following scenarios.

Unlike WeightedTargetSelector (which picks the "best" target each frame),
PersonFollowingSelector locks onto a specific person by track_id and holds
that lock until the person is truly lost.

Lock acquisition methods (in priority order):
  1. lock_track(track_id)           — direct lock by FusionMOT track ID
  2. lock_by_crop_similarity(...)   — CLIP/Re-ID feature match from crops
  3. Auto-lock highest-confidence person when no explicit lock is set

Lock lifecycle:
  UNLOCKED → lock_track() or auto-lock → LOCKED
  LOCKED   → target visible each frame → LOCKED (stable)
  LOCKED   → target missing → SEARCHING (FusionMOT Re-ID recovery window)
  SEARCHING → target reappears (same track_id) → LOCKED
  SEARCHING → timeout exceeded → LOST (needs VLM re-selection)
  LOST     → lock_track() → LOCKED (re-acquired)

The select() method always returns the locked target when visible,
regardless of whether other targets score higher.  This prevents
the jitter problem where a following robot keeps switching between
nearby people.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from qp_perception.types import TargetObservation, Track

logger = logging.getLogger(__name__)


@dataclass
class FollowingConfig:
    """Configuration for person following selector.

    Attributes
    ----------
    search_timeout_s : float
        How long to wait for a lost track to reappear (via FusionMOT Re-ID)
        before declaring LOST and requiring VLM re-selection.
    auto_lock : bool
        If True, automatically lock onto the highest-confidence person
        track when no explicit lock is set.  Useful for "follow nearest
        person" without VLM selection.
    min_confidence : float
        Minimum detection confidence to accept for auto-lock.
    person_classes : frozenset[str]
        Class labels considered as "person" for filtering.
    """

    search_timeout_s: float = 8.0
    auto_lock: bool = True
    min_confidence: float = 0.3
    person_classes: frozenset[str] = frozenset(
        {"person", "people", "human", "pedestrian"}
    )


class PersonFollowingSelector:
    """Target selector that locks onto one person for continuous following.

    Implements the ``TargetSelector`` protocol::

        selector = PersonFollowingSelector()
        selector.lock_track(42)  # lock onto FusionMOT track #42
        obs = selector.select(tracks, timestamp)  # always returns track #42

    Designed to work with FusionMOT's stable track IDs — when a person is
    occluded briefly, FusionMOT maintains the track_id through Kalman
    prediction + Re-ID recovery, so the selector just keeps returning the
    same ID without any logic duplication.
    """

    STATE_UNLOCKED = "unlocked"
    STATE_LOCKED = "locked"
    STATE_SEARCHING = "searching"
    STATE_LOST = "lost"

    def __init__(self, config: FollowingConfig | None = None) -> None:
        self._cfg = config or FollowingConfig()
        self._locked_id: int | None = None
        self._state: str = self.STATE_UNLOCKED
        self._last_seen_ts: float = 0.0
        self._lock_ts: float = 0.0
        self._description: str = ""

    # ------------------------------------------------------------------
    # Lock management
    # ------------------------------------------------------------------

    def lock_track(self, track_id: int, description: str = "") -> None:
        """Explicitly lock onto a FusionMOT track ID."""
        self._locked_id = track_id
        self._state = self.STATE_LOCKED
        self._last_seen_ts = time.time()
        self._lock_ts = self._last_seen_ts
        self._description = description
        logger.info(
            "PersonFollowing: locked track_id=%d desc='%s'",
            track_id, description,
        )

    def lock_by_crop_similarity(
        self,
        tracks: list[Track],
        query_feature,
        reid_extractor=None,
        frame=None,
    ) -> int | None:
        """Lock onto the track whose crop is most similar to a query feature.

        Useful when VLM has selected a person crop and we need to find the
        corresponding FusionMOT track ID.

        Parameters
        ----------
        tracks : list[Track]
            Current active tracks from FusionMOT.
        query_feature : np.ndarray
            L2-normalized feature vector of the target person.
        reid_extractor : ReIDExtractor, optional
            For extracting features from track crops.
        frame : np.ndarray, optional
            Current BGR frame for crop extraction.

        Returns
        -------
        int or None
            Locked track_id, or None if no match found.
        """
        if reid_extractor is None or frame is None or not tracks:
            return None

        import numpy as np

        person_tracks = self._filter_persons(tracks)
        if not person_tracks:
            return None

        bbox_tuples = [
            (t.bbox.x, t.bbox.y, t.bbox.w, t.bbox.h) for t in person_tracks
        ]
        try:
            features = reid_extractor.extract(frame, bbox_tuples)
        except Exception as e:
            logger.debug("Crop similarity extraction failed: %s", e)
            return None

        if len(features) == 0:
            return None

        # Cosine similarity (features are L2-normalized)
        sims = features @ query_feature
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim < 0.2:
            logger.debug("No similar person found (best=%.3f)", best_sim)
            return None

        best_track = person_tracks[best_idx]
        self.lock_track(best_track.track_id, description=f"sim={best_sim:.2f}")
        return best_track.track_id

    def unlock(self) -> None:
        """Release the current lock."""
        if self._locked_id is not None:
            logger.info("PersonFollowing: unlocked track_id=%d", self._locked_id)
        self._locked_id = None
        self._state = self.STATE_UNLOCKED
        self._description = ""

    # ------------------------------------------------------------------
    # TargetSelector protocol
    # ------------------------------------------------------------------

    def select(
        self, tracks: list[Track], timestamp: float,
    ) -> TargetObservation | None:
        """Select the locked person target from active tracks.

        State machine:
          UNLOCKED  → auto_lock if enabled, else return None
          LOCKED    → return locked track (or transition to SEARCHING)
          SEARCHING → return None, wait for Re-ID recovery (or → LOST)
          LOST      → return None, needs external re-lock
        """
        person_tracks = self._filter_persons(tracks)

        # ── UNLOCKED: auto-lock or wait ──
        if self._state == self.STATE_UNLOCKED:
            if not self._cfg.auto_lock or not person_tracks:
                return None
            best = max(person_tracks, key=lambda t: t.confidence)
            if best.confidence >= self._cfg.min_confidence:
                self.lock_track(best.track_id, description="auto")
                return self._to_observation(best, timestamp)
            return None

        # ── LOST: needs external re-lock ──
        if self._state == self.STATE_LOST:
            return None

        # ── LOCKED / SEARCHING: find locked track ──
        locked = self._find_locked(person_tracks)

        if locked is not None:
            # Target visible — stay/return to LOCKED
            self._state = self.STATE_LOCKED
            self._last_seen_ts = timestamp
            return self._to_observation(locked, timestamp)

        # Target not in current tracks
        if self._state == self.STATE_LOCKED:
            self._state = self.STATE_SEARCHING
            logger.debug(
                "PersonFollowing: track_id=%d missing, searching...",
                self._locked_id,
            )

        # SEARCHING: check timeout
        elapsed = timestamp - self._last_seen_ts
        if elapsed > self._cfg.search_timeout_s:
            self._state = self.STATE_LOST
            logger.info(
                "PersonFollowing: track_id=%d lost after %.1fs",
                self._locked_id, elapsed,
            )

        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def locked_track_id(self) -> int | None:
        return self._locked_id

    @property
    def description(self) -> str:
        return self._description

    @property
    def is_locked(self) -> bool:
        return self._state == self.STATE_LOCKED

    @property
    def is_lost(self) -> bool:
        return self._state == self.STATE_LOST

    @property
    def needs_reselect(self) -> bool:
        """True when target is lost and VLM re-selection is needed."""
        return self._state == self.STATE_LOST

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _filter_persons(self, tracks: list[Track]) -> list[Track]:
        """Filter tracks to person-class only."""
        classes = self._cfg.person_classes
        return [t for t in tracks if t.class_id.lower() in classes]

    def _find_locked(self, tracks: list[Track]) -> Track | None:
        """Find the locked track by ID."""
        if self._locked_id is None:
            return None
        for t in tracks:
            if t.track_id == self._locked_id:
                return t
        return None

    @staticmethod
    def _to_observation(track: Track, timestamp: float) -> TargetObservation:
        return TargetObservation(
            timestamp=timestamp,
            track_id=track.track_id,
            bbox=track.bbox,
            confidence=track.confidence,
            class_id=track.class_id,
            velocity_px_per_s=track.velocity_px_per_s,
            acceleration_px_per_s2=track.acceleration_px_per_s2,
            mask_center=track.mask_center,
        )
