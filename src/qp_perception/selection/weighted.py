"""Multi-target weighted scoring with anti-switch jitter."""

from __future__ import annotations

import logging
import math

from qp_perception.config import SelectorConfig
from qp_perception.types import TargetObservation, Track

logger = logging.getLogger(__name__)


class WeightedTargetSelector:
    def __init__(self, frame_width: int, frame_height: int, config: SelectorConfig | None = None) -> None:
        self._w = frame_width
        self._h = frame_height
        self._cfg = config if config is not None else SelectorConfig()
        self._current_id: int | None = None
        self._current_since = 0.0
        self._last_score = 0.0

    @staticmethod
    def _effective_center(track: Track) -> tuple[float, float]:
        """Return mask centroid if available, otherwise bbox center."""
        if track.mask_center is not None:
            return track.mask_center
        return track.bbox.center

    def _score(self, track: Track) -> float:
        weights = self._cfg.weights
        cx, cy = self._effective_center(track)
        nx = abs(cx - self._w * 0.5) / max(self._w * 0.5, 1.0)
        ny = abs(cy - self._h * 0.5) / max(self._h * 0.5, 1.0)
        center_proximity = max(1.0 - (nx + ny) * 0.5, 0.0)
        size = min(track.bbox.area / max(self._w * self._h, 1.0), 1.0)
        age_norm = min(track.age_frames / max(self._cfg.age_norm_frames, 1), 1.0)
        class_bonus = self._cfg.class_weights().get(track.class_id, 0.0)

        # Velocity approach score: reward targets moving toward frame center.
        # A target closing on the crosshair is a higher threat and easier to lock.
        # Score = 0 (moving away) .. 0.5 (stationary) .. 1.0 (approaching fast).
        vx, vy = track.velocity_px_per_s
        speed = math.sqrt(vx * vx + vy * vy)
        if speed > 1.0:
            dx = self._w * 0.5 - cx
            dy = self._h * 0.5 - cy
            dist_to_center = math.sqrt(dx * dx + dy * dy) + 1e-6
            cos_theta = (vx * dx + vy * dy) / (speed * dist_to_center)
            # Normalise speed: half-frame-size/s counts as "fast"
            speed_norm = min(speed / (max(self._w, self._h) * 0.5), 1.0)
            approach = (cos_theta + 1.0) * 0.5 * speed_norm
        else:
            approach = 0.0

        # Stability penalty: a track with recent missed detections (misses > 0)
        # is partially occluded or uncertain.  Decay the raw score so that a
        # fully-confirmed track is preferred over an equally-scored occluded one.
        # misses=0 → no penalty; misses=3 (lost_patience) → ~22% penalty.
        stability = 1.0 / (1.0 + track.misses * 0.08)

        score = stability * (
            weights.confidence * track.confidence
            + weights.size * size
            + weights.center_proximity * center_proximity
            + weights.track_age * age_norm
            + weights.class_weight * class_bonus
            + weights.velocity_approach * approach
        )
        if self._current_id is not None and track.track_id != self._current_id:
            score -= weights.switch_penalty
        return score

    def select(self, tracks: list[Track], timestamp: float) -> TargetObservation | None:
        if not tracks:
            self._current_id = None
            self._last_score = 0.0
            return None

        scored = [(self._score(t), t) for t in tracks]
        ranked = sorted(scored, key=lambda x: x[0], reverse=True)
        top_score, top_track = ranked[0]

        if self._current_id is None:
            self._current_id = top_track.track_id
            self._current_since = timestamp
            self._last_score = top_score
        elif top_track.track_id != self._current_id:
            hold_elapsed = timestamp - self._current_since
            should_switch = hold_elapsed >= self._cfg.min_hold_time_s and (
                top_score >= self._last_score + self._cfg.delta_threshold
            )
            if should_switch:
                logger.debug(
                    "target switch: ID %s -> %s (score %.3f -> %.3f, hold %.2fs)",
                    self._current_id,
                    top_track.track_id,
                    self._last_score,
                    top_score,
                    hold_elapsed,
                )
                self._current_id = top_track.track_id
                self._current_since = timestamp
                self._last_score = top_score

        # Look up cached score for the selected track instead of re-computing
        score_map = {t.track_id: s for s, t in scored}
        selected = next((t for t in tracks if t.track_id == self._current_id), top_track)
        self._last_score = score_map.get(selected.track_id, top_score)
        return TargetObservation(
            timestamp=timestamp,
            track_id=selected.track_id,
            bbox=selected.bbox,
            confidence=selected.confidence,
            class_id=selected.class_id,
            velocity_px_per_s=selected.velocity_px_per_s,
            acceleration_px_per_s2=selected.acceleration_px_per_s2,
            mask_center=selected.mask_center,
        )
