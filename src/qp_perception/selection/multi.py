"""Multi-target selection and allocation for coordinated multi-gimbal systems."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from qp_perception.config import SelectorConfig
from qp_perception.types import TargetObservation, Track

logger = logging.getLogger(__name__)


@dataclass
class TargetAssignment:
    """A target assigned to a specific executor (gimbal/weapon station)."""

    executor_id: int
    target: TargetObservation
    cost: float  # Assignment cost (e.g., angular distance)


class MultiTargetSelector:
    """Selects multiple targets from tracked objects.

    Unlike TargetSelector which returns a single target, this returns
    a ranked list of targets suitable for multi-gimbal coordination.
    """

    def select_multiple(
        self, tracks: list[Track], timestamp: float, max_targets: int = 3
    ) -> list[TargetObservation]:
        """Select up to max_targets best targets from tracks.

        Parameters
        ----------
        tracks : List[Track]
            Available tracked objects
        timestamp : float
            Current timestamp
        max_targets : int
            Maximum number of targets to return

        Returns
        -------
        List[TargetObservation]
            Ranked list of targets (best first), up to max_targets
        """
        ...


class TargetAllocator:
    """Allocates multiple targets to multiple executors using optimal assignment.

    Uses the Hungarian algorithm to minimize total assignment cost.
    Cost can be based on angular distance, priority, or other metrics.
    """

    def __init__(self, num_executors: int):
        """Initialize allocator.

        Parameters
        ----------
        num_executors : int
            Number of available executors (gimbals/weapon stations)
        """
        self.num_executors = num_executors
        self._last_assignments: list[int | None] = [None] * num_executors

    def allocate(
        self,
        targets: list[TargetObservation],
        executor_positions: list[tuple[float, float]],  # (yaw_deg, pitch_deg)
    ) -> list[TargetAssignment]:
        """Allocate targets to executors using Hungarian algorithm.

        Parameters
        ----------
        targets : List[TargetObservation]
            Available targets to assign
        executor_positions : List[Tuple[float, float]]
            Current position of each executor (yaw_deg, pitch_deg)

        Returns
        -------
        List[TargetAssignment]
            Assignments for each executor (may be empty if no suitable target)
        """
        if not targets:
            self._last_assignments = [None] * self.num_executors
            return []

        # Build cost matrix: executors x targets
        n_executors = len(executor_positions)
        n_targets = len(targets)

        cost_matrix = np.zeros((n_executors, n_targets))

        for i, (exec_yaw, exec_pitch) in enumerate(executor_positions):
            for j, target in enumerate(targets):
                # Cost = angular distance from executor to target
                # (simplified - assumes target bbox center maps to angle)
                # In real implementation, would use coordinate transform
                cost = self._compute_cost(exec_yaw, exec_pitch, target, i, j)
                cost_matrix[i, j] = cost

        # Solve assignment problem
        assignments = self._hungarian_algorithm(cost_matrix)

        # Build result
        result = []
        for executor_id, target_idx in enumerate(assignments):
            if target_idx is not None and target_idx < len(targets):
                result.append(
                    TargetAssignment(
                        executor_id=executor_id,
                        target=targets[target_idx],
                        cost=cost_matrix[executor_id, target_idx],
                    )
                )
                self._last_assignments[executor_id] = targets[target_idx].track_id
            else:
                self._last_assignments[executor_id] = None

        return result

    def _compute_cost(
        self,
        exec_yaw: float,
        exec_pitch: float,
        target: TargetObservation,
        executor_id: int,
        target_idx: int,
    ) -> float:
        """Compute assignment cost.

        Lower cost = better assignment.
        Factors:
        - Angular distance (primary)
        - Continuity bonus (prefer keeping same target)
        - Target confidence
        """
        # Simplified: use bbox center as proxy for angle
        # Real implementation would use coordinate transform
        cx, cy = target.bbox.center

        # Normalize to approximate angular error (very rough)
        # Assume 1280x720 image, ~60° FOV
        approx_yaw_error = (cx - 640) / 640 * 30  # degrees
        approx_pitch_error = (cy - 360) / 360 * 20  # degrees

        # Angular distance
        angular_dist = np.sqrt((approx_yaw_error) ** 2 + (approx_pitch_error) ** 2)

        cost = angular_dist

        # Continuity bonus: prefer keeping same target
        if self._last_assignments[executor_id] == target.track_id:
            cost *= 0.7  # 30% discount for continuity

        # Confidence bonus
        cost *= 2.0 - target.confidence  # Higher confidence = lower cost

        return cost

    def _hungarian_algorithm(self, cost_matrix: np.ndarray) -> list[int | None]:
        """Solve assignment problem using Hungarian algorithm.

        Parameters
        ----------
        cost_matrix : np.ndarray
            Shape (n_executors, n_targets), cost[i, j] = cost of assigning target j to executor i

        Returns
        -------
        List[Optional[int]]
            For each executor, the assigned target index (or None if unassigned)
        """
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            # Fallback: greedy assignment
            return self._greedy_assignment(cost_matrix)

        n_executors, n_targets = cost_matrix.shape

        if n_targets == 0:
            return [None] * n_executors

        # Pad matrix if needed (more executors than targets)
        if n_executors > n_targets:
            # Add dummy targets with high cost
            padding = np.full((n_executors, n_executors - n_targets), 1e6)
            cost_matrix_padded = np.hstack([cost_matrix, padding])
        else:
            cost_matrix_padded = cost_matrix

        # Solve
        row_ind, col_ind = linear_sum_assignment(cost_matrix_padded)

        # Build result
        assignments = [None] * n_executors
        for executor_id, target_idx in zip(row_ind, col_ind):
            if target_idx < n_targets:  # Not a dummy target
                # Only assign if cost is reasonable (not too far)
                if cost_matrix[executor_id, target_idx] < 50.0:  # Max 50° distance
                    assignments[executor_id] = target_idx

        return assignments

    def _greedy_assignment(self, cost_matrix: np.ndarray) -> list[int | None]:
        """Fallback greedy assignment (when scipy not available)."""
        n_executors, n_targets = cost_matrix.shape
        assignments = [None] * n_executors
        assigned_targets = set()

        # Sort all (executor, target) pairs by cost
        pairs = []
        for i in range(n_executors):
            for j in range(n_targets):
                pairs.append((cost_matrix[i, j], i, j))
        pairs.sort()

        # Greedily assign
        for cost, executor_id, target_idx in pairs:
            if executor_id not in [a for a in assignments if a is not None]:
                if target_idx not in assigned_targets:
                    if cost < 50.0:  # Max distance threshold
                        assignments[executor_id] = target_idx
                        assigned_targets.add(target_idx)

        return assignments


class WeightedMultiTargetSelector:
    """Selects multiple targets using weighted scoring.

    Scores each track based on:
    - Confidence
    - Size (bbox area)
    - Distance from center
    - Age (track stability)
    - Class preference

    Returns top N targets sorted by score.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        config: SelectorConfig | None = None,
    ):
        self._w = frame_width
        self._h = frame_height
        self._cfg = config if config is not None else SelectorConfig()
        self._last_selected_ids: list[int] = []

    def select_multiple(
        self, tracks: list[Track], timestamp: float, max_targets: int = 3
    ) -> list[TargetObservation]:
        """Select up to max_targets best targets.

        Parameters
        ----------
        tracks : List[Track]
            Available tracked objects
        timestamp : float
            Current timestamp
        max_targets : int
            Maximum number of targets to return

        Returns
        -------
        List[TargetObservation]
            Ranked list of targets (best first)
        """
        if not tracks:
            self._last_selected_ids = []
            return []

        # Score all tracks
        scored = []
        for track in tracks:
            score = self._compute_score(track, timestamp)
            scored.append((score, track))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Take top N
        selected = []
        for _score, track in scored[:max_targets]:
            obs = TargetObservation(
                timestamp=timestamp,
                track_id=track.track_id,
                bbox=track.bbox,
                confidence=track.confidence,
                class_id=track.class_id,
                velocity_px_per_s=track.velocity_px_per_s,
                acceleration_px_per_s2=track.acceleration_px_per_s2,
                mask_center=track.mask_center,
            )
            selected.append(obs)

        # Update tracking
        self._last_selected_ids = [obs.track_id for obs in selected]

        logger.debug(
            "Selected %d targets: %s (scores: %s)",
            len(selected),
            [obs.track_id for obs in selected],
            [f"{score:.2f}" for score, _ in scored[:max_targets]],
        )

        return selected

    # Backwards-compatible alias
    def select(self, tracks: list[Track], timestamp: float, max_targets: int = 3) -> list[TargetObservation]:
        return self.select_multiple(tracks, timestamp, max_targets=max_targets)

    def _compute_score(self, track: Track, timestamp: float) -> float:
        """Compute weighted score for a track."""
        cx, cy = track.bbox.center
        area = track.bbox.area

        # 1. Confidence weight
        conf_score = track.confidence

        # 2. Size weight (normalized)
        max_area = self._w * self._h
        size_score = min(area / max_area, 1.0)

        # 3. Center distance weight (closer to center = higher score)
        dx = (cx - self._w / 2) / (self._w / 2)
        dy = (cy - self._h / 2) / (self._h / 2)
        dist = (dx**2 + dy**2) ** 0.5
        center_score = max(0.0, 1.0 - dist)

        # 4. Age weight (older tracks = more stable)
        age_norm = min(track.age_frames / self._cfg.age_norm_frames, 1.0)
        age_score = age_norm

        # 5. Class preference
        preferred = self._cfg.preferred_classes or {}
        class_bonus = preferred.get(track.class_id, 0.5)

        # 6. Continuity bonus (was selected before)
        continuity_bonus = 1.2 if track.track_id in self._last_selected_ids else 1.0

        # Weighted combination using SelectorWeights
        weights = self._cfg.weights
        score = (
            conf_score * weights.confidence
            + size_score * weights.size
            + center_score * weights.center_proximity
            + age_score * weights.track_age
            + class_bonus * weights.class_weight
        ) * continuity_bonus

        return score
