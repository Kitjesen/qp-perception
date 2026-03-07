"""
Appearance Gallery for Re-ID based track recovery.
=====================================================

Maintains per-track appearance feature vectors and provides matching logic
for recovering track IDs after occlusion.

Two pools:
    - **Active pool**: Features of currently visible tracks, updated via EMA
      every frame to adapt to appearance changes (lighting, pose).
    - **Lost pool**: When a track disappears, its feature moves here and is
      kept for ``max_lost_age`` seconds.  New detections are matched against
      this pool to recover the original track ID.

Matching uses cosine similarity on L2-normalized 576-dim feature vectors.
A match is accepted only when similarity exceeds ``match_threshold``.
When multiple lost tracks could match, the highest similarity wins,
subject to a second-best margin check to avoid ambiguous matches.

This module is intentionally stateless regarding tracking mechanics — it
only manages feature vectors and returns match decisions.  The actual
ID remapping is performed by the tracker that calls into this gallery.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GalleryConfig:
    """Configuration for the appearance gallery.

    Attributes
    ----------
    match_threshold : float
        Minimum cosine similarity to accept a Re-ID match.
    match_threshold_relaxed : float
        Relaxed threshold for recently lost tracks (< ``cascade_recent_s``).
        Recent losses are more likely the same person, so we can be lenient.
    cascade_recent_s : float
        Tracks lost for less than this duration use ``match_threshold_relaxed``.
        Older lost tracks use the stricter ``match_threshold``.
    ema_alpha : float
        Exponential moving average weight for feature updates.
        0.9 → slow adaptation (stable appearance), 0.1 → fast adaptation.
    max_lost_age : float
        Maximum seconds to keep a lost track in the gallery.
    second_best_margin : float
        Minimum gap between best and second-best match similarity.
        Prevents ambiguous 1-to-many matches.
    min_track_age_frames : int
        Minimum number of frames a track must be visible before its feature
        is considered reliable enough for Re-ID.
    spatial_gate_px : float
        Maximum pixel distance between predicted lost position and new
        detection center.  Pairs exceeding this are rejected regardless of
        cosine similarity.  Set to 0 to disable spatial gating.
    spatial_gate_grow_rate : float
        The spatial gate expands over time: effective radius =
        ``spatial_gate_px + spatial_gate_grow_rate * lost_duration``.
        Accounts for increasing uncertainty the longer a track is lost.
    appearance_weight : float
        Weight of appearance similarity in the fused score.
    motion_weight : float
        Weight of motion consistency in the fused score.
    iou_weight : float
        Weight of predicted-box IoU in the fused score.
    min_fused_score : float
        Minimum fused score required to accept a Re-ID match.
    short_ema_alpha : float
        EMA alpha for short-term prototype (fast adaptation).
    long_ema_alpha : float
        EMA alpha for long-term prototype (stable identity memory).
    uncertainty_app_boost : float
        How much to increase appearance weight when motion residual is high.
    dir_consistency_min_cos : float
        Minimum cosine between lost velocity and query velocity to accept
        high-motion matches. Set to -1 to disable.
    direction_gate_min_speed : float
        Direction gate is only applied when both speeds exceed this value.
    scale_ratio_min : float
        Minimum query/lost size ratio for scale consistency gate.
    scale_ratio_max : float
        Maximum query/lost size ratio for scale consistency gate.
    """

    match_threshold: float = 0.35
    match_threshold_relaxed: float = 0.28
    cascade_recent_s: float = 1.5
    ema_alpha: float = 0.85
    max_lost_age: float = 5.0
    second_best_margin: float = 0.04
    min_track_age_frames: int = 3
    spatial_gate_px: float = 300.0
    spatial_gate_grow_rate: float = 150.0
    appearance_weight: float = 0.60
    motion_weight: float = 0.25
    iou_weight: float = 0.15
    min_fused_score: float = 0.35
    short_ema_alpha: float = 0.65
    long_ema_alpha: float = 0.92
    uncertainty_app_boost: float = 0.25
    dir_consistency_min_cos: float = -0.35
    direction_gate_min_speed: float = 40.0
    scale_ratio_min: float = 0.55
    scale_ratio_max: float = 1.90

    # --- Deep OC-SORT: Dynamic Appearance (DA) ---
    da_alpha_fixed: float = 0.95
    da_confidence_sigma: float = 0.40

    # --- Deep OC-SORT: Adaptive Weighting (AW) ---
    aw_epsilon: float = 0.5
    aw_base_weight: float = 0.55

    # --- OC-SORT: Observation-Centric Momentum (OCM) ---
    ocm_window: int = 5

    # --- Temporal Feature Bank (multi-frame aggregation) ---
    feature_bank_size: int = 10
    feature_bank_query_top_k: int = 5

    # --- Hybrid-SORT weak cues ---
    height_weight: float = 0.10


@dataclass
class _ActiveEntry:
    """Feature state for a currently visible track."""

    feature_short: np.ndarray  # short-term prototype
    feature_long: np.ndarray  # long-term prototype
    feature_bank: list = field(default_factory=list)  # temporal bank: [(feat, conf, ts), ...]
    frames_seen: int = 1
    last_ts: float = 0.0
    recent_obs: list = field(default_factory=list)  # OCM: [(x, y, ts), ...]
    height_history: list = field(default_factory=list)  # Hybrid-SORT: stable height cue


@dataclass
class _LostEntry:
    """Feature state for a recently disappeared track."""

    feature_short: np.ndarray
    feature_long: np.ndarray
    feature_bank: list = field(default_factory=list)  # [(feat, conf, ts), ...]
    lost_ts: float = 0.0
    last_position: tuple[float, float] = (0.0, 0.0)
    last_velocity: tuple[float, float] = (0.0, 0.0)
    last_size: tuple[float, float] = (0.0, 0.0)
    median_height: float = 0.0  # Hybrid-SORT: stable height cue


@dataclass
class ReIDMatch:
    """Result of a Re-ID gallery match query."""

    old_track_id: int
    similarity: float
    fused_score: float = 0.0
    position_pred: tuple[float, float] = (0.0, 0.0)


class AppearanceGallery:
    """Manages appearance features for active and lost tracks.

    Usage flow (called by the tracker each frame)::

        # 1. Update active features for all currently visible tracks
        gallery.update_active(track_id, feature, timestamp)

        # 2. For "new" tracks (first frame), query lost gallery
        match = gallery.query_lost(new_feature, timestamp)
        if match is not None:
            remap new_track_id → match.old_track_id

        # 3. After processing all tracks, retire disappeared ones
        gallery.retire_missing(current_track_ids, timestamp)

        # 4. Purge expired lost entries
        gallery.purge_expired(timestamp)
    """

    def __init__(self, config: GalleryConfig | None = None) -> None:
        self._cfg = config or GalleryConfig()
        self._active: dict[int, _ActiveEntry] = {}
        self._lost: dict[int, _LostEntry] = {}

        logger.info(
            "AppearanceGallery ready  threshold=%.2f  ema=%.2f  max_lost=%.1fs",
            self._cfg.match_threshold,
            self._cfg.ema_alpha,
            self._cfg.max_lost_age,
        )

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def lost_count(self) -> int:
        return len(self._lost)

    def _dynamic_alpha(self, base_alpha: float, confidence: float) -> float:
        """Deep OC-SORT Dynamic Appearance (Eq.3): confidence-dependent EMA alpha.

        Low confidence (near σ) → alpha→1 (reject noisy embedding from occluded crop).
        High confidence (near 1) → alpha→base_alpha (normal update).
        """
        sigma = self._cfg.da_confidence_sigma
        af = self._cfg.da_alpha_fixed
        if confidence <= sigma:
            return 1.0
        return af + (1.0 - af) * (1.0 - (confidence - sigma) / (1.0 - sigma))

    def update_active(
        self,
        track_id: int,
        feature: np.ndarray,
        timestamp: float,
        confidence: float = 1.0,
        position: tuple[float, float] | None = None,
        feature_decay: bool = False,
        bbox_height: float = 0.0,
    ) -> None:
        """Update (or create) the appearance feature for an active track.

        Uses **Dynamic Appearance** (Deep OC-SORT, Eq.2-3): EMA alpha is
        modulated by detection confidence.  Low-confidence detections (partial
        occlusion, blur) contribute less to the feature model, preventing
        "gallery pollution" that degrades Re-ID accuracy.

        **Feature Decay** (Fast-Deep-OC-SORT, Sec.3.4): When ``feature_decay``
        is True, the feature wasn't freshly extracted this frame (Selective ReID
        skipped extraction).  We still apply a "phantom" EMA step:
        ``α' ← α' × α`` — the old feature's weight decays exponentially, so
        stale features lose influence even without a new observation.

        **Temporal Feature Bank**: stores last K high-confidence features
        explicitly.  When the track is lost, the bank provides a robust
        multi-sample representation for Re-ID matching (inspired by
        SambaMOTR/MOTIP multi-frame context).

        **Hybrid-SORT height cue**: stores bbox height as a stable property
        that doesn't change during occlusion.
        """
        if track_id in self._active:
            entry = self._active[track_id]

            if feature_decay:
                a_s = self._cfg.short_ema_alpha
                a_l = self._cfg.long_ema_alpha
                entry.feature_short *= a_s
                entry.feature_long *= a_l
                n_s = np.linalg.norm(entry.feature_short)
                n_l = np.linalg.norm(entry.feature_long)
                if n_s > 1e-6:
                    entry.feature_short /= n_s
                if n_l > 1e-6:
                    entry.feature_long /= n_l
            else:
                a_s = self._dynamic_alpha(self._cfg.short_ema_alpha, confidence)
                a_l = self._dynamic_alpha(self._cfg.long_ema_alpha, confidence)
                entry.feature_short = a_s * entry.feature_short + (1.0 - a_s) * feature
                entry.feature_long = a_l * entry.feature_long + (1.0 - a_l) * feature
                n_s = np.linalg.norm(entry.feature_short)
                n_l = np.linalg.norm(entry.feature_long)
                if n_s > 1e-6:
                    entry.feature_short /= n_s
                if n_l > 1e-6:
                    entry.feature_long /= n_l

                # Temporal feature bank: store high-confidence features
                if confidence > self._cfg.da_confidence_sigma:
                    bank = entry.feature_bank
                    bank.append((feature.copy(), confidence, timestamp))
                    max_k = self._cfg.feature_bank_size
                    if len(bank) > max_k:
                        entry.feature_bank = bank[-max_k:]

            entry.frames_seen += 1
            entry.last_ts = timestamp
            if position is not None:
                entry.recent_obs.append((position[0], position[1], timestamp))
                if len(entry.recent_obs) > self._cfg.ocm_window:
                    entry.recent_obs = entry.recent_obs[-self._cfg.ocm_window:]
            if bbox_height > 1.0:
                entry.height_history.append(bbox_height)
                if len(entry.height_history) > 20:
                    entry.height_history = entry.height_history[-20:]
        else:
            feat = feature.copy()
            norm = np.linalg.norm(feat)
            if norm > 1e-6:
                feat /= norm
            obs: list = []
            if position is not None:
                obs = [(position[0], position[1], timestamp)]
            heights: list = []
            if bbox_height > 1.0:
                heights = [bbox_height]
            self._active[track_id] = _ActiveEntry(
                feature_short=feat.copy(),
                feature_long=feat.copy(),
                feature_bank=[(feature.copy(), confidence, timestamp)],
                frames_seen=1,
                last_ts=timestamp,
                recent_obs=obs,
                height_history=heights,
            )

    @staticmethod
    def _box_iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        """IoU for XYWH boxes."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 1e-6 else 0.0

    def query_lost(
        self,
        query_feature: np.ndarray,
        timestamp: float,
        position_hint: tuple[float, float] | None = None,
        bbox_hint: tuple[float, float, float, float] | None = None,
        query_velocity: tuple[float, float] | None = None,
    ) -> ReIDMatch | None:
        """Check if a new detection matches any lost track.

        Uses a two-tier cascade strategy:
        1. Compute cosine similarity against all lost entries.
        2. Apply spatial gating: predict where the lost target *should* be
           (linear extrapolation from last position + velocity), and reject
           matches whose predicted-vs-actual distance exceeds the gate radius.
        3. Apply cascade thresholds: recently lost tracks get a relaxed
           cosine threshold; older ones require stricter similarity.
        4. Enforce second-best margin to avoid ambiguous matches.

        Parameters
        ----------
        query_feature : np.ndarray
            L2-normalized feature vector of the new detection.
        timestamp : float
            Current time.
        position_hint : optional
            (cx, cy) of the new detection for spatial gating.

        Returns
        -------
        ReIDMatch or None
        """
        if not self._lost:
            return None

        lost_ids = list(self._lost.keys())
        # Build per-candidate scores accounting for spatial gate + cascade threshold
        candidates: list[tuple[int, float, float, float, float, float]] = []
        # (idx, fused, raw_sim, dt, pred_x, pred_y)
        fallback_candidates: list[tuple[int, float, float, float, float]] = []
        # (idx, raw_sim, dt, pred_x, pred_y)

        for idx, tid in enumerate(lost_ids):
            entry = self._lost[tid]
            dt = timestamp - entry.lost_ts
            sim_short = float(entry.feature_short @ query_feature)
            sim_long = float(entry.feature_long @ query_feature)
            sim = max(sim_short, 0.7 * sim_short + 0.3 * sim_long)

            # Cascade threshold: recent losses get relaxed cosine bar
            threshold = (
                self._cfg.match_threshold_relaxed
                if dt < self._cfg.cascade_recent_s
                else self._cfg.match_threshold
            )

            if sim < threshold:
                continue

            pred_x = entry.last_position[0] + entry.last_velocity[0] * dt
            pred_y = entry.last_position[1] + entry.last_velocity[1] * dt
            fallback_candidates.append((idx, sim, dt, pred_x, pred_y))

            # Spatial gating: reject if predicted position is too far from detection
            if position_hint is not None and self._cfg.spatial_gate_px > 0:
                gate_radius = self._cfg.spatial_gate_px + self._cfg.spatial_gate_grow_rate * dt
                dx = position_hint[0] - pred_x
                dy = position_hint[1] - pred_y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist > gate_radius:
                    logger.debug(
                        "Re-ID spatial reject: lost_id=%d  dist=%.0f > gate=%.0f  sim=%.3f",
                        tid, dist, gate_radius, sim,
                    )
                    continue

            gate_radius = self._cfg.spatial_gate_px + self._cfg.spatial_gate_grow_rate * dt
            motion_score = 1.0
            if position_hint is not None and self._cfg.spatial_gate_px > 0:
                dx = position_hint[0] - pred_x
                dy = position_hint[1] - pred_y
                dist = (dx * dx + dy * dy) ** 0.5
                motion_score = max(0.0, 1.0 - dist / max(gate_radius, 1e-6))
            else:
                dist = 0.0

            # Direction consistency gate (helps avoid over-merge after occlusion)
            if query_velocity is not None:
                lvx, lvy = entry.last_velocity
                qvx, qvy = query_velocity
                ls = (lvx * lvx + lvy * lvy) ** 0.5
                qs = (qvx * qvx + qvy * qvy) ** 0.5
                if ls > self._cfg.direction_gate_min_speed and qs > self._cfg.direction_gate_min_speed:
                    cos_dir = (lvx * qvx + lvy * qvy) / max(ls * qs, 1e-6)
                    if cos_dir < self._cfg.dir_consistency_min_cos:
                        continue

            iou_score = 0.0
            if bbox_hint is not None and entry.last_size[0] > 1.0 and entry.last_size[1] > 1.0:
                lw, lh = entry.last_size
                q_w, q_h = bbox_hint[2], bbox_hint[3]
                if q_w > 1.0 and q_h > 1.0:
                    ratio = (q_w * q_h) / max(lw * lh, 1e-6)
                    if ratio < self._cfg.scale_ratio_min or ratio > self._cfg.scale_ratio_max:
                        continue
                pred_box = (pred_x - lw * 0.5, pred_y - lh * 0.5, lw, lh)
                iou_score = self._box_iou_xywh(pred_box, bbox_hint)

            app_score = (sim + 1.0) * 0.5
            uncertainty = min(1.0, max(0.0, dist / max(gate_radius, 1e-6)))
            w_app = self._cfg.appearance_weight + self._cfg.uncertainty_app_boost * uncertainty
            w_motion = max(0.05, self._cfg.motion_weight * (1.0 - 0.6 * uncertainty))
            w_iou = self._cfg.iou_weight
            w_sum = w_app + w_motion + w_iou
            fused = (w_app * app_score + w_motion * motion_score + w_iou * iou_score) / max(w_sum, 1e-6)
            if fused < self._cfg.min_fused_score:
                continue

            candidates.append((idx, fused, sim, dt, pred_x, pred_y))

        if not candidates:
            if not fallback_candidates:
                return None
            fallback_candidates.sort(key=lambda c: -c[1])
            fb = fallback_candidates[0]
            if len(fallback_candidates) > 1 and (
                fb[1] - fallback_candidates[1][1] < self._cfg.second_best_margin
            ):
                return None
            fb_tid = lost_ids[fb[0]]
            logger.info(
                "Re-ID fallback match: lost_id=%d  sim=%.3f  lost_for=%.2fs",
                fb_tid,
                fb[1],
                fb[2],
            )
            del self._lost[fb_tid]
            return ReIDMatch(
                old_track_id=fb_tid,
                similarity=fb[1],
                fused_score=fb[1],
                position_pred=(fb[3], fb[4]),
            )

        # --- Deep OC-SORT: Adaptive Weighting (Eq.4-6) ---
        # Boost fused score for discriminative matches: if the gap between
        # the best and second-best *cosine similarity* is large, the appearance
        # signal is informative and deserves more weight.
        candidates.sort(key=lambda c: -c[2])  # sort by raw similarity for AW
        if len(candidates) >= 2:
            sim_diff = min(candidates[0][2] - candidates[1][2], self._cfg.aw_epsilon)
        else:
            sim_diff = self._cfg.aw_epsilon

        aw_boosted: list[tuple[int, float, float, float, float, float]] = []
        for idx, fused, sim, dt, px, py in candidates:
            aw_boost = sim_diff * self._cfg.aw_base_weight
            boosted_fused = fused + aw_boost * ((sim + 1.0) * 0.5)
            aw_boosted.append((idx, boosted_fused, sim, dt, px, py))

        aw_boosted.sort(key=lambda c: -c[1])
        best = aw_boosted[0]

        if len(aw_boosted) > 1:
            if best[1] - aw_boosted[1][1] < self._cfg.second_best_margin:
                logger.debug(
                    "Re-ID ambiguous after AW: best=%.3f second=%.3f (margin=%.3f < %.3f)",
                    best[1], aw_boosted[1][1],
                    best[1] - aw_boosted[1][1],
                    self._cfg.second_best_margin,
                )
                return None

        best_tid = lost_ids[best[0]]

        logger.info(
            "Re-ID match (AW): lost_id=%d  fused=%.3f  sim=%.3f  lost_for=%.2fs  aw_boost=%.3f",
            best_tid, best[1], best[2], best[3], sim_diff * self._cfg.aw_base_weight,
        )

        del self._lost[best_tid]

        return ReIDMatch(
            old_track_id=best_tid,
            similarity=best[2],
            fused_score=best[1],
            position_pred=(best[4], best[5]),
        )

    @staticmethod
    def _ocm_velocity(obs: list) -> tuple[float, float]:
        """Observation-Centric Momentum: compute velocity from raw observations.

        OC-SORT (Sec.3, OCM) shows that Kalman-predicted velocity drifts
        quadratically during occlusion.  Using the last Δt raw detections'
        angular velocity is far more robust for extrapolation.

        Returns (vx, vy) in pixels/second computed from the first and last
        entries of the observation window.
        """
        if len(obs) < 2:
            return (0.0, 0.0)
        x0, y0, t0 = obs[0]
        x1, y1, t1 = obs[-1]
        dt = t1 - t0
        if dt < 1e-4:
            return (0.0, 0.0)
        return ((x1 - x0) / dt, (y1 - y0) / dt)

    def retire_missing(
        self,
        current_ids: set[int],
        timestamp: float,
        positions: dict[int, tuple[float, float]] | None = None,
        velocities: dict[int, tuple[float, float]] | None = None,
        sizes: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        """Move tracks that are no longer visible from active to lost pool.

        Only tracks with enough accumulated frames (min_track_age_frames) are
        saved — very brief detections are noise and not worth recovering.

        Velocity is computed via **OCM** (Observation-Centric Momentum) from
        the stored raw observation window when available, falling back to the
        Kalman-estimated velocity provided by the caller.
        """
        positions = positions or {}
        velocities = velocities or {}
        sizes = sizes or {}

        disappeared = [tid for tid in self._active if tid not in current_ids]
        for tid in disappeared:
            entry = self._active.pop(tid)
            if entry.frames_seen >= self._cfg.min_track_age_frames:
                ocm_vel = self._ocm_velocity(entry.recent_obs)
                use_ocm = abs(ocm_vel[0]) + abs(ocm_vel[1]) > 1e-3
                final_vel = ocm_vel if use_ocm else velocities.get(tid, (0.0, 0.0))
                final_pos = positions.get(tid, (0.0, 0.0))
                if entry.recent_obs:
                    final_pos = (entry.recent_obs[-1][0], entry.recent_obs[-1][1])

                self._lost[tid] = _LostEntry(
                    feature_short=entry.feature_short.copy(),
                    feature_long=entry.feature_long.copy(),
                    lost_ts=timestamp,
                    last_position=final_pos,
                    last_velocity=final_vel,
                    last_size=sizes.get(tid, (0.0, 0.0)),
                )
                logger.debug(
                    "Track %d retired to lost gallery (seen %d frames, OCM=%s)",
                    tid,
                    entry.frames_seen,
                    use_ocm,
                )
            else:
                logger.debug(
                    "Track %d dropped (only %d frames, below threshold %d)",
                    tid,
                    entry.frames_seen,
                    self._cfg.min_track_age_frames,
                )

    def purge_expired(self, timestamp: float) -> None:
        """Remove lost entries that have exceeded max_lost_age."""
        expired = [
            tid
            for tid, e in self._lost.items()
            if (timestamp - e.lost_ts) > self._cfg.max_lost_age
        ]
        for tid in expired:
            del self._lost[tid]
            logger.debug("Lost track %d expired (>%.1fs)", tid, self._cfg.max_lost_age)

    def transfer_identity(self, new_id: int, old_id: int) -> None:
        """After an ID remap, transfer the active entry from old semantics.

        When BoT-SORT assigns new_id but Re-ID says it's actually old_id,
        we need to keep the gallery consistent.
        """
        if new_id in self._active:
            entry = self._active.pop(new_id)
            self._active[old_id] = entry

    def clear(self) -> None:
        """Reset all state (useful for scene changes)."""
        self._active.clear()
        self._lost.clear()
