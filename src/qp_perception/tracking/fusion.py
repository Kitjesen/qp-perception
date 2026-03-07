"""
FusionMOT: Multi-cue Fused Multi-Object Tracker.
==================================================

A self-contained MOT algorithm that replaces external trackers (BoT-SORT/
ByteTrack) with a unified cost matrix approach.  All association cues are
fused **before** the Hungarian algorithm runs, so the optimal global
assignment already accounts for appearance, motion, shape, and spatial
information — no post-hoc ID repair needed.

Design references:
    - Deep OC-SORT (ICASSP 2023): fused cost = IoU + α·appearance
    - ByteTrack (ECCV 2022): two-stage matching (high + low confidence)
    - Hybrid-SORT (AAAI 2024): weak cues (confidence state, height state)
    - DeconfuseTrack (CVPR 2024): occlusion-aware NMS + decomposed matching
    - OC-SORT (CVPR 2023): observation-centric momentum for lost tracks
    - MOTIP (CVPR 2025): multi-frame temporal context for ID prediction

Architecture::

    YOLO detect (raw)
         │
         ▼
    ┌─────────────────────────────┐
    │  Stage 1: High-conf dets    │  IoU + Appearance + Motion + Height
    │  vs ALL active tracks       │  → Hungarian assignment
    │  (fused cost matrix)        │
    └────────────┬────────────────┘
                 │ unmatched tracks
                 ▼
    ┌─────────────────────────────┐
    │  Stage 2: Low-conf dets     │  IoU-only (ByteTrack style)
    │  vs remaining tracks        │  → Hungarian assignment
    └────────────┬────────────────┘
                 │
                 ▼
    Kalman update matched tracks
    Init new tracks / retire lost
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

import cv2 as _cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from qp_perception.kalman import CentroidKalmanCA, KalmanCAConfig
from qp_perception.reid.extractor import ReIDConfig, ReIDExtractor
from qp_perception.types import BoundingBox, Track

logger = logging.getLogger(__name__)

_UNMATCHED = 1e6

# COCO-17 keypoint indices used for pose-guided tracking
_KP_NOSE = 0
_KP_SHOULDER_L = 5
_KP_SHOULDER_R = 6
_KP_ELBOW_L = 7
_KP_ELBOW_R = 8
_KP_HIP_L = 11
_KP_HIP_R = 12
_KP_KNEE_L = 13
_KP_KNEE_R = 14

# 8 bone pairs for skeleton proportion descriptor (normalized by torso diagonal)
_SKELETON_BONES: tuple[tuple[int, int], ...] = (
    (_KP_SHOULDER_L, _KP_SHOULDER_R),  # shoulder span
    (_KP_HIP_L, _KP_HIP_R),            # hip span
    (_KP_SHOULDER_L, _KP_HIP_L),       # left torso
    (_KP_SHOULDER_R, _KP_HIP_R),       # right torso
    (_KP_SHOULDER_L, _KP_ELBOW_L),     # left upper arm
    (_KP_SHOULDER_R, _KP_ELBOW_R),     # right upper arm
    (_KP_HIP_L, _KP_KNEE_L),           # left thigh
    (_KP_HIP_R, _KP_KNEE_R),           # right thigh
)


@dataclass
class FusionMOTConfig:
    """Tuning knobs for the fused tracker."""

    # Detection thresholds
    high_conf: float = 0.35
    low_conf: float = 0.15

    # Cost matrix weights (Deep OC-SORT Eq.6 style)
    w_iou: float = 0.35
    w_appearance: float = 0.35
    w_motion: float = 0.20
    w_height: float = 0.10

    # Gating thresholds
    iou_gate: float = 0.05
    appearance_gate: float = 0.15
    motion_gate_px: float = 250.0
    use_mahalanobis: bool = True
    mahalanobis_sigma: float = 6.0
    min_motion_gate_px: float = 80.0

    # Track lifecycle
    confirm_frames: int = 2
    lost_patience: int = 3
    max_lost_frames: int = 60
    max_lost_seconds: float = 8.0

    # Feature bank
    feature_bank_size: int = 15
    ema_alpha: float = 0.90

    # ByteTrack Stage 2: IoU-only for low-conf
    stage2_iou_gate: float = 0.12

    # Lost track recovery (Stage 3)
    lost_motion_gate_multiplier: float = 3.0
    lost_appearance_gate: float = 0.10
    lost_match_threshold: float = 0.80

    # Kalman filter
    kalman_config: KalmanCAConfig = field(default_factory=KalmanCAConfig)

    # Skeleton / pose-guided tracking
    # Set w_skeleton > 0 when YOLO-pose keypoints are available.
    # use_hip_center: use hip midpoint as Kalman anchor (more stable than bbox centroid).
    # skeleton_gate: max normalized L2 descriptor distance to allow a match.
    # kp_visibility_thresh: minimum YOLO keypoint visibility score to treat a
    #     keypoint as valid (YOLO outputs [x, y, vis] with vis in [0, 1]).
    w_skeleton: float = 0.0
    use_hip_center: bool = True
    skeleton_gate: float = 0.8
    kp_visibility_thresh: float = 0.2

    # Dynamic ReID weight during occlusion.
    # When a confirmed track misses detections for more than occlusion_onset_frames,
    # gradually shift weight FROM w_iou TOWARD w_appearance (max boost =
    # occlusion_app_boost).  IoU becomes unreliable as the bbox drifts; the
    # feature bank provides a more stable identity signal.
    # At full occlusion (onset + 5 frames): w_app += occlusion_app_boost,
    #                                        w_iou -= occlusion_app_boost.
    occlusion_onset_frames: int = 2
    occlusion_app_boost: float = 0.15

    # OC-SORT Observation-Centric Re-Update (ORU / OCM).
    # After an occlusion gap, the Kalman velocity state is corrected by mixing
    # the observed displacement velocity (pos_delta / dt_gap) with the
    # Kalman-predicted velocity.  This undoes the CA drift during occlusion.
    # oru_alpha : weight given to observed velocity (0 = pure Kalman, 1 = pure obs).
    # oru_min_gap_frames : minimum missed-frame count before ORU is applied.
    oru_alpha: float = 0.6
    oru_min_gap_frames: int = 2


class _Tracklet:
    """Internal tracklet state with embedded Kalman CA filter."""

    __slots__ = (
        "track_id", "kf", "feature_ema", "feature_bank", "bbox",
        "height_ema", "confidence", "state", "frames_since_update",
        "total_frames", "last_ts", "created_ts",
        "keypoints_ema",   # EMA-smoothed COCO-17 keypoints (17, 2) or None
        "last_obs_cx", "last_obs_cy", "last_obs_ts",  # OC-SORT ORU anchor
    )

    def __init__(self, track_id: int, cx: float, cy: float,
                 kalman_cfg: KalmanCAConfig) -> None:
        self.track_id = track_id
        self.kf = CentroidKalmanCA(cx0=cx, cy0=cy, config=kalman_cfg)
        self.feature_ema: np.ndarray | None = None
        self.feature_bank: list = []
        self.bbox = np.zeros(4, dtype=np.float64)  # x, y, w, h
        self.height_ema: float = 0.0
        self.confidence: float = 0.0
        self.state: str = "tentative"
        self.frames_since_update: int = 0
        self.total_frames: int = 0
        self.last_ts: float = 0.0
        self.created_ts: float = 0.0
        self.keypoints_ema: np.ndarray | None = None  # shape (17, 2)
        self.last_obs_cx: float = cx
        self.last_obs_cy: float = cy
        self.last_obs_ts: float = 0.0

    @property
    def position(self) -> np.ndarray:
        """Kalman-filtered position (cx, cy)."""
        p = self.kf.position
        return np.array([p[0], p[1]], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        """Kalman-filtered velocity (vx, vy) in px/s."""
        v = self.kf.velocity
        return np.array([v[0], v[1]], dtype=np.float64)

    @property
    def pos_covariance(self) -> np.ndarray:
        """2x2 position covariance from Kalman state."""
        return self.kf.covariance_2x2

    def predicted_bbox(self) -> np.ndarray:
        """Bbox with Kalman-predicted center, preserving last w/h."""
        pos = self.position
        w, h = self.bbox[2], self.bbox[3]
        return np.array([pos[0] - w / 2, pos[1] - h / 2, w, h],
                        dtype=np.float64)


class FusionMOT:
    """Multi-Object Tracker with fused multi-cue cost matrix.

    Usage::

        tracker = FusionMOT(config, reid_extractor)

        # Each frame:
        detections = yolo_model(frame)  # raw detections
        tracks = tracker.update(detections, features, timestamp)
    """

    def __init__(self, config: FusionMOTConfig | None = None,
                 feature_dim: int = 512) -> None:
        self._cfg = config or FusionMOTConfig()
        self._feature_dim = feature_dim
        self._tracklets: dict[int, _Tracklet] = {}
        self._next_id = 1
        self._frame_count = 0
        self._last_global_ts: float = 0.0

        logger.info("FusionMOT initialized  weights=(iou=%.2f, app=%.2f, "
                     "motion=%.2f, height=%.2f)  kalman=CA(6-state)  "
                     "mahalanobis=%s(σ=%.1f)",
                     self._cfg.w_iou, self._cfg.w_appearance,
                     self._cfg.w_motion, self._cfg.w_height,
                     self._cfg.use_mahalanobis, self._cfg.mahalanobis_sigma)

    @property
    def active_tracks(self) -> dict[int, _Tracklet]:
        return self._tracklets

    def update(
        self,
        bboxes: np.ndarray,
        confidences: np.ndarray,
        features: np.ndarray | None,
        timestamp: float,
        keypoints: np.ndarray | None = None,
        gimbal_delta_yaw_deg: float = 0.0,
        gimbal_delta_pitch_deg: float = 0.0,
        camera_fx: float = 0.0,
        camera_fy: float = 0.0,
    ) -> list[tuple[int, np.ndarray, float]]:
        """Run one frame of tracking.

        Parameters
        ----------
        bboxes : (N, 4) float array — [x, y, w, h] per detection
        confidences : (N,) float array
        features : (N, D) float array or None — appearance features
        timestamp : float
        keypoints : (N, 17, 2) or (N, 17, 3) float array or None.
            COCO-17 keypoints from a pose model.  Shape (N, 17, 2) contains
            [x, y] pixel coords; shape (N, 17, 3) also includes a visibility
            score in the third channel.  When provided and
            ``use_hip_center=True``, the hip midpoint replaces the bbox
            centroid as the Kalman measurement anchor.
        gimbal_delta_yaw_deg : float, optional
            Camera yaw change since last frame (deg). When non-zero and
            camera_fx > 0, enables EMAP ego-motion compensation in the
            Kalman predict step so gimbal rotation is not confused with
            target motion.
        gimbal_delta_pitch_deg : float, optional
            Camera pitch change since last frame (deg).
        camera_fx, camera_fy : float, optional
            Focal lengths (pixels) required for EMAP pixel displacement math.

        Returns
        -------
        list of (track_id, bbox_xywh, confidence) for confirmed tracks
        """
        self._frame_count += 1
        dt = max(timestamp - self._last_global_ts, 1e-3) if self._last_global_ts > 0 else 0.033
        self._last_global_ts = timestamp

        # Kalman predict: advance ALL tracklets using CA model (parabolic).
        # When gimbal ego-motion is provided, use EMAP to decouple platform
        # rotation from target motion in the Kalman prediction step.
        use_emap = (
            (abs(gimbal_delta_yaw_deg) > 1e-4 or abs(gimbal_delta_pitch_deg) > 1e-4)
            and camera_fx > 0.0
        )
        for t in self._tracklets.values():
            if use_emap:
                t.kf.predict_with_ego_motion(
                    dt, gimbal_delta_yaw_deg, gimbal_delta_pitch_deg,
                    camera_fx, camera_fy,
                )
            else:
                t.kf.predict(dt)
            # Sync bbox position from Kalman state
            pos = t.position
            t.bbox[0] = pos[0] - t.bbox[2] / 2
            t.bbox[1] = pos[1] - t.bbox[3] / 2

        N = len(bboxes)
        if N == 0:
            self._mark_unmatched_tracks(timestamp)
            self._cleanup(timestamp)
            return self._get_output(timestamp)

        # Split detections by confidence
        high_mask = confidences >= self._cfg.high_conf
        low_mask = (~high_mask) & (confidences >= self._cfg.low_conf)

        high_idx = np.where(high_mask)[0]
        low_idx = np.where(low_mask)[0]

        # Get active + lost tracklet IDs for matching
        track_ids = [tid for tid, t in self._tracklets.items()
                     if t.state in ("confirmed", "tentative", "lost")]

        matched_track_set: set[int] = set()
        matched_det_set: set[int] = set()

        # Helper: safely index keypoints array
        def _kpts(idx: int) -> np.ndarray | None:
            return keypoints[idx] if keypoints is not None else None

        def _kpts_slice(idx_arr: np.ndarray) -> np.ndarray | None:
            return keypoints[idx_arr] if keypoints is not None else None

        # ═══ Stage 1: High-conf detections vs ALL tracks (full cost) ═══
        if len(high_idx) > 0 and len(track_ids) > 0:
            cost = self._build_cost_matrix(
                track_ids, bboxes[high_idx],
                features[high_idx] if features is not None else None,
                det_keypoints=_kpts_slice(high_idx),
            )
            t_matched, d_matched = self._hungarian_match(cost, threshold=0.70)

            for ti, di in zip(t_matched, d_matched):
                tid = track_ids[ti]
                det_i = high_idx[di]
                self._update_tracklet(
                    tid, bboxes[det_i], confidences[det_i],
                    features[det_i] if features is not None else None,
                    timestamp,
                    kpts=_kpts(det_i),
                )
                matched_track_set.add(tid)
                matched_det_set.add(int(det_i))

        # ═══ Stage 2: Low-conf detections vs UNMATCHED tracks (IoU only) ═══
        remaining_tracks = [tid for tid in track_ids
                           if tid not in matched_track_set]

        if len(low_idx) > 0 and len(remaining_tracks) > 0:
            cost_s2 = self._build_iou_cost(remaining_tracks, bboxes[low_idx])
            t_matched2, d_matched2 = self._hungarian_match(
                cost_s2, threshold=1.0 - self._cfg.stage2_iou_gate)

            for ti, di in zip(t_matched2, d_matched2):
                tid = remaining_tracks[ti]
                det_i = low_idx[di]
                self._update_tracklet(
                    tid, bboxes[det_i], confidences[det_i],
                    None,  # no appearance update for low-conf
                    timestamp,
                    # no keypoint update for low-conf to avoid noisy poses
                )
                matched_track_set.add(tid)
                matched_det_set.add(int(det_i))

        # ═══ Stage 3: Unmatched high-conf dets vs lost tracks (Re-ID recovery) ═══
        unmatched_high = [int(i) for i in high_idx if int(i) not in matched_det_set]
        lost_tracks = [tid for tid, t in self._tracklets.items()
                       if t.state == "lost" and tid not in matched_track_set]

        # Also include confirmed tracks that weren't matched (brief occlusion)
        patience_tracks = [tid for tid, t in self._tracklets.items()
                           if t.state == "confirmed"
                           and t.frames_since_update > 0
                           and tid not in matched_track_set]
        recovery_tracks = lost_tracks + patience_tracks

        if unmatched_high and recovery_tracks:
            uh_bboxes = bboxes[unmatched_high]
            uh_feats = features[unmatched_high] if features is not None else None
            uh_kpts = (keypoints[unmatched_high]
                       if keypoints is not None else None)
            cost_s3 = self._build_recovery_cost(
                recovery_tracks, uh_bboxes, uh_feats,
                det_keypoints=uh_kpts,
            )
            t_m3, d_m3 = self._hungarian_match(
                cost_s3, threshold=self._cfg.lost_match_threshold)

            for ti, di in zip(t_m3, d_m3):
                tid = recovery_tracks[ti]
                det_i = unmatched_high[di]
                self._update_tracklet(
                    tid, bboxes[det_i], confidences[det_i],
                    features[det_i] if features is not None else None,
                    timestamp,
                    kpts=_kpts(det_i),
                )
                matched_track_set.add(tid)
                matched_det_set.add(det_i)

        # ═══ Init new tracks from unmatched high-conf detections ═══
        for i in high_idx:
            if int(i) not in matched_det_set:
                self._init_tracklet(
                    bboxes[i], confidences[i],
                    features[i] if features is not None else None,
                    timestamp,
                    kpts=_kpts(i),
                )

        # Mark unmatched tracks (with patience before declaring lost)
        for tid in track_ids:
            if tid not in matched_track_set:
                t = self._tracklets[tid]
                t.frames_since_update += 1
                if (t.state == "confirmed"
                        and t.frames_since_update > self._cfg.lost_patience):
                    t.state = "lost"

        self._cleanup(timestamp)
        return self._get_output(timestamp)

    # ------------------------------------------------------------------
    # Cost matrix construction
    # ------------------------------------------------------------------

    def _build_cost_matrix(
        self,
        track_ids: list[int],
        det_bboxes: np.ndarray,
        det_features: np.ndarray | None,
        det_keypoints: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build fused cost matrix using Kalman-predicted positions.

        Motion distance uses **Mahalanobis distance** when enabled: the
        Kalman covariance provides an uncertainty-aware gate that adapts
        per-track — a track with high uncertainty (just initialized or
        recovering) gets a wider gate automatically, while a well-tracked
        target has a tight gate that rejects false matches.

        cost = w_iou·(1-IoU) + w_app·(1-cosine)/2
             + w_motion·mahal_norm + w_height·h_diff
             [+ w_skeleton·skel_dist  (when keypoints available)]
        """
        M = len(track_ids)
        N = len(det_bboxes)
        cost = np.full((M, N), _UNMATCHED, dtype=np.float64)

        cfg = self._cfg
        det_centers = det_bboxes[:, :2] + det_bboxes[:, 2:4] / 2
        det_heights = det_bboxes[:, 3]

        # Precompute skeleton descriptors for all detections (once per frame)
        compute_skel = (cfg.w_skeleton > 0.0 and det_keypoints is not None)
        det_skel_descs: list[np.ndarray | None] = []
        det_vis: list[np.ndarray | None] = []
        if compute_skel:
            for j in range(N):
                kp = det_keypoints[j]           # (17, 2) or (17, 3)
                v = kp[:, 2] if kp.shape[1] == 3 else None
                xy = kp[:, :2]
                det_vis.append(v)
                det_skel_descs.append(
                    self._skeleton_descriptor(xy, v, cfg.kp_visibility_thresh))
        else:
            det_vis = [None] * N
            det_skel_descs = [None] * N

        for i, tid in enumerate(track_ids):
            t = self._tracklets[tid]
            t_pred_bbox = t.predicted_bbox()
            t_center = t.position
            t_height = t.height_ema if t.height_ema > 1.0 else t.bbox[3]

            # Precompute per-track skeleton descriptor from EMA keypoints
            t_skel_desc: np.ndarray | None = None
            if compute_skel and t.keypoints_ema is not None:
                t_skel_desc = self._skeleton_descriptor(
                    t.keypoints_ema, None, cfg.kp_visibility_thresh)

            # Precompute inverse covariance for Mahalanobis
            if cfg.use_mahalanobis:
                cov = t.pos_covariance
                cov_reg = cov + np.eye(2) * 1e-4
                try:
                    cov_inv = np.linalg.inv(cov_reg)
                except np.linalg.LinAlgError:
                    cov_inv = np.eye(2) / (cfg.motion_gate_px ** 2)
            else:
                cov_inv = None

            # Dynamic ReID weight: during occlusion (missed detections), shift
            # weight from IoU toward appearance.  IoU reliability degrades as the
            # Kalman-predicted bbox drifts; appearance (feature bank) is more
            # identity-stable.  Only active when features are available.
            if (t.frames_since_update > cfg.occlusion_onset_frames
                    and det_features is not None):
                occ_alpha = min(
                    (t.frames_since_update - cfg.occlusion_onset_frames) / 5.0, 1.0
                )
                transfer = cfg.occlusion_app_boost * occ_alpha
                w_iou_eff = max(cfg.w_iou - transfer, 0.05)
                w_app_eff = cfg.w_appearance + transfer
            else:
                w_iou_eff = cfg.w_iou
                w_app_eff = cfg.w_appearance

            for j in range(N):
                # --- IoU distance (using Kalman-predicted bbox) ---
                iou = self._iou(t_pred_bbox, det_bboxes[j])
                if iou < cfg.iou_gate and t.state != "lost":
                    continue
                iou_cost = 1.0 - iou

                # --- Motion distance (Kalman-predicted center) ---
                delta = det_centers[j] - t_center
                eucl_dist = float(np.linalg.norm(delta))
                if cov_inv is not None:
                    mahal_sq = float(delta @ cov_inv @ delta)
                    mahal_ok = mahal_sq <= cfg.mahalanobis_sigma ** 2
                    pixel_ok = eucl_dist <= cfg.min_motion_gate_px
                    if not (mahal_ok or pixel_ok) and t.state != "lost":
                        continue
                    motion_cost = min(mahal_sq ** 0.5 / cfg.mahalanobis_sigma, 1.0)
                    if pixel_ok and not mahal_ok:
                        motion_cost = min(eucl_dist / cfg.motion_gate_px, 1.0)
                else:
                    gate_r = cfg.motion_gate_px
                    if t.state == "lost":
                        gate_r *= (1.0 + t.frames_since_update * 0.5)
                    if eucl_dist > gate_r:
                        continue
                    motion_cost = min(eucl_dist / max(gate_r, 1.0), 1.0)

                # --- Appearance distance ---
                app_cost = 0.5
                if det_features is not None and t.feature_ema is not None:
                    cos_sim = float(t.feature_ema @ det_features[j])
                    if t.feature_bank:
                        bank_sims = [float(f @ det_features[j])
                                     for f, _, _ in t.feature_bank[-cfg.feature_bank_size:]]
                        bank_best = max(bank_sims)
                        cos_sim = max(cos_sim, 0.7 * cos_sim + 0.3 * bank_best)
                    if cos_sim < cfg.appearance_gate and t.state != "lost":
                        continue
                    app_cost = (1.0 - cos_sim) / 2.0

                # --- Height distance (Hybrid-SORT) ---
                h_cost = 0.0
                if t_height > 1.0 and det_heights[j] > 1.0:
                    h_ratio = min(t_height, det_heights[j]) / max(t_height, det_heights[j])
                    h_cost = 1.0 - h_ratio

                # --- Skeleton proportion distance (pose-guided) ---
                skel_cost = 0.5
                if compute_skel and t_skel_desc is not None and det_skel_descs[j] is not None:
                    dist = float(np.linalg.norm(t_skel_desc - det_skel_descs[j]))
                    if dist > cfg.skeleton_gate and t.state != "lost":
                        continue  # biomechanical gate: body proportions too different
                    skel_cost = min(dist / max(cfg.skeleton_gate, 1e-6), 1.0)

                total = (w_iou_eff * iou_cost
                         + w_app_eff * app_cost
                         + cfg.w_motion * motion_cost
                         + cfg.w_height * h_cost
                         + cfg.w_skeleton * skel_cost)
                cost[i, j] = total

        return cost

    def _build_recovery_cost(
        self,
        track_ids: list[int],
        det_bboxes: np.ndarray,
        det_features: np.ndarray | None,
        det_keypoints: np.ndarray | None = None,
    ) -> np.ndarray:
        """Cost matrix for lost track recovery (Stage 3).

        Uses Kalman-predicted position (which keeps drifting forward even
        during occlusion thanks to the CA model), with a much wider
        Mahalanobis gate — lost tracks accumulate covariance, so the gate
        widens naturally.  Skeleton descriptor is weighted more heavily here
        since it is the most person-identity-stable signal.
        """
        M = len(track_ids)
        N = len(det_bboxes)
        cost = np.full((M, N), _UNMATCHED, dtype=np.float64)

        cfg = self._cfg
        det_centers = det_bboxes[:, :2] + det_bboxes[:, 2:4] / 2
        det_heights = det_bboxes[:, 3]

        compute_skel = (cfg.w_skeleton > 0.0 and det_keypoints is not None)
        det_skel_descs: list[np.ndarray | None] = []
        if compute_skel:
            for j in range(N):
                kp = det_keypoints[j]
                v = kp[:, 2] if kp.shape[1] == 3 else None
                det_skel_descs.append(
                    self._skeleton_descriptor(kp[:, :2], v, cfg.kp_visibility_thresh))
        else:
            det_skel_descs = [None] * N

        for i, tid in enumerate(track_ids):
            t = self._tracklets[tid]
            t_center = t.position  # Kalman-predicted even when lost
            t_height = t.height_ema if t.height_ema > 1.0 else t.bbox[3]

            t_skel_desc: np.ndarray | None = None
            if compute_skel and t.keypoints_ema is not None:
                t_skel_desc = self._skeleton_descriptor(
                    t.keypoints_ema, None, cfg.kp_visibility_thresh)

            for j in range(N):
                delta = det_centers[j] - t_center
                dist = float(np.linalg.norm(delta))
                gate_r = cfg.motion_gate_px * cfg.lost_motion_gate_multiplier
                gate_r *= (1.0 + t.frames_since_update * 0.3)
                if dist > gate_r:
                    continue
                motion_cost = min(dist / max(gate_r, 1.0), 1.0)

                app_cost = 0.5
                if det_features is not None and t.feature_ema is not None:
                    cos_sim = float(t.feature_ema @ det_features[j])
                    if t.feature_bank:
                        bank_sims = [float(f @ det_features[j])
                                     for f, _, _ in t.feature_bank]
                        bank_best = max(bank_sims)
                        cos_sim = max(cos_sim, bank_best)
                    if cos_sim < cfg.lost_appearance_gate:
                        continue
                    app_cost = (1.0 - cos_sim) / 2.0

                h_cost = 0.0
                if t_height > 1.0 and det_heights[j] > 1.0:
                    h_ratio = min(t_height, det_heights[j]) / max(t_height, det_heights[j])
                    h_cost = 1.0 - h_ratio

                skel_cost = 0.5
                if compute_skel and t_skel_desc is not None and det_skel_descs[j] is not None:
                    dist_s = float(np.linalg.norm(t_skel_desc - det_skel_descs[j]))
                    skel_cost = min(dist_s / max(cfg.skeleton_gate, 1e-6), 1.0)

                pred_bbox = t.predicted_bbox()
                # In recovery, skeleton descriptor carries extra weight as a
                # person-identity anchor (body proportions are stable post-occlusion)
                skel_w = cfg.w_skeleton * 1.5 if compute_skel else 0.0
                total = (0.15 * motion_cost
                         + 0.65 * app_cost
                         + 0.10 * h_cost
                         + 0.10 * (1.0 - self._iou(pred_bbox, det_bboxes[j]))
                         + skel_w * skel_cost)
                cost[i, j] = total

        return cost

    def _build_iou_cost(self, track_ids: list[int],
                        det_bboxes: np.ndarray) -> np.ndarray:
        """IoU-only cost matrix for ByteTrack Stage 2 (Kalman-predicted bboxes)."""
        M = len(track_ids)
        N = len(det_bboxes)
        cost = np.full((M, N), _UNMATCHED, dtype=np.float64)

        for i, tid in enumerate(track_ids):
            pred_bbox = self._tracklets[tid].predicted_bbox()
            for j in range(N):
                iou = self._iou(pred_bbox, det_bboxes[j])
                if iou > self._cfg.stage2_iou_gate:
                    cost[i, j] = 1.0 - iou

        return cost

    @staticmethod
    def _hungarian_match(cost: np.ndarray,
                         threshold: float) -> tuple[list[int], list[int]]:
        """Run Hungarian algorithm with gating threshold."""
        if cost.size == 0:
            return [], []

        row_idx, col_idx = linear_sum_assignment(cost)
        t_matched: list[int] = []
        d_matched: list[int] = []

        for r, c in zip(row_idx, col_idx):
            if cost[r, c] < threshold:
                t_matched.append(int(r))
                d_matched.append(int(c))

        return t_matched, d_matched

    # ------------------------------------------------------------------
    # Tracklet lifecycle
    # ------------------------------------------------------------------

    def _init_tracklet(self, bbox: np.ndarray, conf: float,
                       feat: np.ndarray | None, ts: float,
                       kpts: np.ndarray | None = None) -> int:
        tid = self._next_id
        self._next_id += 1

        # Prefer hip center as Kalman anchor when keypoints are available
        center = bbox[:2] + bbox[2:4] / 2
        if kpts is not None and self._cfg.use_hip_center:
            xy = kpts[:, :2]
            vis = kpts[:, 2] if kpts.shape[1] == 3 else None
            hip = self._hip_center(xy, vis, self._cfg.kp_visibility_thresh)
            if hip is not None:
                center = np.array(hip, dtype=np.float64)

        t = _Tracklet(
            track_id=tid,
            cx=float(center[0]),
            cy=float(center[1]),
            kalman_cfg=self._cfg.kalman_config,
        )
        t.bbox = bbox.copy()
        t.height_ema = float(bbox[3])
        t.confidence = conf
        t.state = "tentative"
        t.total_frames = 1
        t.last_ts = ts
        t.created_ts = ts

        if feat is not None:
            f = feat.copy()
            n = np.linalg.norm(f)
            if n > 1e-6:
                f /= n
            t.feature_ema = f
            t.feature_bank = [(f.copy(), conf, ts)]

        if kpts is not None:
            t.keypoints_ema = kpts[:, :2].copy().astype(np.float64)

        self._tracklets[tid] = t
        return tid

    def _update_tracklet(self, tid: int, bbox: np.ndarray, conf: float,
                         feat: np.ndarray | None, ts: float,
                         kpts: np.ndarray | None = None) -> None:
        t = self._tracklets[tid]

        # Prefer hip center as Kalman measurement when keypoints are available
        center = bbox[:2] + bbox[2:4] / 2
        if kpts is not None and self._cfg.use_hip_center:
            xy = kpts[:, :2]
            vis = kpts[:, 2] if kpts.shape[1] == 3 else None
            hip = self._hip_center(xy, vis, self._cfg.kp_visibility_thresh)
            if hip is not None:
                center = np.array(hip, dtype=np.float64)

        # OC-SORT ORU (Observation-Centric Re-Update).
        # After an occlusion gap the CA model may have let the velocity state
        # drift.  Correct it by mixing the observed displacement velocity
        # (Δpos / Δt over the gap) with the Kalman-predicted velocity.
        # Only applied when the gap is at least oru_min_gap_frames long.
        if (t.frames_since_update >= self._cfg.oru_min_gap_frames
                and t.last_obs_ts > 0.0):
            dt_gap = ts - t.last_obs_ts
            if dt_gap > 0.01:
                v_obs_x = (float(center[0]) - t.last_obs_cx) / dt_gap
                v_obs_y = (float(center[1]) - t.last_obs_cy) / dt_gap
                alpha = self._cfg.oru_alpha
                t.kf.blend_velocity(v_obs_x, v_obs_y, alpha)
                logger.debug(
                    "Track %d ORU: gap=%d frames dt=%.3fs "
                    "v_obs=(%.1f,%.1f) kf_vel_after=(%.1f,%.1f)",
                    tid, t.frames_since_update, dt_gap,
                    v_obs_x, v_obs_y,
                    *t.kf.velocity,
                )

        # Kalman measurement update — confidence-scaled R so high-conf detections
        # move the state estimate more aggressively while low-conf borderline
        # detections are smoothed (Kalman prediction trusted more).
        t.kf.update_with_confidence(float(center[0]), float(center[1]), conf)

        # Store observed bbox (w/h from detection, position from Kalman)
        t.bbox = bbox.copy()
        t.confidence = conf
        t.frames_since_update = 0
        t.total_frames += 1
        t.last_ts = ts

        # Height EMA (Hybrid-SORT stable weak cue)
        if bbox[3] > 1.0:
            t.height_ema = 0.9 * t.height_ema + 0.1 * float(bbox[3])

        # Appearance update (Dynamic Appearance EMA, Deep OC-SORT)
        if feat is not None and t.feature_ema is not None:
            f = feat.copy()
            n = np.linalg.norm(f)
            if n > 1e-6:
                f /= n
            sigma = 0.4
            if conf > sigma:
                a = 0.9 + 0.1 * (1.0 - (conf - sigma) / (1.0 - sigma))
            else:
                a = 1.0
            t.feature_ema = a * t.feature_ema + (1.0 - a) * f
            en = np.linalg.norm(t.feature_ema)
            if en > 1e-6:
                t.feature_ema /= en

            if conf > sigma:
                t.feature_bank.append((f.copy(), conf, ts))
                if len(t.feature_bank) > self._cfg.feature_bank_size:
                    t.feature_bank = t.feature_bank[-self._cfg.feature_bank_size:]
        elif feat is not None and t.feature_ema is None:
            f = feat.copy()
            n = np.linalg.norm(f)
            if n > 1e-6:
                f /= n
            t.feature_ema = f
            t.feature_bank = [(f.copy(), conf, ts)]

        # Keypoints EMA update (high-conf frames only, α=0.7 for stability)
        if kpts is not None:
            xy = kpts[:, :2].astype(np.float64)
            if t.keypoints_ema is None:
                t.keypoints_ema = xy.copy()
            else:
                t.keypoints_ema = 0.7 * t.keypoints_ema + 0.3 * xy

        # State transition
        if t.state == "tentative" and t.total_frames >= self._cfg.confirm_frames:
            t.state = "confirmed"
        elif t.state == "lost":
            t.state = "confirmed"
            logger.info("Track %d recovered from lost state", tid)

        # Update ORU anchor: store the current matched observation position
        # so it can be used for velocity correction after the next occlusion gap.
        t.last_obs_cx = float(center[0])
        t.last_obs_cy = float(center[1])
        t.last_obs_ts = ts

    def _mark_unmatched_tracks(self, timestamp: float) -> None:
        for t in self._tracklets.values():
            if t.state in ("confirmed", "tentative"):
                t.frames_since_update += 1
                if (t.state == "confirmed"
                        and t.frames_since_update > self._cfg.lost_patience):
                    t.state = "lost"

    def _cleanup(self, timestamp: float) -> None:
        """Remove dead tracklets."""
        to_delete: list[int] = []
        cfg = self._cfg

        for tid, t in self._tracklets.items():
            if t.state == "tentative" and t.frames_since_update > 2:
                to_delete.append(tid)
            elif t.state == "lost":
                if t.frames_since_update > cfg.max_lost_frames:
                    to_delete.append(tid)
                elif (timestamp - t.last_ts) > cfg.max_lost_seconds:
                    to_delete.append(tid)

        for tid in to_delete:
            del self._tracklets[tid]

    def _get_output(self, timestamp: float) -> list[tuple[int, np.ndarray, float]]:
        """Return confirmed tracks with Kalman-smoothed bboxes.

        During patience frames (no measurement yet), the Kalman CA model
        provides parabolic extrapolation — much more accurate than the
        old linear velocity × dt.
        """
        results: list[tuple[int, np.ndarray, float]] = []
        for t in self._tracklets.values():
            if t.state == "confirmed":
                if t.frames_since_update <= self._cfg.lost_patience:
                    results.append((t.track_id, t.predicted_bbox(), t.confidence))
        return results

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        """IoU for (x, y, w, h) boxes."""
        ax1, ay1, aw, ah = a[0], a[1], a[2], a[3]
        bx1, by1, bw, bh = b[0], b[1], b[2], b[3]
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 1e-6 else 0.0

    @staticmethod
    def _hip_center(
        kpts: np.ndarray,
        vis: np.ndarray | None = None,
        vis_thresh: float = 0.3,
    ) -> tuple[float, float] | None:
        """Extract hip midpoint from COCO-17 keypoints.

        Parameters
        ----------
        kpts : (17, 2) array — x, y pixel coordinates.
        vis  : (17,) array — per-keypoint visibility [0, 1], or None.
        vis_thresh : minimum visibility score to treat a keypoint as valid.

        Returns
        -------
        (cx, cy) hip midpoint, or None if both hips are invisible.
        """
        if kpts is None or kpts.shape[0] < 13:
            return None

        def _valid(idx: int) -> bool:
            if vis is not None:
                if float(vis[idx]) < vis_thresh:
                    return False
            # Guard: YOLO outputs (0, 0) for invisible keypoints regardless of
            # visibility score — a pixel coordinate of (0,0) is almost certainly
            # the top-left image corner and not a real body landmark.
            px, py = float(kpts[idx, 0]), float(kpts[idx, 1])
            return px > 1.0 and py > 1.0

        lh_ok = _valid(_KP_HIP_L)
        rh_ok = _valid(_KP_HIP_R)

        if lh_ok and rh_ok:
            cx = (float(kpts[_KP_HIP_L, 0]) + float(kpts[_KP_HIP_R, 0])) / 2.0
            cy = (float(kpts[_KP_HIP_L, 1]) + float(kpts[_KP_HIP_R, 1])) / 2.0
            return (cx, cy)
        if lh_ok:
            return (float(kpts[_KP_HIP_L, 0]), float(kpts[_KP_HIP_L, 1]))
        if rh_ok:
            return (float(kpts[_KP_HIP_R, 0]), float(kpts[_KP_HIP_R, 1]))
        return None

    @staticmethod
    def _skeleton_descriptor(
        kpts: np.ndarray,
        vis: np.ndarray | None = None,
        vis_thresh: float = 0.3,
    ) -> np.ndarray | None:
        """Compute an 8-D bone-proportion descriptor normalized by torso diagonal.

        The descriptor captures body proportions (shoulder/hip span, limb lengths)
        relative to the torso — scale-invariant and stable across frames.
        Invisible bones are encoded as 0 (neutral), so a partially occluded
        person still produces a partial (but usable) descriptor.

        Returns None only when the torso is too small to serve as normalizer
        (both shoulders or both hips are invisible / out of frame).
        """
        if kpts is None or kpts.shape[0] < 17:
            return None

        def _vis(idx: int) -> bool:
            if vis is not None:
                if float(vis[idx]) < vis_thresh:
                    return False
            px, py = float(kpts[idx, 0]), float(kpts[idx, 1])
            return px > 1.0 and py > 1.0

        # Torso diagonal: average of (right_shoulder → left_hip) and
        # (left_shoulder → right_hip) for robustness against single-side occlusion.
        diag_candidates: list[float] = []
        if _vis(_KP_SHOULDER_R) and _vis(_KP_HIP_L):
            diag_candidates.append(float(np.linalg.norm(
                kpts[_KP_SHOULDER_R] - kpts[_KP_HIP_L])))
        if _vis(_KP_SHOULDER_L) and _vis(_KP_HIP_R):
            diag_candidates.append(float(np.linalg.norm(
                kpts[_KP_SHOULDER_L] - kpts[_KP_HIP_R])))

        if not diag_candidates:
            return None
        normalizer = sum(diag_candidates) / len(diag_candidates)
        if normalizer < 5.0:
            return None

        desc = np.zeros(len(_SKELETON_BONES), dtype=np.float64)
        for k, (a, b) in enumerate(_SKELETON_BONES):
            if _vis(a) and _vis(b):
                desc[k] = float(np.linalg.norm(kpts[a] - kpts[b])) / normalizer
            # invisible bone stays 0 (neutral, not penalised in distance)

        return desc

    def reset(self) -> None:
        """Clear all state."""
        self._tracklets.clear()
        self._next_id = 1
        self._frame_count = 0
        self._last_global_ts = 0.0


# ======================================================================
# FusionSegTracker: YOLO-Seg detection + FusionMOT tracking.
# ======================================================================

class FusionSegTracker:
    """YOLO-Seg + FusionMOT (with internal Kalman CA) + OSNet.

    Parameters
    ----------
    model_path : str
        YOLO-Seg model weights (e.g. "yolo11n-seg.pt").
    confidence_threshold : float
        High-confidence detection threshold for Stage 1.
    low_confidence_threshold : float
        Low-confidence threshold for ByteTrack Stage 2.
    class_whitelist : optional
        Only track these COCO class names (e.g. ["person"]).
    device : str
        PyTorch device ("cuda:0", "cpu", "" for auto).
    img_size : int
        YOLO input size (longer side).
    reid_config : ReIDConfig
        Re-ID feature extraction configuration.
    mot_config : FusionMOTConfig
        Tracking algorithm configuration (includes kalman_config).
    """

    def __init__(
        self,
        model_path: str = "yolo11n-seg.pt",
        confidence_threshold: float = 0.35,
        low_confidence_threshold: float = 0.15,
        class_whitelist: Sequence[str] | None = None,
        device: str = "",
        img_size: int = 640,
        reid_config: ReIDConfig | None = None,
        mot_config: FusionMOTConfig | None = None,
    ) -> None:
        from ultralytics import YOLO  # type: ignore[import-untyped]

        self._model = YOLO(model_path)
        self._conf_high = confidence_threshold
        self._conf_low = low_confidence_threshold
        self._device = device
        self._img_size = img_size

        self._id_to_name: dict[int, str] = self._model.names
        self._allowed_ids: list[int] | None = None
        if class_whitelist is not None:
            name_map = {v.lower(): k for k, v in self._id_to_name.items()}
            self._allowed_ids = [
                name_map[n.lower()] for n in class_whitelist if n.lower() in name_map
            ]

        self._reid = ReIDExtractor(reid_config or ReIDConfig())

        mot_cfg = mot_config or FusionMOTConfig()
        mot_cfg.high_conf = confidence_threshold
        mot_cfg.low_conf = low_confidence_threshold
        self._mot = FusionMOT(mot_cfg, feature_dim=self._reid.feature_dim)

        self._first_seen: dict[int, float] = {}
        self._last_raw_results: list | None = None
        self._total_extractions = 0
        self._total_skips = 0

        logger.info(
            "FusionSegTracker ready  model=%s  reid=%s  device=%s  kalman=internal_CA",
            model_path, (reid_config or ReIDConfig()).backbone, device or "auto",
        )

    @property
    def last_raw_results(self) -> list | None:
        return self._last_raw_results

    @property
    def reid_stats(self) -> dict[str, int]:
        active = sum(1 for t in self._mot.active_tracks.values()
                     if t.state == "confirmed")
        lost = sum(1 for t in self._mot.active_tracks.values()
                   if t.state == "lost")
        return {
            "enabled": 1,
            "active": active,
            "lost": lost,
            "remaps": 0,
            "extractions": self._total_extractions,
            "skips": self._total_skips,
        }

    def detect_and_track(self, frame: np.ndarray,
                         timestamp: float) -> list[Track]:
        """Run full pipeline: detect → extract features → FusionMOT(Kalman).

        Automatically extracts COCO-17 keypoints when the loaded model is a
        pose model (``result.keypoints`` is not None).  Keypoints are passed
        to FusionMOT so that hip center is used as the Kalman anchor and the
        skeleton proportion descriptor is included in the cost matrix
        (requires ``FusionMOTConfig.w_skeleton > 0``).
        """
        if not isinstance(frame, np.ndarray):
            return []

        results = self._model(
            source=frame,
            conf=self._conf_low,
            iou=0.45,
            imgsz=self._img_size,
            device=self._device or None,
            classes=self._allowed_ids,
            verbose=False,
        )
        self._last_raw_results = results

        all_bboxes: list[np.ndarray] = []
        all_confs: list[float] = []
        all_keypoints: list[np.ndarray] = []   # (17, 3) per detection: [x, y, vis]
        has_pose = False

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            # Pose model outputs result.keypoints; seg/detect models do not
            kpts_data = None
            if result.keypoints is not None:
                has_pose = True
                # shape: (M, 17, 3) with [x, y, visibility] per keypoint
                kpts_data = result.keypoints.data.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue
                all_bboxes.append(np.array([x1, y1, w, h], dtype=np.float64))
                all_confs.append(float(confs[i]))
                if kpts_data is not None and i < len(kpts_data):
                    all_keypoints.append(kpts_data[i].astype(np.float64))  # (17, 3)

        N = len(all_bboxes)
        if N == 0:
            self._mot.update(np.empty((0, 4)), np.empty(0), None, timestamp)
            return []

        bboxes_arr = np.stack(all_bboxes)
        confs_arr = np.array(all_confs)

        # Build keypoints array — only when all detections have pose data
        keypoints_arr: np.ndarray | None = None
        if has_pose and len(all_keypoints) == N:
            keypoints_arr = np.stack(all_keypoints)  # (N, 17, 3)

        bbox_tuples = [(b[0], b[1], b[2], b[3]) for b in all_bboxes]
        features = self._reid.extract(frame, bbox_tuples)
        self._total_extractions += N

        mot_results = self._mot.update(
            bboxes_arr, confs_arr, features, timestamp,
            keypoints=keypoints_arr,
        )

        tracks: list[Track] = []
        for tid, bbox_xywh, conf in mot_results:
            if tid not in self._first_seen:
                self._first_seen[tid] = timestamp

            tracklet = self._mot.active_tracks.get(tid)
            vel = tracklet.kf.velocity if tracklet else (0.0, 0.0)
            acc = tracklet.kf.acceleration if tracklet else (0.0, 0.0)
            kf_pos = tracklet.kf.position if tracklet else (
                float(bbox_xywh[0] + bbox_xywh[2] / 2),
                float(bbox_xywh[1] + bbox_xywh[3] / 2),
            )

            tracks.append(Track(
                track_id=tid,
                bbox=BoundingBox(
                    x=float(bbox_xywh[0]), y=float(bbox_xywh[1]),
                    w=float(bbox_xywh[2]), h=float(bbox_xywh[3]),
                ),
                confidence=conf,
                class_id="person",
                first_seen_ts=self._first_seen[tid],
                last_seen_ts=timestamp,
                age_frames=tracklet.total_frames if tracklet else 1,
                misses=tracklet.frames_since_update if tracklet else 0,
                velocity_px_per_s=vel,
                acceleration_px_per_s2=acc,
                mask_center=kf_pos,
            ))

        # Purge first_seen for tracks that are fully deleted
        active_ids = set(self._mot.active_tracks.keys())
        stale = [k for k in self._first_seen if k not in active_ids]
        for k in stale:
            del self._first_seen[k]

        return tracks

    @staticmethod
    def _compute_mask_centroid(masks: object, idx: int) -> tuple[float, float] | None:
        if masks is None:
            return None
        try:
            mask_tensor = masks.data[idx].cpu().numpy()
            binary = (mask_tensor > 0.5).astype(np.uint8)
            M = _cv2.moments(binary)
            if M["m00"] < 1.0:
                return None
            mcx = M["m10"] / M["m00"]
            mcy = M["m01"] / M["m00"]
            orig_h, orig_w = masks.orig_shape
            mask_h, mask_w = mask_tensor.shape
            return (float(mcx * orig_w / mask_w), float(mcy * orig_h / mask_h))
        except (IndexError, AttributeError, _cv2.error):
            return None
