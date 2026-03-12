"""
YoloSegTracker: YOLO11-Seg + BoT-SORT + Re-ID recovery.
=========================================================

Responsibilities (single):
    - Run YOLO-Seg inference with built-in BoT-SORT/ByteTrack tracking.
    - Output Track list with stable IDs, Kalman-smoothed bboxes, mask centroids.
    - Replaces separate YoloDetector + SimpleIoUTracker for production use.

Key features:
    - BoT-SORT maintains stable track IDs (first-pass association).
    - **Second-pass Re-ID recovery**: when BoT-SORT assigns a new ID after
      occlusion, the appearance gallery matches it against recently lost tracks
      using cosine similarity on MobileNetV3-Small features.  If a confident
      match is found, the original ID is restored — solving the occlusion
      ID-switch problem without modifying BoT-SORT internals.
    - Instance segmentation masks give pixel-tight contours (no oversized boxes).
    - Per-track **Constant-Acceleration Kalman filter** on mask centroid:
      smooth position, robust velocity, estimated acceleration, and
      parabolic inter-frame prediction for curved trajectory at low FPS.
    - cv2.moments for sub-pixel precise centroid computation.
    - Single model.track() call — simpler and faster than two-step pipeline.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Union

import cv2 as _cv2
import numpy as np

from qp_perception.kalman import (
    CentroidKalman2D,
    CentroidKalmanCA,
    KalmanCAConfig,
    KalmanConfig,
)
from qp_perception.reid.extractor import ReIDConfig, ReIDExtractor
from qp_perception.reid.gallery import AppearanceGallery, GalleryConfig
from qp_perception.tracking.cmc import CameraMotionCompensator
from qp_perception.types import BoundingBox, Track

logger = logging.getLogger(__name__)

# Type alias: either CV or CA filter
_KalmanFilter = Union[CentroidKalman2D, CentroidKalmanCA]


class YoloSegTracker:
    """
    Combined YOLO-Seg detection + BoT-SORT tracking in one call.

    Each detected track maintains its own Kalman filter that fuses raw mask
    centroids into a smooth (position, velocity[, acceleration]) estimate.

    Parameters
    ----------
    model_path : str
        Path to a seg model weight (e.g. ``"yolo11n-seg.pt"``).
    confidence_threshold : float
        Detections below this confidence are discarded.
    nms_iou_threshold : float
        IoU threshold for NMS.
    tracker : str
        Tracker config name: ``"botsort.yaml"`` (default) or ``"bytetrack.yaml"``.
    class_whitelist : optional sequence of class names
        If provided, only these COCO class names are kept.
    device : str
        ``"cuda:0"``, ``"cpu"``, or ``""`` for auto.
    img_size : int
        Input image size (longer side).
    kalman_config : KalmanCAConfig or KalmanConfig
        Pass ``KalmanCAConfig`` (default) to use the 6-state Constant-Acceleration
        model, or ``KalmanConfig`` for the 4-state Constant-Velocity model.
    """

    def __init__(
        self,
        model_path: str = "yolo11n-seg.pt",
        confidence_threshold: float = 0.40,
        nms_iou_threshold: float = 0.45,
        tracker: str = "botsort.yaml",
        class_whitelist: Sequence[str] | None = None,
        device: str = "",
        img_size: int = 640,
        kalman_config: KalmanCAConfig | KalmanConfig | None = None,
        enable_reid: bool = False,
        reid_config: ReIDConfig | None = None,
        gallery_config: GalleryConfig | None = None,
        enable_cmc: bool = False,
    ) -> None:
        from ultralytics import YOLO  # type: ignore[import-untyped]

        self._model = YOLO(model_path)
        self._conf = confidence_threshold
        self._iou = nms_iou_threshold
        self._tracker = tracker
        self._device = device
        self._img_size = img_size

        self._id_to_name: dict[int, str] = self._model.names
        self._allowed_ids: list[int] | None = None
        if class_whitelist is not None:
            name_lower_map = {v.lower(): k for k, v in self._id_to_name.items()}
            self._allowed_ids = [
                name_lower_map[n.lower()] for n in class_whitelist if n.lower() in name_lower_map
            ]
            if not self._allowed_ids:
                logger.warning(
                    "class_whitelist %s matched no model classes; detections will be empty.",
                    class_whitelist,
                )

        # Kalman filter configuration
        self._kalman_cfg = kalman_config or KalmanCAConfig()
        self._use_ca = isinstance(self._kalman_cfg, KalmanCAConfig)
        self._filters: dict[int, _KalmanFilter] = {}
        self._filter_last_seen: dict[int, float] = {}
        self._first_seen: dict[int, float] = {}
        self._last_ts: float = 0.0

        # Cache last raw ultralytics results (for visualization)
        self._last_raw_results: list | None = None

        # Re-ID recovery layer (second-pass after BoT-SORT)
        self._reid_enabled = enable_reid
        self._reid_extractor: ReIDExtractor | None = None
        self._gallery: AppearanceGallery | None = None
        self._known_botsort_ids: set[int] = set()
        self._id_remap: dict[int, int] = {}
        self._feature_cache: dict[int, np.ndarray] = {}
        self._feature_cache_age: dict[int, int] = {}
        self._feature_refresh_interval: int = 3

        # Selective ReID (Fast-Deep-OC-SORT, Bayar 2024)
        self._prev_bboxes: dict[int, tuple[float, float, float, float]] = {}
        self._selective_iou_threshold: float = 0.20
        self._selective_extractions: int = 0
        self._selective_skips: int = 0

        # CMC (Deep OC-SORT Sec.3.2): compensate camera ego-motion
        self._cmc_enabled = enable_cmc
        self._cmc: CameraMotionCompensator | None = None
        if enable_cmc:
            self._cmc = CameraMotionCompensator(downscale=2)
            logger.info("Camera Motion Compensation ENABLED")

        if enable_reid:
            self._reid_extractor = ReIDExtractor(reid_config)
            self._gallery = AppearanceGallery(gallery_config)
            logger.info("Re-ID recovery layer ENABLED")

        model_name = "CA (6-state)" if self._use_ca else "CV (4-state)"
        logger.info(
            "YoloSegTracker ready  model=%s  tracker=%s  conf=%.2f  "
            "kalman=%s  reid=%s  whitelist=%s  device=%s",
            model_path,
            tracker,
            self._conf,
            model_name,
            enable_reid,
            class_whitelist,
            device or "auto",
        )

    @property
    def last_raw_results(self) -> list | None:
        """Last raw ultralytics results (for visualization / debug)."""
        return self._last_raw_results

    @property
    def filters(self) -> dict[int, _KalmanFilter]:
        """Per-track Kalman filters (read-only access for visualization)."""
        return self._filters

    @property
    def gallery(self) -> AppearanceGallery | None:
        """Appearance gallery (read-only access for stats / debug)."""
        return self._gallery

    @property
    def reid_stats(self) -> dict[str, int]:
        """Live Re-ID statistics for monitoring."""
        if not self._reid_enabled or self._gallery is None:
            return {"enabled": 0, "active": 0, "lost": 0, "remaps": len(self._id_remap),
                    "extractions": 0, "skips": 0}
        return {
            "enabled": 1,
            "active": self._gallery.active_count,
            "lost": self._gallery.lost_count,
            "remaps": len(self._id_remap),
            "extractions": self._selective_extractions,
            "skips": self._selective_skips,
        }

    # ------------------------------------------------------------------
    # Public API — matches the CombinedTracker protocol
    # ------------------------------------------------------------------

    def detect_and_track(self, frame: object, timestamp: float) -> list[Track]:
        """
        Run YOLO-Seg + BoT-SORT on a single BGR frame.

        Returns a list of ``Track`` objects with:
          - Stable ``track_id`` from BoT-SORT.
          - ``mask_center`` from Kalman-filtered centroid (smooth & predicted).
          - ``velocity_px_per_s`` from Kalman state.
        """
        if not isinstance(frame, np.ndarray):
            logger.warning("YoloSegTracker received non-ndarray frame; returning empty.")
            return []

        # ── CMC (Deep OC-SORT Sec.3.2): correct Kalman states ──
        if self._cmc is not None:
            warp = self._cmc.compute(frame)
            R = warp[:2, :2]
            T = warp[:2, 2]
            for kf in self._filters.values():
                pos = np.array(kf.position, dtype=np.float64)
                kf._x[0:2] = R @ pos + T
                vel = np.array(kf.velocity, dtype=np.float64)
                kf._x[2:4] = R @ vel
                if hasattr(kf, 'acceleration') and len(kf._x) >= 6:
                    acc = np.array(kf.acceleration, dtype=np.float64)
                    kf._x[4:6] = R @ acc
                # Also rotate covariance (Deep OC-SORT Eq.1)
                n = len(kf._x)
                for block_start in range(0, n, 2):
                    if block_start + 1 < n:
                        blk = kf._P[block_start:block_start+2, block_start:block_start+2]
                        kf._P[block_start:block_start+2, block_start:block_start+2] = R @ blk @ R.T

        # ── Kalman predict step for ALL existing tracks ──
        dt = max(timestamp - self._last_ts, 1e-3) if self._last_ts > 0.0 else 0.033
        for kf in self._filters.values():
            kf.predict(dt)

        # ── YOLO-Seg + BoT-SORT inference ──
        results = self._model.track(
            source=frame,
            persist=True,
            tracker=self._tracker,
            conf=self._conf,
            iou=self._iou,
            imgsz=self._img_size,
            device=self._device or None,
            classes=self._allowed_ids,
            verbose=False,
        )
        self._last_raw_results = results

        tracks: list[Track] = []
        seen_ids: set = set()

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            has_ids = boxes.id is not None
            track_ids = boxes.id.cpu().numpy().astype(int) if has_ids else None

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)
            masks = result.masks

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                w, h = x2 - x1, y2 - y1
                if w <= 0 or h <= 0:
                    continue

                tid = int(track_ids[i]) if track_ids is not None else (i + 1)
                cls_name = self._id_to_name.get(int(cls_ids[i]), "unknown")
                conf = float(confs[i])
                seen_ids.add(tid)

                # Raw mask centroid via cv2.moments (sub-pixel precise)
                raw_mask_center = self._compute_mask_centroid(masks, i)
                bbox = BoundingBox(x=float(x1), y=float(y1), w=float(w), h=float(h))
                raw_center = raw_mask_center if raw_mask_center is not None else bbox.center

                # ── Kalman update step ──
                if tid not in self._filters:
                    self._filters[tid] = self._create_filter(raw_center[0], raw_center[1])
                else:
                    self._filters[tid].update(raw_center[0], raw_center[1])

                self._filter_last_seen[tid] = timestamp
                if tid not in self._first_seen:
                    self._first_seen[tid] = timestamp

                kf = self._filters[tid]
                filtered_pos = kf.position
                filtered_vel = kf.velocity
                filtered_acc = kf.acceleration if hasattr(kf, "acceleration") else (0.0, 0.0)
                mask_center = filtered_pos if raw_mask_center is not None else None

                tracks.append(
                    Track(
                        track_id=tid,
                        bbox=bbox,
                        confidence=conf,
                        class_id=cls_name,
                        first_seen_ts=self._first_seen[tid],
                        last_seen_ts=timestamp,
                        age_frames=1,
                        misses=0,
                        velocity_px_per_s=filtered_vel,
                        acceleration_px_per_s2=filtered_acc,
                        mask_center=mask_center,
                    )
                )

        # ── Re-ID second-pass recovery ──
        if self._reid_enabled and isinstance(frame, np.ndarray):
            tracks = self._apply_reid_recovery(frame, tracks, seen_ids, timestamp)

        # Purge Kalman filters for tracks not seen recently (grace period)
        stale_ids = [
            tid
            for tid in self._filters
            if tid not in seen_ids and (timestamp - self._filter_last_seen.get(tid, 0.0)) > 0.5
        ]
        for stale_id in stale_ids:
            del self._filters[stale_id]
            self._filter_last_seen.pop(stale_id, None)
            self._first_seen.pop(stale_id, None)

        self._last_ts = timestamp
        return tracks

    # ------------------------------------------------------------------
    # Re-ID recovery layer
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_iou(a: tuple[float, float, float, float],
                  b: tuple[float, float, float, float]) -> float:
        """IoU for (x, y, w, h) boxes."""
        ax1, ay1, aw, ah = a
        bx1, by1, bw, bh = b
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0.0:
            return 0.0
        union = aw * ah + bw * bh - inter
        return float(inter / union) if union > 1e-6 else 0.0

    def _is_risky(self, track: Track, all_prev_bboxes: dict) -> bool:
        """Selective ReID (Fast-Deep-OC-SORT, Bayar 2024, Sec.3.1-3.3).

        A detection is "non-risky" (safe to skip feature extraction) when:
          1. It has exactly ONE confirmed tracklet with IoU > threshold.
          2. Aspect ratio similarity is high (Eq.2-3 from the paper).

        Non-risky detections can be matched purely by motion/IoU — extracting
        their features wastes compute and can even *hurt* accuracy by confusing
        the feature matcher with similar-looking people.
        """
        if not all_prev_bboxes:
            return True

        cur_bbox = (track.bbox.x, track.bbox.y, track.bbox.w, track.bbox.h)
        iou_threshold = self._selective_iou_threshold

        candidates = 0
        for prev_bbox in all_prev_bboxes.values():
            iou = self._bbox_iou(cur_bbox, prev_bbox)
            if iou > iou_threshold:
                pw, ph = prev_bbox[2], prev_bbox[3]
                cw, ch = cur_bbox[2], cur_bbox[3]
                if pw > 1 and ph > 1 and cw > 1 and ch > 1:
                    import math
                    v = 1.0 - (4.0 / (math.pi * math.pi)) * (
                        math.atan(cw / ch) - math.atan(pw / ph)
                    ) ** 2
                    alpha_ars = v / ((1.0 - iou) + v) if (1.0 - iou + v) > 1e-6 else 1.0
                    if alpha_ars < 0.6:
                        continue
                candidates += 1

        return candidates != 1

    def _apply_reid_recovery(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        seen_ids: set[int],
        timestamp: float,
    ) -> list[Track]:
        """Second-pass Re-ID with Selective extraction and Feature Decay.

        Paper-guided optimizations:
        - **Selective ReID** (Fast-Deep-OC-SORT): Only extract features for
          "risky" detections (new IDs, ambiguous IoU, near other tracks).
          Non-risky detections skip extraction entirely → ~50% fewer CNN calls.
        - **Feature Decay** (ibid., Sec.3.4): When extraction is skipped for a
          track, decay the gallery feature's influence: ``α' ← α' × α``.
          Prevents stale features from dominating after long non-extraction.
        - **Dynamic Appearance** (Deep OC-SORT): confidence-modulated EMA.
        - **Adaptive Weighting** (Deep OC-SORT): discriminativeness boost.
        - **OCM** (OC-SORT): raw-observation velocity for lost track prediction.
        """
        assert self._reid_extractor is not None
        assert self._gallery is not None

        if not tracks:
            self._gallery.retire_missing(set(), timestamp)
            self._gallery.purge_expired(timestamp)
            return tracks

        need_extract_idx: list[int] = []
        skip_idx: list[int] = []
        cached_idx: list[int] = []

        for i, track in enumerate(tracks):
            bid = track.track_id
            is_new = bid not in self._known_botsort_ids and bid not in self._id_remap

            if is_new:
                need_extract_idx.append(i)
                continue

            risky = self._is_risky(track, self._prev_bboxes)

            if not risky and bid in self._feature_cache:
                skip_idx.append(i)
                self._selective_skips += 1
                continue

            cache_age = self._feature_cache_age.get(bid, 999)
            if cache_age >= self._feature_refresh_interval or bid not in self._feature_cache:
                need_extract_idx.append(i)
            else:
                cached_idx.append(i)

        features_map: dict[int, np.ndarray] = {}

        if need_extract_idx:
            bboxes = [(tracks[i].bbox.x, tracks[i].bbox.y,
                        tracks[i].bbox.w, tracks[i].bbox.h) for i in need_extract_idx]
            feats = self._reid_extractor.extract(frame, bboxes)
            for j, i in enumerate(need_extract_idx):
                bid = tracks[i].track_id
                if j < len(feats):
                    features_map[bid] = feats[j]
                    self._feature_cache[bid] = feats[j]
                    self._feature_cache_age[bid] = 0
            self._selective_extractions += len(need_extract_idx)

        for i in cached_idx:
            bid = tracks[i].track_id
            features_map[bid] = self._feature_cache[bid]
            self._feature_cache_age[bid] = self._feature_cache_age.get(bid, 0) + 1

        for i in skip_idx:
            bid = tracks[i].track_id
            features_map[bid] = self._feature_cache[bid]
            self._feature_cache_age[bid] = self._feature_cache_age.get(bid, 0) + 1

        new_tracks: list[Track] = []
        current_visible_ids: set[int] = set()
        positions: dict[int, tuple[float, float]] = {}
        velocities: dict[int, tuple[float, float]] = {}
        sizes: dict[int, tuple[float, float]] = {}
        next_prev_bboxes: dict[int, tuple[float, float, float, float]] = {}

        for track in tracks:
            botsort_id = track.track_id
            feat = features_map.get(botsort_id)

            final_id = botsort_id
            center = track.mask_center or track.bbox.center
            cur_bbox = (track.bbox.x, track.bbox.y, track.bbox.w, track.bbox.h)

            if botsort_id in self._id_remap:
                final_id = self._id_remap[botsort_id]

            elif botsort_id not in self._known_botsort_ids and feat is not None:
                match = self._gallery.query_lost(
                    feat,
                    timestamp,
                    position_hint=center,
                    bbox_hint=cur_bbox,
                    query_velocity=track.velocity_px_per_s,
                )
                if match is not None:
                    old_id = match.old_track_id
                    self._id_remap[botsort_id] = old_id
                    final_id = old_id

                    if botsort_id in self._filters:
                        self._filters[old_id] = self._filters.pop(botsort_id)
                        self._filter_last_seen[old_id] = self._filter_last_seen.pop(
                            botsort_id, timestamp
                        )
                        self._first_seen[old_id] = self._first_seen.pop(
                            botsort_id, timestamp
                        )

                    seen_ids.discard(botsort_id)
                    seen_ids.add(old_id)

                    logger.info(
                        "Re-ID REMAP: botsort_id=%d → recovered_id=%d  fused=%.3f  sim=%.3f",
                        botsort_id, old_id, match.fused_score, match.similarity,
                    )

            self._known_botsort_ids.add(botsort_id)

            was_skipped = botsort_id in {tracks[i].track_id for i in skip_idx}

            if feat is not None:
                self._gallery.update_active(
                    final_id, feat, timestamp,
                    confidence=track.confidence,
                    position=center,
                    feature_decay=was_skipped,
                )
            current_visible_ids.add(final_id)

            positions[final_id] = center
            velocities[final_id] = track.velocity_px_per_s
            sizes[final_id] = (track.bbox.w, track.bbox.h)
            next_prev_bboxes[final_id] = cur_bbox

            if final_id != botsort_id:
                track = Track(
                    track_id=final_id,
                    bbox=track.bbox,
                    confidence=track.confidence,
                    class_id=track.class_id,
                    first_seen_ts=self._first_seen.get(final_id, track.first_seen_ts),
                    last_seen_ts=track.last_seen_ts,
                    age_frames=track.age_frames,
                    misses=track.misses,
                    velocity_px_per_s=track.velocity_px_per_s,
                    acceleration_px_per_s2=track.acceleration_px_per_s2,
                    mask_center=track.mask_center,
                )

            new_tracks.append(track)

        self._prev_bboxes = next_prev_bboxes

        active_bids = {t.track_id for t in tracks}
        stale_cache = [k for k in self._feature_cache if k not in active_bids]
        for k in stale_cache:
            del self._feature_cache[k]
            self._feature_cache_age.pop(k, None)

        self._gallery.retire_missing(current_visible_ids, timestamp, positions, velocities, sizes)
        self._gallery.purge_expired(timestamp)

        return new_tracks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_filter(self, cx0: float, cy0: float) -> _KalmanFilter:
        """Instantiate the appropriate Kalman filter for a new track."""
        if self._use_ca:
            return CentroidKalmanCA(cx0=cx0, cy0=cy0, config=self._kalman_cfg)
        return CentroidKalman2D(cx0=cx0, cy0=cy0, config=self._kalman_cfg)

    @staticmethod
    def _compute_mask_centroid(masks: object, idx: int) -> tuple[float, float] | None:
        """
        Extract the centroid of the i-th segmentation mask using cv2.moments.

        cv2.moments gives sub-pixel precision by computing the first-order
        spatial moments of the binary mask, which is more accurate than a
        simple np.mean of pixel coordinates.
        """
        if masks is None:
            return None
        try:
            mask_tensor = masks.data[idx].cpu().numpy()
            # Convert to uint8 for cv2.moments
            binary = (mask_tensor > 0.5).astype(np.uint8)
            M = _cv2.moments(binary)
            if M["m00"] < 1.0:
                return None
            # Centroid in mask coordinates
            mcx = M["m10"] / M["m00"]
            mcy = M["m01"] / M["m00"]
            # Scale to original image coordinates
            orig_h, orig_w = masks.orig_shape
            mask_h, mask_w = mask_tensor.shape
            cx = mcx * (orig_w / mask_w)
            cy = mcy * (orig_h / mask_h)
            return (float(cx), float(cy))
        except (IndexError, AttributeError, _cv2.error) as exc:
            logger.warning("mask centroid computation failed for idx=%d: %s", idx, exc)
            return None
