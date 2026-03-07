"""IoU-based multi-object tracker with stable IDs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import linear_sum_assignment

from qp_perception.types import BoundingBox, Detection, Track


def _iou(a: BoundingBox, b: BoundingBox) -> float:
    ax2, ay2 = a.x + a.w, a.y + a.h
    bx2, by2 = b.x + b.w, b.y + b.h
    ix1, iy1 = max(a.x, b.x), max(a.y, b.y)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0.0 else 0.0


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.2, max_misses: int = 8) -> None:
        self._iou_threshold = iou_threshold
        self._max_misses = max_misses
        self._next_id = 1
        self._tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection], timestamp: float) -> list[Track]:
        matches: dict[int, int] = {}
        used_det: set[int] = set()
        track_ids = list(self._tracks.keys())

        if track_ids and detections:
            # Build IoU cost matrix and solve with Hungarian algorithm
            cost = np.zeros((len(track_ids), len(detections)))
            for i, tid in enumerate(track_ids):
                for j, det in enumerate(detections):
                    cost[i, j] = _iou(self._tracks[tid].bbox, det.bbox)
            row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
            for i, j in zip(row_ind, col_ind):
                if cost[i, j] >= self._iou_threshold:
                    matches[track_ids[i]] = j
                    used_det.add(j)

        for tid, det_idx in matches.items():
            det = detections[det_idx]
            prev = self._tracks[tid]
            prev_cx, prev_cy = prev.bbox.center
            new_cx, new_cy = det.bbox.center
            dt = max(timestamp - prev.last_seen_ts, 1e-3)
            vx, vy = (new_cx - prev_cx) / dt, (new_cy - prev_cy) / dt
            self._tracks[tid] = Track(
                track_id=tid,
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                first_seen_ts=prev.first_seen_ts,
                last_seen_ts=timestamp,
                age_frames=prev.age_frames + 1,
                misses=0,
                velocity_px_per_s=(vx, vy),
            )

        for tid in track_ids:
            if tid in matches:
                continue
            self._tracks[tid] = replace(self._tracks[tid], misses=self._tracks[tid].misses + 1)

        for det_idx, det in enumerate(detections):
            if det_idx in used_det:
                continue
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                track_id=tid,
                bbox=det.bbox,
                confidence=det.confidence,
                class_id=det.class_id,
                first_seen_ts=timestamp,
                last_seen_ts=timestamp,
            )

        stale_ids = [tid for tid, t in self._tracks.items() if t.misses > self._max_misses]
        for tid in stale_ids:
            del self._tracks[tid]

        return sorted(self._tracks.values(), key=lambda t: t.track_id)
