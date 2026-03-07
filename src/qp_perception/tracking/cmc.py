"""
Camera Motion Compensation (CMC) — Deep OC-SORT, Sec.3.2.
==========================================================

Estimates frame-to-frame camera motion using sparse optical flow and
applies the inverse transform to Kalman filter states, so the motion
model only needs to capture *object* motion, not camera ego-motion.

Pipeline per frame:
    1. Detect good-to-track keypoints in the previous grayscale frame.
    2. Track them to the current frame via pyramidal Lucas-Kanade.
    3. Estimate a similarity (4-DOF) or affine (6-DOF) transform with RANSAC.
    4. Return the 2×3 warp matrix for downstream Kalman state correction.

Lightweight: ~1ms per 640px frame on CPU.  No GPU or external libs required.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraMotionCompensator:
    """Sparse-optical-flow based camera motion estimator.

    Parameters
    ----------
    downscale : int
        Factor to downscale frames before keypoint detection.
        2 → half resolution (faster, slightly less precise).
    max_keypoints : int
        Maximum number of Shi-Tomasi corners to detect.
    use_affine : bool
        If True, estimate full 6-DOF affine (rotation + scale + shear + tx/ty).
        If False, estimate 4-DOF similarity (rotation + uniform scale + tx/ty).
        Similarity is more robust for typical camera vibration.
    """

    def __init__(
        self,
        downscale: int = 2,
        max_keypoints: int = 200,
        use_affine: bool = False,
    ) -> None:
        self._downscale = max(1, downscale)
        self._max_kp = max_keypoints
        self._use_affine = use_affine
        self._prev_gray: np.ndarray | None = None

        self._lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

        self._feature_params = dict(
            maxCorners=self._max_kp,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7,
        )

    def compute(self, frame: np.ndarray) -> np.ndarray:
        """Compute the camera motion warp matrix from previous to current frame.

        Returns
        -------
        np.ndarray
            2×3 affine/similarity matrix.  Identity if no previous frame or
            estimation failed.
        """
        identity = np.eye(2, 3, dtype=np.float64)

        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        if self._downscale > 1:
            h, w = gray.shape[:2]
            small = cv2.resize(gray, (w // self._downscale, h // self._downscale))
        else:
            small = gray

        if self._prev_gray is None:
            self._prev_gray = small
            return identity

        prev = self._prev_gray
        curr = small

        pts = cv2.goodFeaturesToTrack(prev, **self._feature_params)
        if pts is None or len(pts) < 4:
            self._prev_gray = curr
            return identity

        pts_next, status, _ = cv2.calcOpticalFlowPyrLK(
            prev, curr, pts, None, **self._lk_params
        )
        if pts_next is None:
            self._prev_gray = curr
            return identity

        mask = status.flatten() == 1
        if mask.sum() < 4:
            self._prev_gray = curr
            return identity

        good_old = pts[mask]
        good_new = pts_next[mask]

        if self._use_affine:
            M, inliers = cv2.estimateAffine2D(
                good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
        else:
            M, inliers = cv2.estimateAffinePartial2D(
                good_old, good_new, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )

        if M is None:
            self._prev_gray = curr
            return identity

        if self._downscale > 1:
            M[0, 2] *= self._downscale
            M[1, 2] *= self._downscale

        self._prev_gray = curr
        return M

    def reset(self) -> None:
        """Reset internal state (e.g. on scene change)."""
        self._prev_gray = None
