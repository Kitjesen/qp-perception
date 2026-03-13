"""
Lightweight 2D Kalman Filters for centroid tracking.
=====================================================

Two models provided:

1. **CentroidKalman2D** — Constant-Velocity (CV), 4 states ``[cx, cy, vx, vy]``.
   Good for steady-speed targets.

2. **CentroidKalmanCA** — Constant-Acceleration (CA), 6 states
   ``[cx, cy, vx, vy, ax, ay]``.  Better for targets that speed up, slow down,
   or turn — acceleration is estimated and used for parabolic prediction.

Common API
----------
Both classes share the same public interface:

- ``predict(dt)``  — extrapolate state forward (call every frame, even without
  a measurement).
- ``update(cx, cy)`` — fuse a new centroid measurement.
- ``position``  — ``(cx, cy)`` best estimate.
- ``velocity``  — ``(vx, vy)`` best estimate in px/s.
- ``predict_position(dt_ahead)`` — read-only future extrapolation.

``CentroidKalmanCA`` additionally exposes:

- ``acceleration`` — ``(ax, ay)`` best estimate in px/s².

Pure numpy, no external dependencies, O(1) per step.

Units: position in pixels, velocity in px/s, acceleration in px/s², dt in seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanConfig:
    """Tuning knobs for the 2D centroid Kalman filter.

    Attributes
    ----------
    process_noise_pos : float
        Standard deviation of position process noise (pixels).
        Models random jitter in the centroid measurement.
    process_noise_vel : float
        Standard deviation of velocity process noise (pixels/s).
        Models how quickly the target can change speed.
    measurement_noise : float
        Standard deviation of measurement noise (pixels).
        Reflects the expected noise in the raw mask centroid.
    initial_velocity_var : float
        Initial variance for velocity states (px/s)^2.
        Large value → filter learns velocity quickly from first observations.
    """

    process_noise_pos: float = 3.0
    process_noise_vel: float = 15.0
    measurement_noise: float = 8.0
    initial_velocity_var: float = 200.0


class CentroidKalman2D:
    """
    4-state linear Kalman filter for pixel centroid + velocity.

    Typical usage::

        kf = CentroidKalman2D(cx0, cy0)
        # each frame:
        kf.predict(dt)          # always call first
        if measurement_available:
            kf.update(meas_cx, meas_cy)
        pos = kf.position       # smoothed (cx, cy)
        vel = kf.velocity       # estimated (vx, vy) px/s
    """

    def __init__(
        self,
        cx0: float,
        cy0: float,
        vx0: float = 0.0,
        vy0: float = 0.0,
        config: KalmanConfig = KalmanConfig(),
    ) -> None:
        # State: [cx, cy, vx, vy]
        self._x = np.array([cx0, cy0, vx0, vy0], dtype=np.float64)

        # Covariance
        iv = config.initial_velocity_var
        self._P = np.diag(
            [
                config.measurement_noise**2,
                config.measurement_noise**2,
                iv,
                iv,
            ]
        ).astype(np.float64)

        # Measurement matrix: observe [cx, cy]
        self._H = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
            ],
            dtype=np.float64,
        )

        # Measurement noise
        r = config.measurement_noise**2
        self._R = np.diag([r, r]).astype(np.float64)

        # Process noise config (used to build Q each predict step)
        self._q_pos = config.process_noise_pos
        self._q_vel = config.process_noise_vel

        # Identity for convenience
        self._I4 = np.eye(4, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Propagate state forward by *dt* seconds (constant-velocity model)."""
        if dt <= 0.0:
            return

        F = self._transition(dt)
        Q = self._process_noise(dt)

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q

    def update(self, cx: float, cy: float) -> None:
        """Fuse a new centroid measurement."""
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self._H @ self._x  # innovation
        S = self._H @ self._P @ self._H.T + self._R  # innovation covariance
        K = self._P @ self._H.T @ np.linalg.inv(S)  # Kalman gain

        self._x = self._x + K @ y
        self._P = (self._I4 - K @ self._H) @ self._P

    @property
    def position(self) -> tuple[float, float]:
        """Current estimated position (cx, cy) in pixels."""
        return (float(self._x[0]), float(self._x[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        """Current estimated velocity (vx, vy) in px/s."""
        return (float(self._x[2]), float(self._x[3]))

    @property
    def state(self) -> np.ndarray:
        """Full state vector [cx, cy, vx, vy] (read-only copy)."""
        return self._x.copy()

    def predict_position(self, dt_ahead: float) -> tuple[float, float]:
        """Extrapolate position *dt_ahead* seconds into the future (read-only)."""
        cx = self._x[0] + self._x[2] * dt_ahead
        cy = self._x[1] + self._x[3] * dt_ahead
        return (float(cx), float(cy))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        """State transition matrix F(dt)."""
        return np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        """
        Discrete-time process noise Q(dt).

        Uses the piecewise-white-noise model where acceleration is the noise
        source:  q_vel drives velocity noise, and position noise is coupled
        through dt.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        qp = self._q_pos**2
        qv = self._q_vel**2
        return np.array(
            [
                [qp + qv * dt3 / 3, 0, qv * dt2 / 2, 0],
                [0, qp + qv * dt3 / 3, 0, qv * dt2 / 2],
                [qv * dt2 / 2, 0, qv * dt, 0],
                [0, qv * dt2 / 2, 0, qv * dt],
            ],
            dtype=np.float64,
        )


# ======================================================================
# Constant-Acceleration (CA) model — 6 states
# ======================================================================


@dataclass
class KalmanCAConfig:
    """Tuning knobs for the Constant-Acceleration Kalman filter.

    Attributes
    ----------
    process_noise_pos : float
        Position process noise σ (pixels).
    process_noise_vel : float
        Velocity process noise σ (pixels/s).
    process_noise_acc : float
        Acceleration process noise σ (pixels/s²).
        Controls how rapidly the filter adapts to acceleration changes.
    measurement_noise : float
        Measurement noise σ (pixels).
    initial_velocity_var : float
        Initial velocity variance (px/s)².
    initial_accel_var : float
        Initial acceleration variance (px/s²)².
    """

    process_noise_pos: float = 2.0
    process_noise_vel: float = 10.0
    process_noise_acc: float = 30.0
    measurement_noise: float = 6.0
    initial_velocity_var: float = 200.0
    initial_accel_var: float = 500.0

    # Adaptive Q: scale acceleration process noise by estimated target speed.
    # When True, faster-moving targets receive proportionally larger Q_acc
    # so the filter adapts more aggressively to sudden maneuvers.
    # speed_ref_px_s: reference speed (px/s); at this speed Q_acc is doubled.
    # max_scale: maximum Q_acc multiplier (applied at >= 2×speed_ref).
    adaptive_q_enabled: bool = False
    adaptive_q_speed_ref_px_s: float = 150.0
    adaptive_q_max_scale: float = 3.0


class CentroidKalmanCA:
    """
    6-state Kalman filter: Constant-Acceleration model.

    State vector  x = [cx, cy, vx, vy, ax, ay]^T

    Compared to CentroidKalman2D (CV):
      - Tracks acceleration explicitly → better during speed changes & turns.
      - predict_position uses parabolic extrapolation (not linear).
      - predict_trajectory returns a list of future points for smooth arc display.

    Typical usage::

        kf = CentroidKalmanCA(cx0, cy0)
        kf.predict(dt)
        kf.update(meas_cx, meas_cy)
        pos = kf.position        # (cx, cy)
        vel = kf.velocity        # (vx, vy)
        acc = kf.acceleration    # (ax, ay)
        trail = kf.predict_trajectory(horizon_s=0.5, steps=8)
    """

    def __init__(
        self,
        cx0: float,
        cy0: float,
        vx0: float = 0.0,
        vy0: float = 0.0,
        config: KalmanCAConfig = KalmanCAConfig(),
    ) -> None:
        # State: [cx, cy, vx, vy, ax, ay]
        self._x = np.array([cx0, cy0, vx0, vy0, 0.0, 0.0], dtype=np.float64)

        # Covariance
        r2 = config.measurement_noise**2
        iv = config.initial_velocity_var
        ia = config.initial_accel_var
        self._P = np.diag([r2, r2, iv, iv, ia, ia]).astype(np.float64)

        # Measurement matrix: observe [cx, cy]
        self._H = np.zeros((2, 6), dtype=np.float64)
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0

        # Measurement noise
        self._R = np.diag([r2, r2]).astype(np.float64)

        # Process noise parameters
        self._q_pos = config.process_noise_pos
        self._q_vel = config.process_noise_vel
        self._q_acc = config.process_noise_acc
        self._adaptive_q_enabled: bool = config.adaptive_q_enabled
        self._adaptive_q_speed_ref: float = config.adaptive_q_speed_ref_px_s
        self._adaptive_q_max_scale: float = config.adaptive_q_max_scale

        self._I6 = np.eye(6, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Propagate state forward by *dt* seconds (constant-acceleration model).

        When ``adaptive_q_enabled`` is True (via KalmanCAConfig), the
        acceleration process noise Q_acc is scaled by the current estimated
        target speed: faster targets → larger Q_acc → faster adaptation to
        sudden maneuvers.  The base ``_q_acc`` value is restored after the
        predict step so it remains stable for inspection.
        """
        if dt <= 0.0:
            return
        q_acc_save = self._q_acc
        if self._adaptive_q_enabled:
            speed = math.sqrt(float(self._x[2]) ** 2 + float(self._x[3]) ** 2)
            scale = min(1.0 + speed / self._adaptive_q_speed_ref, self._adaptive_q_max_scale)
            self._q_acc = q_acc_save * scale
        F = self._transition(dt)
        Q = self._process_noise(dt)
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._q_acc = q_acc_save  # restore base value

    def predict_with_ego_motion(
        self,
        dt: float,
        d_yaw_deg: float,
        d_pitch_deg: float,
        fx: float,
        fy: float,
    ) -> None:
        """Ego-Motion Aware Prediction (EMAP).

        Applies standard CA prediction then corrects the predicted position
        for camera rotation, so that world-fixed objects are not mistaken
        for moving targets due to gimbal platform motion.

        Physics (small-angle approximation):
            Camera rotates right by Δψ → world-fixed point shifts left:
                Δcx = -fx · tan(Δψ_rad)   (pixels)
            Camera tilts up by Δθ   → world-fixed point shifts down:
                Δcy =  fy · tan(Δθ_rad)   (pixels)

        Parameters
        ----------
        dt : float
            Integration timestep (seconds).
        d_yaw_deg : float
            Camera yaw change since last frame (degrees, positive = rightward).
        d_pitch_deg : float
            Camera pitch change since last frame (degrees, positive = upward).
        fx, fy : float
            Focal lengths in pixels (camera intrinsics).
        """
        self.predict(dt)  # standard CA prediction first
        # Ego-motion compensation: offset predicted position by the pixel
        # displacement that camera rotation would impose on a world-fixed point.
        dx_cam = -fx * math.tan(math.radians(d_yaw_deg))
        dy_cam = fy * math.tan(math.radians(d_pitch_deg))
        self._x[0] += dx_cam
        self._x[1] += dy_cam

    def update(self, cx: float, cy: float) -> None:
        """Fuse a new centroid measurement."""
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (self._I6 - K @ self._H) @ self._P

    def update_with_confidence(self, cx: float, cy: float, confidence: float = 1.0) -> None:
        """Fuse a centroid measurement with confidence-scaled noise.

        Higher detection confidence → smaller R → measurement trusted more.
        Lower confidence → larger R → Kalman prediction trusted more.

        Parameters
        ----------
        cx, cy : float
            Measured centroid position (pixels).
        confidence : float
            Detection confidence in [0, 1]. At 1.0 behaves identically to
            ``update()``.  At 0.35 (typical threshold) R is scaled ×2.9,
            letting the Kalman prediction dominate over a noisy bbox.
        """
        if confidence <= 0.0:
            return
        # r_scale: confidence=1.0 → 1.0x, confidence=0.5 → 2.0x, clamped to [0.5, 4.0]
        r_scale = max(0.5, min(4.0, 1.0 / max(confidence, 0.1)))
        z = np.array([cx, cy], dtype=np.float64)
        y = z - self._H @ self._x
        R_scaled = self._R * r_scale
        S = self._H @ self._P @ self._H.T + R_scaled
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self._x = self._x + K @ y
        self._P = (self._I6 - K @ self._H) @ self._P

    @property
    def position(self) -> tuple[float, float]:
        return (float(self._x[0]), float(self._x[1]))

    @property
    def velocity(self) -> tuple[float, float]:
        return (float(self._x[2]), float(self._x[3]))

    @property
    def acceleration(self) -> tuple[float, float]:
        return (float(self._x[4]), float(self._x[5]))

    @property
    def state(self) -> np.ndarray:
        return self._x.copy()

    def predict_position(self, dt_ahead: float) -> tuple[float, float]:
        """Parabolic extrapolation: p + v*t + 0.5*a*t²."""
        t = dt_ahead
        t2 = t * t
        cx = self._x[0] + self._x[2] * t + 0.5 * self._x[4] * t2
        cy = self._x[1] + self._x[3] * t + 0.5 * self._x[5] * t2
        return (float(cx), float(cy))

    def predict_trajectory(
        self, horizon_s: float = 0.5, steps: int = 8
    ) -> list[tuple[float, float]]:
        """Return a list of predicted future positions (parabolic arc)."""
        dt_step = horizon_s / steps
        return [self.predict_position(dt_step * i) for i in range(1, steps + 1)]

    def blend_velocity(self, vx: float, vy: float, alpha: float) -> None:
        """Blend observed velocity into the Kalman velocity state.

        Used by OC-SORT ORU (Observation-Centric Re-Update) after an occlusion
        gap to correct velocity drift without exposing the internal state vector.

        Parameters
        ----------
        vx, vy : float
            Observed velocity to blend in (px/s).
        alpha : float
            Weight given to observed velocity in [0, 1].
            0 = keep Kalman unchanged, 1 = replace with observed.
        """
        self._x[2] = alpha * vx + (1.0 - alpha) * float(self._x[2])
        self._x[3] = alpha * vy + (1.0 - alpha) * float(self._x[3])

    @property
    def covariance_2x2(self) -> np.ndarray:
        """2×2 position covariance submatrix [P_cx, P_cy] (copy)."""
        return self._P[:2, :2].copy()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        """State transition F(dt) for constant-acceleration model."""
        dt2 = 0.5 * dt * dt
        return np.array(
            [
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )

    def _process_noise(self, dt: float) -> np.ndarray:
        """
        Discrete-time process noise Q(dt) for CA model.

        Jerk (da/dt) is the noise source.  The covariance couples
        through the kinematic integrals:
          pos  += 0.5 * dt² * jerk
          vel  += dt * jerk
          acc  += jerk
        Each axis is independent, so Q is block-diagonal 2x(3x3).
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        dt5 = dt4 * dt

        qp = self._q_pos**2
        qv = self._q_vel**2
        qa = self._q_acc**2

        # Per-axis 3x3 block (pos, vel, acc coupling from jerk noise)
        q_block = qa * np.array(
            [
                [dt5 / 20, dt4 / 8, dt3 / 6],
                [dt4 / 8, dt3 / 3, dt2 / 2],
                [dt3 / 6, dt2 / 2, dt],
            ],
            dtype=np.float64,
        )

        # Add direct pos/vel noise
        q_block[0, 0] += qp * dt + qv * dt3 / 3
        q_block[0, 1] += qv * dt2 / 2
        q_block[1, 0] += qv * dt2 / 2
        q_block[1, 1] += qv * dt

        Q = np.zeros((6, 6), dtype=np.float64)
        # x-axis: indices 0, 2, 4
        # y-axis: indices 1, 3, 5
        idx_x = [0, 2, 4]
        idx_y = [1, 3, 5]
        for i in range(3):
            for j in range(3):
                Q[idx_x[i], idx_x[j]] = q_block[i, j]
                Q[idx_y[i], idx_y[j]] = q_block[i, j]

        return Q


# ---------------------------------------------------------------------------
# Backward-compatible aliases / adapters
# ---------------------------------------------------------------------------


class ConstantVelocityKalman2D:
    """Adapter that mirrors the benchmark/test API for CentroidKalman2D.

    Constructor accepts ``process_noise`` and ``measurement_noise`` scalars
    instead of a ``KalmanConfig`` object, and starts at the origin.
    ``update()`` accepts an optional ``timestamp`` keyword arg (ignored).
    """

    def __init__(
        self,
        cx0: float = 0.0,
        cy0: float = 0.0,
        *,
        process_noise: float = 3.0,
        measurement_noise: float = 8.0,
    ) -> None:
        config = KalmanConfig(
            process_noise_pos=process_noise,
            process_noise_vel=process_noise * 5.0,
            measurement_noise=measurement_noise,
        )
        self._inner = CentroidKalman2D(cx0, cy0, config=config)

    def predict(self, dt: float) -> None:
        self._inner.predict(dt)

    def update(self, cx: float, cy: float, *, timestamp: float | None = None) -> None:
        self._inner.update(cx, cy)

    @property
    def position(self) -> tuple[float, float]:
        return self._inner.position

    @property
    def velocity(self) -> tuple[float, float]:
        return self._inner.velocity

    def predict_position(self, dt_ahead: float) -> tuple[float, float]:
        return self._inner.predict_position(dt_ahead)


ConstantAccelerationKalman2D = CentroidKalmanCA
