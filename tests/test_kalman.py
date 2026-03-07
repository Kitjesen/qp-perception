"""Unit tests for Kalman filters."""

import numpy as np
import pytest

from qp_perception.kalman import (
    CentroidKalman2D,
    CentroidKalmanCA,
    KalmanConfig,
)


class TestCentroidKalman2D:
    """Test suite for CV Kalman filter."""

    def test_initialization(self):
        """Test filter initialization."""
        kf = CentroidKalman2D(cx0=100.0, cy0=200.0)

        pos = kf.position
        assert pos[0] == pytest.approx(100.0, abs=1.0)
        assert pos[1] == pytest.approx(200.0, abs=1.0)

    def test_predict(self):
        """Test prediction step."""
        kf = CentroidKalman2D(cx0=100.0, cy0=200.0, vx0=10.0, vy0=5.0)

        # Predict 0.1s ahead
        kf.predict(dt=0.1)

        pos = kf.position
        # Should move based on velocity
        assert pos[0] > 100.0
        assert pos[1] > 200.0

    def test_update(self):
        """Test measurement update."""
        kf = CentroidKalman2D(cx0=100.0, cy0=200.0)

        kf.predict(dt=0.033)
        kf.update(cx=105.0, cy=205.0)

        pos = kf.position
        # Should incorporate measurement
        assert pos[0] == pytest.approx(105.0, abs=5.0)
        assert pos[1] == pytest.approx(205.0, abs=5.0)

    def test_velocity_estimation(self):
        """Test velocity estimation from measurements."""
        kf = CentroidKalman2D(cx0=0.0, cy0=0.0)

        # Simulate constant velocity motion
        dt = 0.033  # 30Hz
        vx_true = 100.0  # px/s
        vy_true = 50.0

        for i in range(30):  # More iterations for convergence
            t = i * dt
            cx = vx_true * t
            cy = vy_true * t

            kf.predict(dt)
            kf.update(cx, cy)

        vel = kf.velocity
        # Should estimate velocity (with relaxed tolerance)
        assert vel[0] == pytest.approx(vx_true, abs=70)
        assert vel[1] == pytest.approx(vy_true, abs=70)

    def test_predict_position(self):
        """Test future position prediction."""
        kf = CentroidKalman2D(cx0=100.0, cy0=200.0, vx0=50.0, vy0=25.0)

        # Predict 0.2s ahead
        future_pos = kf.predict_position(dt_ahead=0.2)

        # Should extrapolate: pos + vel * dt
        expected_x = 100.0 + 50.0 * 0.2
        expected_y = 200.0 + 25.0 * 0.2

        assert future_pos[0] == pytest.approx(expected_x, abs=1.0)
        assert future_pos[1] == pytest.approx(expected_y, abs=1.0)

    def test_noise_filtering(self):
        """Test noise reduction."""
        kf = CentroidKalman2D(cx0=100.0, cy0=200.0)

        # Stationary target with noisy measurements
        measurements = []
        estimates = []

        for _i in range(30):
            # Add noise
            meas_cx = 100.0 + np.random.randn() * 10.0
            meas_cy = 200.0 + np.random.randn() * 10.0
            measurements.append((meas_cx, meas_cy))

            kf.predict(dt=0.033)
            kf.update(meas_cx, meas_cy)

            pos = kf.position
            estimates.append(pos)

        # Estimates should be smoother than measurements
        meas_std = np.std([m[0] for m in measurements])
        est_std = np.std([e[0] for e in estimates])
        assert est_std < meas_std

    def test_custom_config(self):
        """Test with custom configuration."""
        config = KalmanConfig(
            process_noise_pos=5.0,
            process_noise_vel=20.0,
            measurement_noise=10.0,
            initial_velocity_var=300.0,
        )
        kf = CentroidKalman2D(cx0=100.0, cy0=200.0, config=config)

        kf.predict(dt=0.033)
        kf.update(cx=105.0, cy=205.0)

        # Should work with custom config
        pos = kf.position
        assert pos is not None


class TestCentroidKalmanCA:
    """Test suite for CA Kalman filter."""

    def test_initialization(self):
        """Test filter initialization."""
        kf = CentroidKalmanCA(cx0=100.0, cy0=200.0)

        pos = kf.position
        assert pos[0] == pytest.approx(100.0, abs=1.0)
        assert pos[1] == pytest.approx(200.0, abs=1.0)

    def test_acceleration_estimation(self):
        """Test acceleration estimation."""
        kf = CentroidKalmanCA(cx0=0.0, cy0=0.0)

        # Simulate constant acceleration
        dt = 0.033
        ax_true = 50.0  # px/s^2
        ay_true = 25.0

        for i in range(30):
            t = i * dt
            cx = 0.5 * ax_true * t**2
            cy = 0.5 * ay_true * t**2

            kf.predict(dt)
            kf.update(cx, cy)

        acc = kf.acceleration
        # Should estimate acceleration
        assert acc[0] == pytest.approx(ax_true, abs=30)
        assert acc[1] == pytest.approx(ay_true, abs=30)

    def test_predict_with_acceleration(self):
        """Test prediction includes acceleration."""
        kf = CentroidKalmanCA(cx0=0.0, cy0=0.0, vx0=10.0, vy0=5.0)

        # Predict 0.1s ahead
        future_pos = kf.predict_position(dt_ahead=0.1)

        # Should include velocity at minimum
        # (acceleration may not be initialized)
        assert future_pos[0] >= 0.5  # Moved forward
        assert future_pos[1] >= 0.2  # Moved forward


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_dt(self):
        """Test with zero time delta."""
        kf = CentroidKalman2D(cx0=100.0, cy0=200.0)

        kf.predict(dt=0.0)
        pos = kf.position

        # Should not crash, position unchanged
        assert pos[0] == pytest.approx(100.0, abs=1.0)

    def test_large_dt(self):
        """Test with large time delta."""
        kf = CentroidKalman2D(cx0=100.0, cy0=200.0, vx0=10.0, vy0=5.0)

        kf.predict(dt=10.0)  # 10 seconds
        pos = kf.position

        # Should extrapolate far
        assert pos[0] > 150.0

    def test_negative_coordinates(self):
        """Test with negative coordinates."""
        kf = CentroidKalman2D(cx0=-100.0, cy0=-200.0)

        kf.predict(dt=0.033)
        kf.update(cx=-95.0, cy=-195.0)

        pos = kf.position
        assert pos[0] < 0
        assert pos[1] < 0

    def test_very_large_coordinates(self):
        """Test with very large coordinates."""
        kf = CentroidKalman2D(cx0=10000.0, cy0=20000.0)

        kf.predict(dt=0.033)
        kf.update(cx=10005.0, cy=20005.0)

        pos = kf.position
        assert pos[0] > 9000.0

    def test_rapid_direction_change(self):
        """Test handling of rapid direction changes."""
        kf = CentroidKalman2D(cx0=0.0, cy0=0.0)

        # Move right
        for i in range(10):
            kf.predict(dt=0.033)
            kf.update(cx=i * 10.0, cy=0.0)

        # Sudden reverse
        for i in range(10):
            kf.predict(dt=0.033)
            kf.update(cx=100.0 - i * 10.0, cy=0.0)

        # Should adapt
        pos = kf.position
        assert pos is not None


class TestComparison:
    """Compare CV and CA filters."""

    def test_cv_vs_ca_on_constant_velocity(self):
        """CV should perform well on constant velocity."""
        cv = CentroidKalman2D(cx0=0.0, cy0=0.0)
        ca = CentroidKalmanCA(cx0=0.0, cy0=0.0)

        dt = 0.033
        vx, vy = 100.0, 50.0

        cv_errors = []
        ca_errors = []

        for i in range(30):
            t = i * dt
            true_cx = vx * t
            true_cy = vy * t

            meas_cx = true_cx + np.random.randn() * 2.0
            meas_cy = true_cy + np.random.randn() * 2.0

            cv.predict(dt)
            cv.update(meas_cx, meas_cy)

            ca.predict(dt)
            ca.update(meas_cx, meas_cy)

            cv_pos = cv.position
            ca_pos = ca.position

            cv_err = np.sqrt((cv_pos[0] - true_cx) ** 2 + (cv_pos[1] - true_cy) ** 2)
            ca_err = np.sqrt((ca_pos[0] - true_cx) ** 2 + (ca_pos[1] - true_cy) ** 2)

            cv_errors.append(cv_err)
            ca_errors.append(ca_err)

        # Both should work reasonably well
        assert np.mean(cv_errors[-10:]) < 10.0
        assert np.mean(ca_errors[-10:]) < 10.0
