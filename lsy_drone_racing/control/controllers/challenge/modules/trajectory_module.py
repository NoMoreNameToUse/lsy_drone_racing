from scipy.interpolate import PchipInterpolator
import numpy as np


class SplineTrajectory:
    """
    Shape-preserving trajectory interpolation.

    Uses PCHIP instead of CubicSpline to reduce overshoot / S-shaped wiggles
    near gate entry-center-exit waypoints.
    """

    def __init__(self, waypoints, t, freq):
        self._t_total = float(t[-1])
        self._freq = freq

        waypoints = np.asarray(waypoints, dtype=float)
        t = np.asarray(t, dtype=float)

        if len(waypoints) != len(t):
            raise ValueError(
                f"waypoints and t must have same length, got "
                f"{len(waypoints)} and {len(t)}"
            )

        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing")

        self._pos_spline = PchipInterpolator(t, waypoints, axis=0)
        self._vel_spline = self._pos_spline.derivative()

        self._time_grid = np.linspace(
            0.0,
            self._t_total,
            int(freq * self._t_total),
        )

        self._pos = self._pos_spline(self._time_grid)
        self._vel = self._vel_spline(self._time_grid)

        # Keep yaw simple for now.
        self._yaw = np.zeros(len(self._pos))

    def sample_horizon(self, tick, N):
        i = min(tick, len(self._pos) - 1 - N)
        i = max(i, 0)

        return {
            "pos": self._pos[i : i + N],
            "vel": self._vel[i : i + N],
            "yaw": self._yaw[i : i + N],
            "pos_terminal": self._pos[i + N],
            "vel_terminal": self._vel[i + N],
            "yaw_terminal": self._yaw[i + N],
        }