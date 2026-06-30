"""Improved spline trajectory with geometry/timing separation and yaw.

``ImprovedSplineTrajectory`` replaces the simple "PCHIP-through-waypoints-in-time"
approach with a clean three-stage construction that decouples *where* the path
goes from *when* the drone is there:

1. **Geometric path** ``p(s)`` -- the waypoints are fitted with a cubic B-spline
   and re-parameterized by cumulative **arc length** ``s`` (so ``||dp/ds|| ~= 1``,
   a unit-speed geometric curve independent of timing).
2. **Timing map** ``s(t)`` -- the per-waypoint timestamps ``t`` are turned into a
   **monotone** arc-length schedule via shape-preserving PCHIP, so the drone only
   ever moves forward along the path.
3. **Timed trajectory** -- the two are composed, ``r(t) = p(s(t))``, and sampled
   on the control grid to give position, velocity (unit tangent x scheduled
   speed), acceleration and a generated **yaw** (path heading) -- no longer zero.

The public surface (``_pos``, ``_vel``, ``_yaw``, ``_time_grid``,
``sample_horizon``) matches the legacy ``SplineTrajectory`` so it is a drop-in
replacement; extra signals (``_acc``, ``_yaw_rate``, ``_s_grid``) and the raw
maps (``path``, ``s_of_t``) are exposed for planning/diagnostics.

The legacy ``SplineTrajectory`` is kept below unchanged for comparison.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator, make_interp_spline

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ImprovedSplineTrajectory:
    """Arc-length B-spline path composed with a monotone timing map, plus yaw."""

    def __init__(
        self,
        waypoints: NDArray[np.floating],
        t: NDArray[np.floating],
        freq: float,
        *,
        min_yaw_speed: float = 0.15,
        yaw_smoothing: float = 0.0,
    ):
        """Build the timed trajectory from waypoints and their timestamps.

        Args:
            waypoints: ``(M, 3)`` path waypoints in world coordinates.
            t: ``(M,)`` strictly increasing timestamps, one per waypoint.
            freq: Control/sampling frequency (Hz) of the dense output grid.
            min_yaw_speed: Horizontal speed (m/s) below which yaw is held at its
                previous value instead of following the (ill-defined) heading.
            yaw_smoothing: Optional 0..1 exponential smoothing factor for yaw
                (0 = none).
        """
        waypoints = np.asarray(waypoints, dtype=float)
        t = np.asarray(t, dtype=float)

        if len(waypoints) != len(t):
            raise ValueError(
                f"waypoints and t must have same length, got {len(waypoints)} and {len(t)}"
            )
        if np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing")

        self._freq = freq
        self._t_total = float(t[-1])
        self._min_yaw_speed = float(min_yaw_speed)
        self._yaw_smoothing = float(yaw_smoothing)

        # Drop consecutive coincident waypoints (and their timestamps); a cubic
        # B-spline needs a strictly increasing parameter.
        seg = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        keep = np.concatenate([[True], seg > 1e-9])
        wp = waypoints[keep]
        tk = t[keep]

        # Stage 1 + 2.
        s_wp = self._build_geometric_path(wp)
        self._build_timing_map(tk, s_wp)
        # Stage 3.
        self._build_timed_trajectory()

    # ------------------------------------------------------------------ stage 1
    def _build_geometric_path(self, wp: NDArray[np.floating]) -> NDArray[np.floating]:
        """Fit p(u) (cubic B-spline) and re-parameterize by arc length s.

        Returns the arc length at each input waypoint (used by the timing map).
        """
        if len(wp) < 2:
            # Degenerate: a single point -> constant path.
            self._geom = None
            self._point = wp[0] if len(wp) else np.zeros(3)
            self._path_length = 0.0
            self._u_of_s = None
            self._geom_d = None
            return np.array([0.0])

        # Chord-length parameterization conditions the fit far better than a
        # uniform one when waypoint spacing is uneven (gate entry/center/exit).
        chord = np.linalg.norm(np.diff(wp, axis=0), axis=1)
        u = np.concatenate([[0.0], np.cumsum(chord)])

        k = min(3, len(wp) - 1)  # cubic when possible, lower degree for few points
        geom = make_interp_spline(u, wp, k=k, axis=0)
        geom_d = geom.derivative()

        # True arc length by dense sampling of the fitted curve.
        n_dense = max(200, 30 * (len(wp) - 1))
        u_dense = np.linspace(u[0], u[-1], n_dense)
        p_dense = geom(u_dense)
        ds = np.linalg.norm(np.diff(p_dense, axis=0), axis=1)
        s_dense = np.concatenate([[0.0], np.cumsum(ds)])
        length = float(s_dense[-1])

        # Monotone maps between the fit parameter u and arc length s.
        self._geom = geom
        self._geom_d = geom_d
        self._u_of_s = PchipInterpolator(s_dense, u_dense)
        s_of_u = PchipInterpolator(u_dense, s_dense)
        self._path_length = length

        return np.asarray(s_of_u(u), dtype=float)  # arc length at each waypoint

    # ------------------------------------------------------------------ stage 2
    def _build_timing_map(self, tk: NDArray[np.floating], s_wp: NDArray[np.floating]) -> None:
        """Monotone arc-length schedule s(t) from waypoint (time, arc length)."""
        if len(tk) < 2 or self._path_length <= 1e-9:
            # Constant or single-waypoint path: no motion.
            self._s_of_t = None
            self._s_dot_of_t = None
            return
        # PCHIP preserves monotonicity of the (increasing) arc-length samples, so
        # the drone never backtracks along the path.
        self._s_of_t = PchipInterpolator(tk, s_wp)
        self._s_dot_of_t = self._s_of_t.derivative()

    # ------------------------------------------------------------------ stage 3
    def _build_timed_trajectory(self) -> None:
        """Sample r(t) = p(s(t)) on the control grid: pos, vel, acc, yaw."""
        n = max(2, int(self._freq * self._t_total))
        self._time_grid = np.linspace(0.0, self._t_total, n)

        if self._geom is None or self._s_of_t is None:
            # Degenerate path: hold position, zero velocity, zero yaw.
            point = getattr(self, "_point", np.zeros(3))
            self._pos = np.tile(point, (n, 1))
            self._vel = np.zeros((n, 3))
            self._acc = np.zeros((n, 3))
            self._s_grid = np.zeros(n)
            self._yaw = np.zeros(n)
            self._yaw_rate = np.zeros(n)
            return

        s_t = np.clip(self._s_of_t(self._time_grid), 0.0, self._path_length)
        u_t = self._u_of_s(s_t)

        self._pos = self._geom(u_t)
        self._s_grid = s_t

        # Unit tangent from the geometry; scheduled speed from the timing map.
        dp_du = self._geom_d(u_t)
        speed_u = np.linalg.norm(dp_du, axis=1)
        tangent = dp_du / np.maximum(speed_u[:, None], 1e-9)
        s_dot = self._s_dot_of_t(self._time_grid)
        self._vel = tangent * s_dot[:, None]
        self._acc = np.gradient(self._vel, self._time_grid, axis=0)

        self._yaw = self._compute_yaw(tangent, s_dot)
        self._yaw_rate = np.gradient(np.unwrap(self._yaw), self._time_grid)

    def _compute_yaw(
        self, tangent: NDArray[np.floating], s_dot: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Yaw from the horizontal path heading, held through slow/vertical bits.

        On near-vertical segments the horizontal heading is ill-defined and can
        flip; a hysteresis band (resume only above ``2 * min_yaw_speed``) keeps the
        yaw held smoothly through such regions instead of snapping 180 degrees.
        """
        heading = np.arctan2(tangent[:, 1], tangent[:, 0])
        horiz_speed = np.hypot(tangent[:, 0], tangent[:, 1]) * np.abs(s_dot)

        enter_hold = self._min_yaw_speed
        resume = 2.0 * self._min_yaw_speed
        strong = horiz_speed > resume
        if not strong.any():
            return np.zeros_like(heading)

        # Hysteresis hold: keep the last confident heading until the horizontal
        # speed clearly recovers (> resume), then track again.
        yaw = heading.copy()
        last_yaw = heading[int(np.argmax(strong))]
        holding = True
        for i in range(len(yaw)):
            if holding:
                if horiz_speed[i] > resume:
                    holding = False
                    last_yaw = heading[i]
            elif horiz_speed[i] < enter_hold:
                holding = True
            else:
                last_yaw = heading[i]
            yaw[i] = last_yaw

        yaw = np.unwrap(yaw)
        if self._yaw_smoothing > 0.0:
            a = self._yaw_smoothing
            for i in range(1, len(yaw)):
                yaw[i] = a * yaw[i - 1] + (1.0 - a) * yaw[i]
        return yaw

    # --------------------------------------------------------------- public API
    def path(self, s: NDArray[np.floating] | float) -> NDArray[np.floating]:
        """Evaluate the geometric path p(s) at arc length(s) ``s``."""
        if self._geom is None:
            point = getattr(self, "_point", np.zeros(3))
            return np.broadcast_to(point, np.shape(s) + (3,)) if np.ndim(s) else point
        s = np.clip(np.asarray(s, dtype=float), 0.0, self._path_length)
        return self._geom(self._u_of_s(s))

    def s_of_t(self, t: NDArray[np.floating] | float) -> NDArray[np.floating] | float:
        """Evaluate the timing map s(t) (arc length reached at time ``t``)."""
        if self._s_of_t is None:
            return np.zeros_like(t) if np.ndim(t) else 0.0
        return np.clip(self._s_of_t(t), 0.0, self._path_length)

    def sample_horizon(self, tick: int, N: int) -> dict:
        """Return the next ``N`` setpoints starting at ``tick`` (+ the terminal)."""
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
