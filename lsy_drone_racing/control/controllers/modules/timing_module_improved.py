"""Dynamics-aware timing: a feasible, clearance/curvature-shaped speed profile.

``DynamicTiming`` replaces the constant-speed ``DistanceTiming`` (which ignored
dynamics, so sharp turns demanded impossible decelerations and straights were left
crawling). It assigns per-waypoint timestamps ``t`` from a speed profile that:

* **slows for geometry** -- per-waypoint curvature (turn angle / segment length)
  caps speed at ``sqrt(a_lat_max / kappa)``, so tight corners and reversals are
  taken slowly;
* **slows for tightness** -- the clearance "tube" from the post-processor scales
  speed down in narrow passages / squeezes between obstacles;
* **handles reversals** -- a near-180 deg fold is both caught by the curvature term
  and hard-capped at ``reversal_speed``; the backward acceleration pass then makes
  the drone decelerate *before* reaching it ("slow down when reversing is coming
  up"), while still allowing full speed through an open straight gate;
* **stays feasible** -- a forward/backward pass limits tangential
  acceleration/deceleration to ``a_max``, so the commanded speed never changes
  faster than the tracker can follow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class DynamicTiming:
    """Per-waypoint timing from a curvature/clearance-shaped, acceleration-limited profile."""

    def __init__(
        self,
        v_max: float = 1.3,
        a_max: float = 2.5,
        a_lat_max: float = 4.0,
        curvature_window: float = 0.3,
        clearance_ref: float = 0.40,
        clearance_floor_speed: float = 0.6,
        v_start: float = 0.3,
        v_end: float = 0.6,
        reversal_angle_deg: float = 140.0,
        reversal_speed: float = 0.4,
        abs_min_speed: float = 0.2,
        min_segment_time: float = 0.05,
    ):
        """Configure the speed profile.

        Args:
            v_max: Cruise speed ceiling (keep at/below the tracker's stable limit).
            a_max: Max tangential accel/decel used by the feasibility pass (m/s^2).
            a_lat_max: Max lateral accel for the curvature speed cap (m/s^2).
            curvature_window: Arc-length half-window (m) over which curvature is
                measured, so the speed profile is independent of waypoint density
                (a densified 10cm path gives the same speeds as a sparse one).
            clearance_ref: Clearance (m) at/above which speed is unrestricted.
            clearance_floor_speed: Speed at zero clearance (narrowest squeeze).
            v_start: Speed cap at the first waypoint (drone starts ~slow).
            v_end: Speed cap at the last waypoint.
            reversal_angle_deg: Turn angle at/above which a waypoint is a reversal.
            reversal_speed: Hard speed cap at a reversal waypoint.
            abs_min_speed: Absolute floor so segments never take ~infinite time.
            min_segment_time: Lower bound on any segment's duration (s).
        """
        self.v_max = v_max
        self.a_max = a_max
        self.a_lat_max = a_lat_max
        self.curvature_window = curvature_window
        self.clearance_ref = clearance_ref
        self.clearance_floor_speed = clearance_floor_speed
        self.v_start = v_start
        self.v_end = v_end
        self.reversal_angle_deg = reversal_angle_deg
        self.reversal_speed = reversal_speed
        self.abs_min_speed = abs_min_speed
        self.min_segment_time = min_segment_time

    def compute(
        self,
        waypoints: NDArray[np.floating],
        clearance: NDArray[np.floating] | None = None,
        t_total: float | None = None,
        v_init: float | None = None,
    ) -> NDArray[np.floating]:
        """Return per-waypoint timestamps for ``waypoints``.

        Args:
            waypoints: ``(M, 3)`` path waypoints.
            clearance: Optional ``(M,)`` per-waypoint clearance tube (from the
                post-processor); if ``None`` the clearance term is skipped.
            t_total: Optional total time to rescale to (kept for compatibility).
            v_init: Drone's actual current speed (m/s) to seed the profile with on
                a replan. If ``None`` the profile starts from ``v_start`` (assumes a
                near-stop); pass ``norm(obs["vel"])`` so the post-replan schedule
                matches reality instead of asking the drone to slow to a crawl.

        Returns:
            ``(M,)`` strictly increasing timestamps, ``t[0] == 0``.
        """
        wp = np.asarray(waypoints, dtype=float)
        m = len(wp)
        if m <= 1:
            return np.zeros(m, dtype=float)

        ds = np.maximum(np.linalg.norm(np.diff(wp, axis=0), axis=1), 1e-6)  # (M-1,)
        s = np.concatenate([[0.0], np.cumsum(ds)])  # (M,) arc length

        v_lim = self._speed_limits(wp, s, clearance, m)
        v = self._accel_limited_pass(v_lim, ds, v_init)
        t = self._integrate(v, ds)

        if t_total is not None and t[-1] > 1e-9:
            t *= float(t_total) / t[-1]
        return t

    # ------------------------------------------------------------------ helpers
    def _speed_limits(
        self,
        wp: NDArray[np.floating],
        s: NDArray[np.floating],
        clearance: NDArray[np.floating] | None,
        m: int,
    ) -> NDArray[np.floating]:
        """Per-waypoint speed cap from curvature, reversals, clearance and endpoints."""
        v_lim = np.full(m, self.v_max)

        # Curvature over a fixed arc-length window (density-independent): for each
        # waypoint compare the heading ~curvature_window before vs. after it. Dense
        # paths average over the window (no per-10cm noise); sparse paths fall back
        # to the adjacent points.
        if m >= 3:
            idx = np.arange(m)
            back = np.minimum(np.searchsorted(s, s - self.curvature_window), idx - 1)
            fwd = np.maximum(np.searchsorted(s, s + self.curvature_window), idx + 1)
            jb = np.clip(back, 0, m - 1)
            jf = np.clip(fwd, 0, m - 1)
            a = wp - wp[jb]
            b = wp[jf] - wp
            na = np.linalg.norm(a, axis=1)
            nb = np.linalg.norm(b, axis=1)
            valid = (na > 1e-6) & (nb > 1e-6)
            cos_turn = np.clip(np.sum(a * b, axis=1) / np.maximum(na * nb, 1e-9), -1.0, 1.0)
            theta = np.arccos(cos_turn)
            kappa = np.where(valid, theta / np.maximum(0.5 * (na + nb), 1e-6), 0.0)
            v_curve = np.sqrt(self.a_lat_max / np.maximum(kappa, 1e-6))
            v_lim = np.minimum(v_lim, v_curve)

            # Explicit reversal cap (also caught by curvature, but guaranteed here).
            reversal = valid & (np.degrees(theta) >= self.reversal_angle_deg)
            v_lim = np.where(reversal, np.minimum(v_lim, self.reversal_speed), v_lim)

        # Tightness: scale speed with the clearance tube.
        if clearance is not None:
            clr = np.asarray(clearance, dtype=float)
            if clr.shape[0] == m:
                frac = np.clip(clr / max(self.clearance_ref, 1e-6), 0.0, 1.0)
                span = self.v_max - self.clearance_floor_speed
                v_clear = self.clearance_floor_speed + span * frac
                v_lim = np.minimum(v_lim, v_clear)

        v_lim = np.clip(v_lim, self.abs_min_speed, self.v_max)
        v_lim[0] = min(v_lim[0], self.v_start)
        v_lim[-1] = min(v_lim[-1], self.v_end)
        return v_lim

    def _accel_limited_pass(
        self,
        v_lim: NDArray[np.floating],
        ds: NDArray[np.floating],
        v_init: float | None = None,
    ) -> NDArray[np.floating]:
        """Forward/backward sweep capping tangential accel/decel at ``a_max``.

        With ``v_init`` the start speed is pinned to the drone's actual speed (the
        boundary condition), so the forward sweep accel-limits from reality and the
        first segment's time reflects the true entry speed; the backward sweep still
        shapes the deceleration into upcoming slow points.
        """
        v = v_lim.copy()
        if v_init is not None:
            v[0] = float(np.clip(v_init, self.abs_min_speed, self.v_max))
        # Forward: cannot accelerate faster than a_max.
        for i in range(1, len(v)):
            v[i] = min(v[i], np.sqrt(v[i - 1] ** 2 + 2.0 * self.a_max * ds[i - 1]))
        # Backward: cannot decelerate faster than a_max.
        for i in range(len(v) - 2, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1] ** 2 + 2.0 * self.a_max * ds[i]))
        # Restore the initial condition: the drone is at v_init now, so the schedule
        # must start there even if the backward pass would dip it for a near turn.
        if v_init is not None:
            v[0] = float(np.clip(v_init, self.abs_min_speed, self.v_max))
        return np.maximum(v, self.abs_min_speed)

    def _integrate(
        self, v: NDArray[np.floating], ds: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Cumulative time from per-segment average speed."""
        v_avg = 0.5 * (v[:-1] + v[1:])
        dt = np.maximum(ds / np.maximum(v_avg, 1e-6), self.min_segment_time)
        t = np.zeros(len(v))
        t[1:] = np.cumsum(dt)
        return t

class DistanceTiming:
    """
    Assigns time based on path segment length.

    This avoids giving the same time to tiny gate-entry segments and long
    between-gate travel segments.
    """

    def __init__(self, nominal_speed=0.6, min_segment_time=0.15):
        self.nominal_speed = nominal_speed
        self.min_segment_time = min_segment_time

    def compute(self, waypoints, t_total=None):
        waypoints = np.asarray(waypoints, dtype=float)

        if len(waypoints) <= 1:
            return np.zeros(len(waypoints), dtype=float)

        segment_lengths = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
        segment_times = segment_lengths / self.nominal_speed

        segment_times = np.maximum(segment_times, self.min_segment_time)

        t = np.zeros(len(waypoints))
        t[1:] = np.cumsum(segment_times)

        if t_total is not None and t[-1] > 1e-9:
            t *= float(t_total) / t[-1]

        return t
    
class UniformTiming:
    # Dead simple, I still remember how to use linespace! :D 
    # Update: too simple :C did not work well
    def compute(self, waypoints, t_total):
        return np.linspace(0, t_total, len(waypoints))