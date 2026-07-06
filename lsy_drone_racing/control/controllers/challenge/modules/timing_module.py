"""Simple waypoint timing schemes for the initial-challenge pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


class UniformTiming:
    """Assigns waypoints uniformly spaced timestamps over a fixed total time."""

    def compute(self, waypoints: NDArray[np.floating], t_total: float) -> NDArray[np.floating]:
        """Return ``len(waypoints)`` timestamps evenly spaced in ``[0, t_total]``."""
        return np.linspace(0, t_total, len(waypoints))


class DistanceTiming:
    """Assigns time based on path segment length.

    This avoids giving the same time to tiny gate-entry segments and long
    between-gate travel segments.
    """

    def __init__(self, nominal_speed: float = 0.6, min_segment_time: float = 0.15):
        """Configure the cruise speed and the per-segment minimum duration."""
        self.nominal_speed = nominal_speed
        self.min_segment_time = min_segment_time

    def compute(
        self, waypoints: NDArray[np.floating], t_total: float | None = None
    ) -> NDArray[np.floating]:
        """Return per-waypoint timestamps proportional to segment length."""
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
