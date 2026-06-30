import numpy as np


class UniformTiming:
    def compute(self, waypoints, t_total):
        return np.linspace(0, t_total, len(waypoints))



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