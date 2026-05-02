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


class MotionAwareTiming:
    """
    Assign time based on distance, direction change, and vertical motion.

    This keeps the simplicity of distance-based timing but allocates extra time
    around sharp corners and stronger climb/descent segments, which makes the
    resulting reference easier for the drone to track.
    """

    def __init__(
        self,
        nominal_speed: float = 0.6,
        min_segment_time: float = 0.15,
        turn_time_gain: float = 0.5,
        turn_exponent: float = 1.5,
        vertical_time_gain: float = 0.5,
        vertical_ratio_power: float = 2.0,
    ):
        self.nominal_speed = nominal_speed
        self.min_segment_time = min_segment_time
        self.turn_time_gain = turn_time_gain
        self.turn_exponent = turn_exponent
        self.vertical_time_gain = vertical_time_gain
        self.vertical_ratio_power = vertical_ratio_power

    def compute(self, waypoints, t_total=None):
        waypoints = np.asarray(waypoints, dtype=float)

        if len(waypoints) <= 1:
            return np.zeros(len(waypoints), dtype=float)

        segment_vectors = np.diff(waypoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)

        safe_lengths = np.maximum(segment_lengths, 1e-9)
        base_times = segment_lengths / self.nominal_speed

        vertical_ratio = np.abs(segment_vectors[:, 2]) / safe_lengths
        vertical_multiplier = 1.0 + self.vertical_time_gain * np.power(
            vertical_ratio,
            self.vertical_ratio_power,
        )

        turn_penalties = np.zeros(len(segment_lengths), dtype=float)

        for i in range(1, len(waypoints) - 1):
            incoming = segment_vectors[i - 1]
            outgoing = segment_vectors[i]

            incoming_norm = np.linalg.norm(incoming)
            outgoing_norm = np.linalg.norm(outgoing)
            if incoming_norm < 1e-9 or outgoing_norm < 1e-9:
                continue

            cosine = np.dot(incoming, outgoing) / (incoming_norm * outgoing_norm)
            cosine = np.clip(cosine, -1.0, 1.0)
            turn_fraction = (1.0 - cosine) * 0.5
            turn_fraction = np.power(turn_fraction, self.turn_exponent)

            shared_penalty = self.turn_time_gain * turn_fraction
            turn_penalties[i - 1] += 0.5 * shared_penalty
            turn_penalties[i] += 0.5 * shared_penalty

        segment_times = base_times * vertical_multiplier
        segment_times *= 1.0 + turn_penalties
        segment_times = np.maximum(segment_times, self.min_segment_time)

        t = np.zeros(len(waypoints), dtype=float)
        t[1:] = np.cumsum(segment_times)

        if t_total is not None and t[-1] > 1e-9:
            t *= float(t_total) / t[-1]

        return t