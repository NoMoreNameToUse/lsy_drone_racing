import numpy as np

# Baseline fixed waypoint
class WaypointPathGenerator:
    def generate(self, obs, config):
        return np.array(
            [
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.7, 1.2],
                [0.5, -0.75, 1.2],
            ]
        )

from scipy.spatial.transform import Rotation as R


class GatePassingPathGenerator:
    """
    Baseline gate-passing waypoint generator with simple local obstacle nudging.

    Generates:
        start -> gate entry -> gate center -> gate exit -> ...

    The gate center is fixed.
    Entry/exit points may be nudged away from nearby pole obstacles.
    """

    def __init__(
        self,
        gate_entry_distance: float = 0.18,
        obstacle_detection_radius: float = 0.35,
        obstacle_clearance_radius: float = 0.28,
        max_nudge: float = 0.20,
        min_z: float = 0.08,
        max_z: float = 1.60,
        gate_forward_axis: np.ndarray | None = None,
    ):
        self.gate_entry_distance = gate_entry_distance
        self.obstacle_detection_radius = obstacle_detection_radius
        self.obstacle_clearance_radius = obstacle_clearance_radius
        self.max_nudge = max_nudge
        self.min_z = min_z
        self.max_z = max_z

        self.gate_forward_axis = (
            np.array([1.0, 0.0, 0.0])
            if gate_forward_axis is None
            else np.asarray(gate_forward_axis, dtype=float)
        )

    def generate(self, obs, config=None):
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float)
        start_pos = np.asarray(obs["pos"], dtype=float)

        waypoints = [self._clip_z(start_pos.copy())]

        for gate_pos, gate_quat in zip(gates_pos, gates_quat):
            entry, center, exit_ = self._gate_passing_points(
                gate_pos=gate_pos,
                gate_quat=gate_quat,
                obstacles_pos=obstacles_pos,
            )

            waypoints.extend([entry, center, exit_])

        return np.asarray(waypoints, dtype=float)

    def _gate_passing_points(self, gate_pos, gate_quat, obstacles_pos):
        gate_pos = np.asarray(gate_pos, dtype=float)
        forward = self._gate_forward(gate_quat)

        entry = gate_pos - self.gate_entry_distance * forward
        center = gate_pos.copy()
        exit_ = gate_pos + self.gate_entry_distance * forward

        # Keep the actual gate crossing point fixed.
        center = self._clip_z(center)

        # Nudge only entry/exit helper points.
        entry = self._nudge_away_from_obstacles(entry, obstacles_pos)
        exit_ = self._nudge_away_from_obstacles(exit_, obstacles_pos)

        return self._clip_z(entry), center, self._clip_z(exit_)

    def _nudge_away_from_obstacles(self, point, obstacles_pos):
        """
        Move point away from nearby obstacle poles in XY.

        This is intentionally local and conservative:
        - only uses XY distance
        - preserves z
        - caps maximum movement
        """
        p = np.asarray(point, dtype=float).copy()
        original_xy = p[:2].copy()

        total_push = np.zeros(2)

        for obs in obstacles_pos:
            obs_xy = obs[:2]
            delta = p[:2] - obs_xy
            dist = np.linalg.norm(delta)

            if dist < 1e-9:
                # Degenerate case: choose arbitrary push direction.
                direction = np.array([1.0, 0.0])
                dist = 1e-9
            else:
                direction = delta / dist

            if dist < self.obstacle_detection_radius:
                # Push stronger when closer.
                required_push = self.obstacle_clearance_radius - dist

                if required_push > 0.0:
                    total_push += required_push * direction

        push_norm = np.linalg.norm(total_push)

        if push_norm > self.max_nudge:
            total_push = total_push / push_norm * self.max_nudge

        p[:2] = original_xy + total_push
        return self._clip_z(p)

    def _gate_forward(self, gate_quat):
        rot = R.from_quat(gate_quat)
        forward = rot.apply(self.gate_forward_axis)
        return self._normalize(forward)

    def _clip_z(self, p):
        p = np.asarray(p, dtype=float).copy()
        p[2] = np.clip(p[2], self.min_z, self.max_z)
        return p

    @staticmethod
    def _normalize(v):
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return v
        return v / norm