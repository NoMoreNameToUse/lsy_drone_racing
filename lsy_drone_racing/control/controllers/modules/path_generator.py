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
from lsy_drone_racing.control.controllers.modules.occupancy_grid_3d import OccupancyGrid3D
from lsy_drone_racing.control.controllers.modules.astar_3d import astar_3d


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
        gate_entry_distance: float = 0.2,
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

class AStarGatePathGenerator:
    """
    GatePassingPathGenerator gives mandatory gate points.
    A* connects only the free-space segments:
        current -> next gate entry

    Gate traversal itself remains direct:
        entry -> center -> exit
    """

    def __init__(
        self,
        gate_passing_generator=None,
        grid_resolution: float = 0.075,
        safety_margin: float = 0.04,
        obstacle_radius: float = 0.20,
        prune_path: bool = True,
        final_extension_distance: float = 0.60,
    ):
        self.gate_passing_generator = (
            GatePassingPathGenerator(max_nudge=0.0)
            if gate_passing_generator is None
            else gate_passing_generator
        )
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin
        self.obstacle_radius = obstacle_radius
        self.prune_path = prune_path
        self.final_extension_distance = final_extension_distance

    def generate(self, obs, config=None):
        mandatory = self.gate_passing_generator.generate(obs, config)

        grid = OccupancyGrid3D(
            obs=obs,
            config=config,
            resolution=self.grid_resolution,
            safety_margin=self.safety_margin,
            obstacle_radius=self.obstacle_radius,
        )

        final_path = [mandatory[0]]

        num_gates = (len(mandatory) - 1) // 3
        current = mandatory[0]

        for gate_i in range(num_gates):
            base = 1 + 3 * gate_i

            entry = mandatory[base + 0]
            center = mandatory[base + 1]
            exit_ = mandatory[base + 2]

            astar_segment = self.plan_astar(grid, current, entry)

            if len(astar_segment) > 1:
                final_path.extend(astar_segment[1:])

            # Keep gate crossing direct and ordered.
            final_path.append(center)
            final_path.append(exit_)

            current = exit_

        # Final extension after last gate.
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)

        if len(gates_pos) > 0 and self.final_extension_distance > 0.0:
            last_gate_pos = gates_pos[-1]
            last_gate_quat = gates_quat[-1]

            rot = R.from_quat(last_gate_quat)
            forward = rot.apply(np.array([1.0, 0.0, 0.0]))
            forward = forward / max(np.linalg.norm(forward), 1e-9)

            finish = last_gate_pos + self.final_extension_distance * forward
            finish = self.gate_passing_generator._clip_z(finish)

            astar_segment = self.plan_astar(grid, current, finish)

            if len(astar_segment) > 1:
                final_path.extend(astar_segment[1:])

            current = finish

        path = np.asarray(final_path, dtype=float)

        if self.prune_path:
            path = self._prune_path(path, grid)

        return path

    def plan_astar(self, grid, start, goal):
        start_idx = grid.world_to_grid(start)
        goal_idx = grid.world_to_grid(goal)

        idx_path = astar_3d(grid, start_idx, goal_idx)

        if idx_path is None:
            print(
                "WARNING: A* failed; falling back to straight segment:",
                "start=", start,
                "goal=", goal,
            )
            return np.vstack([start, goal])

        return np.asarray([grid.grid_to_world(idx) for idx in idx_path], dtype=float)

    def _prune_path(self, path, grid):
        if len(path) <= 2:
            return path

        pruned = [path[0]]
        i = 0

        while i < len(path) - 1:
            j = len(path) - 1

            while j > i + 1:
                if self._line_is_free(path[i], path[j], grid):
                    break
                j -= 1

            pruned.append(path[j])
            i = j

        return np.asarray(pruned, dtype=float)

    def _line_is_free(self, a, b, grid, step=None):
        if step is None:
            step = grid.resolution * 0.5

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)

        dist = np.linalg.norm(b - a)
        if dist < 1e-9:
            return grid.is_free(grid.world_to_grid(a))

        n = max(2, int(np.ceil(dist / step)))

        for alpha in np.linspace(0.0, 1.0, n):
            p = (1.0 - alpha) * a + alpha * b
            if not grid.is_free(grid.world_to_grid(p)):
                return False

        return True