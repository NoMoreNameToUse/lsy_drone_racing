"""Gate-aware path generators for the initial-challenge pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controllers.challenge.modules.astar_3d import astar_3d
from lsy_drone_racing.control.controllers.challenge.modules.occupancy_grid_3d import OccupancyGrid3D

if TYPE_CHECKING:
    from numpy.typing import NDArray


class WaypointPathGenerator:
    """Baseline generator returning a fixed, hand-placed waypoint list."""

    def generate(self, obs: dict, config: object) -> NDArray[np.floating]:
        """Return the fixed nominal-track waypoints (ignores the observation)."""
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


class GatePassingPathGenerator:
    """Baseline gate-passing waypoint generator with simple local obstacle nudging.

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
        max_nudge: float = 0.25,
        min_z: float = 0.08,
        max_z: float = 1.60,
        gate_forward_axis: np.ndarray | None = None,
    ):
        """Configure gate-anchor offsets, obstacle-nudge radii, and the z-clip range."""
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

    def generate(self, obs: dict, config: object = None) -> NDArray[np.floating]:
        """Return the waypoint path through the gates remaining ahead of the drone."""
        target_gate = int(np.asarray(obs.get("target_gate", 0)).item())

        gates_pos_all = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat_all = np.asarray(obs["gates_quat"], dtype=float)

        # Only remaining gates are mandatory targets.
        gates_pos = gates_pos_all[target_gate:]
        gates_quat = gates_quat_all[target_gate:]

        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float)
        start_pos = np.asarray(obs["pos"], dtype=float)

        waypoints = [self._clip_z(start_pos.copy())]

        for gate_pos, gate_quat in zip(gates_pos, gates_quat):
            entry, center, exit_ = self._gate_passing_points(
                gate_pos=gate_pos, gate_quat=gate_quat, obstacles_pos=obstacles_pos
            )

            waypoints.extend([entry, center, exit_])

        return np.asarray(waypoints, dtype=float)

    def _gate_passing_points(
        self,
        gate_pos: NDArray[np.floating],
        gate_quat: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
        """Return (entry, center, exit) anchors for one gate; entry/exit obstacle-nudged."""
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

    def _nudge_away_from_obstacles(
        self, point: NDArray[np.floating], obstacles_pos: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Move point away from nearby obstacle poles in XY.

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

    def _gate_forward(self, gate_quat: NDArray[np.floating]) -> NDArray[np.floating]:
        """Unit gate normal in world frame from the gate quaternion."""
        rot = R.from_quat(gate_quat)
        forward = rot.apply(self.gate_forward_axis)
        return self._normalize(forward)

    def _clip_z(self, p: NDArray[np.floating]) -> NDArray[np.floating]:
        """Clamp a point's height into ``[min_z, max_z]``."""
        p = np.asarray(p, dtype=float).copy()
        p[2] = np.clip(p[2], self.min_z, self.max_z)
        return p

    @staticmethod
    def _normalize(v: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return the unit vector of ``v`` (unchanged if near zero)."""
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return v
        return v / norm


class AStarGatePathGenerator:
    """Gate-aware planner: A* connects the free-space segments between gate anchors.

    ``GatePassingPathGenerator`` supplies the mandatory gate points; A* plans only
    ``current -> next gate entry`` while the gate traversal (entry -> center -> exit)
    stays direct. Optional pruning is applied only within each A* free-space segment,
    never across the mandatory gate anchors.
    """

    def __init__(
        self,
        gate_passing_generator: GatePassingPathGenerator | None = None,
        grid_resolution: float = 0.075,
        safety_margin: float = 0.04,
        obstacle_radius: float = 0.20,
        heuristic_weight: float = 1.15,
        max_astar_iterations: int = 200_000,
        endpoint_snap_distance: float = 0.30,
        prune_path: bool = False,
        final_extension_distance: float = 0.80,
        velocity_bias_weight: float = 0.20,
        velocity_bias_decay: float = 8.0,
        min_velocity_for_bias: float = 0.10,
    ):
        """Configure the A* grid inflation, endpoint snapping, and velocity bias."""
        self.gate_passing_generator = (
            GatePassingPathGenerator(max_nudge=0.0)
            if gate_passing_generator is None
            else gate_passing_generator
        )
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin
        self.obstacle_radius = obstacle_radius
        self.heuristic_weight = heuristic_weight
        self.max_astar_iterations = max_astar_iterations
        self.endpoint_snap_distance = endpoint_snap_distance
        self.prune_path = prune_path
        self.final_extension_distance = final_extension_distance
        self.velocity_bias_weight = velocity_bias_weight
        self.velocity_bias_decay = velocity_bias_decay
        self.min_velocity_for_bias = min_velocity_for_bias

    def generate(self, obs: dict, config: object = None) -> NDArray[np.floating]:
        """Return the waypoint path through the gates remaining ahead of the drone."""
        mandatory = self.gate_passing_generator.generate(obs, config)
        current_velocity = np.asarray(obs.get("vel", np.zeros(3)), dtype=float)

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

            segment_velocity = current_velocity if gate_i == 0 else None
            astar_segment = self.plan_astar(
                grid, current, entry, preferred_velocity=segment_velocity
            )
            astar_segment = self._maybe_prune_segment(astar_segment, grid)

            if len(astar_segment) > 1:
                final_path.extend(astar_segment[1:])

            # Keep gate crossing direct and ordered.
            final_path.append(center)
            final_path.append(exit_)

            current = exit_

        # Final extension after the last remaining gate.
        target_gate = int(np.asarray(obs.get("target_gate", 0)).item())

        gates_pos_all = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat_all = np.asarray(obs["gates_quat"], dtype=float)

        remaining_gates_pos = gates_pos_all[target_gate:]
        remaining_gates_quat = gates_quat_all[target_gate:]

        if len(remaining_gates_pos) > 0 and self.final_extension_distance > 0.0:
            last_gate_pos = remaining_gates_pos[-1]
            last_gate_quat = remaining_gates_quat[-1]

            rot = R.from_quat(last_gate_quat)
            forward = rot.apply(np.array([1.0, 0.0, 0.0]))
            forward = forward / max(np.linalg.norm(forward), 1e-9)

            finish = last_gate_pos + self.final_extension_distance * forward
            finish = self.gate_passing_generator._clip_z(finish)

            final_segment_velocity = current_velocity if num_gates == 0 else None
            astar_segment = self.plan_astar(
                grid, current, finish, preferred_velocity=final_segment_velocity
            )
            astar_segment = self._maybe_prune_segment(astar_segment, grid)

            if len(astar_segment) > 1:
                final_path.extend(astar_segment[1:])

            current = finish

        return np.asarray(final_path, dtype=float)

    def plan_astar(
        self,
        grid: OccupancyGrid3D,
        start: NDArray[np.floating],
        goal: NDArray[np.floating],
        preferred_velocity: NDArray[np.floating] | None = None,
    ) -> NDArray[np.floating]:
        """Plan the free-space segment ``start -> goal`` (straight-line on failure)."""
        start_idx = grid.world_to_grid(start)
        goal_idx = grid.world_to_grid(goal)

        snapped_start_idx = grid.nearest_free_idx(
            start_idx, max_distance=self.endpoint_snap_distance
        )
        snapped_goal_idx = grid.nearest_free_idx(goal_idx, max_distance=self.endpoint_snap_distance)

        if snapped_start_idx is None:
            print(f"A*: no free start cell near {start_idx}, world={start}")
            return np.vstack([start, goal])

        if snapped_goal_idx is None:
            print(f"A*: no free goal cell near {goal_idx}, world={goal}")
            return np.vstack([start, goal])

        if snapped_start_idx != start_idx:
            print(
                "A*: snapped start",
                start_idx,
                "->",
                snapped_start_idx,
                "world=",
                grid.grid_to_world(snapped_start_idx),
            )

        if snapped_goal_idx != goal_idx:
            print(
                "A*: snapped goal",
                goal_idx,
                "->",
                snapped_goal_idx,
                "world=",
                grid.grid_to_world(snapped_goal_idx),
            )

        idx_path = astar_3d(
            grid,
            snapped_start_idx,
            snapped_goal_idx,
            max_iterations=self.max_astar_iterations,
            heuristic_weight=self.heuristic_weight,
            preferred_direction=preferred_velocity,
            direction_bias_weight=self.velocity_bias_weight,
            direction_bias_decay=self.velocity_bias_decay,
            min_direction_speed=self.min_velocity_for_bias,
        )

        if idx_path is None:
            print(
                "WARNING: A* failed; falling back to straight segment:",
                "start=",
                start,
                "goal=",
                goal,
            )
            return np.vstack([start, goal])

        return np.asarray([grid.grid_to_world(idx) for idx in idx_path], dtype=float)

    def _maybe_prune_segment(
        self, path: NDArray[np.floating], grid: OccupancyGrid3D
    ) -> NDArray[np.floating]:
        """Prune the segment if pruning is enabled, else return it unchanged."""
        if not self.prune_path:
            return path

        return self._prune_path(path, grid)

    def _prune_path(
        self, path: NDArray[np.floating], grid: OccupancyGrid3D
    ) -> NDArray[np.floating]:
        """Drop intermediate waypoints whose skip stays collision-free (line-of-sight)."""
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

    def _line_is_free(
        self,
        a: NDArray[np.floating],
        b: NDArray[np.floating],
        grid: OccupancyGrid3D,
        step: float | None = None,
    ) -> bool:
        """True if the straight segment ``a -> b`` passes only through free cells."""
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
