"""Gate-anchored path generator (the pipeline's main planner).

Connects mandatory gate anchors through free space and adds the reversal /
side-commitment handling the tracker needs:
- mandatory gate anchors from ``GatePassingPathGenerator``,
- shortest-path segment connection with the ``AStar3DBarebone`` solver,
- reversal/flyback opening blocks and obstacle side-commitment (per segment),
- optional conservative path pruning inside each free-space segment.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controllers.modules.astar_3d_barebone import AStar3DBarebone
from lsy_drone_racing.control.controllers.modules.occupancy_grid_3d_improved import OccupancyGrid3D


class GatePassingPathGenerator:
    """Gate entry / center / exit anchors for the remaining gates.

    For each gate still ahead of the drone, emits three anchors -- a point
    ``gate_entry_distance`` before the gate along its (world) normal, the gate
    centre, and a point the same distance after -- prefixed by the drone's start:
    ``[start, entry_0, center_0, exit_0, entry_1, ...]``
    """

    def __init__(
        self,
        gate_entry_distance: float = 0.2,
        min_z: float = 0.08,
        max_z: float = 1.60,
        gate_forward_axis=None,
    ):
        self.gate_entry_distance = gate_entry_distance
        self.min_z = min_z
        self.max_z = max_z
        self.gate_forward_axis = (
            np.array([1.0, 0.0, 0.0])
            if gate_forward_axis is None
            else np.asarray(gate_forward_axis, dtype=float)
        )

    def generate(self, obs, config=None):
        target_gate = int(np.asarray(obs.get("target_gate", 0)).item())
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)[target_gate:]
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)[target_gate:]
        start_pos = np.asarray(obs["pos"], dtype=float)

        waypoints = [self._clip_z(start_pos.copy())]
        for gate_pos, gate_quat in zip(gates_pos, gates_quat):
            forward = self._gate_forward(gate_quat)
            waypoints.append(self._clip_z(gate_pos - self.gate_entry_distance * forward))
            waypoints.append(self._clip_z(gate_pos.copy()))
            waypoints.append(self._clip_z(gate_pos + self.gate_entry_distance * forward))
        return np.asarray(waypoints, dtype=float)

    def _gate_forward(self, gate_quat):
        forward = R.from_quat(gate_quat).apply(self.gate_forward_axis)
        return self._normalize(forward)

    def _clip_z(self, p):
        p = np.asarray(p, dtype=float).copy()
        p[2] = np.clip(p[2], self.min_z, self.max_z)
        return p

    @staticmethod
    def _normalize(v):
        norm = np.linalg.norm(v)
        return v if norm < 1e-9 else v / norm


class AStarImprovedPathGenerator:
    """Gate-anchored path generator: anchors + barebone A* + reversal/commit + pruning."""

    def __init__(
        self,
        gate_passing_generator=None,
        grid_resolution: float = 0.075,
        safety_margin: float = 0.04,
        obstacle_radius: float = 0.20,
        heuristic_weight: float = 1.0,
        endpoint_snap_distance: float = 0.30,
        prune_path: bool = False,
        final_extension_distance: float = 1,
        # --- Reversal / flyback handling ---
        block_passed_gate: bool = True,
        block_all_gates: bool = True,
        reversal_cone_deg: float = 30.0,
        forward_cone_deg: float = 30.0,
        forward_cone_length: float = 1.0,
        block_opening_half: float = 0.22,
        block_depth: float = 0.25,
        # --- Obstacle side-commitment (anti late-flip) ---
        # When the drone approaches an obstacle fast, lock the side it is already
        # passing on so a late sensor reveal can't flip the path to the other side
        # (a lateral jump the tracker can't follow). A wall is blocked on the
        # *wrong* side of imminent obstacles; if that makes the segment infeasible
        # the block is dropped (the flip was genuinely necessary).
        commit_obstacles: bool = True,
        commit_distance: float = 1.0,
        commit_min_speed: float = 0.8,
        commit_lateral_band: float = 0.5,
        commit_wall_extent: float = 0.5,
        commit_wall_thickness: float = 0.12,
        commit_z_half: float = 0.85,
    ):
        self.gate_passing_generator = (
            GatePassingPathGenerator()
            if gate_passing_generator is None
            else gate_passing_generator
        )
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin
        self.obstacle_radius = obstacle_radius
        self.heuristic_weight = heuristic_weight
        self.endpoint_snap_distance = endpoint_snap_distance
        self.prune_path = prune_path
        self.final_extension_distance = final_extension_distance

        self.block_passed_gate = block_passed_gate
        self.block_all_gates = block_all_gates
        self.reversal_cone_deg = reversal_cone_deg
        self.forward_cone_deg = forward_cone_deg
        self.forward_cone_length = forward_cone_length
        self.block_opening_half = block_opening_half
        self.block_depth = block_depth

        self.commit_obstacles = commit_obstacles
        self.commit_distance = commit_distance
        self.commit_min_speed = commit_min_speed
        self.commit_lateral_band = commit_lateral_band
        self.commit_wall_extent = commit_wall_extent
        self.commit_wall_thickness = commit_wall_thickness
        self.commit_z_half = commit_z_half

        self._solver = AStar3DBarebone()

    def generate(self, obs, config=None):
        mandatory = self.gate_passing_generator.generate(obs, config)

        grid = OccupancyGrid3D(
            obs=obs,
            config=config,
            resolution=self.grid_resolution,
            safety_margin=self.safety_margin,
            obstacle_radius=self.obstacle_radius,
        )

        target_gate = int(np.asarray(obs.get("target_gate", 0)).item())
        final_path = [mandatory[0]]
        num_gates = (len(mandatory) - 1) // 3
        current = mandatory[0]

        for gate_i in range(num_gates):
            base = 1 + 3 * gate_i
            entry = mandatory[base + 0]
            center = mandatory[base + 1]
            exit_ = mandatory[base + 2]

            # The gate left behind on this segment (target_gate - 1 on the first
            # segment = the physical gate just crossed; the previous remaining gate
            # thereafter). The reversal/flyback check is applied to it per-segment.
            # With block_all_gates off, only the first segment is checked (the old
            # behaviour) -- per-gate blocking changes later segments and, via the
            # shared spline, can perturb the RL tracker through an earlier gate.
            left_gate_idx = target_gate + gate_i - 1
            if not self.block_all_gates and gate_i > 0:
                left_gate_idx = -1  # disable: only the just-passed gate is checked
            astar_segment = self._plan_segment(
                grid, current, entry, obs, left_gate_idx, is_approach=(gate_i == 0)
            )

            if len(astar_segment) > 1:
                final_path.extend(astar_segment[1:])

            final_path.append(center)
            final_path.append(exit_)
            current = exit_

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

            finish_left_idx = target_gate + num_gates - 1
            if not self.block_all_gates and num_gates > 0:
                finish_left_idx = -1  # only the just-passed gate is checked
            astar_segment = self._plan_segment(
                grid, current, finish, obs, finish_left_idx, is_approach=(num_gates == 0)
            )
            if len(astar_segment) > 1:
                final_path.extend(astar_segment[1:])

        return np.asarray(final_path, dtype=float)

    def _plan_segment(self, grid, current, goal, obs, left_gate_idx, is_approach=False):
        """Plan ``current -> goal`` with any temporary, segment-local blocks.

        Two block sources, applied to a scratch copy of the grid (restored after):
          * gate reversal/flyback -- block the just-left gate's opening so A* turns
            around it rather than diving back through (see ``_should_block``);
          * obstacle side-commitment (only on the approach segment) -- block a wall
            on the *wrong* side of imminent obstacles so a late sensor reveal can't
            flip the pass side under the drone (see ``_commit_blocks``).
        If the blocks make the segment infeasible (A* falls back to a colliding
        straight line) they are dropped and the segment is replanned unblocked --
        i.e. only commit/block when there is genuinely room to.
        """
        blocks = self._gate_blocks(obs, current, goal, left_gate_idx)
        if is_approach:
            blocks += self._commit_blocks(obs, current)
        if not blocks:
            seg = self.plan_astar(grid, current, goal)
            return self._maybe_prune_segment(seg, grid)

        saved = grid.occupied
        grid.occupied = saved.copy()
        try:
            for center, quat, half in blocks:
                grid.block_oriented_box(center, quat, half)
            seg = self.plan_astar(grid, current, goal)
            seg = self._maybe_prune_segment(seg, grid)
        finally:
            grid.occupied = saved

        # Blocking made it infeasible -> the constraint was wrong, plan unblocked.
        if len(seg) <= 2 and not self._line_is_free(seg[0], seg[-1], grid):
            seg = self.plan_astar(grid, current, goal)
            seg = self._maybe_prune_segment(seg, grid)
        return seg

    def _gate_blocks(self, obs, current, goal, left_gate_idx):
        """Reversal/flyback block as a ``[(center, quat, half_extents)]`` list."""
        block = self._should_block(obs, current, goal, left_gate_idx)
        if block is None:
            return []
        center, quat, exit_dir = block
        half_depth = self.block_depth / 2.0
        return [(
            center - exit_dir * half_depth,
            quat,
            np.array([half_depth, self.block_opening_half, self.block_opening_half]),
        )]

    def _commit_blocks(self, obs, current):
        """Wall-block the wrong side of imminent obstacles to lock the pass side.

        Returns ``[(center, quat, half_extents)]`` for each obstacle that is ahead,
        within ``commit_distance``, roughly in the drone's path, and approached
        fast enough to matter. The committed side is the side the drone is already
        on (from its current lateral offset); the wall sits just past the obstacle
        on the opposite side, aligned with the travel direction.
        """
        if not self.commit_obstacles:
            return []
        vel = np.asarray(obs.get("vel", np.zeros(3)), dtype=float)[:2]
        speed = float(np.linalg.norm(vel))
        if speed < self.commit_min_speed:
            return []
        obstacles = np.asarray(obs.get("obstacles_pos", np.empty((0, 3))), dtype=float)
        if obstacles.size == 0:
            return []

        pos = np.asarray(current, dtype=float)
        u = vel / speed  # travel direction (xy)
        w = np.array([-u[1], u[0]])  # left normal

        blocks = []
        for o in obstacles:
            rel = o[:2] - pos[:2]
            along = float(rel @ u)
            lat = float(rel @ w)  # >0: obstacle to the drone's left
            if along <= 0.0 or along > self.commit_distance:
                continue  # behind, or too far to be imminent
            if abs(lat) > self.commit_lateral_band or abs(lat) < 1e-3:
                continue  # well off to the side, or dead-centre (no committed side)

            wrong_dir = np.sign(lat) * w  # block the side the obstacle is on
            center_xy = o[:2] + wrong_dir * (self.obstacle_radius + self.commit_wall_extent / 2.0)
            center = np.array([center_xy[0], center_xy[1], self.commit_z_half])
            # Box axes: local x = travel (u), local y = left normal (w), z = up. Use
            # the fixed right-handed frame [u, w, z] (the box is symmetric in y, so
            # the center offset alone puts the wall on the wrong side).
            mat = np.array([[u[0], w[0], 0.0], [u[1], w[1], 0.0], [0.0, 0.0, 1.0]])
            quat = R.from_matrix(mat).as_quat()
            half = np.array([
                self.commit_wall_thickness / 2.0,
                self.commit_wall_extent / 2.0,
                self.commit_z_half,
            ])
            blocks.append((center, quat, half))
        return blocks

    def _should_block(self, obs, current, goal, left_gate_idx):
        """Reversal/flyback decision for the gate left behind on a segment.

        Returns ``(center, quat, exit_dir)`` of the gate whose opening to block, or
        ``None`` to plan normally. Two checks on gate ``left_gate_idx``:
          1. forward runway clear -- a cone along the exit direction is obstacle
             free (the drone could keep flying forward); and
          2. next gate behind -- ``goal`` lies within ``reversal_cone_deg`` of
             straight behind the gate (a genuine reversal).
        A runway-clear reversal plans normally (A* may loop forward on its own);
        otherwise the opening is blocked. A clean forward fly-through is unaffected
        either way (its forward path never revisits the gate's entry-side opening).
        """
        if not self.block_passed_gate or left_gate_idx < 0:
            return None

        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        if left_gate_idx >= len(gates_pos):
            return None

        center = gates_pos[left_gate_idx]
        quat = gates_quat[left_gate_idx]
        current = np.asarray(current, dtype=float)
        goal = np.asarray(goal, dtype=float)

        # Exit direction: gate normal, signed toward where the drone now is.
        forward = R.from_quat(quat).apply(np.array([1.0, 0.0, 0.0]))
        sign = np.sign(np.dot(forward, current - center))
        exit_dir = forward * (sign if sign != 0 else 1.0)
        norm = np.linalg.norm(exit_dir)
        if norm < 1e-9:
            return None
        exit_dir = exit_dir / norm

        forward_clear = self._forward_cone_free(current, exit_dir, obs["obstacles_pos"])

        to_next = goal - center
        n = np.linalg.norm(to_next)
        if n < 1e-9:
            return None
        cos_behind = float(np.dot(to_next / n, -exit_dir))
        is_reversal = cos_behind >= np.cos(np.radians(self.reversal_cone_deg))

        if forward_clear and is_reversal:
            return None  # genuine flyback with a clear runway: plan normally
        return center, quat, exit_dir

    def _forward_cone_free(self, apex, direction, obstacles):
        """True if no obstacle lies in the forward cone from ``apex``."""
        obstacles = np.asarray(obstacles, dtype=float)
        if obstacles.size == 0:
            return True

        apex = np.asarray(apex, dtype=float)
        v = obstacles - apex
        along = v @ direction
        dist = np.linalg.norm(v, axis=1)
        cos_ang = np.where(dist > 1e-9, along / np.maximum(dist, 1e-9), 1.0)
        cos_thr = np.cos(np.radians(self.forward_cone_deg))
        in_cone = (along > 0.0) & (along <= self.forward_cone_length) & (cos_ang >= cos_thr)
        return not bool(np.any(in_cone))

    def plan_astar(self, grid, start, goal):
        start_idx = grid.world_to_grid(start)
        goal_idx = grid.world_to_grid(goal)

        snapped_start_idx = grid.nearest_free_idx(
            start_idx, max_distance=self.endpoint_snap_distance
        )
        snapped_goal_idx = grid.nearest_free_idx(
            goal_idx, max_distance=self.endpoint_snap_distance
        )

        if snapped_start_idx is None:
            print(f"A*(barebone): no free start cell near {start_idx}, world={start}")
            return np.vstack([start, goal])

        if snapped_goal_idx is None:
            print(f"A*(barebone): no free goal cell near {goal_idx}, world={goal}")
            return np.vstack([start, goal])

        idx_path = self._solver.plan(
            grid.occupied,
            snapped_start_idx,
            snapped_goal_idx,
            heuristic_weight=self.heuristic_weight,
        )
        if idx_path is None:
            print(
                "WARNING: A*(barebone) failed; falling back to straight segment:",
                "start=", start,
                "goal=", goal,
            )
            return np.vstack([start, goal])

        return np.asarray([grid.grid_to_world(idx) for idx in idx_path], dtype=float)

    def _maybe_prune_segment(self, path, grid):
        if not self.prune_path:
            return path
        return self._prune_path(path, grid)

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