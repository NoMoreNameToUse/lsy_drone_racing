"""Tier-0 optimized A* gate path generator.

``AStarImprovedPathGenerator`` produces the same kind of path as
``AStarGatePathGenerator`` (gate entry/center/exit anchors connected by A*
through free space), but adds the Tier-0 efficiency improvements:

- the optimized kernel :func:`astar_3d_improved` (no per-edge numpy allocation),
- a per-segment **search window** so A* only explores / allocates the local
  region between the current point and the next anchor,
- optional **coarse-to-fine** planning: a quick coarse-resolution A* defines a
  corridor inside which the fine-resolution A* searches.

The existing ``GatePassingPathGenerator`` and ``OccupancyGrid3D`` are reused
unmodified, and the original ``AStarGatePathGenerator`` is left untouched for
comparison.

Determinism / comparison note:
    The inner-loop optimization is bit-for-bit identical to the original. The
    search window and coarse-to-fine corridor are search-space restrictions and
    *can* change the result if the true optimum leaves the restricted region
    (both have straight-line / unrestricted fallbacks). Set
    ``use_search_window=False`` and ``use_coarse_to_fine=False`` to reproduce the
    original full-grid A* exactly while still benefiting from the inner-loop
    speedup.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controllers.modules.astar_3d_improved import AStar3DImproved
from lsy_drone_racing.control.controllers.modules.occupancy_grid_3d_improved import OccupancyGrid3D
from lsy_drone_racing.control.controllers.modules.initial_challenge.path_generator import GatePassingPathGenerator


class _BareboneSolverAdapter:
    """Adapt ``AStar3DBarebone`` to the ``AStar3DImproved.plan(grid, ...)`` API.

    The barebone solver is shortest-path only and works on ``grid.occupied``, so
    the velocity-bias / window / mask / cost-field keyword arguments used by the
    pure-Python solver are accepted and ignored.
    """

    def __init__(self):
        self._solver = AStar3DBarebone()

    def plan(self, grid, start_idx, goal_idx, *, heuristic_weight=1.0, **_ignored):
        return self._solver.plan(grid.occupied, start_idx, goal_idx, heuristic_weight=heuristic_weight)


class AStarImprovedPathGenerator:
    """Gate-anchored A* path generator with Tier-0 efficiency improvements."""

    def __init__(
        self,
        gate_passing_generator=None,
        grid_resolution: float = 0.075,
        safety_margin: float = 0.04,
        obstacle_radius: float = 0.20,
        heuristic_weight: float = 1.15,
        max_astar_iterations: int = 200_000,
        endpoint_snap_distance: float = 0.30,
        prune_path: bool = False,
        final_extension_distance: float = 0.60,
        velocity_bias_weight: float = 0.20,
        velocity_bias_decay: float = 8.0,
        min_velocity_for_bias: float = 0.10,
        # --- Tier-0 options ---
        # With reusable full-grid buffers the window no longer saves allocation
        # and weighted A* already focuses exploration, so the window is net
        # neutral-to-negative here; off by default. Kept for much larger grids.
        use_search_window: bool = False,
        window_margin: float = 0.40,
        # Coarse-to-fine has per-segment overhead (a coarse plan + corridor mask)
        # that does not pay off at this grid scale; left off by default. It is
        # kept available for much larger / finer grids where the fine search
        # dominates. See module docstring.
        use_coarse_to_fine: bool = False,
        coarse_factor: int = 2,
        corridor_radius: float = 0.15,
        # --- Clearance / risk cost (A: reshape A*'s cost) ---
        # Penalize cells close to the obstacle poles so the path stays in the
        # middle of free space, giving the tracker room to carry speed. Off by
        # default so that (with window/coarse also off) the path stays identical
        # to the original A*. The penalty is an extra cost rate that ramps from 0
        # at ``clearance_dist`` metres outside the inflated pole up to
        # ``clearance_weight`` at the inflated pole surface (raised to
        # ``clearance_exponent``). Gate frames are deliberately NOT penalized:
        # the drone must fly close through the aperture, and penalizing gates
        # both fights the objective and blows up the search.
        use_clearance_cost: bool = False,
        clearance_weight: float = 0.4,
        clearance_dist: float = 0.35,
        clearance_exponent: float = 2.0,

        # --- Momentum-aware replanning (two-candidate compare by est. time) ---
        # On a replan the shortest path can demand a near-reversal (e.g. back
        # through the gate just passed) because A* ignores the current velocity.
        # When enabled, the velocity-bearing first segment is planned twice --
        # the unconstrained shortest path vs a forward-committed path (routed
        # through a look-ahead waypoint along the current velocity) -- and the
        # faster one under a cheap time proxy is kept. The forward candidate is
        # dropped when its departure is blocked, so a genuine detour/reversal
        # still wins ("keep speed, but reverse if you must"). Off by default.
        use_momentum_compare: bool = False,
        momentum_min_speed: float = 0.5,
        momentum_lookahead_time: float = 0.3,
        momentum_min_dist: float = 0.2,
        momentum_max_dist: float = 0.8,
        speed_estimate: float = 2.0,
        turn_penalty: float = 0.15,
        # Seconds of penalty per (m/s of speed) per unit (1 - cos) of initial
        # deviation from the current velocity. Speed-scaled (see _estimate_time).
        initial_turn_penalty: float = 1000,
        # --- Backend: barebone numba A* (shortest path only, 26-connected) ---
        # Swaps the pure-Python solver for the JIT-compiled barebone A*. The
        # velocity-bias keyword args are ignored by it (shortest path only).
        use_barebone: bool = False,
    ):
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

        self.use_search_window = use_search_window
        self.window_margin = window_margin
        self.use_coarse_to_fine = use_coarse_to_fine
        self.coarse_factor = max(1, int(coarse_factor))
        self.corridor_radius = corridor_radius

        self.use_clearance_cost = use_clearance_cost
        self.clearance_weight = clearance_weight
        self.clearance_dist = clearance_dist
        self.clearance_exponent = clearance_exponent

        self.use_momentum_compare = use_momentum_compare
        self.momentum_min_speed = momentum_min_speed
        self.momentum_lookahead_time = momentum_lookahead_time
        self.momentum_min_dist = momentum_min_dist
        self.momentum_max_dist = momentum_max_dist
        self.speed_estimate = speed_estimate
        self.turn_penalty = turn_penalty
        self.initial_turn_penalty = initial_turn_penalty


        # Persistent solvers so the A* scratch buffers are reused across replans.
        # Separate instances for fine / coarse grids to avoid buffer-size thrash.
        self.use_barebone = use_barebone
        if use_barebone:
            self._fine_solver = _BareboneSolverAdapter()
            self._coarse_solver = _BareboneSolverAdapter()
        else:
            self._fine_solver = AStar3DImproved()
            self._coarse_solver = AStar3DImproved()

        # Per-generate clearance cost field (built in generate, used by the fine
        # solver). None when clearance shaping is disabled.
        self._cost_field = None

    def generate(self, obs, config=None):
        mandatory = self.gate_passing_generator.generate(obs, config)
        current_velocity = np.asarray(obs.get("vel", np.zeros(3)), dtype=float)

        grid = OccupancyGrid3D(
            obs=obs,
            config=config,
            resolution=self.grid_resolution,
            safety_margin=self.safety_margin,
            obstacle_radius=self.obstacle_radius,
        )
        self._cost_field = self._build_cost_field(grid)

        coarse_grid = None
        if self.use_coarse_to_fine and self.coarse_factor > 1:
            coarse_grid = OccupancyGrid3D(
                obs=obs,
                config=config,
                resolution=self.grid_resolution * self.coarse_factor,
                safety_margin=self.safety_margin,
                obstacle_radius=self.obstacle_radius,
            )

        final_path = [mandatory[0]]
        locked = [0]  # indices of gate anchors (entry/center/exit) + start/end

        num_gates = (len(mandatory) - 1) // 3
        current = mandatory[0]

        for gate_i in range(num_gates):
            base = 1 + 3 * gate_i

            entry = mandatory[base + 0]
            center = mandatory[base + 1]
            exit_ = mandatory[base + 2]

            segment_velocity = current_velocity if gate_i == 0 else None
            astar_segment = self._plan_segment(grid, coarse_grid, current, entry, segment_velocity)
            astar_segment = self._maybe_prune_segment(astar_segment, grid)

            if len(astar_segment) > 1:
                final_path.extend(astar_segment[1:])
                locked.append(len(final_path) - 1)  # gate entry: lock approach

            # Keep gate crossing direct and ordered, and locked during refinement.
            final_path.append(center)
            locked.append(len(final_path) - 1)
            final_path.append(exit_)
            locked.append(len(final_path) - 1)

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
            astar_segment = self._plan_segment(
                grid, coarse_grid, current, finish, final_segment_velocity
            )
            astar_segment = self._maybe_prune_segment(astar_segment, grid)

            if len(astar_segment) > 1:
                final_path.extend(astar_segment[1:])

            current = finish

        path = np.asarray(final_path, dtype=float)

        return path

    def _plan_segment(self, grid, coarse_grid, current, goal, velocity):
        """Plan one segment, using the momentum-aware compare when applicable."""
        if (
            self.use_momentum_compare
            and velocity is not None
            and np.linalg.norm(velocity) >= self.momentum_min_speed
        ):
            return self._two_candidate_segment(grid, coarse_grid, current, goal, velocity)
        return self.plan_astar(grid, coarse_grid, current, goal, preferred_velocity=velocity)

    def _two_candidate_segment(self, grid, coarse_grid, current, goal, velocity):
        """Return the faster of the shortest path and a forward-committed path."""
        cand_shortest = self.plan_astar(grid, coarse_grid, current, goal, preferred_velocity=None)
        cand_forward = self._plan_forward_committed(grid, coarse_grid, current, goal, velocity)

        if cand_forward is None or len(cand_forward) < 2:
            print("Momentum compare: forward commit infeasible, falling back to shortest path")
            print(cand_forward)
            return cand_shortest

        t_short = self._estimate_time(cand_shortest, velocity)
        t_forward = self._estimate_time(cand_forward, velocity)
        return cand_forward if t_forward < t_short else cand_shortest

    def _plan_forward_committed(self, grid, coarse_grid, current, goal, velocity):
        """Plan via a look-ahead waypoint along the current velocity.

        Returns None when the forward commitment is infeasible (look-ahead point
        blocked, or the straight departure is not collision-free) so the caller
        falls back to the shortest path -- i.e. reverse only when necessary.
        """
        current = np.asarray(current, dtype=float)
        speed = float(np.linalg.norm(velocity))
        if speed < 1e-9:
            return None

        v_unit = np.asarray(velocity, dtype=float) / speed
        look = float(
            np.clip(
                speed * self.momentum_lookahead_time,
                self.momentum_min_dist,
                self.momentum_max_dist,
            )
        )
        momentum = self.gate_passing_generator._clip_z(current + look * v_unit)
        momentum = np.clip(momentum, grid.bounds_low, grid.bounds_high)

        # The forward departure must be clear of obstacle POLES (real hazards).
        # Gate frames are deliberately ignored: just after passing a gate the
        # drone sits inside that gate's inflated frame, so a full-occupancy check
        # would falsely reject the forward commit exactly when it is needed. A
        # pole genuinely ahead still rejects it -> fall back to the shortest
        # (possibly reversing) path, i.e. reverse only when necessary.
        if not self._segment_pole_clear(current, momentum, grid):
            return None

        # plan_astar snaps the (possibly gate-frame) momentum point internally.
        onward = self.plan_astar(grid, coarse_grid, momentum, goal, preferred_velocity=velocity)
        return np.vstack([current[None, :], np.asarray(onward, dtype=float)])

    def _segment_pole_clear(self, a, b, grid, margin=0.0):
        """True if the segment a->b stays clear of every obstacle pole (XY).

        Pole-only by design (ignores gate frames); see _plan_forward_committed.
        """
        obstacles = np.asarray(grid.obstacles_pos, dtype=float)
        if obstacles.size == 0:
            return True

        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        dist = float(np.linalg.norm(b - a))
        n = max(2, int(np.ceil(dist / max(grid.resolution, 1e-6))))
        ts = np.linspace(0.0, 1.0, n)
        pts = a[None, :] * (1.0 - ts)[:, None] + b[None, :] * ts[:, None]  # (n, 3)

        d_xy = np.linalg.norm(pts[:, None, :2] - obstacles[None, :, :2], axis=2)  # (n, P)
        return bool(np.all(d_xy.min(axis=1) > self.obstacle_radius + margin))

    def _estimate_time(self, path, velocity):
        """Cheap traversal-time proxy: length term plus turn/deceleration penalties.

        The initial-turn term (deviation of the first move from the current
        velocity) is the key discriminator: a reversal incurs a large penalty.
        """
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return 0.0

        segs = np.diff(path, axis=0)
        seg_len = np.linalg.norm(segs, axis=1)
        t = float(np.sum(seg_len)) / max(self.speed_estimate, 1e-6)

        dirs = segs / np.maximum(seg_len, 1e-9)[:, None]

        # Interior turns.
        if len(dirs) >= 2:
            cos_int = np.clip(np.sum(dirs[:-1] * dirs[1:], axis=1), -1.0, 1.0)
            t += self.turn_penalty * float(np.sum(1.0 - cos_int))

        # Initial deviation from the current velocity. Scaled by speed: the
        # faster you are going, the more momentum a turn/reversal must shed, so
        # the costlier it is in time. This is the dominant discriminator.
        speed = float(np.linalg.norm(velocity))
        if speed > 1e-9 and seg_len[0] > 1e-9:
            cos0 = float(np.clip(np.dot(np.asarray(velocity, dtype=float) / speed, dirs[0]), -1.0, 1.0))
            t += self.initial_turn_penalty * speed * (1.0 - cos0)

        return t

    def plan_astar(self, grid, coarse_grid, start, goal, preferred_velocity=None):
        start_idx = grid.world_to_grid(start)
        goal_idx = grid.world_to_grid(goal)

        snapped_start_idx = grid.nearest_free_idx(
            start_idx, max_distance=self.endpoint_snap_distance
        )
        snapped_goal_idx = grid.nearest_free_idx(goal_idx, max_distance=self.endpoint_snap_distance)

        if snapped_start_idx is None:
            print(f"A*(improved): no free start cell near {start_idx}, world={start}")
            return np.vstack([start, goal])

        if snapped_goal_idx is None:
            print(f"A*(improved): no free goal cell near {goal_idx}, world={goal}")
            return np.vstack([start, goal])

        win_min, win_max = self._segment_window(grid, snapped_start_idx, snapped_goal_idx)

        allowed_mask = None
        if coarse_grid is not None:
            allowed_mask = self._coarse_corridor(
                coarse_grid,
                grid,
                start,
                goal,
                snapped_start_idx,
                snapped_goal_idx,
                preferred_velocity,
            )

        idx_path = self._run_astar(
            grid, snapped_start_idx, snapped_goal_idx, win_min, win_max, allowed_mask,
            preferred_velocity,
        )

        # If the corridor was too tight, retry once on the full window.
        if idx_path is None and allowed_mask is not None:
            print("A*(improved): corridor search failed, retrying on full window")
            idx_path = self._run_astar(
                grid, snapped_start_idx, snapped_goal_idx, win_min, win_max, None,
                preferred_velocity,
            )

        if idx_path is None:
            print(
                "WARNING: A*(improved) failed; falling back to straight segment:",
                "start=", start, "goal=", goal,
            )
            return np.vstack([start, goal])

        return np.asarray([grid.grid_to_world(idx) for idx in idx_path], dtype=float)

    def _run_astar(self, grid, start_idx, goal_idx, win_min, win_max, allowed_mask, preferred_velocity):
        return self._fine_solver.plan(
            grid,
            start_idx,
            goal_idx,
            win_min=win_min,
            win_max=win_max,
            allowed_mask=allowed_mask,
            cost_field=self._cost_field,
            max_iterations=self.max_astar_iterations,
            heuristic_weight=self.heuristic_weight,
            preferred_direction=preferred_velocity,
            direction_bias_weight=self.velocity_bias_weight,
            direction_bias_decay=self.velocity_bias_decay,
            min_direction_speed=self.min_velocity_for_bias,
        )

    def _build_cost_field(self, grid):
        """Pole-clearance extra cost rate per cell, or None if disabled.

        Analytic XY distance to the nearest obstacle pole (poles are vertical),
        minus the pole inflation radius. The cost ramps from 0 at
        ``clearance_dist`` metres outside the inflated pole up to
        ``clearance_weight`` at its surface. Built once per grid (cheap, no 3D
        distance transform) and shared across all segments. Gate frames are not
        penalized; they are handled by the hard occupancy only.
        """
        if not self.use_clearance_cost or self.clearance_weight <= 0.0:
            return None

        obstacles = np.asarray(grid.obstacles_pos, dtype=float)
        if obstacles.size == 0:
            return None

        sx_dim, sy_dim, sz_dim = (int(v) for v in grid.shape.tolist())
        xs = grid.bounds_low[0] + np.arange(sx_dim) * grid.resolution
        ys = grid.bounds_low[1] + np.arange(sy_dim) * grid.resolution

        # Min XY distance from each (x, y) column to any pole center.
        dx = xs[:, None, None] - obstacles[None, None, :, 0]  # (Sx, 1, P)
        dy = ys[None, :, None] - obstacles[None, None, :, 1]  # (1, Sy, P)
        clearance_xy = np.sqrt(dx * dx + dy * dy).min(axis=2)  # (Sx, Sy)

        # Distance outside the inflated pole, then ramp to a cost.
        clearance = np.maximum(clearance_xy - self.obstacle_radius, 0.0)
        d = max(self.clearance_dist, 1e-6)
        ramp = np.clip((d - clearance) / d, 0.0, 1.0) ** self.clearance_exponent
        field_2d = self.clearance_weight * ramp  # (Sx, Sy)

        # Broadcast over z into a contiguous (Sx, Sy, Sz) field.
        field = np.empty((sx_dim, sy_dim, sz_dim), dtype=float)
        field[:] = field_2d[:, :, None]
        return field

    def _segment_window(self, grid, start_idx, goal_idx):
        """Inclusive index box around start/goal, expanded by ``window_margin``."""
        shape = tuple(int(v) for v in grid.shape.tolist())
        if not self.use_search_window:
            return (0, 0, 0), (shape[0] - 1, shape[1] - 1, shape[2] - 1)

        margin = int(np.ceil(self.window_margin / grid.resolution))
        lo, hi = [], []
        for d in range(3):
            a = min(start_idx[d], goal_idx[d]) - margin
            b = max(start_idx[d], goal_idx[d]) + margin
            lo.append(int(max(0, a)))
            hi.append(int(min(shape[d] - 1, b)))
        return tuple(lo), tuple(hi)

    def _coarse_corridor(
        self,
        coarse_grid,
        fine_grid,
        start,
        goal,
        fine_start_idx,
        fine_goal_idx,
        preferred_velocity,
    ):
        """Plan coarsely and build a full-grid corridor mask for the fine pass.

        Returns the boolean mask (shape equal to the fine grid), or ``None`` if
        no usable corridor was found (in which case the caller searches the full
        window unrestricted).
        """
        c_start = coarse_grid.nearest_free_idx(
            coarse_grid.world_to_grid(start), max_distance=self.endpoint_snap_distance
        )
        c_goal = coarse_grid.nearest_free_idx(
            coarse_grid.world_to_grid(goal), max_distance=self.endpoint_snap_distance
        )
        if c_start is None or c_goal is None:
            return None

        coarse_idx_path = self._coarse_solver.plan(
            coarse_grid,
            c_start,
            c_goal,
            max_iterations=self.max_astar_iterations,
            heuristic_weight=self.heuristic_weight,
            preferred_direction=preferred_velocity,
            direction_bias_weight=self.velocity_bias_weight,
            direction_bias_decay=self.velocity_bias_decay,
            min_direction_speed=self.min_velocity_for_bias,
        )
        if not coarse_idx_path:
            return None

        coarse_world = np.asarray(
            [coarse_grid.grid_to_world(i) for i in coarse_idx_path], dtype=float
        )
        dense_world = self._densify(coarse_world, step=fine_grid.resolution)

        mask = np.zeros(tuple(int(v) for v in fine_grid.shape.tolist()), dtype=bool)
        r = int(np.ceil(self.corridor_radius / fine_grid.resolution))

        # Inflate the coarse centerline into the fine grid.
        for w in dense_world:
            self._mark_ball(mask, fine_grid.world_to_grid(w), r)

        # Always keep the snapped fine endpoints reachable.
        self._mark_ball(mask, fine_start_idx, r)
        self._mark_ball(mask, fine_goal_idx, r)

        if not mask.any():
            return None
        return mask

    @staticmethod
    def _mark_ball(mask, center_idx, r):
        """Mark an axis-aligned cube of half-size ``r`` around a grid cell."""
        wnx, wny, wnz = mask.shape
        cx, cy, cz = center_idx
        x0, x1 = max(0, cx - r), min(wnx - 1, cx + r)
        y0, y1 = max(0, cy - r), min(wny - 1, cy + r)
        z0, z1 = max(0, cz - r), min(wnz - 1, cz + r)
        if x0 <= x1 and y0 <= y1 and z0 <= z1:
            mask[x0 : x1 + 1, y0 : y1 + 1, z0 : z1 + 1] = True

    @staticmethod
    def _densify(points, step):
        """Resample a polyline so consecutive samples are <= ``step`` apart."""
        points = np.asarray(points, dtype=float)
        if len(points) < 2:
            return points

        out = [points[0]]
        step = max(float(step), 1e-9)
        for i in range(1, len(points)):
            a, b = points[i - 1], points[i]
            dist = float(np.linalg.norm(b - a))
            n = max(1, int(np.ceil(dist / step)))
            for k in range(1, n + 1):
                out.append(a + (b - a) * (k / n))
        return np.asarray(out, dtype=float)

    # --- Pruning helpers (copied verbatim from AStarGatePathGenerator) ---

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
