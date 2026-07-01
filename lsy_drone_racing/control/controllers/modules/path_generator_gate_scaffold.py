"""Gate-passage path generator using a P0-P4 scaffold per gate.

A new approach to defining gate-passage waypoints (vs. the entry/center/exit
scaffold in ``path_generator.GatePassingPathGenerator``). For each remaining gate
it builds five anchors:

    P0  far approach   - on the crossing axis, leaned toward the previous anchor
    P1  near approach  - on the crossing axis just before the gate
    P2  crossing       - racing-line apex, shifted inside the aperture
    P3  near exit      - on the crossing axis just after the gate
    P4  far exit/turn  - on the crossing axis, leaned toward the next gate (if safe)

Key differences from the old scaffold:
- The crossing axis ``d`` is oriented along the **direction of travel** (its sign
  flips based on the approach), instead of the gate's fixed stored normal -- so
  the approach is never accidentally reversed.
- The crossing point is a **racing-line apex** within the aperture (not the
  dead-center), letting the tracker carry more speed through a turn.
- Every anchor is **validated** against poles and the gate frame; an invalid
  point is repaired by backing off its discretionary shift, then nudging in XY,
  then snapping to the nearest free cell.

Gaps between gates (previous P4 -> next P0, and start -> first P0) are connected
with the fast :class:`AStar3DBarebone` over the improved occupancy grid. The
result is a full ``(N, 3)`` world-space path, a drop-in replacement for
``AStarImprovedPathGenerator`` in a controller. Existing generators are untouched.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controllers.modules.astar_3d_barebone import AStar3DBarebone
from lsy_drone_racing.control.controllers.modules.occupancy_grid_3d_improved import OccupancyGrid3D


class GateScaffoldPathGenerator:
    """P0-P4 gate-passage path generator with travel-aligned, apex-shifted crossings."""

    def __init__(
        self,
        grid_resolution: float = 0.05,
        safety_margin: float = 0.05,
        obstacle_radius: float = 0.215,
        heuristic_weight: float = 1.0,
        prune: bool = True,
        # reverse-through-gate deliberation
        deliberate_reverse: bool = True,
        reverse_min_speed: float = 0.3,
        reverse_aperture_margin: float = 0.0,
        speed_estimate: float = 2.0,
        turn_penalty: float = 0.05,
        initial_turn_penalty: float = 1000.0,
        # scaffold geometry
        near_dist: float = 0.15,
        far_dist: float = 0.25,
        apex_max_shift: float = 0.08,
        apex_margin: float = 0.03,
        gate_inner_width: float = 0.40,
        prev_bias_gain: float = 0.4,
        next_bias_gain: float = 0.4,
        max_lean: float = 0.30,
        # validity / repair
        pole_clearance: float = 0.05,
        max_nudge: float = 0.20,
        snap_distance: float = 0.30,
        z_min: float = 0.08,
        z_max: float = 1.60,
        # extras
        final_extension_distance: float = 0.60,
        use_postprocess: bool = False,
        pp_iterations: int = 10,
        pp_smooth: float = 0.2,
        pp_repulse: float = 0.15,
        pp_influence: float = 0.35,
        pp_max_step: float = 0.04,
    ):
        self.grid_resolution = grid_resolution
        self.safety_margin = safety_margin
        self.obstacle_radius = obstacle_radius
        self.heuristic_weight = heuristic_weight
        self.prune = prune

        self.deliberate_reverse = deliberate_reverse
        self.reverse_min_speed = reverse_min_speed
        self.reverse_aperture_margin = reverse_aperture_margin
        self.speed_estimate = speed_estimate
        self.turn_penalty = turn_penalty
        self.initial_turn_penalty = initial_turn_penalty

        self.near_dist = near_dist
        self.far_dist = far_dist
        self.apex_max_shift = apex_max_shift
        self.apex_margin = apex_margin
        self.gate_inner_half = gate_inner_width / 2.0
        self.prev_bias_gain = prev_bias_gain
        self.next_bias_gain = next_bias_gain
        self.max_lean = max_lean

        self.pole_clearance = pole_clearance
        self.max_nudge = max_nudge
        self.snap_distance = snap_distance
        self.z_min = z_min
        self.z_max = z_max

        self.final_extension_distance = final_extension_distance

        self._solver = AStar3DBarebone()

        self.use_postprocess = use_postprocess

    # ------------------------------------------------------------------ public

    def generate(self, obs, config=None):
        """Return a full (N, 3) world-space path through the remaining gates.

        If the natural (shortest) path reverses back through the gate the drone
        just passed (``target_gate - 1``), a second path is built with that gate's
        aperture blocked -- forcing a forward detour -- and whichever of the two
        is faster by :meth:`_estimate_time` (which penalizes shedding momentum) is
        returned.
        """
        target = int(np.asarray(obs.get("target_gate", 0)).item())
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        start = self._clip_z(np.asarray(obs["pos"], dtype=float))
        approach_start = np.asarray(obs["pos"], dtype=float)
        velocity = np.asarray(obs.get("vel", np.zeros(3)), dtype=float)

        rem_pos = gates_pos[target:]
        rem_quat = gates_quat[target:]
        if len(rem_pos) == 0:
            return np.asarray([start], dtype=float)

        grid = OccupancyGrid3D(
            obs=obs,
            config=config,
            resolution=self.grid_resolution,
            safety_margin=self.safety_margin,
            obstacle_radius=self.obstacle_radius,
        )
        path = self._build_path(grid, start, approach_start, rem_pos, rem_quat)

        if self._should_deliberate(path, target, gates_pos, gates_quat, velocity):
            grid_blocked = OccupancyGrid3D(
                obs=obs,
                config=config,
                resolution=self.grid_resolution,
                safety_margin=self.safety_margin,
                obstacle_radius=self.obstacle_radius,
            )
            self._block_gate_aperture(grid_blocked, gates_pos[target - 1], gates_quat[target - 1])
            forward = self._build_path(grid_blocked, start, approach_start, rem_pos, rem_quat)
            if self._estimate_time(forward, velocity) < self._estimate_time(path, velocity):
                path = forward

        return path

    def _build_path(self, grid, start, approach_start, rem_pos, rem_quat):
        """Build the full (N, 3) path through ``rem_pos`` over a given grid."""
        n_gates = len(rem_pos)
        path = [start]
        locked = []
        prev = start
        approach_ref = approach_start  # apex chord origin
        last_d = np.array([1.0, 0.0, 0.0])

        for i in range(n_gates):
            center = rem_pos[i]
            quat = rem_quat[i]
            next_center = rem_pos[i + 1] if i + 1 < n_gates else None

            (p0, p1, p2, p3, p4), d = self._gate_scaffold(
                grid, center, quat, approach_ref, next_center
            )
            last_d = d

            seg = self._connect(grid, prev, p0)
            if len(seg) > 1:
                path.extend(seg[1:])
            else:
                path.append(p0)

            path.append(p1); locked.append(len(path) - 1)
            path.append(p2); locked.append(len(path) - 1)
            path.append(p3); locked.append(len(path) - 1)
            path.append(p4)

            prev = p4
            approach_ref = center  # next gate's apex chord starts at this center

        if self.final_extension_distance > 0.0:
            finish = self._clip_z(prev + self.final_extension_distance * last_d)
            seg = self._connect(grid, prev, finish)
            if len(seg) > 1:
                path.extend(seg[1:])

        out = np.asarray(path, dtype=float)
        if self.use_postprocess:
            out = self._smoother.smooth(out, grid, locked=locked)
        return out

    # --------------------------------------------------- reverse deliberation

    def _should_deliberate(self, path, target, gates_pos, gates_quat, velocity):
        """True when a forward-detour alternative is worth building.

        Only when reverse deliberation is enabled, a previous gate exists, the
        drone is moving fast enough that shedding momentum matters, and the
        natural path actually reverses through that previous gate's aperture.
        """
        if not self.deliberate_reverse or target < 1:
            return False
        if float(np.linalg.norm(velocity)) < self.reverse_min_speed:
            return False
        return self._path_crosses_gate(path, gates_pos[target - 1], gates_quat[target - 1])

    def _path_crosses_gate(self, path, center, quat):
        """True if any path segment pierces the gate plane within its aperture."""
        normal, lateral, vertical = self._axes(quat)
        half = self.gate_inner_half + self.reverse_aperture_margin
        pts = np.asarray(path, dtype=float)
        if len(pts) < 2:
            return False
        sn = (pts - center) @ normal  # signed distance to gate plane, per point
        for i in range(len(pts) - 1):
            s0, s1 = sn[i], sn[i + 1]
            if (s0 > 0.0) == (s1 > 0.0) or s0 == s1:
                continue  # both on one side -> no crossing
            t = s0 / (s0 - s1)
            off = (pts[i] + t * (pts[i + 1] - pts[i])) - center
            if abs(float(off @ lateral)) <= half and abs(float(off @ vertical)) <= half:
                return True
        return False

    def _block_gate_aperture(self, grid, center, quat, thickness=0.16):
        """Mark a gate's open aperture as occupied so A* cannot route through it."""
        normal, lateral, vertical = self._axes(quat)
        reach = self.gate_inner_half + thickness
        lo = np.clip(grid.world_to_grid(center - reach), 0, grid.shape - 1)
        hi = np.clip(grid.world_to_grid(center + reach), 0, grid.shape - 1)
        xs = np.arange(lo[0], hi[0] + 1)
        ys = np.arange(lo[1], hi[1] + 1)
        zs = np.arange(lo[2], hi[2] + 1)
        if xs.size == 0 or ys.size == 0 or zs.size == 0:
            return
        gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = grid.bounds_low + np.stack([gx, gy, gz], axis=-1) * grid.resolution
        rel = pts - center
        dn = np.abs(rel @ normal)
        dl = np.abs(rel @ lateral)
        dv = np.abs(rel @ vertical)
        mask = (dn <= 0.5 * thickness) & (dl <= self.gate_inner_half) & (dv <= self.gate_inner_half)
        grid.occupied[gx[mask], gy[mask], gz[mask]] = True

    def _estimate_time(self, path, velocity):
        """Cheap traversal-time proxy: length plus turn/momentum-shedding penalties.

        The initial-turn term (deviation of the first move from the current
        velocity, scaled by speed) is the dominant discriminator: a reversal that
        sheds momentum incurs a large penalty, so a longer forward detour can win.
        """
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return 0.0
        segs = np.diff(path, axis=0)
        seg_len = np.linalg.norm(segs, axis=1)
        t = float(np.sum(seg_len)) / max(self.speed_estimate, 1e-6)
        dirs = segs / np.maximum(seg_len, 1e-9)[:, None]

        if len(dirs) >= 2:
            cos_int = np.clip(np.sum(dirs[:-1] * dirs[1:], axis=1), -1.0, 1.0)
            t += self.turn_penalty * float(np.sum(1.0 - cos_int))

        speed = float(np.linalg.norm(velocity))
        if speed > 1e-9 and seg_len[0] > 1e-9:
            cos0 = float(np.clip(np.dot(velocity / speed, dirs[0]), -1.0, 1.0))
            t += self.initial_turn_penalty * speed * (1.0 - cos0)
        return t

    # ------------------------------------------------------------ scaffold

    def _gate_scaffold(self, grid, center, quat, approach_ref, next_center):
        """Build [P0, P1, P2, P3, P4] for one gate plus the travel axis ``d``."""
        normal, lateral, vertical = self._axes(quat)

        # Travel direction: orient the crossing axis along the approach.
        to_gate = center - approach_ref
        s = float(np.dot(to_gate, normal))
        if abs(s) < 1e-6:
            if next_center is not None and abs(np.dot(next_center - center, normal)) > 1e-6:
                d = normal * (1.0 if np.dot(next_center - center, normal) > 0 else -1.0)
            else:
                d = normal.copy()
        else:
            d = normal * (1.0 if s > 0 else -1.0)

        # Racing-line apex (P2): where the approach->exit chord pierces the plane.
        exit_ref = next_center if next_center is not None else (center + d)
        apex_offset = self._apex_offset(approach_ref, exit_ref, center, normal, lateral, vertical)
        p2 = self._validate_offset_point(center, apex_offset, grid)

        # Near anchors on the crossing axis.
        p1 = self._validate_axis_point(p2 - self.near_dist * d, grid)
        p3 = self._validate_axis_point(p2 + self.near_dist * d, grid)

        # Far approach: lean toward the previous anchor.
        p0 = self._lean_and_validate(p2 - self.far_dist * d, approach_ref, d, self.prev_bias_gain, grid)

        # Far exit: lean toward the next gate only when safe (validation backs off).
        if next_center is not None:
            p4 = self._lean_and_validate(
                p2 + self.far_dist * d, next_center, d, self.next_bias_gain, grid
            )
        else:
            p4 = self._validate_axis_point(p2 + self.far_dist * d, grid)

        return [p0, p1, p2, p3, p4], d

    def _apex_offset(self, a, b, center, normal, lateral, vertical):
        """Lateral/vertical offset of the apex within the aperture (clamped)."""
        ab = b - a
        denom = float(np.dot(ab, normal))
        if abs(denom) < 1e-6:
            return np.zeros(3)
        t = float(np.dot(center - a, normal)) / denom
        x = a + t * ab
        off = x - center
        lim = min(max(0.0, self.gate_inner_half - self.apex_margin), self.apex_max_shift)
        lat = float(np.clip(np.dot(off, lateral), -lim, lim))
        vert = float(np.clip(np.dot(off, vertical), -lim, lim))
        return lat * lateral + vert * vertical

    def _lean_and_validate(self, base, toward, d, gain, grid):
        """Lean ``base`` toward ``toward`` (perp to ``d``), then validate/repair."""
        v = np.asarray(toward, dtype=float) - base
        v_perp = v - np.dot(v, d) * d
        lean = gain * v_perp
        nrm = float(np.linalg.norm(lean))
        if nrm > self.max_lean:
            lean = lean / nrm * self.max_lean
        return self._validate_offset_point(base, lean, grid)

    # ------------------------------------------------------------ validity

    def _validate_offset_point(self, base, offset, grid):
        """Back off the discretionary ``offset`` toward ``base``, then nudge/snap."""
        for scale in (1.0, 0.75, 0.5, 0.25, 0.0):
            p = self._clip_z(base + scale * offset)
            if self._is_valid(p, grid):
                return p
        return self._nudge_then_snap(self._clip_z(base), grid)

    def _validate_axis_point(self, p, grid):
        """Validate an on-axis point (no discretionary shift): nudge then snap."""
        p = self._clip_z(p)
        if self._is_valid(p, grid):
            return p
        return self._nudge_then_snap(p, grid)

    def _nudge_then_snap(self, p, grid):
        nudged = self._nudge_from_poles(p, grid)
        if self._is_valid(nudged, grid):
            return nudged
        snapped = self._snap(nudged, grid)
        return snapped if snapped is not None else nudged

    def _is_valid(self, p, grid):
        if p[2] < self.z_min or p[2] > self.z_max:
            return False
        if self._min_pole_dist(p, grid) <= self.obstacle_radius + self.pole_clearance:
            return False
        return grid.is_free(grid.world_to_grid(p))

    def _nudge_from_poles(self, p, grid):
        p = np.asarray(p, dtype=float).copy()
        obstacles = np.asarray(grid.obstacles_pos, dtype=float)
        thresh = self.obstacle_radius + self.pole_clearance
        push = np.zeros(2)
        for ob in obstacles:
            delta = p[:2] - ob[:2]
            dist = float(np.linalg.norm(delta))
            if dist < 1e-9:
                direction = np.array([1.0, 0.0])
                dist = 1e-9
            else:
                direction = delta / dist
            if dist < thresh:
                push += (thresh - dist) * direction
        nrm = float(np.linalg.norm(push))
        if nrm > self.max_nudge:
            push = push / nrm * self.max_nudge
        p[:2] += push
        return self._clip_z(p)

    def _snap(self, p, grid):
        idx = grid.nearest_free_idx(grid.world_to_grid(p), max_distance=self.snap_distance)
        if idx is None:
            return None
        return grid.grid_to_world(idx)

    @staticmethod
    def _min_pole_dist(p, grid):
        obstacles = np.asarray(grid.obstacles_pos, dtype=float)
        if obstacles.size == 0:
            return np.inf
        return float(np.min(np.linalg.norm(obstacles[:, :2] - np.asarray(p)[:2], axis=1)))

    # ------------------------------------------------------------ connection

    def _connect(self, grid, start, goal):
        """Barebone A* from start to goal (world), with snapping + straight fallback.

        The dense A* output is line-of-sight pruned to the fewest waypoints whose
        connecting straight segments stay collision-free (endpoints kept).
        """
        si = grid.nearest_free_idx(grid.world_to_grid(start), max_distance=self.snap_distance)
        gi = grid.nearest_free_idx(grid.world_to_grid(goal), max_distance=self.snap_distance)
        if si is None or gi is None:
            return np.vstack([start, goal])
        idx_path = self._solver.plan(grid.occupied, si, gi, heuristic_weight=self.heuristic_weight)
        if not idx_path:
            return np.vstack([start, goal])
        seg = np.asarray([grid.grid_to_world(idx) for idx in idx_path], dtype=float)
        if self.prune:
            seg = self._prune(seg, grid)
        return seg

    def _prune(self, path, grid):
        """Greedy line-of-sight shortcut: drop waypoints a straight segment can skip."""
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
        """True if the straight segment a->b stays in free cells (vectorized)."""
        if step is None:
            step = grid.resolution * 0.5
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        dist = float(np.linalg.norm(b - a))
        if dist < 1e-9:
            return grid.is_free(grid.world_to_grid(a))
        n = max(2, int(np.ceil(dist / step)))
        ts = np.linspace(0.0, 1.0, n)
        pts = a[None, :] * (1.0 - ts)[:, None] + b[None, :] * ts[:, None]
        idx = np.round((pts - grid.bounds_low) * grid.inv_resolution).astype(int)
        shape = np.asarray(grid.shape)
        if not np.all((idx >= 0) & (idx < shape)):
            return False
        return not grid.occupied[idx[:, 0], idx[:, 1], idx[:, 2]].any()

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _axes(quat):
        rot = R.from_quat(quat)
        normal = GateScaffoldPathGenerator._normalize(rot.apply(np.array([1.0, 0.0, 0.0])))
        lateral = GateScaffoldPathGenerator._normalize(rot.apply(np.array([0.0, 1.0, 0.0])))
        vertical = GateScaffoldPathGenerator._normalize(rot.apply(np.array([0.0, 0.0, 1.0])))
        return normal, lateral, vertical

    @staticmethod
    def _normalize(v):
        norm = float(np.linalg.norm(v))
        return v if norm < 1e-9 else v / norm

    def _clip_z(self, p):
        p = np.asarray(p, dtype=float).copy()
        p[2] = np.clip(p[2], self.z_min, self.z_max)
        return p
