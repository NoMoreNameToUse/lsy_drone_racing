"""Curve-style gate waypoint generator for controller_rl."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controllers.modules.zihan.path_generator import GatePassingPathGenerator


class CurveGatePathGenerator:
    """Lightweight curve-style gate planner for RL.

    It follows the same high-level idea as path.py:
        start -> gate entry -> gate center -> gate exit -> ...

    Then it inserts simple lateral detour points around obstacle poles for
    free-space segments. The output remains a compact waypoint list, so
    controller_rl can still use DistanceTiming + SplineTrajectory afterwards.
    """

    def __init__(
        self,
        gate_entry_distance: float = 0.24,
        obstacle_clearance: float = 0.32,
        obstacle_detection_padding: float = 0.08,
        detour_side_bias: float = 0.18,
        final_extension_distance: float = 0.60,
        gate_inner_width: float = 0.40,
        gate_outer_width: float = 0.72,
        gate_frame_clearance: float = 0.05,
        gate_bypass_margin: float = 0.10,
        min_z: float = 0.08,
        max_z: float = 1.60,
        max_avoidance_passes: int = 2,
        max_gate_repair_passes: int = 2,
    ):
        """Initialize curve-style gate planning parameters."""
        self.gate_passing_generator = GatePassingPathGenerator(
            gate_entry_distance=gate_entry_distance,
            max_nudge=0.0,
            min_z=min_z,
            max_z=max_z,
        )
        self.obstacle_clearance = obstacle_clearance
        self.obstacle_detection_padding = obstacle_detection_padding
        self.detour_side_bias = detour_side_bias
        self.final_extension_distance = final_extension_distance
        self.gate_inner_half = gate_inner_width / 2.0
        self.gate_outer_half = gate_outer_width / 2.0
        self.gate_frame_clearance = gate_frame_clearance
        self.gate_bypass_margin = gate_bypass_margin
        self.min_z = min_z
        self.max_z = max_z
        self.max_avoidance_passes = max_avoidance_passes
        self.max_gate_repair_passes = max_gate_repair_passes

    def generate(self, obs: dict[str, Any], config: Any = None) -> np.ndarray:
        """Generate compact gate waypoints with geometric obstacle detours."""
        mandatory = self.gate_passing_generator.generate(obs, config)
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float)

        path = [mandatory[0]]
        num_gates = (len(mandatory) - 1) // 3
        current = mandatory[0]

        for gate_i in range(num_gates):
            base = 1 + 3 * gate_i
            entry = mandatory[base + 0]
            center = mandatory[base + 1]
            exit_ = mandatory[base + 2]

            path.extend(self._avoid_segment(current, entry, obstacles_pos)[1:])
            path.append(center)
            path.append(exit_)
            current = exit_

        finish = self._finish_point(obs)
        if finish is not None:
            path.extend(self._avoid_segment(current, finish, obstacles_pos)[1:])

        path = self._repair_gate_frame_crossings(path, obs)
        return np.asarray(self._remove_near_duplicates(path), dtype=float)

    def _finish_point(self, obs: dict[str, Any]) -> np.ndarray | None:
        target_gate = int(np.asarray(obs.get("target_gate", 0)).item())
        gates_pos_all = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat_all = np.asarray(obs["gates_quat"], dtype=float)

        remaining_gates_pos = gates_pos_all[target_gate:]
        remaining_gates_quat = gates_quat_all[target_gate:]
        if len(remaining_gates_pos) == 0 or self.final_extension_distance <= 0.0:
            return None

        last_gate_pos = remaining_gates_pos[-1]
        last_gate_quat = remaining_gates_quat[-1]
        rot = R.from_quat(last_gate_quat)
        forward = rot.apply(np.array([1.0, 0.0, 0.0]))
        forward = forward / max(np.linalg.norm(forward), 1e-9)

        finish = last_gate_pos + self.final_extension_distance * forward
        return self._clip_z(finish)

    def _avoid_segment(
        self, start: np.ndarray, goal: np.ndarray, obstacles_pos: np.ndarray
    ) -> list[np.ndarray]:
        segment = [self._clip_z(start), self._clip_z(goal)]

        for _ in range(self.max_avoidance_passes):
            updated = [segment[0]]
            inserted_any = False

            for a, b in zip(segment[:-1], segment[1:]):
                detour = self._detour_for_segment(a, b, obstacles_pos)
                if detour is not None:
                    updated.append(detour)
                    inserted_any = True
                updated.append(b)

            segment = self._remove_near_duplicates(updated)
            if not inserted_any:
                break

        return segment

    def _detour_for_segment(
        self, a: np.ndarray, b: np.ndarray, obstacles_pos: np.ndarray
    ) -> np.ndarray | None:
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        ab_xy = b[:2] - a[:2]
        ab_norm = np.linalg.norm(ab_xy)
        if ab_norm < 1e-9:
            return None

        best_obs = None
        best_dist = np.inf
        best_alpha = 0.0

        for obs in obstacles_pos:
            alpha = float(np.dot(obs[:2] - a[:2], ab_xy) / (ab_norm * ab_norm))
            alpha = np.clip(alpha, 0.0, 1.0)
            closest_xy = a[:2] + alpha * ab_xy
            dist = float(np.linalg.norm(closest_xy - obs[:2]))

            if dist < best_dist:
                best_dist = dist
                best_obs = obs
                best_alpha = alpha

        trigger_dist = self.obstacle_clearance + self.obstacle_detection_padding
        if best_obs is None or best_dist >= trigger_dist:
            return None

        direction = ab_xy / ab_norm
        left = np.array([-direction[1], direction[0]])
        right = -left

        midpoint_xy = (a[:2] + b[:2]) * 0.5
        side = left if np.dot(midpoint_xy - best_obs[:2], left) >= 0.0 else right

        detour_xy = best_obs[:2] + side * (self.obstacle_clearance + self.detour_side_bias)
        detour_z = (1.0 - best_alpha) * a[2] + best_alpha * b[2]
        return self._clip_z(np.array([detour_xy[0], detour_xy[1], detour_z], dtype=float))

    def _repair_gate_frame_crossings(
        self, points: list[np.ndarray], obs: dict[str, Any]
    ) -> list[np.ndarray]:
        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        repaired = self._remove_near_duplicates(points)

        for _ in range(self.max_gate_repair_passes):
            updated = [repaired[0]]
            inserted_any = False

            for a, b in zip(repaired[:-1], repaired[1:]):
                bypass = self._gate_frame_bypass(a, b, gates_pos, gates_quat)
                if bypass is not None:
                    updated.append(bypass)
                    inserted_any = True
                updated.append(b)

            repaired = self._remove_near_duplicates(updated)
            if not inserted_any:
                break

        return repaired

    def _gate_frame_bypass(
        self,
        a: np.ndarray,
        b: np.ndarray,
        gates_pos: np.ndarray,
        gates_quat: np.ndarray,
    ) -> np.ndarray | None:
        for gate_pos, gate_quat in zip(gates_pos, gates_quat):
            rot = R.from_quat(gate_quat)
            local_a = rot.inv().apply(np.asarray(a, dtype=float) - gate_pos)
            local_b = rot.inv().apply(np.asarray(b, dtype=float) - gate_pos)

            denom = local_b[0] - local_a[0]
            if abs(denom) < 1e-9:
                continue

            alpha = float(-local_a[0] / denom)
            if not 0.0 <= alpha <= 1.0:
                continue

            crossing = local_a + alpha * (local_b - local_a)
            if self._crosses_safe_gate_opening(crossing):
                continue

            if not self._crosses_gate_frame_region(crossing):
                continue

            return self._gate_bypass_point(crossing, gate_pos, rot)

        return None

    def _crosses_safe_gate_opening(self, local_point: np.ndarray) -> bool:
        safe_inner = max(0.0, self.gate_inner_half - self.gate_frame_clearance)
        return abs(local_point[1]) < safe_inner and abs(local_point[2]) < safe_inner

    def _crosses_gate_frame_region(self, local_point: np.ndarray) -> bool:
        outer = self.gate_outer_half + self.gate_frame_clearance
        return abs(local_point[1]) < outer and abs(local_point[2]) < outer

    def _gate_bypass_point(
        self, local_crossing: np.ndarray, gate_pos: np.ndarray, rot: R
    ) -> np.ndarray:
        outer = self.gate_outer_half + self.gate_frame_clearance
        y_sign = 1.0 if local_crossing[1] >= 0.0 else -1.0

        bypass_local = local_crossing.copy()
        bypass_local[0] = 0.0
        bypass_local[1] = y_sign * (outer + self.gate_bypass_margin)

        bypass_world = rot.apply(bypass_local) + gate_pos
        return self._clip_z(bypass_world)

    def _clip_z(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=float).copy()
        p[2] = np.clip(p[2], self.min_z, self.max_z)
        return p

    @staticmethod
    def _remove_near_duplicates(points: list[np.ndarray], tol: float = 1e-6) -> list[np.ndarray]:
        cleaned = []
        for point in points:
            point = np.asarray(point, dtype=float)
            if not cleaned or np.linalg.norm(point - cleaned[-1]) > tol:
                cleaned.append(point)
        return cleaned
