"""Occupancy grid for the improved A* path generator.

Copy of ``occupancy_grid_3d.OccupancyGrid3D`` with the unused per-point test
helpers (``_inside_inflated_pole`` and ``_inside_gate_frame``) removed. The grid
is built via the vectorized ``_mark_*`` routines and queried through
``occupied`` / ``is_free``; the removed methods were dead code. The original
module is left untouched for comparison.
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation as R


class OccupancyGrid3D:
    """3D occupancy grid (inflated obstacle poles + gate frames, openings free)."""

    INFLATION_RELIEF = 0.05

    # Gate support stand, from the gate MJCF (gate.xml `gate_stand`, a real
    # collision geom): a 0.10 x 0.10 m box, in gate-local coords centred at
    # (0, 0, -0.86) with half-extents (0.05, 0.05, 0.5) -> it runs from
    # gate_z - 1.36 up to gate_z - 0.36 (just below the frame) at the gate's
    # (x, y), rotating with the gate.
    STAND_HALF = (0.05, 0.05, 0.5)
    STAND_CENTER_LOCAL = (0.0, 0.0, -0.86)

    def __init__(
        self,
        obs,
        config=None,
        resolution: float = 0.075,
        safety_margin: float = 0.05,
        obstacle_radius: float = 0.23,
        gate_inner_width: float = 0.40,
        gate_outer_width: float = 0.72,
        gate_thickness: float = 0.08,
        gate_corner_radius: float = 0.05,
        model_gate_stand: bool = True,
        gate_stand_margin: float = 0.06,
        z_min: float = 0.05,
        z_max: float = 1.75,
    ):
        self.resolution = float(resolution)
        self.inv_resolution = 1.0 / self.resolution
        self.safety_margin = float(safety_margin)
        self.obstacle_radius = float(obstacle_radius)

        self.gate_inner_half = gate_inner_width / 2.0
        self.gate_outer_half = gate_outer_width / 2.0
        self.gate_thickness = gate_thickness
        self.gate_corner_radius = float(gate_corner_radius)
        self.model_gate_stand = bool(model_gate_stand)
        self.gate_stand_margin = float(gate_stand_margin)

        self.z_min = z_min
        self.z_max = z_max

        self.gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        self.gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        self.obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float)

        gate_outer_margin: float = 0.25
        gate_inner_margin: float = 0.17
        gate_plane_margin: float = 0.25

        self.gate_outer_margin = gate_outer_margin
        self.gate_inner_margin = gate_inner_margin
        self.gate_plane_margin = gate_plane_margin
        self.effective_obstacle_margin = max(0.0, self.safety_margin - self.INFLATION_RELIEF)
        self.effective_gate_outer_margin = max(0.0, self.gate_outer_margin - self.INFLATION_RELIEF)
        self.effective_gate_inner_margin = max(0.0, self.gate_inner_margin - self.INFLATION_RELIEF)
        self.effective_gate_plane_margin = max(0.0, self.gate_plane_margin - self.INFLATION_RELIEF)

        self.bounds_low, self.bounds_high = self._get_bounds(config)

        self.shape = np.ceil((self.bounds_high - self.bounds_low) / self.resolution).astype(int) + 1
        self.occupied = np.zeros(tuple(self.shape.tolist()), dtype=bool)

        self._build_occupancy_grid()

    def _get_bounds(self, config):
        if config is not None:
            try:
                low = np.asarray(config.env.track.safety_limits.pos_limit_low, dtype=float)
                high = np.asarray(config.env.track.safety_limits.pos_limit_high, dtype=float)

                low[2] = max(low[2], self.z_min)
                high[2] = min(high[2], self.z_max)
                return low, high
            except Exception:
                pass

        return (
            np.array([-2.5, -1.5, self.z_min], dtype=float),
            np.array([2.5, 1.5, self.z_max], dtype=float),
        )

    def world_to_grid(self, p):
        p = np.asarray(p, dtype=float)
        idx = np.round((p - self.bounds_low) * self.inv_resolution).astype(int)
        return tuple(idx.tolist())

    def clip_idx(self, idx):
        idx_array = np.asarray(idx, dtype=int)
        clipped = np.clip(idx_array, 0, self.shape - 1)
        return tuple(int(v) for v in clipped)

    def grid_to_world(self, idx):
        idx = np.asarray(idx, dtype=float)
        return self.bounds_low + idx * self.resolution

    def in_bounds_idx(self, idx):
        ix, iy, iz = idx
        return 0 <= ix < self.shape[0] and 0 <= iy < self.shape[1] and 0 <= iz < self.shape[2]

    def is_free(self, idx):
        if not self.in_bounds_idx(idx):
            return False

        return not self.occupied[idx]

    def nearest_free_idx(self, idx, max_distance: float = 0.30):
        clipped_idx = self.clip_idx(idx)

        if self.is_free(clipped_idx):
            return clipped_idx

        max_radius = max(1, int(np.ceil(max_distance * self.inv_resolution)))
        cx, cy, cz = clipped_idx

        for radius in range(1, max_radius + 1):
            best_idx = None
            best_dist_sq = math.inf

            x_min = max(0, cx - radius)
            x_max = min(self.shape[0] - 1, cx + radius)
            y_min = max(0, cy - radius)
            y_max = min(self.shape[1] - 1, cy + radius)
            z_min = max(0, cz - radius)
            z_max = min(self.shape[2] - 1, cz + radius)

            for ix in range(x_min, x_max + 1):
                for iy in range(y_min, y_max + 1):
                    for iz in range(z_min, z_max + 1):
                        if max(abs(ix - cx), abs(iy - cy), abs(iz - cz)) != radius:
                            continue

                        if self.occupied[ix, iy, iz]:
                            continue

                        dist_sq = (ix - cx) ** 2 + (iy - cy) ** 2 + (iz - cz) ** 2
                        if dist_sq < best_dist_sq:
                            best_dist_sq = dist_sq
                            best_idx = (ix, iy, iz)

            if best_idx is not None:
                return best_idx

        return None

    def block_oriented_box(
        self, center: np.ndarray, quat: np.ndarray, half_extents: np.ndarray
    ) -> None:
        """Mark cells inside an oriented box as occupied.

        Used to close just the opening of a gate the drone has passed: a thin,
        gate-aligned slab over the aperture (rather than a sphere) so the planner
        is forced around the gate without sacrificing the surrounding free space.
        ``half_extents`` are in gate-local axes (x = gate normal, y/z = opening).
        """
        center = np.asarray(center, dtype=float)
        half = np.asarray(half_extents, dtype=float)
        rot = R.from_quat(quat)
        inv_rot = rot.inv().as_matrix()

        signs = np.array(
            [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
        )
        world_corners = rot.apply(signs * half) + center
        lo = self._world_to_grid_clipped(world_corners.min(axis=0))
        hi = self._world_to_grid_clipped(world_corners.max(axis=0))

        x_idx = np.arange(lo[0], hi[0] + 1)
        y_idx = np.arange(lo[1], hi[1] + 1)
        z_idx = np.arange(lo[2], hi[2] + 1)
        if x_idx.size == 0 or y_idx.size == 0 or z_idx.size == 0:
            return

        offsets = np.stack(
            np.meshgrid(
                self.bounds_low[0] + x_idx * self.resolution - center[0],
                self.bounds_low[1] + y_idx * self.resolution - center[1],
                self.bounds_low[2] + z_idx * self.resolution - center[2],
                indexing="ij",
            ),
            axis=-1,
        )
        local = offsets @ inv_rot.T
        mask = np.all(np.abs(local) < half, axis=-1)
        self.occupied[
            x_idx[:, None, None], y_idx[None, :, None], z_idx[None, None, :]
        ] |= mask

    def _build_occupancy_grid(self):
        self._mark_obstacle_poles()
        self._mark_gate_frames()
        if self.model_gate_stand:
            self._mark_gate_stands()

    @staticmethod
    def _box_sdf_2d(
        ay: np.ndarray, az: np.ndarray, hy: float, hz: float
    ) -> np.ndarray:
        """Signed distance to an axis-aligned rectangle (negative inside).

        ``ay``/``az`` are |y|/|z| (>= 0); ``hy``/``hz`` are half-extents. Using
        the true Euclidean distance (vs. a per-axis box test) is what rounds the
        corners: ``_box_sdf_2d(...) < margin`` is a rounded rectangle with corner
        radius ``margin``, not a square.
        """
        qy = ay - hy
        qz = az - hz
        outside = np.hypot(np.maximum(qy, 0.0), np.maximum(qz, 0.0))
        inside = np.minimum(np.maximum(qy, qz), 0.0)
        return outside + inside

    @staticmethod
    def _box_dist_3d(local: np.ndarray, center: np.ndarray, half: np.ndarray) -> np.ndarray:
        """Euclidean distance from points ``local`` to an axis-aligned box (0 inside)."""
        q = np.abs(local - np.asarray(center)) - np.asarray(half)
        return np.linalg.norm(np.maximum(q, 0.0), axis=-1)

    def _world_to_grid_clipped(self, p):
        idx = np.round(
            (np.asarray(p, dtype=float) - self.bounds_low) * self.inv_resolution
        ).astype(int)
        return np.clip(idx, 0, self.shape - 1)

    def _mark_obstacle_poles(self):
        radius = self.obstacle_radius + self.effective_obstacle_margin
        radius_sq = radius * radius

        for obs_p in self.obstacles_pos:
            xy_low = self._world_to_grid_clipped([obs_p[0] - radius, obs_p[1] - radius, self.z_min])
            xy_high = self._world_to_grid_clipped([obs_p[0] + radius, obs_p[1] + radius, self.z_max])

            x_idx = np.arange(xy_low[0], xy_high[0] + 1)
            y_idx = np.arange(xy_low[1], xy_high[1] + 1)

            if x_idx.size == 0 or y_idx.size == 0:
                continue

            x_world = self.bounds_low[0] + x_idx * self.resolution
            y_world = self.bounds_low[1] + y_idx * self.resolution

            dist_sq = (x_world[:, None] - obs_p[0]) ** 2 + (y_world[None, :] - obs_p[1]) ** 2
            mask_xy = dist_sq < radius_sq

            if np.any(mask_xy):
                self.occupied[x_idx[:, None], y_idx[None, :], :] |= mask_xy[:, :, None]

    def _mark_gate_frames(self):
        plane_limit = self.gate_thickness / 2.0 + self.effective_gate_plane_margin
        outer = self.gate_outer_half + self.effective_gate_outer_margin
        inner = max(0.0, self.gate_inner_half - self.effective_gate_inner_margin)

        local_corners = np.array(
            [
                [sx * plane_limit, sy * outer, sz * outer]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=float,
        )

        for gate_pos, gate_quat in zip(self.gates_pos, self.gates_quat):
            rot = R.from_quat(gate_quat)
            inv_rot = rot.inv().as_matrix()
            world_corners = rot.apply(local_corners) + gate_pos

            bbox_low = self._world_to_grid_clipped(np.min(world_corners, axis=0))
            bbox_high = self._world_to_grid_clipped(np.max(world_corners, axis=0))

            x_idx = np.arange(bbox_low[0], bbox_high[0] + 1)
            y_idx = np.arange(bbox_low[1], bbox_high[1] + 1)
            z_idx = np.arange(bbox_low[2], bbox_high[2] + 1)

            if x_idx.size == 0 or y_idx.size == 0 or z_idx.size == 0:
                continue

            x_world = self.bounds_low[0] + x_idx * self.resolution
            y_world = self.bounds_low[1] + y_idx * self.resolution
            z_world = self.bounds_low[2] + z_idx * self.resolution

            world_offsets = np.stack(
                np.meshgrid(
                    x_world - gate_pos[0],
                    y_world - gate_pos[1],
                    z_world - gate_pos[2],
                    indexing="ij",
                ),
                axis=-1,
            )
            local = world_offsets @ inv_rot.T

            # Rounded (Euclidean) inflation instead of axis-aligned box tests, so
            # the frame's outer corners and the opening corners are rounded rather
            # than sharp squares. Straight-edge clearances are unchanged: the outer
            # reach stays gate_outer_half + outer_margin and the opening edge stays
            # at `inner`; only the corners round (radius outer_margin outside, and
            # gate_corner_radius around the opening). This lets a side/angled
            # approach curve around the frame corner instead of being blocked by a
            # protruding square corner.
            ay = np.abs(local[..., 1])
            az = np.abs(local[..., 2])
            outer_margin = max(0.0, outer - self.gate_outer_half)

            near_gate_plane = np.abs(local[..., 0]) < plane_limit
            within_outer = self._box_sdf_2d(ay, az, self.gate_outer_half, self.gate_outer_half) \
                < outer_margin
            rc = min(self.gate_corner_radius, inner)
            inside_inner_opening = self._box_sdf_2d(ay, az, inner - rc, inner - rc) < rc

            gate_mask = near_gate_plane & within_outer & ~inside_inner_opening
            self.occupied[
                x_idx[:, None, None],
                y_idx[None, :, None],
                z_idx[None, None, :],
            ] |= gate_mask

    def _mark_gate_stands(self) -> None:
        """Mark the support stand below each gate (the post it stands on).

        The stand is a real collision geom that the frame-only marking ignored, so
        A* could route a path straight through the post under a gate. It is a box
        in gate-local coords (see ``STAND_*``) that rotates with the gate; only the
        portion inside the grid's z-range is marked (the stand runs down to the
        floor, which is below ``z_min``).
        """
        center_local = np.array(self.STAND_CENTER_LOCAL, dtype=float)
        half = np.array(self.STAND_HALF, dtype=float)
        margin = self.gate_stand_margin
        inflated = half + margin

        corner_signs = np.array(
            [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)]
        )
        local_corners = center_local + corner_signs * inflated

        for gate_pos, gate_quat in zip(self.gates_pos, self.gates_quat):
            rot = R.from_quat(gate_quat)
            inv_rot = rot.inv().as_matrix()
            world_corners = rot.apply(local_corners) + gate_pos

            bbox_low = self._world_to_grid_clipped(np.min(world_corners, axis=0))
            bbox_high = self._world_to_grid_clipped(np.max(world_corners, axis=0))

            x_idx = np.arange(bbox_low[0], bbox_high[0] + 1)
            y_idx = np.arange(bbox_low[1], bbox_high[1] + 1)
            z_idx = np.arange(bbox_low[2], bbox_high[2] + 1)

            if x_idx.size == 0 or y_idx.size == 0 or z_idx.size == 0:
                continue

            world_offsets = np.stack(
                np.meshgrid(
                    self.bounds_low[0] + x_idx * self.resolution - gate_pos[0],
                    self.bounds_low[1] + y_idx * self.resolution - gate_pos[1],
                    self.bounds_low[2] + z_idx * self.resolution - gate_pos[2],
                    indexing="ij",
                ),
                axis=-1,
            )
            local = world_offsets @ inv_rot.T

            stand_mask = self._box_dist_3d(local, center_local, half) < margin
            self.occupied[
                x_idx[:, None, None],
                y_idx[None, :, None],
                z_idx[None, None, :],
            ] |= stand_mask
