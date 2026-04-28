import math

import numpy as np
from scipy.spatial.transform import Rotation as R


class OccupancyGrid3D:
    """
    Simple 3D occupancy grid for drone racing.

    Occupied:
        - outside world bounds
        - ground / too high
        - inflated obstacle poles
        - gate frames, with gate openings left free
    """

    INFLATION_RELIEF = 0.05

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

        self.z_min = z_min
        self.z_max = z_max

        self.gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        self.gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        self.obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float)

        gate_outer_margin: float = 0.25
        gate_inner_margin: float = 0.16
        gate_plane_margin: float = 0.25

        self.gate_outer_margin = gate_outer_margin
        self.gate_inner_margin = gate_inner_margin
        self.gate_plane_margin = gate_plane_margin
        self.effective_obstacle_margin = max(0.0, self.safety_margin - self.INFLATION_RELIEF)
        self.effective_gate_outer_margin = max(0.0, self.gate_outer_margin - self.INFLATION_RELIEF)
        self.effective_gate_inner_margin = max(0.0, self.gate_inner_margin - self.INFLATION_RELIEF)
        self.effective_gate_plane_margin = max(0.0, self.gate_plane_margin - self.INFLATION_RELIEF)

        self.bounds_low, self.bounds_high = self._get_bounds(config)

        self.shape = np.ceil(
            (self.bounds_high - self.bounds_low) / self.resolution
        ).astype(int) + 1
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
        return (
            0 <= ix < self.shape[0]
            and 0 <= iy < self.shape[1]
            and 0 <= iz < self.shape[2]
        )

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

    def _build_occupancy_grid(self):
        self._mark_obstacle_poles()
        self._mark_gate_frames()

    def _world_to_grid_clipped(self, p):
        idx = np.round((np.asarray(p, dtype=float) - self.bounds_low) * self.inv_resolution).astype(int)
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

            near_gate_plane = np.abs(local[..., 0]) < plane_limit
            inside_outer_square = np.abs(local[..., 1]) < outer
            inside_outer_square &= np.abs(local[..., 2]) < outer
            inside_inner_opening = np.abs(local[..., 1]) < inner
            inside_inner_opening &= np.abs(local[..., 2]) < inner

            gate_mask = near_gate_plane & inside_outer_square & ~inside_inner_opening
            self.occupied[
                x_idx[:, None, None],
                y_idx[None, :, None],
                z_idx[None, None, :],
            ] |= gate_mask

    def _inside_inflated_pole(self, p):
        """
        Obstacles are vertical poles.
        Check XY distance and z range.
        """
        for obs_p in self.obstacles_pos:
            d_xy = np.linalg.norm(p[:2] - obs_p[:2])

            if d_xy < self.obstacle_radius + self.effective_obstacle_margin:
                # Poles are tall; treat them as vertical obstacles through useful flight region.
                return True

        return False

    def _inside_gate_frame(self, p):
        """
        Gate frame model:
            occupied if point lies in the inflated square frame volume,
            while the shrunken inner opening remains free.

        Gate local frame:
            x = normal direction
            y = lateral
            z = vertical relative to gate center
        """
        for gate_pos, gate_quat in zip(self.gates_pos, self.gates_quat):
            rot = R.from_quat(gate_quat)
            local = rot.inv().apply(p - gate_pos)

            x, y, z = local

            near_gate_plane = abs(x) < (
                self.gate_thickness / 2.0 + self.effective_gate_plane_margin
            )

            outer = self.gate_outer_half + self.effective_gate_outer_margin
            inner = max(0.0, self.gate_inner_half - self.effective_gate_inner_margin)

            inside_outer_square = abs(y) < outer and abs(z) < outer
            inside_inner_opening = abs(y) < inner and abs(z) < inner

            if near_gate_plane and inside_outer_square and not inside_inner_opening:
                return True

        return False