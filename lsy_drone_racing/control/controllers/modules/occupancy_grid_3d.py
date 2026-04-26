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
        gate_inner_margin: float = 0.1
        gate_plane_margin: float = 0.25

        self.gate_outer_margin = gate_outer_margin
        self.gate_inner_margin = gate_inner_margin
        self.gate_plane_margin = gate_plane_margin

        self.bounds_low, self.bounds_high = self._get_bounds(config)

        self.shape = np.ceil(
            (self.bounds_high - self.bounds_low) / self.resolution
        ).astype(int) + 1

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
        idx = np.round((p - self.bounds_low) / self.resolution).astype(int)
        return tuple(idx.tolist())

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

        p = self.grid_to_world(idx)

        if p[2] < self.z_min or p[2] > self.z_max:
            return False

        if self._inside_inflated_pole(p):
            return False

        if self._inside_gate_frame(p):
            return False

        return True

    def _inside_inflated_pole(self, p):
        """
        Obstacles are vertical poles.
        Check XY distance and z range.
        """
        for obs_p in self.obstacles_pos:
            d_xy = np.linalg.norm(p[:2] - obs_p[:2])

            if d_xy < self.obstacle_radius + self.safety_margin:
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
                self.gate_thickness / 2.0 + self.gate_plane_margin
            )

            outer = self.gate_outer_half + self.gate_outer_margin
            inner = max(0.0, self.gate_inner_half - self.gate_inner_margin)

            inside_outer_square = abs(y) < outer and abs(z) < outer
            inside_inner_opening = abs(y) < inner and abs(z) < inner

            if near_gate_plane and inside_outer_square and not inside_inner_opening:
                return True

        return False