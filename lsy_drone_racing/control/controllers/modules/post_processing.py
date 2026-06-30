"""Post-process an A* path to prepare it for timing / trajectory generation.

``PathPostProcessor`` refines the raw waypoints from the path generator in two
lean stages and returns, alongside the refined path, a per-point **clearance
tube** (distance to the nearest obstacle) for a later clearance/curvature-aware
timing module to exploit:

1. **Gravity / repulsion nudge** (soft): obstacle poles push interior waypoints
   into freer space -- nearer poles push harder (inverse-distance potential) --
   so the path carries more clearance and can be flown faster. The nudge is
   deliberately *gentle* and per-step capped: the RL tracker is sensitive to
   waypoint-shape changes and mistracks under aggressive nudging.
2. **Clearance floor** (hard, best-effort guarantee): any waypoint closer than
   ``min_clearance`` to a pole is pushed radially out to the safe distance. If the
   push cannot improve clearance (e.g. squeezed between two poles), the waypoint
   is left untouched -- *fail: do nothing*, never make it worse.

Gate-aperture waypoints and the path endpoints are locked so gate crossings stay
straight. Pole distances are analytic (poles are vertical cylinders), so no grid
is built -- the processor is lean and sub-millisecond at our path sizes.

The returned ``clearance`` array (free space beyond the inflated pole, per point)
is the "tube radius": large where the drone can open up, small in tight squeezes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import PchipInterpolator

if TYPE_CHECKING:
    from numpy.typing import NDArray


class PathPostProcessor:
    """Two-stage waypoint refiner: gentle pole repulsion + clearance floor + tube."""

    def __init__(
        self,
        resample_step: float = 0.10,
        pole_radius: float = 0.15,
        min_clearance: float = 0.18,
        influence: float = 0.6,
        repulse_gain: float = 0.03,
        max_step: float = 0.02,
        iterations: int = 3,
        gate_lock_radius: float = 0.30,
        max_tube: float = 0.8,
        enabled: bool = True,
    ):
        """Configure the post-processor.

        Args:
            resample_step: Densify the (sparse, pruned) A* path by fitting a PCHIP
                geometry spline and resampling at this arc-length step (m); 0
                disables. Dense waypoints give the clearance tube and the timing
                curvature a fine, per-10cm resolution. Runs regardless of ``enabled``.
            pole_radius: Effective obstacle radius (pole + drone half-width); the
                tube clearance is measured beyond this.
            min_clearance: Required free clearance beyond ``pole_radius`` (the hard
                floor); the safe pole-centre distance is ``pole_radius + min_clearance``.
            influence: Radius (m) within which a pole repels a waypoint.
            repulse_gain: Strength of the repulsion (keep small for the RL tracker).
            max_step: Per-iteration cap on a waypoint's displacement (m).
            iterations: Number of nudge sweeps (0 disables the nudge, floor only).
            gate_lock_radius: Waypoints within this of a gate centre are locked.
            max_tube: Upper cap on the stored clearance value.
            enabled: If False, the nudge/floor refinement is skipped; the path is
                still densified and the tube still returned.
        """
        self.resample_step = resample_step
        self.pole_radius = pole_radius
        self.min_clearance = min_clearance
        self.influence = influence
        self.repulse_gain = repulse_gain
        self.max_step = max_step
        self.iterations = iterations
        self.gate_lock_radius = gate_lock_radius
        self.max_tube = max_tube
        self.enabled = enabled

    @property
    def safe_distance(self) -> float:
        """Pole-centre distance the clearance floor pushes violating points to."""
        return self.pole_radius + self.min_clearance

    def process(
        self, path: NDArray[np.floating], obs: dict, config: object = None
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Refine ``path`` and return ``(refined_path, clearance)``.

        Args:
            path: ``(M, 3)`` waypoints from the path generator.
            obs: Observation dict (uses ``obstacles_pos`` and ``gates_pos``).
            config: Unused; kept for a uniform module signature.

        Returns:
            ``(path, clearance)`` -- the refined ``(M, 3)`` waypoints and the
            ``(M,)`` per-point clearance tube (free space beyond ``pole_radius``).
        """
        path = np.asarray(path, dtype=float).copy()
        if self.resample_step > 0.0:
            path = self._densify(path)
        poles = np.asarray(obs.get("obstacles_pos", np.empty((0, 3))), dtype=float)
        gates = np.asarray(obs.get("gates_pos", np.empty((0, 3))), dtype=float)

        if not self.enabled or len(path) < 3 or poles.size == 0:
            return path, self._clearance(path, poles)

        movable = self._movable_mask(path, gates)
        for _ in range(self.iterations):
            path = self._nudge(path, poles, movable)
        path = self._enforce_floor(path, poles, movable)
        return path, self._clearance(path, poles)

    # ------------------------------------------------------------------ helpers
    def _densify(self, path: NDArray[np.floating]) -> NDArray[np.floating]:
        """Resample the path at ~``resample_step`` via a PCHIP geometry spline.

        PCHIP is shape-preserving (no overshoot past the pruned vertices), so the
        dense points lie on the same smooth curve the trajectory will follow -- we
        just sample it finely (per ~10 cm) so the clearance tube and timing
        curvature are no longer blind between sparse waypoints.
        """
        path = np.asarray(path, dtype=float)
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        pts = path[np.concatenate([[True], seg > 1e-9])]  # drop coincident points
        if len(pts) < 2:
            return path

        u = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        n = max(2, int(np.ceil(u[-1] / self.resample_step)) + 1)
        return PchipInterpolator(u, pts, axis=0)(np.linspace(0.0, u[-1], n))

    def _movable_mask(
        self, path: NDArray[np.floating], gates: NDArray[np.floating]
    ) -> NDArray[np.bool_]:
        """Interior waypoints that are not locked near a gate (or the endpoints)."""
        movable = np.ones(len(path), dtype=bool)
        movable[0] = movable[-1] = False
        if gates.size:
            d = np.linalg.norm(path[:, None, :] - gates[None, :, :], axis=2)  # (M, G)
            movable &= np.all(d >= self.gate_lock_radius, axis=1)
        return movable

    def _nudge(
        self, path: NDArray[np.floating], poles: NDArray[np.floating], movable: NDArray[np.bool_]
    ) -> NDArray[np.floating]:
        """One sweep of capped, gentle inverse-distance pole repulsion (XY only)."""
        p = path[:, :2]
        diff = p[:, None, :] - poles[None, :, :2]  # (M, P, 2)
        dist = np.linalg.norm(diff, axis=2)  # (M, P)
        active = (dist < self.influence) & (dist > 1e-6)
        # Repulsion potential gradient ~ (1/d - 1/influence), zero at the influence radius.
        inv = 1.0 / np.maximum(dist, 1e-6) - 1.0 / self.influence
        weight = np.where(active, self.repulse_gain * inv, 0.0)
        dirs = diff / np.maximum(dist, 1e-9)[..., None]
        force = np.sum(weight[..., None] * dirs, axis=1)  # (M, 2)

        norm = np.linalg.norm(force, axis=1)
        scale = np.where(norm > self.max_step, self.max_step / np.maximum(norm, 1e-9), 1.0)
        force *= scale[:, None]
        force[~movable] = 0.0

        path = path.copy()
        path[:, :2] += force
        return path

    def _enforce_floor(
        self, path: NDArray[np.floating], poles: NDArray[np.floating], movable: NDArray[np.bool_]
    ) -> NDArray[np.floating]:
        """Push under-clearance waypoints radially out; skip if it cannot improve."""
        path = path.copy()
        idx = np.where(movable)[0]
        for i in idx:
            d, nearest = self._nearest_pole(path[i, :2], poles)
            if d >= self.safe_distance:
                continue
            direction = path[i, :2] - nearest
            n = np.linalg.norm(direction)
            if n < 1e-6:
                continue  # exactly on a pole axis: no well-defined push direction
            candidate = nearest + (direction / n) * self.safe_distance
            new_d, _ = self._nearest_pole(candidate, poles)
            if new_d > d:  # only move when it genuinely improves clearance
                path[i, :2] = candidate
        return path

    def _nearest_pole(
        self, p_xy: NDArray[np.floating], poles: NDArray[np.floating]
    ) -> tuple[float, NDArray[np.floating]]:
        """Nearest pole (XY) to a point: returns (centre distance, pole XY)."""
        d = np.linalg.norm(poles[:, :2] - p_xy, axis=1)
        j = int(np.argmin(d))
        return float(d[j]), poles[j, :2]

    def _clearance(
        self, path: NDArray[np.floating], poles: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Per-point free clearance beyond the inflated pole (the tube radius)."""
        if poles.size == 0:
            return np.full(len(path), self.max_tube)
        d = np.linalg.norm(path[:, None, :2] - poles[None, :, :2], axis=2)
        return np.clip(d.min(axis=1) - self.pole_radius, 0.0, self.max_tube)
