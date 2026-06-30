"""Tier-0 optimized 3D A* kernel.

Standalone, dependency-free optimization of ``astar_3d.astar_3d``. The original
module is left untouched for comparison.

What changed vs. the original (none of these alter the search result for the
full-grid / no-mask case; see the determinism note in
``path_generator_improved``):

1. No per-edge numpy allocation in the inner loop. The 18 move directions carry
   precomputed scalar unit components, so the velocity-bias dot product uses
   plain floats instead of allocating an array and calling ``numpy.dot``.

2. Flat (1-D) cell indexing. ``g_score`` / ``closed`` are flat arrays indexed by
   a single integer ``flat = x*Sy*Sz + y*Sz + z``, which is faster to read/write
   than 3-D numpy indexing. Neighbor indices are derived incrementally
   (``nflat = cflat + offset``). The parent map stores a single integer flat
   index per cell instead of three.

3. Buffer reuse. The ``g_score`` / ``closed`` / ``parent`` arrays live on the
   solver and are reused across calls (reset with ``fill``), so repeated replans
   on a fixed grid avoid re-allocating ~1 MB per call.

Optional search window (an inclusive index box) bounds exploration, and an
optional full-grid boolean ``allowed_mask`` restricts the search to a corridor
(used by coarse-to-fine planning). The 18-connectivity, weighted heuristic, and
path-history-dependent velocity bias are preserved exactly.
"""

import heapq
import math

import numpy as np

# (dx, dy, dz, step_cost, ux, uy, uz) with (ux, uy, uz) the unit move direction.
# Precomputing the unit components removes the per-edge numpy allocation the
# original kernel performed inside the neighbor loop.
NEIGHBOR_STEPS_18 = tuple(
    (
        dx,
        dy,
        dz,
        math.sqrt(dx * dx + dy * dy + dz * dz),
        dx / math.sqrt(dx * dx + dy * dy + dz * dz),
        dy / math.sqrt(dx * dx + dy * dy + dz * dz),
        dz / math.sqrt(dx * dx + dy * dy + dz * dz),
    )
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0) and (int(dx != 0) + int(dy != 0) + int(dz != 0) <= 2)
)


def _heuristic(ax, ay, az, bx, by, bz):
    dx = ax - bx
    dy = ay - by
    dz = az - bz
    return math.sqrt(dx * dx + dy * dy + dz * dz)


class AStar3DImproved:
    """18-connected 3D A* with flat indexing and reusable scratch buffers.

    One instance should be reused across replans on a fixed-resolution grid so
    that the bookkeeping arrays are allocated once. Separate instances should be
    used for grids of different shapes (e.g. fine vs. coarse), to avoid
    reallocating the buffers back and forth.
    """

    def __init__(self):
        self._n = -1
        self._g = None
        self._closed = None
        self._parent = None

    def _ensure(self, n):
        if self._n != n:
            self._g = np.empty(n, dtype=float)
            self._closed = np.empty(n, dtype=bool)
            self._parent = np.empty(n, dtype=np.int32)
            self._n = n

    def plan(
        self,
        grid,
        start_idx,
        goal_idx,
        *,
        win_min=None,
        win_max=None,
        allowed_mask=None,
        cost_field=None,
        max_iterations=200_000,
        heuristic_weight=1.0,
        preferred_direction=None,
        direction_bias_weight=1.0,
        direction_bias_decay=8.0,
        min_direction_speed=0.10,
    ):
        """Plan a path. See module docstring. Returns a list of (x, y, z) grid
        indices, or None on failure.

        Args:
            grid: OccupancyGrid3D.
            start_idx: tuple[int, int, int] grid index (must be free).
            goal_idx: tuple[int, int, int] grid index (must be free).
            win_min: Inclusive lower index bound of the search window, or None
                for the full grid. ``start_idx`` and ``goal_idx`` must lie inside.
            win_max: Inclusive upper index bound of the search window, or None.
            allowed_mask: Optional full-grid boolean array; a neighbor is
                searchable only where it is True. Used for coarse-to-fine
                corridors.
            cost_field: Optional full-grid float array of an extra cost *rate*
                per cell (cost per unit path length). The cost added to enter a
                neighbor is ``step_cost * cost_field[neighbor]``. Used for
                clearance / risk shaping (penalize cells near obstacles). The
                Euclidean heuristic stays admissible as long as the field is
                non-negative.
            max_iterations: Iteration cap before giving up.
            heuristic_weight: Weight on the Euclidean heuristic.
            preferred_direction: Optional 3D direction to bias early expansion.
            direction_bias_weight: Soft penalty weight for misaligned moves.
            direction_bias_decay: Exponential decay length (path-cost units).
            min_direction_speed: Minimum norm to activate ``preferred_direction``.
        """
        start_idx = tuple(int(v) for v in start_idx)
        goal_idx = tuple(int(v) for v in goal_idx)

        if not grid.is_free(start_idx):
            print(f"A*(improved): start occupied {start_idx}, world={grid.grid_to_world(start_idx)}")
            return None
        if not grid.is_free(goal_idx):
            print(f"A*(improved): goal occupied {goal_idx}, world={grid.grid_to_world(goal_idx)}")
            return None

        shape = tuple(int(v) for v in grid.shape.tolist())
        sx_dim, sy_dim, sz_dim = shape
        sxstep = sy_dim * sz_dim  # flat stride along x
        systep = sz_dim  # flat stride along y
        n = sx_dim * sxstep

        self._ensure(n)
        g = self._g
        closed = self._closed
        parent = self._parent
        g.fill(np.inf)
        closed.fill(False)

        # Flat views of the (C-contiguous) occupancy, corridor mask, cost field.
        occ_flat = grid.occupied.reshape(-1)
        mask_flat = None if allowed_mask is None else np.asarray(allowed_mask).reshape(-1)
        cost_flat = None if cost_field is None else np.asarray(cost_field).reshape(-1)

        # Search window (inclusive). Defaults to the full grid.
        if win_min is None:
            win_min = (0, 0, 0)
        if win_max is None:
            win_max = (sx_dim - 1, sy_dim - 1, sz_dim - 1)
        ox, oy, oz = int(win_min[0]), int(win_min[1]), int(win_min[2])
        hx, hy, hz = int(win_max[0]), int(win_max[1]), int(win_max[2])

        stx, sty, stz = start_idx
        gtx, gty, gtz = goal_idx
        if not (ox <= stx <= hx and oy <= sty <= hy and oz <= stz <= hz):
            print("A*(improved): start outside window")
            return None
        if not (ox <= gtx <= hx and oy <= gty <= hy and oz <= gtz <= hz):
            print("A*(improved): goal outside window")
            return None

        # Neighbor flat offsets depend on the grid strides, so build them here.
        steps = [
            (dx, dy, dz, cost, ux, uy, uz, dx * sxstep + dy * systep + dz)
            for (dx, dy, dz, cost, ux, uy, uz) in NEIGHBOR_STEPS_18
        ]

        # Preferred direction as scalar unit components (avoids numpy in loop).
        use_bias = False
        pdx = pdy = pdz = 0.0
        if preferred_direction is not None and direction_bias_weight > 0.0:
            pd = np.asarray(preferred_direction, dtype=float).reshape(-1)
            if pd.size >= 3:
                pd = pd[:3]
                pnorm = float(np.linalg.norm(pd))
                if pnorm >= min_direction_speed:
                    pdx, pdy, pdz = pd[0] / pnorm, pd[1] / pnorm, pd[2] / pnorm
                    use_bias = True
        decay_len = max(float(direction_bias_decay), 1e-6)

        start_flat = stx * sxstep + sty * systep + stz
        goal_flat = gtx * sxstep + gty * systep + gtz
        g[start_flat] = 0.0

        start_h = _heuristic(stx, sty, stz, gtx, gty, gtz)
        open_heap = [(heuristic_weight * start_h, start_h, stx, sty, stz)]

        iterations = 0
        while open_heap:
            iterations += 1
            if iterations > max_iterations:
                print("A*(improved): exceeded max iterations")
                return None

            _, _, cx, cy, cz = heapq.heappop(open_heap)
            cflat = cx * sxstep + cy * systep + cz

            if closed[cflat]:
                continue
            closed[cflat] = True

            if cx == gtx and cy == gty and cz == gtz:
                return _reconstruct(parent, start_flat, goal_flat, sxstep, systep)

            current_g = g[cflat]

            for dx, dy, dz, step_cost, ux, uy, uz, off in steps:
                nx = cx + dx
                ny = cy + dy
                nz = cz + dz

                # Stay inside the search window.
                if nx < ox or ny < oy or nz < oz or nx > hx or ny > hy or nz > hz:
                    continue

                nflat = cflat + off

                if closed[nflat] or occ_flat[nflat]:
                    continue
                if mask_flat is not None and not mask_flat[nflat]:
                    continue

                directional_penalty = 0.0
                if use_bias:
                    alignment = ux * pdx + uy * pdy + uz * pdz
                    misalignment = 0.5 * (1.0 - alignment)
                    decay = math.exp(-current_g / decay_len)
                    directional_penalty = direction_bias_weight * misalignment * decay

                tentative_g = current_g + step_cost + directional_penalty
                if cost_flat is not None:
                    tentative_g += step_cost * cost_flat[nflat]

                if tentative_g >= g[nflat]:
                    continue

                g[nflat] = tentative_g
                parent[nflat] = cflat

                h_score = _heuristic(nx, ny, nz, gtx, gty, gtz)
                f_score = tentative_g + heuristic_weight * h_score
                heapq.heappush(open_heap, (f_score, h_score, nx, ny, nz))

        print("A*(improved): failed to find path")
        return None


def _reconstruct(parent, start_flat, goal_flat, sxstep, systep):
    """Walk single-int flat parents from goal to start, decoding to (x, y, z)."""
    path = []
    cur = goal_flat
    while cur != start_flat:
        x = cur // sxstep
        rem = cur - x * sxstep
        y = rem // systep
        z = rem - y * systep
        path.append((int(x), int(y), int(z)))
        p = int(parent[cur])
        if p < 0:
            return None
        cur = p

    x = start_flat // sxstep
    rem = start_flat - x * sxstep
    y = rem // systep
    z = rem - y * systep
    path.append((int(x), int(y), int(z)))
    path.reverse()
    return path


# Module-level convenience wrapper around a shared solver (back-compatible with
# the previous functional API). Callers that replan repeatedly should hold their
# own AStar3DImproved instance instead, to get buffer reuse without size thrash.
_default_solver = AStar3DImproved()


def astar_3d_improved(grid, start_idx, goal_idx, **kwargs):
    """Plan with a shared default solver. See AStar3DImproved.plan."""
    return _default_solver.plan(grid, start_idx, goal_idx, **kwargs)
