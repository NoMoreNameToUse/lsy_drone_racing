"""Minimal numba-JIT 3D grid A* used for fast replanning.

A custom implementation is used because off-the-shelf planners were either too
slow or did not expose the internal grid data the pipeline needs; this one runs
consistently under ~10 ms, small enough to replan within the 50 Hz control loop.
It is a fast grid A* designed for repeated replanning on a fixed-shape grid, so it
preallocates scratch buffers and neighbor tables:

- flat (1-D) cell indexing with precomputed neighbor offsets,
- a hand-rolled binary heap on plain arrays (numba has no heapq),
- reusable scratch buffers (reset in-kernel, O(n) but compiled),
- 26-connectivity with Euclidean step costs and an Euclidean heuristic
  (optimal at ``heuristic_weight == 1.0``; weighted A* for speed when > 1.0).

It operates directly on a boolean occupancy array indexed by integer grid
coordinates, so it has no dependency on the rest of the planner. The first call
triggers a one-time numba compile (~0.5 s, then disk-cached).
"""

import math

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _astar_kernel(
    occ: np.ndarray,  # (n,) bool flattened occupancy (True = blocked)
    sx: int,
    sy: int,
    sz: int,  # grid dims
    sxstep: int,
    systep: int,  # flat strides: flat = x*sxstep + y*systep + z
    si: int,
    gi: int,  # flat start / goal indices
    hw: float,  # heuristic weight
    g: np.ndarray,
    closed: np.ndarray,
    parent: np.ndarray,  # (n,) scratch buffers
    heap_f: np.ndarray,
    heap_id: np.ndarray,  # (cap,) binary-heap arrays
    ddx: np.ndarray,
    ddy: np.ndarray,
    ddz: np.ndarray,
    dcost: np.ndarray,
    doff: np.ndarray,  # (ndir,) neighbor tables
) -> int:
    """Run A*; fill ``parent``; return 1 if goal reached, 0 if not, -1 on heap overflow."""
    n = g.shape[0]
    for k in range(n):
        g[k] = 1e18
        closed[k] = False

    gx = gi // sxstep
    rg = gi - gx * sxstep
    gy = rg // systep
    gz = rg - gy * systep

    sxx = si // sxstep
    rs = si - sxx * sxstep
    syy = rs // systep
    szz = rs - syy * systep

    g[si] = 0.0
    h0 = math.sqrt(float((sxx - gx) ** 2 + (syy - gy) ** 2 + (szz - gz) ** 2))
    heap_f[0] = hw * h0
    heap_id[0] = si
    hsize = 1
    cap = heap_f.shape[0]
    ndir = doff.shape[0]

    while hsize > 0:
        # pop min
        ci = heap_id[0]
        hsize -= 1
        heap_f[0] = heap_f[hsize]
        heap_id[0] = heap_id[hsize]
        i = 0
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            sm = i
            if left < hsize and heap_f[left] < heap_f[sm]:
                sm = left
            if right < hsize and heap_f[right] < heap_f[sm]:
                sm = right
            if sm == i:
                break
            tf = heap_f[i]
            heap_f[i] = heap_f[sm]
            heap_f[sm] = tf
            ti = heap_id[i]
            heap_id[i] = heap_id[sm]
            heap_id[sm] = ti
            i = sm

        if closed[ci]:
            continue
        closed[ci] = True
        if ci == gi:
            return 1

        cg = g[ci]
        cx = ci // sxstep
        rc = ci - cx * sxstep
        cy = rc // systep
        cz = rc - cy * systep

        for d in range(ndir):
            nx = cx + ddx[d]
            ny = cy + ddy[d]
            nz = cz + ddz[d]
            if nx < 0 or ny < 0 or nz < 0 or nx >= sx or ny >= sy or nz >= sz:
                continue
            ni = ci + doff[d]
            if closed[ni] or occ[ni]:
                continue
            ng = cg + dcost[d]
            if ng < g[ni]:
                g[ni] = ng
                parent[ni] = ci
                hh = math.sqrt(float((nx - gx) ** 2 + (ny - gy) ** 2 + (nz - gz) ** 2))
                f = ng + hw * hh
                if hsize >= cap:
                    return -1
                j = hsize
                heap_f[j] = f
                heap_id[j] = ni
                hsize += 1
                while j > 0:
                    p = (j - 1) // 2
                    if heap_f[p] <= heap_f[j]:
                        break
                    tf = heap_f[p]
                    heap_f[p] = heap_f[j]
                    heap_f[j] = tf
                    ti = heap_id[p]
                    heap_id[p] = heap_id[j]
                    heap_id[j] = ti
                    j = p

    return 0


class AStar3DBarebone:
    """Reusable 3D grid A* solver that caches scratch buffers across replans."""

    def __init__(self):
        """Initialize with empty buffers; they are (re)allocated on the first plan."""
        self._n = -1
        self._shape = None

    def _ensure(self, shape: tuple[int, int, int]) -> None:
        """(Re)allocate scratch buffers and neighbor tables for a grid of ``shape``."""
        sx, sy, sz = shape
        n = sx * sy * sz
        if self._n != n:
            self._g = np.empty(n, dtype=np.float64)
            self._closed = np.empty(n, dtype=np.bool_)
            self._parent = np.empty(n, dtype=np.int64)
            self._heap_f = np.empty(n, dtype=np.float64)
            self._heap_id = np.empty(n, dtype=np.int64)
            self._n = n
        if self._shape != shape:
            self._build_neighbors(sy, sz)
            self._shape = shape

    def _build_neighbors(self, sy: int, sz: int) -> None:
        """Precompute the 26-neighbor offsets, step costs and flat strides."""
        self._sxstep = sy * sz
        self._systep = sz
        ddx, ddy, ddz, dcost, doff = [], [], [], [], []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    ddx.append(dx)
                    ddy.append(dy)
                    ddz.append(dz)
                    dcost.append(math.sqrt(dx * dx + dy * dy + dz * dz))
                    doff.append(dx * self._sxstep + dy * self._systep + dz)
        self._ddx = np.array(ddx, dtype=np.int64)
        self._ddy = np.array(ddy, dtype=np.int64)
        self._ddz = np.array(ddz, dtype=np.int64)
        self._dcost = np.array(dcost, dtype=np.float64)
        self._doff = np.array(doff, dtype=np.int64)

    def plan(
        self,
        occupied: np.ndarray,
        start: tuple[int, int, int],
        goal: tuple[int, int, int],
        heuristic_weight: float = 1.0,
    ) -> list[tuple[int, int, int]] | None:
        """Plan a path on a boolean occupancy grid.

        Args:
            occupied: (sx, sy, sz) boolean array (True = blocked).
            start: (ix, iy, iz) integer grid index (must be free).
            goal: (ix, iy, iz) integer grid index (must be free).
            heuristic_weight: 1.0 for optimal; > 1.0 for faster, slightly longer.

        Returns:
            List of (ix, iy, iz) grid indices start..goal, or None.
        """
        occ = np.ascontiguousarray(occupied, dtype=np.bool_)
        self._ensure(occ.shape)
        sx, sy, sz = occ.shape

        si = int(start[0]) * self._sxstep + int(start[1]) * self._systep + int(start[2])
        gi = int(goal[0]) * self._sxstep + int(goal[1]) * self._systep + int(goal[2])

        occ_flat = occ.reshape(-1)
        if occ_flat[si] or occ_flat[gi]:
            return None

        result = _astar_kernel(
            occ_flat,
            sx,
            sy,
            sz,
            self._sxstep,
            self._systep,
            si,
            gi,
            float(heuristic_weight),
            self._g,
            self._closed,
            self._parent,
            self._heap_f,
            self._heap_id,
            self._ddx,
            self._ddy,
            self._ddz,
            self._dcost,
            self._doff,
        )
        if result != 1:
            return None
        return self._trace(si, gi)

    def _trace(self, si: int, gi: int) -> list[tuple[int, int, int]] | None:
        """Reconstruct the start..goal index path from the ``parent`` buffer."""
        parent = self._parent
        sxstep = self._sxstep
        systep = self._systep
        path = []
        cur = gi
        while cur != si:
            x = cur // sxstep
            r = cur - x * sxstep
            y = r // systep
            z = r - y * systep
            path.append((int(x), int(y), int(z)))
            cur = int(parent[cur])
            if cur < 0:
                return None
        x = si // sxstep
        r = si - x * sxstep
        y = r // systep
        z = r - y * systep
        path.append((int(x), int(y), int(z)))
        path.reverse()
        return path


_solver = AStar3DBarebone()


def astar_3d_barebone(
    occupied: np.ndarray,
    start: tuple[int, int, int],
    goal: tuple[int, int, int],
    heuristic_weight: float = 1.0,
) -> list[tuple[int, int, int]] | None:
    """Plan with a shared module-level solver. See ``AStar3DBarebone.plan``."""
    return _solver.plan(occupied, start, goal, heuristic_weight)
