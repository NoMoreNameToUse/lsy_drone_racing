"""D* Lite path planning on the existing 3D occupancy grid."""

from __future__ import annotations

import heapq
from typing import Any

import numpy as np

from lsy_drone_racing.control.controllers.modules.zihan.astar_3d import NEIGHBOR_STEPS_18, heuristic

GridIdx = tuple[int, int, int]


def d_star_lite_3d(
    grid: Any,
    start_idx: GridIdx,
    goal_idx: GridIdx,
    max_iterations: int = 200_000,
    heuristic_weight: float = 1.0,
    search_bounds: tuple[GridIdx, GridIdx] | None = None,
) -> list[GridIdx] | None:
    """Plan a static-grid path with the D* Lite update equations.

    This implementation uses D* Lite's reverse-search value updates on the
    current occupancy grid. It does not yet persist state across replans, but
    keeps the algorithm isolated so later incremental replanning can build on it.
    """
    start_idx = tuple(start_idx)
    goal_idx = tuple(goal_idx)

    if not grid.is_free(start_idx):
        print(f"D* Lite: start occupied {start_idx}, world={grid.grid_to_world(start_idx)}")
        return None

    if not grid.is_free(goal_idx):
        print(f"D* Lite: goal occupied {goal_idx}, world={grid.grid_to_world(goal_idx)}")
        return None

    shape = tuple(grid.shape.tolist())
    g = np.full(shape, np.inf, dtype=float)
    rhs = np.full(shape, np.inf, dtype=float)
    rhs[goal_idx] = 0.0

    open_heap: list[tuple[float, float, int, int, int]] = []
    push(open_heap, goal_idx, calculate_key(goal_idx, start_idx, g, rhs, heuristic_weight))

    iterations = 0
    while open_heap:
        iterations += 1
        if iterations > max_iterations:
            print("D* Lite: exceeded max iterations")
            return None

        top_key = open_heap[0][:2]
        start_key = calculate_key(start_idx, start_idx, g, rhs, heuristic_weight)
        if not key_less(top_key, start_key) and rhs[start_idx] == g[start_idx]:
            break

        k_old, u = pop(open_heap)
        k_new = calculate_key(u, start_idx, g, rhs, heuristic_weight)

        # Lazy priority queues can contain old entries for nodes that have
        # already become consistent. Skip those instead of invalidating them.
        if g[u] == rhs[u]:
            continue

        if key_less(k_old, k_new):
            push(open_heap, u, k_new)
            continue

        if g[u] > rhs[u]:
            g[u] = rhs[u]
            for pred, _ in neighbors(grid, u, search_bounds):
                update_vertex(
                    grid,
                    pred,
                    goal_idx,
                    start_idx,
                    g,
                    rhs,
                    open_heap,
                    heuristic_weight,
                    search_bounds,
                )
        else:
            g[u] = np.inf
            update_vertex(
                grid, u, goal_idx, start_idx, g, rhs, open_heap, heuristic_weight, search_bounds
            )
            for pred, _ in neighbors(grid, u, search_bounds):
                update_vertex(
                    grid,
                    pred,
                    goal_idx,
                    start_idx,
                    g,
                    rhs,
                    open_heap,
                    heuristic_weight,
                    search_bounds,
                )

    if not np.isfinite(g[start_idx]) and not np.isfinite(rhs[start_idx]):
        print("D* Lite: failed to find path")
        return None

    return reconstruct_path(grid, start_idx, goal_idx, g, search_bounds)


def calculate_key(
    idx: GridIdx,
    start_idx: GridIdx,
    g: np.ndarray,
    rhs: np.ndarray,
    heuristic_weight: float,
) -> tuple[float, float]:
    """Return D* Lite priority key for a grid cell."""
    value = min(g[idx], rhs[idx])
    return value + heuristic_weight * heuristic(idx, start_idx), value


def update_vertex(
    grid: Any,
    idx: GridIdx,
    goal_idx: GridIdx,
    start_idx: GridIdx,
    g: np.ndarray,
    rhs: np.ndarray,
    open_heap: list[tuple[float, float, int, int, int]],
    heuristic_weight: float,
    search_bounds: tuple[GridIdx, GridIdx] | None,
) -> None:
    """Update one vertex rhs value and lazily push inconsistent states."""
    if idx != goal_idx:
        best = np.inf
        for succ, step_cost in neighbors(grid, idx, search_bounds):
            best = min(best, step_cost + g[succ])
        rhs[idx] = best

    if g[idx] != rhs[idx]:
        push(open_heap, idx, calculate_key(idx, start_idx, g, rhs, heuristic_weight))


def neighbors(
    grid: Any, idx: GridIdx, search_bounds: tuple[GridIdx, GridIdx] | None = None
) -> list[tuple[GridIdx, float]]:
    """Return free 18-connected neighbors and their step costs."""
    ix, iy, iz = idx
    result = []
    for dx, dy, dz, step_cost in NEIGHBOR_STEPS_18:
        nidx = (ix + dx, iy + dy, iz + dz)
        if search_bounds is not None and not in_search_bounds(nidx, search_bounds):
            continue
        if grid.is_free(nidx):
            result.append((nidx, step_cost))
    return result


def reconstruct_path(
    grid: Any,
    start_idx: GridIdx,
    goal_idx: GridIdx,
    g: np.ndarray,
    search_bounds: tuple[GridIdx, GridIdx] | None = None,
) -> list[GridIdx] | None:
    """Follow the lowest cost-to-go values from start to goal."""
    current = start_idx
    path = [current]
    visited = {current}
    max_steps = int(np.sum(grid.shape)) * 4

    for _ in range(max_steps):
        if current == goal_idx:
            return path

        best_idx = None
        best_cost = np.inf
        for succ, step_cost in neighbors(grid, current, search_bounds):
            cost = step_cost + g[succ]
            if cost < best_cost:
                best_cost = cost
                best_idx = succ

        if best_idx is None or not np.isfinite(best_cost) or best_idx in visited:
            print("D* Lite: failed to reconstruct path")
            return None

        current = best_idx
        visited.add(current)
        path.append(current)

    print("D* Lite: path reconstruction exceeded max steps")
    return None


def push(
    open_heap: list[tuple[float, float, int, int, int]], idx: GridIdx, key: tuple[float, float]
) -> None:
    """Push a keyed grid cell onto the lazy priority queue."""
    heapq.heappush(open_heap, (key[0], key[1], idx[0], idx[1], idx[2]))


def pop(open_heap: list[tuple[float, float, int, int, int]]) -> tuple[tuple[float, float], GridIdx]:
    """Pop the next keyed grid cell from the priority queue."""
    k1, k2, ix, iy, iz = heapq.heappop(open_heap)
    return (k1, k2), (ix, iy, iz)


def key_less(a: tuple[float, float], b: tuple[float, float]) -> bool:
    """Return lexicographic D* Lite key ordering."""
    return a[0] < b[0] or (a[0] == b[0] and a[1] < b[1])


def in_search_bounds(idx: GridIdx, search_bounds: tuple[GridIdx, GridIdx]) -> bool:
    """Return whether idx is inside inclusive local search bounds."""
    low, high = search_bounds
    return all(low[i] <= idx[i] <= high[i] for i in range(3))
