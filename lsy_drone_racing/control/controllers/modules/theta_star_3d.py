"""Theta* path planning on the existing 3D occupancy grid."""

import heapq
import math
from typing import Any

import numpy as np

from lsy_drone_racing.control.controllers.modules.initial_challenge.astar_3d import NEIGHBOR_STEPS_18, heuristic

GridIdx = tuple[int, int, int]


def theta_star_3d(
    grid: Any,
    start_idx: GridIdx,
    goal_idx: GridIdx,
    max_iterations: int = 200_000,
    heuristic_weight: float = 1.0,
) -> list[GridIdx] | None:
    """Plan a 3D any-angle path with Theta*.

    Theta* is an any-angle variant of A*: it uses the same occupancy grid, but
    connects a node directly to its grandparent when line of sight is free.
    This usually creates smoother, less stair-stepped paths than grid A*.
    """
    start_idx = tuple(start_idx)
    goal_idx = tuple(goal_idx)

    if not grid.is_free(start_idx):
        print(f"Theta*: start occupied {start_idx}, world={grid.grid_to_world(start_idx)}")
        return None

    if not grid.is_free(goal_idx):
        print(f"Theta*: goal occupied {goal_idx}, world={grid.grid_to_world(goal_idx)}")
        return None

    shape = tuple(grid.shape.tolist())
    occupied = grid.occupied

    g_score = np.full(shape, np.inf, dtype=float)
    closed = np.zeros(shape, dtype=bool)
    parents = np.full(shape + (3,), -1, dtype=np.int32)

    sx, sy, sz = start_idx
    g_score[sx, sy, sz] = 0.0
    parents[sx, sy, sz] = start_idx

    start_h = heuristic(start_idx, goal_idx)
    open_heap = [(heuristic_weight * start_h, start_h, sx, sy, sz)]

    iterations = 0

    while open_heap:
        iterations += 1
        if iterations > max_iterations:
            print("Theta*: exceeded max iterations")
            return None

        _, _, cx, cy, cz = heapq.heappop(open_heap)

        if closed[cx, cy, cz]:
            continue

        closed[cx, cy, cz] = True

        current = (cx, cy, cz)
        if current == goal_idx:
            return reconstruct_path(parents, start_idx, goal_idx)

        px, py, pz = parents[cx, cy, cz]
        parent = (int(px), int(py), int(pz))

        for dx, dy, dz, step_cost in NEIGHBOR_STEPS_18:
            nx = cx + dx
            ny = cy + dy
            nz = cz + dz

            if nx < 0 or ny < 0 or nz < 0:
                continue

            if nx >= shape[0] or ny >= shape[1] or nz >= shape[2]:
                continue

            if closed[nx, ny, nz] or occupied[nx, ny, nz]:
                continue

            neighbor = (nx, ny, nz)

            if parent != current and line_of_sight(grid, parent, neighbor):
                candidate_parent = parent
                tentative_g = g_score[parent] + idx_distance(parent, neighbor)
            else:
                candidate_parent = current
                tentative_g = g_score[current] + step_cost

            if tentative_g >= g_score[neighbor]:
                continue

            g_score[neighbor] = tentative_g
            parents[neighbor] = candidate_parent

            h_score = heuristic(neighbor, goal_idx)
            f_score = tentative_g + heuristic_weight * h_score
            heapq.heappush(open_heap, (f_score, h_score, nx, ny, nz))

    print("Theta*: failed to find path")
    return None


def line_of_sight(grid: Any, a_idx: GridIdx, b_idx: GridIdx) -> bool:
    """Return whether the straight segment between two grid cells is free."""
    a = grid.grid_to_world(a_idx)
    b = grid.grid_to_world(b_idx)

    dist = np.linalg.norm(b - a)
    if dist < 1e-9:
        return grid.is_free(tuple(a_idx))

    step = grid.resolution * 0.5
    n = max(2, int(np.ceil(dist / step)))

    for alpha in np.linspace(0.0, 1.0, n):
        p = (1.0 - alpha) * a + alpha * b
        if not grid.is_free(grid.world_to_grid(p)):
            return False

    return True


def idx_distance(a: GridIdx, b: GridIdx) -> float:
    """Return Euclidean distance in grid-index space."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def reconstruct_path(
    parents: np.ndarray, start_idx: GridIdx, goal_idx: GridIdx
) -> list[GridIdx] | None:
    """Reconstruct a grid-index path from a parent array."""
    current = goal_idx
    path = [current]

    while current != start_idx:
        px, py, pz = parents[current]

        if px < 0:
            return None

        current = (int(px), int(py), int(pz))
        path.append(current)

    path.reverse()
    return path
