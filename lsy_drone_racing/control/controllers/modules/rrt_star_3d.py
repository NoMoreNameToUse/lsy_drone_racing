"""RRT* path planning on the existing 3D occupancy grid."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class RRTNode:
    """One node in the RRT* search tree."""

    point: np.ndarray
    parent: int
    cost: float


def rrt_star_3d(
    grid: Any,
    start: np.ndarray,
    goal: np.ndarray,
    step_size: float = 0.12,
    goal_sample_rate: float = 0.25,
    max_iterations: int = 800,
    search_radius: float = 0.35,
    early_exit: bool = True,
    rng: np.random.Generator | None = None,
) -> np.ndarray | None:
    """Plan a continuous-space RRT* path using OccupancyGrid3D collision checks."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    rng = np.random.default_rng() if rng is None else rng

    if not is_point_free(grid, start):
        print(f"RRT*: start occupied, world={start}")
        return None

    if not is_point_free(grid, goal):
        print(f"RRT*: goal occupied, world={goal}")
        return None

    if collision_free(grid, start, goal):
        return np.vstack([start, goal])

    nodes = [RRTNode(point=start, parent=-1, cost=0.0)]
    best_goal_idx = None

    for _ in range(max_iterations):
        sample = goal if rng.random() < goal_sample_rate else sample_free(grid, rng)

        nearest_idx = nearest_node(nodes, sample)
        new_point = steer(nodes[nearest_idx].point, sample, step_size)

        if not is_point_free(grid, new_point):
            continue

        if not collision_free(grid, nodes[nearest_idx].point, new_point):
            continue

        near_indices = near_nodes(nodes, new_point, search_radius)
        parent_idx = nearest_idx
        best_cost = nodes[nearest_idx].cost + distance(nodes[nearest_idx].point, new_point)

        for idx in near_indices:
            candidate_cost = nodes[idx].cost + distance(nodes[idx].point, new_point)
            if candidate_cost >= best_cost:
                continue

            if collision_free(grid, nodes[idx].point, new_point):
                parent_idx = idx
                best_cost = candidate_cost

        nodes.append(RRTNode(point=new_point, parent=parent_idx, cost=best_cost))
        new_idx = len(nodes) - 1

        for idx in near_indices:
            rewired_cost = best_cost + distance(new_point, nodes[idx].point)
            if rewired_cost >= nodes[idx].cost:
                continue

            if collision_free(grid, new_point, nodes[idx].point):
                nodes[idx].parent = new_idx
                nodes[idx].cost = rewired_cost

        if distance(new_point, goal) <= step_size and collision_free(grid, new_point, goal):
            goal_cost = best_cost + distance(new_point, goal)
            nodes.append(RRTNode(point=goal, parent=new_idx, cost=goal_cost))
            goal_idx = len(nodes) - 1

            if best_goal_idx is None or nodes[goal_idx].cost < nodes[best_goal_idx].cost:
                best_goal_idx = goal_idx

            if early_exit:
                return reconstruct_path(nodes, goal_idx)

    if best_goal_idx is None:
        print("RRT*: failed to find path")
        return None

    return reconstruct_path(nodes, best_goal_idx)


def sample_free(grid: Any, rng: np.random.Generator) -> np.ndarray:
    """Sample a free world-frame point from the grid bounds."""
    for _ in range(200):
        point = rng.uniform(grid.bounds_low, grid.bounds_high)
        if is_point_free(grid, point):
            return point

    return rng.uniform(grid.bounds_low, grid.bounds_high)


def nearest_node(nodes: list[RRTNode], point: np.ndarray) -> int:
    """Return the index of the nearest tree node."""
    distances = [distance(node.point, point) for node in nodes]
    return int(np.argmin(distances))


def near_nodes(nodes: list[RRTNode], point: np.ndarray, radius: float) -> list[int]:
    """Return tree node indices within the rewiring radius."""
    return [idx for idx, node in enumerate(nodes) if distance(node.point, point) <= radius]


def steer(start: np.ndarray, target: np.ndarray, step_size: float) -> np.ndarray:
    """Move from start toward target by at most step_size."""
    delta = target - start
    dist = np.linalg.norm(delta)
    if dist < 1e-9 or dist <= step_size:
        return target.copy()

    return start + delta / dist * step_size


def collision_free(grid: Any, a: np.ndarray, b: np.ndarray, step: float | None = None) -> bool:
    """Return whether the straight segment between two points is free."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    dist = np.linalg.norm(b - a)
    if dist < 1e-9:
        return is_point_free(grid, a)

    if step is None:
        step = grid.resolution * 0.5

    n = max(2, int(np.ceil(dist / step)))
    for alpha in np.linspace(0.0, 1.0, n):
        p = (1.0 - alpha) * a + alpha * b
        if not is_point_free(grid, p):
            return False

    return True


def is_point_free(grid: Any, point: np.ndarray) -> bool:
    """Return whether a world-frame point is in a free grid cell."""
    return grid.is_free(grid.world_to_grid(point))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Return Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def reconstruct_path(nodes: list[RRTNode], goal_idx: int) -> np.ndarray:
    """Reconstruct a world-frame path from tree nodes."""
    path = []
    idx = goal_idx

    while idx >= 0:
        node = nodes[idx]
        path.append(node.point)
        idx = node.parent

    path.reverse()
    return np.asarray(path, dtype=float)
