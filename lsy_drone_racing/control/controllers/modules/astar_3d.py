import heapq
import math
import numpy as np


def astar_3d(grid, start_idx, goal_idx, max_iterations=200_000):
    """
    Basic 26-connected 3D A*.

    Args:
        grid: OccupancyGrid3D
        start_idx: tuple[int, int, int]
        goal_idx: tuple[int, int, int]

    Returns:
        list of grid indices, or None if planning failed.
    """
    start_idx = tuple(start_idx)
    goal_idx = tuple(goal_idx)

    if not grid.is_free(start_idx):
        print(f"A*: start occupied {start_idx}, world={grid.grid_to_world(start_idx)}")
        return None

    if not grid.is_free(goal_idx):
        print(f"A*: goal occupied {goal_idx}, world={grid.grid_to_world(goal_idx)}")
        return None

    open_heap = []
    heapq.heappush(open_heap, (0.0, start_idx))

    came_from = {}
    g_score = {start_idx: 0.0}
    visited = set()

    iterations = 0

    while open_heap:
        iterations += 1

        if iterations > max_iterations:
            print("A*: exceeded max iterations")
            return None

        _, current = heapq.heappop(open_heap)

        if current in visited:
            continue

        visited.add(current)

        if current == goal_idx:
            return reconstruct_path(came_from, current)

        for neighbor, step_cost in neighbors_26(current):
            if neighbor in visited:
                continue

            if not grid.is_free(neighbor):
                continue

            tentative_g = g_score[current] + step_cost

            if tentative_g < g_score.get(neighbor, math.inf):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                f_score = tentative_g + heuristic(neighbor, goal_idx)
                heapq.heappush(open_heap, (f_score, neighbor))

    print("A*: failed to find path")
    return None


def neighbors_26(idx):
    ix, iy, iz = idx

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == 0 and dy == 0 and dz == 0:
                    continue

                neighbor = (ix + dx, iy + dy, iz + dz)
                cost = math.sqrt(dx * dx + dy * dy + dz * dz)
                yield neighbor, cost


def heuristic(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.linalg.norm(a - b))


def reconstruct_path(came_from, current):
    path = [current]

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path