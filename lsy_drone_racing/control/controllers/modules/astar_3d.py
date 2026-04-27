import heapq
import math
import numpy as np


NEIGHBOR_STEPS_18 = tuple(
    (dx, dy, dz, math.sqrt(dx * dx + dy * dy + dz * dz))
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0) and (int(dx != 0) + int(dy != 0) + int(dz != 0) <= 2)
)


def astar_3d(grid, start_idx, goal_idx, max_iterations=200_000, heuristic_weight=1.0):
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

    shape = tuple(grid.shape.tolist())
    occupied = grid.occupied

    g_score = np.full(shape, np.inf, dtype=float)
    closed = np.zeros(shape, dtype=bool)
    parents = np.full(shape + (3,), -1, dtype=np.int32)

    sx, sy, sz = start_idx
    gx, gy, gz = goal_idx

    g_score[sx, sy, sz] = 0.0

    start_h = heuristic(start_idx, goal_idx)
    open_heap = [(heuristic_weight * start_h, start_h, sx, sy, sz)]

    iterations = 0

    while open_heap:
        iterations += 1

        if iterations > max_iterations:
            print("A*: exceeded max iterations")
            return None

        _, _, cx, cy, cz = heapq.heappop(open_heap)

        if closed[cx, cy, cz]:
            continue

        closed[cx, cy, cz] = True

        if (cx, cy, cz) == goal_idx:
            return reconstruct_path(parents, start_idx, goal_idx)

        current_g = g_score[cx, cy, cz]

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

            tentative_g = current_g + step_cost

            if tentative_g >= g_score[nx, ny, nz]:
                continue

            g_score[nx, ny, nz] = tentative_g
            parents[nx, ny, nz] = (cx, cy, cz)

            h_score = heuristic((nx, ny, nz), goal_idx)
            f_score = tentative_g + heuristic_weight * h_score
            heapq.heappush(open_heap, (f_score, h_score, nx, ny, nz))

    print("A*: failed to find path")
    return None


def heuristic(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def reconstruct_path(parents, start_idx, goal_idx):
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