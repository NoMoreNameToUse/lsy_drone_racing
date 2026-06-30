"""Interactive 3D view of the A* occupancy grid for a logged plan.

Reconstructs the ``OccupancyGrid3D`` the path planner used at a chosen planning
event (the initial plan or any replan) directly from a ``flight_seed<N>.npz``
log -- no controller changes, no live sim. It rebuilds the grid from the gate /
obstacle poses the planner saw at that moment and renders a self-contained,
orbitable HTML:

* occupied voxels, coloured by source -- inflated obstacle poles (brown) vs.
  gate frames with the aperture carved free (blue),
* the gates (aperture + frame) and obstacle poles drawn to scale,
* the A* path / waypoints produced for that plan,
* the grid bounding box.

Because obs gate/obstacle poses are sensor-masked until in range, the grid for an
early plan shows nominal poses; later replans show the revealed (true) poses --
exactly what A* searched. Inspect the replan right before a failure to see, e.g.,
an obstacle inflated across a gate aperture (an infeasible corridor).

Run (inside ``pixi shell``):

    python lsy_drone_racing/control/controllers/utility/sim/visualize_grid.py --seed 4
    python .../visualize_grid.py --seed 4 --replan 3      # grid at replan #3
    python .../visualize_grid.py --file debug_outputs/flight_seed4.npz --list

Grid parameters default to the controller's
(``grid_resolution=0.05, safety_margin=0.05, obstacle_radius=0.18``); override
with the matching flags if you change the controller.
"""

from __future__ import annotations

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

from typing import TYPE_CHECKING  # noqa: E402

import fire  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from lsy_drone_racing.control.controllers.modules.occupancy_grid_3d_improved import (  # noqa: E402
    OccupancyGrid3D,
)
from lsy_drone_racing.control.controllers.utility.sim.analyze_flight import (  # noqa: E402
    OBSTACLE_TOP,
    REPO_ROOT,
    _gate_loop,
    _resolve_path,
    load_flight,
)
from lsy_drone_racing.utils import load_config  # noqa: E402

if TYPE_CHECKING:
    from pathlib import Path

GATE_APERTURE_HALF = 0.2
GATE_OUTER_HALF = 0.36


def _nearest_log_row(flight: dict, global_tick: int) -> int:
    """Index of the log row whose global tick is closest to ``global_tick``."""
    rows_global = np.rint(flight["log_t"] * flight["freq"]).astype(int)
    return int(np.argmin(np.abs(rows_global - global_tick)))


def reconstruct_scene(flight: dict, replan: int) -> dict:
    """Rebuild the gate/obstacle poses + A* path for one planning event.

    Args:
        flight: Loaded flight log.
        replan: Replan index to inspect; ``-1`` selects the initial (pre-flight)
            plan built in the controller constructor.

    Returns:
        Dict with ``gates_pos/quat``, ``obstacles_pos``, ``start``, ``waypoints``
        and a human-readable ``label``.
    """
    n_replans = int(flight["n_replans"])
    if replan >= n_replans:
        raise IndexError(f"replan {replan} out of range (log has {n_replans} replans; "
                         f"use -1 for the initial plan)")

    if replan < 0:
        return {
            "gates_pos": flight["gates_pos"],
            "gates_quat": flight["gates_quat"],
            "obstacles_pos": flight["obstacles_pos"],
            "start": flight["start_pos"],
            "waypoints": flight["initial_raw_waypoints"],
            "label": "initial plan (t=0.00s)",
        }

    gtick = int(flight["replan_ticks"][replan])
    row = _nearest_log_row(flight, gtick)
    return {
        "gates_pos": flight["sensed_gates_pos"][row],
        "gates_quat": flight["sensed_gates_quat"][row],
        "obstacles_pos": flight["sensed_obstacles_pos"][row],
        "start": flight["executed_pos"][row],
        "waypoints": flight[f"replan_{replan}_waypoints"],
        "label": f"replan #{replan} (t={flight['log_t'][row]:.2f}s, "
                 f"target_gate={int(flight['replan_target_gate'][replan])})",
    }


def _empty(cols: int) -> np.ndarray:
    return np.zeros((0, cols), dtype=float)


def build_grids(scene: dict, config: object, resolution: float, safety_margin: float,
                obstacle_radius: float) -> dict:
    """Build the full grid plus obstacle-only / gate-only grids for colouring.

    The obstacle-only and gate-only grids isolate each source's contribution; the
    A* occupancy grid is their union.
    """
    common = dict(config=config, resolution=resolution, safety_margin=safety_margin,
                  obstacle_radius=obstacle_radius)
    full = OccupancyGrid3D(obs={
        "gates_pos": scene["gates_pos"], "gates_quat": scene["gates_quat"],
        "obstacles_pos": scene["obstacles_pos"],
    }, **common)
    obst_only = OccupancyGrid3D(obs={
        "gates_pos": _empty(3), "gates_quat": _empty(4),
        "obstacles_pos": scene["obstacles_pos"],
    }, **common)
    gate_only = OccupancyGrid3D(obs={
        "gates_pos": scene["gates_pos"], "gates_quat": scene["gates_quat"],
        "obstacles_pos": _empty(3),
    }, **common)
    return {"full": full, "obstacle": obst_only, "gate": gate_only}


def _occupied_points(grid: OccupancyGrid3D, max_voxels: int) -> np.ndarray:
    """World-space centres of occupied cells, randomly thinned to ``max_voxels``."""
    xi, yi, zi = np.nonzero(grid.occupied)
    if xi.size == 0:
        return _empty(3)
    if xi.size > max_voxels:
        sel = np.random.default_rng(0).choice(xi.size, max_voxels, replace=False)
        xi, yi, zi = xi[sel], yi[sel], zi[sel]
    return grid.bounds_low + np.stack([xi, yi, zi], axis=1) * grid.resolution


def _bbox_traces(grid: OccupancyGrid3D) -> list[go.Scatter3d]:
    """12 edges of the grid bounding box as a single dashed trace."""
    lo, hi = grid.bounds_low, grid.bounds_high
    c = np.array([[lo[0], lo[1], lo[2]], [hi[0], lo[1], lo[2]], [hi[0], hi[1], lo[2]],
                  [lo[0], hi[1], lo[2]], [lo[0], lo[1], hi[2]], [hi[0], lo[1], hi[2]],
                  [hi[0], hi[1], hi[2]], [lo[0], hi[1], hi[2]]])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    xs, ys, zs = [], [], []
    for a, b in edges:
        xs += [c[a, 0], c[b, 0], None]
        ys += [c[a, 1], c[b, 1], None]
        zs += [c[a, 2], c[b, 2], None]
    return [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                         line=dict(color="lightgray", width=2), name="grid bounds",
                         hoverinfo="skip")]


def plot_grid_3d(flight: dict, scene: dict, grids: dict, out_html: Path,
                 voxel_size: float, max_voxels: int) -> None:
    """Render and save the interactive 3D occupancy-grid view."""
    full = grids["full"]
    traces: list[go.Scatter3d] = []

    # Occupied voxels by source (their union is the A* occupancy grid).
    for key, color, name in (("obstacle", "saddlebrown", "occupied: obstacle"),
                             ("gate", "royalblue", "occupied: gate frame")):
        pts = _occupied_points(grids[key], max_voxels)
        if pts.size:
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                marker=dict(size=voxel_size, color=color, symbol="square", opacity=0.35),
                name=name, hoverinfo="skip",
            ))

    # Gates to scale (aperture + frame), oriented by quaternion.
    gpos, gquat = scene["gates_pos"], scene["gates_quat"]
    for g in range(gpos.shape[0]):
        for half, c, w, nm in ((GATE_APERTURE_HALF, "deepskyblue", 6, "gate aperture"),
                               (GATE_OUTER_HALF, "lightskyblue", 2, "gate frame")):
            loop = _gate_loop(gpos[g], gquat[g], half)
            traces.append(go.Scatter3d(
                x=loop[:, 0], y=loop[:, 1], z=loop[:, 2], mode="lines",
                line=dict(color=c, width=w), name=nm, legendgroup=nm,
                showlegend=(g == 0), hoverinfo="skip"))

    # Obstacle poles to scale.
    for k, o in enumerate(scene["obstacles_pos"]):
        traces.append(go.Scatter3d(
            x=[o[0], o[0]], y=[o[1], o[1]], z=[0.0, OBSTACLE_TOP], mode="lines",
            line=dict(color="black", width=4), name="obstacle pole",
            legendgroup="obstacle pole", showlegend=(k == 0),
            text=f"obstacle {k}", hoverinfo="text"))

    # A* path / waypoints for this plan.
    wp = np.asarray(scene["waypoints"], dtype=float)
    if wp.size:
        traces.append(go.Scatter3d(
            x=wp[:, 0], y=wp[:, 1], z=wp[:, 2], mode="lines+markers",
            line=dict(color="limegreen", width=5), marker=dict(size=3, color="green"),
            name="A* path", hoverinfo="skip"))

    # Start marker + bounding box.
    s = np.asarray(scene["start"], dtype=float)
    traces.append(go.Scatter3d(x=[s[0]], y=[s[1]], z=[s[2]], mode="markers",
                               marker=dict(size=6, color="red"), name="start",
                               hoverinfo="skip"))
    traces.extend(_bbox_traces(full))

    n_occ = int(full.occupied.sum())
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(f"A* occupancy grid: seed {flight['seed']} | {scene['label']} | "
               f"res={full.resolution:.3f} m, shape={tuple(int(v) for v in full.shape)}, "
               f"{n_occ} occupied cells"),
        scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                   aspectmode="data"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_html, include_plotlyjs=True)


def visualize(
    seed: int | None = None,
    file: str | None = None,
    replan: int = -1,
    config: str = "level3.toml",
    resolution: float = 0.05,
    safety_margin: float = 0.05,
    obstacle_radius: float = 0.18,
    voxel_size: float = 2.5,
    max_voxels: int = 40000,
    list_replans: bool = False,
) -> dict:
    """Rebuild and render the A* occupancy grid for a logged plan.

    Args:
        seed: Seed whose ``flight_seed<seed>.npz`` to load.
        file: Explicit flight npz path (overrides ``seed``).
        replan: Planning event to inspect; ``-1`` = initial plan, ``0..n-1`` = replans.
        config: Config file (for the grid bounds / safety limits).
        resolution: Grid cell size (m); match the controller's path generator.
        safety_margin: Obstacle/gate safety margin (m).
        obstacle_radius: Inflated obstacle radius (m).
        voxel_size: Plotly marker size for occupied cells (px).
        max_voxels: Cap on plotted voxels per source (random thinning above this).
        list_replans: Print the available planning events and exit.

    Returns:
        Dict with the output ``html`` path and occupied-cell count.
    """
    path = _resolve_path(seed, file)
    flight = load_flight(path)

    if list_replans:
        print(f"{path.name}: {int(flight['n_replans'])} replans "
              f"(use --replan -1 for the initial plan)")
        for k in range(int(flight["n_replans"])):
            print(f"  replan {k}: global_tick={int(flight['replan_ticks'][k])}, "
                  f"target_gate={int(flight['replan_target_gate'][k])}")
        return {"html": None, "n_occupied": 0}

    scene = reconstruct_scene(flight, replan)
    cfg = load_config(REPO_ROOT / "config" / config)
    grids = build_grids(scene, cfg, resolution, safety_margin, obstacle_radius)

    tag = "initial" if replan < 0 else f"replan{replan}"
    out_html = path.with_suffix("").parent / f"{path.with_suffix('').name}_grid_{tag}.html"
    plot_grid_3d(flight, scene, grids, out_html, voxel_size, max_voxels)

    n_occ = int(grids["full"].occupied.sum())
    print(f"Grid: shape={tuple(int(v) for v in grids['full'].shape)}, "
          f"res={resolution} m, {n_occ} occupied cells  |  {scene['label']}")
    print(f"Saved 3D grid view -> {out_html}  (open in a browser, drag to orbit)")
    return {"html": str(out_html), "n_occupied": n_occ}


if __name__ == "__main__":
    fire.Fire(visualize, serialize=lambda _: None)
