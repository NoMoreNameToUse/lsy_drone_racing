"""Compare multiple saved path-debug npz files in one 3D Plotly view.

Run:
    pixi run python debug_outputs/compare_debug_paths.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R


DEFAULT_FILES = (
    "astar_debug.npz",
    "theta_star_debug.npz",
    "rrt_star_debug.npz",
    "d_star_lite_debug.npz",
    "curve_gate_debug.npz",
)

COLORS = (
    "#2563eb",
    "#16a34a",
    "#dc2626",
    "#9333ea",
    "#f97316",
    "#0891b2",
)


def add_points(fig, points, name, color, size=5):
    points = np.asarray(points)
    if points.size == 0:
        return None

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            name=name,
            marker=dict(size=size, color=color),
        )
    )
    return len(fig.data) - 1


def add_line(
    fig,
    points,
    name,
    color,
    width=4,
    dash=None,
    showlegend=True,
    legendgroup=None,
    hovertemplate=None,
):
    points = np.asarray(points)
    if points.size == 0:
        return None

    line = dict(color=color, width=width)
    if dash is not None:
        line["dash"] = dash

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            name=name,
            line=line,
            showlegend=showlegend,
            legendgroup=legendgroup,
            hovertemplate=hovertemplate,
        )
    )
    return len(fig.data) - 1


def add_length_label(fig, point, label, length_m, color, legendgroup=None):
    point = np.asarray(point, dtype=float)
    if point.size != 3:
        return None

    fig.add_trace(
        go.Scatter3d(
            x=[point[0]],
            y=[point[1]],
            z=[point[2]],
            mode="text",
            name=f"{label} length label",
            text=[f"{label}<br>{length_m:.2f} m"],
            textfont=dict(color=color, size=12),
            textposition="top center",
            showlegend=False,
            legendgroup=legendgroup,
            hovertemplate=f"{label}<br>trajectory length: {length_m:.3f} m<extra></extra>",
        )
    )
    return len(fig.data) - 1


def add_gate_frame(fig, gate_pos, gate_quat, name, color="rgba(37, 99, 235, 0.45)"):
    gate_pos = np.asarray(gate_pos, dtype=float)
    rot = R.from_quat(gate_quat)

    outer = 0.72 / 2.0
    inner = 0.40 / 2.0

    def local_to_world(local):
        return gate_pos + rot.apply(np.asarray(local, dtype=float))

    outer_local = np.array(
        [
            [0.0, -outer, -outer],
            [0.0, outer, -outer],
            [0.0, outer, outer],
            [0.0, -outer, outer],
            [0.0, -outer, -outer],
        ]
    )
    inner_local = np.array(
        [
            [0.0, -inner, -inner],
            [0.0, inner, -inner],
            [0.0, inner, inner],
            [0.0, -inner, inner],
            [0.0, -inner, -inner],
        ]
    )

    outer_world = np.array([local_to_world(p) for p in outer_local])
    inner_world = np.array([local_to_world(p) for p in inner_local])

    add_line(fig, outer_world, f"{name} outer", color, width=5)
    add_line(fig, inner_world, f"{name} opening", color, width=3, dash="dash")


def add_obstacle_cylinder(
    fig,
    center,
    radius=0.015,
    height=1.52,
    color="rgba(249, 115, 22, 0.65)",
    resolution=24,
):
    center = np.asarray(center, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z_bottom = np.zeros_like(theta)
    z_top = np.ones_like(theta) * height

    add_line(fig, np.column_stack([x, y, z_bottom]), "obstacle poles", color, width=2)
    add_line(fig, np.column_stack([x, y, z_top]), "obstacle poles", color, width=2, showlegend=False)

    for i in range(0, resolution, max(1, resolution // 6)):
        add_line(
            fig,
            np.array([[x[i], y[i], 0.0], [x[i], y[i], height]]),
            "obstacle poles",
            color,
            width=2,
            showlegend=False,
        )


def add_inflated_obstacle_cylinder(
    fig,
    center,
    radius=0.25,
    height=1.52,
    color="rgba(249, 115, 22, 0.18)",
    resolution=36,
):
    center = np.asarray(center, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, resolution)
    z = np.linspace(0.0, height, 2)
    theta_grid, z_grid = np.meshgrid(theta, z)

    x = center[0] + radius * np.cos(theta_grid)
    y = center[1] + radius * np.sin(theta_grid)

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z_grid,
            name="inflated obstacle safety zone",
            opacity=0.18,
            colorscale=[[0, "orange"], [1, "orange"]],
            showscale=False,
            showlegend=False,
        )
    )
    return len(fig.data) - 1


def path_length(points):
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def min_obstacle_clearance_xy(traj_pos, obstacles_pos):
    traj_pos = np.asarray(traj_pos, dtype=float)
    obstacles_pos = np.asarray(obstacles_pos, dtype=float)
    if traj_pos.size == 0 or obstacles_pos.size == 0:
        return np.nan

    best = np.inf
    for obs in obstacles_pos:
        d_xy = np.linalg.norm(traj_pos[:, :2] - obs[:2], axis=1)
        best = min(best, float(np.min(d_xy)))
    return best


def load_debug_file(path):
    data = np.load(path, allow_pickle=False)
    required = ("gates_pos", "gates_quat", "obstacles_pos", "start_pos", "raw_waypoints", "traj_pos")
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"{path} missing keys: {missing}")
    return data


def label_for(path, data):
    if "planner_name" in data.files:
        planner_name = np.asarray(data["planner_name"]).item()
        return str(planner_name)
    return path.stem.replace("_debug", "")


def seed_for(data):
    if "env_seed" not in data.files:
        return "unknown"
    return str(np.asarray(data["env_seed"]).item())


def default_existing_files(script_dir):
    files = [script_dir / name for name in DEFAULT_FILES]
    existing = [path for path in files if path.exists()]
    if existing:
        return existing

    fallback = script_dir / "level1_path_debug.npz"
    return [fallback] if fallback.exists() else []


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Debug npz files to compare. Defaults to astar/theta_star/rrt_star debug files.",
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=None,
        help="Optional HTML output path. If omitted, opens Plotly's default viewer.",
    )
    return parser.parse_args()


def write_html_with_path_checkboxes(fig, output_html, path_groups):
    """Write Plotly HTML and inject checkbox controls for planner trace groups."""
    controls = json.dumps(path_groups)
    post_script = f"""
    (function() {{
      var plot = document.getElementById('{{plot_id}}');
      var groups = {controls};

      var panel = document.createElement('div');
      panel.style.position = 'fixed';
      panel.style.top = '12px';
      panel.style.right = '12px';
      panel.style.zIndex = '1000';
      panel.style.maxWidth = '280px';
      panel.style.padding = '12px 14px';
      panel.style.background = 'rgba(255, 255, 255, 0.94)';
      panel.style.border = '1px solid #d0d7de';
      panel.style.borderRadius = '6px';
      panel.style.boxShadow = '0 6px 18px rgba(0, 0, 0, 0.12)';
      panel.style.font = '13px system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';

      var title = document.createElement('div');
      title.textContent = 'Displayed paths';
      title.style.fontWeight = '600';
      title.style.marginBottom = '8px';
      panel.appendChild(title);

      groups.forEach(function(group) {{
        var row = document.createElement('label');
        row.style.display = 'flex';
        row.style.alignItems = 'center';
        row.style.gap = '7px';
        row.style.margin = '6px 0';
        row.style.cursor = 'pointer';

        var checkbox = document.createElement('input');
        checkbox.type = 'checkbox';
        checkbox.checked = true;
        checkbox.style.accentColor = group.color;

        var swatch = document.createElement('span');
        swatch.style.width = '10px';
        swatch.style.height = '10px';
        swatch.style.borderRadius = '50%';
        swatch.style.background = group.color;
        swatch.style.display = 'inline-block';

        var text = document.createElement('span');
        text.textContent = group.label + ' (' + group.traj_len.toFixed(2) + ' m)';

        checkbox.addEventListener('change', function() {{
          Plotly.restyle(plot, {{visible: checkbox.checked ? true : 'legendonly'}}, group.trace_indices);
        }});

        row.appendChild(checkbox);
        row.appendChild(swatch);
        row.appendChild(text);
        panel.appendChild(row);
      }});

      var hint = document.createElement('div');
      hint.textContent = 'Legend clicks also toggle each path group.';
      hint.style.marginTop = '8px';
      hint.style.color = '#57606a';
      hint.style.fontSize = '12px';
      panel.appendChild(hint);

      document.body.appendChild(panel);
    }})();
    """
    fig.write_html(output_html, post_script=post_script)


def main():
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    paths = args.files or default_existing_files(script_dir)

    if not paths:
        print("No debug files found.")
        print("Expected files like:")
        for name in DEFAULT_FILES:
            print(f"  {script_dir / name}")
        return

    loaded = []
    for path in paths:
        path = path if path.is_absolute() else Path.cwd() / path
        data = load_debug_file(path)
        loaded.append((path, data, label_for(path, data)))

    if len(loaded) == 1:
        print("\nOnly one debug file was loaded, so the figure shows one path.")
        print("To compare planners, run each planner once so these files exist:")
        for name in DEFAULT_FILES:
            print(f"  {script_dir / name}")

    reference = loaded[0][1]
    env_seed = seed_for(reference)
    gates_pos = reference["gates_pos"]
    gates_quat = reference["gates_quat"]
    obstacles_pos = reference["obstacles_pos"]
    start_pos = reference["start_pos"].reshape(1, 3)

    fig = go.Figure()

    add_points(fig, start_pos, "start", "black", size=7)
    add_points(fig, gates_pos, "gate centers", "rgba(37, 99, 235, 0.7)", size=4)
    add_points(fig, obstacles_pos, "obstacle markers", "rgba(249, 115, 22, 0.8)", size=4)

    for i, (gate_pos, gate_quat) in enumerate(zip(gates_pos, gates_quat)):
        add_gate_frame(fig, gate_pos, gate_quat, name=f"gate {i}")

    for obstacle in obstacles_pos:
        add_inflated_obstacle_cylinder(fig, obstacle)
        add_obstacle_cylinder(fig, obstacle)

    print("\nPath comparison:")
    path_groups = []
    for i, (path, data, label) in enumerate(loaded):
        color = COLORS[i % len(COLORS)]
        raw_waypoints = data["raw_waypoints"]
        traj_pos = data["traj_pos"]
        traj_plot = traj_pos[:: max(1, len(traj_pos) // 1500)]
        raw_len = path_length(raw_waypoints)
        traj_len = path_length(traj_pos)
        min_obs_xy = min_obstacle_clearance_xy(traj_pos, obstacles_pos)
        legendgroup = f"path-{i}"
        trace_indices = []

        waypoint_idx = add_points(fig, raw_waypoints, f"{label} waypoints", color, size=3)
        if waypoint_idx is not None:
            fig.data[waypoint_idx].legendgroup = legendgroup
            fig.data[waypoint_idx].showlegend = False
            fig.data[waypoint_idx].hovertemplate = (
                f"{label} waypoint<br>"
                "x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>"
            )
            trace_indices.append(waypoint_idx)

        raw_idx = add_line(
            fig,
            raw_waypoints,
            f"{label} raw path ({raw_len:.2f} m)",
            color,
            width=3,
            dash="dot",
            showlegend=False,
            legendgroup=legendgroup,
            hovertemplate=(
                f"{label} raw path<br>"
                f"raw length: {raw_len:.3f} m<br>"
                "x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>"
            ),
        )
        if raw_idx is not None:
            trace_indices.append(raw_idx)

        traj_idx = add_line(
            fig,
            traj_plot,
            f"{label} trajectory ({traj_len:.2f} m)",
            color,
            width=6,
            showlegend=True,
            legendgroup=legendgroup,
            hovertemplate=(
                f"{label} trajectory<br>"
                f"trajectory length: {traj_len:.3f} m<br>"
                f"min obstacle XY: {min_obs_xy:.3f} m<br>"
                "x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>"
            ),
        )
        if traj_idx is not None:
            trace_indices.append(traj_idx)

        label_idx = add_length_label(fig, traj_pos[-1], label, traj_len, color, legendgroup=legendgroup)
        if label_idx is not None:
            trace_indices.append(label_idx)

        path_groups.append(
            {
                "label": label,
                "color": color,
                "traj_len": traj_len,
                "trace_indices": trace_indices,
            }
        )

        print(
            f"  {label}: file={path.name}, "
            f"waypoints={len(raw_waypoints)}, "
            f"raw_len={raw_len:.3f} m, "
            f"traj_len={traj_len:.3f} m, "
            f"min_obs_xy={min_obs_xy:.3f} m"
        )

    fig.update_layout(
        title=(
            "Drone Racing Debug Path Comparison "
            f"(seed={env_seed}): "
            + ", ".join(label for _, _, label in loaded)
        ),
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant", groupclick="togglegroup"),
    )

    if args.output_html is not None:
        write_html_with_path_checkboxes(fig, args.output_html, path_groups)
        print(f"\nWrote: {args.output_html}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
