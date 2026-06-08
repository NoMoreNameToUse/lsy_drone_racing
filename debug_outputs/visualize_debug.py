import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
from pathlib import Path


DEBUG_FILE = Path(__file__).parent / "level1_path_debug.npz"


def add_points(fig, points, name, color, size=5):
    points = np.asarray(points)
    if points.size == 0:
        return

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


def add_line(fig, points, name, color, width=4):
    points = np.asarray(points)
    if points.size == 0:
        return

    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            name=name,
            line=dict(color=color, width=width),
        )
    )


def add_gate_frame(
    fig,
    gate_pos,
    gate_quat,
    name,
    outer_width=0.72,
    inner_width=0.40,
    color="blue",
):
    """
    Draws the gate outer square and inner opening square in 3D.
    Assumes gate-local:
        x = normal direction
        y = lateral
        z = vertical relative to gate center
    """
    gate_pos = np.asarray(gate_pos, dtype=float)
    rot = R.from_quat(gate_quat)

    outer = outer_width / 2.0
    inner = inner_width / 2.0

    def local_to_world(local):
        return gate_pos + rot.apply(np.asarray(local, dtype=float))

    # Outer square corners in local frame, x = 0 plane
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

    fig.add_trace(
        go.Scatter3d(
            x=outer_world[:, 0],
            y=outer_world[:, 1],
            z=outer_world[:, 2],
            mode="lines",
            name=f"{name} outer frame",
            line=dict(color=color, width=6),
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=inner_world[:, 0],
            y=inner_world[:, 1],
            z=inner_world[:, 2],
            mode="lines",
            name=f"{name} inner opening",
            line=dict(color=color, width=3, dash="dash"),
        )
    )

    # Draw gate normal direction
    normal_start = gate_pos
    normal_end = gate_pos + 0.25 * rot.apply([1.0, 0.0, 0.0])

    fig.add_trace(
        go.Scatter3d(
            x=[normal_start[0], normal_end[0]],
            y=[normal_start[1], normal_end[1]],
            z=[normal_start[2], normal_end[2]],
            mode="lines",
            name=f"{name} forward",
            line=dict(color=color, width=4),
        )
    )


def add_obstacle_cylinder(
    fig,
    center,
    radius=0.015,
    height=1.52,
    color="orange",
    name="obstacle",
    resolution=24,
):
    """
    Draws a simple vertical cylinder.
    obstacle pos in env is marker/top-ish position, but XY is what matters.
    We draw from z=0 to height.
    """
    center = np.asarray(center, dtype=float)
    theta = np.linspace(0, 2 * np.pi, resolution)

    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)

    z_bottom = np.zeros_like(theta)
    z_top = np.ones_like(theta) * height

    # Bottom/top rings
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z_bottom,
            mode="lines",
            name=f"{name} base",
            line=dict(color=color, width=2),
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z_top,
            mode="lines",
            name=f"{name} top",
            line=dict(color=color, width=2),
            showlegend=False,
        )
    )

    # Vertical sides
    for i in range(0, resolution, max(1, resolution // 6)):
        fig.add_trace(
            go.Scatter3d(
                x=[x[i], x[i]],
                y=[y[i], y[i]],
                z=[0.0, height],
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                showlegend=False,
            )
        )


def add_inflated_obstacle_cylinder(
    fig,
    center,
    radius=0.25,
    height=1.52,
    color="rgba(255, 140, 0, 0.18)",
    name="inflated obstacle",
    resolution=32,
):
    """
    Draws a translucent inflated safety cylinder as a mesh.
    """
    center = np.asarray(center, dtype=float)

    theta = np.linspace(0, 2 * np.pi, resolution)
    z = np.linspace(0.0, height, 2)

    theta_grid, z_grid = np.meshgrid(theta, z)

    x = center[0] + radius * np.cos(theta_grid)
    y = center[1] + radius * np.sin(theta_grid)

    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z_grid,
            name=name,
            opacity=0.18,
            colorscale=[[0, "orange"], [1, "orange"]],
            showscale=False,
            showlegend=False,
        )
    )


def print_clearance_report(traj_pos, obstacles_pos):
    print("\nObstacle clearance report:")
    for i, obs in enumerate(obstacles_pos):
        d_xy = np.linalg.norm(traj_pos[:, :2] - obs[:2], axis=1)
        idx = int(np.argmin(d_xy))
        print(
            f"  obstacle {i}: min_xy={d_xy[idx]:.3f} m, "
            f"traj_idx={idx}, traj_pos={traj_pos[idx]}, obs={obs}"
        )


def print_gate_frame_report(
    traj_pos,
    gates_pos,
    gates_quat,
    inner_half=0.20,
    outer_half=0.36,
    plane_tol=0.08,
    margin=0.035,
):
    print("\nGate frame clearance report:")
    for gate_idx, (gate_pos, gate_quat) in enumerate(zip(gates_pos, gates_quat)):
        rot = R.from_quat(gate_quat)
        local = rot.inv().apply(traj_pos - gate_pos)

        x = local[:, 0]
        y = local[:, 1]
        z = local[:, 2]

        near_gate_plane = np.abs(x) < plane_tol

        inside_outer = (
            (np.abs(y) < outer_half + margin)
            & (np.abs(z) < outer_half + margin)
        )

        inside_safe_opening = (
            (np.abs(y) < inner_half - margin)
            & (np.abs(z) < inner_half - margin)
        )

        hits_frame = near_gate_plane & inside_outer & (~inside_safe_opening)

        if np.any(hits_frame):
            hit_indices = np.where(hits_frame)[0]
            idx = int(hit_indices[0])
            print(
                f"  gate {gate_idx}: WARNING {len(hit_indices)} suspicious samples; "
                f"first idx={idx}, world={traj_pos[idx]}, local={local[idx]}"
            )
        else:
            print(f"  gate {gate_idx}: no suspicious samples")


def main():
    data = np.load(DEBUG_FILE)

    gates_pos = data["gates_pos"]
    gates_quat = data["gates_quat"]
    obstacles_pos = data["obstacles_pos"]
    start_pos = data["start_pos"]
    raw_waypoints = data["raw_waypoints"]
    traj_pos = data["traj_pos"]
    planner_name = str(np.asarray(data["planner_name"]).item()) if "planner_name" in data.files else "unknown"

    print("Loaded:", DEBUG_FILE)
    print("planner_name:", planner_name)
    print("gates_pos:", gates_pos.shape)
    print("obstacles_pos:", obstacles_pos.shape)
    print("raw_waypoints:", raw_waypoints.shape)
    print("traj_pos:", traj_pos.shape)

    print_clearance_report(traj_pos, obstacles_pos)
    print_gate_frame_report(traj_pos, gates_pos, gates_quat)

    fig = go.Figure()

    # Main geometry
    add_points(fig, gates_pos, "gate centers", "blue", size=5)
    add_points(fig, obstacles_pos, "obstacle markers", "orange", size=5)
    add_points(fig, start_pos.reshape(1, 3), "start", "black", size=7)

    # Waypoints and trajectory
    add_points(fig, raw_waypoints, "raw waypoints", "magenta", size=4)
    add_line(fig, raw_waypoints, "raw waypoint path", "magenta", width=4)

    # Downsample trajectory for plotting if huge
    traj_plot = traj_pos[:: max(1, len(traj_pos) // 1500)]
    add_line(fig, traj_plot, "trajectory", "green", width=5)

    # Gates
    for i, (gate_pos, gate_quat) in enumerate(zip(gates_pos, gates_quat)):
        add_gate_frame(
            fig,
            gate_pos,
            gate_quat,
            name=f"gate {i}",
            color="blue",
        )

    # Obstacles
    for i, obs in enumerate(obstacles_pos):
        add_obstacle_cylinder(
            fig,
            obs,
            radius=0.015,
            height=1.52,
            color="orange",
            name=f"obstacle {i}",
        )

        add_inflated_obstacle_cylinder(
            fig,
            obs,
            radius=0.25,
            height=1.52,
            name=f"obstacle {i} inflated",
        )

    fig.update_layout(
        title=f"Drone Racing Debug Path - 3D View ({planner_name})",
        scene=dict(
            xaxis_title="x [m]",
            yaxis_title="y [m]",
            zaxis_title="z [m]",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
    )

    fig.show()


if __name__ == "__main__":
    main()
