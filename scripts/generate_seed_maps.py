"""Generate one top-down race-track map per seed.

Example:
    pixi run python scripts/generate_seed_maps.py --config level3.toml --seed-count 20

For selected seeds:
    pixi run python scripts/generate_seed_maps.py --config level2.toml --seeds 1 7 42 --save-npz
"""

from __future__ import annotations

import argparse
import csv
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import gymnasium
import matplotlib
import numpy as np
import toml
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.utils import load_config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="level3.toml",
        help="Config file name inside config/, or a direct path to a toml file.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Explicit seed list. Overrides --seed-start/--seed-count.",
    )
    parser.add_argument("--seed-start", type=int, default=1, help="First generated seed.")
    parser.add_argument("--seed-count", type=int, default=10, help="Number of consecutive seeds.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "debug_outputs" / "seed_maps",
        help="Directory where maps and metadata are written.",
    )
    parser.add_argument(
        "--save-npz",
        action="store_true",
        help="Also save raw start/gate/obstacle arrays for each seed.",
    )
    parser.add_argument(
        "--save-toml",
        action="store_true",
        help="Also save a frozen config for each generated map.",
    )
    parser.add_argument(
        "--obstacle-margin",
        type=float,
        default=0.25,
        help="Optional inflated obstacle radius drawn in the top-down map.",
    )
    return parser.parse_args()


def resolve_config_path(config_arg: str) -> Path:
    """Resolve a config filename or direct path to an absolute path."""
    path = Path(config_arg)
    if path.exists():
        return path
    return Path(__file__).resolve().parents[1] / "config" / config_arg


def make_env(config: Any) -> Any:
    """Create a single-drone racing environment from config."""
    env = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    return JaxToNumpy(env)


def get_actual_track(env: Any) -> dict[str, np.ndarray]:
    """Read the true post-reset track state from the unwrapped environment."""
    data = env.unwrapped.data
    return {
        "start_pos": np.asarray(data.sim_data.states.pos, dtype=float)[0, 0],
        "gates_pos": np.asarray(data.gates_pos, dtype=float)[0],
        "gates_quat": np.asarray(data.gates_quat, dtype=float)[0],
        "obstacles_pos": np.asarray(data.obstacles_pos, dtype=float)[0],
    }


def gate_yaw(gate_quat: np.ndarray) -> float:
    """Return the yaw angle of an xyzw gate quaternion."""
    return float(R.from_quat(gate_quat).as_euler("xyz")[2])


def draw_map(
    output_path: Path,
    seed: int,
    config_name: str,
    track: dict[str, np.ndarray],
    safety_low: np.ndarray,
    safety_high: np.ndarray,
    obstacle_margin: float,
) -> None:
    """Draw and save a top-down PNG for one track."""
    gates_pos = track["gates_pos"]
    gates_quat = track["gates_quat"]
    obstacles_pos = track["obstacles_pos"]
    start_pos = track["start_pos"]

    fig, ax = plt.subplots(figsize=(8.0, 5.2), dpi=160)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{config_name} | seed {seed}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    width = safety_high[0] - safety_low[0]
    height = safety_high[1] - safety_low[1]
    ax.add_patch(
        Rectangle(
            safety_low[:2],
            width,
            height,
            fill=False,
            linewidth=1.6,
            linestyle="-",
            edgecolor="#4a5568",
            label="safety limits",
        )
    )

    route_xy = np.vstack([start_pos[:2], gates_pos[:, :2]])
    ax.plot(
        route_xy[:, 0],
        route_xy[:, 1],
        "--",
        color="#718096",
        linewidth=1.2,
        alpha=0.75,
        label="start-to-gate order",
    )

    ax.scatter(
        [start_pos[0]],
        [start_pos[1]],
        marker="*",
        s=160,
        color="#111827",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
        label="start",
    )

    for i, obs_pos in enumerate(obstacles_pos, start=1):
        if obstacle_margin > 0:
            ax.add_patch(
                Circle(
                    obs_pos[:2],
                    obstacle_margin,
                    facecolor="#f59e0b",
                    edgecolor="none",
                    alpha=0.14,
                    zorder=1,
                )
            )
        ax.scatter(
            [obs_pos[0]],
            [obs_pos[1]],
            s=58,
            color="#f59e0b",
            edgecolor="#7c2d12",
            linewidth=0.8,
            zorder=4,
            label="obstacle" if i == 1 else None,
        )
        ax.text(obs_pos[0], obs_pos[1] + 0.06, f"O{i}", ha="center", va="bottom", fontsize=8)

    gate_half_width = 0.36
    normal_len = 0.28
    for i, (gate_pos, gate_quat) in enumerate(zip(gates_pos, gates_quat, strict=True), start=1):
        rot = R.from_quat(gate_quat)
        lateral = rot.apply([0.0, 1.0, 0.0])[:2]
        normal = rot.apply([1.0, 0.0, 0.0])[:2]
        a = gate_pos[:2] - gate_half_width * lateral
        b = gate_pos[:2] + gate_half_width * lateral
        ax.plot(
            [a[0], b[0]],
            [a[1], b[1]],
            color="#2563eb",
            linewidth=3.0,
            solid_capstyle="round",
            zorder=6,
            label="gate" if i == 1 else None,
        )
        ax.arrow(
            gate_pos[0],
            gate_pos[1],
            normal_len * normal[0],
            normal_len * normal[1],
            width=0.012,
            head_width=0.08,
            head_length=0.08,
            length_includes_head=True,
            color="#1d4ed8",
            zorder=6,
        )
        ax.text(gate_pos[0], gate_pos[1] - 0.08, f"G{i}", ha="center", va="top", fontsize=8)

    pad = 0.25
    ax.set_xlim(safety_low[0] - pad, safety_high[0] + pad)
    ax.set_ylim(safety_low[1] - pad, safety_high[1] + pad)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_npz(output_path: Path, seed: int, track: dict[str, np.ndarray]) -> None:
    """Save the raw arrays for one generated track."""
    np.savez(
        output_path,
        seed=np.asarray(seed, dtype=int),
        start_pos=track["start_pos"],
        gates_pos=track["gates_pos"],
        gates_quat=track["gates_quat"],
        obstacles_pos=track["obstacles_pos"],
    )


def save_frozen_config(
    output_path: Path, config: Any, seed: int, track: dict[str, np.ndarray]
) -> None:
    """Save a config with the generated track written as fixed object poses."""
    frozen = deepcopy(config).to_dict()
    frozen["env"]["seed"] = seed
    frozen["env"]["track"]["randomize"] = False

    randomizations = frozen["env"].get("randomizations", {})
    for key in ("gate_pos", "gate_rpy", "obstacle_pos"):
        randomizations.pop(key, None)

    for i, gate in enumerate(frozen["env"]["track"]["gates"]):
        gate["pos"] = track["gates_pos"][i].tolist()
        gate["rpy"] = [0.0, 0.0, gate_yaw(track["gates_quat"][i])]
    for i, obstacle in enumerate(frozen["env"]["track"]["obstacles"]):
        obstacle["pos"] = track["obstacles_pos"][i].tolist()
    for i, drone in enumerate(frozen["env"]["track"]["drones"]):
        if i == 0:
            drone["pos"] = track["start_pos"].tolist()

    with open(output_path, "w") as f:
        toml.dump(frozen, f)


def seed_list(args: argparse.Namespace) -> list[int]:
    """Return the requested seeds."""
    if args.seeds:
        return args.seeds
    return list(range(args.seed_start, args.seed_start + args.seed_count))


def write_manifest(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Write one CSV row per generated map."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "png",
                "npz",
                "toml",
                "start_x",
                "start_y",
                "n_gates",
                "n_obstacles",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Generate one map image per seed."""
    args = parse_args()
    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    config.sim.render = False

    args.output_dir.mkdir(parents=True, exist_ok=True)
    seeds = seed_list(args)
    safety_low = np.asarray(config.env.track.safety_limits.pos_limit_low, dtype=float)
    safety_high = np.asarray(config.env.track.safety_limits.pos_limit_high, dtype=float)

    env = make_env(config)
    manifest_rows: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            env.reset(seed=seed)
            track = get_actual_track(env)

            stem = f"{config_path.stem}_seed_{seed:04d}"
            png_path = args.output_dir / f"{stem}.png"
            npz_path = args.output_dir / f"{stem}.npz"
            toml_path = args.output_dir / f"{stem}.toml"

            draw_map(
                png_path,
                seed,
                config_path.name,
                track,
                safety_low,
                safety_high,
                args.obstacle_margin,
            )
            saved_npz = ""
            saved_toml = ""
            if args.save_npz:
                save_npz(npz_path, seed, track)
                saved_npz = str(npz_path)
            if args.save_toml:
                save_frozen_config(toml_path, config, seed, track)
                saved_toml = str(toml_path)

            manifest_rows.append(
                {
                    "seed": seed,
                    "png": str(png_path),
                    "npz": saved_npz,
                    "toml": saved_toml,
                    "start_x": f"{track['start_pos'][0]:.6f}",
                    "start_y": f"{track['start_pos'][1]:.6f}",
                    "n_gates": len(track["gates_pos"]),
                    "n_obstacles": len(track["obstacles_pos"]),
                }
            )
            print(f"seed {seed}: {png_path}")
    finally:
        env.close()

    manifest_path = args.output_dir / "manifest.csv"
    write_manifest(manifest_path, manifest_rows)
    print(f"\nWrote {len(manifest_rows)} map(s) to {args.output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
