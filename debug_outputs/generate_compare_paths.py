"""Generate comparable planner debug files from one shared environment reset.

Run:
    SCIPY_ARRAY_API=1 pixi run python debug_outputs/generate_compare_paths.py \
        --config level2.toml --seed 1
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.control.controllers.modules.path_generator import make_path_generator
from lsy_drone_racing.control.controllers.modules.timing_module import DistanceTiming
from lsy_drone_racing.control.controllers.modules.trajectory_module import SplineTrajectory
from lsy_drone_racing.utils import load_config

PLANNERS = ("astar", "theta_star", "rrt_star", "d_star_lite", "curve_gate")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="level2.toml",
        help="Config file name inside config/, or a direct path to a toml file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional env seed override. Use this for repeatable maps.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where planner debug npz files are written.",
    )
    parser.add_argument(
        "--planner",
        action="append",
        choices=PLANNERS,
        help="Planner to generate. Can be passed multiple times. Defaults to all.",
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


def save_planner_debug(
    planner_name: str, obs: dict, config: Any, output_dir: Path
) -> tuple[Path, np.ndarray, SplineTrajectory]:
    """Generate and save one planner's path debug file."""
    planner = make_path_generator(planner_name)
    timing = DistanceTiming(nominal_speed=1.28, min_segment_time=0.123)

    waypoints = np.asarray(planner.generate(obs, config), dtype=float)
    t = timing.compute(waypoints)
    trajectory = SplineTrajectory(waypoints, t, config.env.freq)

    output_dir.mkdir(exist_ok=True)
    path = output_dir / f"{planner_name}_debug.npz"

    np.savez(
        path,
        planner_name=np.asarray(planner_name),
        env_seed=np.asarray(config.env.seed),
        gates_pos=np.asarray(obs["gates_pos"], dtype=float),
        gates_quat=np.asarray(obs["gates_quat"], dtype=float),
        obstacles_pos=np.asarray(obs["obstacles_pos"], dtype=float),
        start_pos=np.asarray(obs["pos"], dtype=float),
        raw_waypoints=waypoints,
        traj_pos=np.asarray(trajectory._pos, dtype=float),
        traj_vel=np.asarray(trajectory._vel, dtype=float),
        time_grid=np.asarray(getattr(trajectory, "_time_grid", []), dtype=float),
    )

    return path, waypoints, trajectory


def path_length(points: np.ndarray) -> float:
    """Return the polyline length of a point sequence."""
    if len(points) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def main() -> None:
    """Generate all requested planner debug files from one shared reset."""
    args = parse_args()
    config = load_config(resolve_config_path(args.config))
    if args.seed is not None:
        config.env.seed = args.seed
    elif int(config.env.seed) < 0:
        config.env.seed = int(np.random.default_rng().integers(1, 2**31 - 1))
    config.sim.render = False

    planners = tuple(args.planner) if args.planner else PLANNERS

    env = make_env(config)
    try:
        obs, _ = env.reset()

        print("Shared map:")
        print(f"  config={args.config}")
        print(f"  seed={config.env.seed}")
        print(f"  start_pos={np.asarray(obs['pos'], dtype=float)}")
        print(f"  gates={len(obs['gates_pos'])}, obstacles={len(obs['obstacles_pos'])}")

        print("\nGenerated planner debug files:")
        for planner_name in planners:
            path, waypoints, trajectory = save_planner_debug(
                planner_name,
                obs,
                config,
                args.output_dir,
            )
            print(
                f"  {planner_name}: {path}, "
                f"waypoints={len(waypoints)}, "
                f"raw_len={path_length(waypoints):.3f} m, "
                f"traj_samples={len(trajectory._pos)}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
