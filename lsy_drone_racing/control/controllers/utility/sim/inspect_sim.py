"""Interactive simulation runner with seed control and pre-flight track inspection.

A debugging-oriented variant of ``scripts/sim.py``. It adds two things on top of
the normal simulate loop:

1. **Seed control** -- you can pass an explicit ``seed`` to reproduce a track, or
   omit it to draw (and print) a fresh random seed for every new random sim. The
   seed that produced the current track is always printed so it can be reused
   (e.g. saved into ``references/funny_seeds.md``).

2. **Pre-flight map inspection** -- after the track is randomized but *before* the
   controller flies, the live MuJoCo viewer is opened so you can orbit/zoom the
   track, an automated obstacle<->gate clipping report is printed, and you decide
   from the terminal whether to fly, regenerate (new seed), jump to a specific
   seed, or quit. This catches the failure mode where a randomized obstacle clips
   into a gate aperture and makes the track infeasible.

Run (inside ``pixi shell``):

    python lsy_drone_racing/control/controllers/utility/sim/inspect_sim.py \
        --config level2.toml --seed 42

    # or, for a fresh random seed printed back to you:
    python lsy_drone_racing/control/controllers/utility/sim/inspect_sim.py --config level3.toml

Terminal commands during inspection:
    [Enter] / y   fly the current track with the controller
    r             regenerate the track with a new random seed
    s <int>       reset the track to a specific seed
    q             quit without flying
"""

from __future__ import annotations

import os

# crazyflow requires this flag to be set before scipy is imported (it is imported
# transitively below). Set it defensively so the module is import-order safe.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import logging  # noqa: E402
import select  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import fire  # noqa: E402
import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

from lsy_drone_racing.utils import load_config, load_controller  # noqa: E402

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)

# Repo root: .../lsy_drone_racing/control/controllers/utility/sim/inspect_sim.py
#            parents[0]=sim [1]=utility [2]=controllers [3]=control
#            [4]=lsy_drone_racing (package) [5]=repo root
REPO_ROOT = Path(__file__).resolve().parents[5]

# Geometry constants (see config TOML comments).
GATE_APERTURE = 0.4  # inner opening width/height (m)
GATE_OUTER = 0.72  # outer frame width (m)


def _new_seed() -> int:
    """Draw a fresh, printable, reproducible-elsewhere random seed."""
    return int(np.random.default_rng().integers(0, 2**31 - 1))


def _true_track(env: DroneRaceEnv, obs: dict) -> dict:
    """Return the *true* randomized track, not the sensor-masked observation.

    At reset the drone is far from every object, so ``obs`` reports the nominal
    (un-randomized) gate/obstacle poses for level >= 2. Inspecting that would hide
    the very randomization (e.g. an obstacle clipped into a gate) we want to see.
    The real poses live in the underlying env data; the viewer already draws them.

    The returned dict reuses the ``obs`` key names so it is a drop-in for the
    report/summary helpers.
    """
    d = env.unwrapped.data
    return {
        "pos": np.asarray(obs["pos"], dtype=float),
        "gates_pos": np.asarray(d.gates_pos)[0],
        "gates_quat": np.asarray(d.gates_quat)[0],
        "obstacles_pos": np.asarray(d.obstacles_pos)[0],
        "target_gate": obs.get("target_gate", 0),
    }


def clipping_report(
    track: dict,
    obstacle_radius: float = 0.18,
    drone_margin: float = 0.10,
) -> tuple[str, bool]:
    """Build a textual obstacle<->gate clipping report for the current track.

    For every (gate, obstacle) pair, the obstacle's vertical line is projected
    into the gate's local frame and the planar distance from the obstacle centre
    to the gate *aperture* line segment is measured. Obstacles are tall cylinders
    spanning the full gate height, so XY proximity to the opening is what matters.

    Args:
        track: Track dict with ``gates_pos``, ``gates_quat``, ``obstacles_pos``
            (use the *true* randomized poses, see :func:`_true_track`).
        obstacle_radius: Inflated obstacle radius used by the planner. An obstacle
            whose centre is within this distance of the opening leaves no safe
            corridor -> the track is treated as infeasible.
        drone_margin: Extra clearance below which a pair is flagged as TIGHT.

    Returns:
        ``(report_text, feasible)``. ``feasible`` is False if any obstacle blocks
        an aperture.
    """
    gates_pos = np.asarray(track["gates_pos"], dtype=float)
    gates_quat = np.asarray(track["gates_quat"], dtype=float)
    obstacles_pos = np.asarray(track["obstacles_pos"], dtype=float)

    half = GATE_APERTURE / 2.0
    lines = []
    feasible = True

    for gi, (gp, gq) in enumerate(zip(gates_pos, gates_quat)):
        forward = R.from_quat(gq).apply([1.0, 0.0, 0.0])[:2]
        n = np.linalg.norm(forward)
        forward = forward / n if n > 1e-9 else np.array([1.0, 0.0])
        lateral = np.array([-forward[1], forward[0]])  # +90 deg in XY

        for oi, op in enumerate(obstacles_pos):
            rel = op[:2] - gp[:2]
            along = float(rel @ forward)  # distance out of the gate plane
            lat = float(rel @ lateral)  # offset across the opening
            # Distance from obstacle centre to the aperture segment in XY.
            lat_clamped = float(np.clip(lat, -half, half))
            dist = float(np.hypot(along, lat - lat_clamped))
            clearance = dist - obstacle_radius

            if dist < obstacle_radius:
                tag = "BLOCKS APERTURE -> INFEASIBLE"
                feasible = False
            elif clearance < drone_margin:
                tag = "TIGHT"
            elif dist < GATE_OUTER / 2.0 + obstacle_radius:
                tag = "near frame"
            else:
                tag = "ok"

            if tag != "ok":
                lines.append(
                    f"  gate {gi} <-> obstacle {oi}: "
                    f"aperture_dist={dist:.3f} m, clearance={clearance:+.3f} m "
                    f"(along={along:+.3f}, lateral={lat:+.3f})  [{tag}]"
                )

    header = "Obstacle/gate clipping report:"
    if not lines:
        body = "  all obstacles clear of every gate aperture."
    else:
        body = "\n".join(lines)
    verdict = "FEASIBLE" if feasible else "INFEASIBLE (an obstacle blocks a gate)"
    return f"{header}\n{body}\n  verdict: {verdict}", feasible


def _print_track_summary(track: dict, seed: int) -> None:
    """Print seed, gate poses, obstacle positions and the clipping report."""
    gates_pos = np.asarray(track["gates_pos"], dtype=float)
    gates_quat = np.asarray(track["gates_quat"], dtype=float)
    obstacles_pos = np.asarray(track["obstacles_pos"], dtype=float)

    print("\n" + "=" * 70)
    print(f"TRACK PREVIEW  |  seed = {seed}")
    print("=" * 70)
    print(f"start pos: {np.asarray(track['pos'], dtype=float).round(3)}")
    print("gates (pos | yaw deg):")
    for gi, (gp, gq) in enumerate(zip(gates_pos, gates_quat)):
        yaw = np.degrees(R.from_quat(gq).as_euler("xyz")[2])
        print(f"  [{gi}] pos={gp.round(3)}  yaw={yaw:+6.1f}")
    print("obstacles (pos):")
    for oi, op in enumerate(obstacles_pos):
        print(f"  [{oi}] pos={op.round(3)}")
    report, _ = clipping_report(track)
    print(report)
    print("=" * 70)


def _read_command_nonblocking() -> str | None:
    """Return a stripped line from stdin if one is ready, else None (Linux)."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        return sys.stdin.readline().strip()
    return None


def _inspect_loop(
    env: DroneRaceEnv,
    controller: Controller | None,
    render: bool,
) -> str:
    """Block on the live viewer until the user issues a command.

    Returns one of: ``"fly"``, ``"regen"``, ``"quit"``, or ``"seed:<int>"``.
    """
    print(
        "\nInspect the track in the viewer. Commands (type here + Enter):\n"
        "  [Enter]/y  fly      r  new random seed      s <int>  set seed      q  quit"
    )
    if not render:
        # Headless: no viewer to orbit, just ask once.
        cmd = input("> ").strip()
        return _parse_command(cmd)

    while True:
        if controller is not None:
            try:
                controller.render_callback(env.unwrapped.sim)
            except Exception as exc:  # noqa: BLE001 - preview must never crash inspect
                if not getattr(_inspect_loop, "_warned", False):
                    print(f"(path preview unavailable: {exc})")
                    _inspect_loop._warned = True
        env.render()

        cmd = _read_command_nonblocking()
        if cmd is not None:
            return _parse_command(cmd)
        time.sleep(1 / 60)


def _parse_command(cmd: str) -> str:
    """Map a raw terminal command to an inspect-loop action token."""
    low = cmd.lower()
    if low in ("", "y", "yes", "fly", "f"):
        return "fly"
    if low in ("r", "regen", "regenerate"):
        return "regen"
    if low in ("q", "quit", "exit"):
        return "quit"
    if low.startswith("s"):
        parts = cmd.split()
        if len(parts) >= 2:
            try:
                return f"seed:{int(parts[1])}"
            except ValueError:
                pass
        print("  usage: s <integer seed>")
        return "stay"
    print(f"  unknown command: {cmd!r}")
    return "stay"


def _fly(
    env: DroneRaceEnv,
    controller: Controller,
    config: ConfigDict,
    obs: dict,
    info: dict,
) -> float | None:
    """Run one episode with the controller, mirroring scripts/sim.py."""
    i = 0
    fps = 60
    curr_time = 0.0
    while True:
        curr_time = i / config.env.freq
        action = controller.compute_control(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        controller_finished = controller.step_callback(
            action, obs, reward, terminated, truncated, info
        )
        if terminated or truncated or controller_finished:
            break
        if config.sim.render and ((i * fps) % config.env.freq) < fps:
            controller.render_callback(env.unwrapped.sim)
            env.render()
        i += 1

    controller.episode_callback()
    log_episode_stats(obs, info, config, curr_time)
    controller.episode_reset()
    return curr_time if obs["target_gate"] == -1 else None


def simulate(
    config: str = "level2.toml",
    controller: str | None = None,
    seed: int | None = None,
    n_runs: int = 1,
    inspect: bool = True,
    preview_path: bool = True,
) -> list[float | None]:
    """Run the controller with seed control and pre-flight track inspection.

    Args:
        config: Config file name inside ``config/`` (e.g. ``level2.toml``).
        controller: Controller file name in ``lsy_drone_racing/control/`` or None
            to use the one named in the config.
        seed: Explicit seed for a reproducible track. None draws a fresh random
            seed per run (and prints it).
        n_runs: Number of episodes to run.
        inspect: If True, open the viewer and prompt before each flight. If False,
            behave like ``scripts/sim.py`` (just print the seed + clipping report).
        preview_path: If True, build the controller before inspection so its
            planned path/trajectory is drawn over the track during inspection.

    Returns:
        Per-run finish times (None for runs that did not finish / were skipped).
    """
    config = load_config(REPO_ROOT / "config" / config)
    render = bool(config.sim.render)
    if inspect:
        # Inspection needs the live viewer; force render on for it.
        config.sim.render = True
        render = True

    control_path = REPO_ROOT / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)

    env: DroneRaceEnv = gymnasium.make(
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
    env = JaxToNumpy(env)

    ep_times: list[float | None] = []
    run = 0
    while run < n_runs:
        run_seed = seed if seed is not None else _new_seed()

        # Inspection lets the user flip through seeds before committing to a run.
        while True:
            obs, info = env.reset(seed=run_seed)
            # Expose the seed to the controller so it tags its flight log.
            config.env.seed = run_seed
            track = _true_track(env, obs)
            _print_track_summary(track, run_seed)

            controller_obj: Controller | None = None
            if preview_path or not inspect:
                try:
                    controller_obj = controller_cls(obs, info, config)
                except Exception as exc:  # noqa: BLE001
                    print(f"(controller build failed for this track: {exc})")
                    controller_obj = None

            if not inspect:
                break

            action = _inspect_loop(env, controller_obj, render)
            if action == "fly":
                break
            if action == "quit":
                print("Quit requested; closing.")
                env.close()
                return ep_times
            if action == "regen":
                seed = None  # subsequent regen also random
                run_seed = _new_seed()
                continue
            if action.startswith("seed:"):
                run_seed = int(action.split(":", 1)[1])
                continue
            # "stay": redraw same track
            continue

        if controller_obj is None:
            controller_obj = controller_cls(obs, info, config)

        print(f"\nFlying seed {run_seed} (run {run + 1}/{n_runs})...")
        ep_times.append(_fly(env, controller_obj, config, obs, info))
        run += 1

    env.close()
    print("\nFinish times:", ep_times)
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode (matches scripts/sim.py)."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
