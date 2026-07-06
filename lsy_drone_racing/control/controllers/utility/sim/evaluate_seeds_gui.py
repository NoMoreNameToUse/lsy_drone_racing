"""GUI batch evaluation of the controller over N random seeds, with a report.

Same report machinery as ``evaluate_seeds`` (pass rate, speeds, per-failure
breakdown, infeasible-track detection), but:

* draws ``n_seeds`` random seeds from a *seeded* RNG so the run is
  reproducible (override the master seed with ``--rng_seed``), and
* keeps the live MuJoCo viewer open so you can watch every flight.

Run (inside ``pixi shell``):

    python lsy_drone_racing/control/controllers/utility/sim/evaluate_seeds_gui.py \
        --config level3.toml

    # different sample size / master seed / controller:
    python .../evaluate_seeds_gui.py --n_seeds 20 --rng_seed 7
    python .../evaluate_seeds_gui.py --controller controllers/controller_rl.py

The report is written to ``debug_outputs/seed_eval_gui_report.{txt,json}``.
"""

from __future__ import annotations

import json
import logging
import os

# crazyflow requires this before scipy is imported (see inspect_sim).
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import time as _time  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import fire  # noqa: E402
import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402

from lsy_drone_racing.control.controllers.utility.sim.evaluate_seeds import (  # noqa: E402
    _build_report,
    _classify_failure,
    _quiet,
)
from lsy_drone_racing.control.controllers.utility.sim.inspect_sim import (  # noqa: E402
    REPO_ROOT,
    _true_track,
    clipping_report,
)
from lsy_drone_racing.utils import load_config, load_controller  # noqa: E402

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv

logger = logging.getLogger(__name__)


def _run_one(
    env: DroneRaceEnv,
    controller_cls: type[Controller],
    config: ConfigDict,
    seed: int,
    safety_low: np.ndarray,
    safety_high: np.ndarray,
    verbose: bool,
    render: bool,
) -> dict:
    """Run a single seeded episode (optionally rendered) and return its metrics."""
    obs, info = env.reset(seed=seed)
    track = _true_track(env, obs)
    _, feasible = clipping_report(track)

    # Expose the seed to the controller so it tags its flight log (flight_seedN.npz).
    config.env.seed = seed

    with _quiet(verbose):
        controller: Controller = controller_cls(obs, info, config)

    n_gates = len(config.env.track.gates)
    freq = config.env.freq

    last_pos = np.asarray(obs["pos"], dtype=float)
    path_len = 0.0
    max_speed = 0.0
    i = 0
    terminated = truncated = controller_finished = False

    with _quiet(verbose):
        while True:
            cur_pos = np.asarray(obs["pos"], dtype=float)
            cur_vel = np.asarray(obs["vel"], dtype=float)
            max_speed = max(max_speed, float(np.linalg.norm(cur_vel)))

            action = controller.compute_control(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )

            if render:
                try:
                    controller.render_callback(env.unwrapped.sim)
                except Exception:  # noqa: BLE001 - preview must never crash eval
                    pass
                env.render()

            new_pos = np.asarray(obs["pos"], dtype=float)
            # Disabled drones are warped below ground; only count valid motion.
            if np.all(np.abs(new_pos) < 10.0) and new_pos[2] > -0.5:
                path_len += float(np.linalg.norm(new_pos - cur_pos))
                last_pos = cur_pos
            i += 1
            if terminated or truncated or controller_finished:
                break

        controller.episode_callback()
        controller.episode_reset()

    final_gate = int(np.asarray(obs["target_gate"]).item())
    passed = final_gate == -1
    gates_passed = n_gates if passed else final_gate
    curr_time = i / freq
    avg_speed = path_len / curr_time if curr_time > 0 else 0.0

    reason = _classify_failure(
        passed=passed,
        truncated=truncated,
        terminated=terminated and not passed,
        controller_finished=controller_finished,
        final_gate=final_gate,
        last_pos=last_pos,
        track=track,
        safety_low=safety_low,
        safety_high=safety_high,
    )

    return {
        "seed": seed,
        "passed": passed,
        "gates_passed": gates_passed,
        "n_gates": n_gates,
        "time_s": round(curr_time, 3),
        "avg_speed": round(avg_speed, 3),
        "max_speed": round(max_speed, 3),
        "path_len": round(path_len, 3),
        "feasible_track": feasible,
        "reason": reason,
        "fail_gate": None if passed else gates_passed,
    }


def evaluate(
    config: str = "level3.toml",
    controller: str | None = None,
    n_seeds: int = 50,
    rng_seed: int = 0,
    verbose: bool = False,
    render: bool = True,
    out: str | None = "debug_outputs/seed_eval_gui_report",
) -> dict:
    """Evaluate the controller across ``n_seeds`` random seeds with the GUI on.

    Args:
        config: Config file name inside ``config/`` (default ``level3.toml``).
        controller: Controller file in ``lsy_drone_racing/control/`` or None to
            use the one named in the config.
        n_seeds: Number of random episode seeds to draw (default 50).
        rng_seed: Master seed for the RNG that draws the episode seeds, so the
            whole evaluation is reproducible (default 0).
        verbose: If True, let the controller print during runs (noisy).
        render: If True (default), keep the live MuJoCo viewer open.
        out: Output path stem for the ``.txt`` / ``.json`` report, or None to
            skip writing files.

    Returns:
        Dict with ``results`` (per-seed) and ``report`` (text).
    """
    rng = np.random.default_rng(rng_seed)
    # Draw distinct episode seeds in the same 32-bit range the env expects.
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=n_seeds)]

    cfg = load_config(REPO_ROOT / "config" / config)
    cfg.sim.render = bool(render)

    control_path = REPO_ROOT / "lsy_drone_racing/control"
    controller_path = control_path / (controller or cfg.controller.file)
    controller_cls = load_controller(controller_path)

    env: DroneRaceEnv = gymnasium.make(
        cfg.env.id,
        freq=cfg.env.freq,
        sim_config=cfg.sim,
        sensor_range=cfg.env.sensor_range,
        control_mode=cfg.env.control_mode,
        track=cfg.env.track,
        disturbances=cfg.env.get("disturbances"),
        randomizations=cfg.env.get("randomizations"),
        seed=cfg.env.seed,
    )
    env = JaxToNumpy(env)

    safety_low = np.asarray(cfg.env.track.safety_limits.pos_limit_low, dtype=float)
    safety_high = np.asarray(cfg.env.track.safety_limits.pos_limit_high, dtype=float)

    results = []
    t0 = _time.perf_counter()
    for k, seed in enumerate(seeds):
        r = _run_one(env, controller_cls, cfg, seed, safety_low, safety_high, verbose, bool(render))
        results.append(r)
        status = "PASS" if r["passed"] else f"FAIL@g{r['fail_gate']}"
        print(
            f"[{k + 1:>2}/{len(seeds)}] seed {seed:>11}  {status:<8} "
            f"t={r['time_s']:>5.2f}s  vmax={r['max_speed']:.2f}  -> {r['reason']}"
        )
    env.close()
    elapsed = _time.perf_counter() - t0

    report = _build_report(results, f"{config} (rng_seed={rng_seed}, n={n_seeds})")
    report += f"\nWall-clock eval time: {elapsed:.1f} s for {len(seeds)} seeds"
    print("\n" + report)

    if out is not None:
        out_path = REPO_ROOT / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.with_suffix(".txt").write_text(report + "\n")
        out_path.with_suffix(".json").write_text(
            json.dumps(
                {"config": config, "rng_seed": rng_seed, "seeds": seeds, "results": results},
                indent=2,
            )
        )
        print(f"\nSaved report to {out_path.with_suffix('.txt')} and .json")

    return {"results": results, "report": report}


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.WARNING)
    fire.Fire(evaluate, serialize=lambda _: None)
