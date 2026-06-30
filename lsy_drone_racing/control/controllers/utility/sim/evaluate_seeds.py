"""Batch evaluation of the controller over a fixed list of seeds, with a report.

Runs the controller headlessly across a curated seed list (the "funny seeds")
and prints + saves a debugging report covering:

* Total passed / pass rate
* Median speed (and median finish time) over completed runs
* For every failed run: which gate it failed on and why (collision vs out of
  bounds vs timeout vs trajectory ran out early)
* Extra debugging metrics: max speed, distance flown, time survived, and whether
  the *track itself* was infeasible (an obstacle clipped into a gate aperture),
  which directly explains otherwise-mysterious failures.

Run (inside ``pixi shell``):

    python lsy_drone_racing/control/controllers/utility/sim/evaluate_seeds.py \
        --config level3.toml

    # custom seeds / controller / verbose controller logs:
    python .../evaluate_seeds.py --config level3.toml --controller controllers/controller_rl.py
    python .../evaluate_seeds.py --seeds '[4, 10, 1234]' --verbose True

The report is written to ``debug_outputs/seed_eval_report.{txt,json}``.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os

# crazyflow requires this before scipy is imported (see inspect_sim).
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import statistics  # noqa: E402
import time as _time  # noqa: E402
from datetime import datetime  # noqa: E402
from typing import TYPE_CHECKING  # noqa: E402

import fire  # noqa: E402
import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402

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

# Curated "funny" seeds (level 3). Override with --seeds on the CLI.
DEFAULT_SEEDS = [
    4, 10, 537525082, 548046238, 437125220, 334327978, 1898163101, 380878779,
    368113776, 1522539070, 394335597, 2070717517, 1756798599, 601977100,
    447542283, 1943230833, 1008680058, 992857546, 1674664546, 223246164,
    1243516087, 168155228, 1015453627, 502743744, 1749845046, 1609583425,
    826175110, 1450051549, 780252868, 1345132680, 350327268, 673638594,
    196241771, 191876941, 848575542, 72996710,
]


def _quiet(verbose: bool) -> contextlib.AbstractContextManager:
    """Fresh context manager that swallows stdout unless ``verbose``."""
    return contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())


def _nearest_object(pos: np.ndarray, track: dict) -> tuple[str, float]:
    """Return (label, distance) of the gate/obstacle nearest to ``pos``.

    Gates use full 3D distance; obstacles (vertical poles) use XY distance.
    """
    best_label, best_dist = "none", np.inf
    for i, g in enumerate(np.asarray(track["gates_pos"], dtype=float)):
        d = float(np.linalg.norm(pos - g))
        if d < best_dist:
            best_label, best_dist = f"gate {i}", d
    for i, o in enumerate(np.asarray(track["obstacles_pos"], dtype=float)):
        d = float(np.linalg.norm(pos[:2] - o[:2]))
        if d < best_dist:
            best_label, best_dist = f"obstacle {i}", d
    return best_label, best_dist


def _classify_failure(
    *,
    passed: bool,
    truncated: bool,
    terminated: bool,
    controller_finished: bool,
    final_gate: int,
    last_pos: np.ndarray,
    track: dict,
    safety_low: np.ndarray,
    safety_high: np.ndarray,
) -> str:
    """Human-readable reason for the run outcome, inferred from final state.

    The env's ``info`` is empty and ``terminated`` conflates success, collision
    and out-of-bounds; we disambiguate from the last valid pose and flags.
    """
    if passed:
        return "finished"
    if truncated:
        return f"timeout: still targeting gate {final_gate} after the time limit"
    if terminated:
        below = last_pos < safety_low
        above = last_pos > safety_high
        if below.any() or above.any():
            axis = "xyz"
            viol = [
                f"{axis[i]}={last_pos[i]:.2f}"
                for i in range(3)
                if below[i] or above[i]
            ]
            return f"out of bounds ({', '.join(viol)}) while targeting gate {final_gate}"
        label, dist = _nearest_object(last_pos, track)
        return f"collision near {label} (d={dist:.2f} m) while targeting gate {final_gate}"
    if controller_finished:
        return f"controller ended trajectory early at gate {final_gate} (never reached the finish)"
    return f"unknown termination at gate {final_gate}"


def _run_one(
    env: DroneRaceEnv,
    controller_cls: type[Controller],
    config: ConfigDict,
    seed: int,
    safety_low: np.ndarray,
    safety_high: np.ndarray,
    verbose: bool,
) -> dict:
    """Run a single seeded episode and return its metrics dict."""
    obs, info = env.reset(seed=seed)
    track = _true_track(env, obs)
    _, feasible = clipping_report(track)

    # Expose the seed to the controller so it tags its flight log (flight_seedN.npz).
    config.env.seed = seed

    # Build the controller (and silence its constructor chatter unless verbose).
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

            new_pos = np.asarray(obs["pos"], dtype=float)
            # Disabled drones are warped below ground; only count valid motion.
            if np.all(np.abs(new_pos) < 10.0) and new_pos[2] > -0.5:
                path_len += float(np.linalg.norm(new_pos - cur_pos))
                last_pos = cur_pos  # last pose the controller actually saw
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


def _build_report(results: list[dict], config_label: str) -> str:
    """Assemble the human-readable report string from per-seed results."""
    n = len(results)
    passed = [r for r in results if r["passed"]]
    failed = [r for r in results if not r["passed"]]

    pass_times = [r["time_s"] for r in passed]
    pass_speeds = [r["avg_speed"] for r in passed]
    all_speeds = [r["avg_speed"] for r in results]

    def _median(xs: list[float]) -> float | None:
        return round(statistics.median(xs), 3) if xs else None

    lines = []
    lines.append("=" * 78)
    lines.append("SEED EVALUATION REPORT")
    stamp = datetime.now().isoformat(timespec="seconds")
    lines.append(f"config={config_label}  seeds={n}  generated={stamp}")
    lines.append("=" * 78)
    lines.append(f"Total passed     : {len(passed)} / {n}  ({100.0 * len(passed) / n:.1f}%)")
    lines.append(f"Median speed     : {_median(pass_speeds)} m/s (passed runs)   "
                 f"| all runs: {_median(all_speeds)} m/s")
    lines.append(f"Median finish    : {_median(pass_times)} s (passed runs)")
    if pass_times:
        lines.append(f"Fastest / slowest finish: {min(pass_times):.2f} s / {max(pass_times):.2f} s")

    # Failure breakdown by gate and by category.
    if failed:
        by_gate: dict[int, int] = {}
        cat = {"collision": 0, "out of bounds": 0, "timeout": 0, "early": 0, "other": 0}
        infeasible_fails = 0
        for r in failed:
            by_gate[r["fail_gate"]] = by_gate.get(r["fail_gate"], 0) + 1
            rs = r["reason"]
            if rs.startswith("collision"):
                cat["collision"] += 1
            elif rs.startswith("out of bounds"):
                cat["out of bounds"] += 1
            elif rs.startswith("timeout"):
                cat["timeout"] += 1
            elif rs.startswith("controller ended"):
                cat["early"] += 1
            else:
                cat["other"] += 1
            if not r["feasible_track"]:
                infeasible_fails += 1

        lines.append("")
        lines.append(f"Not-passed gate histogram (gate -> #fails): "
                     f"{ {k: by_gate[k] for k in sorted(by_gate)} }")
        lines.append(f"Failure categories: { {k: v for k, v in cat.items() if v} }")
        if infeasible_fails:
            lines.append(
                f"** {infeasible_fails} of {len(failed)} failures were on INFEASIBLE "
                f"tracks (obstacle clipped a gate) -- not a controller bug. **"
            )

    # Per-failure detail table.
    lines.append("")
    lines.append("-" * 78)
    lines.append("FAILED RUNS")
    lines.append("-" * 78)
    if not failed:
        lines.append("(none -- all seeds passed)")
    else:
        for r in sorted(failed, key=lambda x: (x["fail_gate"], x["seed"])):
            flag = "" if r["feasible_track"] else "  [INFEASIBLE TRACK]"
            lines.append(
                f"seed {r['seed']:>11}  gate {r['fail_gate']}/{r['n_gates']}  "
                f"t={r['time_s']:>5.2f}s  vmax={r['max_speed']:.2f}  "
                f"reason: {r['reason']}{flag}"
            )

    # Per-pass summary table (compact).
    lines.append("")
    lines.append("-" * 78)
    lines.append("PASSED RUNS (seed: time s @ avg m/s, vmax)")
    lines.append("-" * 78)
    if not passed:
        lines.append("(none)")
    else:
        for r in sorted(passed, key=lambda x: x["time_s"]):
            lines.append(
                f"seed {r['seed']:>11}  t={r['time_s']:>5.2f}s  "
                f"avg={r['avg_speed']:.2f} m/s  vmax={r['max_speed']:.2f}"
            )
    lines.append("=" * 78)
    return "\n".join(lines)


def evaluate(
    config: str = "level3.toml",
    controller: str | None = None,
    seeds: list[int] | None = None,
    verbose: bool = False,
    out: str | None = "debug_outputs/seed_eval_report",
) -> dict:
    """Evaluate the controller across a seed list and emit a debugging report.

    Args:
        config: Config file name inside ``config/`` (default ``level3.toml``).
        controller: Controller file in ``lsy_drone_racing/control/`` or None to
            use the one named in the config.
        seeds: List of integer seeds. None uses the curated ``DEFAULT_SEEDS``.
        verbose: If True, let the controller print during runs (noisy).
        out: Output path stem for the ``.txt`` / ``.json`` report, or None to
            skip writing files.

    Returns:
        Dict with ``results`` (per-seed) and ``report`` (text).
    """
    seeds = list(DEFAULT_SEEDS if seeds is None else seeds)
    cfg = load_config(REPO_ROOT / "config" / config)
    cfg.sim.render = False  # batch eval is always headless

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
        r = _run_one(env, controller_cls, cfg, seed, safety_low, safety_high, verbose)
        results.append(r)
        status = "PASS" if r["passed"] else f"FAIL@g{r['fail_gate']}"
        print(f"[{k + 1:>2}/{len(seeds)}] seed {seed:>11}  {status:<8} "
              f"t={r['time_s']:>5.2f}s  vmax={r['max_speed']:.2f}  -> {r['reason']}")
    env.close()
    elapsed = _time.perf_counter() - t0

    report = _build_report(results, config)
    report += f"\nWall-clock eval time: {elapsed:.1f} s for {len(seeds)} seeds"
    print("\n" + report)

    if out is not None:
        out_path = REPO_ROOT / out
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.with_suffix(".txt").write_text(report + "\n")
        out_path.with_suffix(".json").write_text(
            json.dumps({"config": config, "seeds": seeds, "results": results}, indent=2)
        )
        print(f"\nSaved report to {out_path.with_suffix('.txt')} and .json")

    return {"results": results, "report": report}


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.WARNING)
    fire.Fire(evaluate, serialize=lambda _: None)
