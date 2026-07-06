"""Measure controller ``compute_control`` latency (real-time / control-rate check).

Runs a controller over one or more seeds, times every ``compute_control`` call,
and reports the latency distribution in milliseconds against the control period
(taken from ``config.env.freq`` -- e.g. 50 Hz -> 20 ms). To fly on real hardware
each control step must finish within that period.

The report separates:
  * **steady-state** steps (the typical per-tick cost the loop must sustain), and
  * **spikes** -- the first call per seed (warmup) and mid-episode **replans**,
    which regenerate the path/trajectory inside ``compute_control`` and dominate
    the tail. They are reported but excluded from the steady-state stats, since on
    hardware a replan is usually run off the hot loop / at a lower rate.

Run (inside ``pixi shell``):

    python .../time_controller.py --controller controllers/controller_mppi.py
    python .../time_controller.py --controller controllers/controller_mpc_astar.py --seeds '[1,2,3]'
"""

from __future__ import annotations

import contextlib
import io
import os

# crazyflow requires this before scipy is imported (see inspect_sim).
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import sys  # noqa: E402
import time as _time  # noqa: E402

import fire  # noqa: E402
import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402

from lsy_drone_racing.control.controllers.utility.sim.inspect_sim import REPO_ROOT  # noqa: E402
from lsy_drone_racing.utils import load_config, load_controller  # noqa: E402


def _quiet(verbose: bool) -> contextlib.AbstractContextManager:
    """Context manager that swallows stdout unless ``verbose``."""
    return contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())


def _time_seed(
    env: object,
    controller_cls: type,
    cfg: object,
    seed: int,
    verbose: bool,
    live: bool = True,
    live_out: object | None = None,
) -> np.ndarray:
    """Run one seeded episode, return per-step ``compute_control`` latency (ms).

    When ``live`` is set, prints the average latency over each simulated second
    (a window of ``config.env.freq`` steps) to ``live_out`` -- the real stdout,
    captured before the controller-silencing redirect below.
    """
    obs, info = env.reset(seed=seed)
    cfg.env.seed = seed
    steps_per_sec = max(1, int(round(float(cfg.env.freq))))
    with _quiet(verbose):
        controller = controller_cls(obs, info, cfg)

    latencies = []
    with _quiet(verbose):
        while True:
            t0 = _time.perf_counter()
            action = controller.compute_control(obs, info)
            latencies.append((_time.perf_counter() - t0) * 1000.0)

            # Once per simulated second, print the average over that second.
            if live and len(latencies) % steps_per_sec == 0:
                window = latencies[-steps_per_sec:]
                avg = sum(window) / len(window)
                worst = max(window)
                sec = len(latencies) // steps_per_sec
                print(
                    f"[seed {seed} | sim {sec:>3d}s] avg {avg:6.2f} ms "
                    f"({1000.0 / avg:6.1f} Hz)  max {worst:6.2f} ms  "
                    f"over last {steps_per_sec} steps",
                    file=live_out,
                    flush=True,
                )

            obs, reward, terminated, truncated, info = env.step(action)
            finished = controller.step_callback(action, obs, reward, terminated, truncated, info)
            if terminated or truncated or finished:
                break
        controller.episode_callback()
        controller.episode_reset()
    return np.asarray(latencies, dtype=float)


def time_controller(
    config: str = "level3.toml",
    controller: str | None = None,
    seeds: list[int] | None = None,
    seed: int = 2,
    spike_factor: float = 2.0,
    live: bool = True,
    verbose: bool = False,
) -> dict:
    """Time ``compute_control`` and report latency vs the control-rate budget.

    Args:
        config: Config file in ``config/`` (default ``level3.toml``).
        controller: Controller file under ``lsy_drone_racing/control/`` or None to
            use the one named in the config.
        seeds: List of seeds to average over. None uses ``[seed]``.
        seed: Single seed used when ``seeds`` is None.
        spike_factor: Steps slower than ``spike_factor x`` the median are treated
            as spikes (replans) and split out of the steady-state stats.
        live: If True (default), print the average latency once per simulated
            second (a window of ``config.env.freq`` steps) as the run proceeds.
        verbose: Let the controller print during the run.

    Returns:
        Dict of the key steady-state metrics (ms) and the real-time verdict.
    """
    seeds = [seed] if seeds is None else list(seeds)
    cfg = load_config(REPO_ROOT / "config" / config)
    cfg.sim.render = False

    control_path = REPO_ROOT / "lsy_drone_racing/control"
    controller_path = control_path / (controller or cfg.controller.file)
    controller_cls = load_controller(controller_path)

    env = JaxToNumpy(
        gymnasium.make(
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
    )

    live_out = sys.stdout  # captured before the controller-silencing redirect
    per_seed = []
    for s in seeds:
        lat = _time_seed(env, controller_cls, cfg, s, verbose, live=live, live_out=live_out)
        if len(lat):
            per_seed.append(lat)
    env.close()

    if not per_seed:
        raise RuntimeError("No steps were timed.")

    all_lat = np.concatenate(per_seed)
    first_calls = np.array([lat[0] for lat in per_seed])
    # Drop the first call of each seed (warmup), then split spikes (replans).
    body = np.concatenate([lat[1:] for lat in per_seed if len(lat) > 1])
    median = float(np.median(body))
    spike_thresh = spike_factor * median
    steady = body[body <= spike_thresh]
    spikes = body[body > spike_thresh]

    budget = 1000.0 / float(cfg.env.freq)
    p95 = float(np.percentile(steady, 95))
    p99 = float(np.percentile(steady, 99))

    if steady.max() <= budget:
        verdict = "PASS -- every steady step within budget"
    elif p95 <= budget:
        verdict = "PASS (typical) -- p95 within budget; check spikes below"
    elif median <= budget:
        verdict = "MARGINAL -- median within budget but p95 over"
    else:
        verdict = "FAIL -- median over budget"

    line = "=" * 72
    print(line)
    print(f"compute_control latency   controller={controller or cfg.controller.file}")
    print(f"seeds={seeds}   budget={budget:.1f} ms ({float(cfg.env.freq):.0f} Hz control rate)")
    print(line)
    print(f"steady-state steps: {len(steady)}   (of {len(all_lat)} total)")
    print(f"  mean    {steady.mean():7.2f} ms    ->  {1000.0 / steady.mean():6.1f} Hz sustainable")
    print(f"  median  {median:7.2f} ms    ->  {1000.0 / median:6.1f} Hz")
    print(f"  p95     {p95:7.2f} ms")
    print(f"  p99     {p99:7.2f} ms")
    print(f"  max     {steady.max():7.2f} ms")
    print("-" * 72)
    print(
        f"warmup (first call/seed): mean {first_calls.mean():.1f} ms, "
        f"max {first_calls.max():.1f} ms"
    )
    if len(spikes):
        print(
            f"replan spikes (> {spike_thresh:.1f} ms): {len(spikes)} steps, "
            f"median {np.median(spikes):.1f} ms, max {spikes.max():.1f} ms"
        )
    else:
        print("replan spikes: none")
    print("-" * 72)
    over = int(np.sum(steady > budget))
    print(f"Real-time @ {float(cfg.env.freq):.0f} Hz: {verdict}")
    print(f"  steady steps over budget: {over}/{len(steady)}")
    print(line)

    return {
        "median_ms": median,
        "mean_ms": float(steady.mean()),
        "p95_ms": p95,
        "max_ms": float(steady.max()),
        "budget_ms": budget,
        "n_spikes": int(len(spikes)),
        "verdict": verdict,
    }


if __name__ == "__main__":
    fire.Fire(time_controller)
