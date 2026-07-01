"""Systematic tuning of the attitude MPC (``controller_mpc_astar.AttitudeMPC``).

The MPC's tunable surface is exposed as ``controller_mpc_astar.MPC_HYPERPARAMS``
(state weights ``q_diag``, input weights ``r_diag``, horizon ``N``). This tool
searches that surface against a *continuous* score so an optimizer gets signal
even when the raw pass count plateaus, and it splits the seeds into train/val so
we can tell tuning from overfitting.

Score (higher is better), aggregated over a seed set::

    score =  gates_passed_fraction          # primary, dominates
           - 0.15 * mean_tracking_rmse       # setpoint-vs-actual (m), smooth
           - 0.05 * median_finish_norm       # speed tie-break (t / episode_len)
           - 0.30 * mean_clearance_deficit   # reward pole margin, not just survival

Tracking RMSE is the key smooth term: it measures the MPC's actual job (follow
the reference) and moves continuously with the weights, unlike pass/fail.

Build cache: acados regenerates C code per solver build (~3 s). Each distinct
``N`` gets its own codegen (see ``create_ocp_solver``), so we build once per N and
reuse the compiled solver across seeds/trials, updating the LINEAR_LS weight
matrix ``W`` at runtime (``cost_set``) instead of rebuilding. This turns a
~100-trial search from hours into minutes.

Run (inside ``pixi shell``)::

    # baseline horizon sweep (default weights)
    python .../tune_mpc.py sweep_n
    # Bayesian search, then validate the winner on held-out seeds
    python .../tune_mpc.py tune --n_trials 120
    # score a single explicit config
    python .../tune_mpc.py score_one --n 25
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os

# crazyflow requires this before scipy is imported (see inspect_sim).
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("MUJOCO_GL", "egl")

import statistics  # noqa: E402
import time as _time  # noqa: E402

import fire  # noqa: E402
import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402

import lsy_drone_racing.control.controllers.controller_mpc_astar as ctl  # noqa: E402
from lsy_drone_racing.control.controllers.utility.sim.evaluate_seeds import (  # noqa: E402
    DEFAULT_SEEDS,
)
from lsy_drone_racing.control.controllers.utility.sim.inspect_sim import (  # noqa: E402
    REPO_ROOT,
    _true_track,
    clipping_report,
)
from lsy_drone_racing.utils import load_config  # noqa: E402

logger = logging.getLogger(__name__)

# --- score weights (see module docstring) ---
W_RMSE = 0.15
W_TIME = 0.05
W_CLEAR = 0.30
CLEAR_TARGET = 0.25  # metres: pole-center distance we want to keep
EPISODE_LEN_S = 12.0  # rough episode cap, for normalizing finish time

# Default seed split: interleave so the hard "funny" seeds land in both sets.
_ALL = list(DEFAULT_SEEDS)
TRAIN_SEEDS = _ALL[0::2]
VAL_SEEDS = _ALL[1::2]

# The current committed defaults (baseline for the N sweep / generalization check).
DEFAULT_N = int(ctl.MPC_HYPERPARAMS["N"])
DEFAULT_Q = tuple(ctl.MPC_HYPERPARAMS["q_diag"])
DEFAULT_R = tuple(ctl.MPC_HYPERPARAMS["r_diag"])
DEFAULT_TIMING = dict(ctl.TIMING_HYPERPARAMS)


# ---------------------------------------------------------------------------
# acados build cache: one compiled solver per horizon N; weights set at runtime.
# ---------------------------------------------------------------------------
_real_create = ctl.create_ocp_solver
_SOLVER_CACHE: dict[int, tuple] = {}


def _cached_create(Tf, N, parameters, q_diag=None, r_diag=None, verbose=False):
    q = tuple(ctl.MPC_HYPERPARAMS["q_diag"] if q_diag is None else q_diag)
    r = tuple(ctl.MPC_HYPERPARAMS["r_diag"] if r_diag is None else r_diag)
    W = scipy.linalg.block_diag(np.diag(q), np.diag(r))
    W_e = np.diag(q)

    if N in _SOLVER_CACHE:
        solver, ocp = _SOLVER_CACHE[N]
        # Update LINEAR_LS weights at runtime -- no rebuild.
        for i in range(N):
            solver.cost_set(i, "W", W)
        solver.cost_set(N, "W", W_e)
        solver.reset()  # clear warm start carried over from the previous seed
        return solver, ocp

    solver, ocp = _real_create(Tf, N, parameters, q_diag=q, r_diag=r, verbose=verbose)
    _SOLVER_CACHE[N] = (solver, ocp)
    return solver, ocp


ctl.create_ocp_solver = _cached_create


# ---------------------------------------------------------------------------
# environment + single-seed runner (records tracking error + pole clearance)
# ---------------------------------------------------------------------------
def _make_env(cfg):
    env = gymnasium.make(
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
    return JaxToNumpy(env)


def _run_seed(env, cfg, seed) -> dict:
    """Run one seeded episode; return outcome + tracking RMSE + pole clearance."""
    obs, info = env.reset(seed=seed)
    track = _true_track(env, obs)
    _, feasible = clipping_report(track)
    cfg.env.seed = seed

    with contextlib.redirect_stdout(io.StringIO()):
        controller = ctl.AttitudeMPC(obs, info, cfg)

    n_gates = len(cfg.env.track.gates)
    freq = cfg.env.freq
    obstacles = np.asarray(track["obstacles_pos"], dtype=float)

    sq_err = 0.0
    n_steps = 0
    min_clear = np.inf
    i = 0
    terminated = truncated = finished = False

    with contextlib.redirect_stdout(io.StringIO()):
        while True:
            action = controller.compute_control(obs, info)

            # Tracking error: current setpoint vs actual pose (pre-step).
            traj = controller.trajectory
            k = min(controller._tick, len(traj._pos) - 1)
            sp = traj._pos[k]
            sq_err += float(np.sum((np.asarray(obs["pos"], dtype=float) - sp) ** 2))
            n_steps += 1

            if obstacles.size:
                d = np.linalg.norm(
                    obstacles[:, :2] - np.asarray(obs["pos"], dtype=float)[:2], axis=1
                ).min()
                min_clear = min(min_clear, float(d))

            obs, reward, terminated, truncated, info = env.step(action)
            finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            i += 1
            if terminated or truncated or finished:
                break

        controller.episode_callback()
        controller.episode_reset()

    final_gate = int(np.asarray(obs["target_gate"]).item())
    passed = final_gate == -1
    gates_passed = n_gates if passed else final_gate

    return {
        "seed": seed,
        "passed": passed,
        "gates_passed": gates_passed,
        "n_gates": n_gates,
        "time_s": i / freq,
        "rmse": (sq_err / max(n_steps, 1)) ** 0.5,
        "min_clear": None if not obstacles.size else min_clear,
        "feasible": feasible,
    }


def _score(results: list[dict]) -> tuple[float, dict]:
    """Continuous score + breakdown from per-seed results (see module docstring)."""
    gates_frac = float(np.mean([r["gates_passed"] / r["n_gates"] for r in results]))
    rmse = float(np.mean([r["rmse"] for r in results]))
    passed_times = [r["time_s"] for r in results if r["passed"]]
    t_norm = (statistics.median(passed_times) / EPISODE_LEN_S) if passed_times else 1.5
    clears = [max(0.0, CLEAR_TARGET - r["min_clear"]) for r in results if r["min_clear"] is not None]
    clear_deficit = float(np.mean(clears)) if clears else 0.0

    score = gates_frac - W_RMSE * rmse - W_TIME * t_norm - W_CLEAR * clear_deficit
    meta = {
        "score": round(score, 4),
        "npass": sum(r["passed"] for r in results),
        "n": len(results),
        "gates_frac": round(gates_frac, 3),
        "rmse": round(rmse, 4),
        "t_norm": round(t_norm, 3),
        "clear_deficit": round(clear_deficit, 4),
    }
    return score, meta


def _eval_config(env, cfg, seeds, N, q_diag, r_diag) -> tuple[float, dict, list]:
    """Set hyperparameters, run the seed set, return (score, meta, results)."""
    ctl.MPC_HYPERPARAMS = {"N": int(N), "q_diag": tuple(q_diag), "r_diag": tuple(r_diag)}
    results = [_run_seed(env, cfg, s) for s in seeds]
    score, meta = _score(results)
    return score, meta, results


def _load(config: str):
    cfg = load_config(REPO_ROOT / "config" / config)
    cfg.sim.render = False
    return cfg


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def sweep_n(config: str = "level3.toml", ns=(15, 18, 20, 22, 25, 28, 32), seeds=None):
    """Coordinate sweep over the horizon N with the default weights (baseline)."""
    cfg = _load(config)
    seeds = TRAIN_SEEDS if seeds is None else list(seeds)
    env = _make_env(cfg)
    print(f"N sweep on {len(seeds)} train seeds (default weights)\n")
    rows = []
    for N in ns:
        t0 = _time.perf_counter()
        score, meta, _ = _eval_config(env, cfg, seeds, N, DEFAULT_Q, DEFAULT_R)
        dt = _time.perf_counter() - t0
        rows.append((N, meta))
        print(f"  N={N:>3}  score={meta['score']:+.4f}  pass={meta['npass']:>2}/{meta['n']}  "
              f"gates={meta['gates_frac']:.3f}  rmse={meta['rmse']:.4f}  "
              f"clr_def={meta['clear_deficit']:.4f}  ({dt:.0f}s)")
    env.close()
    best = max(rows, key=lambda kv: kv[1]["score"])
    print(f"\nBest N by score: {best[0]}  {best[1]}")
    return rows


def sweep_timing(
    config: str = "level3.toml",
    param: str = "a_lat_max",
    values=(2.0, 2.5, 3.0, 3.5, 4.0, 5.0),
    seeds=None,
    full: bool = True,
):
    """One-knob sensitivity sweep over a DynamicTiming param (MPC weights fixed).

    Holds the committed MPC weights constant and varies a single timing
    hyperparameter (e.g. ``a_lat_max``, the corner-speed cap), printing the
    pass/score/RMSE curve. ``full`` uses all seeds (low variance, matches the
    tuning protocol); else the train split.
    """
    cfg = _load(config)
    if seeds is not None:
        seeds = list(seeds)
    elif full:
        seeds = list(_ALL)
    else:
        seeds = TRAIN_SEEDS
    env = _make_env(cfg)

    # Pin MPC weights to the committed default for the whole sweep.
    ctl.MPC_HYPERPARAMS = {"N": DEFAULT_N, "q_diag": DEFAULT_Q, "r_diag": DEFAULT_R}
    base_timing = dict(DEFAULT_TIMING)
    base_val = base_timing.get(param)

    print(f"{param} sweep on {len(seeds)} seeds "
          f"(committed default {param}={base_val}; MPC weights fixed)\n")
    rows = []
    for v in values:
        ctl.TIMING_HYPERPARAMS = {**base_timing, param: v}
        t0 = _time.perf_counter()
        results = [_run_seed(env, cfg, s) for s in seeds]
        score, meta = _score(results)
        dt = _time.perf_counter() - t0
        rows.append((v, meta))
        mark = "  <- current" if base_val is not None and abs(v - base_val) < 1e-9 else ""
        print(f"  {param}={v:<5}  score={meta['score']:+.4f}  pass={meta['npass']:>2}/{meta['n']}  "
              f"gates={meta['gates_frac']:.3f}  rmse={meta['rmse']:.4f}  "
              f"t_norm={meta['t_norm']:.3f}  ({dt:.0f}s){mark}")
    ctl.TIMING_HYPERPARAMS = base_timing
    env.close()
    best = max(rows, key=lambda kv: kv[1]["score"])
    print(f"\nBest {param} by score: {best[0]}  {best[1]}")
    return rows


def score_one(config: str = "level3.toml", n: int = 25, q=None, r=None, seeds=None):
    """Score one explicit config on the given (default: train) seeds."""
    cfg = _load(config)
    seeds = TRAIN_SEEDS if seeds is None else list(seeds)
    env = _make_env(cfg)
    score, meta, results = _eval_config(
        env, cfg, seeds, n, DEFAULT_Q if q is None else q, DEFAULT_R if r is None else r
    )
    env.close()
    print(f"score={score:+.4f}  {meta}")
    for rr in results:
        tag = "P" if rr["passed"] else f"g{rr['gates_passed']}"
        print(f"    seed {rr['seed']:>11} {tag:>3}  t={rr['time_s']:5.2f}  rmse={rr['rmse']:.3f}")
    return meta


def tune(
    config: str = "level3.toml",
    n_trials: int = 120,
    train=None,
    val=None,
    full: bool = False,
    out: str = "debug_outputs/mpc_tune",
):
    """Bayesian (Optuna TPE) search over weights + horizon, then validate.

    Search space (log-scale weights): pos-xy, pos-z, vel, attitude (rpy), body
    rates (drpy), thrust-input; horizon N categorical. The rpy input weight is
    fixed at 1.0 as the reference scale.

    ``full=True`` tunes on ALL seeds (train == val == the whole curated set). Use
    this when per-seed variance makes a 25-seed subset too noisy to tune on (a
    6-seed split swing was observed at v_max=4); the winner must then be confirmed
    on the independent ``evaluate_seeds`` path before wiring, since there is no
    held-out set.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = _load(config)
    if full:
        train = list(_ALL)
        val = list(_ALL)
    else:
        train = TRAIN_SEEDS if train is None else list(train)
        val = VAL_SEEDS if val is None else list(val)
    env = _make_env(cfg)

    def objective(trial):
        # Wider space for the faster (v_max=4, a_max=4) regime: higher-N horizons
        # for lookahead at speed, and looser lower bounds on attitude/rate/vel
        # (the v_max=3 winner sat near att=0.18/rate=0.31/vel=1.3).
        N = trial.suggest_categorical("N", [18, 20, 22, 25, 28, 32, 36])
        pos_xy = trial.suggest_float("pos_xy", 5.0, 2000.0, log=True)
        pos_z = trial.suggest_float("pos_z", 50.0, 3000.0, log=True)
        vel = trial.suggest_float("vel", 0.2, 200.0, log=True)
        att = trial.suggest_float("att", 0.02, 20.0, log=True)
        rate = trial.suggest_float("rate", 0.02, 20.0, log=True)
        thrust = trial.suggest_float("thrust_R", 5.0, 300.0, log=True)
        q = (pos_xy, pos_xy, pos_z, att, att, att, vel, vel, vel, rate, rate, rate)
        r = (1.0, 1.0, 1.0, thrust)
        score, meta, _ = _eval_config(env, cfg, train, N, q, r)
        trial.set_user_attr("meta", meta)
        return score

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    # Anchor the search on the current committed config so TPE starts from a
    # known-good point (and we get an apples-to-apples trial for it) instead of
    # rediscovering it. Only enqueued if it lies inside the search space.
    study.enqueue_trial(
        {
            "N": DEFAULT_N,
            "pos_xy": DEFAULT_Q[0],
            "pos_z": DEFAULT_Q[2],
            "vel": DEFAULT_Q[6],
            "att": DEFAULT_Q[3],
            "rate": DEFAULT_Q[9],
            "thrust_R": DEFAULT_R[3],
        }
    )
    t0 = _time.perf_counter()
    best_so_far = [-1e9]

    def _cb(study, trial):
        s = trial.value if trial.value is not None else -1e9
        if s > best_so_far[0]:
            best_so_far[0] = s
            print(f"[trial {trial.number:>3}] NEW BEST score={s:+.4f}  "
                  f"{trial.user_attrs.get('meta')}  params={trial.params}")

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
    elapsed = _time.perf_counter() - t0

    bp = study.best_params
    N = bp["N"]
    q = (bp["pos_xy"], bp["pos_xy"], bp["pos_z"], bp["att"], bp["att"], bp["att"],
         bp["vel"], bp["vel"], bp["vel"], bp["rate"], bp["rate"], bp["rate"])
    r = (1.0, 1.0, 1.0, bp["thrust_R"])

    print(f"\n=== Best on TRAIN (score {study.best_value:+.4f}) after {n_trials} "
          f"trials in {elapsed:.0f}s ===\n  {study.best_trial.user_attrs['meta']}\n  {bp}")

    # Validate the winner + the current default on held-out seeds.
    tr_s, tr_m, _ = _eval_config(env, cfg, train, N, q, r)
    va_s, va_m, _ = _eval_config(env, cfg, val, N, q, r)
    db_s, db_m, _ = _eval_config(env, cfg, val, DEFAULT_HP()["N"], DEFAULT_Q, DEFAULT_R)
    env.close()

    print("\n=== Generalization check ===")
    print(f"  tuned  TRAIN : {tr_m}")
    print(f"  tuned  VAL   : {va_m}")
    print(f"  default VAL  : {db_m}  (baseline to beat on held-out)")

    result = {
        "config": config, "n_trials": n_trials,
        "best_params": bp, "q_diag": q, "r_diag": r, "N": N,
        "train": tr_m, "val": va_m, "default_val": db_m,
        "train_seeds": train, "val_seeds": val,
    }
    out_path = REPO_ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved {out_path.with_suffix('.json')}")
    return result


def DEFAULT_HP():
    return {"N": DEFAULT_N, "q_diag": DEFAULT_Q, "r_diag": DEFAULT_R}


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.WARNING)
    fire.Fire(
        {
            "sweep_n": sweep_n,
            "sweep_timing": sweep_timing,
            "tune": tune,
            "score_one": score_one,
        }
    )
