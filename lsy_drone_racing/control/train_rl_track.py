"""Closed-loop RL training: PPO tracker on references from the real path planner.

The stock ``train_rl`` trains the tracker to follow *random cubic splines*. At
deploy the tracker follows the A*+timing planner instead, so its training and
test reference distributions differ. This module closes that gap:

1. ``build_track_bank`` samples many randomized race tracks (level3), runs the
   actual planner pipeline on each (``AStarImprovedPathGenerator`` -> clearance
   ``PathPostProcessor`` -> dynamics-aware ``DynamicTiming`` -> ``SplineTrajectory``,
   identical to ``controller_rl_astar``), and stores the resulting dense position
   references as an offline bank ``(N, n_steps, 3)``. The numpy A* is too slow to
   run inside the 1024-env JAX loop, so it is done once, offline.
2. ``TrackTrajEnv`` is a drop-in ``RandTrajEnv`` whose ``reset`` samples a
   reference from the bank per world instead of generating a random spline.
3. ``make_track_envs`` wraps it with the same wrapper stack as ``make_envs``, and
   ``main`` trains via ``train_rl.train_ppo(..., env_fn=make_track_envs)``.

The tracker thus learns the exact reference shapes / speed profiles / reversals
the planner produces. It is still open-loop w.r.t. gate passing (the reward is
distance-to-reference, as in the stock env) -- only the reference distribution is
made realistic; in-loop replanning is out of scope.

Run (inside the ``rl`` pixi env)::

    # build the bank once (writes control/rl_track_bank.npz)
    pixi run -e rl python .../train_rl_track.py build_bank --n_tracks 2000
    # train on it (builds the bank first if missing)
    pixi run -e rl python .../train_rl_track.py train --total_timesteps 1500000
"""

from __future__ import annotations

import os

# crazyflow requires this before scipy is imported (see inspect_sim).
os.environ.setdefault("SCIPY_ARRAY_API", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.55")  # share VRAM with torch

import time  # noqa: E402
from pathlib import Path  # noqa: E402

import fire  # noqa: E402
import gymnasium  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from crazyflow.envs.drone_env import DroneEnv  # noqa: E402
from crazyflow.envs.norm_actions_wrapper import NormalizeActions  # noqa: E402
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy  # noqa: E402
from gymnasium.wrappers.vector.jax_to_torch import JaxToTorch  # noqa: E402

from lsy_drone_racing.control.controllers.modules.initial_challenge.trajectory_module import (  # noqa: E402
    SplineTrajectory,
)
from lsy_drone_racing.control.controllers.modules.path_generator_improved import (  # noqa: E402
    AStarImprovedPathGenerator,
)
from lsy_drone_racing.control.controllers.modules.post_processing import (  # noqa: E402
    PathPostProcessor,
)
from lsy_drone_racing.control.controllers.modules.timing_module_improved import (  # noqa: E402
    DynamicTiming,
)
from lsy_drone_racing.control.controllers.utility.sim.inspect_sim import (  # noqa: E402
    REPO_ROOT,
    _true_track,
)
from lsy_drone_racing.control.train_rl import (  # noqa: E402
    ActionPenalty,
    AngleReward,
    Args,
    FlattenJaxObservation,
    RandTrajEnv,
    StackObs,
    train_ppo,
)
from lsy_drone_racing.utils import load_config  # noqa: E402

BANK_PATH = REPO_ROOT / "lsy_drone_racing" / "control" / "rl_track_bank.npz"


# ---------------------------------------------------------------------------
# planner pipeline (mirrors controller_rl_astar exactly)
# ---------------------------------------------------------------------------
# Non-speed pipeline params. To target MPC-level speed, the bank now mirrors
# controller_mpc_astar.TIMING_HYPERPARAMS (incl. the a_lat_max corner-speed cap the
# RL bank previously left at default) so the RL tracker trains on the same fast
# reference profiles the MPC flies. The RL controller's deploy timing is set to
# match. Path-gen still uses the RL controller's inflation (obstacle_radius=0.215).
_PATHGEN_KW = dict(grid_resolution=0.05, safety_margin=0.05, obstacle_radius=0.215, prune_path=True)
_TIMING_KW = dict(
    a_lat_max=4.85, clearance_ref=0.495, clearance_floor_speed=1.606,
    reversal_speed=0.833, min_segment_time=0.01,
)


def _make_pathgen() -> AStarImprovedPathGenerator:
    """The controller_rl_astar path generator (built once, speed-independent)."""
    return AStarImprovedPathGenerator(**_PATHGEN_KW)


def _make_timing(v_max: float, a_max: float) -> DynamicTiming:
    """DynamicTiming at a given speed cap (mirrors the MPC clearance/corner params)."""
    return DynamicTiming(v_max=v_max, a_max=a_max, **_TIMING_KW)


def _race_env(cfg):
    """Build the (single-world) race env used only to sample randomized tracks."""
    return JaxToNumpy(
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


def build_track_bank(
    config: str = "level3.toml",
    n_tracks: int = 5000,
    out: str | None = None,
    v_max_range: tuple[float, float] = (1.5, 3.0),
    a_max_range: tuple[float, float] = (2.5, 4.0),
    pad_percentile: float = 98.0,
    max_time: float = 15.0,
) -> dict:
    """Generate an offline bank of planner reference trajectories over random tracks.

    For each seed: reset the race env, read the *true* randomized track, run the
    planner on it (full knowledge, from the start gate), and sample the resulting
    ``SplineTrajectory`` at the env frequency. All references are padded (hold the
    final point) / truncated to a common ``n_steps`` so they stack into one array.

    The **speed cap is domain-randomized per track** (``v_max``/``a_max`` sampled
    from the given ranges): the tracker then learns to follow references spanning
    slow-to-fast, so the deploy ``v_max`` can be pushed up without the policy
    seeing a speed it never trained on. Set both range bounds equal for a fixed
    speed. Non-speed params mirror the deployed controller_rl_astar pipeline.

    Args:
        config: Race config to sample tracks from (default level3, full random).
        n_tracks: Number of tracks (seeds) to attempt.
        out: Output ``.npz`` path (default ``control/rl_track_bank.npz``).
        v_max_range: Per-track ``v_max`` sampled uniformly in this range (m/s).
        a_max_range: Per-track ``a_max`` sampled uniformly in this range (m/s^2).
        pad_percentile: ``n_steps`` is set to this percentile of reference lengths
            (capped at ``max_time``), so most references are not truncated.
        max_time: Hard cap on reference length in seconds.

    Returns:
        Summary dict (path, shape, freq, length + speed stats).
    """
    out_path = Path(out) if out else BANK_PATH
    cfg = load_config(REPO_ROOT / "config" / config)
    cfg.sim.render = False
    freq = int(cfg.env.freq)
    env = _race_env(cfg)
    path_gen = _make_pathgen()
    post_proc = PathPostProcessor(enabled=False)  # refinement off (RL tracker is nudge-sensitive)
    rng = np.random.default_rng(0)

    refs: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    v_maxes: list[float] = []
    n_fail = 0
    t0 = time.perf_counter()
    for seed in range(n_tracks):
        try:
            v_max = float(rng.uniform(*v_max_range))
            a_max = float(rng.uniform(*a_max_range))
            obs, _ = env.reset(seed=seed)
            track = _true_track(env, obs)  # true randomized poses, from start gate
            planner_obs = {
                "pos": np.asarray(track["pos"], dtype=float),
                "gates_pos": np.asarray(track["gates_pos"], dtype=float),
                "gates_quat": np.asarray(track["gates_quat"], dtype=float),
                "obstacles_pos": np.asarray(track["obstacles_pos"], dtype=float),
                "target_gate": 0,
            }
            waypoints = path_gen.generate(planner_obs, cfg)
            waypoints, clearance = post_proc.process(waypoints, planner_obs, cfg)
            t = _make_timing(v_max, a_max).compute(waypoints, clearance=clearance)
            traj = SplineTrajectory(waypoints, t, freq)
            pos = np.asarray(traj._pos, dtype=np.float32)  # (L, 3), sampled at freq
            if pos.ndim != 2 or pos.shape[0] < 2:
                n_fail += 1
                continue
            refs.append(pos)
            starts.append(pos[0].copy())
            v_maxes.append(v_max)
        except Exception as e:  # noqa: BLE001 - a bad track is skipped, not fatal
            n_fail += 1
            if n_fail <= 5:
                print(f"  seed {seed}: planner failed ({type(e).__name__}: {e}) -- skipped")
        if (seed + 1) % 500 == 0:
            print(f"  {seed + 1}/{n_tracks} tracks ({len(refs)} ok, {n_fail} failed, "
                  f"{time.perf_counter() - t0:.0f}s)")
    env.close()

    if not refs:
        raise RuntimeError("No planner references were generated.")

    lengths = np.array([r.shape[0] for r in refs])
    n_steps = int(min(np.percentile(lengths, pad_percentile), max_time * freq))
    n_steps = max(n_steps, 2)

    bank = np.empty((len(refs), n_steps, 3), dtype=np.float32)
    for i, r in enumerate(refs):
        if r.shape[0] >= n_steps:
            bank[i] = r[:n_steps]
        else:  # pad by holding the final (finish) point -> the drone hovers there
            bank[i, : r.shape[0]] = r
            bank[i, r.shape[0] :] = r[-1]

    start = np.median(np.stack(starts), axis=0).astype(np.float32)
    v_maxes_arr = np.asarray(v_maxes, dtype=np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        bank=bank,
        freq=freq,
        start=start,
        lengths=lengths,
        v_maxes=v_maxes_arr,
        config=config,
    )
    summary = {
        "path": str(out_path),
        "n_refs": len(refs),
        "n_fail": n_fail,
        "shape": list(bank.shape),
        "freq": freq,
        "n_steps": n_steps,
        "sec": round(n_steps / freq, 2),
        "len_s_min_med_max": [
            round(float(lengths.min()) / freq, 2),
            round(float(np.median(lengths)) / freq, 2),
            round(float(lengths.max()) / freq, 2),
        ],
        "v_max_range": [round(float(v_maxes_arr.min()), 2), round(float(v_maxes_arr.max()), 2)],
        "start": start.tolist(),
    }
    print(f"\nSaved bank -> {out_path}\n  {summary}")
    return summary


# ---------------------------------------------------------------------------
# training env: sample a planner reference per world at reset
# ---------------------------------------------------------------------------
class TrackTrajEnv(RandTrajEnv):
    """RandTrajEnv variant that follows references sampled from the planner bank."""

    def __init__(self, bank_path: str | Path = BANK_PATH, **kwargs):
        """Load the bank, size the episode to it, and reset the drone to its start.

        ``kwargs`` are forwarded to ``RandTrajEnv`` (num_envs, freq, drone_model,
        physics, disturbances, device); ``trajectory_time``/``max_episode_time``
        are derived from the bank so shapes line up.
        """
        data = np.load(Path(bank_path))
        self._bank = data["bank"].astype(np.float32)  # (N, n_steps, 3)
        bank_freq = int(data["freq"])
        bank_start = np.asarray(data["start"], dtype=float)
        n_steps = self._bank.shape[1]

        freq = int(kwargs.get("freq", bank_freq))
        if freq != bank_freq:
            raise ValueError(f"bank freq {bank_freq} != env freq {freq}; rebuild the bank")
        bank_time = n_steps / freq
        kwargs.setdefault("trajectory_time", bank_time)
        kwargs.setdefault("max_episode_time", bank_time)

        super().__init__(**kwargs)
        # Override RandTrajEnv's fixed n_steps/takeoff with the bank's.
        self.n_steps = n_steps
        self.takeoff_pos = bank_start.astype(np.float32)
        self.trajectories = np.zeros((self.num_envs, n_steps, 3), dtype=np.float32)
        self._set_takeoff(self.takeoff_pos)

    def _set_takeoff(self, pos: np.ndarray) -> None:
        """Reset all drones' default spawn to ``pos`` (mirrors RandTrajEnv.__init__)."""
        data = self.sim.data
        self.sim.data = data.replace(
            states=data.states.replace(
                pos=np.broadcast_to(pos, (data.core.n_worlds, data.core.n_drones, 3))
            )
        )
        self.sim.build_default_data()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Sample one planner reference per world, then reset the drones."""
        idx = np.random.randint(0, self._bank.shape[0], size=self.sim.n_worlds)
        self.trajectories = self._bank[idx]  # (n_worlds, n_steps, 3)

        # Bypass RandTrajEnv.reset (which regenerates a random spline).
        DroneEnv.reset(self, seed=seed)
        if seed is not None:
            self.sim.seed(seed)
        self._reset(options=options)
        self._marked_for_reset = self._marked_for_reset.at[...].set(False)
        return self.obs(), {}


def make_track_envs(
    config: str = "level3.toml",
    num_envs: int = None,
    jax_device: str = "cpu",
    torch_device: torch.device = torch.device("cpu"),
    coefs: dict = {},
    bank_path: str | Path = BANK_PATH,
):
    """Make training envs that follow planner references (same wrappers as make_envs)."""
    cfg = load_config(REPO_ROOT / "config" / config)
    env = TrackTrajEnv(
        bank_path=bank_path,
        n_samples=10,
        num_envs=num_envs,
        freq=cfg.env.freq,
        drone_model=cfg.sim.drone_model,
        physics=cfg.sim.physics,
        disturbances=cfg.env.disturbances,
        device=jax_device,
    )
    env = NormalizeActions(env)
    env = StackObs(env, n_obs=coefs.get("n_obs", 0))
    env = AngleReward(env, rpy_coef=coefs.get("rpy_coef", 0.04))
    env = ActionPenalty(
        env,
        act_coef=coefs.get("act_coef", 0.04),
        d_act_th_coef=coefs.get("d_act_th_coef", 0.4),
        d_act_xy_coef=coefs.get("d_act_xy_coef", 1.0),
    )
    env = FlattenJaxObservation(env)
    env = JaxToTorch(env, torch_device)
    return env


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------
def build_bank(
    config: str = "level3.toml",
    n_tracks: int = 5000,
    out: str | None = None,
    v_max_lo: float = 1.5,
    v_max_hi: float = 3.0,
    a_max_lo: float = 2.5,
    a_max_hi: float = 4.0,
):
    """CLI: build the planner-reference bank.

    For a bank *matched* to a fixed deploy speed, set lo==hi (e.g.
    ``--v_max_lo 2.0 --v_max_hi 2.0 --a_max_lo 3.0 --a_max_hi 3.0``).
    """
    return build_track_bank(
        config=config, n_tracks=n_tracks, out=out,
        v_max_range=(v_max_lo, v_max_hi), a_max_range=(a_max_lo, a_max_hi),
    )


def train(
    config: str = "level3.toml",
    total_timesteps: int = 1_500_000,
    num_envs: int = 1024,
    n_tracks: int = 5000,
    out_ckpt: str = "ppo_drone_racing_track.ckpt",
    bank_path: str | None = None,
    rebuild_bank: bool = False,
    wandb_enabled: bool = False,
):
    """CLI: train the PPO tracker on planner references (builds the bank if missing).

    Saves to ``control/<out_ckpt>`` -- NOT the deployed ``ppo_drone_racing.ckpt`` --
    so the working checkpoint is preserved until you validate this one on the track.
    Use ``--rebuild_bank True`` to regenerate the (speed-randomized) bank; large
    ``--total_timesteps`` (e.g. 150_000_000) gives a long run.
    """
    bank = Path(bank_path) if bank_path else BANK_PATH
    if rebuild_bank or not bank.exists():
        print(f"Building planner-reference bank ({n_tracks} tracks) -> {bank}")
        build_track_bank(config=config, n_tracks=n_tracks, out=str(bank))
    else:
        print(f"Using existing bank {bank}")

    args = Args.create(total_timesteps=total_timesteps, num_envs=num_envs, jax_device="gpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = REPO_ROOT / "lsy_drone_racing" / "control" / out_ckpt

    def env_fn(num_envs, jax_device, torch_device, coefs):
        return make_track_envs(
            config=config, num_envs=num_envs, jax_device=jax_device,
            torch_device=torch_device, coefs=coefs, bank_path=bank,
        )

    print(f"iters={args.num_iterations} envs={args.num_envs} steps={args.total_timesteps}")
    t0 = time.time()
    train_ppo(args, ckpt, device, "gpu", wandb_enabled=wandb_enabled, env_fn=env_fn)
    print(f"TRAIN DONE in {time.time() - t0:.0f}s -> {ckpt}")
    return str(ckpt)


if __name__ == "__main__":
    fire.Fire({"build_bank": build_bank, "train": train})
