"""

The Cavemen Approach for a exploratory controller 

The controller is designed to be modular, with separate components for path generation, timing, and trajectory generation and quick swappable 

Current architecture for the entry challenge  

- Simple three point entry center exit waypoint selection with simple collision avoidance for the gates
- Between gates A* Brute Force Oooga Booga path generator using a 3D Occupancy grid and point snapping and waypoint pruning
- Very simple distance based timing
- And still using baseline MPC that is given as example

A lot of future improvement possible, but I was low on time and this should be enough for the entry test

Idea collection: 
- Weighted and dynamic aware A*
- Better timing and trajectory generation
- Better modularity
- RL based optimizatio

"""

from __future__ import annotations  # Python 3.10 type hints

from pathlib import Path
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.train_rl import Agent

## Modular imports
from lsy_drone_racing.control.controllers.modules.path_generator_improved import AStarImprovedPathGenerator
from lsy_drone_racing.control.controllers.modules.path_generator_barebone import AStarBarebonePathGenerator
from lsy_drone_racing.control.controllers.modules.initial_challenge.timing_module import DistanceTiming
from lsy_drone_racing.control.controllers.modules.timing_module_improved import DynamicTiming
from lsy_drone_racing.control.controllers.modules.initial_challenge.trajectory_module import SplineTrajectory
from lsy_drone_racing.control.controllers.modules.trajectory_module_improved import ImprovedSplineTrajectory
from lsy_drone_racing.control.controllers.modules.post_processing import PathPostProcessor

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


class AttitudeMPC(Controller):
    """RL attitude controller with spline planning and event-triggered replanning."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self.freq = config.env.freq

        # modules
        # self.path_gen = AStarGatePathGenerator()
        # self.path_gen = WaypointPathGenerator()
        self.path_gen = AStarBarebonePathGenerator(
            grid_resolution=0.05,
            safety_margin=0.05,
            obstacle_radius=0.215,
            prune_path=True,
            )

        # Dynamics-aware timing: fast on open straights, slows for curvature, tight
        # clearance and reversals, with an acceleration-limited (feasible) profile.
        # Uses the post-processor's per-waypoint clearance tube (self._clearance).
        # A/B over 10 level3 seeds vs the old DistanceTiming(0.8): same pass rate,
        # ~8% faster (median 11.76 -> 10.78 s). v_max=1.5 was faster still but
        # dropped a gate (tracker ceiling ~1.3-1.5); the clearance floor is kept at
        # 0.8 (== old cruise) so tight spots are never slower than before, only
        # true reversals slow hard.
        self.timing = DynamicTiming(
            v_max=1.5,
            a_max=2.5,
            clearance_ref=0.35,
            clearance_floor_speed=0.8,
            reversal_speed=0.3,
            min_segment_time=0.01,
        )

        # Clearance "tube" provider for the upcoming clearance-aware timing module.
        # A/B over 10 level3 seeds: refinement OFF = 10/10; floor-only and gentle
        # nudge both = 9/10 (drop seed 368113776) -- the RL tracker mistracks any
        # waypoint shift (see rl-controller-waypoint-sensitivity). So refinement is
        # disabled here (path unchanged), but process() still returns the tube in
        # self._clearance. Enable the nudge/floor for a more robust tracker (MPC).
        self.post_proc = PathPostProcessor(enabled=False)

        # generate once
        waypoints = self.path_gen.generate(obs, config)
        waypoints, self._clearance = self.post_proc.process(waypoints, obs, config)
        self._raw_waypoints = np.asarray(waypoints, dtype=float)
        ##t = self.timing.compute(waypoints, t_total=25) # Simpleixed timing with total time
        # NOTE: DynamicTiming supports v_init=norm(obs["vel"]) to start the profile
        # from the drone's actual speed. A/B (10 level3 seeds) showed it is ~8%
        # faster BUT drops 2-3 gates: this RL tracker needs the gentle near-stop
        # ramp after a (re)plan to keep the setpoint near the drone. Left unseeded
        # here (10/10); pass v_init for a robust tracker (MPC).
        t = self.timing.compute(waypoints, clearance=self._clearance)

        self.trajectory = SplineTrajectory(waypoints, t, config.env.freq)
        self._track_gates = np.asarray(obs.get("gates_pos", []), dtype=float)
        self._track_obstacles = np.asarray(obs.get("obstacles_pos", []), dtype=float)

        self._last_target_gate = int(np.asarray(obs["target_gate"]).item())
        self._last_gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._last_gates_quat = np.asarray(obs["gates_quat"], dtype=float).copy()
        self._last_obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()

        self._replan_count = 0
        self._min_replan_interval_ticks = int(0.5 * config.env.freq)  # at most every 0.5s
        self._last_replan_tick = 0

        ## Debug outputs -- single, seed-named flight log written at episode end.
        self._init_flight_log(obs, config)

        self.print_trajectory_obstacle_clearance(
            self.trajectory._pos,
            np.asarray(obs["obstacles_pos"], dtype=float),
            min_clearance=0.25,
        )
        ## End debug outputs
        self._N = 30
        self.n_obs = 2
        self.n_samples = 10
        self.samples_dt = 0.1
        self.sample_offsets = np.array(
            np.arange(self.n_samples) * self.freq * self.samples_dt, dtype=int
        )

        self.drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.drone_mass = self.drone_params["mass"]
        self.thrust_min = self.drone_params["thrust_min"] * 4
        self.thrust_max = self.drone_params["thrust_max"] * 4

        self.agent = Agent((13 + 3 * self.n_samples + self.n_obs * 13 + 4,), (4,)).to("cpu")
        model_path = Path(__file__).resolve().parents[1] / "ppo_drone_racing.ckpt"
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.agent.eval()

        self.basic_obs_key = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1).astype(np.float32)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1))
        self.last_action = np.zeros(4, dtype=np.float32)

        self._tick = 0
        self._tick_max = len(self.trajectory._pos) - 1
        self._global_tick = 0
        self._config = config
        self._finished = False

    ## Debug
    def print_trajectory_obstacle_clearance(self, traj_pos, obstacles, min_clearance=0.25):
        for j, obs_p in enumerate(obstacles):
            d_xy = np.linalg.norm(traj_pos[:, :2] - obs_p[:2], axis=1)
            idx = int(np.argmin(d_xy))
            if d_xy[idx] < min_clearance:
                print(
                    f"WARNING: trajectory too close to obstacle {j}: "
                    f"min_xy={d_xy[idx]:.3f}, idx={idx}, "
                    f"traj_pos={traj_pos[idx]}, obs={obs_p}"
                )

    # --- Single-file, seed-named flight logging -------------------------------
    # The whole flight (initial plan, every replan, and the per-tick executed vs.
    # commanded state) is accumulated in memory and dumped to one .npz at episode
    # end: debug_outputs/flight_<tag>.npz, where <tag> is `seed<N>` when the seed
    # is known (set config.env.seed before constructing the controller; the sim/
    # eval utilities do this) and a timestamp otherwise.

    def _init_flight_log(self, obs: dict[str, NDArray[np.floating]], config: dict) -> None:
        """Set up the in-memory flight log and resolve the seed-based file tag."""
        self._debug_dir = Path("debug_outputs")
        self._debug_dir.mkdir(exist_ok=True)

        seed = None
        try:
            s = int(np.asarray(config.env.seed).item())
            if s >= 0:
                seed = s
        except Exception:  # noqa: BLE001 - seed is best-effort metadata
            seed = None
        self._debug_seed = seed if seed is not None else -1
        self._debug_tag = f"seed{seed}" if seed is not None else time.strftime("run_%Y%m%d_%H%M%S")

        # Static snapshot at construction (the initial, pre-flight plan).
        self._initial_track = {
            "gates_pos": np.asarray(obs["gates_pos"], dtype=float),
            "gates_quat": np.asarray(obs["gates_quat"], dtype=float),
            "obstacles_pos": np.asarray(obs["obstacles_pos"], dtype=float),
            "start_pos": np.asarray(obs["pos"], dtype=float),
        }
        self._initial_plan = {
            "raw_waypoints": np.asarray(self._raw_waypoints, dtype=float),
            "traj_pos": np.asarray(self.trajectory._pos, dtype=float),
            "traj_vel": np.asarray(self.trajectory._vel, dtype=float),
            "time_grid": np.asarray(getattr(self.trajectory, "_time_grid", []), dtype=float),
        }

        # Per-tick execution log (filled in step_callback).
        self._flight = {k: [] for k in (
            "t", "tick", "target_gate", "executed_pos", "executed_vel",
            "executed_quat", "target_pos", "cmd", "sensed_gates_pos",
            "sensed_gates_quat", "sensed_obstacles_pos",
        )}
        # Per-replan snapshots (filled in _replan_trajectory).
        self._replans = []
        self._flight_written = False

    def _log_step(
        self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]]
    ) -> None:
        """Record one tick of executed state vs. commanded setpoint/action."""
        # Disabled drones are warped to z=-1 on the terminating step; skip that
        # sentinel so the logged executed trajectory stays physical.
        if float(np.asarray(obs["pos"])[2]) < -0.5:
            return
        idx = min(self._tick, len(self.trajectory._pos) - 1)
        f = self._flight
        f["t"].append(self._global_tick / self.freq)
        f["tick"].append(int(self._tick))
        f["target_gate"].append(int(np.asarray(obs["target_gate"]).item()))
        f["executed_pos"].append(np.asarray(obs["pos"], dtype=float))
        f["executed_vel"].append(np.asarray(obs["vel"], dtype=float))
        f["executed_quat"].append(np.asarray(obs["quat"], dtype=float))
        f["target_pos"].append(np.asarray(self.trajectory._pos[idx], dtype=float))
        f["cmd"].append(np.asarray(action, dtype=float))
        f["sensed_gates_pos"].append(np.asarray(obs["gates_pos"], dtype=float))
        f["sensed_gates_quat"].append(np.asarray(obs["gates_quat"], dtype=float))
        f["sensed_obstacles_pos"].append(np.asarray(obs["obstacles_pos"], dtype=float))

    def _log_replan(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Snapshot the plan produced by a replan (called after it is built)."""
        self._replans.append({
            "tick": int(self._global_tick),
            "target_gate": int(np.asarray(obs["target_gate"]).item()),
            "waypoints": np.asarray(self._raw_waypoints, dtype=float),
            "traj_pos": np.asarray(self.trajectory._pos, dtype=float),
        })

    def _write_flight_log(self, reason: str) -> None:
        """Dump the accumulated flight to a single seed-named .npz file."""
        if self._flight_written:
            return
        self._flight_written = True

        def _stack(key: str) -> NDArray[np.floating]:
            seq = self._flight[key]
            return np.asarray(seq) if seq else np.empty((0,))

        data = {
            "seed": self._debug_seed,
            "reason": reason,
            "freq": self.freq,
            # Initial (pre-flight) plan + track.
            "start_pos": self._initial_track["start_pos"],
            "gates_pos": self._initial_track["gates_pos"],
            "gates_quat": self._initial_track["gates_quat"],
            "obstacles_pos": self._initial_track["obstacles_pos"],
            "initial_raw_waypoints": self._initial_plan["raw_waypoints"],
            "initial_traj_pos": self._initial_plan["traj_pos"],
            "initial_traj_vel": self._initial_plan["traj_vel"],
            "initial_time_grid": self._initial_plan["time_grid"],
            # Final plan in effect when the flight ended.
            "final_raw_waypoints": np.asarray(self._raw_waypoints, dtype=float),
            "final_traj_pos": np.asarray(self.trajectory._pos, dtype=float),
            # Per-tick execution log.
            "log_t": _stack("t"),
            "log_tick": _stack("tick"),
            "log_target_gate": _stack("target_gate"),
            "executed_pos": _stack("executed_pos"),
            "executed_vel": _stack("executed_vel"),
            "executed_quat": _stack("executed_quat"),
            "target_pos": _stack("target_pos"),
            "cmd": _stack("cmd"),
            "sensed_gates_pos": _stack("sensed_gates_pos"),
            "sensed_gates_quat": _stack("sensed_gates_quat"),
            "sensed_obstacles_pos": _stack("sensed_obstacles_pos"),
            # Replan history.
            "n_replans": len(self._replans),
            "replan_ticks": np.asarray([r["tick"] for r in self._replans], dtype=int),
            "replan_target_gate": np.asarray(
                [r["target_gate"] for r in self._replans], dtype=int
            ),
        }
        for k, rp in enumerate(self._replans):
            data[f"replan_{k}_waypoints"] = rp["waypoints"]
            data[f"replan_{k}_traj_pos"] = rp["traj_pos"]

        path = self._debug_dir / f"flight_{self._debug_tag}.npz"
        np.savez(path, **data)
        print(
            f"[debug] wrote flight log {path.name}: "
            f"{len(self._flight['t'])} ticks, {len(self._replans)} replans, reason={reason}"
        )

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired roll/pitch/yaw and collective thrust.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """

        self._global_tick += 1

        if self._should_replan(obs):
            self._replan_trajectory(obs)

        if self._tick >= self._tick_max:
            self._finished = True

        obs_rl = torch.tensor(self._obs_rl(obs), dtype=torch.float32).unsqueeze(0).to("cpu")
        with torch.no_grad():
            raw_action, _, _, _ = self.agent.get_action_and_value(obs_rl, deterministic=True)
            raw_action[..., 2] = 0.0
            self.last_action = raw_action.squeeze(0).cpu().numpy().astype(np.float32)

        return self._scale_actions(self.last_action)

    def _obs_rl(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Build the flattened PPO observation used during training."""
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1).astype(np.float32)
        idx = np.clip(self._tick + self.sample_offsets, 0, len(self.trajectory._pos) - 1)
        local_samples = (self.trajectory._pos[idx] - obs["pos"]).reshape(-1).astype(np.float32)
        prev_obs = self.prev_obs.reshape(-1).astype(np.float32)
        last_action = self.last_action.astype(np.float32)

        obs_rl = np.concatenate([basic_obs, local_samples, prev_obs, last_action], axis=-1)
        self.prev_obs = np.concatenate([self.prev_obs[1:, :], basic_obs[None, :]], axis=0)
        return obs_rl.astype(np.float32)

    def _scale_actions(self, actions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Rescale normalized policy actions to simulator commands."""
        scale = np.array(
            [np.pi / 2, np.pi / 2, np.pi / 2, (self.thrust_max - self.thrust_min) / 2.0],
            dtype=np.float32,
        )
        mean = np.array(
            [0.0, 0.0, 0.0, (self.thrust_max + self.thrust_min) / 2.0],
            dtype=np.float32,
        )
        return (np.clip(actions, -1.0, 1.0) * scale + mean).astype(np.float32)

    def _replan_trajectory(self, obs):
        """Regenerate path and trajectory from the current observation."""
        t0 = time.perf_counter()
        waypoints = self.path_gen.generate(obs, self._config)
        waypoints, self._clearance = self.post_proc.process(waypoints, obs, self._config)
        t1 = time.perf_counter()

        self._raw_waypoints = np.asarray(waypoints, dtype=float)

        # Unseeded on purpose: seeding v_init=norm(obs["vel"]) here is faster but
        # drops gates -- the RL tracker needs the gentle post-replan ramp (see the
        # note in __init__). Pass v_init for a robust tracker.
        t = self.timing.compute(self._raw_waypoints, clearance=self._clearance)
        self.trajectory = SplineTrajectory(self._raw_waypoints, t, self._config.env.freq)

        t2 = time.perf_counter()
        print(
            f"Replan #{self._replan_count}: "
            f"path={1000*(t1-t0):.1f} ms, "
            f"traj={1000*(t2-t1):.1f} ms, "
            f"waypoints={len(self._raw_waypoints)}"
        )
        self._tick = 0
        self._tick_max = len(self.trajectory._pos) - 1
        self._finished = False

        self._track_gates = np.asarray(obs.get("gates_pos", []), dtype=float)
        self._track_obstacles = np.asarray(obs.get("obstacles_pos", []), dtype=float)

        self._last_target_gate = int(np.asarray(obs["target_gate"]).item())
        self._last_gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._last_gates_quat = np.asarray(obs["gates_quat"], dtype=float).copy()
        self._last_obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()

        self._last_replan_tick = self._global_tick
        self._replan_count += 1
        self._log_replan(obs)

        print(f"Replanned trajectory #{self._replan_count}. target_gate: ", {self._last_target_gate})

    def _should_replan(self, obs):
        """Event-triggered replanning for level 2."""
        current_target_gate = int(np.asarray(obs["target_gate"]).item())

        # Avoid replanning every tick.
        if self._global_tick - self._last_replan_tick < self._min_replan_interval_ticks:
            return False

        # Replan when a gate has been passed.
        if current_target_gate != self._last_target_gate:
            print("Replan trigger: target gate changed")
            return True

        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float)

        gate_pos_shift = np.linalg.norm(gates_pos - self._last_gates_pos, axis=1)
        obstacle_shift = np.linalg.norm(obstacles_pos - self._last_obstacles_pos, axis=1)

        # Thresholds should be larger than tiny sensor/noise changes.
        if np.any(gate_pos_shift > 0.05):
            print("Replan trigger: gate position changed", gate_pos_shift)
            return True

        if np.any(obstacle_shift > 0.05):
            print("Replan trigger: obstacle position changed", obstacle_shift)
            return True

        # Quaternion change check, rough but good enough.
        quat_diff = np.linalg.norm(gates_quat - self._last_gates_quat, axis=1)
        if np.any(quat_diff > 0.05):
            print("Replan trigger: gate orientation changed", quat_diff)
            return True

        return False
    
    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Log this tick, advance the tick counter, and flush on episode end."""
        # Log before incrementing so target_pos matches the tick that produced
        # the applied action.
        self._log_step(action, obs)
        self._tick += 1

        if terminated or truncated or self._finished:
            if obs["target_gate"] == -1:
                reason = "finished"
            elif truncated:
                reason = "truncated"
            elif terminated:
                reason = "terminated"
            else:
                reason = "controller_finished"
            self._write_flight_log(reason)

        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0

    def render_callback(self, sim: Sim):
        """Visualize track, generated waypoints, smooth trajectory, and current setpoint."""

        if self._track_gates.size > 0:
            draw_points(sim, self._track_gates, rgba=(0.1, 0.6, 1.0, 0.9), size=0.025)
            draw_line(sim, self._track_gates, rgba=(0.1, 0.6, 1.0, 0.35))

        if self._track_obstacles.size > 0:
            draw_points(sim, self._track_obstacles, rgba=(1.0, 0.5, 0.0, 0.9), size=0.02)

        if self._raw_waypoints.size > 0:
            draw_points(sim, self._raw_waypoints, rgba=(0.9, 0.0, 0.9, 0.85), size=0.015)
            draw_line(sim, self._raw_waypoints, rgba=(0.9, 0.0, 0.9, 0.35))

        # Full trajectory, heavily downsampled
        traj_vis = self.trajectory._pos[::20]
        draw_line(sim, traj_vis, rgba=(0.0, 1.0, 0.0, 0.8))

        # Current MPC horizon, short enough to draw directly
        horizon = self.trajectory.sample_horizon(self._tick, self._N)["pos"]
        draw_line(sim, horizon, rgba=(1.0, 0.0, 0.0, 0.7))

        # Current setpoint
        i = min(self._tick, len(self.trajectory._pos) - 1)
        setpoint = self.trajectory._pos[i].reshape(1, -1)
        draw_points(sim, setpoint, rgba=(1.0, 0.0, 0.0, 1.0), size=0.02)
