"""Initial-challenge RL controller: A* pipeline + the example RL tracker.

Modular pipeline (gate-anchored A* path generator on a 3D occupancy grid ->
distance-based timing -> spline trajectory) tracked by the course's example RL
policy, with event-triggered replanning. Superseded by ``controller_rl_astar``;
kept so the initial-challenge submission remains runnable on its own.
"""

from __future__ import annotations  # Python 3.10 type hints

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params

from lsy_drone_racing.control import Controller

## Modular imports
from lsy_drone_racing.control.controllers.challenge.modules.path_generator import (
    AStarGatePathGenerator,
)
from lsy_drone_racing.control.controllers.challenge.modules.timing_module import DistanceTiming
from lsy_drone_racing.control.controllers.modules.trajectory_module_improved import SplineTrajectory
from lsy_drone_racing.control.train_rl import Agent

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
        self.path_gen = AStarGatePathGenerator(
            grid_resolution=0.05, safety_margin=0.06, obstacle_radius=0.21, prune_path=True
        )
        ##self.timing = UniformTiming()

        self.timing = DistanceTiming(nominal_speed=1.2, min_segment_time=0.12)

        # generate once
        waypoints = self.path_gen.generate(obs, config)
        self._raw_waypoints = np.asarray(waypoints, dtype=float)
        ##t = self.timing.compute(waypoints, t_total=25) # Simpleixed timing with total time
        t = self.timing.compute(waypoints)

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

        ## Debug outputs
        debug_dir = Path("debug_outputs")
        debug_dir.mkdir(exist_ok=True)

        np.savez(
            debug_dir / "level1_path_debug.npz",
            gates_pos=np.asarray(obs["gates_pos"], dtype=float),
            gates_quat=np.asarray(obs["gates_quat"], dtype=float),
            obstacles_pos=np.asarray(obs["obstacles_pos"], dtype=float),
            start_pos=np.asarray(obs["pos"], dtype=float),
            raw_waypoints=np.asarray(self._raw_waypoints, dtype=float),
            traj_pos=np.asarray(self.trajectory._pos, dtype=float),
            traj_vel=np.asarray(self.trajectory._vel, dtype=float),
            time_grid=np.asarray(getattr(self.trajectory, "_time_grid", []), dtype=float),
        )

        self.print_trajectory_obstacle_clearance(
            self.trajectory._pos, np.asarray(obs["obstacles_pos"], dtype=float), min_clearance=0.25
        )
        self._executed_pos = []

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
        model_path = Path(__file__).resolve().parents[2] / "ppo_drone_racing.ckpt"
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
    def print_trajectory_obstacle_clearance(
        self,
        traj_pos: NDArray[np.floating],
        obstacles: NDArray[np.floating],
        min_clearance: float = 0.25,
    ) -> None:
        """Print a warning for each obstacle the trajectory passes closer than the floor."""
        for j, obs_p in enumerate(obstacles):
            d_xy = np.linalg.norm(traj_pos[:, :2] - obs_p[:2], axis=1)
            idx = int(np.argmin(d_xy))
            if d_xy[idx] < min_clearance:
                print(
                    f"WARNING: trajectory too close to obstacle {j}: "
                    f"min_xy={d_xy[idx]:.3f}, idx={idx}, "
                    f"traj_pos={traj_pos[idx]}, obs={obs_p}"
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
            [0.0, 0.0, 0.0, (self.thrust_max + self.thrust_min) / 2.0], dtype=np.float32
        )
        return (np.clip(actions, -1.0, 1.0) * scale + mean).astype(np.float32)

    def _replan_trajectory(self, obs: dict) -> None:
        """Regenerate path and trajectory from the current observation."""
        t0 = time.perf_counter()
        waypoints = self.path_gen.generate(obs, self._config)
        t1 = time.perf_counter()

        self._raw_waypoints = np.asarray(waypoints, dtype=float)

        t = self.timing.compute(self._raw_waypoints)
        self.trajectory = SplineTrajectory(self._raw_waypoints, t, self._config.env.freq)

        t2 = time.perf_counter()
        print(
            f"Replan #{self._replan_count}: "
            f"path={1000 * (t1 - t0):.1f} ms, "
            f"traj={1000 * (t2 - t1):.1f} ms, "
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

        print(
            f"Replanned trajectory #{self._replan_count}. target_gate: ", {self._last_target_gate}
        )

    def _should_replan(self, obs: dict) -> bool:
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
        """Increment the tick counter."""
        self._tick += 1

        # Debug
        self._executed_pos.append(np.asarray(obs["pos"], dtype=float).copy())
        if terminated or truncated or self._finished:
            np.savez(
                "debug_outputs/level1_execution_debug.npz",
                gates_pos=self._track_gates,
                obstacles_pos=self._track_obstacles,
                raw_waypoints=self._raw_waypoints,
                traj_pos=self.trajectory._pos,
                executed_pos=np.asarray(self._executed_pos),
            )

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
