"""Learned attitude tracker following the modular A* planning pipeline.

Second best performing controller and also validated in real,
a modular pipeline (gate-aware A* path generator -> clearance post-processor
-> dynamics-aware timing -> spline trajectory) produces the reference, and the
trajectory is tracked by a PPO policy (trained in ``train_rl``) instead of the
acados MPC. At each step the policy sees the drone state plus the next reference
samples relative to its position and outputs a roll/pitch/yaw/thrust command. The
pipeline replans event-triggered when sensed gate/obstacle poses deviate from the
plan.

The timing is deliberately softer than the MPC's: the learned tracker has a lower
achievable-speed ceiling, so aggressive corner/reversal speeds cause crashes.

This could well be caused by the the early stopping implemented in the training as
our team have the combined GPU compute power of a single mobile RTX4050 on my
Laptop, which is heavily thernal throttled :C thus we only ran small trainings.

This is also the reason why we didn't persue training an RL controller tnat also
handles trajectory generation and replanning as my poor computer will just die.

"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.controllers.modules.path_generator_improved import (
    AStarImprovedPathGenerator,
)
from lsy_drone_racing.control.controllers.modules.post_processing import PathPostProcessor
from lsy_drone_racing.control.controllers.modules.timing_module_improved import DynamicTiming

# modular pipeline imports
from lsy_drone_racing.control.controllers.modules.trajectory_module_improved import SplineTrajectory
from lsy_drone_racing.control.train_rl import Agent

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray

# Trained policy checkpoint (in controllers/utility/ckpt/)
CHECKPOINT = "ppo_track_final.ckpt"

# Trajectory timing hyperparameters
RL_TIMING_HYPERPARAMS = {
    "v_max": 2.1,
    "a_max": 3.2,
    "a_lat_max": 3.0,
    "clearance_ref": 0.45,
    "clearance_floor_speed": 0.8,
    "reversal_speed": 0.5,
    "min_segment_time": 0.01,
}

# Path-generator geometry
PATHGEN_HYPERPARAMS = {
    "grid_resolution": 0.05,
    "safety_margin": 0.05,
    "obstacle_radius": 0.215,
    "prune_path": True,
}


class AStarRLController(Controller):
    """RL attitude tracker for the modular pipeline, with event-triggered replanning."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the pipeline, plan the first trajectory, and load the policy.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._config = config
        self.freq = config.env.freq

        # Pipeline modules (shared design with controller_mpc_astar).
        self.path_gen = AStarImprovedPathGenerator(**PATHGEN_HYPERPARAMS)
        self.timing = DynamicTiming(**RL_TIMING_HYPERPARAMS)
        self.post_proc = PathPostProcessor(enabled=True)

        # Initial plan from the nominal track.
        self._replan_count = 0
        self._min_replan_interval_ticks = int(0.5 * config.env.freq)  # at most every 0.5 s
        self._last_replan_tick = 0
        self._plan_trajectory(obs)

        # Policy observation layout: [state(13), local ref samples(3*n), history, last action].
        self._N = 30  # horizon used only for rendering
        self.n_obs = 2
        self.n_samples = 10
        self.samples_dt = 0.1
        self.sample_offsets = np.array(
            np.arange(self.n_samples) * self.freq * self.samples_dt, dtype=int
        )

        self.drone_params = load_params(config.sim.physics, config.sim.drone_model)
        self.thrust_min = self.drone_params["thrust_min"] * 4
        self.thrust_max = self.drone_params["thrust_max"] * 4

        # Load the trained PPO policy (CPU inference is fast enough for one drone).
        self.agent = Agent((13 + 3 * self.n_samples + self.n_obs * 13 + 4,), (4,)).to("cpu")
        model_path = Path(__file__).resolve().parent / "utility" / "ckpt" / CHECKPOINT
        self.agent.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.agent.eval()

        # Rolling state history and last action fed back into the observation.
        self.basic_obs_key = ["pos", "quat", "vel", "ang_vel"]
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1).astype(np.float32)
        self.prev_obs = np.tile(basic_obs[None, :], (self.n_obs, 1))
        self.last_action = np.zeros(4, dtype=np.float32)

        self._tick = 0
        self._global_tick = 0
        self._finished = False

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
            self._replan(obs)
        if self._tick >= self._tick_max:
            self._finished = True

        obs_rl = torch.tensor(self._policy_obs(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            raw_action, _, _, _ = self.agent.get_action_and_value(obs_rl, deterministic=True)
            raw_action[..., 2] = 0.0  # yaw is fixed to zero (not used for racing)
            self.last_action = raw_action.squeeze(0).cpu().numpy().astype(np.float32)

        return self._scale_action(self.last_action)

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
        return self._finished

    def episode_callback(self):
        """Reset the tick counter at the end of an episode."""
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

        draw_line(sim, self.trajectory._pos[::20], rgba=(0.0, 1.0, 0.0, 0.8))
        horizon = self.trajectory.sample_horizon(self._tick, self._N)["pos"]
        draw_line(sim, horizon, rgba=(1.0, 0.0, 0.0, 0.7))

        i = min(self._tick, len(self.trajectory._pos) - 1)
        draw_points(
            sim, self.trajectory._pos[i].reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.02
        )

    # -------------------------------------------------------------- policy input
    def _policy_obs(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Build the flattened observation the PPO policy was trained on.

        Layout: current state (pos, quat, vel, ang_vel), the next ``n_samples``
        reference points relative to the drone (spaced ``samples_dt``), the state
        history, and the previous action.
        """
        basic_obs = np.concatenate([obs[k] for k in self.basic_obs_key], axis=-1).astype(np.float32)
        idx = np.clip(self._tick + self.sample_offsets, 0, len(self.trajectory._pos) - 1)
        local_samples = (self.trajectory._pos[idx] - obs["pos"]).reshape(-1).astype(np.float32)

        policy_obs = np.concatenate(
            [basic_obs, local_samples, self.prev_obs.reshape(-1), self.last_action]
        ).astype(np.float32)
        self.prev_obs = np.concatenate([self.prev_obs[1:, :], basic_obs[None, :]], axis=0)
        return policy_obs

    def _scale_action(self, action: NDArray[np.floating]) -> NDArray[np.floating]:
        """Rescale a normalized policy action in [-1, 1] to simulator commands."""
        half_thrust = (self.thrust_max - self.thrust_min) / 2.0
        hover_thrust = (self.thrust_max + self.thrust_min) / 2.0
        scale = np.array([np.pi / 2, np.pi / 2, np.pi / 2, half_thrust], dtype=np.float32)
        mean = np.array([0.0, 0.0, 0.0, hover_thrust], dtype=np.float32)
        return (np.clip(action, -1.0, 1.0) * scale + mean).astype(np.float32)

    # ------------------------------------------------------------------ planning
    def _plan_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Run the pipeline (path -> post-process -> timing -> spline) from ``obs``.

        Rebuilds ``self.trajectory`` and stores the gate/obstacle poses the plan was
        based on, so ``_should_replan`` can detect when they change.
        """
        waypoints = self.path_gen.generate(obs, self._config)
        waypoints, self._clearance = self.post_proc.process(waypoints, obs, self._config)
        self._raw_waypoints = np.asarray(waypoints, dtype=float)
        t = self.timing.compute(self._raw_waypoints, clearance=self._clearance)
        self.trajectory = SplineTrajectory(self._raw_waypoints, t, self._config.env.freq)
        self._tick_max = len(self.trajectory._pos) - 1

        self._track_gates = np.asarray(obs.get("gates_pos", []), dtype=float)
        self._track_obstacles = np.asarray(obs.get("obstacles_pos", []), dtype=float)
        self._last_target_gate = int(np.asarray(obs["target_gate"]).item())
        self._last_gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._last_gates_quat = np.asarray(obs["gates_quat"], dtype=float).copy()
        self._last_obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()

    def _replan(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Replan from the current observation and restart trajectory tracking."""
        t0 = time.perf_counter()
        self._plan_trajectory(obs)
        self._tick = 0
        self._finished = False
        self._last_replan_tick = self._global_tick
        self._replan_count += 1
        print(
            f"Replan #{self._replan_count}: {1000 * (time.perf_counter() - t0):.1f} ms, "
            f"{len(self._raw_waypoints)} waypoints, target_gate={self._last_target_gate}"
        )

    def _should_replan(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Event-triggered replanning: gate passed, or sensed poses moved > 5 cm."""
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

        # Thresholds must exceed sensor noise; quaternion check is rough but sufficient.
        gate_shift = np.linalg.norm(gates_pos - self._last_gates_pos, axis=1)
        obstacle_shift = np.linalg.norm(obstacles_pos - self._last_obstacles_pos, axis=1)
        quat_diff = np.linalg.norm(gates_quat - self._last_gates_quat, axis=1)

        if np.any(gate_shift > 0.05):
            print("Replan trigger: gate position changed", gate_shift)
            return True
        if np.any(obstacle_shift > 0.05):
            print("Replan trigger: obstacle position changed", obstacle_shift)
            return True
        if np.any(quat_diff > 0.05):
            print("Replan trigger: gate orientation changed", quat_diff)
            return True
        return False
