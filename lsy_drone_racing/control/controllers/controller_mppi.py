"""MPPI controller for the modular planning pipeline.

Due to my teammate not having a GPU in his laptop this implementation is focused on CPU compute :D

The controller combines the modular pipeline trajectory generation with a short-horizon MPPI tracker

MPPI uses a vectorized point-mass model, so it does not require a GPU
and does not depend on acados.  The command order is ``[roll, pitch, yaw,
collective_thrust]`` as expected by the attitude-control environment.

Note that due to time limitation we didn't really tune this one, only implemented it to learn
about how MPPI works as we want to explore all the possible implemenration, and that
we only have access to a single mobile RTX4050 GPU on my laptop, which is heavily thermal throttled.

Maybe its time to convince my teammate to get a better laptop with a proper GPU :D

"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.controllers.modules.curve_gate import CurveGatePathGenerator

# --- controller_mpc_astar reference pipeline (selectable via path_planner) ---
# These are light (no acados), so importing them keeps the MPPI GPU/acados-free.
from lsy_drone_racing.control.controllers.modules.path_generator_improved import (
    AStarImprovedPathGenerator,
)
from lsy_drone_racing.control.controllers.modules.path_generator_mppi import (
    AStarGatePathGenerator,
    DStarLiteGatePathGenerator,
    GatePassingPathGenerator,
    RRTStarGatePathGenerator,
    ThetaStarGatePathGenerator,
)
from lsy_drone_racing.control.controllers.modules.post_processing import PathPostProcessor
from lsy_drone_racing.control.controllers.modules.timing_module_improved import DynamicTiming
from lsy_drone_racing.control.controllers.modules.timing_module_mppi import MotionAwareTiming
from lsy_drone_racing.control.controllers.modules.trajectory_module_improved import (
    ImprovedSplineTrajectory,
    SplineTrajectory,
)

# Reference-speed profile for the "mpc_astar" pipeline under the MPPI tracker.
# Slower than controller_mpc_astar's own timing: the CPU point-mass MPPI has a
# lower achievable-speed ceiling, so an MPC-speed reference is untrackable.
_MPC_ASTAR_TIMING = {
    "v_max": 1.6,
    "a_max": 2.5,
    "a_lat_max": 2.2,
    "clearance_ref": 0.3,
    "clearance_floor_speed": 0.8,
    "reversal_speed": 0.3,
    "min_segment_time": 0.01,
}

if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


@dataclass(frozen=True)
class MPPIConfig:
    """Parameters chosen for a 50 Hz controller running on a laptop CPU."""

    # Reference pipeline the MPPI tracker follows. Options:
    # Native-MPPI planners (motion-aware timing + PCHIP spline):
    #   "theta_star" (default), "astar", "d_star_lite", "rrt_star", "curve".
    # Full controller_mpc_astar pipeline (A*+reversal/commit path gen, clearance
    # post-processing, dynamics-aware timing, arc-length spline):
    #   "mpc_astar".

    path_planner: str = "curve"

    # --- MPPI core (active in every mode) ---
    horizon: int = 35  # rollout steps (~0.70 s @ 50 Hz lookahead)
    num_samples: int = 256  # rollouts/step; 256 > 100 on reliability (5/6 vs 4/6)
    temperature: float = 6.0
    iterations: int = 1
    max_tilt: float = 0.504

    # --- cost / rollout clearance (active in every mode) ---
    obstacle_clearance: float = 0.215
    gate_constraint_distance: float = 0.3

    # --- replanning (active in every mode) ---
    replan_position_threshold: float = 0.05
    replan_interval_s: float = 0.25
    visualization_rollout_stride: int = 3
    seed: int = 7

    # --- native-MPPI stack ONLY (unused when path_planner == "mpc_astar")
    nominal_speed: float = 1.0
    turn_time_gain: float = 0.524
    gate_entry_distance: float = 0.3


def load_mppi_config(config: Any) -> MPPIConfig:
    """Load optional ``[controller.mppi]`` overrides from a race config."""
    controller_config = config.get("controller", {})
    overrides = controller_config.get("mppi", {})
    allowed = {field.name for field in fields(MPPIConfig)}
    unknown = set(overrides) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown controller.mppi parameter(s): {names}")
    return MPPIConfig(**{name: overrides[name] for name in overrides})


class AttitudeMPPI(Controller):
    """Gate-aware MPPI controller using desired attitude and collective thrust."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the path planner, dynamics approximation and sample sequence."""
        super().__init__(obs, info, config)
        self.cfg = load_mppi_config(config)
        self._config = config
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        self._rng = np.random.default_rng(self.cfg.seed)

        params = load_params("so_rpy", config.sim.drone_model)
        self._mass = float(params["mass"])
        self._gravity = -float(np.asarray(params["gravity_vec"])[2])
        self._thrust_min = float(params["thrust_min"]) * 4.0
        self._thrust_max = float(params["thrust_max"]) * 4.0
        self._hover_thrust = self._mass * self._gravity

        # Noise is expressed directly in roll, pitch, yaw and Newtons.
        self._noise_std = np.array([0.12, 0.12, 0.08, 0.08 * self._hover_thrust], dtype=float)
        self._u_low = np.array(
            [-self.cfg.max_tilt, -self.cfg.max_tilt, -np.pi / 2, self._thrust_min]
        )
        self._u_high = np.array([self.cfg.max_tilt, self.cfg.max_tilt, np.pi / 2, self._thrust_max])

        limits = config.env.track.safety_limits
        self._bounds_low = np.asarray(limits.pos_limit_low, dtype=float)
        self._bounds_high = np.asarray(limits.pos_limit_high, dtype=float)

        self._build_reference_pipeline()

        self._tick = 0
        self._last_replan_tick = -(10**9)
        self._pending_gate_replan_tick: int | None = None
        self._last_target_gate = 0
        self._last_gates_pos = np.empty((0, 3))
        self._last_gates_quat = np.empty((0, 4))
        self._last_obstacles_pos = np.empty((0, 3))
        self._control_sequence = self._hover_sequence()
        self._predicted_path = np.empty((0, 3), dtype=float)
        self._candidate_paths = np.empty((0, self.cfg.horizon, 3), dtype=float)
        self._plan_trajectory(obs)

    def _build_reference_pipeline(self):
        """Set up the reference stack (path gen + timing + post-proc + trajectory).

        ``cfg.path_planner == "mpc_astar"`` selects the full pipeline used by
        controller_mpc_astar (A* with reversal/commit handling, clearance
        post-processing, dynamics-aware ``DynamicTiming``, arc-length
        ``ImprovedSplineTrajectory``). Every other value uses the native MPPI
        stack (the selected planner + ``MotionAwareTiming`` + PCHIP
        ``SplineTrajectory``). ``_plan_trajectory`` and the MPPI tracker consume
        ``self.trajectory`` identically for either stack.
        """
        if self.cfg.path_planner == "mpc_astar":
            self._path_generator = AStarImprovedPathGenerator(
                grid_resolution=0.05, safety_margin=0.05, obstacle_radius=0.18, prune_path=True
            )
            self._post_proc = PathPostProcessor(enabled=True)
            self._timing = DynamicTiming(**_MPC_ASTAR_TIMING)
            self._trajectory_cls = ImprovedSplineTrajectory
        else:
            self._path_generator = self._build_path_generator()
            self._post_proc = None
            self._timing = MotionAwareTiming(
                nominal_speed=self.cfg.nominal_speed,
                min_segment_time=0.16,
                turn_time_gain=self.cfg.turn_time_gain,
                vertical_time_gain=0.6,
            )
            self._trajectory_cls = SplineTrajectory

    def _build_path_generator(self) -> AStarGatePathGenerator | CurveGatePathGenerator:
        """Construct the native-MPPI path generator selected by ``cfg.path_planner``.

        The A*-family planners (``astar`` / ``theta_star`` / ``d_star_lite``)
        share the grid-based construction; ``rrt_star`` (sampling-based) and
        ``curve`` (geometric detours) have their own parameter sets. All expose
        ``generate(obs, config)`` so the tracker is agnostic to the choice. To add
        a planner, import it above and register a ``case`` here.
        """
        gate_passing = GatePassingPathGenerator(
            gate_entry_distance=self.cfg.gate_entry_distance, max_nudge=0.0
        )
        grid_kwargs = dict(
            gate_passing_generator=gate_passing,
            grid_resolution=0.075,
            safety_margin=0.10,
            obstacle_radius=0.23,
            heuristic_weight=1.15,
            prune_path=False,
        )
        match self.cfg.path_planner:
            case "theta_star":
                return ThetaStarGatePathGenerator(**grid_kwargs)
            case "astar":
                return AStarGatePathGenerator(**grid_kwargs)
            case "d_star_lite":
                return DStarLiteGatePathGenerator(**grid_kwargs)
            case "rrt_star":
                return RRTStarGatePathGenerator(
                    gate_passing_generator=gate_passing,
                    collision_resolution=0.05,
                    safety_margin=0.10,
                    obstacle_radius=0.23,
                    prune_path=False,
                )
            case "curve":
                return CurveGatePathGenerator(gate_entry_distance=self.cfg.gate_entry_distance)
            case other:
                raise ValueError(
                    f"Unknown path_planner {other!r}; choose from: "
                    "theta_star, astar, d_star_lite, rrt_star, curve"
                )

    def _hover_sequence(self) -> NDArray[np.float64]:
        """Control sequence that holds hover thrust with zero attitude (the safe fallback)."""
        u = np.zeros((self.cfg.horizon, 4), dtype=float)
        u[:, 3] = self._hover_thrust
        return u

    def _plan_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Generate the path (+ timing) and reset the tracker state for a fresh plan."""
        waypoints = self._path_generator.generate(obs, self._config)
        if len(waypoints) < 2:
            # This is only expected after the final gate; keep a stationary reference valid.
            p = np.asarray(obs["pos"], dtype=float)
            waypoints = np.vstack((p, p + np.array([0.0, 0.0, 1e-4])))
        if self._post_proc is not None:
            # mpc_astar pipeline: densify + clearance tube, then clearance-aware timing.
            waypoints, clearance = self._post_proc.process(waypoints, obs, self._config)
            times = self._timing.compute(waypoints, clearance=clearance)
        else:
            times = self._timing.compute(waypoints)
        self.trajectory = self._trajectory_cls(waypoints, times, self._freq)
        self._trajectory_tick = 0
        self._control_sequence = self._hover_sequence()
        self._cache_layout(obs)
        self._pending_gate_replan_tick = None
        self._last_replan_tick = self._tick

    def _cache_layout(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Snapshot the gate/obstacle layout the current plan is based on."""
        self._last_target_gate = int(np.asarray(obs.get("target_gate", 0)).item())
        self._last_gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._last_gates_quat = np.asarray(obs["gates_quat"], dtype=float).copy()
        self._last_obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()

    def _should_replan(self, obs: dict[str, NDArray[np.floating]]) -> bool:
        """Event-triggered replanning: gate passed (with a delay), or sensed poses moved."""
        target_gate = int(np.asarray(obs.get("target_gate", 0)).item())
        if target_gate < 0:
            return False
        if target_gate != self._last_target_gate:
            # The current trajectory already contains all gates in order.  Do
            # not replan while physically inside the gate just passed: its
            # inflated frame can make A* snap the start to a discontinuous
            # point.  Pose-discovery changes below still trigger safe replans.
            self._last_target_gate = target_gate
            self._pending_gate_replan_tick = self._tick + int(0.4 * self._freq)

        if self._pending_gate_replan_tick is not None:
            if self._tick < self._pending_gate_replan_tick:
                return False
            return True

        min_ticks = max(1, int(self.cfg.replan_interval_s * self._freq))
        if self._tick - self._last_replan_tick < min_ticks:
            return False

        gates_pos = np.asarray(obs["gates_pos"], dtype=float)
        gates_quat = np.asarray(obs["gates_quat"], dtype=float)
        obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float)
        threshold = self.cfg.replan_position_threshold

        if np.any(np.linalg.norm(gates_pos - self._last_gates_pos, axis=1) > threshold):
            return True
        if np.any(np.linalg.norm(obstacles_pos - self._last_obstacles_pos, axis=1) > threshold):
            return True

        # q and -q represent the same rotation, hence the absolute dot product.
        quat_alignment = np.abs(np.sum(gates_quat * self._last_gates_quat, axis=1))
        return bool(np.any(quat_alignment < np.cos(0.5 * 0.08)))

    def _reference_horizon(
        self, position: NDArray[np.floating]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return (pos, vel, acc) reference samples for the next ``horizon`` steps.

        Re-anchors the trajectory clock to the nearest upcoming sample (never
        backwards) before slicing the horizon, so timing drift is corrected without
        the reference jumping to a different, distant path branch.
        """
        positions = self.trajectory._pos
        velocities = self.trajectory._vel

        # Correct forward timing drift without jumping to a different, distant
        # path branch.  Never move the reference clock backwards: around bends
        # the nearest sample can alternate between two branches, which otherwise
        # makes the target retreat and causes the drone to loiter in one area.
        lo = self._trajectory_tick
        hi = min(len(positions), self._trajectory_tick + int(0.8 * self._freq) + 1)
        if hi > lo:
            nearest = lo + int(np.argmin(np.linalg.norm(positions[lo:hi] - position, axis=1)))
            self._trajectory_tick = max(self._trajectory_tick, nearest)

        indices = np.minimum(
            self._trajectory_tick + np.arange(self.cfg.horizon), len(positions) - 1
        )
        ref_pos = positions[indices]
        ref_vel = velocities[indices]
        ref_acc = np.gradient(ref_vel, self._dt, axis=0)
        return ref_pos, ref_vel, ref_acc

    def _feedback_nominal(
        self,
        position: NDArray[np.floating],
        velocity: NDArray[np.floating],
        ref_pos: NDArray[np.floating],
        ref_vel: NDArray[np.floating],
        ref_acc: NDArray[np.floating],
    ) -> NDArray[np.float64]:
        """Roll out a PD policy to give MPPI a useful proposal distribution."""
        p = np.asarray(position, dtype=float).copy()
        v = np.asarray(velocity, dtype=float).copy()
        nominal = np.empty((self.cfg.horizon, 4), dtype=float)

        for i in range(self.cfg.horizon):
            # Acceleration-domain gains.  Their force-domain equivalents are
            # close to the repository's proven attitude PID baseline.
            pos_gain = np.array([6.0, 6.0, 18.0])
            vel_gain = np.array([7.0, 7.0, 8.0])
            acceleration = ref_acc[i] + pos_gain * (ref_pos[i] - p)
            acceleration += vel_gain * (ref_vel[i] - v)
            acceleration = np.clip(acceleration, -5.0, 5.0)
            yaw = self._path_yaw(ref_vel[i], nominal[i - 1, 2] if i else 0.0)
            nominal[i] = self._acceleration_to_attitude(acceleration, yaw)
            p, v = self._single_dynamics_step(p, v, nominal[i])
        return nominal

    @staticmethod
    def _path_yaw(velocity: NDArray[np.floating], fallback: float) -> float:
        """Desired yaw for the nominal rollout: always zero.

        Translation is fully controllable at zero yaw and the course baseline
        controllers use the same convention. Keeping yaw fixed also avoids angle
        wrapping injecting unnecessary attitude transients. Kept as a function (with
        unused ``velocity``/``fallback`` parameters) so a heading-following policy
        can be dropped in later without changing the caller.
        """
        del velocity, fallback
        return 0.0

    def _acceleration_to_attitude(
        self, acceleration: NDArray[np.floating], yaw: float
    ) -> NDArray[np.float64]:
        """Differentially-flat inverse: desired acceleration + yaw -> [roll, pitch, yaw, thrust]."""
        force = self._mass * (np.asarray(acceleration, dtype=float) + [0.0, 0.0, self._gravity])
        thrust = float(np.linalg.norm(force))
        if thrust < 1e-8:
            return np.array([0.0, 0.0, yaw, self._thrust_min])

        z_axis = force / thrust
        x_heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        y_axis = np.cross(z_axis, x_heading)
        y_norm = np.linalg.norm(y_axis)
        if y_norm < 1e-8:
            y_axis = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        else:
            y_axis /= y_norm
        x_axis = np.cross(y_axis, z_axis)

        # XYZ Euler angles from the desired rotation matrix.
        pitch = np.arcsin(np.clip(-x_axis[2], -1.0, 1.0))
        roll = np.arctan2(y_axis[2], z_axis[2])
        command = np.array([roll, pitch, yaw, thrust], dtype=float)
        return np.clip(command, self._u_low, self._u_high)

    def _single_dynamics_step(
        self,
        position: NDArray[np.floating],
        velocity: NDArray[np.floating],
        command: NDArray[np.floating],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Advance one point-mass Euler step (with linear drag) under a single command."""
        z_axis = self._body_z_axis(command[None, :])[0]
        acceleration = command[3] / self._mass * z_axis
        acceleration[2] -= self._gravity
        acceleration -= 0.12 * velocity
        next_position = position + velocity * self._dt + 0.5 * acceleration * self._dt**2
        next_velocity = velocity + acceleration * self._dt
        return next_position, next_velocity

    @staticmethod
    def _body_z_axis(commands: NDArray[np.floating]) -> NDArray[np.float64]:
        """Body z-axis (thrust direction) in world frame for a batch of [r, p, y, T] commands."""
        roll, pitch, yaw = commands[:, 0], commands[:, 1], commands[:, 2]
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.column_stack((cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr))

    def _sample_noise(self) -> NDArray[np.float64]:
        """Antithetic Gaussian control perturbations: ``[+noise, -noise]`` pairs.

        Mirroring each sample around zero halves the variance of the MPPI cost
        estimate for the same sample budget (a standard antithetic-variate trick),
        at the cost of only using ``num_samples // 2`` independent draws.
        """
        half = self.cfg.num_samples // 2
        noise = self._rng.normal(size=(half, self.cfg.horizon, 4)) * self._noise_std
        noise = np.concatenate((noise, -noise), axis=0)
        if len(noise) < self.cfg.num_samples:
            noise = np.concatenate((noise, np.zeros((1, self.cfg.horizon, 4))), axis=0)
        return noise[: self.cfg.num_samples]

    def _rollout_cost(
        self,
        position: NDArray[np.floating],
        velocity: NDArray[np.floating],
        controls: NDArray[np.floating],
        ref_pos: NDArray[np.floating],
        ref_vel: NDArray[np.floating],
        obstacles: NDArray[np.floating],
        gate_position: NDArray[np.floating] | None,
        gate_rotation: NDArray[np.floating] | None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Roll out every sampled control sequence and return (cost, path) per sample.

        Forward-simulates the point-mass dynamics for all ``controls`` in parallel
        (vectorized over the sample axis) and accumulates a per-step cost from:
        position/velocity tracking error, control effort and control-rate (smoothness),
        obstacle clearance (quadratic penalty inside ``obstacle_clearance`` plus a hard
        indicator below 12 cm), the gate-opening corridor (only near the target gate's
        plane, via ``gate_position``/``gate_rotation``), and a soft world-bounds
        penalty; plus a terminal position-error cost. The fixed weights (35.0, 65.0,
        1.8, ...) were hand-tuned for the 50 Hz attitude interface and are not
        currently exposed as parameters.
        """
        samples = len(controls)
        p = np.repeat(np.asarray(position, dtype=float)[None, :], samples, axis=0)
        v = np.repeat(np.asarray(velocity, dtype=float)[None, :], samples, axis=0)
        cost = np.zeros(samples, dtype=float)
        paths = np.empty((samples, self.cfg.horizon, 3), dtype=float)

        previous = np.repeat(self._control_sequence[0][None, :], samples, axis=0)
        thrust_scale = max(self._hover_thrust, 1e-6)

        for i in range(self.cfg.horizon):
            u = controls[:, i]
            z_axis = self._body_z_axis(u)
            acceleration = u[:, 3, None] / self._mass * z_axis
            acceleration[:, 2] -= self._gravity
            acceleration -= 0.12 * v
            p += v * self._dt + 0.5 * acceleration * self._dt**2
            v += acceleration * self._dt
            paths[:, i] = p

            pos_error = p - ref_pos[i]
            vel_error = v - ref_vel[i]
            cost += 35.0 * np.sum(pos_error[:, :2] ** 2, axis=1)
            cost += 65.0 * pos_error[:, 2] ** 2
            cost += 1.8 * np.sum(vel_error**2, axis=1)
            cost += 0.08 * np.sum(u[:, :2] ** 2, axis=1)
            cost += 0.05 * ((u[:, 3] - self._hover_thrust) / thrust_scale) ** 2

            delta_u = (u - previous) / np.array([0.2, 0.2, 0.3, thrust_scale])
            cost += 0.04 * np.sum(delta_u**2, axis=1)
            previous = u

            if obstacles.size:
                distance_xy = np.linalg.norm(p[:, None, :2] - obstacles[None, :, :2], axis=2)
                intrusion = np.maximum(self.cfg.obstacle_clearance - distance_xy, 0.0)
                cost += 1400.0 * np.sum(intrusion**2, axis=1)
                cost += 60.0 * np.sum(distance_xy < 0.12, axis=1)

            if gate_position is not None and gate_rotation is not None:
                # In gate coordinates x is normal to the opening.  Near its
                # plane, strongly prefer the central 20 cm by 20 cm corridor;
                # this accounts for the drone body, not only its center point.
                local = (p - gate_position) @ gate_rotation
                plane_weight = np.exp(-0.5 * (local[:, 0] / self.cfg.gate_constraint_distance) ** 2)
                lateral_excess = np.maximum(np.abs(local[:, 1]) - 0.05, 0.0)
                vertical_excess = np.maximum(np.abs(local[:, 2]) - 0.05, 0.0)
                cost += 5000.0 * plane_weight * (lateral_excess**2 + vertical_excess**2)

            below = np.maximum(self._bounds_low + [0.05, 0.05, 0.06] - p, 0.0)
            above = np.maximum(p - (self._bounds_high - [0.05, 0.05, 0.05]), 0.0)
            cost += 600.0 * np.sum(below**2 + above**2, axis=1)

        terminal_error = p - ref_pos[-1]
        cost += 45.0 * np.sum(terminal_error**2, axis=1)
        return cost, paths

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.float32]:
        """Return ``[roll, pitch, yaw, collective_thrust]``."""
        if self._should_replan(obs):
            self._plan_trajectory(obs)

        position = np.asarray(obs["pos"], dtype=float)
        velocity = np.asarray(obs["vel"], dtype=float)
        ref_pos, ref_vel, ref_acc = self._reference_horizon(position)
        target_gate = int(np.asarray(obs.get("target_gate", -1)).item())
        if target_gate == len(obs["gates_pos"]) - 1 and self._pending_gate_replan_tick is None:
            # Recover time spent on online replans while the tighter gate cost
            # below still protects the final opening.  Return to nominal speed
            # before the gate rather than carrying the cruise boost through it.
            gate_distance = np.linalg.norm(position - obs["gates_pos"][target_gate])
            cruise_blend = np.clip((gate_distance - 0.55) / 0.65, 0.0, 1.0)
            obstacle_distance = np.min(
                np.linalg.norm(position[None, :2] - obs["obstacles_pos"][:, :2], axis=1)
            )
            cruise_blend *= np.clip((obstacle_distance - 0.30) / 0.25, 0.0, 1.0)
            ref_vel = (1.0 + 0.50 * cruise_blend) * ref_vel
        feedback = self._feedback_nominal(position, velocity, ref_pos, ref_vel, ref_acc)
        self._control_sequence = np.clip(
            0.20 * self._control_sequence + 0.80 * feedback, self._u_low, self._u_high
        )

        obstacles = np.asarray(obs.get("obstacles_pos", []), dtype=float).reshape(-1, 3)
        cost_gate = target_gate
        if self._pending_gate_replan_tick is not None and target_gate > 0:
            # Continue constraining the opening until the vehicle has cleared
            # the gate whose passage triggered the delayed replan.
            cost_gate = target_gate - 1
        if 0 <= cost_gate < len(obs["gates_pos"]):
            gate_position = np.asarray(obs["gates_pos"][cost_gate], dtype=float)
            gate_rotation = Rotation.from_quat(obs["gates_quat"][cost_gate]).as_matrix()
        else:
            gate_position = None
            gate_rotation = None
        for _ in range(self.cfg.iterations):
            noise = self._sample_noise()
            candidates = np.clip(
                self._control_sequence[None, :, :] + noise, self._u_low, self._u_high
            )
            costs, paths = self._rollout_cost(
                position,
                velocity,
                candidates,
                ref_pos,
                ref_vel,
                obstacles,
                gate_position,
                gate_rotation,
            )
            shifted_cost = costs - np.min(costs)
            weights = np.exp(-shifted_cost / max(self.cfg.temperature, 1e-6))
            weights /= np.sum(weights) + 1e-12
            applied_noise = candidates - self._control_sequence[None, :, :]
            self._control_sequence += np.tensordot(weights, applied_noise, axes=(0, 0))
            self._control_sequence = np.clip(self._control_sequence, self._u_low, self._u_high)
            self._candidate_paths = paths
            self._predicted_path = paths[int(np.argmin(costs))]

        return self._control_sequence[0].astype(np.float32)

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Advance the reference and warm-start sequence after an environment step."""
        self._tick += 1
        self._trajectory_tick = min(self._trajectory_tick + 1, len(self.trajectory._pos) - 1)
        self._control_sequence[:-1] = self._control_sequence[1:]
        self._control_sequence[-1] = self._control_sequence[-2]
        return False

    def episode_callback(self) -> None:
        """Discard the optimized sequence after an episode."""
        self._control_sequence = self._hover_sequence()

    def episode_reset(self) -> None:
        """Reset counters used by the receding-horizon reference."""
        self._tick = 0
        self._trajectory_tick = 0

    def render_callback(self, sim: Sim) -> None:
        """Visualize the MPPI rollout cloud, global reference and best rollout."""
        trajectory = self.trajectory._pos

        # Downsample the full reference to at most 300 points.  Keeping a hard
        # upper bound matters because a replan may change the trajectory length
        # after the viewer capacity has already been fixed.
        trajectory_indices = np.linspace(
            0, len(trajectory) - 1, min(len(trajectory), 300), dtype=int
        )
        trajectory_vis = trajectory[trajectory_indices]

        # Keep every sampled rollout, but draw fewer temporal points on each one.
        rollout_indices = np.arange(0, self.cfg.horizon, self.cfg.visualization_rollout_stride)
        if rollout_indices[-1] != self.cfg.horizon - 1:
            rollout_indices = np.append(rollout_indices, self.cfg.horizon - 1)

        # The viewer capacity is fixed when its first frame is created.  This
        # callback runs immediately before that frame, so reserve enough room
        # for the complete rollout cloud and the reference overlays.
        if sim.viewer is None:
            rollout_geoms = len(self._candidate_paths) * (len(rollout_indices) - 1)
            overlay_geoms = (
                max(0, len(trajectory_vis) - 1) + max(0, len(self._predicted_path) - 1) + 1
            )
            # mjv_updateScene also consumes slots for the track, drone and other
            # model geometry.  Leave extra headroom for MuJoCo decorations.
            scene_geoms = int(sim.mj_model.ngeom) + 512
            sim.max_visual_geom = max(
                sim.max_visual_geom, rollout_geoms + overlay_geoms + scene_geoms
            )

        # Cyan: all candidate rollouts from the latest MPPI sampling step.
        for path in self._candidate_paths:
            draw_line(
                sim,
                path[rollout_indices],
                rgba=(0.1, 0.65, 1.0, 0.10),
                start_size=1.0,
                end_size=1.0,
            )

        # Green: the global A* path after spline interpolation.
        if len(trajectory_vis) >= 2:
            draw_line(sim, trajectory_vis, rgba=(0.0, 1.0, 0.0, 0.75), start_size=2.0, end_size=2.0)

        # The lowest-cost sampled rollout from the latest MPPI update.
        if len(self._predicted_path) >= 2:
            draw_line(
                sim, self._predicted_path, rgba=(1.0, 0.1, 0.0, 0.95), start_size=3.0, end_size=3.0
            )

        # Current point on the time-parameterized reference trajectory.
        index = min(self._trajectory_tick, len(trajectory) - 1)
        draw_points(sim, trajectory[index].reshape(1, 3), rgba=(0.0, 0.3, 1.0, 1.0), size=0.025)
