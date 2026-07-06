"""Attitude MPC tracking a trajectory from the modular planning pipeline.

Final racing controller: a modular pipeline (gate-aware A* path generator ->
clearance post-processor -> dynamics-aware timing -> spline trajectory) produces
the reference, and an acados attitude MPC (extended from the course's example
``attitude_mpc.py``) tracks it. The pipeline replans event-triggered when newly
sensed gate/obstacle poses deviate from the plan.

All hyperparameters are exposed as module-level dicts so the tuning harness
(``utility/sim/tune_mpc.py``) can override them per trial; the committed values
are the Optuna results aiming for best lap time subject to a min 60% success-rate.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from crazyflow.sim.visualize import draw_line, draw_points
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

# modular pipeline imports
from lsy_drone_racing.control.controllers.modules.path_generator_improved import (
    AStarImprovedPathGenerator,
)
from lsy_drone_racing.control.controllers.modules.post_processing import PathPostProcessor
from lsy_drone_racing.control.controllers.modules.timing_module_improved import DynamicTiming
from lsy_drone_racing.control.controllers.modules.trajectory_module_improved import (
    ImprovedSplineTrajectory,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from crazyflow import Sim
    from numpy.typing import NDArray

# --- Tunable hyperparameters (module-level so the tuning harness can override) ---

# MPC cost weights (state: pos xyz, rpy, vel xyz, drpy; input: rpy, thrust) and horizon.
MPC_HYPERPARAMS = {
    "N": 20,
    "q_diag": (18.32, 18.32, 488.73, 0.180, 0.180, 0.180, 1.297, 1.297, 1.297, 0.313, 0.313, 0.313),
    "r_diag": (1.0, 1.0, 1.0, 110.85),
}

# Trajectory timing (DynamicTiming): speed caps and clearance/reversal shaping.
TIMING_HYPERPARAMS = {
    "v_max": 2.83,
    "a_max": 3.19,
    "a_lat_max": 4.85,
    "clearance_ref": 0.495,
    "clearance_floor_speed": 1.606,
    "reversal_speed": 0.833,
    "min_segment_time": 0.01,
}

# Path-generator geometry (A* collision inflation).
PATHGEN_HYPERPARAMS = {"safety_margin": 0.013, "obstacle_radius": 0.140}

# Post-processor (densify, pole repulsion, hard clearance floor).
POSTPROC_HYPERPARAMS = {
    "enabled": True,
    "resample_step": 0.2,
    "smooth_iterations": 0,
    "smooth_weight": 0.4,
    "smooth_max_step": 0.03,
    "repulse_gain": 0.03,
    "iterations": 3,
    "max_step": 0.02,
    "influence": 0.6,
    "min_clearance": 0.18,
    "pole_radius": 0.15,
}


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    # For more info on the models, check out https://github.com/utiasDSL/drone-models
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "basic_example_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_ocp_solver(
    Tf: float,
    N: int,
    parameters: dict,
    q_diag: Sequence[float] | None = None,
    r_diag: Sequence[float] | None = None,
    verbose: bool = False,
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver.

    ``q_diag`` / ``r_diag`` default to ``MPC_HYPERPARAMS`` so a plain call reproduces
    the tuned controller; the tuning harness passes explicit weight vectors. Each
    horizon ``N`` gets its own codegen name/json so several compiled solvers (one
    per N) can coexist in-process without clobbering each other's shared library.
    """
    q_diag = MPC_HYPERPARAMS["q_diag"] if q_diag is None else q_diag
    r_diag = MPC_HYPERPARAMS["r_diag"] if r_diag is None else r_diag

    ocp = AcadosOcp()

    # Set model (unique name per horizon so per-N libs don't collide).
    ocp.model = create_acados_model(parameters)
    ocp.model.name = f"mpc_attitude_N{N}"

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights (state: pos xyz, rpy, vel xyz, drpy; input: rpy, thrust).
    Q = np.diag(np.asarray(q_diag, dtype=float))
    # Input weights (reference is upright orientation and hover thrust).
    R = np.diag(np.asarray(r_diag, dtype=float))

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references. We will overwrite these later to track the trajectory
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < ~30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < ~30°, plus collective thrust bounds)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file=f"c_generated_code/{ocp.model.name}.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AStarAttitudeMPC(Controller):
    """Attitude MPC tracking the A* pipeline's trajectory, with online replanning."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Build the planning pipeline, plan the first trajectory, and set up the solver.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._config = config
        self._N = MPC_HYPERPARAMS["N"]

        # Pipeline modules (shared design with controller_rl_astar; only the tracker
        # differs). Parameters come from the module-level hyperparameter dicts.
        self.path_gen = AStarImprovedPathGenerator(
            grid_resolution=0.05, prune_path=True, **PATHGEN_HYPERPARAMS
        )
        self.timing = DynamicTiming(**TIMING_HYPERPARAMS)
        self.post_proc = PathPostProcessor(**POSTPROC_HYPERPARAMS)

        # Initial plan from the nominal track.
        self._replan_count = 0
        self._min_replan_interval_ticks = int(0.5 * config.env.freq)  # at most every 0.5 s
        self._last_replan_tick = 0
        self._plan_trajectory(obs)
        self._warn_low_clearance(self.trajectory._pos, self._track_obstacles)

        # acados solver over the tuned horizon.
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._global_tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust
            [r_des, p_des, y_des, t_des] as a numpy array.
        """
        self._global_tick += 1

        # modular replan logic
        if self._should_replan(obs):
            self._replan(obs)

        if self._tick >= self._tick_max:
            self._finished = True

        # Reference over the horizon, sampled from the current trajectory.
        traj = self.trajectory.sample_horizon(self._tick, self._N)
        assert traj["pos"].shape == (self._N, 3), traj["pos"].shape
        assert traj["vel"].shape == (self._N, 3), traj["vel"].shape
        assert traj["yaw"].shape == (self._N,), traj["yaw"].shape

        # Current state as initial condition.
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Stage references: position/velocity/yaw from the trajectory, zero rpy and
        # body rates, hover thrust as input reference.
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = traj["pos"]
        yref[:, 5] = traj["yaw"]
        yref[:, 6:9] = traj["vel"]
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Terminal reference.
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = traj["pos_terminal"]
        yref_e[5] = traj["yaw_terminal"]
        yref_e[6:9] = traj["vel_terminal"]
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Solve and apply the first input.
        self._acados_ocp_solver.solve()
        return self._acados_ocp_solver.get(0, "u")

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

        # Full trajectory, heavily downsampled
        draw_line(sim, self.trajectory._pos[::20], rgba=(0.0, 1.0, 0.0, 0.8))

        # Current MPC horizon, short enough to draw directly
        horizon = self.trajectory.sample_horizon(self._tick, self._N)["pos"]
        draw_line(sim, horizon, rgba=(1.0, 0.0, 0.0, 0.7))

        # Current setpoint
        i = min(self._tick, len(self.trajectory._pos) - 1)
        draw_points(
            sim, self.trajectory._pos[i].reshape(1, -1), rgba=(1.0, 0.0, 0.0, 1.0), size=0.02
        )

    # ---------------------------- pipeline trajectory planning ----------------------------
    def _plan_trajectory(self, obs: dict[str, NDArray[np.floating]]) -> None:
        """Run the pipeline (path -> post-process -> timing -> spline) from ``obs``.

        Rebuilds ``self.trajectory`` and remembers the gate/obstacle poses the plan
        was based on, so ``_should_replan`` can detect when they change.
        """
        waypoints = self.path_gen.generate(obs, self._config)
        waypoints, self._clearance = self.post_proc.process(waypoints, obs, self._config)
        self._raw_waypoints = np.asarray(waypoints, dtype=float)
        t = self.timing.compute(self._raw_waypoints, clearance=self._clearance)
        self.trajectory = ImprovedSplineTrajectory(self._raw_waypoints, t, self._config.env.freq)
        self._tick_max = len(self.trajectory._pos) - 1 - self._N

        # Poses the plan is based on (for change detection and rendering).
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

    # ---------------------------- debugging ----------------------------
    @staticmethod
    def _warn_low_clearance(
        traj_pos: NDArray[np.floating], obstacles: NDArray[np.floating], min_clearance: float = 0.25
    ) -> None:
        """Print a warning for every obstacle the trajectory passes closer than the floor."""
        for j, obstacle in enumerate(obstacles):
            d_xy = np.linalg.norm(traj_pos[:, :2] - obstacle[:2], axis=1)
            idx = int(np.argmin(d_xy))
            if d_xy[idx] < min_clearance:
                print(
                    f"WARNING: trajectory too close to obstacle {j}: "
                    f"min_xy={d_xy[idx]:.3f}, idx={idx}, "
                    f"traj_pos={traj_pos[idx]}, obs={obstacle}"
                )
