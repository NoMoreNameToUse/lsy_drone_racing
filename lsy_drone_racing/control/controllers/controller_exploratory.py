"""This module implements an example MPC using attitude control for a quadrotor.

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

## Modular imports
from lsy_drone_racing.control.controllers.modules.path_generator import WaypointPathGenerator, GatePassingPathGenerator, AStarGatePathGenerator
from lsy_drone_racing.control.controllers.modules.timing_module import UniformTiming, DistanceTiming
from lsy_drone_racing.control.controllers.modules.trajectory_module import SplineTrajectory

## Debug stuff
from pathlib import Path
import time


if TYPE_CHECKING:
    from crazyflow import Sim
    from numpy.typing import NDArray


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
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

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
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            50.0,  # pos
            50.0,  # pos
            400.0,  # pos
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            10.0,  # vel
            10.0,  # vel
            10.0,  # vel
            5.0,  # drpy
            5.0,  # drpy
            5.0,  # drpy
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            50.0,  # thrust
        ]
    )

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

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
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
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)

        # modules
        # self.path_gen = AStarGatePathGenerator()
        # self.path_gen = WaypointPathGenerator()
        self.path_gen = AStarGatePathGenerator(
            grid_resolution=0.05,
            safety_margin=0.05,
            obstacle_radius=0.21,
            prune_path=True,
            )
        ##self.timing = UniformTiming()

        self.timing = DistanceTiming(
            nominal_speed=1,
            min_segment_time=0.2,
        )

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
            self.trajectory._pos,
            np.asarray(obs["obstacles_pos"], dtype=float),
            min_clearance=0.25,
        )
        self._executed_pos = []

        ## End debug outputs 
        self._N = 25
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
        self._tick_max = len(self.trajectory._pos) - 1 - self._N
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

        if self._should_replan(obs):
            self._replan_trajectory(obs)

        if self._tick >= self._tick_max:
            self._finished = True

        traj = self.trajectory.sample_horizon(self._tick, self._N)

        assert traj["pos"].shape == (self._N, 3), traj["pos"].shape
        assert traj["vel"].shape == (self._N, 3), traj["vel"].shape
        assert traj["yaw"].shape == (self._N,), traj["yaw"].shape

        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))

        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = traj["pos"]
        yref[:, 6:9] = traj["vel"]
        yref[:, 5] = traj["yaw"]
        # zero drpy

        # Setting input reference (index > self._nx)
        # zero rpy
        # hover thrust
        yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = traj["pos_terminal"]
        yref_e[6:9] = traj["vel_terminal"]
        yref_e[5] = traj["yaw_terminal"]
        # zero drpy
        self._acados_ocp_solver.set(self._N, "y_ref", yref_e)

        # Solving problem and getting first input
        self._acados_ocp_solver.solve()
        u0 = self._acados_ocp_solver.get(0, "u")

        return u0

    def _replan_trajectory(self, obs):
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
            f"path={1000*(t1-t0):.1f} ms, "
            f"traj={1000*(t2-t1):.1f} ms, "
            f"waypoints={len(self._raw_waypoints)}"
        )
        self._tick = 0
        self._tick_max = len(self.trajectory._pos) - 1 - self._N
        self._finished = False

        self._track_gates = np.asarray(obs.get("gates_pos", []), dtype=float)
        self._track_obstacles = np.asarray(obs.get("obstacles_pos", []), dtype=float)

        self._last_target_gate = int(np.asarray(obs["target_gate"]).item())
        self._last_gates_pos = np.asarray(obs["gates_pos"], dtype=float).copy()
        self._last_gates_quat = np.asarray(obs["gates_quat"], dtype=float).copy()
        self._last_obstacles_pos = np.asarray(obs["obstacles_pos"], dtype=float).copy()

        self._last_replan_tick = self._global_tick
        self._replan_count += 1

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
