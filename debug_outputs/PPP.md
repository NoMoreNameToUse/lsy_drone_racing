# Progress Presentation PPT Plan

本文件是当前 progress presentation 的推荐内容。它已经按你的项目现状更新：

- 你的程序跑过 **Level 2** 和 **Level 3**。
- Level 2 比较过 `controller_exploratory.py`、`controller_rl.py`、`attitude_mpcc.py`。
- Level 3 只保留 `controller_exploratory.py` 和 `controller_rl.py`，放弃 `attitude_mpcc.py`。
- 汇报重点不只是 “我们写了 controller”，而是 “我们建立了模块化 planning/debug/control pipeline，并比较 Level 2 和 Level 3 下不同 controller 的表现”。

建议总共做 **6 页**：1 页封面 + 5 页正式内容。这样仍然贴合模板要求：

- Methodology / Approach: 3 slides, about 60%
- Progress to Date: 1 slide, about 20%
- Project Milestones and SMART Goals: 1 slide, about 20%

## Overall Narrative

英文主线可以这样讲：

> We built a modular drone racing pipeline that separates path planning, timing, trajectory generation, and control. We evaluated three controller families on Level 2: MPC, RL, and MPCC. For Level 3, where the track is randomized and online planning becomes more important, we kept the modular MPC and RL controllers, while dropping MPCC because it was too sensitive to reference quality, replanning changes, and solver feasibility.

中文理解：

> 我们建立了一个模块化无人机竞速控制流程，把路径规划、时间分配、轨迹生成和控制分开。Level 2 中我们比较了 MPC、RL 和 MPCC 三种路线。Level 3 中赛道随机化，对 online planning 和 replanning 的要求更高，所以保留了模块化 MPC 和 RL，放弃了 MPCC，因为它对参考轨迹质量、replanning 变化和优化器可行性过于敏感。

## Slide Overview

| Slide | Title | Format | Purpose |
|---|---|---|---|
| 1 | Drone Racing Control Progress | Cover slide | 项目名、组员、日期 |
| 2 | Modular Control Pipeline | Flowchart | 展示 Path / Timing / Trajectory / Control 的模块化架构 |
| 3 | 3D Debug Path Model | 3D figure + planner table | 展示 gates、obstacles、inflated safety zones 和 planner comparison |
| 4 | Controller Designs and Level Applicability | Controller comparison table | 解释 RL / MPC / MPCC 的模块组成，以及为什么 Level 3 放弃 MPCC |
| 5 | Progress: Level 2 vs Level 3 | Speed + success-rate tables | 展示当前结果和 Level 2 / Level 3 差异 |
| 6 | Challenges and SMART Next Steps | Challenge table + milestone table | 说明 Level 3 难点和后续计划 |

## Slide 1: Drone Racing Control Progress

Use the template cover page.

Recommended content:

- Project title: Drone Racing Control Progress
- Team members
- Date
- Course/lab information if needed

Do not include technical details here.

## Slide 2: Modular Control Pipeline

这页是 Methodology 的第 1 页。建议用 **流程图**，不要堆文字。

Recommended flow:

```text
Observation
(drone state, gates, obstacles)
        |
        v
Path Module
(A*, Theta*, RRT*, D* Lite, CurveGate)
        |
        v
Timing Module
(UniformTiming / DistanceTiming / MotionAwareTiming)
        |
        v
Trajectory Module
(PCHIP SplineTrajectory or arc-length spline)
        |
        v
Control Module
(MPC / RL policy / MPCC)
        |
        v
Attitude + Thrust Command
```

Recommended bullets:

- The pipeline is modular: path planning, timing, trajectory generation, and control can be swapped independently.
- This allows direct comparison between classical control, RL control, and racing-oriented MPCC.
- Replanning is triggered by target gate changes or changes in observed gates/obstacles.

Suggested speaking points:

- We intentionally designed the controller around modules instead of one fixed hand-written trajectory.
- This made it possible to run different planners and different controllers on the same task.
- The same debug outputs can be used to compare planned paths and executed trajectories.

## Slide 3: 3D Debug Path Model

这页是 Methodology 的第 2 页。重点是说明你们如何建模、可视化和比较路径规划。

Recommended visual:

- Use `debug_outputs/compare_debug_paths.py`.
- Show a 3D figure with:
  - gate square frames
  - gate inner openings
  - obstacle vertical cylinders
  - inflated obstacle safety cylinders
  - raw waypoint paths
  - smooth trajectories
  - trajectory length labels

Recommended command:

```bash
cd /home/deng0517/Study/Drone_racing/Caogao/lsy_drone_racing
pixi run python debug_outputs/compare_debug_paths.py --output-html /tmp/path_compare_with_controls.html
```

The generated HTML has checkboxes for manually showing or hiding each planner path.

Recommended table:

| Planner | Waypoints | Traj. Length | Comment |
|---|---:|---:|---|
| A* | 22 | 12.80 m | Short grid-based path |
| Theta* | 22 | 12.76 m | Similar length, potentially smoother |
| RRT* | 25 | 15.72 m | More exploratory, less direct |
| D* Lite | 23 | 12.74 m | Useful for replanning idea |
| CurveGate | 29 | 14.11 m | More conservative around gates |

Note:

- The exact values depend on the saved `.npz` debug files.
- Use the displayed values if you regenerate the data.
- The simple XY obstacle clearance metric is a debug comparison, not a final safety proof.

Suggested speaking points:

- The 3D debug model helps us check if the path passes through gate openings instead of cutting through gate frames.
- Inflated obstacle cylinders make the safety margin visible.
- Path length and shape can be compared before running the controller.

## Slide 4: Controller Designs and Level Applicability

这页是 Methodology 的第 3 页。建议用 **controller comparison table**，并明确区分 Level 2 和 Level 3。

Recommended table:

| Controller | Module Composition | Used in Level 2 | Used in Level 3 | Why |
|---|---|---:|---:|---|
| `controller_exploratory.py` | AStarGatePathGenerator + DistanceTiming + SplineTrajectory + MPC/acados | Yes | Yes | Interpretable, stable baseline, easier to replan |
| `controller_rl.py` | RRT*/planner module + DistanceTiming + SplineTrajectory + PPO RL policy | Yes | Yes | Fast inference, good empirical success, flexible with replanned trajectory samples |
| `attitude_mpcc.py` | PathPlanner + gate waypoints + CubicSpline/arc-length + MPCC/acados | Yes | No | Strong on fixed/reference-quality paths, but too sensitive for randomized online Level 3 |

Recommended detailed notes:

### RL controller

```text
Path:       RRTStarGatePathGenerator / selectable planner
Timing:     DistanceTiming
Trajectory: SplineTrajectory
Control:    PPO RL policy
```

Strength:

- Fast control output after policy inference.
- Works well when trajectory samples are reasonable.
- Current best candidate for Level 2 and Level 3 based on success rate.

Limitation:

- Weak formal safety guarantee.
- Policy can suffer from distribution shift when replanning changes the trajectory shape.
- Current yaw command is simplified / forced toward zero.
- RL policy has no explicit obstacle reasoning; it relies on planned trajectory samples.

### MPC controller

```text
Path:       WaypointPathGenerator / GatePassingPathGenerator / AStarGatePathGenerator
Timing:     UniformTiming / DistanceTiming
Trajectory: SplineTrajectory
Control:    MPC / acados
```

Strength:

- More interpretable than RL.
- Easier to debug tracking error and trajectory following.
- Useful fallback controller.

Limitation:

- Slower than RL in current tests.
- Still depends strongly on trajectory smoothness and timing.

### MPCC controller

```text
Path:       PathPlanner
Waypoints:  gate-based waypoints
Trajectory: CubicSpline + arc-length parameterization
Control:    MPCC / acados
```

Strength:

- Directly optimizes progress along the path.
- Conceptually suitable for racing because it balances contour error, lag error, and progress reward.

Why it was dropped for Level 3:

- Level 3 randomizes the full track, so the reference path changes more aggressively.
- MPCC depends on a high-quality arc-length reference; poor or changing references can make contour tracking unstable.
- The nonlinear optimization problem is sensitive to cost weights, local projection, gate-stage logic, and solver feasibility.
- Online replanning can change the theta/reference distribution abruptly, which is harder for MPCC than for the simpler modular MPC/RL setup.
- In practice it did not give reliable enough behavior compared with RL/MPC on Level 3.

Suggested speaking points:

- MPCC is attractive theoretically, but Level 3 emphasizes robustness under randomized tracks and online replanning.
- For this reason we kept the simpler modular MPC and RL controllers for Level 3.
- This is a design decision based on reliability, not because MPCC is unimportant.

## Slide 5: Progress: Level 2 vs Level 3

这页是 Progress to Date。建议把你目前的结果做成 **两个小表格**：一个速度，一个成功率。

Important:

- 这里的数值先按你当前大纲填写。
- 如果后续有正式 benchmark，请替换为最终测量值。
- 建议在 PPT 中标注：`preliminary simulation results`。

### Speed / Completion Time

| Level | RL | MPC | MPCC |
|---|---:|---:|---:|
| Level 2 | < 7 s | < 10 s | < 9 s |
| Level 3 | < 9 s | < 14 s | Not used |

### Success Rate

| Level | RL | MPC | MPCC |
|---|---:|---:|---:|
| Level 2 | > 90% | > 70% | > 50% |
| Level 3 | > 90% | > 70% | Not used |

Recommended visual:

- Use one execution/debug figure, not the same multi-planner image from Slide 3.
- Good visual choices:
  - planned trajectory vs executed path
  - Level 2 and Level 3 example trajectories side by side
  - one screenshot for RL execution if it is the best-performing controller

Suggested speaking points:

- RL is currently the strongest empirical candidate in both Level 2 and Level 3.
- MPC remains useful as a more interpretable and stable fallback.
- MPCC was tested in Level 2 but not carried to Level 3 because reliability was worse under randomized online-planning conditions.
- The main progress is that the same modular framework can run both fixed-track and randomized-track scenarios.

## Slide 6: Challenges and SMART Next Steps

这页是 Milestones and SMART Goals。建议上半部分放 Level 3 challenges，下半部分放 next steps。

### Level 3 Challenges

| Challenge | Why It Matters | Current Response |
|---|---|---|
| Weak safety guarantee for RL | RL outputs actions without explicit constraint solving | Keep planner safety margins and consider MPC fallback |
| Policy distribution shift | Replanning changes the trajectory samples seen by the policy | Benchmark over randomized seeds |
| Randomized mass/inertia | Level 3 changes dynamics properties | Prefer robust controller settings and evaluate success rate |
| Trajectory dependence | RL relies heavily on local trajectory samples | Improve trajectory smoothing and timing |
| Yaw simplification | Current RL command simplifies yaw behavior | Add yaw-aware reference or training improvement |
| No explicit obstacle reasoning in RL policy | Obstacle avoidance is only represented through planned path | Strengthen path planning and clearance validation |
| Replanning discontinuity | New path can change abruptly after detection updates | Add smoothing or continuity constraints between old and new references |

### SMART Next Steps

| Area | Goal | Success Metric | Responsible |
|---|---|---|---|
| Path planning | Add curvature-aware smoothing | Reduced sharp turns in planned trajectory | Planning |
| Path planning | Tune adaptive obstacle margin | No collisions in randomized Level 3 seeds | Planning |
| Path planning | Add clearance validation | Report min clearance for each run | Evaluation |
| Control | Fine-tune RL behavior | Maintain >90% success with lower average time | Control |
| Control | Add safety fallback | Switch to MPC or safer mode when trajectory becomes risky | Control |
| Evaluation | Benchmark Level 2 / Level 3 | Fixed table of speed and success rate over multiple seeds | Evaluation |
| Evaluation | Compare speed vs success rate | Choose final controller based on reliability, not only speed | All |

Suggested speaking points:

- The next step is not to add many unrelated controller variants.
- The priority is robust evaluation across Level 2 and Level 3.
- We will choose the final controller based on speed and success rate together.

## Figure Usage Summary

| Slide | Figure Type | Main Message |
|---|---|---|
| Slide 3 | Multi-planner 3D comparison | How we model and compare planning methods |
| Slide 5 | Planned vs executed trajectory, or Level 2 vs Level 3 execution examples | What currently works and how performance differs |

Avoid using the same image on Slide 3 and Slide 5.

- Slide 3 is methodology/debug modeling.
- Slide 5 is result/progress evidence.

## Recommended 5-Minute Timing

| Slide | Time |
|---|---:|
| Slide 1 | 15-20 s |
| Slide 2 | 45-50 s |
| Slide 3 | 55-60 s |
| Slide 4 | 65-70 s |
| Slide 5 | 60 s |
| Slide 6 | 45-50 s |

If the presentation becomes too long, reduce Slide 4 details and keep only the main controller table.

## Final Checklist

- Mention both Level 2 and Level 3 explicitly.
- Do not present MPCC as a final Level 3 candidate.
- Explain that MPCC was dropped from Level 3 because of robustness and replanning sensitivity.
- Use figures instead of long paragraphs.
- Mark current speed/success numbers as preliminary unless you have final benchmark data.
- Make sure all members have a clear speaking part.
- Include explicit future tasks and responsibilities on the final slide.
