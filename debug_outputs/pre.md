# Progress Presentation Speaker Script

This script is written for the current `canva_drone.pptx`.

Recommended total timing:

- Main flow without staying long on linked debug screenshots: about **5 minutes**.
- Slides 4 and 5 are linked from Slide 3 through `CurveGate` and `A*`. Treat them as short detail views. If time is tight, spend only **10-15 seconds** on each and return to Slide 3.

## Slide 1: Title

**Target time: 15 seconds**

Good morning everyone.  
We are Yichen Zhang and Zihan Deng, and today we will present our progress on the autonomous drone racing project.  
Our focus is the control pipeline, the path planning debug tools, and the comparison between Level 2 and Level 3 performance.

## Slide 2: Modular Control Pipeline

**Target time: 40 seconds**

Our main methodology is a modular control pipeline.  
Instead of building one fixed controller, we separate the problem into several replaceable modules.

First, the observation contains the drone state, gate positions, and obstacle positions.  
Then the path module generates a path using planners such as A*, Theta*, RRT*, D* Lite, or CurveGate.  
After that, the timing module assigns timestamps to the waypoints. We mainly use distance-based timing, but the framework also supports uniform and motion-aware timing.  
The trajectory module converts the waypoints into a smooth reference trajectory, for example using PCHIP spline interpolation.  
Finally, the control module tracks this reference using MPC, an RL policy, or MPCC.

The key point is that this structure lets us compare different paths and different controllers under the same framework.  
It also supports replanning when the target gate changes or when the observed environment changes.

## Slide 3: 3D Debug Path Model

**Target time: 40 seconds**

This slide shows how we debug the path planning part before running the controller.  
The table compares several planners on the same Level 2 randomized seed.

A* and Theta* produce the shortest trajectories, around 8.7 meters.  
RRT* and D* Lite are slightly longer in this run, and CurveGate is the most conservative around gates, with a trajectory length of about 10.63 meters.

The purpose of this comparison is not only to find the shortest path.  
We also need to check whether the trajectory passes through gate openings, avoids gate frames, and stays away from inflated obstacle regions.  
That is why the 3D debug visualization is useful: it allows us to inspect the planned path before sending it to the controller.

If we click on `CurveGate` or `A*`, we can jump to the corresponding 3D debug screenshot.

## Slide 4: Linked Debug Screenshot - CurveGate

**Target time: 15 seconds**

Here we show the CurveGate path in the 3D debug model.  
The orange curve is more conservative: it keeps a larger detour around gates and obstacles, so the total trajectory length becomes longer.  
This is useful for safety inspection, but it may reduce speed compared with shorter planners.

After this detail view, return to the planner comparison slide.

## Slide 5: Linked Debug Screenshot - A* and Planner Comparison

**Target time: 15 seconds**

This screenshot shows the planner comparison in the same 3D environment.  
The legend gives the trajectory length for each planner.  
A* and Theta* are shorter and more direct, while CurveGate takes a wider route around critical regions.

This view helps us choose a path planner based on both path length and geometric safety, not only one metric.

After this detail view, return to the main flow.

## Slide 6: Controller Designs and Level Applicability

**Target time: 55 seconds**

Next, we compare the three controller designs.

The first one is the MPC-based controller.  
It uses A* path planning, distance timing, spline trajectory generation, and an acados MPC controller.  
Its advantage is interpretability: when it fails, it is easier to inspect whether the problem comes from the path, timing, trajectory, or tracking.

The second one is the RL controller.  
It uses a planned trajectory as input samples and then applies an RL policy to output attitude and thrust commands.  
It has fast inference and gives the best empirical performance in our current tests, especially in terms of success rate.

The third one is MPCC.  
MPCC is attractive because it directly optimizes progress along the path, using contour error, lag error, and progress reward.  
However, we only keep it for Level 2 comparison.  
For Level 3, we dropped MPCC because the full track is randomized and online replanning changes the reference more aggressively.  
MPCC depends strongly on a good arc-length reference and stable solver behavior, so it became less reliable under Level 3 conditions.

## Slide 7: Progress - Level 2 vs Level 3

**Target time: 45 seconds**

This slide summarizes our current progress in Level 2 and Level 3.

For completion time, RL is currently the fastest: below 7 seconds on Level 2 and below 9 seconds on Level 3.  
MPC is slower, but still useful as a stable and interpretable baseline.  
MPCC can finish Level 2 below 9 seconds in successful runs, but we do not use it for Level 3.

For success rate, RL is also the strongest candidate, with more than 90 percent in both levels.  
MPC reaches more than 70 percent, so it remains a reasonable fallback.  
MPCC is above 50 percent in Level 2, but not robust enough for Level 3.

The main conclusion is that RL is our best current candidate, while MPC is our reliable backup.  
MPCC is useful as a racing-oriented idea, but not yet robust enough for randomized online planning.

## Slide 8: Challenges

**Target time: 35 seconds**

The main remaining challenges are especially related to Level 3.

First, the RL controller has a weak explicit safety guarantee.  
It outputs actions directly, so safety mainly comes from the planned trajectory and obstacle margins.

Second, RL depends heavily on local trajectory samples.  
If replanning changes the trajectory abruptly, the input distribution of the policy can shift.

Third, the RL policy does not reason about obstacles directly.  
Obstacle avoidance is encoded through the planned path, so the path planner and clearance validation must be reliable.

Finally, replanning discontinuity is important.  
When a new path is generated, it should connect smoothly to the old reference, otherwise the controller may receive a sudden change.

## Slide 9: Next Steps

**Target time: 40 seconds**

Our next steps are organized into path planning, control, and evaluation.

For path planning, we will add curvature-aware smoothing and tune the adaptive obstacle margin.  
The goal is to reduce sharp turns and avoid collisions in randomized Level 3 seeds.

For control, we will fine-tune the RL behavior and add a safety fallback.  
If the trajectory becomes risky, the controller should switch to MPC or another safer behavior.

For evaluation, we will benchmark Level 2 and Level 3 over multiple seeds.  
We will compare both speed and success rate, because the final controller should be selected based on reliability, not only minimum time.

The final goal is a reproducible controller configuration that works robustly across randomized tracks.

## Slide 10: Thank You

**Target time: 10 seconds**

Thank you for listening.  
We are happy to answer questions about the controller design, the 3D debug visualization, or the Level 2 and Level 3 comparison.

## Timing Summary

| Slide | Topic | Target Time |
|---|---|---:|
| 1 | Title | 15 s |
| 2 | Modular Control Pipeline | 40 s |
| 3 | 3D Debug Path Model | 40 s |
| 4 | CurveGate linked screenshot | 15 s |
| 5 | A* / planner comparison linked screenshot | 15 s |
| 6 | Controller designs | 55 s |
| 7 | Progress Level 2 vs Level 3 | 45 s |
| 8 | Challenges | 35 s |
| 9 | Next steps | 40 s |
| 10 | Thank you | 10 s |

Total with linked screenshots: about **5 minutes 15 seconds**.  
If you need exactly 5 minutes, shorten Slides 4 and 5 to one sentence each, or skip one of them during the live presentation.
