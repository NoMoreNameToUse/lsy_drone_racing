# Short 3-Minute Speaker Script

This is a shorter version for a slower speaking speed.  
Target total time: about **3 minutes**.

Slides 4 and 5 are linked debug screenshots from Slide 3.  
If time is limited, show only one of them or skip them.

## Slide 1: Title

**Target time: 10 seconds**

Good morning everyone.  
We are Yichen Zhang and Zihan Deng.  
Today we present our progress on autonomous drone racing, focusing on our controller pipeline and the comparison between Level 2 and Level 3.

## Slide 2: Modular Control Pipeline

**Target time: 30 seconds**

Our method is based on a modular control pipeline.

The observation gives us the drone state, gates, and obstacles.  
Then the path module generates a path with planners such as A*, Theta*, RRT*, D* Lite, or CurveGate.  
The timing module assigns timestamps to the path.  
The trajectory module converts it into a smooth reference.  
Finally, the control module tracks this reference using MPC, RL, or MPCC.

The advantage is that we can replace each module independently and compare different controllers and planners in the same framework.

## Slide 3: 3D Debug Path Model

**Target time: 30 seconds**

Here we show the 3D debug path model.

The table compares several planners on the same Level 2 random seed.  
A* and Theta* are the shortest, around 8.7 meters.  
CurveGate is longer, about 10.63 meters, because it is more conservative around gates.

This debug model helps us check the gates, obstacle cylinders, safety margins, and trajectory length before running the controller.

The `CurveGate` and `A*` links open two detailed screenshots.

## Slide 4: CurveGate Debug Screenshot

**Target time: 8 seconds**

This is the CurveGate path.  
It chooses a safer and wider path around gates and obstacles, but the path length is longer.

## Slide 5: A* / Planner Debug Screenshot

**Target time: 8 seconds**

This screenshot shows the planner comparison.  
It helps us compare path length and geometry, not only the final controller result.

## Slide 6: Controller Designs

**Target time: 40 seconds**

We tested three controller designs.

First, MPC uses A* path planning, distance timing, spline trajectory, and acados MPC.  
It is stable and easy to debug.

Second, RL uses planned trajectory samples and a learned policy to output attitude and thrust.  
It is fast and gives the best current empirical performance.

Third, MPCC directly optimizes progress along the path.  
It is useful for racing in theory, but we only keep it for Level 2.  
For Level 3, we dropped MPCC because randomized tracks and online replanning make the reference path change more strongly, and MPCC became too sensitive to reference quality and solver feasibility.

## Slide 7: Progress Level 2 vs Level 3

**Target time: 35 seconds**

This slide summarizes our current results.

For speed, RL is the fastest: below 7 seconds on Level 2 and below 9 seconds on Level 3.  
MPC is slower, but still useful as a stable baseline.  
MPCC works in some Level 2 runs, but is not used in Level 3.

For success rate, RL is currently above 90 percent in both levels.  
MPC is above 70 percent.  
So our current best candidate is RL, with MPC as a safer backup.

## Slide 8: Challenges

**Target time: 25 seconds**

The main challenges are related to Level 3.

RL has no explicit safety guarantee, so it depends on the path planner and safety margins.  
It also depends heavily on local trajectory samples.  
If replanning changes the trajectory suddenly, the policy input can change a lot.

Also, RL does not reason about obstacles directly.  
Obstacle avoidance is mainly handled by the planned path.

## Slide 9: Next Steps

**Target time: 25 seconds**

Our next steps are path planning, control, and evaluation.

For path planning, we will add curvature-aware smoothing and tune obstacle margins.  
For control, we will fine-tune RL and add a safety fallback, for example switching to MPC when the trajectory becomes risky.  
For evaluation, we will benchmark Level 2 and Level 3 over multiple random seeds.

The final goal is a robust and reproducible controller configuration.

## Slide 10: Thank You

**Target time: 7 seconds**

Thank you for listening.  
We are happy to answer your questions.

## Timing Summary

| Slide | Topic | Target Time |
|---|---|---:|
| 1 | Title | 10 s |
| 2 | Pipeline | 30 s |
| 3 | 3D debug model | 30 s |
| 4 | CurveGate screenshot | 8 s |
| 5 | Planner screenshot | 8 s |
| 6 | Controller designs | 40 s |
| 7 | Progress | 35 s |
| 8 | Challenges | 25 s |
| 9 | Next steps | 25 s |
| 10 | Thank you | 7 s |

Total: about **3 minutes 18 seconds**.  
For exactly 3 minutes, skip Slide 4 or Slide 5 during the live presentation.
