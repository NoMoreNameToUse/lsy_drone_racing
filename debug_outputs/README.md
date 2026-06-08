# Debug path tools

This folder is used to save and visualize planned paths.

## Generate planner paths

Generate paths for all planners on the same map:

```bash
SCIPY_ARRAY_API=1 pixi run python debug_outputs/generate_compare_paths.py --config level2.toml
```

Use a fixed seed for reproducible comparison:

```bash
SCIPY_ARRAY_API=1 pixi run python debug_outputs/generate_compare_paths.py --config level2.toml --seed 2
```

Output files are saved here, for example:

```text
debug_outputs/astar_debug.npz
debug_outputs/theta_star_debug.npz
debug_outputs/rrt_star_debug.npz
debug_outputs/d_star_lite_debug.npz
debug_outputs/curve_gate_debug.npz
```

## Visualize planner comparison

Show the saved planner paths:

```bash
pixi run python debug_outputs/compare_debug_paths.py
```

This script only reads existing `.npz` files. Run `generate_compare_paths.py` first if you want fresh data.

## Run normal simulation

Run the controller in the simulator:

```bash
pixi run python scripts/sim.py --config level2.toml
```

The simulation uses the controller selected in `config/level2.toml`.

## Change path planner

Open:

```text
lsy_drone_racing/control/controllers/controller_rl.py
```

Find:

```python
planner_name = "rrt_star"
```

Change it to one of:

```python
planner_name = "astar"
planner_name = "theta_star"
planner_name = "rrt_star"
planner_name = "d_star_lite"
planner_name = "curve_gate"
```

Planner parameters are configured in:

```text
lsy_drone_racing/control/controllers/modules/path_generator.py
```

Look for:

```python
def make_path_generator(name: str):
```

Then edit the matching block, for example `if name == "astar":` or `if name == "theta_star":`.

## Typical workflow

1. Choose `planner_name` in `controller_rl.py`.
2. Run simulation:

```bash
pixi run python scripts/sim.py --config level2.toml
```

3. Generate comparison paths:

```bash
SCIPY_ARRAY_API=1 pixi run python debug_outputs/generate_compare_paths.py --config level2.toml
```

4. Visualize comparison:

```bash
pixi run python debug_outputs/compare_debug_paths.py
```
