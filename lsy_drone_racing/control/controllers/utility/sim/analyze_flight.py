"""Visualize and analyze a single flight log, highlighting risk/quality events.

Consumes a ``debug_outputs/flight_<tag>.npz`` produced by the RL+A* controller
(one file per seed; see ``controller_rl_astar.py``) and produces:

1. An interactive 3D view ``flight_<tag>_3d.html`` -- open in a browser and drag
   to orbit/zoom the full 3D track: executed path coloured by speed, the planned
   path, gates (aperture + frame, correctly oriented), obstacle poles, and every
   detected event as a hoverable marker.
2. A static figure ``flight_<tag>_analysis.png`` with
   * a top-down (XY) map: executed path coloured by speed, planned path, gates
     (aperture segments) and obstacles, with every detected event marked;
   * timelines of executed vs. planned speed, obstacle/gate clearance, position
     tracking error, and the velocity-reversal metric -- thresholds drawn in.
3. A textual event report ``flight_<tag>_events.txt`` (also printed) listing,
   with timestamps and positions, the four event classes requested:
   * NEAR_OBSTACLE / NEAR_GATE  -- clearance below a safe threshold,
   * COLLISION / CONTACT        -- clearance ~0 or a terminated-by-crash ending,
   * REVERSAL                   -- rapid velocity-direction flips (executed AND
                                   planned -- the latter surfaces the timing
                                   module budgeting an impossible reversal),
   * DEVIATION                  -- large position/speed error vs. the plan.

Run (inside ``pixi shell``):

    python lsy_drone_racing/control/controllers/utility/sim/analyze_flight.py --seed 4
    python .../analyze_flight.py --file debug_outputs/flight_seed4.npz
    python .../analyze_flight.py            # picks the most recent flight_*.npz
"""

from __future__ import annotations

import os

# Harmless for a pure numpy/scipy/matplotlib script; kept for consistency with
# the other utilities that import crazyflow.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

import glob  # noqa: E402
from pathlib import Path  # noqa: E402

import fire  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from scipy.spatial.transform import Rotation as R  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
DEBUG_DIR = REPO_ROOT / "debug_outputs"

# Geometry (see config TOML).
GATE_APERTURE_HALF = 0.2  # 0.4 m opening
GATE_OUTER_HALF = 0.36  # 0.72 m frame
POLE_RADIUS = 0.015  # obstacle cylinders are 0.03 m diameter
OBSTACLE_TOP = 1.55  # obstacle poles run from the ground to ~1.55 m

# Per-event 3D marker style: (plotly symbol, colour).
_EVENT_3D = {
    "NEAR_OBSTACLE": ("x", "red"),
    "CONTACT_OBSTACLE": ("diamond", "red"),
    "NEAR_GATE": ("x", "purple"),
    "CONTACT_GATE": ("diamond", "purple"),
    "REVERSAL_EXECUTED": ("cross", "magenta"),
    "REVERSAL_PLANNED": ("cross", "orange"),
    "DEVIATION_POS": ("circle", "darkorange"),
    "DEVIATION_SPEED": ("square", "gold"),
    "COLLISION_END": ("diamond", "black"),
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _resolve_path(seed: int | None, file: str | None) -> Path:
    if file is not None:
        p = Path(file)
        return p if p.is_absolute() else REPO_ROOT / p
    if seed is not None:
        return DEBUG_DIR / f"flight_seed{seed}.npz"
    candidates = sorted(glob.glob(str(DEBUG_DIR / "flight_*.npz")), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"No flight_*.npz found in {DEBUG_DIR}")
    return Path(candidates[-1])


def load_flight(path: Path) -> dict:
    """Load a flight npz into a plain dict, normalizing scalar fields."""
    d = np.load(path, allow_pickle=True)
    out = {k: d[k] for k in d.files}
    out["seed"] = int(out["seed"])
    out["freq"] = float(out["freq"])
    out["reason"] = str(out["reason"])
    out["_path"] = path
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _square_loop_distance(lat: np.ndarray, vert: np.ndarray, half: float) -> np.ndarray:
    """In-plane distance from (lat, vert) to a square loop of the given half-size.

    Zero on the loop, positive both inside and outside (it is a |distance|).
    """
    m = np.maximum(np.abs(lat), np.abs(vert))
    inside = m <= half
    # Inside: distance to the nearest edge is half - max(|lat|, |vert|).
    d_inside = half - m
    # Outside: Euclidean distance to the square's boundary.
    dx = np.maximum(np.abs(lat) - half, 0.0)
    dy = np.maximum(np.abs(vert) - half, 0.0)
    d_outside = np.hypot(dx, dy)
    return np.where(inside, d_inside, d_outside)


def gate_rim_clearance(pos: np.ndarray, gpos: np.ndarray, gquat: np.ndarray) -> np.ndarray:
    """Per-tick 3D distance from the drone to the nearest point of the gate rim.

    The rim is the inner aperture edge (half-size ``GATE_APERTURE_HALF``) in the
    gate plane. Flying cleanly through the centre keeps this near the aperture
    half-width; grazing the frame drives it toward zero.

    Args:
        pos: (T, 3) drone positions.
        gpos: (T, 3) gate centre positions (per tick).
        gquat: (T, 4) gate quaternions (per tick, scipy [x, y, z, w]).
    """
    rot = R.from_quat(gquat)
    rel = pos - gpos
    fwd = rot.apply([1.0, 0.0, 0.0])
    lat = rot.apply([0.0, 1.0, 0.0])
    up = rot.apply([0.0, 0.0, 1.0])
    along = np.einsum("ij,ij->i", rel, fwd)
    o_lat = np.einsum("ij,ij->i", rel, lat)
    o_up = np.einsum("ij,ij->i", rel, up)
    in_plane = _square_loop_distance(o_lat, o_up, GATE_APERTURE_HALF)
    return np.hypot(along, in_plane)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(flight: dict) -> dict:
    """Derive per-tick analysis signals from the raw flight log."""
    pos = flight["executed_pos"]
    vel = flight["executed_vel"]
    target = flight["target_pos"]
    freq = flight["freq"]
    t = flight["log_t"]
    T = len(pos)

    sensed_obst = flight["sensed_obstacles_pos"]  # (T, O, 3)
    sensed_gpos = flight["sensed_gates_pos"]  # (T, G, 3)
    sensed_gquat = flight["sensed_gates_quat"]  # (T, G, 4)

    # Obstacle clearance: XY distance from drone to nearest pole centre per tick.
    d_obst = np.linalg.norm(pos[:, None, :2] - sensed_obst[:, :, :2], axis=2)  # (T, O)
    obst_clear = d_obst.min(axis=1)
    obst_idx = d_obst.argmin(axis=1)

    # Gate rim clearance: min over gates of the 3D rim distance per tick.
    n_gates = sensed_gpos.shape[1]
    gate_d = np.full((T, n_gates), np.inf)
    for g in range(n_gates):
        gate_d[:, g] = gate_rim_clearance(pos, sensed_gpos[:, g, :], sensed_gquat[:, g, :])
    gate_clear = gate_d.min(axis=1)
    gate_idx = gate_d.argmin(axis=1)

    # Tracking error and speeds.
    pos_err = np.linalg.norm(pos - target, axis=1)
    speed = np.linalg.norm(vel, axis=1)

    # Planned velocity from the setpoint stream. The setpoint index (log_tick)
    # resets to 0 on every replan, so the target_pos jump across a replan is not
    # a real planned motion -- invalidate those boundaries so they don't masquerade
    # as huge speed errors or reversals.
    log_tick = flight["log_tick"].astype(int)
    planned_vel = np.zeros_like(vel)
    if T >= 2:
        planned_vel[:-1] = (target[1:] - target[:-1]) * freq
        planned_vel[-1] = planned_vel[-2]
        boundary = np.zeros(T, dtype=bool)
        boundary[:-1] = log_tick[1:] <= log_tick[:-1]  # tick reset/stall == replan
        planned_vel[boundary] = np.nan
    planned_speed = np.linalg.norm(planned_vel, axis=1)
    speed_err = np.abs(speed - planned_speed)

    # Velocity-reversal metric: angle between v(t) and v(t+W) over a short window.
    w = max(1, int(round(0.15 * freq)))
    turn_exec = _reversal_angle(vel, w)
    turn_plan = _reversal_angle(planned_vel, w)

    return {
        "t": t,
        "T": T,
        "pos": pos,
        "speed": speed,
        "planned_speed": planned_speed,
        "speed_err": speed_err,
        "pos_err": pos_err,
        "obst_clear": obst_clear,
        "obst_idx": obst_idx,
        "gate_clear": gate_clear,
        "gate_idx": gate_idx,
        "turn_exec": turn_exec,
        "turn_plan": turn_plan,
        "reversal_window": w,
    }


def _reversal_angle(vel: np.ndarray, w: int) -> np.ndarray:
    """Per-tick angle (deg) between velocity now and ``w`` ticks later."""
    T = len(vel)
    speed = np.linalg.norm(vel, axis=1)
    unit = vel / np.maximum(speed[:, None], 1e-9)
    ang = np.zeros(T)
    if T > w:
        dots = np.einsum("ij,ij->i", unit[:-w], unit[w:])
        ang[:-w] = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return ang


# ---------------------------------------------------------------------------
# Event detection
# ---------------------------------------------------------------------------
def _intervals(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs of a boolean mask as inclusive (i0, i1) index pairs."""
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def detect_events(flight: dict, m: dict, p: dict) -> list[dict]:
    """Detect and summarize the four highlighted event classes."""
    t, pos = m["t"], m["pos"]
    events: list[dict] = []

    def _add(kind: str, i0: int, i1: int, metric: np.ndarray, extreme: str, extra: str = ""):
        seg = metric[i0 : i1 + 1]
        k = int(np.argmin(seg) if extreme == "min" else np.argmax(seg))
        idx = i0 + k
        events.append({
            "kind": kind,
            "i0": i0,
            "i1": i1,
            "t0": float(t[i0]),
            "t1": float(t[i1]),
            "value": float(metric[idx]),
            "idx": idx,
            "pos": pos[idx].tolist(),
            "extra": extra,
        })

    # NEAR_OBSTACLE / CONTACT.
    for i0, i1 in _intervals(m["obst_clear"] < p["near_obstacle"]):
        seg = m["obst_clear"][i0 : i1 + 1]
        oi = int(m["obst_idx"][i0 + int(np.argmin(seg))])
        kind = "CONTACT_OBSTACLE" if seg.min() < p["contact"] else "NEAR_OBSTACLE"
        _add(kind, i0, i1, m["obst_clear"], "min", extra=f"obstacle {oi}")

    # NEAR_GATE / CONTACT.
    for i0, i1 in _intervals(m["gate_clear"] < p["near_gate"]):
        seg = m["gate_clear"][i0 : i1 + 1]
        gi = int(m["gate_idx"][i0 + int(np.argmin(seg))])
        kind = "CONTACT_GATE" if seg.min() < p["contact_gate"] else "NEAR_GATE"
        _add(kind, i0, i1, m["gate_clear"], "min", extra=f"gate {gi}")

    # REVERSAL -- executed and planned, where fast enough to matter.
    fast = m["speed"] >= p["reversal_min_speed"]
    for i0, i1 in _intervals((m["turn_exec"] > p["reversal_angle"]) & fast):
        _add("REVERSAL_EXECUTED", i0, i1, m["turn_exec"], "max")
    plan_fast = m["planned_speed"] >= p["reversal_min_speed"]
    plan_rev = (m["turn_plan"] > p["reversal_angle"]) & plan_fast & np.isfinite(m["turn_plan"])
    for i0, i1 in _intervals(plan_rev):
        _add("REVERSAL_PLANNED", i0, i1, m["turn_plan"], "max",
             extra="planned trajectory reverses (timing/plan)")

    # DEVIATION -- position and speed (speed_err is NaN across replan boundaries).
    for i0, i1 in _intervals(m["pos_err"] > p["pos_dev"]):
        _add("DEVIATION_POS", i0, i1, m["pos_err"], "max")
    speed_dev = (m["speed_err"] > p["speed_dev"]) & np.isfinite(m["speed_err"])
    for i0, i1 in _intervals(speed_dev):
        _add("DEVIATION_SPEED", i0, i1, m["speed_err"], "max")

    # COLLISION -- terminated-by-crash ending (not a clean finish/timeout).
    if flight["reason"] == "terminated":
        last = pos[-1]
        # Nearest object to the crash point for labelling.
        label, dist = _nearest_object(last, flight)
        events.append({
            "kind": "COLLISION_END",
            "i0": m["T"] - 1,
            "i1": m["T"] - 1,
            "t0": float(t[-1]),
            "t1": float(t[-1]),
            "value": dist,
            "idx": m["T"] - 1,
            "pos": last.tolist(),
            "extra": f"run ended terminated; nearest {label} (d={dist:.2f} m)",
        })

    events.sort(key=lambda e: e["t0"])
    return events


def _nearest_object(pos: np.ndarray, flight: dict) -> tuple[str, float]:
    """Nearest gate (3D) / obstacle (XY) to ``pos`` using last sensed poses."""
    gpos = flight["sensed_gates_pos"][-1]
    opos = flight["sensed_obstacles_pos"][-1]
    best, bd = "none", np.inf
    for i, g in enumerate(gpos):
        d = float(np.linalg.norm(pos - g))
        if d < bd:
            best, bd = f"gate {i}", d
    for i, o in enumerate(opos):
        d = float(np.linalg.norm(pos[:2] - o[:2]))
        if d < bd:
            best, bd = f"obstacle {i}", d
    return best, bd


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def build_report(flight: dict, m: dict, events: list[dict], p: dict) -> str:
    """Assemble the textual event report."""
    lines = []
    lines.append("=" * 78)
    lines.append(f"FLIGHT ANALYSIS  |  {flight['_path'].name}")
    lines.append(f"seed={flight['seed']}  reason={flight['reason']}  "
                 f"ticks={m['T']}  duration={m['t'][-1]:.2f}s  freq={flight['freq']:.0f}Hz")
    lines.append("=" * 78)
    lines.append("Summary metrics:")
    lines.append(f"  min obstacle clearance : {m['obst_clear'].min():.3f} m")
    lines.append(f"  min gate-rim clearance : {m['gate_clear'].min():.3f} m")
    lines.append(f"  max position error     : {m['pos_err'].max():.3f} m   "
                 f"(mean {m['pos_err'].mean():.3f})")
    lines.append(f"  max speed              : {m['speed'].max():.3f} m/s")
    lines.append(f"  max speed error        : {np.nanmax(m['speed_err']):.3f} m/s")
    lines.append(f"  max reversal angle     : exec {m['turn_exec'].max():.0f} deg, "
                 f"planned {np.nanmax(m['turn_plan']):.0f} deg "
                 f"(window {m['reversal_window']} ticks)")
    lines.append("")
    counts: dict[str, int] = {}
    for e in events:
        counts[e["kind"]] = counts.get(e["kind"], 0) + 1
    lines.append(f"Events ({len(events)} total): {counts if counts else 'none'}")
    lines.append("-" * 78)
    if not events:
        lines.append("No threshold events -- clean flight.")
    for e in events:
        span = f"t={e['t0']:.2f}s" if e["i0"] == e["i1"] else f"t={e['t0']:.2f}-{e['t1']:.2f}s"
        pos = e["pos"]
        extra = f"  [{e['extra']}]" if e["extra"] else ""
        lines.append(
            f"  {e['kind']:18s} {span:18s} value={e['value']:7.3f}  "
            f"pos=({pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}){extra}"
        )
    lines.append("=" * 78)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _draw_gate(ax: plt.Axes, gpos: np.ndarray, gquat: np.ndarray) -> None:
    """Draw a gate aperture as a short segment (top-down) at its centre."""
    lat = R.from_quat(gquat).apply([0.0, 1.0, 0.0])[:2]
    n = np.linalg.norm(lat)
    lat = lat / n if n > 1e-9 else np.array([0.0, 1.0])
    a = gpos[:2] - GATE_APERTURE_HALF * lat
    b = gpos[:2] + GATE_APERTURE_HALF * lat
    ax.plot([a[0], b[0]], [a[1], b[1]], color="tab:blue", lw=3, solid_capstyle="round")


_EVENT_STYLE = {
    "NEAR_OBSTACLE": ("red", "x", 60),
    "CONTACT_OBSTACLE": ("red", "*", 200),
    "NEAR_GATE": ("purple", "x", 60),
    "CONTACT_GATE": ("purple", "*", 200),
    "REVERSAL_EXECUTED": ("magenta", "v", 70),
    "REVERSAL_PLANNED": ("orange", "^", 70),
    "DEVIATION_POS": ("darkorange", "o", 40),
    "DEVIATION_SPEED": ("gold", "s", 40),
    "COLLISION_END": ("black", "X", 220),
}


def plot_overview(flight: dict, m: dict, events: list[dict], p: dict, out_png: Path) -> None:
    """Render the top-down map + timelines figure and save it."""
    pos = m["pos"]
    t = m["t"]
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.25, 1.0])

    # --- Top-down map (left, full height) ---
    axm = fig.add_subplot(gs[:, 0])
    seg = np.stack([pos[:-1, :2], pos[1:, :2]], axis=1)
    lc = LineCollection(seg, cmap="viridis", array=m["speed"][:-1], lw=2.5)
    axm.add_collection(lc)
    cb = fig.colorbar(lc, ax=axm, fraction=0.046, pad=0.02)
    cb.set_label("executed speed (m/s)")

    ipath = flight["initial_traj_pos"]
    axm.plot(ipath[:, 0], ipath[:, 1], "--", color="gray", lw=1.2, label="initial plan")

    gpos = flight["sensed_gates_pos"][-1]
    gquat = flight["sensed_gates_quat"][-1]
    for g in range(gpos.shape[0]):
        _draw_gate(axm, gpos[g], gquat[g])
    opos = flight["sensed_obstacles_pos"][-1]
    for o in opos:
        axm.add_patch(plt.Circle(o[:2], 0.06, color="saddlebrown", alpha=0.7))

    seen = set()
    for e in events:
        color, marker, size = _EVENT_STYLE.get(e["kind"], ("k", "o", 40))
        lbl = e["kind"] if e["kind"] not in seen else None
        seen.add(e["kind"])
        axm.scatter(e["pos"][0], e["pos"][1], c=color, marker=marker, s=size,
                    label=lbl, zorder=5)
    axm.scatter(pos[0, 0], pos[0, 1], c="lime", marker="o", s=80, label="start", zorder=6)
    axm.set_aspect("equal")
    axm.set_xlabel("x (m)")
    axm.set_ylabel("y (m)")
    axm.set_title(f"flight_seed{flight['seed']}  ({flight['reason']})  top-down")
    axm.legend(loc="upper right", fontsize=7, ncol=2)
    axm.grid(alpha=0.3)

    # --- Timelines (right column) ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(t, m["speed"], label="executed", color="tab:green")
    ax1.plot(t, m["planned_speed"], label="planned", color="gray", lw=1)
    ax1.set_ylabel("speed\n(m/s)")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1)
    ax2.plot(t, m["obst_clear"], label="obstacle", color="tab:red")
    ax2.plot(t, m["gate_clear"], label="gate rim", color="purple")
    ax2.axhline(p["near_obstacle"], ls=":", color="tab:red", lw=1)
    ax2.axhline(p["near_gate"], ls=":", color="purple", lw=1)
    ax2.set_ylabel("clearance\n(m)")
    ax2.legend(fontsize=7, loc="upper right")
    ax2.grid(alpha=0.3)

    ax3 = fig.add_subplot(gs[2, 1], sharex=ax1)
    ax3.plot(t, m["pos_err"], color="darkorange", label="pos error")
    ax3.axhline(p["pos_dev"], ls=":", color="darkorange", lw=1)
    ax3.set_ylabel("track err\n(m)")
    ax3.legend(fontsize=7, loc="upper right")
    ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[3, 1], sharex=ax1)
    ax4.plot(t, m["turn_exec"], color="magenta", label="exec reversal")
    ax4.plot(t, m["turn_plan"], color="orange", lw=1, label="plan reversal")
    ax4.axhline(p["reversal_angle"], ls=":", color="magenta", lw=1)
    ax4.set_ylabel("reversal\n(deg)")
    ax4.set_xlabel("time (s)")
    ax4.legend(fontsize=7, loc="upper right")
    ax4.grid(alpha=0.3)

    fig.suptitle(f"Flight analysis: seed {flight['seed']}  |  {len(events)} events", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _gate_loop(gpos: np.ndarray, gquat: np.ndarray, half: float) -> np.ndarray:
    """Closed 3D square loop of the given half-size in the gate plane."""
    local = np.array([
        [0.0, half, half], [0.0, -half, half],
        [0.0, -half, -half], [0.0, half, -half], [0.0, half, half],
    ])
    return R.from_quat(gquat).apply(local) + gpos


def plot_3d(flight: dict, m: dict, events: list[dict], out_html: Path) -> None:
    """Write a self-contained, orbitable 3D HTML view of the flight."""
    pos = m["pos"]
    traces: list[go.Scatter3d] = []

    # Executed trajectory, coloured by speed (hover shows time + speed).
    hover = [f"t={tt:.2f}s<br>speed={ss:.2f} m/s" for tt, ss in zip(m["t"], m["speed"])]
    traces.append(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="lines", name="executed",
        line=dict(color=m["speed"], colorscale="Viridis", width=6,
                  colorbar=dict(title="speed<br>(m/s)", x=1.02)),
        text=hover, hoverinfo="text",
    ))

    # Initial planned trajectory (full-track reference).
    ip = flight["initial_traj_pos"]
    traces.append(go.Scatter3d(
        x=ip[:, 0], y=ip[:, 1], z=ip[:, 2], mode="lines", name="initial plan",
        line=dict(color="gray", width=3), hoverinfo="skip",
    ))

    # Gates: aperture (solid) + outer frame (faint), oriented by quaternion.
    gpos = flight["sensed_gates_pos"][-1]
    gquat = flight["sensed_gates_quat"][-1]
    for g in range(gpos.shape[0]):
        for half, color, width, name in (
            (GATE_APERTURE_HALF, "royalblue", 6, "gate aperture"),
            (GATE_OUTER_HALF, "lightskyblue", 2, "gate frame"),
        ):
            loop = _gate_loop(gpos[g], gquat[g], half)
            traces.append(go.Scatter3d(
                x=loop[:, 0], y=loop[:, 1], z=loop[:, 2], mode="lines",
                line=dict(color=color, width=width), name=name,
                legendgroup=name, showlegend=(g == 0), hoverinfo="skip",
            ))

    # Obstacles: vertical poles from the ground up to OBSTACLE_TOP.
    opos = flight["sensed_obstacles_pos"][-1]
    for k, o in enumerate(opos):
        traces.append(go.Scatter3d(
            x=[o[0], o[0]], y=[o[1], o[1]], z=[0.0, OBSTACLE_TOP], mode="lines",
            line=dict(color="saddlebrown", width=8), name="obstacle",
            legendgroup="obstacle", showlegend=(k == 0),
            text=f"obstacle {k}", hoverinfo="text",
        ))

    # Start marker.
    traces.append(go.Scatter3d(
        x=[pos[0, 0]], y=[pos[0, 1]], z=[pos[0, 2]], mode="markers",
        marker=dict(size=6, color="limegreen"), name="start", hoverinfo="skip",
    ))

    # Event markers, grouped by kind.
    by_kind: dict[str, list[dict]] = {}
    for e in events:
        by_kind.setdefault(e["kind"], []).append(e)
    for kind, evs in by_kind.items():
        symbol, color = _EVENT_3D.get(kind, ("circle", "black"))
        pts = np.array([e["pos"] for e in evs])
        txt = [f"{kind}<br>t={e['t0']:.2f}s<br>value={e['value']:.3f}"
               f"{('<br>' + e['extra']) if e['extra'] else ''}" for e in evs]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
            marker=dict(size=5, color=color, symbol=symbol), name=kind,
            text=txt, hoverinfo="text",
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=(f"Flight 3D: seed {flight['seed']} ({flight['reason']})  "
               f"{m['T']} ticks, {len(events)} events"),
        scene=dict(
            xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    # Embed plotly.js so the file opens offline (self-contained, ~3.5 MB).
    fig.write_html(out_html, include_plotlyjs=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def analyze(
    seed: int | None = None,
    file: str | None = None,
    near_obstacle: float = 0.25,
    near_gate: float = 0.12,
    contact: float = 0.10,
    contact_gate: float = 0.05,
    reversal_angle: float = 120.0,
    reversal_min_speed: float = 0.3,
    pos_dev: float = 0.30,
    speed_dev: float = 0.6,
) -> dict:
    """Analyze a flight log: detect highlight events, plot, and write a report.

    Args:
        seed: Seed whose ``flight_seed<seed>.npz`` to load.
        file: Explicit path to a flight npz (overrides ``seed``).
        near_obstacle: Obstacle centre-distance (m) below which a tick is "near".
        near_gate: Gate-rim distance (m) below which a tick is "near".
        contact: Obstacle clearance (m) treated as a contact/collision.
        contact_gate: Gate-rim distance (m) treated as a contact/collision.
        reversal_angle: Velocity-direction change (deg) over the window flagged
            as a rapid reversal.
        reversal_min_speed: Min speed (m/s) for a reversal to count.
        pos_dev: Position tracking error (m) above which a deviation is flagged.
        speed_dev: Speed error (m/s) above which a deviation is flagged.

    Returns:
        Dict with ``events`` and the output ``png`` / ``txt`` paths.
    """
    p = {
        "near_obstacle": near_obstacle, "near_gate": near_gate, "contact": contact,
        "contact_gate": contact_gate, "reversal_angle": reversal_angle,
        "reversal_min_speed": reversal_min_speed, "pos_dev": pos_dev, "speed_dev": speed_dev,
    }
    path = _resolve_path(seed, file)
    flight = load_flight(path)
    m = compute_metrics(flight)
    events = detect_events(flight, m, p)

    report = build_report(flight, m, events, p)
    print(report)

    stem = path.with_suffix("")
    png = stem.parent / f"{stem.name}_analysis.png"
    html = stem.parent / f"{stem.name}_3d.html"
    txt = stem.parent / f"{stem.name}_events.txt"
    plot_overview(flight, m, events, p, png)
    plot_3d(flight, m, events, html)
    txt.write_text(report + "\n")
    print(f"\nSaved 2D figure -> {png}")
    print(f"Saved 3D view   -> {html}  (open in a browser, drag to orbit)")
    print(f"Saved report    -> {txt}")
    return {"events": events, "png": str(png), "html": str(html), "txt": str(txt)}


if __name__ == "__main__":
    fire.Fire(analyze, serialize=lambda _: None)
