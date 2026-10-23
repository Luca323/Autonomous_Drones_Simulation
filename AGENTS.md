# AGENTS.md — Contributor & AI Agent Guide

This document defines architecture, execution patterns, commands, and rules for human contributors and AI agents working on `Simulation_Stuff`.

> [!IMPORTANT]
> **Agent Commandments:** Before making changes, read and adhere to [AGENT_RULES.md](AGENT_RULES.md) (no fake fallbacks, no magic, no silent failures, stay in scope, match house style, protect uncommitted work).

---

## 1. System Overview

A decentralized 2D multi-agent swarm simulator with a C++17 ctypes kernel, double-integrator physics, Voronoi search, distributed Kalman filtering (DKF), and a FastAPI/WebSocket real-time Lattice-style visualizer.

- **Stack:** Python 3.10+, NumPy, FastAPI, Uvicorn, C++17 (`g++`/MinGW), HTML5 Canvas HUD.
- **Architecture:** Zero-centralized coordinator. Each drone computes its own sensing, gossip packets, DKF covariance intersection, CBBA task bids, and CBF safety filter independently.

---

## 2. Essential Commands

| Action | Command |
|---|---|
| **Run Visualizer** | `python -m swarm_sim.server` (Open http://127.0.0.1:8000) |
| **Fast Smoke Check (<0.3s)** | `python -c "from swarm_sim.world import SwarmWorld2D; from swarm_sim.config import SimulationConfig; w = SwarmWorld2D(SimulationConfig(num_drones=4, num_targets=1)); w.step(); print('OK')"` |
| **Run Test Suite** | `python -m pytest tests -q` |
| **Run Monte Carlo Benchmark** | `python -m swarm_sim.benchmark` |
| **Recompile C++ Kernel** | `g++ -O3 -ffast-math -shared -static-libgcc -static-libstdc++ -o swarm_sim/native/fast_core_v2.dll swarm_sim/native/fast_core.cpp` |

---

## 3. Architecture & Data Contracts

### Coordinate System & Geometry
- **Origin `(0, 0)`:** Top-left corner (Canvas convention). Horizontal axis $X \in [0, 1000]$, Vertical axis $Y \in [0, 700]$.
- **Heading / Angles:** 0 radians points along $+X$ (Right/East). Positive rotations follow standard cartesian math, with screen $Y$ pointing down.
- **Grid Mapping:** Grid cell $(r, c)$ corresponds to:
  $$\text{row} = \lfloor y / \text{cell\_size} \rfloor, \quad \text{col} = \lfloor x / \text{cell\_size} \rfloor$$

### Primary Array Shapes & Dtypes
- `positions`: `np.ndarray` shape `(N, 2)`, `float64`
- `velocities`: `np.ndarray` shape `(N, 2)`, `float64`
- `active_mask`: `np.ndarray` shape `(N,)`, `bool`
- `adj_matrix`: `np.ndarray` shape `(N, N)`, `bool`
- `certainty_grid`: `np.ndarray` shape `(grid_rows, grid_cols)`, `float32` (values in $[0.0, 1.0]$)

### Hardware Asset Types vs. Operational Roles
- **Hardware Asset Types** (Fixed airframe specification determined at spawn in `get_asset_spec`):
  - `SCOUT`: High sensor range ($1.55\times$), lower speed ($0.95\times$), comms ($1.15\times$).
  - `INTERCEPTOR`: High speed ($1.35\times$), high acceleration ($1.35\times$), compact sensor ($0.88\times$), comms ($1.00\times$).
  - `RELAY`: Extended radio range ($1.65\times$), standard speed ($1.00\times$), standard sensor ($1.05\times$).
- **Operational Roles** (Dynamic state updated frame-to-frame in `compute_control`):
  - `SEARCHING`: Voronoi / frontier exploration across unmapped grid cells.
  - `PURSUIT`: Proportional Navigation sprint toward allocated target track.
  - `ENCIRCLING`: Standoff orbit around localized target.
  - `STANDOFF_TRACK`: Standoff shadowing without kinetic terminal closure (under `standoff_track` doctrine).
  - `COURIER_EXFIL`: Autonomous dash out of an EW Jammer boundary to restore mesh connectivity and exfiltrate tracks.

---

## 4. Simulation Step Pipeline (`SwarmWorld2D.step`)

Each call to `step()` executes the following sequenced phases:
1. **Update Mesh Topology:** `self.network.update_topology(...)` builds `adj_matrix` based on pairwise distance, LOS occlusion, and active EW Jammer zones (`jammed_mask`).
2. **Gather Gossip Packets:** Active drones build `GossipPacket` instances containing local tracks and CBBA bids.
3. **1-Hop Mesh Exchange:** Packets delivered across wireless mesh respecting packet loss, EW jamming, and EMCON radio-silence rules.
4. **Grid Merge:** `cpp_update_and_merge_grids` applies sensor footprint decay and 1-hop max consensus grid merging (or NumPy fallback).
5. **Perception & 3-Layer Control:** For each active drone:
   - *Layer 1 (Perception):* DKF measurement update, covariance intersection, negative-information ghost clearing.
   - *Layer 2 (Autonomy / AI):* CBBA auction bids, courier exfil checks, role selection, Pro-Nav pursuit or Voronoi search.
   - *Layer 3 (Safety Shield):* Control Barrier Function (CBF) filtering against obstacles, peers, and No-Fly Zones.
6. **Physics Integration:** `step_double_integrator` integrates acceleration into position & velocity with drag and bounding walls.
7. **Target Kinematics & Capture:** Evasive target motion updated, capture criteria evaluated (unless under standoff doctrine).

---

## 5. C++ Native Kernel & Parity Rule

- `swarm_sim/native/fast_core.cpp` compiles to `fast_core_v2.dll` (auto-recompiled by `native/__init__.py` if `.cpp` is newer).
- Exported functions:
  - `cpp_check_los`: Ray-obstacle intersection.
  - `cpp_step_physics`: Vectorized kinematics integration.
  - `cpp_compute_frontier_search`: Voronoi exploration gradient.
  - `cpp_update_and_merge_grids`: Certainty grid fusion.
- **Dual-Execution Parity Rule:** Every C++ routine has a pure Python/NumPy fallback in `physics.py` / `communication.py` / `world.py`. If modifying C++, you **must** update the matching Python fallback to maintain numerical equivalence.
- If editing C++ function signatures, update ctypes `argtypes` and `restype` in `swarm_sim/native/__init__.py`.

---

## 6. WebSocket Protocol

- **Client to Server (`cmd`):**
  `pause`, `step_once`, `reset`, `select_drone` (`drone_id`), `set_param` (`key`, `value`), `spawn_target` (`x`, `y`), `spawn_drone` (`x`, `y`), `add_obstacle` (`x`, `y`, `radius`), `add_jammer` (`x`, `y`, `radius`), `add_nofly_zone` (`x`, `y`, `radius`), `clear_obstacles`, `emp_blast` (`x`, `y`, `radius`), `train_marl`.
- **Server to Client (Broadcast Frame):**
  `sim_time`, `step_count`, `drones`, `targets`, `obstacles`, `jammers`, `nofly_zones`, `network` (`active_links`, `jammed`), `metrics` (`active_drones`, `captured_targets`, `mean_est_error`, `collisions`, `avg_hops`), `inspector` (`drone_id`, `certainty_grid_downsampled`, `tracks`, `bids`).

---

## 7. File Map & Modification Lookup

| File | Primary Responsibility |
|---|---|
| `swarm_sim/config.py` | Master `SimulationConfig` dataclass for all physics, comms, sensor, and mission parameters. |
| `swarm_sim/world.py` | `SwarmWorld2D` top-level orchestrator, step pipeline, and telemetry snapshot generation. |
| `swarm_sim/physics.py` | Kinematics, drag, barrier functions, geometric collision checks, Python ray-casting. |
| `swarm_sim/communication.py` | Mesh topology, EW Jammer masking, 1-hop packet exchange, EMCON logic. |
| `swarm_sim/targets.py` | `TargetManager`, evasive target maneuvers, capture detection. |
| `swarm_sim/agents/controllers.py` | `DecentralizedDroneAgent`, asset specifications, Pro-Nav guidance, CBF safety filter. |
| `swarm_sim/agents/estimation.py` | `LocalWorldEstimator`, Distributed Kalman Filter, Covariance Intersection, ghost clearing. |
| `swarm_sim/agents/task_allocation.py` | `DecentralizedTaskAllocator`, consensus-based auction bidding (CBBA). |
| `swarm_sim/native/fast_core.cpp` | C++17 kernel for high-throughput ray checks, grid merging, and physics. |
| `swarm_sim/native/__init__.py` | Ctypes loader, dynamic compilation trigger, array marshalling. |
| `swarm_sim/server.py` | FastAPI application, WebSocket streaming, browser command router. |
| `swarm_sim/web/app.js` & `index.html` | Canvas HUD, telemetry dashboard, interactive controls. |
| `swarm_sim/rl/marl_env.py` & `policy.py` | PettingZoo parallel environment & attention policy for MARL mode. |
| `swarm_sim/benchmark.py` | Monte Carlo mission benchmark across contested EW, packet loss, and EMP attrition. |
