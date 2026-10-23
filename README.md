# Decentralized Autonomous Drone Swarm Simulator

A decentralized 2D multi-agent swarm simulation engine featuring a C++17 native acceleration kernel, Distributed Kalman Filtering (DKF) with Covariance Intersection, Consensus-Based Bundle / Marginal Utility (CBBA) task allocation, True Proportional Navigation (Pro-Nav) guidance, Control Barrier Function (CBF) safety shielding, Ostrom–Pitt algorithmic self-governance, and a real-time FastAPI/WebSocket Lattice-style C2 visualizer.

---

## Quick Start

1. **Launch the Real-Time Lattice C2 Web Visualizer:**
   ```bash
   python -m swarm_sim.server
   ```
   Open **http://127.0.0.1:8000** in your browser.

2. **Run the Modular Test Suite:**
   ```bash
   python -m pytest tests -q
   ```

3. **Run the Monte Carlo Mission Benchmark:**
   ```bash
   python -m swarm_sim.benchmark
   ```

4. **Recompile the C++17 Native Kernel (auto-compiled on startup if modified):**
   ```bash
   g++ -O3 -ffast-math -shared -static-libgcc -static-libstdc++ -o swarm_sim/native/fast_core_v2.dll swarm_sim/native/fast_core.cpp
   ```

---

## Architecture & Core Subsystems

### 1. Strict Decentralization & Matrix-Free Mesh Networking (`swarm_sim/communication.py`)
- **Zero Central Coordinator:** Each drone operates strictly on its own kinematics, local sensor detections within $R_{\text{sense}}$, and 1-hop wireless mesh packets (`GossipPacket`) within $R_{\text{comm}}$ subject to obstacle Line-of-Sight (LOS) occlusion and stochastic packet loss.
- **Contested EW RF Jamming & EMCON:** Drones inside an active `JammerZone` suffer RF transmit/receive blackouts (`jammed_mask`). Under **EMCON (Radio Silence)** doctrine, drones suppress routine broadcasts unless reporting a fresh direct target contact (`hops <= 1`).
- **Matrix-Free Graph Connectivity:** Connected components ($c$) and the Fiedler algebraic connectivity eigenvalue ($\lambda_2(\mathbf{L})$) are computed in $O(N + |E|)$ time via a 1D Disjoint-Set (Union-Find) pass and edge-list shifted power iteration, avoiding $O(N^3)$ dense Laplacian eigendecomposition.

### 2. Heterogeneous Airframe Roles & Active SHORAD Threats (`swarm_sim/agents/controllers.py`, `swarm_sim/targets.py`)
- **Heterogeneous Fleet (`get_asset_spec`):**
  - **`SCOUT` (ISR):** $1.55\times$ sensor range, $1.15\times$ comm range, $0.95\times$ speed; shadows hostile targets from standoff range outside lethal defense rings.
  - **`INTERCEPTOR` (Strike):** $1.35\times$ max speed & acceleration; executes high-$g$ Proportional Navigation pursuit and terminal wolfpack sprints.
  - **`RELAY` (Comms Data-Mule):** $1.65\times$ comm range; dynamically bridges multi-hop links (`HOLON_TETHER`) between target coalitions and the swarm centroid.
- **EW Courier Exfiltration (`COURIER_EXFIL`):** When a jammed Scout or isolated drone detects a target inside an RF Jammer zone, it autonomously dashes to the nearest jammer boundary to restore mesh contact and exfiltrate the target track.
- **Hostile SHORAD (Armed CIWS Turret) Targets:** Armed targets feature a lethal engagement envelope (`target_weapon_range`) and fire-control cooldown (`target_fire_cooldown`), shooting down lone encroaching drones with unobstructed LOS. Drones counter this via **Standoff-to-Sprint Wolfpack Saturation**: staging outside the weapon ring (`SHORAD_STAGE`) until coalition quorum is reached, then sprinting in simultaneously (`WOLFPACK_STRIKE`) from all bearings.

### 3. Ostrom–Pitt Algorithmic Self-Governance (`swarm_sim/agents/estimation.py`, `swarm_sim/agents/task_allocation.py`)
- **Ostrom Principles #4 & #5 (Mutual Monitoring, Graduated Sanctions & Forgiveness):**
  - Each drone maintains a peer trust ledger $T_{ij} \in [0.05, 1.0]$ and monitors neighbor track reports against its own clear sensor FOV (`clear_ghost_tracks_in_fov` and $0$-hop sensor locks).
  - Byzantine / cyber-spoofed peers (`spoofed = True`) that broadcast false phantom tracks incur graduated sanctions: (1) variance inflation by $1 / T_{ij}^2$ in Covariance Intersection, (2) exclusion from CBBA coalition slot counts ($T_{ij} < 0.45$), and (3) track quarantine ($T_{ij} < 0.20$), with slow trust rehabilitation when reports corroborate.
- **Pitt’s *Cyber-Anarcho-Socialism* & Rescher’s Canons of Distributive Justice:**
  - Task bidding in `DecentralizedTaskAllocator.select_target` combines **Canon of Ability** (airframe match), **Canon of Effort / Sacrifice** (onboard `energy` reserve ledger that rotates depleted drones to `ENERGY_RELIEF` standoff while fresh reserves take high-$g$ sprints), and **Canon of Need** (priority reinforcement for under-strength coalitions).
- **Ostrom Principles #2 & #8 (Polycentric Holon Regimes):**
  - Local coalitions self-organize into dynamic institutional regimes (`COMMONS_RECON`, `COALITION_CONVERGE`, `SHORAD_STAGE`, `WOLFPACK_STRIKE`, `ENERGY_RELIEF`, `HOLON_TETHER`, `EW_COURIER_EXFIL`).

### 4. Three-Layer Onboard Autonomy Pipeline
1. **Layer 1 — Perception (`LocalWorldEstimator`):** Distributed Kalman Filter (DKF), Covariance Intersection track fusion, negative-information ghost clearing, and stigmergic search certainty grid fusion (`cpp_update_and_merge_grids`).
2. **Layer 2 — Guidance & AI (`DecentralizedDroneAgent`, `DecentralizedAttentionPolicy`):**
   - `hybrid`: CBBA + Rescher multi-canon auction, True Proportional Navigation ($a_{\text{pn}} = N' V_c \dot{\lambda}$), wall-aware vortex encirclement, and Voronoi frontier exploration (`cpp_compute_frontier_search`).
   - `marl`: Permutation-invariant DeepSets + Local Attention neural policy with matrix-free SO(2)-equivariant residual projection and live OpenAI-ES rollout training (`DecentralizedSwarmParallelEnv`).
   - `boids_pursuit`: Classical Reynolds Boids + greedy pursuit baseline.
3. **Layer 3 — Safety Filter (`_apply_cbf_safety_shield`):** Reactive Control Barrier Function (CBF) velocity shielding against terrain obstacles, operator No-Fly Zones (`NoFlyZone`), arena boundaries, and peer drones.

---

## Repository Structure

| Path | Description |
|---|---|
| `swarm_sim/config.py` | `SimulationConfig` dataclass for physics, sensors, comms, EW, SHORAD, and ROE parameters. |
| `swarm_sim/world.py` | `SwarmWorld2D` step orchestrator, interactive C2 mutations, and WebSocket snapshot serializer. |
| `swarm_sim/physics.py` | Double-integrator kinematics, obstacle/inter-drone collision checks, and Python LOS ray-casting. |
| `swarm_sim/communication.py` | `MeshNetwork`, EW jamming, EMCON filtering, and matrix-free Union-Find / Fiedler $\lambda_2$ solver. |
| `swarm_sim/targets.py` | `TargetManager`, evasive/patrol maneuvers, Armed SHORAD turret defense, and capture logic. |
| `swarm_sim/agents/controllers.py` | `DecentralizedDroneAgent`, heterogeneous specs, Pro-Nav guidance, energy ledger, and CBF shield. |
| `swarm_sim/agents/estimation.py` | `LocalWorldEstimator`, DKF, Covariance Intersection, ghost clearing, and Ostrom peer trust ledger. |
| `swarm_sim/agents/task_allocation.py` | `DecentralizedTaskAllocator`, CBBA marginal utility + Rescher distributive justice canons. |
| `swarm_sim/native/fast_core.cpp` | C++17 native kernel for LOS ray-casting, vectorized physics, Voronoi search, and grid fusion. |
| `swarm_sim/rl/marl_env.py` & `policy.py` | Parallel MARL environment and `DecentralizedAttentionPolicy` with ES optimizer. |
| `swarm_sim/server.py` & `web/` | FastAPI WebSocket server and modular HTML5 Canvas C2 UI (`index.html`, `app.js`, `renderer.js`, `style.css`). |
| `swarm_sim/benchmark.py` | Headless Monte Carlo benchmark comparing controller modes under contested EW and attrition. |
| `tests/` | Modular pytest suite (`test_physics.py`, `test_communication.py`, `test_controllers.py`, `test_estimation.py`, `test_rl.py`, `test_native_parity.py`). |

For contributor and AI agent guidelines, see [AGENTS.md](AGENTS.md) and [AGENT_RULES.md](AGENT_RULES.md).
