# Plan: Make `Simulation_Stuff` Efficient & Navigable for AI Agents

**Target Executor:** Gemini 3.8 (or any coding agent pair-programming in this repo)  
**Repo root:** `c:\Users\Luca-\OneDrive\Documents\GitHub\Simulation_Stuff`  
**OS / Shell:** Windows (PowerShell). Python 3.10. GCC/MinGW (C++17).

> [!IMPORTANT]
> **Ground Rules for the AI Executor**
> 1. **Protect Uncommitted Work:** The working tree contains active, uncommitted user edits across `.py`, `.js`, `.html` files. **NEVER** run `git checkout`, `git reset --hard`, `git clean`, or `git stash`.
> 2. **Zero Functional Drift in Phases 1–4:** Simulation behavior and outputs must remain 100% identical. Fix documentation to match the code, never change code logic to match outdated comments.
> 3. **Preserve Documentation Integrity:** Maintain existing comments, type annotations, and docstrings unless explicitly fixing identified drifts.
> 4. **No Unrequested Commits:** Stage changes or leave them unstaged for user review. Never commit automatically.
> 5. **Verification Gate:** Run the test suite at the end of every phase. If any test fails, diagnose immediately or revert the specific phase edit.

---

## Architecture & Data Contracts Reference (Quick Agent Briefing)

To navigate and edit this repository without spatial or structural errors, keep these core conventions in mind:

### 1. Coordinate System & Geometry
- **Origin `(0, 0)`:** Top-left corner (Canvas convention). Horizontal axis $X \in [0, 1000]$, Vertical axis $Y \in [0, 700]$.
- **Heading / Angles:** 0 radians points along $+X$ (Right/East). Positive rotations follow standard cartesian math, but screen $Y$ points down.
- **Grid Mapping:** Certainty grid cells $(r, c)$ correspond to:
  $$\text{row} = \lfloor y / \text{cell\_size} \rfloor, \quad \text{col} = \lfloor x / \text{cell\_size} \rfloor$$

### 2. Primary Array Shapes & Dtypes
- `positions`: `np.ndarray` shape `(N, 2)`, `float64`
- `velocities`: `np.ndarray` shape `(N, 2)`, `float64`
- `active_mask`: `np.ndarray` shape `(N,)`, `bool`
- `adj_matrix`: `np.ndarray` shape `(N, N)`, `bool`
- `certainty_grid`: `np.ndarray` shape `(grid_rows, grid_cols)`, `float32` (values in $[0.0, 1.0]$)

### 3. Asset Specifications vs. Operational Roles
- **Hardware Asset Types** (Fixed airframe specification determined at spawn in `get_asset_spec`):
  - `SCOUT`: High sensor range ($1.55\times$), lower speed ($0.95\times$), moderate comms ($1.15\times$).
  - `INTERCEPTOR`: High speed ($1.35\times$), high acceleration ($1.35\times$), compact sensor ($0.88\times$), standard comms ($1.00\times$).
  - `RELAY`: Extended radio range ($1.65\times$), standard speed ($1.00\times$), standard sensor ($1.05\times$).
- **Operational Roles** (Dynamic behavioral state machine updated frame-to-frame in `compute_control`):
  - `SEARCHING`: Voronoi / frontier dispersion across unexplored certainty grid cells.
  - `PURSUIT`: Proportional Navigation sprint toward allocated target track.
  - `ENCIRCLING`: Standoff orbit around localized target.
  - `STANDOFF_TRACK`: Standoff shadowing without kinetic terminal closure (under `standoff_track` doctrine).
  - `COURIER_EXFIL`: Autonomous dash out of an EW Jammer zone boundary to restore mesh connectivity and dump target tracks.

---

## Phase 0 — Baseline & Pre-Flight Check (Read-Only)

1. Check git status to record all currently modified and untracked files:
   ```powershell
   git status --short
   ```
2. Verify Python environment and test suite baseline:
   ```powershell
   python -m pytest tests -q
   ```
   *(Expected baseline: 9 passed in ~4.5–5.0s).*
3. Verify sub-second headless execution sanity:
   ```powershell
   python -c "from swarm_sim.world import SwarmWorld2D; from swarm_sim.config import SimulationConfig; w = SwarmWorld2D(SimulationConfig(num_drones=4, num_targets=1)); w.step(); print('Headless OK')"
   ```
4. Verify C++ compiler status:
   ```powershell
   python -c "from swarm_sim.native import get_cpp_lib; lib = get_cpp_lib(); print('Native C++ Kernel Loaded:', lib is not None)"
   ```

---

## Phase 1 — Repository Hygiene & Noise Elimination

### 1.1 Create Root `.gitignore`
Prevent cache files and compiled binaries from cluttering git diffs and agent search tools:
```gitignore
# Python cache & artifacts
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
.pytest_cache/
.ruff_cache/
.venv/
venv/
env/

# Native C++ binaries (compiled dynamically by swarm_sim/native/__init__.py)
swarm_sim/native/*.dll
swarm_sim/native/*.so
swarm_sim/native/*.dylib

# IDE & OS
.vscode/
.idea/
*.swp
Thumbs.db
.DS_Store
```

### 1.2 Untrack Generated Binaries (Retaining Files on Disk)
Remove tracked `.pyc` and `.dll` files from git cache without deleting local files:
```powershell
git rm -r --cached --quiet swarm_sim/__pycache__ swarm_sim/agents/__pycache__ swarm_sim/native/__pycache__ swarm_sim/rl/__pycache__ tests/__pycache__
git rm --cached --quiet swarm_sim/native/fast_core.dll swarm_sim/native/fast_core_v2.dll
```
*Note: Verify `fast_core.dll` is unused (`fast_core_v2.dll` is the active library in `native/__init__.py`). Remove `swarm_sim/native/fast_core.dll` from disk if it is dead.*

### 1.3 Track Untracked Benchmark Script
`swarm_sim/benchmark.py` is documented in `README.md` and contains the Monte Carlo mission benchmark. Track it:
```powershell
git add swarm_sim/benchmark.py
```

**Verification:**
`git ls-files | Select-String '__pycache__|\.dll$'` must produce no output. Run `python -m pytest tests -q` to confirm baseline remains intact.

---

## Phase 2 — Multi-Agent Instruction & Navigation Maps

Create **`AGENTS.md`** at the repo root (concise, high-density, ~120–150 lines). To ensure universal agent compatibility (Gemini, Claude, Cursor, OpenAI), also link standard pointer files.

### 2.1 Pointer Files
- Create `.agents/rules/project.md` containing:  
  `Please read and adhere to [AGENTS.md](../../AGENTS.md) at the repository root.`
- Create `GEMINI.md` containing:  
  `Please refer to [AGENTS.md](AGENTS.md) for development workflows, architecture, and conventions.`

### 2.2 `AGENTS.md` Mandatory Sections

1. **System Overview:**
   - 2-sentence summary: High-performance decentralized 2D multi-agent swarm simulator with C++17 ctypes kernel, double-integrator physics, Voronoi search, and FastAPI/WebSocket real-time visualizer.
2. **Key Commands:**
   - Run Visualizer: `python -m swarm_sim.server` (Serves on `http://127.0.0.1:8000`)
   - Fast Smoke Check (<0.3s): `python -c "from swarm_sim.world import SwarmWorld2D; from swarm_sim.config import SimulationConfig; w = SwarmWorld2D(SimulationConfig(num_drones=4, num_targets=1)); w.step(); print('OK')"`
   - Full Test Suite: `python -m pytest tests -q`
   - Run Mission Autonomy Benchmark: `python -m swarm_sim.benchmark`
   - Native C++ Recompile & Check:
     ```powershell
     g++ -O3 -ffast-math -shared -static-libgcc -static-libstdc++ -o swarm_sim/native/fast_core_v2.dll swarm_sim/native/fast_core.cpp
     ```
3. **Module & File Map:**
   - Table or concise list with file name, responsibility, and main classes/functions:
     - `swarm_sim/config.py`: `SimulationConfig` master dataclass for physics, communications, sensor specs, and doctrine.
     - `swarm_sim/world.py`: `SwarmWorld2D` top-level orchestrator: step sequencing, global state snapshot generation.
     - `swarm_sim/physics.py`: Double-integrator kinematics, barrier functions, obstacle/jammer/NFZ definitions, pure Python ray-casting fallback.
     - `swarm_sim/communication.py`: `MeshNetwork`, RF Jammer masking, 1-hop packet exchange, EMCON radio-silence logic.
     - `swarm_sim/targets.py`: `TargetManager`, evasive dynamic target pathing, capture bounding-box logic.
     - `swarm_sim/agents/controllers.py`: `DecentralizedDroneAgent`, asset specifications, Pro-Nav terminal guidance, CBF safety filter.
     - `swarm_sim/agents/estimation.py`: `LocalWorldEstimator`, Distributed Kalman Filter (DKF), Covariance Intersection, Negative-information ghost clearing.
     - `swarm_sim/agents/task_allocation.py`: `DecentralizedTaskAllocator`, consensus-based auction bidding (CBBA).
     - `swarm_sim/native/fast_core.cpp`: C++17 accelerated kernel (LOS ray-marching, physics integration, Voronoi grid frontier search, max-consensus grid merging).
     - `swarm_sim/native/__init__.py`: Ctypes bridge, auto-compiler, array packaging.
     - `swarm_sim/server.py`: FastAPI server, WebSocket command dispatch and streaming telemetry.
     - `swarm_sim/web/app.js` & `index.html`: Canvas HUD, telemetry dashboard, user interaction handlers.
     - `swarm_sim/rl/marl_env.py` & `policy.py`: PettingZoo-style decentralized MARL environment & attention policy.
4. **Step Pipeline Flow (`SwarmWorld2D.step`):**
   ```mermaid
   flowchart TD
       A["1. Update Mesh Topology (with EW Jammers & Comm Multipliers)"] --> B["2. Gather Broadcast Packets (GossipPacket)"]
       B --> C["3. 1-Hop Mesh Exchange (Loss & Jammer Drop)"]
       C --> D["3b. Batch C++ Grid Update & Merge (or NumPy fallback)"]
       D --> E["4. Local Sensing, Perception & 3-Layer Control (Perception -> AI -> CBF)"]
       E --> F["5. Double-Integrator Physics Integration"]
       F --> G["6. Target Kinematics & Capture Ring Evaluation"]
   ```
5. **C++ vs. Python Dual-Execution Parity Rule:**
   - Every C++ acceleration function has a pure Python/NumPy counterpart.
   - If modifying `fast_core.cpp`, you **must** update the matching Python fallback in `physics.py` / `communication.py` / `world.py` and verify parity.
   - If C++ signature changes, update `argtypes` and `restype` in `swarm_sim/native/__init__.py`.
6. **WebSocket Protocol Specification:**
   - Document incoming `cmd` types (`pause`, `step_once`, `reset`, `select_drone`, `set_param`, `spawn_target`, `spawn_drone`, `add_obstacle`, `add_jammer`, `add_nofly_zone`, `clear_obstacles`, `emp_blast`, `train_marl`).
   - Document outgoing payload keys (`sim_time`, `drones`, `targets`, `obstacles`, `jammers`, `nofly_zones`, `network`, `metrics`, `inspector`).
7. **Task-to-File Quick Lookup Table:**
   | Desired Change | Target File(s) |
   |---|---|
   | Change drone speed, sensor range, or comm range multipliers | `swarm_sim/agents/controllers.py` (`get_asset_spec`), `swarm_sim/config.py` |
   | Modify Pro-Nav pursuit or CBF safety barrier | `swarm_sim/agents/controllers.py` |
   | Modify RF jamming or mesh routing logic | `swarm_sim/communication.py` |
   | Modify Kalman tracking or ghost clearing | `swarm_sim/agents/estimation.py` |
   | Add a new simulation tunable / parameter | `swarm_sim/config.py`, `server.py`, `web/index.html`, `web/app.js` |
   | Accelerate a numerical hot path | `swarm_sim/native/fast_core.cpp` + `swarm_sim/native/__init__.py` |

---

## Phase 3 — Documentation Alignment & Drift Correction

Align docstrings and README to match the exact ground truth of the codebase (source of truth = **code**).

1. **Role Multipliers Synchronization:**
   - In `swarm_sim/agents/controllers.py`, update module docstring to match actual return values from `get_asset_spec`:
     - `SCOUT`: $0.95\times$ speed, $1.55\times$ sensor, $1.15\times$ comm
     - `RELAY`: $1.00\times$ speed, $1.05\times$ sensor, $1.65\times$ comm
     - `INTERCEPTOR`: $1.35\times$ speed, $0.88\times$ sensor, $1.00\times$ comm
   - In `swarm_sim/config.py`, update docstring comments on `scout_fraction`, `interceptor_fraction`, and `relay_fraction` which currently list out-of-date numbers ($2.1\times$, $1.45\times$, $1.7\times$).
2. **DLL Naming Consistency:**
   - In `swarm_sim/physics.py` line 105, change `fast_core.dll` to `fast_core_v2.dll`.
   - Grep across the entire codebase for any other stale `fast_core.dll` occurrences and normalize them.
3. **Cross-Platform Shared Library Extension Notice:**
   - In `swarm_sim/native/__init__.py`, enhance `DLL_PATH` lookup to support `.so` on Linux and `.dylib` on macOS if cross-platform deployment is required.
4. **README Encoding & Cleanup:**
   - Fix corrupted mojibake character in `README.md`: `Command 24?\"60+` $\rightarrow$ `Command 24–60+`.
   - Add a brief note under "Quick Start" referencing `AGENTS.md` for AI agent guidance.

---

## Phase 4 — Explicit Project Metadata & Tooling Configuration

### 4.1 Create `pyproject.toml`
Create a standard `pyproject.toml` so agents and environments can install dependencies deterministically with editable support:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "swarm_sim"
version = "0.1.0"
description = "Decentralized multi-agent drone swarm simulator with C++17 kernel and real-time web C2 visualizer"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.22",
    "fastapi>=0.95",
    "uvicorn[standard]>=0.22",
    "websockets>=11.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "ruff>=0.1.0",
]

[tool.setuptools.packages.find]
include = ["swarm_sim*"]

[tool.setuptools.package-data]
swarm_sim = ["web/*", "native/*.cpp", "native/*.dll"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-q"

[tool.ruff]
line-length = 110
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "I", "B"]
ignore = ["E501"]
```

### 4.2 Non-Destructive Code Health Inspection
- Run `python -m py_compile` across all Python files to ensure syntax validity without modifying code.
- If `ruff` is available in the environment, run `ruff check swarm_sim` (reporting mode only). **Do not run `--fix`** across the repository to prevent merge conflicts with active uncommitted work. Report findings in the summary.

---

## Phase 5 — Modularized Test Suite & Parity Validation

Currently, all 9 unit and integration tests are crammed into a single 264-line file `tests/test_swarm.py`. Refactor this into domain-specific test modules so agents can run targeted tests instantly when editing a single subsystem.

### 5.1 Test Migration Mapping (Strict 1-to-1 Logic Transfer)
Create `tests/conftest.py` for shared fixtures if needed, and distribute existing tests verbatim:
- **`tests/test_physics.py`**:
  - `test_los_occlusion_blocks_sensors_and_radio`
- **`tests/test_communication.py`**:
  - `test_strict_decentralization_and_multi_hop_dkf_propagation`
  - `test_contested_ew_jamming_and_courier_exfiltration`
- **`tests/test_controllers.py`**:
  - `test_cbf_safety_and_cooperative_encirclement`
  - `test_no_oscillating_role_or_waypoint_loops`
  - `test_idle_drones_decouple_and_spread_out`
- **`tests/test_estimation.py`**:
  - `test_ghost_track_clearing_and_single_target_surplus_decoupling`
- **`tests/test_rl.py`**:
  - `test_marl_parallel_env_and_es_trainer`
  - `test_marl_mode_does_not_bunch_at_top_of_screen`
- **`tests/test_native_parity.py`** (New verification test):
  - Test that `check_los_occluded` and physics stepping produce identical results whether C++ is enabled or forced to pure Python fallback (`lib = None`).

### 5.2 Verification Gate
1. Run `python -m pytest tests -q`.
2. Confirm that exactly 10 tests pass (the original 9 + 1 parity test).
3. Safely delete `tests/test_swarm.py` once equivalence is confirmed.

---

## Phase 6 — Architectural Subsystem Decoupling (Optional / Guarded)

> [!CAUTION]
> **Do not execute this phase without explicit user confirmation.**  
> `swarm_sim/agents/controllers.py` is currently 643 lines long and contains active uncommitted user edits.

If authorized by the user, decompose `controllers.py` into:
1. `swarm_sim/agents/roles.py`: Asset specs, role definitions, multipliers.
2. `swarm_sim/agents/guidance.py`: Pro-Nav pursuit, Voronoi frontier search, standoff encirclement vector math.
3. `swarm_sim/agents/safety.py`: Control Barrier Function (CBF), obstacle/peer avoidance, No-Fly Zone filters.
4. `swarm_sim/agents/controllers.py`: High-level `DecentralizedDroneAgent` orchestrator that imports the above and re-exports symbols for 100% backward compatibility.

---

## Verification & Execution Summary Checklist

Before reporting completion to the user, the executor must verify:
- [ ] Working tree uncommitted user code was untouched (no discarded edits).
- [ ] `.gitignore` is active; `git status` shows no `.pyc`, `__pycache__`, or `.dll` files.
- [ ] `swarm_sim/benchmark.py` is tracked.
- [ ] `AGENTS.md` is created and verified against codebase realities.
- [ ] Pointer files (`.agents/rules/project.md`, `GEMINI.md`) exist and resolve to `AGENTS.md`.
- [ ] Stale multiplier documentation in `controllers.py`, `config.py`, and `README.md` is aligned.
- [ ] `pyproject.toml` is present and valid.
- [ ] `python -m pytest tests -q` runs cleanly and matches or exceeds baseline count.
- [ ] Headless smoke test runs and exits 0 in <0.5s.

---

## Copy-Paste Execution Prompt for Gemini 3.8

When ready to execute this plan, prompt the agent with:
```text
Execute AGENT_IMPROVEMENT_PLAN.md Phases 0 through 5.
Adhere strictly to all Ground Rules: do not discard or overwrite uncommitted user changes, do not run git clean/reset/stash, maintain documentation integrity, and verify all test gates after each phase.
```
