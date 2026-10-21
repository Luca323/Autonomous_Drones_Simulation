# Decentralized AI-Controlled Drone Swarm Simulator

A 2D top-down decentralized drone swarm simulation engine with real-time FastAPI/WebSocket visualization, Distributed Kalman Filtering (DKF), Consensus-Based Bundle / Marginal Utility task allocation, cooperative phase-spaced vortex encirclement, Control Barrier Function (CBF) safety shields, and a Multi-Agent Reinforcement Learning (MARL) DeepSets/Attention policy.

## Quick Start

1. **Run the Real-Time Web Simulator:**
   ```bash
   python -m swarm_sim.server
   ```
   Then open **http://127.0.0.1:8000** in your browser.

2. **Run Automated Tests:**
   ```bash
   python -m pytest tests/test_swarm.py -v
   ```

## Architecture Highlights
- **Strict Decentralization (`SwarmWorld2D` & `DecentralizedDroneAgent`)**: No central coordinator or shared global memory. Each drone only accesses its own state, local sensor hits within $R_{\text{sense}}$, and 1-hop wireless mesh packets within $R_{\text{comm}}$ (with obstacle line-of-sight occlusion and packet drop).
- **Layer 1 — Distributed Estimation (`swarm_sim/agents/estimation.py`)**: Covariance Intersection / Distributed Kalman Filter (DKF) for multi-hop target state tracking, plus a stigmergic search certainty grid merged via neighbor gossip.
- **Layer 2 — Tactical AI (`swarm_sim/agents/task_allocation.py`, `controllers.py`, `rl/policy.py`)**:
  - **Hybrid Mode**: Decentralized marginal utility bidding for multi-target coalition splitting, frontier uncertainty exploration, predictive lead pursuit, and phase-spaced vortex encirclement.
  - **MARL Mode**: Permutation-invariant DeepSets + Local Attention neural policy with live OpenAI-ES rollout optimization and a PettingZoo-compatible `DecentralizedSwarmParallelEnv`.
  - **Classical Mode**: Reynolds Boids (separation, alignment, cohesion) + greedy pursuit.
- **Layer 3 — Reactive Safety Filter (`_apply_cbf_safety_shield`)**: Pairwise inter-drone and obstacle Control Barrier Function (CBF) velocity projection and soft algebraic connectivity preservation.
