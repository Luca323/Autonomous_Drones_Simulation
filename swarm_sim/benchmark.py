"""
Monte Carlo Mission Autonomy Benchmark Suite.
Evaluates the decentralized swarm across contested EW jamming, packet loss, and EMP attrition.
Run via:
    python -m swarm_sim.benchmark
"""

import time
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.world import SwarmWorld2D


def run_scenario(
    name: str,
    cfg: SimulationConfig,
    emp_at_step: int = -1,
    episodes: int = 3,
    max_steps: int = 360,
) -> dict:
    t0 = time.perf_counter()
    capture_rates = []
    first_detect_times = []
    track_rmses = []
    collisions = []

    for ep in range(episodes):
        world = SwarmWorld2D(cfg, seed=100 + ep * 17)
        detected_step = max_steps

        for s in range(max_steps):
            if s == emp_at_step:
                # Simulate mid-mission kinetic/EMP attrition losing ~25% of the fleet
                world.emp_blast(500.0, 350.0, radius=240.0)

            world.step()

            if detected_step == max_steps:
                if any(len(ag.estimator.tracks) > 0 for ag in world.agents):
                    detected_step = s

        snap = world.get_state_snapshot()
        m = snap["metrics"]
        cap_rate = (m["captured_targets"] / max(m["total_targets"], 1)) * 100.0
        capture_rates.append(cap_rate)
        first_detect_times.append(detected_step * cfg.dt)
        track_rmses.append(m["mean_est_error"])
        collisions.append(m["total_collisions"])

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    step_ms = elapsed_ms / (episodes * max_steps)

    return {
        "name": name,
        "capture_pct": round(float(np.mean(capture_rates)), 1),
        "t_detect_s": round(float(np.mean(first_detect_times)), 2),
        "rmse_px": round(float(np.mean(track_rmses)), 1),
        "collisions": round(float(np.mean(collisions)), 2),
        "step_ms": round(step_ms, 2),
    }


def main() -> None:
    print("=" * 86)
    print("  DECENTRALIZED MISSION AUTONOMY - MONTE CARLO BENCHMARK SUITE (C++17 KERNEL)")
    print("=" * 86)

    scenarios = [
        (
            "1. Baseline (Permissive Comms, Heterogeneous Fleet)",
            SimulationConfig(
                num_drones=24,
                num_targets=3,
                packet_loss_prob=0.02,
                enable_heterogeneous=True,
                initial_jammers=[],
            ),
            -1,
        ),
        (
            "2. Contested EW Jamming Zone + 25% Packet Loss",
            SimulationConfig(
                num_drones=24,
                num_targets=3,
                packet_loss_prob=0.25,
                enable_heterogeneous=True,
                initial_jammers=[(550.0, 350.0, 125.0)],
            ),
            -1,
        ),
        (
            "3. Mid-Mission EMP Attrition Strike (t = 2.5s)",
            SimulationConfig(
                num_drones=24,
                num_targets=3,
                packet_loss_prob=0.10,
                enable_heterogeneous=True,
            ),
            50,
        ),
        (
            "4. EMCON Radio Silence Doctrine (Contact-Only RF)",
            SimulationConfig(
                num_drones=24,
                num_targets=3,
                emcon_radio_silence=True,
                enable_heterogeneous=True,
            ),
            -1,
        ),
    ]

    print(
        f"{'Scenario':<48} | {'Capture%':>8} | {'T_Det(s)':>8} | {'RMSE(px)':>8} | {'Step(ms)':>8}"
    )
    print("-" * 86)
    for title, cfg, emp_step in scenarios:
        res = run_scenario(title, cfg, emp_at_step=emp_step)
        print(
            f"{res['name']:<48} | {res['capture_pct']:>7.1f}% | {res['t_detect_s']:>7.2f}s | {res['rmse_px']:>8.1f} | {res['step_ms']:>6.2f}ms"
        )
    print("=" * 86)


if __name__ == "__main__":
    main()
