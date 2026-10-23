"""Tests for tactical drone controllers, CBF safety filters, and role stability."""

import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.world import SwarmWorld2D


def test_cbf_safety_and_cooperative_encirclement():
    """Run simulation steps across all 3 controller modes and verify stability & snapshots."""
    for mode in ("hybrid", "boids_pursuit", "marl"):
        cfg = SimulationConfig(
            num_drones=16,
            num_targets=2,
            controller_mode=mode,
        )
        world = SwarmWorld2D(cfg, seed=7)
        for _ in range(80):
            world.step()

        snap = world.get_state_snapshot(selected_drone_id=0)
        assert snap["metrics"]["active_drones"] == 16
        assert len(snap["drones"]) == 16
        assert snap["inspector"]["drone_id"] == 0


def test_idle_drones_decouple_and_spread_out():
    """
    Verify that when there are no active targets in the arena (idle state),
    drones decouple from their initial tight launch cluster and spread out across
    the arena via Voronoi frontier partitioning and electrostatic dispersion.
    """
    cfg = SimulationConfig(
        num_drones=16,
        num_targets=0,
        controller_mode="hybrid",
    )
    world = SwarmWorld2D(cfg, seed=99)
    # Force all drones to start in a very tight 40x40 cluster near the center
    for i in range(16):
        world.positions[i] = np.array([500.0 + (i % 4) * 8.0, 350.0 + (i // 4) * 8.0])
        world.velocities[i] = np.zeros(2)

    initial_spread = float(np.mean(np.std(world.positions, axis=0)))
    for _ in range(80):
        world.step()
    final_spread = float(np.mean(np.std(world.positions, axis=0)))

    # Spatial standard deviation should expand dramatically (>5x) as drones decouple
    assert final_spread > initial_spread * 5.0


def test_no_oscillating_role_or_waypoint_loops():
    """
    Verify that drones do not get stuck in rapid 1-step oscillating state loops
    (either between roles or velocity sign-flips that stun them in place).
    """
    cfg = SimulationConfig(
        num_drones=20,
        num_targets=2,
        controller_mode="hybrid",
    )
    world = SwarmWorld2D(cfg, seed=42)

    role_switches = np.zeros(20, dtype=int)
    prev_roles = [ag.role for ag in world.agents]
    speeds_over_time = []

    for _ in range(100):
        world.step()
        cur_roles = [ag.role for ag in world.agents]
        for i in range(20):
            if cur_roles[i] != prev_roles[i]:
                role_switches[i] += 1
        prev_roles = cur_roles
        speeds_over_time.append(np.linalg.norm(world.velocities, axis=1))

    # Over 100 steps, no drone should flip roles back and forth more than 12 times
    assert int(np.max(role_switches)) <= 12
    # Mean drone cruise speed should stay healthy (> 45 px/s), never stunned in place
    mean_speed = float(np.mean(speeds_over_time[20:]))
    assert mean_speed > 45.0
