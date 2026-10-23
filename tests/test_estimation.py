"""Tests for distributed Kalman estimation, covariance intersection, and ghost track clearing."""

import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.world import SwarmWorld2D


def test_ghost_track_clearing_and_single_target_surplus_decoupling():
    """
    Verify two key anti-loop properties when there is only 1 target on the board:
      1. Surplus Coalition Cap: With 20 drones and only 1 target, not all 20 drones
         pile onto the same target ring; surplus drones decouple into SEARCHING mode.
      2. Negative Information (Ghost Track Clearing): If a drone has a stale track for a
         target at (x, y) within its sensor footprint, but the target has moved away,
         the drone immediately clears the phantom track instead of orbiting empty space.
    """
    cfg = SimulationConfig(
        num_drones=20,
        num_targets=1,
        controller_mode="hybrid",
        initial_obstacles=[],
        initial_jammers=[],
    )
    world = SwarmWorld2D(cfg, seed=11)

    # Put all 20 drones near the single target at (500, 350)
    world.target_mgr.targets[0].pos = np.array([500.0, 350.0])
    for i in range(20):
        ang = (2.0 * np.pi * i) / 20.0
        r = 40.0 + (i % 3) * 20.0
        world.positions[i] = np.array([500.0 + np.cos(ang) * r, 350.0 + np.sin(ang) * r])

    world.step()
    world.step()

    # Check that surplus drones decoupled into SEARCHING rather than all 20 crowding 1 target
    searching_count = sum(1 for ag in world.agents if ag.role == "SEARCHING")
    assert searching_count >= 6

    # Now teleport the actual target far away to (900, 600) so the old track at (500, 350) is a phantom ghost
    world.target_mgr.targets[0].pos = np.array([900.0, 600.0])
    world.step()
    world.step()
    world.step()

    # Drones sitting at (500, 350) should have cleared the phantom track via negative information!
    drones_near_old_spot = [
        i for i in range(20) if np.linalg.norm(world.positions[i] - np.array([500.0, 350.0])) < 70.0
    ]
    for i in drones_near_old_spot:
        assert 0 not in world.agents[i].estimator.tracks
