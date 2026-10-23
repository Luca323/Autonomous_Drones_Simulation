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


def test_ostrom_pitt_self_governance_trust_and_justice():
    """
    Verify the combined Ostrom-Pitt self-organising governance stack:
      1. Ostrom Principles 4 & 5 (Mutual Monitoring & Graduated Sanctions): When a spoofed peer
         injects false target tracks that contradict a drone's own sensor FOV, its trust score T_ij
         is progressively sanctioned down to quarantine (< 0.20).
      2. Pitt / Rescher Canons of Distributive Justice (Canon of Effort): Fresh high-energy reserves
         bid higher than energy-depleted drones, allowing depleted drones to rotate to ENERGY_RELIEF.
    """
    from swarm_sim.communication import GossipPacket

    cfg = SimulationConfig(num_drones=4, num_targets=1, initial_obstacles=[], initial_jammers=[])
    world = SwarmWorld2D(cfg, seed=21)
    ag0 = world.agents[0]
    own_pos = np.array([300.0, 300.0])

    # 1. Drone 0 scans (300, 300) and sees NO targets
    ag0.estimator.update_from_local_sensors(own_pos, [], world.rng)

    # Neighbor 99 repeatedly broadcasts phantom target tracks at (310, 310) inside Drone 0's clear FOV
    for tid_fake in (10, 11, 12):
        spoof_pkt = GossipPacket(
            sender_id=99,
            pos=np.array([340.0, 300.0]),
            vel=np.zeros(2),
            role="SEARCHING",
            asset_type="SCOUT",
            assigned_target_id=-1,
            target_bids={},
            target_tracks={
                tid_fake: {
                    "pos": [310.0, 310.0],
                    "vel": [0.0, 0.0],
                    "var": 4.0,
                    "hops": 0,
                    "age": 0.0,
                    "captured": False,
                }
            },
        )
        ag0.estimator.fuse_neighbor_gossip([spoof_pkt], skip_grid=True)

    # Drone 0 must have sanctioned Neighbor 99 into quarantine (trust < 0.20)
    assert ag0.estimator.sanctions_applied >= 3
    assert ag0.estimator.peer_trust[99] < 0.20

    # 2. Rescher Canon of Effort: compare task bid of a fresh reserve (energy=1.0) vs depleted (energy=0.22)
    ag0.estimator.update_from_local_sensors(
        own_pos, [(0, np.array([380.0, 300.0]), np.zeros(2), False)], world.rng
    )
    ag0.allocator.select_target(
        own_pos, np.zeros(2), ag0.estimator.tracks, [], asset_type="INTERCEPTOR", own_energy=1.0
    )
    bid_fresh = ag0.allocator.bids[0]

    ag0.allocator.assigned_target_id = -1
    ag0.allocator.select_target(
        own_pos, np.zeros(2), ag0.estimator.tracks, [], asset_type="INTERCEPTOR", own_energy=0.22
    )
    bid_depleted = ag0.allocator.bids[0]
    assert bid_fresh > bid_depleted + 12.0
