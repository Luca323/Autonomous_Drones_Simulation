"""Tests for wireless mesh networking, gossip propagation, and EW jamming."""

import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.world import SwarmWorld2D


def test_strict_decentralization_and_multi_hop_dkf_propagation():
    """
    Verify that when only Drone 0 can directly sense a target, the target state
    estimate propagates across a multi-hop chain of drones (0 -> 1 -> 2) via local
    gossip packets, while an isolated Drone 3 out of comm range remains unaware.
    """
    cfg = SimulationConfig(
        num_drones=4,
        num_targets=1,
        sensor_range=100.0,
        comm_range=120.0,
        packet_loss_prob=0.0,
        los_occlusion=False,
        enable_heterogeneous=False,
        initial_obstacles=[],
        initial_jammers=[],
    )
    world = SwarmWorld2D(cfg, seed=1)

    # Place target at (100, 100)
    world.target_mgr.targets[0].pos = np.array([100.0, 100.0])
    world.target_mgr.targets[0].vel = np.array([0.0, 0.0])

    # Place Drone 0 within sensor range of target: (150, 100) -> dist 50 <= 100
    # Place Drone 1 outside sensor range of target, but within comm range of Drone 0: (250, 100) -> dist to T=150, dist to D0=100
    # Place Drone 2 within comm range of Drone 1: (350, 100) -> 2 hops from Drone 0
    # Place Drone 3 far away at (800, 500) -> isolated
    world.positions[0] = np.array([150.0, 100.0])
    world.positions[1] = np.array([250.0, 100.0])
    world.positions[2] = np.array([350.0, 100.0])
    world.positions[3] = np.array([800.0, 500.0])

    # Step 1: Drone 0 senses target directly
    world.step()
    assert 0 in world.agents[0].estimator.tracks
    assert world.agents[0].estimator.tracks[0].hops == 0
    assert 0 not in world.agents[3].estimator.tracks

    # Step 2: Drone 1 receives 1-hop gossip from Drone 0
    world.step()
    assert 0 in world.agents[1].estimator.tracks
    assert world.agents[1].estimator.tracks[0].hops == 1

    # Step 3: Drone 2 receives 2-hop gossip via Drone 1, while isolated Drone 3 still has 0 tracks
    world.step()
    assert 0 in world.agents[2].estimator.tracks
    assert world.agents[2].estimator.tracks[0].hops == 2
    assert 0 not in world.agents[3].estimator.tracks


def test_contested_ew_jamming_and_courier_exfiltration():
    """
    Verify that inside an EW RF Jamming Zone:
      1. Wireless mesh communication links are severed (`jammed_mask` is True).
      2. When a Scout detects a target inside an EW Jamming Zone, it triggers `COURIER_EXFIL`
         to physically dash out of the jammer boundary and alert the rest of the swarm.
    """
    cfg = SimulationConfig(
        num_drones=4,
        num_targets=1,
        enable_heterogeneous=True,
        initial_obstacles=[],
        initial_jammers=[(500.0, 350.0, 120.0)],
    )
    world = SwarmWorld2D(cfg, seed=5)
    # Drone 0 is a SCOUT (slot 0 % 10 == 0). Place it and Target 0 inside the EW Jammer at (520, 350)
    world.positions[0] = np.array([520.0, 350.0])
    world.positions[1] = np.array([540.0, 350.0])
    world.target_mgr.targets[0].pos = np.array([505.0, 350.0])

    world.step()
    assert world.network.jammed_mask[0] == True
    # Because Drone 0 is jammed, its adjacency row must be all False
    assert not np.any(world.network.adj_matrix[0])
    # And Drone 0 should immediately enter COURIER_EXFIL role to escape the jammer and report the target!
    assert world.agents[0].role == "COURIER_EXFIL"


def test_matrix_free_connectivity_and_fiedler_value():
    """
    Verify that the matrix-free Union-Find + Edge-List Shifted Power Iteration solver
    accurately computes connected components and algebraic connectivity (Fiedler value).
    """
    from swarm_sim.communication import MeshNetwork

    active = np.array([0, 1, 2, 3], dtype=np.int32)

    # 1. Disconnected graph (two components: {0,1} and {2,3}) -> c = 2, lambda_2 = 0.0
    c_disc, lam_disc = MeshNetwork._compute_connectivity_matrix_free(
        active, [(0, 1, 0.8), (2, 3, 0.8)], 4
    )
    assert c_disc == 2
    assert lam_disc == 0.0

    # 2. Path graph P_4 (0-1-2-3): exact Fiedler value is 2 * (1 - cos(pi / 4)) = 2 - sqrt(2) ≈ 0.5858
    c_path, lam_path = MeshNetwork._compute_connectivity_matrix_free(
        active, [(0, 1, 0.8), (1, 2, 0.8), (2, 3, 0.8)], 4
    )
    assert c_path == 1
    np.testing.assert_allclose(lam_path, 2.0 - np.sqrt(2.0), atol=1e-2)

    # 3. Complete graph K_4: exact Fiedler value is 4.0
    k4_links = [(i, j, 0.9) for i in range(4) for j in range(i + 1, 4)]
    c_k4, lam_k4 = MeshNetwork._compute_connectivity_matrix_free(active, k4_links, 4)
    assert c_k4 == 1
    np.testing.assert_allclose(lam_k4, 4.0, atol=1e-5)
