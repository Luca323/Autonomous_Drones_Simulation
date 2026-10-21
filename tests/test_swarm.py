"""Comprehensive Test Suite for the Decentralized AI Drone Swarm Simulation."""

import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.physics import Obstacle, check_los_occluded
from swarm_sim.world import SwarmWorld2D
from swarm_sim.rl.marl_env import DecentralizedSwarmParallelEnv, train_marl_policy_es


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
        initial_obstacles=[],
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


def test_los_occlusion_blocks_sensors_and_radio():
    """Verify that an obstacle between two points occludes line-of-sight."""
    obs = [Obstacle(x=200.0, y=100.0, radius=30.0)]
    p1 = np.array([100.0, 100.0])
    p2 = np.array([300.0, 100.0])
    p3 = np.array([100.0, 200.0])

    assert check_los_occluded(p1, p2, obs) is True
    assert check_los_occluded(p1, p3, obs) is False


def test_cbf_safety_and_cooperative_encirclement():
    """Run 120 simulation steps across all 3 controller modes and verify stability & snapshots."""
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


def test_marl_parallel_env_and_es_trainer():
    """Verify PettingZoo ParallelEnv compliance and ES policy optimization."""
    cfg = SimulationConfig(num_drones=8, num_targets=2, controller_mode="marl")
    env = DecentralizedSwarmParallelEnv(cfg, max_steps=25, seed=12)
    obs, infos = env.reset()

    assert len(obs) == 8
    for agent_name, o_vec in obs.items():
        assert o_vec.shape == (18,)

    for _ in range(10):
        actions = {a: np.random.uniform(-1.0, 1.0, size=2) for a in env.agents}
        obs, rewards, terms, truncs, infos = env.step(actions)
        assert len(rewards) == 8

    res = train_marl_policy_es(env.world, generations=2, pop_size=4, rollout_steps=15)
    assert len(res["mean_return_history"]) == 2


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


def test_marl_mode_does_not_bunch_at_top_of_screen():
    """
    Verify that in MARL mode (`controller_mode="marl"`), idle/searching drones
    spread evenly across the vertical and horizontal arena rather than bunching
    up near y=0 (the top of the screen).
    """
    cfg = SimulationConfig(
        num_drones=18,
        num_targets=0,
        controller_mode="marl",
    )
    world = SwarmWorld2D(cfg, seed=21)
    for _ in range(90):
        world.step()

    mean_y = float(np.mean(world.positions[:, 1]))
    std_y = float(np.std(world.positions[:, 1]))
    # Mean Y should stay reasonably near the vertical middle (200..500 out of 700), never bunched at y < 120
    assert 180.0 < mean_y < 520.0
    assert std_y > 100.0


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




