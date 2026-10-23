"""Tests for MARL PettingZoo environment, policy rollouts, and spatial behavior."""

import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.world import SwarmWorld2D
from swarm_sim.rl.marl_env import DecentralizedSwarmParallelEnv, train_marl_policy_es


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
        if all(terms.values()) or all(truncs.values()):
            break

    # Test small ES training rollout on global world
    world = SwarmWorld2D(cfg, seed=12)
    res = train_marl_policy_es(world, generations=2, pop_size=4, rollout_steps=20)
    assert len(res["mean_return_history"]) == 2


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
