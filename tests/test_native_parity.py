"""Tests verifying numerical equivalence between C++17 kernel and pure Python fallbacks."""

import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.physics import Obstacle, check_los_occluded, step_double_integrator
from swarm_sim.native import get_cpp_lib


def test_cpp_vs_python_los_occlusion_parity(monkeypatch):
    """Verify that C++ and pure Python fallback produce identical LOS occlusion decisions."""
    obs = [
        Obstacle(x=200.0, y=150.0, radius=40.0),
        Obstacle(x=500.0, y=300.0, radius=50.0),
    ]

    test_rays = [
        (np.array([100.0, 150.0]), np.array([300.0, 150.0])),  # Directly crosses obs 0
        (np.array([100.0, 100.0]), np.array([300.0, 100.0])),  # Clears obs 0
        (np.array([400.0, 300.0]), np.array([600.0, 300.0])),  # Directly crosses obs 1
        (np.array([50.0, 50.0]), np.array([800.0, 600.0])),    # Long diagonal
    ]

    for p1, p2 in test_rays:
        # 1. Native C++ evaluation
        res_cpp = check_los_occluded(p1, p2, obs)

        # 2. Pure Python fallback evaluation (forcing get_cpp_lib() -> None)
        monkeypatch.setattr("swarm_sim.physics.get_cpp_lib", lambda: None)
        res_py = check_los_occluded(p1, p2, obs)
        monkeypatch.undo()

        assert res_cpp == res_py, f"Parity mismatch for ray {p1} -> {p2}: cpp={res_cpp}, py={res_py}"


def test_cpp_vs_python_physics_step_parity(monkeypatch):
    """Verify that double-integrator physics kinematics match between C++ and Python fallback."""
    cfg = SimulationConfig(
        world_width=1000.0,
        world_height=700.0,
        dt=0.05,
        max_speed=95.0,
        max_accel=180.0,
        drag_coeff=0.25,
    )
    obs = [Obstacle(x=500.0, y=350.0, radius=50.0)]

    np.random.seed(42)
    pos = np.random.uniform(100.0, 600.0, size=(10, 2))
    vel = np.random.uniform(-40.0, 40.0, size=(10, 2))
    acc = np.random.uniform(-100.0, 100.0, size=(10, 2))

    active_mask = np.ones(10, dtype=bool)

    # 1. Native C++ step
    pos_cpp, vel_cpp, cols_cpp = step_double_integrator(
        positions=pos.copy(),
        velocities=vel.copy(),
        accel_cmds=acc.copy(),
        active_mask=active_mask,
        obstacles=obs,
        entity_radius=cfg.drone_radius,
        max_speed=cfg.max_speed,
        max_accel=cfg.max_accel,
        drag_coeff=cfg.drag_coeff,
        cfg=cfg,
    )

    # 2. Pure Python fallback step
    monkeypatch.setattr("swarm_sim.physics.get_cpp_lib", lambda: None)
    pos_py, vel_py, cols_py = step_double_integrator(
        positions=pos.copy(),
        velocities=vel.copy(),
        accel_cmds=acc.copy(),
        active_mask=active_mask,
        obstacles=obs,
        entity_radius=cfg.drone_radius,
        max_speed=cfg.max_speed,
        max_accel=cfg.max_accel,
        drag_coeff=cfg.drag_coeff,
        cfg=cfg,
    )
    monkeypatch.undo()

    np.testing.assert_allclose(pos_cpp, pos_py, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(vel_cpp, vel_py, rtol=1e-4, atol=1e-4)
    assert cols_cpp == cols_py
