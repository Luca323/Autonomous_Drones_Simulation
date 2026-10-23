"""Tests for geometry, obstacles, and line-of-sight physics."""

import numpy as np
from swarm_sim.physics import Obstacle, check_los_occluded


def test_los_occlusion_blocks_sensors_and_radio():
    """Verify that an obstacle between two points occludes line-of-sight."""
    obs = [Obstacle(x=200.0, y=100.0, radius=30.0)]
    p1 = np.array([100.0, 100.0])
    p2 = np.array([300.0, 100.0])
    p3 = np.array([100.0, 200.0])

    assert check_los_occluded(p1, p2, obs) is True
    assert check_los_occluded(p1, p3, obs) is False
