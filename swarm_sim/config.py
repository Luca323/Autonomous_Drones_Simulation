"""Configuration and hyperparameters for the 2D Decentralized Drone Swarm Simulation."""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class SimulationConfig:
    """Master configuration for physics, sensors, communication, and decentralized AI."""

    # Arena & Time parameters
    world_width: float = 1000.0
    world_height: float = 700.0
    dt: float = 0.05  # 20 Hz physics step (run at 30-60 TPS in real time)

    # Swarm parameters
    num_drones: int = 24
    drone_radius: float = 6.0
    max_speed: float = 95.0
    max_accel: float = 180.0
    drag_coeff: float = 0.25

    # Sensor & Communication parameters
    sensor_range: float = 155.0
    comm_range: float = 210.0
    packet_loss_prob: float = 0.02
    los_occlusion: bool = True  # Line-of-sight obstacle occlusion for sensors & radio
    max_comm_hops: int = 4      # TTL for multi-hop target gossip

    # Target parameters
    num_targets: int = 3
    target_radius: float = 9.0
    target_max_speed: float = 78.0
    target_max_accel: float = 150.0
    target_evasion_range: float = 180.0
    target_behavior: str = "evasive"  # "evasive", "patrol", "mixed"

    # Decentralized Search Grid parameters
    grid_cols: int = 25
    grid_rows: int = 18
    pheromone_decay: float = 0.992   # Per-step uncertainty growth (decay of certainty)
    gossip_merge_rate: float = 0.65  # How strongly neighbors merge search certainty maps

    # Tactical Encirclement & Interception parameters
    desired_encircle_radius: float = 52.0
    capture_radius: float = 22.0
    min_drones_for_capture: int = 3
    encircle_angular_speed: float = 1.15  # rad/s orbital vortex component
    target_memory_timeout: float = 5.5    # Seconds before stale target track is dropped

    # Safety Filter (CBF / APF) parameters
    safe_drone_dist: float = 26.0
    hard_drone_dist: float = 14.0
    safe_obstacle_dist: float = 28.0
    cbf_alpha: float = 3.5
    repulsion_gain: float = 240.0
    connectivity_weight: float = 0.15  # Soft pull to prevent accidental network fracture

    # AI Mode
    # "hybrid" (DKF + CBBA Auction + Phase Encirclement + CBF Safety)
    # "boids_pursuit" (Classical Reynolds Boids + Greedy Pursuit)
    # "marl" (Decentralized Neural Attention Policy + CBF Safety)
    controller_mode: str = "hybrid"

    # Default Obstacles: list of (x, y, radius)
    initial_obstacles: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (300.0, 220.0, 42.0),
            (680.0, 200.0, 48.0),
            (500.0, 380.0, 55.0),
            (260.0, 510.0, 38.0),
            (740.0, 490.0, 45.0),
        ]
    )
