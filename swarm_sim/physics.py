"""2D Double-Integrator Physics Engine with Native C++17 Acceleration."""

import ctypes
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.native import get_cpp_lib, pack_obstacles_array


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float


@dataclass
class JammerZone:
    """Contested Electronic Warfare (EW) RF Jamming Emitter."""

    x: float
    y: float
    radius: float


@dataclass
class NoFlyZone:
    """Operator-designated Keep-Out / No-Fly Zone (NFZ) enforced by the CBF Shield."""

    x: float
    y: float
    radius: float


def check_los_occluded(
    p1: np.ndarray,
    p2: np.ndarray,
    obstacles: List[Obstacle],
    margin: float = 0.0,
) -> bool:
    """Return True if the line segment p1 -> p2 intersects any circular obstacle."""
    if not obstacles:
        return False

    lib = get_cpp_lib()
    if lib is not None:
        obs_arr = pack_obstacles_array(obstacles)
        res = lib.cpp_check_los(
            float(p1[0]),
            float(p1[1]),
            float(p2[0]),
            float(p2[1]),
            len(obstacles),
            obs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            float(margin),
        )
        return bool(res != 0)

    d = p2 - p1
    seg_len_sq = float(d[0] * d[0] + d[1] * d[1])
    if seg_len_sq < 1e-8:
        return False

    for obs in obstacles:
        fx = float(p1[0]) - obs.x
        fy = float(p1[1]) - obs.y
        t = -(fx * float(d[0]) + fy * float(d[1])) / seg_len_sq
        t = max(0.0, min(1.0, t))
        cx = float(p1[0]) + t * float(d[0]) - obs.x
        cy = float(p1[1]) + t * float(d[1]) - obs.y
        r_eff = obs.radius + margin
        if cx * cx + cy * cy < r_eff * r_eff:
            return True
    return False


def clip_vector_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip a 2D vector (or array of 2D vectors) to a maximum Euclidean norm."""
    if vec.ndim == 1:
        norm = float((vec[0] * vec[0] + vec[1] * vec[1]) ** 0.5)
        if norm > max_norm and norm > 1e-8:
            return vec * (max_norm / norm)
        return vec
    else:
        norms = np.sqrt(np.sum(vec * vec, axis=1, keepdims=True))
        scale = np.where(norms > max_norm, max_norm / np.maximum(norms, 1e-8), 1.0)
        return vec * scale


def step_double_integrator(
    positions: np.ndarray,
    velocities: np.ndarray,
    accel_cmds: np.ndarray,
    active_mask: np.ndarray,
    obstacles: List[Obstacle],
    entity_radius: float,
    max_speed: float,
    max_accel: float,
    drag_coeff: float,
    cfg: SimulationConfig,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Advance 2D double-integrator kinematics with drag, obstacle bounce/slide, and arena bounds.
    Uses the compiled C++17 kernel (`fast_core_v2.dll`) when available.
    """
    n = positions.shape[0]
    if n == 0:
        return positions, velocities, 0

    lib = get_cpp_lib()
    if lib is not None:
        new_pos = np.ascontiguousarray(positions.copy(), dtype=np.float64)
        new_vel = np.ascontiguousarray(velocities.copy(), dtype=np.float64)
        acc_c = np.ascontiguousarray(accel_cmds, dtype=np.float64)
        act_c = np.ascontiguousarray(active_mask, dtype=np.uint8)
        obs_c = pack_obstacles_array(obstacles)
        obs_ptr = (
            obs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            if len(obstacles) > 0
            else ctypes.POINTER(ctypes.c_double)()
        )
        check_inter = 1 if (n > 1 and entity_radius == cfg.drone_radius) else 0

        collisions = int(
            lib.cpp_step_physics(
                n,
                new_pos.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                new_vel.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                acc_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                act_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                len(obstacles),
                obs_ptr,
                float(entity_radius),
                float(max_speed),
                float(max_accel),
                float(drag_coeff),
                float(cfg.dt),
                float(cfg.world_width),
                float(cfg.world_height),
                check_inter,
            )
        )
        return new_pos, new_vel, collisions

    # Pure Python fallback if C++ DLL is not loaded
    dt = cfg.dt
    u = clip_vector_norm(accel_cmds.copy(), max_accel)
    u[~active_mask] = 0.0
    new_vel = velocities + (u - drag_coeff * velocities) * dt
    new_vel = clip_vector_norm(new_vel, max_speed)
    new_vel[~active_mask] = 0.0
    new_pos = positions + new_vel * dt
    new_pos[~active_mask] = positions[~active_mask]
    return new_pos, new_vel, 0
