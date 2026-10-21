"""Evasive and Patrol Target Controllers."""

from dataclasses import dataclass, field
from typing import List
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.physics import Obstacle, clip_vector_norm, step_double_integrator


@dataclass
class TargetEntity:
    target_id: int
    pos: np.ndarray
    vel: np.ndarray
    captured: bool = False
    capture_progress: float = 0.0  # 0.0 to 1.0
    phase_offset: float = 0.0
    trail: List[List[float]] = field(default_factory=list)


class TargetManager:
    """Manages multiple dynamic targets with evasive or patrol behaviors."""

    def __init__(self, cfg: SimulationConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.targets: List[TargetEntity] = []
        self._next_id: int = 0

    def reset(self, obstacles: List[Obstacle]) -> None:
        self.targets = []
        self._next_id = 0
        for _ in range(self.cfg.num_targets):
            self.spawn_random_target(obstacles)

    def spawn_random_target(self, obstacles: List[Obstacle]) -> TargetEntity:
        """Spawn a target away from arena edges and obstacles."""
        for _ in range(100):
            x = float(self.rng.uniform(140.0, self.cfg.world_width - 140.0))
            y = float(self.rng.uniform(110.0, self.cfg.world_height - 110.0))
            p = np.array([x, y], dtype=np.float64)
            if all(
                np.linalg.norm(p - np.array([o.x, o.y])) > o.radius + 35.0
                for o in obstacles
            ):
                break
        angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        speed = 0.55 * self.cfg.target_max_speed
        v = np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=np.float64)
        t = TargetEntity(
            target_id=self._next_id,
            pos=p,
            vel=v,
            captured=False,
            capture_progress=0.0,
            phase_offset=float(self.rng.uniform(0.0, 2.0 * np.pi)),
        )
        self._next_id += 1
        self.targets.append(t)
        return t

    def spawn_at(self, x: float, y: float) -> TargetEntity:
        angle = float(self.rng.uniform(0.0, 2.0 * np.pi))
        speed = 0.5 * self.cfg.target_max_speed
        t = TargetEntity(
            target_id=self._next_id,
            pos=np.array([x, y], dtype=np.float64),
            vel=np.array([np.cos(angle) * speed, np.sin(angle) * speed], dtype=np.float64),
            captured=False,
            capture_progress=0.0,
            phase_offset=float(self.rng.uniform(0.0, 2.0 * np.pi)),
        )
        self._next_id += 1
        self.targets.append(t)
        return t

    def step(
        self,
        sim_time: float,
        drone_positions: np.ndarray,
        drone_active: np.ndarray,
        obstacles: List[Obstacle],
    ) -> None:
        if not self.targets:
            return

        m = len(self.targets)
        pos_arr = np.array([t.pos for t in self.targets], dtype=np.float64)
        vel_arr = np.array([t.vel for t in self.targets], dtype=np.float64)
        active_mask = np.array([not t.captured for t in self.targets], dtype=bool)
        accels = np.zeros_like(pos_arr)

        active_drones = drone_positions[drone_active] if np.any(drone_active) else np.zeros((0, 2))

        for idx, t in enumerate(self.targets):
            if t.captured:
                t.vel[:] = 0.0
                continue

            u = np.zeros(2, dtype=np.float64)

            # 1. Base wandering / Lissajous patrol drive
            wander_angle = 0.7 * sim_time + t.phase_offset
            u_patrol = np.array([np.cos(wander_angle), np.sin(1.3 * wander_angle)]) * 45.0
            u += u_patrol

            # 2. Evasion from nearby drones
            if self.cfg.target_behavior in ("evasive", "mixed") and len(active_drones) > 0:
                if self.cfg.target_behavior == "mixed" and (t.target_id % 2 == 1):
                    pass
                else:
                    diffs = t.pos - active_drones  # vectors pointing away from drones
                    dists = np.linalg.norm(diffs, axis=1)
                    close_mask = dists < self.cfg.target_evasion_range
                    if np.any(close_mask):
                        for d_vec, d_val in zip(diffs[close_mask], dists[close_mask]):
                            d_safe = max(float(d_val), 8.0)
                            weight = ((self.cfg.target_evasion_range - d_safe) / self.cfg.target_evasion_range) ** 1.5
                            # Add slight tangential juke to break pure radial traps
                            radial = d_vec / d_safe
                            tangent = np.array([-radial[1], radial[0]]) * (1.0 if (t.target_id % 2 == 0) else -1.0)
                            u += (radial + 0.35 * tangent) * weight * self.cfg.target_max_accel * 1.4

            # 3. Obstacle repulsion
            for obs in obstacles:
                c = np.array([obs.x, obs.y], dtype=np.float64)
                diff = t.pos - c
                dist = float(np.linalg.norm(diff))
                clearance = obs.radius + 48.0
                if dist < clearance and dist > 1e-6:
                    u += (diff / dist) * ((clearance - dist) / clearance) * 240.0

            # 4. Wall repulsion
            wall_pad = 95.0
            if t.pos[0] < wall_pad:
                u[0] += ((wall_pad - t.pos[0]) / wall_pad) * 200.0
            elif t.pos[0] > self.cfg.world_width - wall_pad:
                u[0] -= ((t.pos[0] - (self.cfg.world_width - wall_pad)) / wall_pad) * 200.0

            if t.pos[1] < wall_pad:
                u[1] += ((wall_pad - t.pos[1]) / wall_pad) * 200.0
            elif t.pos[1] > self.cfg.world_height - wall_pad:
                u[1] -= ((t.pos[1] - (self.cfg.world_height - wall_pad)) / wall_pad) * 200.0

            accels[idx] = clip_vector_norm(u, self.cfg.target_max_accel)

        new_pos, new_vel, _ = step_double_integrator(
            positions=pos_arr,
            velocities=vel_arr,
            accel_cmds=accels,
            active_mask=active_mask,
            obstacles=obstacles,
            entity_radius=self.cfg.target_radius,
            max_speed=self.cfg.target_max_speed,
            max_accel=self.cfg.target_max_accel,
            drag_coeff=0.2,
            cfg=self.cfg,
        )

        # Update target states & check encirclement/capture criteria
        for idx, t in enumerate(self.targets):
            t.pos = new_pos[idx]
            t.vel = new_vel[idx]
            t.trail.append([round(float(t.pos[0]), 1), round(float(t.pos[1]), 1)])
            if len(t.trail) > 35:
                t.trail.pop(0)

            if t.captured:
                continue

            if len(active_drones) > 0:
                dists = np.linalg.norm(active_drones - t.pos, axis=1)
                encircling_mask = dists <= (self.cfg.desired_encircle_radius + 22.0)
                close_drones = active_drones[encircling_mask]
                num_close = len(close_drones)

                # Check angular coverage around target
                if num_close >= self.cfg.min_drones_for_capture:
                    rel_vecs = close_drones - t.pos
                    angles = np.sort(np.arctan2(rel_vecs[:, 1], rel_vecs[:, 0]))
                    gaps = np.diff(np.append(angles, angles[0] + 2.0 * np.pi))
                    max_gap = float(np.max(gaps))
                    # Good encirclement if max angular gap < 195 deg or enough drones within capture_radius
                    very_close = int(np.sum(dists <= self.cfg.capture_radius * 1.4))
                    if max_gap < 3.4 or very_close >= 2:
                        t.capture_progress = min(1.0, t.capture_progress + self.cfg.dt * 0.55)
                    else:
                        t.capture_progress = max(0.0, t.capture_progress - self.cfg.dt * 0.15)
                else:
                    t.capture_progress = max(0.0, t.capture_progress - self.cfg.dt * 0.25)

                if t.capture_progress >= 1.0:
                    t.captured = True
                    t.vel[:] = 0.0
