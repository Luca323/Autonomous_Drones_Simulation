"""Master 2D Decentralized Swarm World orchestrating Physics, Mesh Network, Targets, and Agents."""

import ctypes
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.physics import Obstacle, check_los_occluded, step_double_integrator
from swarm_sim.communication import MeshNetwork, GossipPacket
from swarm_sim.targets import TargetManager
from swarm_sim.agents.controllers import DecentralizedDroneAgent
from swarm_sim.rl.policy import DecentralizedAttentionPolicy
from swarm_sim.native import get_cpp_lib


class SwarmWorld2D:
    """
    Main simulation engine. Enforces strict information isolation between drones:
    each `DecentralizedDroneAgent` only receives its own state, local LOS sensor hits,
    and 1-hop wireless mesh packets.
    """

    def __init__(self, cfg: Optional[SimulationConfig] = None, seed: int = 42):
        self.cfg = cfg or SimulationConfig()
        self.rng = np.random.default_rng(seed)

        self.obstacles: List[Obstacle] = [
            Obstacle(x=ox, y=oy, radius=orad)
            for (ox, oy, orad) in self.cfg.initial_obstacles
        ]
        self.network = MeshNetwork(self.cfg, self.rng)
        self.target_mgr = TargetManager(self.cfg, self.rng)
        self.marl_policy = DecentralizedAttentionPolicy(seed=seed)

        self.sim_time: float = 0.0
        self.step_count: int = 0
        self.total_collisions: int = 0

        self.positions: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self.velocities: np.ndarray = np.zeros((0, 2), dtype=np.float64)
        self.active_mask: np.ndarray = np.zeros(0, dtype=bool)
        self.agents: List[DecentralizedDroneAgent] = []

        self.reset()

    def reset(self) -> None:
        """Reset arena, spawn drones in a launch cluster or perimeter, and spawn targets."""
        self.sim_time = 0.0
        self.step_count = 0
        self.total_collisions = 0

        n = self.cfg.num_drones
        self.positions = np.zeros((n, 2), dtype=np.float64)
        self.velocities = np.zeros((n, 2), dtype=np.float64)
        self.active_mask = np.ones(n, dtype=bool)

        # Spawn drones in a deployment zone near the left-center / bottom-left away from obstacles
        for i in range(n):
            for _ in range(100):
                # Spread across launch zone
                px = float(self.rng.uniform(55.0, 220.0))
                py = float(self.rng.uniform(70.0, self.cfg.world_height - 70.0))
                p = np.array([px, py], dtype=np.float64)
                if all(
                    np.linalg.norm(p - np.array([o.x, o.y])) > o.radius + 22.0
                    for o in self.obstacles
                ):
                    self.positions[i] = p
                    break
            ang = float(self.rng.uniform(-0.6, 0.6))
            spd = float(self.rng.uniform(20.0, 45.0))
            self.velocities[i] = np.array([np.cos(ang) * spd, np.sin(ang) * spd])

        self.agents = [DecentralizedDroneAgent(i, self.cfg) for i in range(n)]
        self.target_mgr.reset(self.obstacles)
        self.network.update_topology(self.positions, self.active_mask, self.obstacles)

    def _gather_local_sensors(
        self, drone_idx: int
    ) -> Tuple[
        List[Tuple[int, np.ndarray, np.ndarray, bool]],
        List[Tuple[int, np.ndarray, np.ndarray]],
        List[Obstacle],
    ]:
        """
        Compute strictly local sensory observations for `drone_idx` within `R_sense`
        (respecting obstacle line-of-sight occlusion if enabled).
        """
        own_pos = self.positions[drone_idx]
        r_sense = self.cfg.sensor_range

        # 1. Local target detections
        sensed_targets: List[Tuple[int, np.ndarray, np.ndarray, bool]] = []
        for t in self.target_mgr.targets:
            dist = float(np.linalg.norm(t.pos - own_pos))
            if dist <= r_sense:
                if self.cfg.los_occlusion and check_los_occluded(
                    own_pos, t.pos, self.obstacles, margin=-2.0
                ):
                    continue
                sensed_targets.append((t.target_id, t.pos.copy(), t.vel.copy(), t.captured))

        # 2. Local peer detections (using precomputed C++ sense_matrix when available)
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]] = []
        if self.network.sense_matrix.shape == (len(self.agents), len(self.agents)):
            peer_indices = np.where(self.network.sense_matrix[drone_idx])[0]
            for j in peer_indices:
                sensed_peers.append((int(j), self.positions[j], self.velocities[j]))
        else:
            for j in range(len(self.agents)):
                if j == drone_idx or not self.active_mask[j]:
                    continue
                dist = float(np.linalg.norm(self.positions[j] - own_pos))
                if dist <= r_sense:
                    if self.cfg.los_occlusion and check_los_occluded(
                        own_pos, self.positions[j], self.obstacles, margin=-2.0
                    ):
                        continue
                    sensed_peers.append((j, self.positions[j], self.velocities[j]))

        # 3. Local obstacle detections
        sensed_obstacles: List[Obstacle] = []
        for obs in self.obstacles:
            dx = float(own_pos[0]) - obs.x
            dy = float(own_pos[1]) - obs.y
            if (dx * dx + dy * dy) ** 0.5 - obs.radius <= r_sense:
                sensed_obstacles.append(obs)

        return sensed_targets, sensed_peers, sensed_obstacles

    def step(self) -> None:
        """Advance the entire decentralized simulation by one timestep `dt`."""
        n = len(self.agents)

        # 1. Update wireless mesh network topology
        self.network.update_topology(self.positions, self.active_mask, self.obstacles)

        # 2. Each active drone prepares its broadcast gossip packet from its previous belief
        outbox: Dict[int, GossipPacket] = {}
        for i in range(n):
            if self.active_mask[i]:
                outbox[i] = self.agents[i].prepare_broadcast_packet(
                    self.positions[i], self.velocities[i]
                )

        # 3. Deliver packets across 1-hop wireless mesh links (with packet loss)
        inboxes = self.network.exchange_packets(outbox, self.active_mask)

        # 3b. Batch C++ update of search certainty grids & 1-hop gossip merge
        lib = get_cpp_lib()
        skip_py_grid = False
        if lib is not None and n > 0:
            rows = self.cfg.grid_rows
            cols = self.cfg.grid_cols
            stacked_grids = np.ascontiguousarray(
                [ag.estimator.certainty_grid for ag in self.agents], dtype=np.float32
            )
            pos_c = np.ascontiguousarray(self.positions, dtype=np.float64)
            act_c = np.ascontiguousarray(self.active_mask, dtype=np.uint8)
            adj_c = np.ascontiguousarray(self.network.adj_matrix, dtype=np.uint8)
            lib.cpp_update_and_merge_grids(
                n,
                rows,
                cols,
                pos_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                act_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                adj_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                float(self.cfg.world_width),
                float(self.cfg.world_height),
                float(self.cfg.sensor_range),
                float(self.cfg.pheromone_decay),
                float(self.cfg.gossip_merge_rate),
                stacked_grids.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            for i in range(n):
                if self.active_mask[i]:
                    self.agents[i].estimator.certainty_grid = stacked_grids[i]
            skip_py_grid = True

        # 4. Each drone independently executes its 3-layer perception -> AI -> CBF safety loop
        accel_cmds = np.zeros_like(self.positions)
        for i in range(n):
            if not self.active_mask[i]:
                continue
            sensed_targets, sensed_peers, sensed_obstacles = self._gather_local_sensors(i)
            accel_cmds[i] = self.agents[i].compute_control(
                own_pos=self.positions[i],
                own_vel=self.velocities[i],
                sensed_targets=sensed_targets,
                sensed_peers=sensed_peers,
                sensed_obstacles=sensed_obstacles,
                inbox=inboxes[i],
                rng=self.rng,
                marl_policy=self.marl_policy,
                skip_grid=skip_py_grid,
            )

        # 5. Step drone double-integrator kinematics
        self.positions, self.velocities, step_cols = step_double_integrator(
            positions=self.positions,
            velocities=self.velocities,
            accel_cmds=accel_cmds,
            active_mask=self.active_mask,
            obstacles=self.obstacles,
            entity_radius=self.cfg.drone_radius,
            max_speed=self.cfg.max_speed,
            max_accel=self.cfg.max_accel,
            drag_coeff=self.cfg.drag_coeff,
            cfg=self.cfg,
        )
        self.total_collisions += step_cols

        # 6. Step evasive targets & check encirclement capture progress
        self.target_mgr.step(
            sim_time=self.sim_time,
            drone_positions=self.positions,
            drone_active=self.active_mask,
            obstacles=self.obstacles,
        )

        # Respawn new target if all targets have been captured so the simulation stays lively
        if self.target_mgr.targets and all(t.captured for t in self.target_mgr.targets):
            # Keep captured ones or spawn a fresh wave if user hasn't disabled auto-wave
            pass

        self.sim_time += self.cfg.dt
        self.step_count += 1

    # --- Interactive Mutations from UI ---
    def kill_drone(self, drone_id: int) -> None:
        if 0 <= drone_id < len(self.active_mask):
            self.active_mask[drone_id] = False
            self.velocities[drone_id] = 0.0
            self.agents[drone_id].role = "OFFLINE"

    def emp_blast(self, x: float, y: float, radius: float = 110.0) -> int:
        """Disable all drones within `radius` of `(x, y)` to demonstrate self-healing."""
        center = np.array([x, y], dtype=np.float64)
        killed = 0
        for i in range(len(self.agents)):
            if self.active_mask[i] and float(np.linalg.norm(self.positions[i] - center)) <= radius:
                self.kill_drone(i)
                killed += 1
        return killed

    def add_drone(self, x: float, y: float) -> int:
        new_id = len(self.agents)
        self.positions = np.vstack([self.positions, [[x, y]]])
        self.velocities = np.vstack([self.velocities, [[0.0, 0.0]]])
        self.active_mask = np.append(self.active_mask, True)
        self.agents.append(DecentralizedDroneAgent(new_id, self.cfg))
        self.cfg.num_drones = len(self.agents)
        return new_id

    def add_obstacle(self, x: float, y: float, radius: float = 42.0) -> None:
        self.obstacles.append(Obstacle(x=x, y=y, radius=radius))

    def clear_obstacles(self) -> None:
        self.obstacles = []

    def get_state_snapshot(self, selected_drone_id: int = 0) -> Dict[str, Any]:
        """Serialize simulation state for 60 FPS WebSocket streaming to the Web Visualizer."""
        n = len(self.agents)
        active_indices = np.where(self.active_mask)[0]

        # Compute min inter-drone separation & average uncertainty
        min_sep = 999.0
        if len(active_indices) > 1:
            active_pos = self.positions[active_indices]
            diff = active_pos[:, None, :] - active_pos[None, :, :]
            dists = np.linalg.norm(diff, axis=-1)
            np.fill_diagonal(dists, 999.0)
            min_sep = float(np.min(dists))

        # Global mean certainty across active drones
        if len(active_indices) > 0:
            mean_grid = np.mean(
                [self.agents[i].estimator.certainty_grid for i in active_indices], axis=0
            )
            coverage_pct = float(np.mean(mean_grid) * 100.0)
        else:
            mean_grid = np.zeros((self.cfg.grid_rows, self.cfg.grid_cols))
            coverage_pct = 0.0

        # Compute consensus estimation error across drones that track active targets
        consensus_errors = []
        true_targets = {t.target_id: t for t in self.target_mgr.targets if not t.captured}
        for i in active_indices:
            for tid, trk in self.agents[i].estimator.tracks.items():
                if tid in true_targets and not trk.captured:
                    err = float(np.linalg.norm(trk.pos - true_targets[tid].pos))
                    consensus_errors.append(err)
        mean_est_error = float(np.mean(consensus_errors)) if consensus_errors else 0.0

        # Inspect selected drone's private internal belief
        if not (0 <= selected_drone_id < n):
            selected_drone_id = 0
        sel_agent = self.agents[selected_drone_id] if n > 0 else None
        inspector_data = None
        if sel_agent is not None:
            inspector_data = {
                "drone_id": selected_drone_id,
                "active": bool(self.active_mask[selected_drone_id]),
                "role": sel_agent.role,
                "assigned_target_id": sel_agent.allocator.assigned_target_id,
                "bids": sel_agent.allocator.bids,
                "cbf_intervened": sel_agent.cbf_intervened,
                "planned_waypoint": [
                    round(float(sel_agent.planned_waypoint[0]), 1),
                    round(float(sel_agent.planned_waypoint[1]), 1),
                ],
                "certainty_grid": np.round(sel_agent.estimator.certainty_grid, 2).tolist(),
                "private_tracks": [
                    {
                        "target_id": tid,
                        "pos": [round(float(trk.pos[0]), 1), round(float(trk.pos[1]), 1)],
                        "vel": [round(float(trk.vel[0]), 1), round(float(trk.vel[1]), 1)],
                        "std_radius": round(float(np.sqrt(max(trk.variance, 1.0)) * 3.5), 1),
                        "hops": trk.hops,
                        "age": round(float(trk.age), 2),
                    }
                    for tid, trk in sel_agent.estimator.tracks.items()
                    if not trk.captured
                ],
            }

        role_counts = {"SEARCHING": 0, "PURSUING": 0, "ENCIRCLING": 0, "INTERCEPTING": 0, "OFFLINE": 0}
        drones_payload = []
        for i in range(n):
            ag = self.agents[i]
            role = ag.role if self.active_mask[i] else "OFFLINE"
            role_counts[role] = role_counts.get(role, 0) + 1
            drones_payload.append(
                {
                    "id": i,
                    "x": round(float(self.positions[i, 0]), 1),
                    "y": round(float(self.positions[i, 1]), 1),
                    "vx": round(float(self.velocities[i, 0]), 1),
                    "vy": round(float(self.velocities[i, 1]), 1),
                    "active": bool(self.active_mask[i]),
                    "role": role,
                    "target_id": ag.allocator.assigned_target_id,
                    "cbf": ag.cbf_intervened,
                }
            )

        targets_payload = [
            {
                "id": t.target_id,
                "x": round(float(t.pos[0]), 1),
                "y": round(float(t.pos[1]), 1),
                "vx": round(float(t.vel[0]), 1),
                "vy": round(float(t.vel[1]), 1),
                "captured": t.captured,
                "progress": round(float(t.capture_progress), 3),
                "trail": t.trail[-25:],
            }
            for t in self.target_mgr.targets
        ]

        return {
            "sim_time": round(self.sim_time, 2),
            "step": self.step_count,
            "world": {"width": self.cfg.world_width, "height": self.cfg.world_height},
            "config": {
                "controller_mode": self.cfg.controller_mode,
                "sensor_range": self.cfg.sensor_range,
                "comm_range": self.cfg.comm_range,
                "packet_loss_prob": self.cfg.packet_loss_prob,
                "los_occlusion": self.cfg.los_occlusion,
                "target_behavior": self.cfg.target_behavior,
                "desired_encircle_radius": self.cfg.desired_encircle_radius,
            },
            "metrics": {
                "active_drones": int(len(active_indices)),
                "total_drones": n,
                "captured_targets": sum(1 for t in self.target_mgr.targets if t.captured),
                "total_targets": len(self.target_mgr.targets),
                "algebraic_connectivity": round(self.network.algebraic_connectivity, 3),
                "connected_components": self.network.connected_components,
                "min_separation": round(min_sep, 1) if min_sep < 900 else 0.0,
                "coverage_pct": round(coverage_pct, 1),
                "mean_est_error": round(mean_est_error, 1),
                "total_collisions": self.total_collisions,
                "role_counts": role_counts,
            },
            "drones": drones_payload,
            "targets": targets_payload,
            "obstacles": [
                {"x": round(o.x, 1), "y": round(o.y, 1), "r": round(o.radius, 1)}
                for o in self.obstacles
            ],
            "links": self.network.links,
            "inspector": inspector_data,
        }
