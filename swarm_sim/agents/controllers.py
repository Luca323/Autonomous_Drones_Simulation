"""Decentralized Tactical & Safety Controllers for each Autonomous Drone Agent."""

import ctypes
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.physics import Obstacle, clip_vector_norm
from swarm_sim.communication import GossipPacket
from swarm_sim.agents.estimation import LocalWorldEstimator, TargetTrack
from swarm_sim.agents.task_allocation import DecentralizedTaskAllocator
from swarm_sim.native import get_cpp_lib, pack_obstacles_array


class DecentralizedDroneAgent:
    """
    Autonomous onboard software stack for a single drone `i`.
    Strictly isolated from global state: only uses its own kinematics, local sensor
    measurements, and received 1-hop wireless mesh packets.
    """

    def __init__(self, drone_id: int, cfg: SimulationConfig):
        self.drone_id = drone_id
        self.cfg = cfg
        self.estimator = LocalWorldEstimator(drone_id, cfg)
        self.allocator = DecentralizedTaskAllocator(drone_id, cfg)
        self.role: str = "SEARCHING"
        self.desired_vel: np.ndarray = np.zeros(2, dtype=np.float64)
        self.safe_vel: np.ndarray = np.zeros(2, dtype=np.float64)
        self.planned_waypoint: np.ndarray = np.zeros(2, dtype=np.float64)
        self.cbf_intervened: bool = False

    def prepare_broadcast_packet(
        self, own_pos: np.ndarray, own_vel: np.ndarray
    ) -> GossipPacket:
        """Create the wireless gossip packet to send to 1-hop neighbors."""
        return GossipPacket(
            sender_id=self.drone_id,
            pos=own_pos,
            vel=own_vel,
            role=self.role,
            assigned_target_id=self.allocator.assigned_target_id,
            target_bids=self.allocator.bids,
            target_tracks=self.estimator.export_gossip_tracks(),
            search_grid_summary=self.estimator.certainty_grid,
        )

    def compute_control(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        sensed_targets: List[Tuple[int, np.ndarray, np.ndarray, bool]],
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        sensed_obstacles: List[Obstacle],
        inbox: List[GossipPacket],
        rng: np.random.Generator,
        marl_policy: Optional[Any] = None,
        skip_grid: bool = False,
    ) -> np.ndarray:
        """
        Execute the 3-layer decentralized pipeline:
          Layer 1: Local Perception & Distributed Kalman / Grid Consensus
          Layer 2: Tactical AI / Task Allocation & Encirclement / MARL
          Layer 3: Control Barrier Function (CBF) / APF Reactive Safety Shield
        Returns a 2D acceleration command `u_cmd`.
        """
        # --- LAYER 1: ESTIMATION & CONSENSUS ---
        self.estimator.step_prediction_and_decay(own_pos, skip_grid=skip_grid)
        self.estimator.update_from_local_sensors(own_pos, sensed_targets, rng)
        self.estimator.fuse_neighbor_gossip(inbox, skip_grid=skip_grid)

        # --- LAYER 2: TACTICAL DECISION & DESIRED VELOCITY ---
        mode = self.cfg.controller_mode
        if mode == "boids_pursuit":
            v_des = self._compute_boids_pursuit_velocity(
                own_pos, own_vel, sensed_peers, sensed_obstacles, inbox
            )
        elif mode == "marl" and marl_policy is not None:
            v_des = self._compute_marl_velocity(
                own_pos, own_vel, sensed_peers, sensed_obstacles, inbox, marl_policy
            )
        else:
            v_des = self._compute_hybrid_tactical_velocity(
                own_pos, own_vel, sensed_peers, sensed_obstacles, inbox
            )

        self.desired_vel = clip_vector_norm(v_des, self.cfg.max_speed)

        # --- LAYER 3: REACTIVE CBF / APF SAFETY FILTER ---
        self.safe_vel = self._apply_cbf_safety_shield(
            own_pos, own_vel, self.desired_vel, sensed_peers, sensed_obstacles, inbox
        )

        # Velocity tracking controller -> Acceleration command
        kp_vel = 4.2
        u_cmd = kp_vel * (self.safe_vel - own_vel) + self.cfg.drag_coeff * own_vel
        return clip_vector_norm(u_cmd, self.cfg.max_accel)

    def _compute_hybrid_tactical_velocity(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        sensed_obstacles: List[Obstacle],
        inbox: List[GossipPacket],
    ) -> np.ndarray:
        """
        Hybrid Decentralized AI:
          - CBBA / Marginal Utility Target Assignment
          - Decoupled Voronoi Frontier Exploration when idle/searching
          - Predictive Lead Pursuit when far from assigned target
          - Phase-Spaced Vortex Encirclement & Contraction when surrounding target
        """
        tid = self.allocator.select_target(
            own_pos, own_vel, self.estimator.tracks, inbox
        )

        if tid == -1 or tid not in self.estimator.tracks:
            self.role = "SEARCHING"
            return self._compute_frontier_search_velocity(
                own_pos, own_vel, sensed_peers, sensed_obstacles, inbox
            )

        trk = self.estimator.tracks[tid]
        rel_to_target = trk.pos - own_pos
        dist_to_target = float((rel_to_target[0] ** 2 + rel_to_target[1] ** 2) ** 0.5)

        coalition_positions = [own_pos]
        seen_ids = {self.drone_id}
        for pkt in inbox:
            if pkt.assigned_target_id == tid and pkt.sender_id not in seen_ids:
                coalition_positions.append(pkt.pos)
                seen_ids.add(pkt.sender_id)
        for pid, p_pos, _ in sensed_peers:
            dx = float(p_pos[0] - trk.pos[0])
            dy = float(p_pos[1] - trk.pos[1])
            if pid not in seen_ids and (dx * dx + dy * dy) < (self.cfg.desired_encircle_radius * 2.2) ** 2:
                coalition_positions.append(p_pos)
                seen_ids.add(pid)

        r_encircle = self.cfg.desired_encircle_radius
        near_ring_count = sum(
            1
            for p in coalition_positions
            if ((p[0] - trk.pos[0]) ** 2 + (p[1] - trk.pos[1]) ** 2) < (r_encircle * 1.45) ** 2
        )
        # Schmitt-trigger hysteresis for INTERCEPTING vs ENCIRCLING
        was_intercepting = self.role == "INTERCEPTING"
        min_for_intercept = (
            max(2, self.cfg.min_drones_for_capture - 1)
            if was_intercepting
            else self.cfg.min_drones_for_capture
        )
        age_limit = 0.65 if was_intercepting else 0.30
        is_intercepting = near_ring_count >= min_for_intercept and trk.age <= age_limit

        min_circumference_radius = (near_ring_count * (self.cfg.safe_drone_dist * 0.82)) / (2.0 * np.pi)
        if is_intercepting:
            r_active = max(
                self.cfg.capture_radius * 1.05,
                r_encircle * 0.65,
                min_circumference_radius,
            )
        else:
            r_active = max(r_encircle, min_circumference_radius)

        t_lead = min(dist_to_target / max(self.cfg.max_speed, 1.0), 1.2)
        pred_target_pos = trk.pos + trk.vel * (0.55 * t_lead)

        # Schmitt-trigger distance hysteresis for PURSUING <-> ENCIRCLING transition
        # Always reference `r_encircle` (not `r_active`) so contracting `r_active` never shrinks
        # `pursuit_threshold` and kicks the drone back into PURSUING!
        was_encircling = self.role in ("ENCIRCLING", "INTERCEPTING")
        pursuit_threshold = (r_encircle * 2.35) if was_encircling else (r_encircle * 1.60)

        if dist_to_target > pursuit_threshold:
            self.role = "PURSUING"
            self.planned_waypoint = pred_target_pos.copy()
            diff_p = pred_target_pos - own_pos
            norm_p = max(float((diff_p[0] ** 2 + diff_p[1] ** 2) ** 0.5), 1e-6)
            return (diff_p / norm_p) * self.cfg.max_speed + 0.35 * trk.vel

        self.role = "INTERCEPTING" if is_intercepting else "ENCIRCLING"

        r_vec = own_pos - trk.pos
        r_norm = max(float((r_vec[0] ** 2 + r_vec[1] ** 2) ** 0.5), 1e-4)
        u_radial = r_vec / r_norm
        u_tangent = np.array([-u_radial[1], u_radial[0]], dtype=np.float64)

        radial_err = r_norm - r_active
        v_radial = -1.65 * radial_err * u_radial

        my_angle = float(np.arctan2(r_vec[1], r_vec[0]))
        angular_push = 0.0
        if len(coalition_positions) > 1:
            for p in coalition_positions[1:]:
                rv = p - trk.pos
                if (rv[0] * rv[0] + rv[1] * rv[1]) < (r_encircle * 2.2) ** 2:
                    ang_diff = float(np.arctan2(rv[1], rv[0]) - my_angle)
                    ang_diff = (ang_diff + np.pi) % (2.0 * np.pi) - np.pi
                    if abs(ang_diff) < 1e-4:
                        angular_push += 0.5 if (self.drone_id % 2 == 0) else -0.5
                    elif abs(ang_diff) < (2.0 * np.pi / max(len(coalition_positions), 2)) * 1.2:
                        angular_push -= np.sign(ang_diff) * (0.85 / max(abs(ang_diff), 0.35))

        # Keep orbital direction consistent (>= 0.35 rad/s) so drones never stall or flip spin on the ring
        omega = max(0.35, self.cfg.encircle_angular_speed + float(np.clip(angular_push * 0.35, -0.75, 0.95)))
        v_tangent = omega * r_active * u_tangent
        v_ff = 0.92 * trk.vel

        self.planned_waypoint = trk.pos + u_radial * r_active + u_tangent * (omega * 12.0)
        return v_radial + v_tangent + v_ff

    def _compute_frontier_search_velocity(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        sensed_obstacles: List[Obstacle],
        inbox: List[GossipPacket],
    ) -> np.ndarray:
        """
        Decoupled Voronoi-Partitioned Frontier Exploration & Electrostatic Dispersion:
        When drones have no active target ("idle"), they decouple from any tight formation:
          1. Connectivity constraint is disabled so they are free to break formation and spread out.
          2. Each drone commits to a frontier waypoint with hysteresis until reached or scanned.
          3. Smooth peer dispersion with rotational curl prevents collinear deadlocks.
        Accelerated in C++ (`cpp_compute_frontier_search`).
        """
        peer_list = [p[1] for p in sensed_peers]
        seen_ids = {p[0] for p in sensed_peers}
        for pkt in inbox:
            if pkt.sender_id not in seen_ids and pkt.role == "SEARCHING":
                peer_list.append(pkt.pos)
                seen_ids.add(pkt.sender_id)

        lib = get_cpp_lib()
        if lib is not None:
            grid_c = np.ascontiguousarray(self.estimator.certainty_grid, dtype=np.float32)
            rows, cols = grid_c.shape
            n_peers = len(peer_list)
            peer_arr = (
                np.ascontiguousarray(peer_list, dtype=np.float64)
                if n_peers > 0
                else np.zeros((0, 2), dtype=np.float64)
            )
            peer_ptr = (
                peer_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                if n_peers > 0
                else ctypes.POINTER(ctypes.c_double)()
            )
            obs_arr = pack_obstacles_array(sensed_obstacles)
            obs_ptr = (
                obs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                if len(sensed_obstacles) > 0
                else ctypes.POINTER(ctypes.c_double)()
            )
            out_buf = np.zeros(4, dtype=np.float64)

            lib.cpp_compute_frontier_search(
                int(self.drone_id),
                float(own_pos[0]),
                float(own_pos[1]),
                float(own_vel[0]),
                float(own_vel[1]),
                float(self.planned_waypoint[0]),
                float(self.planned_waypoint[1]),
                rows,
                cols,
                grid_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                n_peers,
                peer_ptr,
                len(sensed_obstacles),
                obs_ptr,
                float(self.cfg.world_width),
                float(self.cfg.world_height),
                float(self.cfg.sensor_range),
                float(self.cfg.max_speed),
                out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            self.planned_waypoint = out_buf[2:4].copy()
            return out_buf[0:2].copy()

        # Pure Python fallback
        grid = self.estimator.certainty_grid
        rows, cols = grid.shape
        cell_centers = np.stack([self.estimator._gx.ravel(), self.estimator._gy.ravel()], axis=1)
        uncertainties = 1.0 - grid.ravel()
        diffs = cell_centers - own_pos
        dists = np.linalg.norm(diffs, axis=1)

        peer_penalty = np.zeros_like(dists)
        for p_pos in peer_list:
            p_dists = np.linalg.norm(cell_centers - p_pos, axis=1)
            peer_penalty[p_dists + 10.0 < dists] += 1.65

        scores = uncertainties * 2.8 - 0.0018 * dists - peer_penalty
        best_idx = int(np.argmax(scores))
        target_cell = cell_centers[best_idx]
        self.planned_waypoint = target_cell.copy()

        dir_vec = target_cell - own_pos
        d_norm = max(float(np.linalg.norm(dir_vec)), 1.0)
        v_search = (dir_vec / d_norm) * (0.9 * self.cfg.max_speed)

        spread_r = self.cfg.sensor_range * 1.25
        for p_pos in peer_list:
            rel = own_pos - p_pos
            d = float(np.linalg.norm(rel))
            if d < spread_r and d > 1e-4:
                strength = (spread_r - d) / spread_r
                v_search += (rel / d) * (self.cfg.max_speed * 0.85 * strength * strength)

        return clip_vector_norm(v_search, self.cfg.max_speed)

    def _compute_boids_pursuit_velocity(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        sensed_obstacles: List[Obstacle],
        inbox: List[GossipPacket],
    ) -> np.ndarray:
        """
        Reynolds Boids + Greedy Pursuit when tracking a target;
        When idle (no known targets), cohesion is disabled so drones decouple and disperse!
        """
        active_tracks = [t for t in self.estimator.tracks.values() if not t.captured]
        if active_tracks:
            nearest = min(
                active_tracks,
                key=lambda t: float((t.pos[0] - own_pos[0]) ** 2 + (t.pos[1] - own_pos[1]) ** 2),
            )
            self.allocator.assigned_target_id = nearest.target_id
            self.role = "PURSUING"
            self.planned_waypoint = nearest.pos.copy()
            rel = nearest.pos - own_pos
            d = max(float((rel[0] ** 2 + rel[1] ** 2) ** 0.5), 1.0)
            v_goal = (rel / d) * self.cfg.max_speed * 0.9

            v_sep = np.zeros(2, dtype=np.float64)
            v_align = np.zeros(2, dtype=np.float64)
            if sensed_peers:
                peer_vel = np.array([p[2] for p in sensed_peers])
                v_align = (np.mean(peer_vel, axis=0) - own_vel) * 0.35
                for _, p_pos, _ in sensed_peers:
                    diff = own_pos - p_pos
                    dist = float((diff[0] ** 2 + diff[1] ** 2) ** 0.5)
                    if dist < 55.0 and dist > 1e-4:
                        v_sep += (diff / dist) * ((55.0 - dist) * 1.4)
            return v_goal + 0.6 * v_sep + 0.25 * v_align
        else:
            self.allocator.assigned_target_id = -1
            self.role = "SEARCHING"
            # Decouple during idle time: zero cohesion, pure Voronoi dispersion search
            return self._compute_frontier_search_velocity(
                own_pos, own_vel, sensed_peers, sensed_obstacles, inbox
            )

    def _compute_marl_velocity(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        sensed_obstacles: List[Obstacle],
        inbox: List[GossipPacket],
        marl_policy: Any,
    ) -> np.ndarray:
        """Run the decentralized neural network policy over local ego-centric observations."""
        tid = self.allocator.select_target(
            own_pos, own_vel, self.estimator.tracks, inbox
        )
        trk = self.estimator.tracks.get(tid) if tid != -1 else None
        if trk is not None:
            dx = float(trk.pos[0] - own_pos[0])
            dy = float(trk.pos[1] - own_pos[1])
            self.role = "ENCIRCLING" if (dx * dx + dy * dy) < 110.0 ** 2 else "PURSUING"
        else:
            self.role = "SEARCHING"

        # Include 1-hop searching inbox peers for Voronoi dispersion during MARL search
        peer_list = list(sensed_peers)
        seen_ids = {p[0] for p in sensed_peers}
        for pkt in inbox:
            if pkt.sender_id not in seen_ids and pkt.role == "SEARCHING":
                peer_list.append((pkt.sender_id, pkt.pos, pkt.vel))
                seen_ids.add(pkt.sender_id)

        v_cmd, new_wp = marl_policy.act(
            own_pos=own_pos,
            own_vel=own_vel,
            target_track=trk,
            sensed_peers=peer_list,
            sensed_obstacles=sensed_obstacles,
            certainty_grid=self.estimator.certainty_grid,
            cfg=self.cfg,
            drone_id=self.drone_id,
            prev_waypoint=self.planned_waypoint,
        )
        self.planned_waypoint = new_wp
        return v_cmd

    def _apply_cbf_safety_shield(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        v_des: np.ndarray,
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        sensed_obstacles: List[Obstacle],
        inbox: List[GossipPacket],
    ) -> np.ndarray:
        """
        Control Barrier Function (CBF) & Potential Field Velocity Filter.
        Note: During `SEARCHING` (idle mode), communication-tether pull is completely
        decoupled so drones are free to break out of clusters and cover the whole map!
        """
        v_safe = v_des.copy()
        self.cbf_intervened = False

        # 1. Inter-drone pairwise barrier constraints
        for _, p_pos, p_vel in sensed_peers:
            dx = float(own_pos[0] - p_pos[0])
            dy = float(own_pos[1] - p_pos[1])
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < self.cfg.safe_drone_dist and dist > 1e-5:
                nx = dx / dist
                ny = dy / dist
                h = dist - self.cfg.hard_drone_dist
                rel_v_n = (v_safe[0] - p_vel[0]) * nx + (v_safe[1] - p_vel[1]) * ny
                min_allowed_vn = -self.cfg.cbf_alpha * h
                if rel_v_n < min_allowed_vn:
                    self.cbf_intervened = True
                    delta = min_allowed_vn - rel_v_n
                    v_safe[0] += delta * nx
                    v_safe[1] += delta * ny
                ratio = (self.cfg.safe_drone_dist - dist) / self.cfg.safe_drone_dist
                repulse = self.cfg.repulsion_gain * (ratio ** 1.5) * 0.28
                v_safe[0] += nx * repulse
                v_safe[1] += ny * repulse

        # 2. Obstacle barrier constraints
        for obs in sensed_obstacles:
            dx = float(own_pos[0]) - obs.x
            dy = float(own_pos[1]) - obs.y
            dist = (dx * dx + dy * dy) ** 0.5
            r_hard = obs.radius + self.cfg.drone_radius + 4.0
            r_influence = obs.radius + self.cfg.safe_obstacle_dist
            if dist < r_influence and dist > 1e-5:
                nx = dx / dist
                ny = dy / dist
                h = dist - r_hard
                v_n = v_safe[0] * nx + v_safe[1] * ny
                min_vn = -self.cfg.cbf_alpha * h
                if v_n < min_vn:
                    self.cbf_intervened = True
                    delta = min_vn - v_n
                    v_safe[0] += delta * nx
                    v_safe[1] += delta * ny
                    tx, ty = -ny, nx
                    if tx * v_des[0] + ty * v_des[1] < 0.0:
                        tx, ty = -tx, -ty
                    v_safe[0] += 0.35 * abs(delta) * tx
                    v_safe[1] += 0.35 * abs(delta) * ty

        # 3. Soft connectivity preservation ONLY when pursuing/encircling a target
        # (Decoupled during SEARCHING so idle drones spread out freely across the arena)
        if (
            self.role != "SEARCHING"
            and len(inbox) == 1
            and self.cfg.connectivity_weight > 0.0
        ):
            nbr_pos = inbox[0].pos
            diff_to_nbr = nbr_pos - own_pos
            dist_nbr = float((diff_to_nbr[0] ** 2 + diff_to_nbr[1] ** 2) ** 0.5)
            if dist_nbr > 0.78 * self.cfg.comm_range:
                pull = (diff_to_nbr / max(dist_nbr, 1.0)) * (
                    (dist_nbr - 0.78 * self.cfg.comm_range)
                    * 1.8
                    * self.cfg.connectivity_weight
                )
                v_safe += pull

        # 4. Arena boundary safety margin
        wall_margin = 28.0
        if own_pos[0] < wall_margin and v_safe[0] < 0.0:
            self.cbf_intervened = True
            v_safe[0] = max(v_safe[0], -self.cfg.cbf_alpha * (own_pos[0] - 10.0))
        elif own_pos[0] > self.cfg.world_width - wall_margin and v_safe[0] > 0.0:
            self.cbf_intervened = True
            v_safe[0] = min(
                v_safe[0],
                self.cfg.cbf_alpha * (self.cfg.world_width - 10.0 - own_pos[0]),
            )

        if own_pos[1] < wall_margin and v_safe[1] < 0.0:
            self.cbf_intervened = True
            v_safe[1] = max(v_safe[1], -self.cfg.cbf_alpha * (own_pos[1] - 10.0))
        elif own_pos[1] > self.cfg.world_height - wall_margin and v_safe[1] > 0.0:
            self.cbf_intervened = True
            v_safe[1] = min(
                v_safe[1],
                self.cfg.cbf_alpha * (self.cfg.world_height - 10.0 - own_pos[1]),
            )

        return clip_vector_norm(v_safe, self.cfg.max_speed)
