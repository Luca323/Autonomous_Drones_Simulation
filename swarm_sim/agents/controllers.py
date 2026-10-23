"""Decentralized Tactical & Safety Controllers for Heterogeneous Autonomous Drone Assets."""

import ctypes
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.physics import Obstacle, JammerZone, NoFlyZone, clip_vector_norm
from swarm_sim.communication import GossipPacket
from swarm_sim.agents.estimation import LocalWorldEstimator, TargetTrack
from swarm_sim.agents.task_allocation import DecentralizedTaskAllocator
from swarm_sim.native import get_cpp_lib, pack_obstacles_array


def get_asset_spec(drone_id: int, cfg: SimulationConfig) -> Tuple[str, float, float, float]:
    """
    Assign heterogeneous platform capabilities (Asset Type, Speed Mult, Sensor Mult, Comm Mult):
      - "SCOUT" (e.g. Anduril Ghost ISR): 0.95x speed, 1.55x sensor range, 1.15x comm range
      - "INTERCEPTOR" (e.g. Anduril Anvil / Altius): 1.35x speed, 0.88x sensor range, 1.00x comm range
      - "RELAY" (e.g. Comms Data-Mule / Mesh Bridge): 1.00x speed, 1.05x sensor range, 1.65x comm range
    """
    if not cfg.enable_heterogeneous:
        return ("INTERCEPTOR", 1.0, 1.0, 1.0)

    slot = drone_id % 10
    if slot in (0, 3, 6):
        return ("SCOUT", 0.95, 1.55, 1.15)
    elif slot in (4, 9):
        return ("RELAY", 1.00, 1.05, 1.65)
    else:
        return ("INTERCEPTOR", 1.35, 0.88, 1.00)


class DecentralizedDroneAgent:
    """
    Autonomous onboard software stack for a single drone `i`.
    Strictly isolated from global state: only uses its own kinematics, local sensor
    measurements, and received 1-hop wireless mesh packets.
    """

    def __init__(self, drone_id: int, cfg: SimulationConfig):
        self.drone_id = drone_id
        self.cfg = cfg
        self.asset_type, self.speed_mult, self.sensor_mult, self.comm_mult = get_asset_spec(
            drone_id, cfg
        )
        self.estimator = LocalWorldEstimator(drone_id, cfg)
        self.allocator = DecentralizedTaskAllocator(drone_id, cfg)
        self.role: str = "SEARCHING"
        self.desired_vel: np.ndarray = np.zeros(2, dtype=np.float64)
        self.safe_vel: np.ndarray = np.zeros(2, dtype=np.float64)
        self.planned_waypoint: np.ndarray = np.zeros(2, dtype=np.float64)
        self.cbf_intervened: bool = False
        self.jammed: bool = False
        self.los_rate_deg_s: float = 0.0  # Line-of-Sight rotation rate for Pro-Nav telemetry

    @property
    def effective_max_speed(self) -> float:
        return self.cfg.max_speed * self.speed_mult

    @property
    def effective_sensor_range(self) -> float:
        return self.cfg.sensor_range * self.sensor_mult

    @property
    def effective_comm_range(self) -> float:
        return self.cfg.comm_range * self.comm_mult

    def prepare_broadcast_packet(
        self, own_pos: np.ndarray, own_vel: np.ndarray
    ) -> GossipPacket:
        """Create the wireless gossip packet to send to 1-hop neighbors."""
        return GossipPacket(
            sender_id=self.drone_id,
            pos=own_pos,
            vel=own_vel,
            role=self.role,
            asset_type=self.asset_type,
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
        is_jammed: bool = False,
        jammers: Optional[List[JammerZone]] = None,
        nofly_zones: Optional[List[NoFlyZone]] = None,
    ) -> np.ndarray:
        """
        Execute the 3-layer decentralized pipeline:
          Layer 1: Local Perception & Distributed Kalman / Grid Consensus
          Layer 2: Tactical AI / Task Allocation, Courier Exfil, & Pro-Nav Guidance
          Layer 3: Control Barrier Function (CBF) / No-Fly Zone Reactive Safety Shield
        Returns a 2D acceleration command `u_cmd`.
        """
        self.asset_type, self.speed_mult, self.sensor_mult, self.comm_mult = get_asset_spec(
            self.drone_id, self.cfg
        )
        self.jammed = is_jammed
        self.los_rate_deg_s = 0.0

        # --- LAYER 1: ESTIMATION & CONSENSUS ---
        self.estimator.step_prediction_and_decay(own_pos, skip_grid=skip_grid)
        self.estimator.update_from_local_sensors(own_pos, sensed_targets, rng)
        if not is_jammed:
            self.estimator.fuse_neighbor_gossip(inbox, skip_grid=skip_grid)

        # Combine obstacle list with operator-defined No-Fly Zones (NFZs) so search & CBF avoid them
        effective_obstacles = list(sensed_obstacles)
        if nofly_zones:
            for nfz in nofly_zones:
                effective_obstacles.append(Obstacle(x=nfz.x, y=nfz.y, radius=nfz.radius))

        # --- LAYER 2: TACTICAL DECISION & DESIRED VELOCITY ---
        mode = self.cfg.controller_mode
        if self.cfg.mission_play == "wide_recon":
            self.allocator.assigned_target_id = -1
            self.role = "SEARCHING"
            v_des = self._compute_frontier_search_velocity(
                own_pos, own_vel, sensed_peers, effective_obstacles, inbox
            )
        elif mode == "boids_pursuit":
            v_des = self._compute_boids_pursuit_velocity(
                own_pos, own_vel, sensed_peers, effective_obstacles, inbox
            )
        elif mode == "marl" and marl_policy is not None:
            v_des = self._compute_marl_velocity(
                own_pos, own_vel, sensed_peers, effective_obstacles, inbox, marl_policy
            )
        else:
            v_des = self._compute_hybrid_tactical_velocity(
                own_pos, own_vel, sensed_peers, effective_obstacles, inbox, jammers
            )

        v_max = self.effective_max_speed
        self.desired_vel = clip_vector_norm(v_des, v_max)

        # --- LAYER 3: REACTIVE CBF / APF / NFZ SAFETY FILTER ---
        self.safe_vel = self._apply_cbf_safety_shield(
            own_pos, own_vel, self.desired_vel, sensed_peers, effective_obstacles, inbox
        )

        kp_vel = 4.5
        u_cmd = kp_vel * (self.safe_vel - own_vel) + self.cfg.drag_coeff * own_vel
        return clip_vector_norm(u_cmd, self.cfg.max_accel * self.speed_mult)

    def _compute_pronav_velocity(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        target_pos: np.ndarray,
        target_vel: np.ndarray,
    ) -> np.ndarray:
        """
        True Proportional Navigation (Pro-Nav / ZEM) Guidance Law used by Anduril Anvil/Altius
        interceptors during terminal engagement:
          1. Line-of-Sight (LOS) vector: r = p_t - p_i
          2. Relative velocity: v_rel = v_t - v_i
          3. Closing velocity: V_c = - (r . v_rel) / ||r||
          4. LOS rotation rate (2D scalar): lambda_dot = (r_x * v_rel_y - r_y * v_rel_x) / ||r||^2
          5. Normal acceleration command: a_pn = N' * V_c * lambda_dot * n_perp
        """
        r_vec = target_pos - own_pos
        r_sq = float(r_vec[0] ** 2 + r_vec[1] ** 2)
        dist = max(r_sq ** 0.5, 1.0)
        u_los = r_vec / dist
        u_perp = np.array([-u_los[1], u_los[0]], dtype=np.float64)

        v_rel = target_vel - own_vel
        v_closing = max(float(-(r_vec[0] * v_rel[0] + r_vec[1] * v_rel[1]) / dist), 25.0)

        # 2D cross product r x v_rel / ||r||^2 gives angular rotation rate of the Line-of-Sight
        lambda_dot = float(r_vec[0] * v_rel[1] - r_vec[1] * v_rel[0]) / max(r_sq, 10.0)
        self.los_rate_deg_s = round(float(np.degrees(lambda_dot)), 1)

        nav_ratio = self.cfg.pronav_nav_ratio  # Typically N' = 4.0
        a_pn_mag = nav_ratio * v_closing * lambda_dot

        v_max = self.effective_max_speed
        # Combine axial closing drive along LOS with Pro-Nav lateral lead correction
        v_axial = u_los * v_max + 0.55 * target_vel
        v_lateral = u_perp * np.clip(a_pn_mag * 0.45, -v_max * 0.85, v_max * 0.85)
        return clip_vector_norm(v_axial + v_lateral, v_max)

    def _compute_hybrid_tactical_velocity(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        sensed_obstacles: List[Obstacle],
        inbox: List[GossipPacket],
        jammers: Optional[List[JammerZone]] = None,
    ) -> np.ndarray:
        """
        Anduril-grade Heterogeneous Mission Autonomy:
          - EW Courier Exfiltration: If an ISR Scout spots a target inside an RF Jamming zone,
            it cannot broadcast over the radio; if no interceptors are nearby, it dashes to the
            nearest jammer boundary (`COURIER_EXFIL`) to upload the target track to the mesh!
          - Mesh Relay Bridging (`RELAY` assets): Position themselves between active target coalitions
            and the rest of the swarm to maintain multi-hop communication links.
          - ISR Standoff Shadowing (`SCOUT` assets or `standoff_track` ROE): Orbit outside the
            target's evasion range at 1.75x encirclement radius to maintain continuous sensor lock.
          - Pro-Nav Terminal Intercept (`INTERCEPTOR` assets): High-speed Proportional Navigation
            pursuit followed by contracting capture ring.
        """
        v_max = self.effective_max_speed
        tid = self.allocator.select_target(
            own_pos, own_vel, self.estimator.tracks, inbox
        )

        # 1. EW Data-Mule / Courier Exfiltration:
        # If this drone is currently RF-jammed AND holds a fresh direct target track (`hops == 0`),
        # check if it is a SCOUT or alone. By dashing to the edge of the jammer zone, it restores
        # mesh radio contact and alerts the swarm!
        if self.jammed and jammers and tid != -1 and tid in self.estimator.tracks:
            trk = self.estimator.tracks[tid]
            if self.asset_type in ("SCOUT", "RELAY") or len(sensed_peers) < 2:
                for jam in jammers:
                    dx = float(own_pos[0]) - jam.x
                    dy = float(own_pos[1]) - jam.y
                    d_jam = (dx * dx + dy * dy) ** 0.5
                    if d_jam <= jam.radius:
                        self.role = "COURIER_EXFIL"
                        exfil_dir = (
                            np.array([dx, dy]) / max(d_jam, 1.0)
                            if d_jam > 1.0
                            else np.array([-1.0, 0.0])
                        )
                        self.planned_waypoint = np.array([jam.x, jam.y]) + exfil_dir * (
                            jam.radius + 25.0
                        )
                        return exfil_dir * v_max

        # 2. Dedicated Mesh Relay Positioning (`RELAY` asset type):
        # If a RELAY knows of a target track and there are already peers assigned to it,
        # the RELAY positions itself at the midpoint between the target coalition and the swarm centroid!
        if self.asset_type == "RELAY" and tid != -1 and tid in self.estimator.tracks and inbox:
            trk = self.estimator.tracks[tid]
            peer_positions = [pkt.pos for pkt in inbox]
            if len(peer_positions) >= 2:
                swarm_center = np.mean(peer_positions, axis=0)
                bridge_pt = 0.55 * trk.pos + 0.45 * swarm_center
                # Keep relay outside jammer zones if possible
                if jammers:
                    for jam in jammers:
                        b_diff = bridge_pt - np.array([jam.x, jam.y])
                        b_dist = float(np.linalg.norm(b_diff))
                        if b_dist < jam.radius + 20.0 and b_dist > 1e-4:
                            bridge_pt = np.array([jam.x, jam.y]) + (b_diff / b_dist) * (
                                jam.radius + 25.0
                            )
                self.role = "RELAYING"
                self.planned_waypoint = bridge_pt.copy()
                diff_b = bridge_pt - own_pos
                d_b = float((diff_b[0] ** 2 + diff_b[1] ** 2) ** 0.5)
                if d_b > 18.0:
                    return (diff_b / d_b) * min(v_max, d_b * 2.2) + 0.3 * trk.vel
                return 0.4 * trk.vel

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

        # Check Rules of Engagement (ROE) & Asset Type:
        # SCOUT assets (or all drones under "standoff_track" ROE) orbit at a wide standoff ISR ring
        # (1.65x r_encircle) to maintain continuous eye-on-target while INTERCEPTORs close in!
        standoff_only = (self.cfg.mission_play == "standoff_track") or (
            self.asset_type == "SCOUT" and near_ring_count >= self.cfg.min_drones_for_capture
        )

        target_near_wall = (
            trk.pos[0] < 75.0
            or trk.pos[0] > self.cfg.world_width - 75.0
            or trk.pos[1] < 75.0
            or trk.pos[1] > self.cfg.world_height - 75.0
        )
        was_intercepting = self.role == "INTERCEPTING"
        min_for_intercept = (
            max(2, self.cfg.min_drones_for_capture - 1)
            if (was_intercepting or target_near_wall)
            else self.cfg.min_drones_for_capture
        )
        age_limit = 0.65 if was_intercepting else 0.30
        is_intercepting = (
            not standoff_only
            and near_ring_count >= min_for_intercept
            and trk.age <= age_limit
        )

        min_circumference_radius = (near_ring_count * (self.cfg.safe_drone_dist * 0.82)) / (2.0 * np.pi)
        if standoff_only:
            r_active = max(r_encircle * 1.60, min_circumference_radius)
        elif is_intercepting:
            r_active = max(
                self.cfg.capture_radius * (0.85 if target_near_wall else 1.05),
                r_encircle * (0.48 if target_near_wall else 0.65),
                min_circumference_radius * (0.75 if target_near_wall else 1.0),
            )
        else:
            r_active = max(r_encircle, min_circumference_radius)

        # Schmitt-trigger distance hysteresis for PURSUING <-> ENCIRCLING transition
        was_encircling = self.role in ("ENCIRCLING", "INTERCEPTING")
        pursuit_threshold = (r_encircle * 2.35) if was_encircling else (r_encircle * 1.60)

        if dist_to_target > pursuit_threshold:
            self.role = "PURSUING"
            t_lead = min(dist_to_target / max(v_max, 1.0), 1.2)
            self.planned_waypoint = trk.pos + trk.vel * (0.55 * t_lead)
            # Use True Proportional Navigation (Pro-Nav) guidance law for high-speed interception!
            return self._compute_pronav_velocity(own_pos, own_vel, trk.pos, trk.vel)

        self.role = "INTERCEPTING" if is_intercepting else "ENCIRCLING"

        r_vec = own_pos - trk.pos
        r_norm = max(float((r_vec[0] ** 2 + r_vec[1] ** 2) ** 0.5), 1e-4)
        u_radial = r_vec / r_norm
        u_tangent = np.array([-u_radial[1], u_radial[0]], dtype=np.float64)

        # Wall-aware orbit direction: if orbiting along +u_tangent drives the drone into an
        # arena boundary (e.g. upward into the top wall y < 35 when the target is near the top),
        # reverse the tangent direction so the drone swings around the open interior side!
        probe_pt = own_pos + u_tangent * 32.0
        if (
            probe_pt[0] < 32.0
            or probe_pt[0] > self.cfg.world_width - 32.0
            or probe_pt[1] < 32.0
            or probe_pt[1] > self.cfg.world_height - 32.0
        ):
            u_tangent = -u_tangent

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
        Decoupled Voronoi-Partitioned Frontier Exploration & Electrostatic Dispersion.
        Accelerated in C++ (`cpp_compute_frontier_search`).
        """
        peer_list = [p[1] for p in sensed_peers]
        seen_ids = {p[0] for p in sensed_peers}
        for pkt in inbox:
            if pkt.sender_id not in seen_ids and pkt.role == "SEARCHING":
                peer_list.append(pkt.pos)
                seen_ids.add(pkt.sender_id)

        v_max = self.effective_max_speed
        r_sense = self.effective_sensor_range
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
                float(r_sense),
                float(v_max),
                out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
            self.planned_waypoint = out_buf[2:4].copy()
            return out_buf[0:2].copy()

        # Pure Python fallback (with border-cell penalty & inward wall buffer parity)
        grid = self.estimator.certainty_grid
        rows, cols = grid.shape
        cell_centers = np.stack([self.estimator._gx.ravel(), self.estimator._gy.ravel()], axis=1)
        uncertainties = 1.0 - grid.ravel()
        diffs = cell_centers - own_pos
        dists = np.linalg.norm(diffs, axis=1)

        border_mask = np.zeros((rows, cols), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        border_penalty = np.where(border_mask.ravel(), 0.85, 0.0)

        peer_penalty = np.zeros_like(dists)
        for p_pos in peer_list:
            p_dists = np.linalg.norm(cell_centers - p_pos, axis=1)
            peer_penalty[p_dists + 10.0 < dists] += 1.65

        scores = uncertainties * 2.8 - 0.0018 * dists - peer_penalty - border_penalty
        scores[dists < 35.0] = -1e9
        best_idx = int(np.argmax(scores))
        target_cell = cell_centers[best_idx]
        self.planned_waypoint = target_cell.copy()

        dir_vec = target_cell - own_pos
        d_norm = max(float(np.linalg.norm(dir_vec)), 1.0)
        u_dir = dir_vec / d_norm

        rep = np.zeros(2, dtype=np.float64)
        wall_pad = 75.0
        if own_pos[0] < wall_pad:
            rep[0] += 0.55 * ((wall_pad - own_pos[0]) / wall_pad)
        elif own_pos[0] > self.cfg.world_width - wall_pad:
            rep[0] -= 0.55 * ((own_pos[0] - (self.cfg.world_width - wall_pad)) / wall_pad)
        if own_pos[1] < wall_pad:
            rep[1] += 0.55 * ((wall_pad - own_pos[1]) / wall_pad)
        elif own_pos[1] > self.cfg.world_height - wall_pad:
            rep[1] -= 0.55 * ((own_pos[1] - (self.cfg.world_height - wall_pad)) / wall_pad)

        comb = u_dir * 0.88 + rep
        comb_norm = max(float(np.linalg.norm(comb)), 1e-4)
        v_search = (comb / comb_norm) * (0.86 * v_max)
        return clip_vector_norm(v_search, v_max)

    def _compute_boids_pursuit_velocity(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        sensed_obstacles: List[Obstacle],
        inbox: List[GossipPacket],
    ) -> np.ndarray:
        """Reynolds Boids + Greedy Pursuit when tracking a target; decoupled search when idle."""
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
            v_goal = (rel / d) * self.effective_max_speed * 0.9

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
        return v_cmd * self.speed_mult

    def _apply_cbf_safety_shield(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        v_des: np.ndarray,
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        sensed_obstacles: List[Obstacle],
        inbox: List[GossipPacket],
    ) -> np.ndarray:
        """Control Barrier Function (CBF) & No-Fly Zone (NFZ) Reactive Safety Filter."""
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

        # 2. Obstacle & Operator No-Fly Zone (NFZ) barrier constraints
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
        if (
            self.role not in ("SEARCHING", "COURIER_EXFIL")
            and len(inbox) == 1
            and self.cfg.connectivity_weight > 0.0
        ):
            nbr_pos = inbox[0].pos
            diff_to_nbr = nbr_pos - own_pos
            dist_nbr = float((diff_to_nbr[0] ** 2 + diff_to_nbr[1] ** 2) ** 0.5)
            r_comm_eff = self.effective_comm_range
            if dist_nbr > 0.78 * r_comm_eff:
                pull = (diff_to_nbr / max(dist_nbr, 1.0)) * (
                    (dist_nbr - 0.78 * r_comm_eff)
                    * 1.8
                    * self.cfg.connectivity_weight
                )
                v_safe += pull

        # 4. Arena boundary safety margin + tangential wall-glide so drones never stall against walls
        wall_margin = 36.0
        if own_pos[0] < wall_margin and v_safe[0] < 0.0:
            self.cbf_intervened = True
            inward = -v_safe[0]
            v_safe[0] = max(v_safe[0], -self.cfg.cbf_alpha * (own_pos[0] - 14.0))
            v_safe[1] += (0.45 * inward) if (own_pos[1] < self.cfg.world_height * 0.5) else (-0.45 * inward)
        elif own_pos[0] > self.cfg.world_width - wall_margin and v_safe[0] > 0.0:
            self.cbf_intervened = True
            inward = v_safe[0]
            v_safe[0] = min(
                v_safe[0],
                self.cfg.cbf_alpha * (self.cfg.world_width - 14.0 - own_pos[0]),
            )
            v_safe[1] += (0.45 * inward) if (own_pos[1] < self.cfg.world_height * 0.5) else (-0.45 * inward)

        if own_pos[1] < wall_margin and v_safe[1] < 0.0:
            self.cbf_intervened = True
            inward = -v_safe[1]
            v_safe[1] = max(v_safe[1], -self.cfg.cbf_alpha * (own_pos[1] - 14.0))
            v_safe[0] += (0.45 * inward) if (own_pos[0] < self.cfg.world_width * 0.5) else (-0.45 * inward)
        elif own_pos[1] > self.cfg.world_height - wall_margin and v_safe[1] > 0.0:
            self.cbf_intervened = True
            inward = v_safe[1]
            v_safe[1] = min(
                v_safe[1],
                self.cfg.cbf_alpha * (self.cfg.world_height - 14.0 - own_pos[1]),
            )
            v_safe[0] += (0.45 * inward) if (own_pos[0] < self.cfg.world_width * 0.5) else (-0.45 * inward)

        return clip_vector_norm(v_safe, self.effective_max_speed)
