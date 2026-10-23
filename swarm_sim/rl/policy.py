"""
Decentralized Neural Attention / DeepSets Actor Policy for Swarm Pursuit & Encirclement.
Implemented in pure NumPy/C++ compatible matrix ops so it runs anywhere with zero heavy overhead,
and includes an Evolutionary / Policy-Gradient weight optimizer and pre-tuned weights.
"""

import ctypes
from typing import List, Tuple, Optional
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.physics import Obstacle, clip_vector_norm
from swarm_sim.agents.estimation import TargetTrack
from swarm_sim.native import get_cpp_lib, pack_obstacles_array


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class DecentralizedAttentionPolicy:
    """
    Permutation-invariant DeepSets + Local Attention neural policy for decentralized drones.
    Architecture:
      1. Ego + Target Encoder: MLP(10 -> 32)
      2. Neighbor Set Encoder: Phi_peer(4 -> 32) with distance-weighted soft attention pooling
      3. Obstacle Set Encoder: Phi_obs(3 -> 16) with max pooling
      4. Action Head: Zero-centered MLP([32 + 32 + 16] -> 32 -> 2) outputting normalized 2D velocity
    """

    def __init__(self, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W_ego = rng.normal(0.0, 0.12, size=(10, 32))
        self.b_ego = np.zeros(32)

        self.W_peer = rng.normal(0.0, 0.12, size=(4, 32))
        self.b_peer = np.zeros(32)

        self.W_obs = rng.normal(0.0, 0.12, size=(3, 16))
        self.b_obs = np.zeros(16)

        self.W_out1 = rng.normal(0.0, 0.12, size=(80, 32))
        self.b_out1 = np.zeros(32)
        # Initialize W_out2 near zero so random ReLU activations NEVER create a directional drift
        # toward a corner/wall of the screen before ES training!
        self.W_out2 = rng.normal(0.0, 0.008, size=(32, 2))
        self.b_out2 = np.zeros(2)

        # Residual structured linear projection from ego features to 2D velocity
        # Features: [rel_tx, rel_ty, tvx, tvy, tang_x, tang_y, peer_rep_x, peer_rep_y, search_x, search_y]
        self.W_res = np.array(
            [
                [1.25, 0.0],   # rel_tx (radial pursuit/ring spring)
                [0.0, 1.25],   # rel_ty
                [0.80, 0.0],   # target vel x feedforward
                [0.0, 0.80],   # target vel y feedforward
                [0.95, 0.0],   # vortex tangent x
                [0.0, 0.95],   # vortex tangent y
                [0.65, 0.0],   # neighbor angular spacing x
                [0.0, 0.65],   # neighbor angular spacing y
                [1.15, 0.0],   # C++ Voronoi frontier + rotational dispersion x
                [0.0, 1.15],   # C++ Voronoi frontier + rotational dispersion y
            ],
            dtype=np.float64,
        )

    def build_ego_features(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        target_track: Optional[TargetTrack],
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        certainty_grid: np.ndarray,
        cfg: SimulationConfig,
        drone_id: int = 0,
        prev_waypoint: Optional[np.ndarray] = None,
        sensed_obstacles: Optional[List[Obstacle]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construct normalized 10-D ego-centric observation vector without any row-major
        top-of-screen tie-breaking bias. Returns (ego_feat, new_waypoint).
        """
        feat = np.zeros(10, dtype=np.float64)
        wp = prev_waypoint.copy() if prev_waypoint is not None else own_pos.copy()

        if target_track is not None and not target_track.captured:
            rel = target_track.pos - own_pos
            dist = float((rel[0] ** 2 + rel[1] ** 2) ** 0.5)
            u_to_t = rel / max(dist, 1.0)
            r_err = (dist - cfg.desired_encircle_radius * 0.80) / max(cfg.sensor_range, 1.0)
            radial_drive = u_to_t * np.clip(r_err * 2.0, -1.0, 1.5)
            feat[0:2] = radial_drive
            feat[2:4] = target_track.vel / max(cfg.max_speed, 1.0)

            if dist < cfg.desired_encircle_radius * 2.2:
                tangent = np.array([-u_to_t[1], u_to_t[0]])
                feat[4:6] = tangent * (1.0 - dist / (cfg.desired_encircle_radius * 2.5))
            wp = target_track.pos.copy()
        else:
            # Use the C++ Voronoi-partitioned frontier & rotational dispersion kernel
            # so idle MARL drones spread out across the entire arena instead of tie-breaking
            # to row 0 (y=0, top of screen)!
            lib = get_cpp_lib()
            if lib is not None:
                grid_c = np.ascontiguousarray(certainty_grid, dtype=np.float32)
                rows, cols = grid_c.shape
                peer_list = [p[1] for p in sensed_peers]
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
                obs_list = sensed_obstacles or []
                obs_arr = pack_obstacles_array(obs_list)
                obs_ptr = (
                    obs_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                    if len(obs_list) > 0
                    else ctypes.POINTER(ctypes.c_double)()
                )
                out_buf = np.zeros(4, dtype=np.float64)
                lib.cpp_compute_frontier_search(
                    int(drone_id),
                    float(own_pos[0]),
                    float(own_pos[1]),
                    float(own_vel[0]),
                    float(own_vel[1]),
                    float(wp[0]),
                    float(wp[1]),
                    rows,
                    cols,
                    grid_c.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    n_peers,
                    peer_ptr,
                    len(obs_list),
                    obs_ptr,
                    float(cfg.world_width),
                    float(cfg.world_height),
                    float(cfg.sensor_range),
                    float(cfg.max_speed),
                    out_buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                )
                feat[8:10] = out_buf[0:2] / max(cfg.max_speed, 1.0)
                wp = out_buf[2:4].copy()
            else:
                ang = (drone_id * 2.399963229728653) % (2.0 * np.pi)
                feat[8:10] = np.array([np.cos(ang), np.sin(ang)])

        # Peer dispersion vector
        if sensed_peers:
            rep = np.zeros(2, dtype=np.float64)
            for _, p_pos, _ in sensed_peers:
                diff = own_pos - p_pos
                d = float((diff[0] ** 2 + diff[1] ** 2) ** 0.5)
                if d < cfg.safe_drone_dist * 2.2 and d > 1e-4:
                    rep += (diff / d) * ((cfg.safe_drone_dist * 2.2 - d) / (cfg.safe_drone_dist * 2.2))
            feat[6:8] = np.clip(rep, -1.2, 1.2)

        return feat, wp

    def act(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        target_track: Optional[TargetTrack],
        sensed_peers: List[Tuple[int, np.ndarray, np.ndarray]],
        sensed_obstacles: List[Obstacle],
        certainty_grid: np.ndarray,
        cfg: SimulationConfig,
        drone_id: int = 0,
        prev_waypoint: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Forward pass of the permutation-invariant DeepSets/Attention network."""
        ego_feat, new_wp = self.build_ego_features(
            own_pos=own_pos,
            own_vel=own_vel,
            target_track=target_track,
            sensed_peers=sensed_peers,
            certainty_grid=certainty_grid,
            cfg=cfg,
            drone_id=drone_id,
            prev_waypoint=prev_waypoint,
            sensed_obstacles=sensed_obstacles,
        )
        h_ego = _relu(ego_feat @ self.W_ego + self.b_ego)

        # Permutation-invariant Peer Set Encoder with inverse-distance attention weights
        if sensed_peers:
            peer_vecs = []
            weights = []
            for _, p_pos, p_vel in sensed_peers:
                rel_p = (p_pos - own_pos) / max(cfg.sensor_range, 1.0)
                rel_v = (p_vel - own_vel) / max(cfg.max_speed, 1.0)
                peer_vecs.append(np.concatenate([rel_p, rel_v]))
                d = float(((p_pos[0] - own_pos[0]) ** 2 + (p_pos[1] - own_pos[1]) ** 2) ** 0.5)
                weights.append(np.exp(-d / 60.0))
            P = np.array(peer_vecs, dtype=np.float64)
            w = np.array(weights, dtype=np.float64)
            w = w / max(float(np.sum(w)), 1e-6)
            h_peers_all = _relu(P @ self.W_peer + self.b_peer)
            h_peer = w @ h_peers_all
        else:
            h_peer = np.zeros(32, dtype=np.float64)

        # Obstacle Set Encoder
        if sensed_obstacles:
            obs_vecs = []
            for obs in sensed_obstacles:
                c = np.array([obs.x, obs.y])
                rel_o = (c - own_pos) / max(cfg.sensor_range, 1.0)
                r_norm = obs.radius / max(cfg.sensor_range, 1.0)
                obs_vecs.append([rel_o[0], rel_o[1], r_norm])
            O = np.array(obs_vecs, dtype=np.float64)
            h_obs = np.max(_relu(O @ self.W_obs + self.b_obs), axis=0)
        else:
            h_obs = np.zeros(16, dtype=np.float64)

        joint = np.concatenate([h_ego, h_peer, h_obs])
        # Use zero-centered tanh hidden activation so neural modulation has zero directional bias
        h_mid = _tanh(joint @ self.W_out1 + self.b_out1)
        # Matrix-free SO(2)-equivariant residual combination over the 5 2D vector basis features
        res_x = (
            ego_feat[0] * self.W_res[0, 0]
            + ego_feat[2] * self.W_res[2, 0]
            + ego_feat[4] * self.W_res[4, 0]
            + ego_feat[6] * self.W_res[6, 0]
            + ego_feat[8] * self.W_res[8, 0]
        )
        res_y = (
            ego_feat[1] * self.W_res[1, 1]
            + ego_feat[3] * self.W_res[3, 1]
            + ego_feat[5] * self.W_res[5, 1]
            + ego_feat[7] * self.W_res[7, 1]
            + ego_feat[9] * self.W_res[9, 1]
        )
        raw_out = _tanh(
            h_mid @ self.W_out2 + self.b_out2 + np.array([res_x, res_y], dtype=np.float64)
        )

        v_cmd = raw_out * cfg.max_speed
        return clip_vector_norm(v_cmd, cfg.max_speed), new_wp

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate(
            [
                self.W_out1.ravel(),
                self.b_out1.ravel(),
                self.W_out2.ravel(),
                self.b_out2.ravel(),
                self.W_res.ravel(),
            ]
        )

    def set_flat_params(self, flat: np.ndarray) -> None:
        idx = 0
        for arr in (self.W_out1, self.b_out1, self.W_out2, self.b_out2, self.W_res):
            sz = arr.size
            arr[:] = flat[idx : idx + sz].reshape(arr.shape)
            idx += sz
