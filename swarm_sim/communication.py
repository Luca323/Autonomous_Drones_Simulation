"""Ad-Hoc Decentralized Mesh Communication Network with Native C++17 Topology Kernel."""

import ctypes
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.physics import Obstacle, check_los_occluded
from swarm_sim.native import get_cpp_lib, pack_obstacles_array


@dataclass
class GossipPacket:
    """Message broadcasted by a single drone to its 1-hop mesh neighbors."""

    sender_id: int
    pos: np.ndarray
    vel: np.ndarray
    role: str
    assigned_target_id: int
    target_bids: Dict[int, float]
    target_tracks: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    search_grid_summary: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))


class MeshNetwork:
    """Simulates local peer-to-peer wireless mesh links between active drones."""

    def __init__(self, cfg: SimulationConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.adj_matrix: np.ndarray = np.zeros((0, 0), dtype=bool)
        self.sense_matrix: np.ndarray = np.zeros((0, 0), dtype=bool)
        self.dist_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.links: List[Tuple[int, int, float]] = []
        self.algebraic_connectivity: float = 0.0
        self.connected_components: int = 1
        self._step_counter: int = 0

    def update_topology(
        self,
        positions: np.ndarray,
        active_mask: np.ndarray,
        obstacles: List[Obstacle],
    ) -> np.ndarray:
        """Compute communication & sensor adjacency matrices in C++ and update spectral connectivity."""
        n = positions.shape[0]
        if n <= 1:
            self.adj_matrix = np.zeros((n, n), dtype=bool)
            self.sense_matrix = np.zeros((n, n), dtype=bool)
            self.dist_matrix = np.zeros((n, n), dtype=np.float64)
            self.links = []
            self.algebraic_connectivity = 0.0
            self.connected_components = int(np.sum(active_mask))
            return self.adj_matrix

        r_comm = self.cfg.comm_range
        r_sense = self.cfg.sensor_range
        lib = get_cpp_lib()

        if lib is not None:
            pos_c = np.ascontiguousarray(positions, dtype=np.float64)
            act_c = np.ascontiguousarray(active_mask, dtype=np.uint8)
            obs_c = pack_obstacles_array(obstacles)
            obs_ptr = (
                obs_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
                if len(obstacles) > 0
                else ctypes.POINTER(ctypes.c_double)()
            )

            adj_comm_u8 = np.zeros((n, n), dtype=np.uint8)
            adj_sense_u8 = np.zeros((n, n), dtype=np.uint8)
            dist_mat = np.zeros((n, n), dtype=np.float64)

            lib.cpp_compute_matrices(
                n,
                pos_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                act_c.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                len(obstacles),
                obs_ptr,
                float(r_sense),
                float(r_comm),
                1 if self.cfg.los_occlusion else 0,
                adj_comm_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                adj_sense_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                dist_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )

            self.adj_matrix = adj_comm_u8.astype(bool)
            self.sense_matrix = adj_sense_u8.astype(bool)
            self.dist_matrix = dist_mat

            # Extract upper-triangle links
            ii, jj = np.where(np.triu(self.adj_matrix, k=1))
            inv_r = 1.0 / max(r_comm, 1.0)
            self.links = [
                (int(i), int(j), round(max(0.0, 1.0 - float(dist_mat[i, j]) * inv_r), 3))
                for i, j in zip(ii, jj)
            ]
        else:
            self.adj_matrix = np.zeros((n, n), dtype=bool)
            self.sense_matrix = np.zeros((n, n), dtype=bool)
            self.dist_matrix = np.full((n, n), 99999.0, dtype=np.float64)
            self.links = []
            for i in range(n):
                if not active_mask[i]:
                    continue
                for j in range(i + 1, n):
                    if not active_mask[j]:
                        continue
                    dist = float(np.linalg.norm(positions[i] - positions[j]))
                    self.dist_matrix[i, j] = self.dist_matrix[j, i] = dist
                    if dist <= max(r_comm, r_sense):
                        if self.cfg.los_occlusion and check_los_occluded(
                            positions[i], positions[j], obstacles, margin=-2.0
                        ):
                            continue
                        if dist <= r_comm:
                            self.adj_matrix[i, j] = self.adj_matrix[j, i] = True
                            quality = max(0.0, 1.0 - dist / max(r_comm, 1.0))
                            self.links.append((i, j, round(quality, 3)))
                        if dist <= r_sense:
                            self.sense_matrix[i, j] = self.sense_matrix[j, i] = True

        # Update Laplacian eigenvalues every 4 steps (saves dense eigvalsh overhead while keeping telemetry live)
        if self._step_counter % 4 == 0:
            active_indices = np.where(active_mask)[0]
            m = len(active_indices)
            if m <= 1:
                self.algebraic_connectivity = 0.0
                self.connected_components = m
            else:
                sub_adj = self.adj_matrix[np.ix_(active_indices, active_indices)].astype(
                    np.float64
                )
                degrees = np.sum(sub_adj, axis=1)
                laplacian = np.diag(degrees) - sub_adj
                eigvals = np.linalg.eigvalsh(laplacian)
                eigvals = np.sort(np.maximum(eigvals, 0.0))
                self.algebraic_connectivity = float(eigvals[1])
                self.connected_components = int(np.sum(eigvals < 1e-4))
        self._step_counter += 1

        return self.adj_matrix

    def exchange_packets(
        self,
        outbox: Dict[int, GossipPacket],
        active_mask: np.ndarray,
    ) -> Dict[int, List[GossipPacket]]:
        """Deliver broadcast packets to 1-hop neighbors subject to stochastic packet loss."""
        n = len(active_mask)
        inboxes: Dict[int, List[GossipPacket]] = {i: [] for i in range(n)}
        p_loss = self.cfg.packet_loss_prob

        for i in range(n):
            if not active_mask[i] or i not in outbox:
                continue
            pkt = outbox[i]
            neighbors = np.where(self.adj_matrix[i])[0]
            for j in neighbors:
                if not active_mask[j]:
                    continue
                if p_loss > 0.0 and self.rng.random() < p_loss:
                    continue
                inboxes[int(j)].append(pkt)

        return inboxes
