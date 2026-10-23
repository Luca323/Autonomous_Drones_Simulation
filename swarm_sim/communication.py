"""Ad-Hoc Decentralized Mesh Communication Network with EW Jamming, Heterogeneous Range & EMCON."""

import ctypes
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.physics import Obstacle, JammerZone, check_los_occluded
from swarm_sim.native import get_cpp_lib, pack_obstacles_array


@dataclass
class GossipPacket:
    """Message broadcasted by a single drone to its 1-hop mesh neighbors."""

    sender_id: int
    pos: np.ndarray
    vel: np.ndarray
    role: str
    asset_type: str
    assigned_target_id: int
    target_bids: Dict[int, float]
    target_tracks: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    search_grid_summary: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))


class MeshNetwork:
    """Simulates local peer-to-peer wireless mesh links subject to LOS occlusion and EW RF Jamming."""

    def __init__(self, cfg: SimulationConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.adj_matrix: np.ndarray = np.zeros((0, 0), dtype=bool)
        self.sense_matrix: np.ndarray = np.zeros((0, 0), dtype=bool)
        self.dist_matrix: np.ndarray = np.zeros((0, 0), dtype=np.float64)
        self.jammed_mask: np.ndarray = np.zeros(0, dtype=bool)
        self.links: List[Tuple[int, int, float]] = []
        self.adj_lists: List[List[int]] = []
        self.sense_lists: List[List[int]] = []
        self.min_separation: float = 999.0
        self.algebraic_connectivity: float = 0.0
        self.connected_components: int = 1
        self._step_counter: int = 0

    @staticmethod
    def _compute_connectivity_matrix_free(
        active_indices: np.ndarray, links: List[Tuple[int, int, float]], n_total: int
    ) -> Tuple[int, float]:
        """
        Matrix-free O(N + |E|) computation of connected components (via 1D Union-Find)
        and Algebraic Connectivity / Fiedler value lambda_2(L) (via edge-list shifted
        power iteration with mean deflation), replacing O(N^3) dense eigvalsh(L).
        """
        m = len(active_indices)
        if m <= 1:
            return m, 0.0

        global_to_local = np.full(n_total, -1, dtype=np.int32)
        for idx, gid in enumerate(active_indices):
            global_to_local[int(gid)] = idx

        parent = list(range(m))
        degrees = np.zeros(m, dtype=np.float64)
        local_edges: List[Tuple[int, int]] = []

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        components = m
        for u_glob, v_glob, _ in links:
            u = int(global_to_local[u_glob])
            v = int(global_to_local[v_glob])
            if u < 0 or v < 0:
                continue
            local_edges.append((u, v))
            degrees[u] += 1.0
            degrees[v] += 1.0
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
                components -= 1

        if components > 1 or not local_edges:
            return components, 0.0

        if len(local_edges) == (m * (m - 1)) // 2:
            return 1, float(m)

        # Matrix-free shifted power iteration on M = mu * I - L over local_edges
        mu = 2.0 * float(np.max(degrees)) + 1.0
        x = np.linspace(-1.0, 1.0, m, dtype=np.float64)
        x -= np.mean(x)
        norm_x = float(np.linalg.norm(x))
        if norm_x < 1e-12:
            return 1, 0.0
        x /= norm_x

        lx = np.zeros(m, dtype=np.float64)
        for _ in range(12):
            lx.fill(0.0)
            for u, v in local_edges:
                diff = x[u] - x[v]
                lx[u] += diff
                lx[v] -= diff
            y = mu * x - lx
            y -= np.mean(y)
            ny = float(np.linalg.norm(y))
            if ny < 1e-12:
                break
            x = y / ny

        lx.fill(0.0)
        for u, v in local_edges:
            diff = x[u] - x[v]
            lx[u] += diff
            lx[v] -= diff
        lambda_2 = max(0.0, float(np.dot(x, lx)))
        return 1, lambda_2

    def update_topology(
        self,
        positions: np.ndarray,
        active_mask: np.ndarray,
        obstacles: List[Obstacle],
        jammers: Optional[List[JammerZone]] = None,
        comm_multipliers: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute communication & sensor adjacency matrices and apply EW jamming blackouts."""
        n = positions.shape[0]
        self.jammed_mask = np.zeros(n, dtype=bool)
        if n <= 1:
            self.adj_matrix = np.zeros((n, n), dtype=bool)
            self.sense_matrix = np.zeros((n, n), dtype=bool)
            self.dist_matrix = np.zeros((n, n), dtype=np.float64)
            self.links = []
            self.algebraic_connectivity = 0.0
            self.connected_components = int(np.sum(active_mask))
            return self.adj_matrix

        # Check which drones are inside an active EW RF Jammer radius
        if jammers:
            for i in range(n):
                if not active_mask[i]:
                    continue
                px, py = float(positions[i, 0]), float(positions[i, 1])
                for jam in jammers:
                    if (px - jam.x) ** 2 + (py - jam.y) ** 2 <= jam.radius ** 2:
                        self.jammed_mask[i] = True
                        break

        r_comm_max = self.cfg.comm_range * (1.75 if self.cfg.enable_heterogeneous else 1.0)
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
                float(r_sense * 1.65),
                float(r_comm_max),
                1 if self.cfg.los_occlusion else 0,
                adj_comm_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                adj_sense_u8.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                dist_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )

            self.adj_matrix = adj_comm_u8.astype(bool)
            self.sense_matrix = adj_sense_u8.astype(bool)
            self.dist_matrix = dist_mat

            # Build compact O(deg) adjacency lists & min_separation in a single upper-triangular pass
            self.adj_lists = [[] for _ in range(n)]
            self.sense_lists = [[] for _ in range(n)]
            self.links = []
            min_sep = 999.0
            inv_r = 1.0 / max(r_comm_max, 1.0)

            for i in range(n):
                if not active_mask[i]:
                    continue
                if self.jammed_mask[i]:
                    self.adj_matrix[i, :] = False
                    self.adj_matrix[:, i] = False
                for j in range(i + 1, n):
                    if not active_mask[j]:
                        continue
                    d_ij = float(dist_mat[i, j])
                    if d_ij < min_sep:
                        min_sep = d_ij
                    if self.sense_matrix[i, j]:
                        self.sense_lists[i].append(j)
                        self.sense_lists[j].append(i)
                    if not self.adj_matrix[i, j]:
                        continue
                    if self.jammed_mask[i] or self.jammed_mask[j]:
                        self.adj_matrix[i, j] = self.adj_matrix[j, i] = False
                        continue
                    mult = (
                        max(float(comm_multipliers[i]), float(comm_multipliers[j]))
                        if comm_multipliers is not None
                        else 1.0
                    )
                    if d_ij > self.cfg.comm_range * mult:
                        self.adj_matrix[i, j] = self.adj_matrix[j, i] = False
                    else:
                        self.adj_lists[i].append(j)
                        self.adj_lists[j].append(i)
                        self.links.append(
                            (i, j, round(max(0.05, 1.0 - d_ij * inv_r), 3))
                        )
            self.min_separation = min_sep
        else:
            self.adj_matrix = np.zeros((n, n), dtype=bool)
            self.sense_matrix = np.zeros((n, n), dtype=bool)
            self.dist_matrix = np.full((n, n), 99999.0, dtype=np.float64)
            self.adj_lists = [[] for _ in range(n)]
            self.sense_lists = [[] for _ in range(n)]
            self.links = []
            self.min_separation = 999.0

        if self._step_counter % 4 == 0:
            active_indices = np.where(active_mask)[0]
            self.connected_components, self.algebraic_connectivity = (
                self._compute_connectivity_matrix_free(active_indices, self.links, n)
            )
        self._step_counter += 1

        return self.adj_matrix

    def exchange_packets(
        self,
        outbox: Dict[int, GossipPacket],
        active_mask: np.ndarray,
    ) -> Dict[int, List[GossipPacket]]:
        """
        Deliver broadcast packets to 1-hop neighbors subject to stochastic packet loss,
        EW RF Jamming blackout, and EMCON (Emissions Control / Radio Silence) rules.
        """
        n = len(active_mask)
        inboxes: Dict[int, List[GossipPacket]] = {i: [] for i in range(n)}
        p_loss = self.cfg.packet_loss_prob

        for i in range(n):
            if not active_mask[i] or i not in outbox:
                continue
            if i < len(self.jammed_mask) and self.jammed_mask[i]:
                continue  # RF transmitter jammed!
            pkt = outbox[i]
            # Under EMCON (Radio Silence), drones suppress routine broadcasts UNLESS they have
            # a high-priority fresh target contact (hops <= 1) to report!
            if self.cfg.emcon_radio_silence:
                has_fresh_contact = any(
                    int(t.get("hops", 99)) <= 1 and not t.get("captured", False)
                    for t in pkt.target_tracks.values()
                )
                if not has_fresh_contact:
                    continue

            neighbors = (
                self.adj_lists[i]
                if i < len(self.adj_lists)
                else np.where(self.adj_matrix[i])[0]
            )
            for j in neighbors:
                if not active_mask[j]:
                    continue
                if j < len(self.jammed_mask) and self.jammed_mask[j]:
                    continue
                if p_loss > 0.0 and self.rng.random() < p_loss:
                    continue
                inboxes[int(j)].append(pkt)

        return inboxes
