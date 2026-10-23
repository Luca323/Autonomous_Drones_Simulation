"""Distributed Kalman Filter (DKF) & Stigmergic Search Certainty Grid for a single drone."""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.communication import GossipPacket


@dataclass
class TargetTrack:
    """A single drone's local state estimate and uncertainty for a target."""

    target_id: int
    pos: np.ndarray        # [x, y]
    vel: np.ndarray        # [vx, vy]
    variance: float        # Scalar position trace uncertainty (lower = more confident)
    hops: int              # 0 if directly observed this step, >0 if via neighbor gossip
    age: float             # Seconds since last direct observation by whoever originated it
    captured: bool = False
    source_id: int = -1    # Neighbor drone_id that relayed this track (-1 if direct sensor)


class LocalWorldEstimator:
    """
    Strictly local world belief maintained onboard each individual drone.
    Handles:
      1. Distributed Kalman Filter (DKF) prediction, local sensor update, and covariance-weighted
         consensus fusion over 1-hop neighbor messages, including Negative Information (Ghost Clearing).
      2. Decentralized Search Pheromone / Certainty Grid (values in [0, 1], where 1 = recently scanned,
         0 = high uncertainty / needs exploration).
    """

    def __init__(self, drone_id: int, cfg: SimulationConfig):
        self.drone_id = drone_id
        self.cfg = cfg
        self.tracks: Dict[int, TargetTrack] = {}
        # Remember recently captured or invalidated target IDs -> timestamp / age so stale
        # neighbor gossip cannot resurrect a ghost track!
        self.cleared_targets: Dict[int, float] = {}
        # Ostrom Principles 4 & 5: Mutual Monitoring peer trust ledger T_ij in [0.05, 1.0]
        self.peer_trust: Dict[int, float] = {}
        self.sanctions_applied: int = 0
        self.certainty_grid: np.ndarray = np.zeros(
            (cfg.grid_rows, cfg.grid_cols), dtype=np.float32
        )
        cell_w = cfg.world_width / cfg.grid_cols
        cell_h = cfg.world_height / cfg.grid_rows
        xs = (np.arange(cfg.grid_cols, dtype=np.float32) + 0.5) * cell_w
        ys = (np.arange(cfg.grid_rows, dtype=np.float32) + 0.5) * cell_h
        self._gx, self._gy = np.meshgrid(xs, ys)

    def step_prediction_and_decay(self, own_pos: np.ndarray, skip_grid: bool = False) -> None:
        """Advance target tracks via damped velocity model and decay search grid."""
        dt = self.cfg.dt

        if not skip_grid:
            self.certainty_grid *= self.cfg.pheromone_decay
            dist_sq = (self._gx - float(own_pos[0])) ** 2 + (self._gy - float(own_pos[1])) ** 2
            sensed_mask = dist_sq <= (self.cfg.sensor_range * 0.9) ** 2
            self.certainty_grid[sensed_mask] = 1.0

        # Pitt Forgiveness / Rehabilitation: slowly recover peer trust toward 1.0 over time
        for pid in list(self.peer_trust.keys()):
            self.peer_trust[pid] = min(1.0, self.peer_trust[pid] + 0.015 * dt)

        # Age out old cleared tombstones after 6 seconds
        expired_tombstones = []
        for tid in list(self.cleared_targets.keys()):
            self.cleared_targets[tid] += dt
            if self.cleared_targets[tid] > 6.0:
                expired_tombstones.append(tid)
        for tid in expired_tombstones:
            del self.cleared_targets[tid]

        # Predict existing target tracks forward in time
        stale_ids = []
        for tid, trk in self.tracks.items():
            if trk.captured:
                stale_ids.append(tid)
                continue
            # Reset direct-observation hop flag at start of step: unless directly sensed
            # on this tick, a track is at least 1 hop / historical
            trk.hops = max(1, trk.hops)
            # Damped velocity prediction (prevents runaway dead-reckoning when target turns)
            trk.vel = trk.vel * (1.0 - 0.8 * dt)
            trk.pos = trk.pos + trk.vel * dt
            trk.pos[0] = min(max(float(trk.pos[0]), 15.0), self.cfg.world_width - 15.0)
            trk.pos[1] = min(max(float(trk.pos[1]), 15.0), self.cfg.world_height - 15.0)
            trk.variance = min(trk.variance + 12.0 * dt, 64.0)
            trk.age += dt
            if trk.age > self.cfg.target_memory_timeout:
                stale_ids.append(tid)

        for tid in stale_ids:
            del self.tracks[tid]

    def update_from_local_sensors(
        self,
        own_pos: np.ndarray,
        observed_targets: List[Tuple[int, np.ndarray, np.ndarray, bool]],
        rng: np.random.Generator,
    ) -> None:
        """
        Update local tracks using direct onboard sensor measurements AND Negative Information:
        If our predicted track for Target `tid` lies inside our clear sensor footprint
        (`dist < 0.72 * R_sense`) yet our sensor did NOT detect `tid`, the track is a phantom
        ghost! Immediately invalidate/clear it so drones never orbit empty space.
        """
        meas_var = 4.0
        directly_seen_ids = set()
        self._last_own_pos = own_pos.copy()

        for tid, t_pos, t_vel, is_captured in observed_targets:
            directly_seen_ids.add(tid)
            if is_captured:
                self.cleared_targets[tid] = 0.0
                if tid in self.tracks:
                    del self.tracks[tid]
                continue

            # Target is actively observed alive -> remove any old tombstone
            self.cleared_targets.pop(tid, None)

            noisy_pos = t_pos + rng.normal(0.0, 1.2, size=2)
            noisy_vel = t_vel + rng.normal(0.0, 2.0, size=2)

            if tid not in self.tracks:
                self.tracks[tid] = TargetTrack(
                    target_id=tid,
                    pos=noisy_pos,
                    vel=noisy_vel,
                    variance=meas_var,
                    hops=0,
                    age=0.0,
                    captured=False,
                )
            else:
                trk = self.tracks[tid]
                k_gain = trk.variance / (trk.variance + meas_var)
                trk.pos = trk.pos + k_gain * (noisy_pos - trk.pos)
                trk.vel = trk.vel + k_gain * (noisy_vel - trk.vel)
                trk.variance = max(meas_var, (1.0 - k_gain) * trk.variance)
                trk.hops = 0
                trk.age = 0.0
                trk.captured = False
                trk.source_id = -1

        # Negative Information (Ghost Track Clearing):
        # If ANY track's estimated position (from our prediction OR just received via gossip)
        # lies inside our clear sensor footprint (`< 0.78 * R_sense`) and we did NOT directly
        # detect `tid` this tick, the target is NOT at `trk.pos`!
        self._directly_seen_this_step = directly_seen_ids
        self.clear_ghost_tracks_in_fov(own_pos)

    def clear_ghost_tracks_in_fov(self, own_pos: np.ndarray) -> None:
        """Delete any non-directly-sensed track whose estimated position lies inside our sensor FOV."""
        directly_seen_ids = getattr(self, "_directly_seen_this_step", set())
        ghost_clear_radius_sq = (self.cfg.sensor_range * 0.78) ** 2
        ghost_ids = []
        for tid, trk in self.tracks.items():
            if tid not in directly_seen_ids:
                dx = float(trk.pos[0] - own_pos[0])
                dy = float(trk.pos[1] - own_pos[1])
                if (dx * dx + dy * dy) < ghost_clear_radius_sq:
                    ghost_ids.append(tid)

        for tid in ghost_ids:
            src = self.tracks[tid].source_id
            if src >= 0:
                # Ostrom Mutual Monitoring: neighbor `src` reported a phantom target inside our
                # clear sensor FOV where no target exists -> apply graduated trust sanction!
                cur_t = self.peer_trust.get(src, 1.0)
                self.peer_trust[src] = max(0.05, cur_t - 0.30)
                self.sanctions_applied += 1
            del self.tracks[tid]
            self.cleared_targets[tid] = 4.5

    def fuse_neighbor_gossip(self, inbox: List[GossipPacket], skip_grid: bool = False) -> None:
        """
        Fuse neighbor target tracks using True Covariance Intersection (preventing artificial
        variance collapse on rumor loops) and propagate negative-information tombstones.
        """
        if not inbox:
            return

        if not skip_grid:
            alpha = np.float32(self.cfg.gossip_merge_rate)
            valid_grids = [
                pkt.search_grid_summary
                for pkt in inbox
                if pkt.search_grid_summary.shape == self.certainty_grid.shape
            ]
            if valid_grids:
                max_nbr_grid = np.maximum.reduce(valid_grids)
                diff = np.maximum(0.0, max_nbr_grid - self.certainty_grid)
                self.certainty_grid += alpha * diff

        for pkt in inbox:
            for tid, n_trk in pkt.target_tracks.items():
                # 1. Propagate tombstones / captured / ghost-cleared notifications
                if n_trk.get("captured", False):
                    tomb_age = float(n_trk.get("age", 0.0))
                    if tid in self.tracks and self.tracks[tid].age > 0.15:
                        del self.tracks[tid]
                    if tid not in self.cleared_targets:
                        self.cleared_targets[tid] = max(tomb_age, 4.5)
                    continue

                n_hops = int(n_trk["hops"]) + 1
                if n_hops > self.cfg.max_comm_hops:
                    continue

                # Each wireless hop takes 1 simulation tick (dt), so the true age of a gossiped
                # track is n_trk["age"] + dt! Without adding dt, a rumor bouncing between two
                # neighbors would keep age == 0.0 forever!
                n_age = float(n_trk["age"]) + self.cfg.dt
                if n_age > self.cfg.target_memory_timeout:
                    continue

                # Ignore stale rumor if we recently verified this target is gone/ghosted,
                # UNLESS the neighbor directly saw the target on the immediately preceding tick (n_hops == 1 and n_trk["age"] == 0)
                if tid in self.cleared_targets and not (n_hops == 1 and float(n_trk["age"]) < 1e-4):
                    continue

                sender_id = pkt.sender_id
                trust = self.peer_trust.get(sender_id, 1.0)
                n_pos = np.array(n_trk["pos"], dtype=np.float64)
                n_vel = np.array(n_trk["vel"], dtype=np.float64)

                # If we have a direct onboard sensor lock (hops == 0), use it to monitor/verify neighbor's report!
                if tid in self.tracks and self.tracks[tid].hops == 0 and self.tracks[tid].age < 1e-4:
                    if n_hops == 1 and n_age <= self.cfg.dt * 1.5:
                        discrepancy = float(np.linalg.norm(self.tracks[tid].pos - n_pos))
                        if discrepancy <= 35.0:
                            # Corroborated report -> boost social trust
                            self.peer_trust[sender_id] = min(1.0, trust + 0.06)
                        elif discrepancy > 85.0:
                            # Contradicts direct sensor truth -> graduated sanction
                            self.peer_trust[sender_id] = max(0.05, trust - 0.30)
                            self.sanctions_applied += 1
                            continue
                    continue

                # Level 3 Graduated Sanction (Quarantine): reject uncorroborated tracks from untrusted peers
                if trust < 0.20:
                    continue

                # Level 1 Graduated Sanction: inflate neighbor variance by 1 / trust^2
                n_var = min((float(n_trk["var"]) + 3.5) / max(trust * trust, 0.04), 64.0)

                if tid not in self.tracks:
                    self.tracks[tid] = TargetTrack(
                        target_id=tid,
                        pos=n_pos,
                        vel=n_vel,
                        variance=n_var,
                        hops=n_hops,
                        age=n_age,
                        captured=False,
                        source_id=sender_id,
                    )
                else:
                    own = self.tracks[tid]
                    # Only fuse if neighbor's track is strictly fresher, or same age with no more hops.
                    # Under Covariance Intersection, fusing neighbor information never inflates variance
                    # above min(own.variance, n_var) when same age, or n_var when refreshing a stale track.
                    if n_age < own.age - 1e-4:
                        inv_own = 1.0 / max(own.variance, 1.0)
                        inv_nbr = 1.0 / max(n_var, 1.0)
                        omega = 0.25 * (inv_own / (inv_own + inv_nbr))
                        own.pos = omega * own.pos + (1.0 - omega) * n_pos
                        own.vel = omega * own.vel + (1.0 - omega) * n_vel
                        own.variance = n_var
                        own.age = n_age
                        own.hops = n_hops
                        own.source_id = sender_id
                    elif abs(n_age - own.age) <= 1e-4 and n_hops <= own.hops:
                        inv_own = 1.0 / max(own.variance, 1.0)
                        inv_nbr = 1.0 / max(n_var, 1.0)
                        omega = inv_own / (inv_own + inv_nbr)
                        own.pos = omega * own.pos + (1.0 - omega) * n_pos
                        own.vel = omega * own.vel + (1.0 - omega) * n_vel
                        own.variance = min(own.variance, n_var)
                        own.hops = min(own.hops, n_hops)

        # Final Negative-Information Check after gossip fusion:
        # Never allow a neighbor's gossip packet to plant a phantom target track inside our own
        # empty sensor circle!
        if hasattr(self, "_last_own_pos"):
            self.clear_ghost_tracks_in_fov(self._last_own_pos)

    def export_gossip_tracks(self) -> Dict[int, Dict[str, Any]]:
        """Serialize local target tracks AND negative-information tombstones for 1-hop neighbors."""
        out: Dict[int, Dict[str, Any]] = {}
        for tid, trk in self.tracks.items():
            out[tid] = {
                "pos": [float(trk.pos[0]), float(trk.pos[1])],
                "vel": [float(trk.vel[0]), float(trk.vel[1])],
                "var": float(trk.variance),
                "hops": int(trk.hops),
                "age": float(trk.age),
                "captured": bool(trk.captured),
            }
        for tid, tomb_age in self.cleared_targets.items():
            if tid not in out:
                out[tid] = {
                    "pos": [0.0, 0.0],
                    "vel": [0.0, 0.0],
                    "var": 999.0,
                    "hops": 1,
                    "age": float(tomb_age),
                    "captured": True,
                }
        return out
