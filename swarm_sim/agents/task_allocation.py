"""Decentralized Multi-Target Coalition Formation & Task Allocation (CBBA / Marginal Utility)."""

from typing import Dict, List
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.communication import GossipPacket
from swarm_sim.agents.estimation import TargetTrack


class DecentralizedTaskAllocator:
    """
    Computes marginal utility bids for known targets and uses 1-hop neighbor
    assignments to self-balance coalition sizes across multiple targets without
    any central coordinator.
    """

    def __init__(self, drone_id: int, cfg: SimulationConfig):
        self.drone_id = drone_id
        self.cfg = cfg
        self.assigned_target_id: int = -1
        self.bids: Dict[int, float] = {}

    def select_target(
        self,
        own_pos: np.ndarray,
        own_vel: np.ndarray,
        tracks: Dict[int, TargetTrack],
        inbox: List[GossipPacket],
    ) -> int:
        """
        Evaluate utility for each active target track in local belief and select
        the highest-utility target. Incorporates diminishing marginal returns
        based on how many nearby peers are already committed to each target.
        """
        active_tracks = {
            tid: trk for tid, trk in tracks.items() if not trk.captured
        }
        if not active_tracks:
            self.assigned_target_id = -1
            self.bids = {}
            return -1

        # Count how many 1-hop neighbors are currently assigned to each target
        # and how many are closer to that target than this drone
        neighbor_counts: Dict[int, int] = {tid: 0 for tid in active_tracks}
        closer_peers: Dict[int, int] = {tid: 0 for tid in active_tracks}

        for tid, trk in active_tracks.items():
            tx = float(trk.pos[0])
            ty = float(trk.pos[1])
            my_dist = float(((own_pos[0] - tx) ** 2 + (own_pos[1] - ty) ** 2) ** 0.5)
            for pkt in inbox:
                peer_knows_tid = (pkt.assigned_target_id == tid) or (
                    pkt.assigned_target_id == -1
                    and tid in pkt.target_tracks
                    and not pkt.target_tracks[tid].get("captured", False)
                )
                if pkt.assigned_target_id == tid:
                    neighbor_counts[tid] += 1
                if peer_knows_tid:
                    peer_dist = float(((pkt.pos[0] - tx) ** 2 + (pkt.pos[1] - ty) ** 2) ** 0.5)
                    # Deterministic tie-breaker using sender_id vs drone_id prevents simultaneous swap oscillations
                    if peer_dist + 8.0 < my_dist or (
                        abs(peer_dist - my_dist) <= 8.0 and pkt.sender_id < self.drone_id
                    ):
                        closer_peers[tid] += 1

        best_tid = -1
        best_utility = -1e9
        new_bids: Dict[int, float] = {}

        ideal_coalition = max(self.cfg.min_drones_for_capture + 2, 5)
        speed = float((own_vel[0] ** 2 + own_vel[1] ** 2) ** 0.5)

        for tid, trk in active_tracks.items():
            rx = float(trk.pos[0] - own_pos[0])
            ry = float(trk.pos[1] - own_pos[1])
            dist = (rx * rx + ry * ry) ** 0.5
            # Time-to-go estimate
            t_go = dist / max(self.cfg.max_speed, 1.0)

            # Heading alignment bonus
            align_bonus = 0.0
            if speed > 5.0 and dist > 5.0:
                align_bonus = 12.0 * float((own_vel[0] * rx + own_vel[1] * ry) / (speed * dist))

            # Diminishing marginal return if coalition already has enough closer peers
            # Ideal ring size is ~6 drones per target. If more than 7 closer peers are already
            # swarming this target, surplus drones should decouple and patrol/search for other targets!
            over_crowding_penalty = 0.0
            if closer_peers[tid] >= ideal_coalition:
                over_crowding_penalty = 52.0 * (closer_peers[tid] - ideal_coalition + 1)
            elif neighbor_counts[tid] > ideal_coalition:
                over_crowding_penalty = 20.0 * (neighbor_counts[tid] - ideal_coalition)

            # Strong hysteresis bonus for sticking to current assignment
            hysteresis = 48.0 if tid == self.assigned_target_id else 0.0

            # Information confidence factor
            conf_penalty = 0.15 * trk.variance + 4.0 * trk.hops

            utility = (
                250.0
                - 32.0 * t_go
                + align_bonus
                + hysteresis
                - over_crowding_penalty
                - conf_penalty
            )
            new_bids[tid] = round(utility, 2)

            if utility > best_utility:
                best_utility = utility
                best_tid = tid

        # Surplus Coalition Cap with Hysteresis:
        # - If already committed to `best_tid`, stay committed unless utility drops below -15.0
        # - If currently SEARCHING (unassigned), only join `best_tid` if utility > +25.0
        drop_threshold = -15.0 if (best_tid == self.assigned_target_id) else 25.0
        if best_utility < drop_threshold:
            self.bids = new_bids
            self.assigned_target_id = -1
            return -1

        self.bids = new_bids
        self.assigned_target_id = best_tid
        return best_tid
