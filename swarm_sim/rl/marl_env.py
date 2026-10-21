"""
PettingZoo-compatible Parallel MARL Environment & Evolutionary/Policy Trainer
for the 2D Decentralized Drone Swarm.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from swarm_sim.config import SimulationConfig
from swarm_sim.world import SwarmWorld2D
from swarm_sim.physics import clip_vector_norm


class DecentralizedSwarmParallelEnv:
    """
    Multi-Agent Reinforcement Learning Parallel Environment following the PettingZoo
    ParallelEnv API (`reset()`, `step(actions)`, `agents`, `observation_space`, `action_space`).

    Each agent receives strictly local 18-dimensional observations:
      - [0:2] normalized ego position (x/W, y/H)
      - [2:4] normalized ego velocity (vx/Vmax, vy/Vmax)
      - [4:14] 10-D ego tactical feature vector (relative target track, vortex tangent,
               neighbor dispersion, frontier uncertainty gradient)
      - [14:16] nearest peer relative vector
      - [16:18] nearest obstacle relative surface vector

    Actions:
      - 2D desired velocity in [-1, 1]^2 (scaled by max_speed and filtered by CBF safety shield).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "name": "decentralized_drone_swarm_v0"}

    def __init__(self, cfg: Optional[SimulationConfig] = None, max_steps: int = 400, seed: int = 42):
        self.cfg = cfg or SimulationConfig()
        self.max_steps = max_steps
        self.world = SwarmWorld2D(self.cfg, seed=seed)
        self.possible_agents: List[str] = [f"drone_{i}" for i in range(self.cfg.num_drones)]
        self.agents: List[str] = list(self.possible_agents)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
        if seed is not None:
            self.world.rng = np.random.default_rng(seed)
        self.world.reset()
        self.agents = [
            f"drone_{i}" for i in range(len(self.world.agents)) if self.world.active_mask[i]
        ]
        obs = self._get_observations()
        infos = {a: {} for a in self.agents}
        return obs, infos

    def _get_observations(self) -> Dict[str, np.ndarray]:
        obs_dict: Dict[str, np.ndarray] = {}
        for i, agent_obj in enumerate(self.world.agents):
            name = f"drone_{i}"
            if not self.world.active_mask[i]:
                continue
            own_pos = self.world.positions[i]
            own_vel = self.world.velocities[i]
            sensed_targets, sensed_peers, sensed_obstacles = self.world._gather_local_sensors(i)

            tid = agent_obj.allocator.assigned_target_id
            trk = agent_obj.estimator.tracks.get(tid) if tid != -1 else None

            ego_feat, _ = self.world.marl_policy.build_ego_features(
                own_pos=own_pos,
                own_vel=own_vel,
                target_track=trk,
                sensed_peers=sensed_peers,
                certainty_grid=agent_obj.estimator.certainty_grid,
                cfg=self.cfg,
                drone_id=i,
                prev_waypoint=agent_obj.planned_waypoint,
                sensed_obstacles=sensed_obstacles,
            )

            near_peer = np.zeros(2, dtype=np.float32)
            if sensed_peers:
                closest_p = min(
                    sensed_peers, key=lambda p: float(np.linalg.norm(p[1] - own_pos))
                )
                near_peer = ((closest_p[1] - own_pos) / max(self.cfg.sensor_range, 1.0)).astype(
                    np.float32
                )

            near_obs = np.zeros(2, dtype=np.float32)
            if sensed_obstacles:
                closest_o = min(
                    sensed_obstacles,
                    key=lambda o: float(np.linalg.norm(np.array([o.x, o.y]) - own_pos)) - o.radius,
                )
                c = np.array([closest_o.x, closest_o.y])
                near_obs = ((c - own_pos) / max(self.cfg.sensor_range, 1.0)).astype(np.float32)

            vec = np.concatenate(
                [
                    [own_pos[0] / self.cfg.world_width, own_pos[1] / self.cfg.world_height],
                    own_vel / max(self.cfg.max_speed, 1.0),
                    ego_feat,
                    near_peer,
                    near_obs,
                ]
            ).astype(np.float32)
            obs_dict[name] = vec

        return obs_dict

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        """Step the environment using external MARL actions filtered through the CBF safety shield."""
        n = len(self.world.agents)
        prev_captured = sum(1 for t in self.world.target_mgr.targets if t.captured)
        prev_cols = self.world.total_collisions

        # 1. Update wireless mesh & exchange packets
        self.world.network.update_topology(
            self.world.positions, self.world.active_mask, self.world.obstacles
        )
        outbox = {
            i: self.world.agents[i].prepare_broadcast_packet(
                self.world.positions[i], self.world.velocities[i]
            )
            for i in range(n)
            if self.world.active_mask[i]
        }
        inboxes = self.world.network.exchange_packets(outbox, self.world.active_mask)

        accel_cmds = np.zeros_like(self.world.positions)
        for i in range(n):
            if not self.world.active_mask[i]:
                continue
            ag = self.world.agents[i]
            own_pos = self.world.positions[i].copy()
            own_vel = self.world.velocities[i].copy()
            sensed_targets, sensed_peers, sensed_obstacles = self.world._gather_local_sensors(i)

            # Layer 1: Local estimation & gossip fusion still runs onboard
            ag.estimator.step_prediction_and_decay(own_pos)
            ag.estimator.update_from_local_sensors(own_pos, sensed_targets, self.world.rng)
            ag.estimator.fuse_neighbor_gossip(inboxes[i])
            ag.allocator.select_target(own_pos, own_vel, ag.estimator.tracks, inboxes[i])

            # Layer 2: External RL action
            act_name = f"drone_{i}"
            raw_act = np.asarray(actions.get(act_name, np.zeros(2)), dtype=np.float64)
            v_des = clip_vector_norm(raw_act * self.cfg.max_speed, self.cfg.max_speed)
            ag.desired_vel = v_des

            # Layer 3: CBF safety filter
            ag.safe_vel = ag._apply_cbf_safety_shield(
                own_pos, own_vel, v_des, sensed_peers, sensed_obstacles, inboxes[i]
            )
            u_cmd = 4.2 * (ag.safe_vel - own_vel) + self.cfg.drag_coeff * own_vel
            accel_cmds[i] = clip_vector_norm(u_cmd, self.cfg.max_accel)

        from swarm_sim.physics import step_double_integrator

        self.world.positions, self.world.velocities, step_cols = step_double_integrator(
            positions=self.world.positions,
            velocities=self.world.velocities,
            accel_cmds=accel_cmds,
            active_mask=self.world.active_mask,
            obstacles=self.world.obstacles,
            entity_radius=self.cfg.drone_radius,
            max_speed=self.cfg.max_speed,
            max_accel=self.cfg.max_accel,
            drag_coeff=self.cfg.drag_coeff,
            cfg=self.cfg,
        )
        self.world.total_collisions += step_cols
        self.world.target_mgr.step(
            sim_time=self.world.sim_time,
            drone_positions=self.world.positions,
            drone_active=self.world.active_mask,
            obstacles=self.world.obstacles,
        )
        self.world.sim_time += self.cfg.dt
        self.world.step_count += 1

        # Compute cooperative + local rewards
        new_captured = sum(1 for t in self.world.target_mgr.targets if t.captured)
        capture_bonus = 25.0 * (new_captured - prev_captured)
        col_penalty = -5.0 if (self.world.total_collisions > prev_cols) else 0.0

        rewards: Dict[str, float] = {}
        terminations: Dict[str, bool] = {}
        truncations: Dict[str, bool] = {}
        infos: Dict[str, Dict[str, Any]] = {}

        all_captured = all(t.captured for t in self.world.target_mgr.targets)
        is_truncated = self.world.step_count >= self.max_steps

        for i in range(n):
            if not self.world.active_mask[i]:
                continue
            name = f"drone_{i}"
            ag = self.world.agents[i]
            r = capture_bonus + col_penalty
            # Shaping reward for approaching encirclement ring of assigned target
            tid = ag.allocator.assigned_target_id
            if tid in ag.estimator.tracks:
                d = float(np.linalg.norm(ag.estimator.tracks[tid].pos - self.world.positions[i]))
                ring_err = abs(d - self.cfg.desired_encircle_radius)
                r += 0.2 * np.exp(-ring_err / 60.0)
            if ag.cbf_intervened:
                r -= 0.05

            rewards[name] = float(r)
            terminations[name] = bool(all_captured)
            truncations[name] = bool(is_truncated)
            infos[name] = {"role": ag.role, "assigned_target": tid}

        if all_captured or is_truncated:
            self.agents = []

        obs = self._get_observations()
        return obs, rewards, terminations, truncations, infos


def train_marl_policy_es(
    world: SwarmWorld2D, generations: int = 8, pop_size: int = 6, rollout_steps: int = 80
) -> Dict[str, Any]:
    """
    Fast OpenAI-ES (Evolution Strategies) optimizer that fine-tunes the decentralized
    neural policy in-place and returns training telemetry.
    """
    base_params = world.marl_policy.get_flat_params().copy()
    sigma = 0.06
    lr = 0.035
    rng = world.rng
    history = []

    eval_cfg = SimulationConfig(
        num_drones=12,
        num_targets=2,
        controller_mode="marl",
    )

    for gen in range(generations):
        noise = rng.normal(0.0, 1.0, size=(pop_size, len(base_params)))
        returns = np.zeros(pop_size)

        for k in range(pop_size):
            cand_params = base_params + sigma * noise[k]
            sim = SwarmWorld2D(eval_cfg, seed=100 + gen * 10 + k)
            sim.marl_policy.set_flat_params(cand_params)

            ep_return = 0.0
            for _ in range(rollout_steps):
                sim.step()
                # Cooperative encirclement & capture progress score
                prog = sum(t.capture_progress + (1.5 if t.captured else 0.0) for t in sim.target_mgr.targets)
                ep_return += prog - 0.1 * sim.total_collisions
            returns[k] = ep_return

        std_r = float(np.std(returns))
        if std_r > 1e-6:
            norm_r = (returns - np.mean(returns)) / std_r
            grad = (noise.T @ norm_r) / (pop_size * sigma)
            base_params += lr * grad

        history.append(round(float(np.mean(returns)), 2))

    world.marl_policy.set_flat_params(base_params)
    return {
        "generations": generations,
        "mean_return_history": history,
        "final_return": history[-1] if history else 0.0,
    }
