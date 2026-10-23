"""FastAPI + WebSocket Real-Time Simulation Server for the Decentralized Drone Swarm."""

import asyncio
from pathlib import Path
from typing import Dict, Any, Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from swarm_sim.config import SimulationConfig
from swarm_sim.world import SwarmWorld2D
from swarm_sim.rl.marl_env import train_marl_policy_es

app = FastAPI(title="Decentralized AI Drone Swarm Simulator")

WEB_DIR = Path(__file__).parent / "web"

# Global simulation instance and playback state
cfg = SimulationConfig()
world = SwarmWorld2D(cfg)

sim_state: Dict[str, Any] = {
    "paused": False,
    "speed_multiplier": 1,
    "selected_drone_id": 0,
    "auto_respawn_targets": True,
}

connected_clients: Set[WebSocket] = set()


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


def handle_client_command(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Process interactive commands sent from the browser UI."""
    cmd = msg.get("cmd", "")

    if cmd == "pause":
        sim_state["paused"] = not sim_state["paused"]
    elif cmd == "step_once":
        world.step()
    elif cmd == "reset":
        world.reset()
    elif cmd == "select_drone":
        sim_state["selected_drone_id"] = int(msg.get("drone_id", 0))
    elif cmd == "set_param":
        key = msg.get("key")
        val = msg.get("value")
        if key == "controller_mode" and val in ("hybrid", "boids_pursuit", "marl"):
            world.cfg.controller_mode = val
        elif key == "mission_play" and val in ("weapons_free", "standoff_track", "wide_recon"):
            world.cfg.mission_play = val
        elif key == "enable_heterogeneous":
            world.cfg.enable_heterogeneous = bool(val)
        elif key == "emcon_radio_silence":
            world.cfg.emcon_radio_silence = bool(val)
        elif key == "target_behavior" and val in ("evasive", "patrol", "mixed"):
            world.cfg.target_behavior = val
        elif key == "sensor_range":
            world.cfg.sensor_range = float(val)
        elif key == "comm_range":
            world.cfg.comm_range = float(val)
        elif key == "packet_loss_prob":
            world.cfg.packet_loss_prob = float(val)
        elif key == "los_occlusion":
            world.cfg.los_occlusion = bool(val)
        elif key == "desired_encircle_radius":
            world.cfg.desired_encircle_radius = float(val)
        elif key == "num_drones":
            new_n = max(3, min(100, int(val)))
            world.cfg.num_drones = new_n
            world.reset()
        elif key == "num_targets":
            new_t = max(1, min(12, int(val)))
            world.cfg.num_targets = new_t
            world.reset()
        elif key == "speed_multiplier":
            sim_state["speed_multiplier"] = max(1, min(5, int(val)))
        elif key == "auto_respawn_targets":
            sim_state["auto_respawn_targets"] = bool(val)
    elif cmd == "spawn_target":
        x = float(msg.get("x", 500.0))
        y = float(msg.get("y", 350.0))
        world.target_mgr.spawn_at(x, y)
    elif cmd == "spawn_drone":
        x = float(msg.get("x", 100.0))
        y = float(msg.get("y", 350.0))
        new_id = world.add_drone(x, y)
        sim_state["selected_drone_id"] = new_id
    elif cmd == "add_obstacle":
        x = float(msg.get("x", 500.0))
        y = float(msg.get("y", 350.0))
        r = float(msg.get("radius", 42.0))
        world.add_obstacle(x, y, r)
    elif cmd == "add_jammer":
        x = float(msg.get("x", 500.0))
        y = float(msg.get("y", 350.0))
        r = float(msg.get("radius", 105.0))
        world.add_jammer(x, y, r)
    elif cmd == "add_nofly_zone":
        x = float(msg.get("x", 500.0))
        y = float(msg.get("y", 350.0))
        r = float(msg.get("radius", 58.0))
        world.add_nofly_zone(x, y, r)
    elif cmd == "clear_obstacles":
        world.clear_obstacles()
    elif cmd == "emp_blast":
        x = float(msg.get("x", 500.0))
        y = float(msg.get("y", 350.0))
        r = float(msg.get("radius", 115.0))
        world.emp_blast(x, y, r)
    elif cmd == "train_marl":
        res = train_marl_policy_es(world, generations=6, pop_size=6, rollout_steps=60)
        world.cfg.controller_mode = "marl"
        return {"event": "marl_trained", "result": res}

    return {}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    connected_clients.add(ws)
    try:
        while True:
            # Non-blocking check for incoming client messages
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                extra = handle_client_command(msg)
                if extra:
                    await ws.send_json(extra)
            except asyncio.TimeoutError:
                pass

            if not sim_state["paused"]:
                for _ in range(sim_state["speed_multiplier"]):
                    world.step()
                    # If all targets captured and auto_respawn_targets is True, spawn a new target
                    if (
                        sim_state["auto_respawn_targets"]
                        and world.target_mgr.targets
                        and all(t.captured for t in world.target_mgr.targets)
                    ):
                        world.target_mgr.reset(world.obstacles)

            snapshot = world.get_state_snapshot(
                selected_drone_id=sim_state["selected_drone_id"]
            )
            snapshot["paused"] = sim_state["paused"]
            snapshot["speed_multiplier"] = sim_state["speed_multiplier"]
            await ws.send_json(snapshot)
            await asyncio.sleep(0.032)  # ~30 FPS broadcast loop
    except WebSocketDisconnect:
        pass
    finally:
        connected_clients.discard(ws)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("swarm_sim.server:app", host="127.0.0.1", port=8000, reload=False)
