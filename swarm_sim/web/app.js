/**
 * Tactical C2 Visualizer & WebSocket Client
 * Connects to the Decentralized Swarm Simulator and drives UI & canvas updates.
 */

import { TacticalRenderer, ROLE_COLORS } from "./renderer.js";

const canvas = document.getElementById("simCanvas");
const renderer = new TacticalRenderer(canvas);

let ws = null;
let latestState = null;
let activeClickTool = "inspect";
let selectedDroneId = 0;

const layers = {
  showLinks: true,
  showHeatmap: true,
  showSensors: false,
  showVectors: true,
};

// Handle high-DPI canvas resizing
window.addEventListener("resize", () => {
  renderer.setupDPI();
  if (latestState) {
    renderer.render(latestState, layers, selectedDroneId);
  }
});

function connectWebSocket() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${window.location.host}/ws`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.event === "marl_trained") {
      const st = document.getElementById("marl-status");
      if (st) {
        st.style.display = "block";
        st.textContent = `ES Policy Optimized! Return: ${data.result.mean_return_history.join(" → ")}`;
      }
      const selMode = document.getElementById("sel-mode");
      if (selMode) selMode.value = "marl";
      return;
    }

    latestState = data;
    updateTelemetryUI(data);
    renderer.render(data, layers, selectedDroneId);
  };

  ws.onclose = () => {
    setTimeout(connectWebSocket, 1000);
  };
}

function sendCmd(payload) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(payload));
  }
}

function updateTelemetryUI(state) {
  const m = state.metrics;
  if (!m) return;

  document.getElementById("m-time").textContent = `${state.sim_time.toFixed(1)}s`;
  document.getElementById("m-drones").textContent = `${m.active_drones} / ${m.total_drones}`;
  document.getElementById("m-jammed").textContent = `${m.jammed_drones || 0}`;
  document.getElementById("m-targets").textContent = `${m.captured_targets} / ${m.total_targets}`;
  document.getElementById("m-lambda").textContent = m.algebraic_connectivity.toFixed(2);
  document.getElementById("m-err").textContent = `${m.mean_est_error.toFixed(1)} px`;
  document.getElementById("m-cov").textContent = `${m.coverage_pct.toFixed(1)}%`;
  document.getElementById("m-sep").textContent = `${m.min_separation.toFixed(1)} px`;

  const pill = document.getElementById("status-pill");
  const pauseBtn = document.getElementById("btn-pause");
  if (state.paused) {
    pill.textContent = "PAUSED";
    pill.className = "status-badge paused";
    pauseBtn.textContent = "Resume";
  } else {
    pill.textContent = "RUNNING";
    pill.className = "status-badge running";
    pauseBtn.textContent = "Pause";
  }

  const insp = state.inspector;
  if (insp) {
    selectedDroneId = insp.drone_id;
    document.getElementById("insp-id").textContent = `Drone #${insp.drone_id} [${insp.asset_type}]`;

    const roleEl = document.getElementById("insp-role");
    roleEl.textContent = insp.role;
    roleEl.style.color = ROLE_COLORS[insp.role] || "#38bdf8";

    const rfEl = document.getElementById("insp-rf");
    if (!insp.active) {
      rfEl.textContent = "OFFLINE (EMP)";
      rfEl.style.color = "#64748b";
    } else if (insp.jammed) {
      rfEl.textContent = "RF JAMMED (EW)";
      rfEl.style.color = "#f43f5e";
    } else if (state.config.emcon_radio_silence) {
      rfEl.textContent = "EMCON SILENT";
      rfEl.style.color = "#f59e0b";
    } else {
      rfEl.textContent = `ONLINE (R=${insp.effective_comm_range}px)`;
      rfEl.style.color = "#10b981";
    }

    const losStr = insp.los_rate_deg_s ? ` | dλ/dt: ${insp.los_rate_deg_s}°/s` : "";
    document.getElementById("insp-target").textContent =
      insp.assigned_target_id >= 0
        ? `Target #${insp.assigned_target_id}${losStr}`
        : "Unassigned (Sector Search)";

    const cbfEl = document.getElementById("insp-cbf");
    if (insp.cbf_intervened) {
      cbfEl.textContent = "SHIELDING ACTIVE";
      cbfEl.style.color = "#f59e0b";
    } else {
      cbfEl.textContent = "Nominal";
      cbfEl.style.color = "#10b981";
    }

    document.getElementById("insp-tracks").textContent = `${insp.private_tracks.length} active`;

    const bidsEl = document.getElementById("insp-bids");
    const bidEntries = Object.entries(insp.bids || {});
    if (bidEntries.length > 0) {
      bidsEl.innerHTML =
        "<div style='margin-bottom:4px;color:var(--text-main);font-weight:600;'>Local CBBA Task Bids:</div>" +
        bidEntries.map(([tid, val]) => `Target #${tid}: <span style="color:#38bdf8">${val}</span>`).join("<br/>");
    } else {
      bidsEl.textContent = "No active target tracks in local memory — executing Voronoi frontier recon.";
    }
  }
}

// --- Tab Switching Logic ---
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

    btn.classList.add("active");
    const targetId = btn.dataset.tab;
    const targetContent = document.getElementById(targetId);
    if (targetContent) {
      targetContent.classList.add("active");
    }
  });
});

// --- UI Command Listeners ---
document.getElementById("btn-pause").addEventListener("click", () => sendCmd({ cmd: "pause" }));
document.getElementById("btn-step").addEventListener("click", () => sendCmd({ cmd: "step_once" }));
document.getElementById("btn-reset").addEventListener("click", () => sendCmd({ cmd: "reset" }));
document.getElementById("btn-clear-obs").addEventListener("click", () => sendCmd({ cmd: "clear_obstacles" }));

document.getElementById("btn-train-marl").addEventListener("click", () => {
  const st = document.getElementById("marl-status");
  st.style.display = "block";
  st.textContent = "Running decentralized ES rollouts...";
  sendCmd({ cmd: "train_marl" });
});

document.querySelectorAll(".play-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".play-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    sendCmd({ cmd: "set_param", key: "mission_play", value: btn.dataset.play });
  });
});

document.getElementById("sel-mode").addEventListener("change", (e) => {
  sendCmd({ cmd: "set_param", key: "controller_mode", value: e.target.value });
});

document.getElementById("sel-target-beh").addEventListener("change", (e) => {
  sendCmd({ cmd: "set_param", key: "target_behavior", value: e.target.value });
});

function bindSlider(sliderId, labelId, paramKey, suffix = "", scale = 1, transformDisplay = null) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);
  if (!slider || !label) return;
  slider.addEventListener("input", () => {
    const v = parseFloat(slider.value);
    label.textContent = transformDisplay ? transformDisplay(v) : `${v * scale}${suffix}`;
    sendCmd({ cmd: "set_param", key: paramKey, value: v });
  });
}

bindSlider("rng-speed", "val-speed", "speed_multiplier", "x");
bindSlider("rng-ndrones", "val-ndrones", "num_drones", "");
bindSlider("rng-sense", "val-sense", "sensor_range", " px");
bindSlider("rng-comm", "val-comm", "comm_range", " px");
bindSlider("rng-loss", "val-loss", "packet_loss_prob", "%", 100, (v) => `${Math.round(v * 100)}%`);

document.getElementById("chk-hetero").addEventListener("change", (e) => {
  sendCmd({ cmd: "set_param", key: "enable_heterogeneous", value: e.target.checked });
});
document.getElementById("chk-emcon").addEventListener("change", (e) => {
  sendCmd({ cmd: "set_param", key: "emcon_radio_silence", value: e.target.checked });
});
document.getElementById("chk-los").addEventListener("change", (e) => {
  sendCmd({ cmd: "set_param", key: "los_occlusion", value: e.target.checked });
});
document.getElementById("chk-respawn").addEventListener("change", (e) => {
  sendCmd({ cmd: "set_param", key: "auto_respawn_targets", value: e.target.checked });
});
document.getElementById("chk-links").addEventListener("change", (e) => (layers.showLinks = e.target.checked));
document.getElementById("chk-heatmap").addEventListener("change", (e) => (layers.showHeatmap = e.target.checked));
document.getElementById("chk-sensors").addEventListener("change", (e) => (layers.showSensors = e.target.checked));
document.getElementById("chk-vectors").addEventListener("change", (e) => (layers.showVectors = e.target.checked));

// Floating Dock tool selection
document.querySelectorAll(".tool-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tool-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeClickTool = btn.dataset.tool;
  });
});

// Canvas Interaction (Click handling)
canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = 1000 / rect.width;
  const scaleY = 700 / rect.height;
  const x = (e.clientX - rect.left) * scaleX;
  const y = (e.clientY - rect.top) * scaleY;

  if (activeClickTool === "inspect" && latestState) {
    let bestId = 0;
    let bestDist = 1e9;
    for (const d of latestState.drones) {
      const dist = Math.hypot(d.x - x, d.y - y);
      if (dist < bestDist) {
        bestDist = dist;
        bestId = d.id;
      }
    }
    selectedDroneId = bestId;
    sendCmd({ cmd: "select_drone", drone_id: bestId });

    // Auto-switch to telemetry tab when user clicks a drone to inspect
    if (window.switchTab) {
      window.switchTab("tab-inspector");
    }
  } else if (activeClickTool === "spawn_target") {
    sendCmd({ cmd: "spawn_target", x, y });
  } else if (activeClickTool === "spawn_drone") {
    sendCmd({ cmd: "spawn_drone", x, y });
  } else if (activeClickTool === "add_obstacle") {
    sendCmd({ cmd: "add_obstacle", x, y, radius: 42 });
  } else if (activeClickTool === "add_jammer") {
    sendCmd({ cmd: "add_jammer", x, y, radius: 105 });
  } else if (activeClickTool === "add_nofly_zone") {
    sendCmd({ cmd: "add_nofly_zone", x, y, radius: 58 });
  } else if (activeClickTool === "emp_blast") {
    sendCmd({ cmd: "emp_blast", x, y, radius: 115 });
  }
});

connectWebSocket();
