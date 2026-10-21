/**
 * Real-Time Tactical Canvas Visualizer & WebSocket Client
 * for the Decentralized AI Drone Swarm Simulator.
 */

const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");

let ws = null;
let latestState = null;
let activeClickTool = "inspect";
let selectedDroneId = 0;

// UI Layer Toggles
const layers = {
  showLinks: true,
  showHeatmap: true,
  showSensors: false,
  showVectors: true,
};

const ROLE_COLORS = {
  SEARCHING: "#38bdf8",
  PURSUING: "#facc15",
  ENCIRCLING: "#a855f7",
  INTERCEPTING: "#10b981",
  OFFLINE: "#475569",
};

function connectWebSocket() {
  const proto = window.location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${window.location.host}/ws`);

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.event === "marl_trained") {
      const st = document.getElementById("marl-status");
      st.style.display = "block";
      st.textContent = `ES Policy Optimized! Return: ${data.result.mean_return_history.join(" → ")}`;
      document.getElementById("sel-mode").value = "marl";
      return;
    }
    latestState = data;
    updateTelemetryUI(data);
    renderFrame(data);
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
  document.getElementById("m-time").textContent = `${state.sim_time.toFixed(1)}s`;
  document.getElementById("m-drones").textContent = `${m.active_drones} / ${m.total_drones}`;
  document.getElementById("m-targets").textContent = `${m.captured_targets} / ${m.total_targets}`;
  document.getElementById("m-lambda").textContent = m.algebraic_connectivity.toFixed(2);
  document.getElementById("m-err").textContent = `${m.mean_est_error.toFixed(1)} px`;
  document.getElementById("m-cov").textContent = `${m.coverage_pct.toFixed(1)}%`;
  document.getElementById("m-sep").textContent = `${m.min_separation.toFixed(1)} px`;

  const pill = document.getElementById("status-pill");
  const pauseBtn = document.getElementById("btn-pause");
  if (state.paused) {
    pill.textContent = "PAUSED";
    pill.style.background = "#78350f";
    pill.style.color = "#fcd34d";
    pauseBtn.textContent = "Resume";
  } else {
    pill.textContent = "RUNNING";
    pill.style.background = "#065f46";
    pill.style.color = "#6ee7b7";
    pauseBtn.textContent = "Pause";
  }

  // Inspector update
  const insp = state.inspector;
  if (insp) {
    selectedDroneId = insp.drone_id;
    document.getElementById("insp-id").textContent = `Drone #${insp.drone_id}`;
    const roleEl = document.getElementById("insp-role");
    roleEl.textContent = insp.role;
    roleEl.style.color = ROLE_COLORS[insp.role] || "#fff";

    document.getElementById("insp-target").textContent =
      insp.assigned_target_id >= 0 ? `Target #${insp.assigned_target_id}` : "Unassigned (Exploring)";

    const cbfEl = document.getElementById("insp-cbf");
    if (insp.cbf_intervened) {
      cbfEl.textContent = "ACTIVE SHIELDING";
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
        "<strong>Local CBBA Marginal Utility Bids:</strong><br/>" +
        bidEntries.map(([tid, val]) => `Target #${tid}: <span style="color:#38bdf8">${val}</span>`).join(" | ");
    } else {
      bidsEl.textContent = "No active target tracks in local memory — executing stigmergic frontier search.";
    }
  }
}

function renderFrame(state) {
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  // 1. Draw Selected Drone's Private Search Certainty Heatmap
  if (layers.showHeatmap && state.inspector && state.inspector.certainty_grid) {
    const grid = state.inspector.certainty_grid;
    const rows = grid.length;
    const cols = grid[0].length;
    const cw = W / cols;
    const ch = H / rows;

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const cert = grid[r][c];
        // Highlight explored certainty in subtle cyan/blue tint, unexplored in dark
        if (cert > 0.05) {
          ctx.fillStyle = `rgba(6, 182, 212, ${cert * 0.14})`;
          ctx.fillRect(c * cw, r * ch, cw + 0.5, ch + 0.5);
        }
      }
    }
  }

  // 2. Subtle tactical background coordinate grid
  ctx.strokeStyle = "rgba(30, 43, 69, 0.45)";
  ctx.lineWidth = 1;
  for (let x = 0; x <= W; x += 100) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
  }
  for (let y = 0; y <= H; y += 100) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(W, y);
    ctx.stroke();
  }

  // 3. Draw Wireless Ad-Hoc Mesh Communication Links
  if (layers.showLinks && state.links) {
    for (const [i, j, qual] of state.links) {
      const d1 = state.drones[i];
      const d2 = state.drones[j];
      if (!d1 || !d2) continue;
      const isSelectedLink = i === selectedDroneId || j === selectedDroneId;
      ctx.strokeStyle = isSelectedLink
        ? `rgba(56, 189, 248, ${0.45 + 0.5 * qual})`
        : `rgba(148, 163, 184, ${0.12 + 0.25 * qual})`;
      ctx.lineWidth = isSelectedLink ? 1.8 : 1.0;
      ctx.beginPath();
      ctx.moveTo(d1.x, d1.y);
      ctx.lineTo(d2.x, d2.y);
      ctx.stroke();
    }
  }

  // 4. Draw Circular Obstacles
  for (const obs of state.obstacles) {
    const grad = ctx.createRadialGradient(obs.x - obs.r * 0.2, obs.y - obs.r * 0.2, 4, obs.x, obs.y, obs.r);
    grad.addColorStop(0, "#334155");
    grad.addColorStop(1, "#0f172a");
    ctx.fillStyle = grad;
    ctx.strokeStyle = "#475569";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(obs.x, obs.y, obs.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }

  // 5. Draw Selected Drone's Private DKF Target Tracks & Covariance Ellipses
  if (state.inspector) {
    const selDrone = state.drones[state.inspector.drone_id];
    if (selDrone && selDrone.active) {
      // Sensor range & Comm range rings for the inspected drone
      ctx.strokeStyle = "rgba(56, 189, 248, 0.32)";
      ctx.setLineDash([5, 5]);
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(selDrone.x, selDrone.y, state.config.sensor_range, 0, Math.PI * 2);
      ctx.stroke();

      ctx.strokeStyle = "rgba(139, 92, 246, 0.22)";
      ctx.beginPath();
      ctx.arc(selDrone.x, selDrone.y, state.config.comm_range, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Planned tactical waypoint line
      const wp = state.inspector.planned_waypoint;
      if (wp) {
        ctx.strokeStyle = "rgba(16, 185, 129, 0.55)";
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(selDrone.x, selDrone.y);
        ctx.lineTo(wp[0], wp[1]);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Private DKF target beliefs
      for (const trk of state.inspector.private_tracks) {
        const [tx, ty] = trk.pos;
        ctx.strokeStyle = "rgba(250, 204, 21, 0.75)";
        ctx.fillStyle = "rgba(250, 204, 21, 0.12)";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(tx, ty, trk.std_radius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        // Crosshair at estimated position
        ctx.beginPath();
        ctx.moveTo(tx - 6, ty);
        ctx.lineTo(tx + 6, ty);
        ctx.moveTo(tx, ty - 6);
        ctx.lineTo(tx, ty + 6);
        ctx.stroke();

        ctx.fillStyle = "#fde047";
        ctx.font = "10px monospace";
        ctx.fillText(`Est T#${trk.target_id} (${trk.hops}h)`, tx + 8, ty - 8);
      }
    }
  }

  // 6. Draw Evasive Targets & Encirclement Rings
  for (const t of state.targets) {
    // Motion trail
    if (t.trail && t.trail.length > 1) {
      ctx.strokeStyle = t.captured ? "rgba(16, 185, 129, 0.2)" : "rgba(239, 68, 68, 0.35)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(t.trail[0][0], t.trail[0][1]);
      for (let k = 1; k < t.trail.length; k++) {
        ctx.lineTo(t.trail[k][0], t.trail[k][1]);
      }
      ctx.stroke();
    }

    // Desired encirclement ring
    ctx.strokeStyle = t.captured ? "rgba(16, 185, 129, 0.45)" : "rgba(239, 68, 68, 0.28)";
    ctx.setLineDash([4, 4]);
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(t.x, t.y, state.config.desired_encircle_radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Capture progress arc
    if (t.progress > 0 && !t.captured) {
      ctx.strokeStyle = "#10b981";
      ctx.lineWidth = 3.5;
      ctx.beginPath();
      ctx.arc(t.x, t.y, 16, -Math.PI / 2, -Math.PI / 2 + t.progress * Math.PI * 2);
      ctx.stroke();
    }

    // Target body
    ctx.fillStyle = t.captured ? "#10b981" : "#ef4444";
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(t.x, t.y, 9, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 11px sans-serif";
    ctx.fillText(t.captured ? `T#${t.id} [CAPTURED]` : `T#${t.id}`, t.x + 12, t.y + 4);
  }

  // 7. Draw Autonomous Swarm Drones
  for (const d of state.drones) {
    const color = ROLE_COLORS[d.role] || "#38bdf8";

    if (layers.showSensors && d.active) {
      ctx.strokeStyle = "rgba(56, 189, 248, 0.08)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(d.x, d.y, state.config.sensor_range, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Velocity vector
    if (layers.showVectors && d.active) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(d.x, d.y);
      ctx.lineTo(d.x + d.vx * 0.28, d.y + d.vy * 0.28);
      ctx.stroke();
    }

    // CBF Safety Shield intervention halo
    if (d.cbf && d.active) {
      ctx.strokeStyle = "rgba(245, 158, 11, 0.75)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(d.x, d.y, 11, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Highlight ring if this is the inspected drone
    if (d.id === selectedDroneId) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(d.x, d.y, 13, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Oriented directional triangle / quadrotor icon
    const angle = Math.atan2(d.vy, d.vx);
    ctx.save();
    ctx.translate(d.x, d.y);
    ctx.rotate(angle);
    ctx.fillStyle = color;
    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(8, 0);
    ctx.lineTo(-6, -5.5);
    ctx.lineTo(-3.5, 0);
    ctx.lineTo(-6, 5.5);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }
}

// --- UI Event Listeners ---
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

document.getElementById("sel-mode").addEventListener("change", (e) => {
  sendCmd({ cmd: "set_param", key: "controller_mode", value: e.target.value });
});

document.getElementById("sel-target-beh").addEventListener("change", (e) => {
  sendCmd({ cmd: "set_param", key: "target_behavior", value: e.target.value });
});

function bindSlider(sliderId, labelId, paramKey, suffix = "", scale = 1, transformDisplay = null) {
  const slider = document.getElementById(sliderId);
  const label = document.getElementById(labelId);
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

// Tool selector buttons
document.querySelectorAll(".tool-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tool-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    activeClickTool = btn.dataset.tool;
  });
});

// Canvas Click Interaction
canvas.addEventListener("click", (e) => {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
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
  } else if (activeClickTool === "spawn_target") {
    sendCmd({ cmd: "spawn_target", x, y });
  } else if (activeClickTool === "spawn_drone") {
    sendCmd({ cmd: "spawn_drone", x, y });
  } else if (activeClickTool === "add_obstacle") {
    sendCmd({ cmd: "add_obstacle", x, y, radius: 42 });
  } else if (activeClickTool === "emp_blast") {
    sendCmd({ cmd: "emp_blast", x, y, radius: 115 });
  }
});

connectWebSocket();
