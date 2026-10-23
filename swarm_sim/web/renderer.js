/**
 * Modular Canvas Renderer for the Decentralized Tactical Swarm C2 Visualizer.
 * Provides high-DPI support, tactical radar grid, EW effects, DKF tracks, and asset silhouettes.
 */

export const ROLE_COLORS = {
  SEARCHING: "#38bdf8",
  PURSUING: "#facc15",
  ENCIRCLING: "#a855f7",
  INTERCEPTING: "#10b981",
  COURIER_EXFIL: "#f97316",
  RELAYING: "#2dd4bf",
  OFFLINE: "#475569",
};

export class TacticalRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.baseWidth = 1000;
    this.baseHeight = 700;
    this.setupDPI();
  }

  setupDPI() {
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = this.baseWidth * dpr;
    this.canvas.height = this.baseHeight * dpr;
    this.ctx.resetTransform();
    this.ctx.scale(dpr, dpr);
  }

  render(state, layers, selectedDroneId) {
    const ctx = this.ctx;
    const W = this.baseWidth;
    const H = this.baseHeight;

    ctx.clearRect(0, 0, W, H);

    // 1. Private Search Certainty Heatmap
    if (layers.showHeatmap && state.inspector && state.inspector.certainty_grid) {
      const grid = state.inspector.certainty_grid;
      const rows = grid.length;
      const cols = grid[0].length;
      const cw = W / cols;
      const ch = H / rows;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const cert = grid[r][c];
          if (cert > 0.05) {
            ctx.fillStyle = `rgba(6, 182, 212, ${cert * 0.14})`;
            ctx.fillRect(c * cw, r * ch, cw + 0.5, ch + 0.5);
          }
        }
      }
    }

    // 2. Tactical Coordinate Grid
    ctx.strokeStyle = "rgba(28, 39, 60, 0.4)";
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

    // 3. Contested EW RF Jamming Zones
    if (state.jammers) {
      const pulse = 0.5 + 0.5 * Math.sin(state.sim_time * 4.0);
      for (const jam of state.jammers) {
        const grad = ctx.createRadialGradient(jam.x, jam.y, 8, jam.x, jam.y, jam.r);
        grad.addColorStop(0, `rgba(244, 63, 94, ${0.22 + 0.06 * pulse})`);
        grad.addColorStop(1, "rgba(244, 63, 94, 0.02)");
        ctx.fillStyle = grad;
        ctx.strokeStyle = "rgba(244, 63, 94, 0.55)";
        ctx.setLineDash([6, 4]);
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(jam.x, jam.y, jam.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);

        // Emitter Center Icon
        ctx.fillStyle = "#f43f5e";
        ctx.beginPath();
        ctx.arc(jam.x, jam.y, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = "rgba(254, 205, 211, 0.85)";
        ctx.font = "bold 9.5px monospace";
        ctx.fillText("EW RF JAMMER", jam.x - 34, jam.y - 10);
      }
    }

    // 4. Operator Keep-Out / No-Fly Zones (NFZ)
    if (state.nofly_zones) {
      for (const nfz of state.nofly_zones) {
        ctx.fillStyle = "rgba(245, 158, 11, 0.10)";
        ctx.strokeStyle = "rgba(245, 158, 11, 0.70)";
        ctx.setLineDash([8, 4]);
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.arc(nfz.x, nfz.y, nfz.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = "#fcd34d";
        ctx.font = "bold 9.5px monospace";
        ctx.fillText("NO-FLY ZONE (CBF)", nfz.x - 44, nfz.y + 4);
      }
    }

    // 5. Wireless Ad-Hoc Mesh Communication Links
    if (layers.showLinks && state.links) {
      for (const [i, j, qual] of state.links) {
        const d1 = state.drones[i];
        const d2 = state.drones[j];
        if (!d1 || !d2) continue;
        const isSelectedLink = i === selectedDroneId || j === selectedDroneId;
        ctx.strokeStyle = isSelectedLink
          ? `rgba(56, 189, 248, ${0.45 + 0.5 * qual})`
          : `rgba(148, 163, 184, ${0.1 + 0.22 * qual})`;
        ctx.lineWidth = isSelectedLink ? 1.8 : 1.0;
        ctx.beginPath();
        ctx.moveTo(d1.x, d1.y);
        ctx.lineTo(d2.x, d2.y);
        ctx.stroke();
      }
    }

    // 6. Physical Terrain Obstacles
    if (state.obstacles) {
      for (const obs of state.obstacles) {
        const grad = ctx.createRadialGradient(obs.x - obs.r * 0.2, obs.y - obs.r * 0.2, 4, obs.x, obs.y, obs.r);
        grad.addColorStop(0, "#283548");
        grad.addColorStop(1, "#0d1422");
        ctx.fillStyle = grad;
        ctx.strokeStyle = "#475569";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(obs.x, obs.y, obs.r, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
    }

    // 7. Selected Drone's Private DKF Target Tracks & Covariance Ellipses
    if (state.inspector) {
      const selDrone = state.drones[state.inspector.drone_id];
      if (selDrone && selDrone.active) {
        ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
        ctx.setLineDash([5, 5]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(selDrone.x, selDrone.y, state.inspector.effective_sensor_range, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = selDrone.jammed ? "rgba(244, 63, 94, 0.35)" : "rgba(139, 92, 246, 0.22)";
        ctx.beginPath();
        ctx.arc(selDrone.x, selDrone.y, state.inspector.effective_comm_range, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

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

        for (const trk of state.inspector.private_tracks) {
          const [tx, ty] = trk.pos;
          ctx.strokeStyle = "rgba(250, 204, 21, 0.75)";
          ctx.fillStyle = "rgba(250, 204, 21, 0.12)";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(tx, ty, trk.std_radius, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();

          ctx.beginPath();
          ctx.moveTo(tx - 6, ty);
          ctx.lineTo(tx + 6, ty);
          ctx.moveTo(tx, ty - 6);
          ctx.lineTo(tx, ty + 6);
          ctx.stroke();

          ctx.fillStyle = "#fde047";
          ctx.font = "9.5px monospace";
          ctx.fillText(`Est T#${trk.target_id} (${trk.hops}h)`, tx + 8, ty - 8);
        }
      }
    }

    // 8. Hostile Targets & Encirclement Rings
    if (state.targets) {
      for (const t of state.targets) {
        if (t.trail && t.trail.length > 1) {
          ctx.strokeStyle = t.captured ? "rgba(16, 185, 129, 0.2)" : "rgba(244, 63, 94, 0.35)";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(t.trail[0][0], t.trail[0][1]);
          for (let k = 1; k < t.trail.length; k++) {
            ctx.lineTo(t.trail[k][0], t.trail[k][1]);
          }
          ctx.stroke();
        }

        // Active SHORAD Lethal Turret Envelope & Kinetic Tracer Fire
        if (t.armed && !t.captured) {
          const wRange = t.weapon_range || 64;
          ctx.fillStyle = "rgba(239, 68, 68, 0.08)";
          ctx.strokeStyle = "rgba(251, 146, 60, 0.65)";
          ctx.setLineDash([6, 3]);
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(t.x, t.y, wRange, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.setLineDash([]);

          if (t.shot_pos && t.shot_pos.length === 2) {
            const [sx, sy] = t.shot_pos;
            ctx.strokeStyle = "#fde047";
            ctx.lineWidth = 2.8;
            ctx.beginPath();
            ctx.moveTo(t.x, t.y);
            ctx.lineTo(sx, sy);
            ctx.stroke();

            ctx.fillStyle = "#f97316";
            ctx.beginPath();
            ctx.arc(sx, sy, 8, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        ctx.strokeStyle = t.captured ? "rgba(16, 185, 129, 0.45)" : "rgba(244, 63, 94, 0.28)";
        ctx.setLineDash([4, 4]);
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(t.x, t.y, state.config.desired_encircle_radius, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);

        if (t.progress > 0 && !t.captured) {
          ctx.strokeStyle = "#10b981";
          ctx.lineWidth = 3.5;
          ctx.beginPath();
          ctx.arc(t.x, t.y, 16, -Math.PI / 2, -Math.PI / 2 + t.progress * Math.PI * 2);
          ctx.stroke();
        }

        ctx.fillStyle = t.captured ? "#10b981" : (t.armed ? "#f97316" : "#f43f5e");
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(t.x, t.y, 9, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 10px sans-serif";
        const tag = t.captured ? `T#${t.id} [NEUTRALIZED]` : (t.armed ? `T#${t.id} [SHORAD]` : `T#${t.id}`);
        ctx.fillText(tag, t.x + 12, t.y + 4);
      }
    }

    // 9. Heterogeneous Autonomous Swarm Assets
    if (state.drones) {
      for (const d of state.drones) {
        const color = ROLE_COLORS[d.role] || "#38bdf8";

        if (layers.showSensors && d.active) {
          ctx.strokeStyle = "rgba(56, 189, 248, 0.08)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(d.x, d.y, d.r_sense || state.config.sensor_range, 0, Math.PI * 2);
          ctx.stroke();
        }

        if (layers.showVectors && d.active) {
          ctx.strokeStyle = color;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.moveTo(d.x, d.y);
          ctx.lineTo(d.x + d.vx * 0.26, d.y + d.vy * 0.26);
          ctx.stroke();
        }

        // Jammed Indicator Ring
        // Spoofed (Byzantine) Halo & Micro Energy Bar (Ostrom-Pitt Self-Governance)
        if (d.spoofed && d.active) {
          ctx.strokeStyle = "#e879f9";
          ctx.setLineDash([3, 2]);
          ctx.lineWidth = 2.0;
          ctx.beginPath();
          ctx.arc(d.x, d.y, 13, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.fillStyle = "#f0abfc";
          ctx.font = "bold 8.5px monospace";
          ctx.fillText("SPOOF", d.x - 14, d.y - 14);
        }
        if (d.active && d.energy !== undefined && d.energy < 0.98) {
          const bw = 14;
          ctx.fillStyle = "rgba(15, 23, 42, 0.75)";
          ctx.fillRect(d.x - bw / 2, d.y + 10, bw, 2.5);
          ctx.fillStyle = d.energy < 0.35 ? "#f43f5e" : "#38bdf8";
          ctx.fillRect(d.x - bw / 2, d.y + 10, bw * d.energy, 2.5);
        }
        if (d.jammed && d.active) {
          ctx.strokeStyle = "rgba(244, 63, 94, 0.85)";
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(d.x, d.y, 11, 0, Math.PI * 2);
          ctx.stroke();
        } else if (d.cbf && d.active) {
          // CBF Safety Shield Intervention Halo
          ctx.strokeStyle = "rgba(245, 158, 11, 0.75)";
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(d.x, d.y, 11, 0, Math.PI * 2);
          ctx.stroke();
        }

        if (d.id === selectedDroneId) {
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2.2;
          ctx.beginPath();
          ctx.arc(d.x, d.y, 13.5, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Distinct Airframe Silhouette
        const angle = Math.atan2(d.vy, d.vx);
        ctx.save();
        ctx.translate(d.x, d.y);
        ctx.rotate(angle);
        ctx.fillStyle = color;
        ctx.strokeStyle = "#070b14";
        ctx.lineWidth = 1.2;
        ctx.beginPath();

        if (d.asset_type === "SCOUT") {
          ctx.moveTo(7, 0);
          ctx.lineTo(-4, -8);
          ctx.lineTo(-2, 0);
          ctx.lineTo(-4, 8);
        } else if (d.asset_type === "RELAY") {
          ctx.moveTo(7, 0);
          ctx.lineTo(0, -6);
          ctx.lineTo(-7, 0);
          ctx.lineTo(0, 6);
        } else {
          // Interceptor
          ctx.moveTo(9.5, 0);
          ctx.lineTo(-6.5, -5.0);
          ctx.lineTo(-3.5, 0);
          ctx.lineTo(-6.5, 5.0);
        }

        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        ctx.restore();
      }
    }
  }
}
