// Native C++17 Fast Geometry, Physics, CBF Safety & Decoupled Search Kernel
// Compiled to a shared library (.dll / .so) with zero external library dependencies.

#include <cmath>
#include <algorithm>

#ifdef _WIN32
#define EXPORT extern "C" __declspec(dllexport)
#else
#define EXPORT extern "C"
#endif

static inline bool segment_occluded(
    double x1, double y1, double x2, double y2,
    int n_obs, const double* obs_xyr, double margin
) {
    if (n_obs <= 0) return false;
    double dx = x2 - x1;
    double dy = y2 - y1;
    double len_sq = dx * dx + dy * dy;
    if (len_sq < 1e-10) return false;

    for (int k = 0; k < n_obs; ++k) {
        double cx = obs_xyr[k * 3 + 0];
        double cy = obs_xyr[k * 3 + 1];
        double r  = obs_xyr[k * 3 + 2] + margin;
        if (r <= 0.0) continue;

        double fx = x1 - cx;
        double fy = y1 - cy;
        double t = -(fx * dx + fy * dy) / len_sq;
        if (t < 0.0) t = 0.0;
        else if (t > 1.0) t = 1.0;

        double px = x1 + t * dx - cx;
        double py = y1 + t * dy - cy;
        if (px * px + py * py < r * r) {
            return true;
        }
    }
    return false;
}

EXPORT int cpp_check_los(
    double x1, double y1, double x2, double y2,
    int n_obs, const double* obs_xyr, double margin
) {
    return segment_occluded(x1, y1, x2, y2, n_obs, obs_xyr, margin) ? 1 : 0;
}

EXPORT void cpp_compute_matrices(
    int n_drones,
    const double* pos,
    const unsigned char* active,
    int n_obs,
    const double* obs_xyr,
    double r_sense,
    double r_comm,
    int los_occlusion,
    unsigned char* adj_comm,
    unsigned char* adj_sense,
    double* dist_mat
) {
    for (int i = 0; i < n_drones * n_drones; ++i) {
        adj_comm[i] = 0;
        adj_sense[i] = 0;
        dist_mat[i] = 99999.0;
    }

    for (int i = 0; i < n_drones; ++i) {
        dist_mat[i * n_drones + i] = 0.0;
        if (!active[i]) continue;
        double xi = pos[i * 2 + 0];
        double yi = pos[i * 2 + 1];

        for (int j = i + 1; j < n_drones; ++j) {
            if (!active[j]) continue;
            double xj = pos[j * 2 + 0];
            double yj = pos[j * 2 + 1];
            double d = std::hypot(xi - xj, yi - yj);
            dist_mat[i * n_drones + j] = d;
            dist_mat[j * n_drones + i] = d;

            if (d <= r_comm || d <= r_sense) {
                bool occ = false;
                if (los_occlusion && n_obs > 0) {
                    occ = segment_occluded(xi, yi, xj, yj, n_obs, obs_xyr, -2.0);
                }
                if (!occ) {
                    if (d <= r_comm) {
                        adj_comm[i * n_drones + j] = 1;
                        adj_comm[j * n_drones + i] = 1;
                    }
                    if (d <= r_sense) {
                        adj_sense[i * n_drones + j] = 1;
                        adj_sense[j * n_drones + i] = 1;
                    }
                }
            }
        }
    }
}

EXPORT int cpp_step_physics(
    int n,
    double* pos,
    double* vel,
    const double* accel,
    const unsigned char* active,
    int n_obs,
    const double* obs_xyr,
    double radius,
    double max_speed,
    double max_accel,
    double drag_coeff,
    double dt,
    double width,
    double height,
    int check_inter_entity
) {
    int collisions = 0;

    for (int i = 0; i < n; ++i) {
        if (!active[i]) {
            vel[i * 2 + 0] = 0.0;
            vel[i * 2 + 1] = 0.0;
            continue;
        }
        double ax = accel[i * 2 + 0];
        double ay = accel[i * 2 + 1];
        double a_norm = std::hypot(ax, ay);
        if (a_norm > max_accel && a_norm > 1e-8) {
            ax *= (max_accel / a_norm);
            ay *= (max_accel / a_norm);
        }

        double vx = vel[i * 2 + 0] + (ax - drag_coeff * vel[i * 2 + 0]) * dt;
        double vy = vel[i * 2 + 1] + (ay - drag_coeff * vel[i * 2 + 1]) * dt;
        double v_norm = std::hypot(vx, vy);
        if (v_norm > max_speed && v_norm > 1e-8) {
            vx *= (max_speed / v_norm);
            vy *= (max_speed / v_norm);
        }

        double px = pos[i * 2 + 0] + vx * dt;
        double py = pos[i * 2 + 1] + vy * dt;

        for (int k = 0; k < n_obs; ++k) {
            double cx = obs_xyr[k * 3 + 0];
            double cy = obs_xyr[k * 3 + 1];
            double min_d = obs_xyr[k * 3 + 2] + radius;
            double dx = px - cx;
            double dy = py - cy;
            double d = std::hypot(dx, dy);
            if (d < min_d) {
                collisions++;
                double nx = (d > 1e-6) ? (dx / d) : 1.0;
                double ny = (d > 1e-6) ? (dy / d) : 0.0;
                px = cx + nx * (min_d + 0.5);
                py = cy + ny * (min_d + 0.5);
                double vn = vx * nx + vy * ny;
                if (vn < 0.0) {
                    vx -= 1.3 * vn * nx;
                    vy -= 1.3 * vn * ny;
                }
            }
        }

        double margin = radius + 2.0;
        if (px < margin) {
            px = margin;
            vx = std::abs(vx) * 0.6;
        } else if (px > width - margin) {
            px = width - margin;
            vx = -std::abs(vx) * 0.6;
        }
        if (py < margin) {
            py = margin;
            vy = std::abs(vy) * 0.6;
        } else if (py > height - margin) {
            py = height - margin;
            vy = -std::abs(vy) * 0.6;
        }

        pos[i * 2 + 0] = px;
        pos[i * 2 + 1] = py;
        vel[i * 2 + 0] = vx;
        vel[i * 2 + 1] = vy;
    }

    if (check_inter_entity && n > 1) {
        double min_d = 2.0 * radius;
        for (int i = 0; i < n; ++i) {
            if (!active[i]) continue;
            for (int j = i + 1; j < n; ++j) {
                if (!active[j]) continue;
                double dx = pos[i * 2 + 0] - pos[j * 2 + 0];
                double dy = pos[i * 2 + 1] - pos[j * 2 + 1];
                double d = std::hypot(dx, dy);
                if (d < min_d) {
                    collisions++;
                    double nx = (d > 1e-6) ? (dx / d) : 1.0;
                    double ny = (d > 1e-6) ? (dy / d) : 0.0;
                    double overlap = 0.5 * (min_d - d + 0.5);
                    pos[i * 2 + 0] += nx * overlap;
                    pos[i * 2 + 1] += ny * overlap;
                    pos[j * 2 + 0] -= nx * overlap;
                    pos[j * 2 + 1] -= ny * overlap;
                }
            }
        }
    }

    return collisions;
}

EXPORT void cpp_compute_frontier_search(
    int drone_id,
    double own_x,
    double own_y,
    double own_vx,
    double own_vy,
    double prev_wp_x,
    double prev_wp_y,
    int rows,
    int cols,
    const float* certainty_grid,
    int n_peers,
    const double* peer_pos,
    int n_obs,
    const double* obs_xyr,
    double world_w,
    double world_h,
    double sensor_range,
    double max_speed,
    double* out_v_and_wp // [vx, vy, wp_x, wp_y]
) {
    double cell_w = world_w / cols;
    double cell_h = world_h / rows;

    // Preferred persistent search compass angle per drone (golden-angle dispersion)
    double pref_ang = std::fmod(drone_id * 2.399963229728653, 6.283185307179586);
    double speed = std::hypot(own_vx, own_vy);
    double hx = (speed > 12.0) ? (own_vx / speed) : std::cos(pref_ang);
    double hy = (speed > 12.0) ? (own_vy / speed) : std::sin(pref_ang);

    // Check if the drone already has a valid uncommitted frontier waypoint:
    // Keep flying toward prev_wp until either we reach it (< 45 px) or its cell certainty is high (> 0.72).
    bool has_prev_wp = (prev_wp_x > 10.0 && prev_wp_y > 10.0);
    if (has_prev_wp) {
        int pc = std::clamp(static_cast<int>(prev_wp_x / cell_w), 0, cols - 1);
        int pr = std::clamp(static_cast<int>(prev_wp_y / cell_h), 0, rows - 1);
        float wp_cert = certainty_grid[pr * cols + pc];
        double dist_to_wp = std::hypot(prev_wp_x - own_x, prev_wp_y - own_y);
        if (dist_to_wp < 48.0 || wp_cert > 0.72f) {
            has_prev_wp = false; // Reached or already scanned -> pick a fresh frontier!
        }
    }

    double best_cx = prev_wp_x;
    double best_cy = prev_wp_y;
    double best_score = -1e9;

    for (int r = 0; r < rows; ++r) {
        double cy = (r + 0.5) * cell_h;
        for (int c = 0; c < cols; ++c) {
            double cx = (c + 0.5) * cell_w;

            // Skip cells inside or too close to obstacles
            bool in_obs = false;
            for (int k = 0; k < n_obs; ++k) {
                double odx = cx - obs_xyr[k * 3 + 0];
                double ody = cy - obs_xyr[k * 3 + 1];
                double safe_r = obs_xyr[k * 3 + 2] + 24.0;
                if (odx * odx + ody * ody < safe_r * safe_r) {
                    in_obs = true;
                    break;
                }
            }
            if (in_obs) continue;

            double uncert = 1.0 - static_cast<double>(certainty_grid[r * cols + c]);
            double dx = cx - own_x;
            double dy = cy - own_y;
            double my_dist = std::hypot(dx, dy);
            if (my_dist < 35.0) continue; // Don't pick a cell right under our nose

            // Continuous, smooth Voronoi territory penalty (no step discontinuity!)
            double voronoi_pen = 0.0;
            for (int p = 0; p < n_peers; ++p) {
                double pdx = cx - peer_pos[p * 2 + 0];
                double pdy = cy - peer_pos[p * 2 + 1];
                double p_dist = std::hypot(pdx, pdy);
                double advantage = my_dist - p_dist; // >0 means peer is closer than us
                if (advantage > -20.0) {
                    // Smooth ramp penalty
                    double ratio = std::clamp((advantage + 20.0) / 80.0, 0.0, 1.5);
                    voronoi_pen += 1.15 * ratio;
                }
            }

            // Strong hysteresis bonus if this cell is near our current committed waypoint
            double wp_hysteresis = 0.0;
            if (has_prev_wp) {
                double d_wp = std::hypot(cx - prev_wp_x, cy - prev_wp_y);
                if (d_wp < 65.0) {
                    wp_hysteresis = 1.45 * (1.0 - d_wp / 65.0);
                }
            }

            double cos_sim = (dx * hx + dy * hy) / std::max(my_dist, 1.0);
            double score = uncert * 3.0
                         - 0.0016 * my_dist
                         + 0.65 * cos_sim
                         + wp_hysteresis
                         - voronoi_pen;

            if (score > best_score) {
                best_score = score;
                best_cx = cx;
                best_cy = cy;
            }
        }
    }

    double dir_x = best_cx - own_x;
    double dir_y = best_cy - own_y;
    double d_norm = std::max(std::hypot(dir_x, dir_y), 1.0);
    double ux = dir_x / d_norm;
    double uy = dir_y / d_norm;

    // 2. Smooth Peer Dispersion with Rotational Curl (breaks collinear deadlock!)
    double rep_x = 0.0;
    double rep_y = 0.0;
    double spread_radius = sensor_range * 1.15;
    for (int p = 0; p < n_peers; ++p) {
        double rx = own_x - peer_pos[p * 2 + 0];
        double ry = own_y - peer_pos[p * 2 + 1];
        double d = std::hypot(rx, ry);
        if (d < spread_radius && d > 1e-4) {
            double strength = (spread_radius - d) / spread_radius;
            double mag = 0.48 * strength * strength;
            double nx = rx / d;
            double ny = ry / d;
            // Add 20% rotational curl so two drones facing each other smoothly swirl past
            rep_x += (nx - 0.22 * ny) * mag;
            rep_y += (ny + 0.22 * nx) * mag;
        }
    }

    // Clamp repulsion magnitude so it never cancels out forward cruise momentum
    double rep_norm = std::hypot(rep_x, rep_y);
    if (rep_norm > 0.55) {
        rep_x *= (0.55 / rep_norm);
        rep_y *= (0.55 / rep_norm);
    }

    double comb_x = ux * 0.88 + rep_x;
    double comb_y = uy * 0.88 + rep_y;
    double comb_norm = std::hypot(comb_x, comb_y);

    double vx, vy;
    if (comb_norm > 0.15) {
        // Always cruise at a healthy minimum speed (75%-92% max_speed) while searching
        vx = (comb_x / comb_norm) * (0.86 * max_speed);
        vy = (comb_y / comb_norm) * (0.86 * max_speed);
    } else {
        vx = ux * (0.82 * max_speed);
        vy = uy * (0.82 * max_speed);
    }

    out_v_and_wp[0] = vx;
    out_v_and_wp[1] = vy;
    out_v_and_wp[2] = best_cx;
    out_v_and_wp[3] = best_cy;
}

EXPORT void cpp_update_and_merge_grids(
    int n_drones,
    int rows,
    int cols,
    const double* pos,
    const unsigned char* active,
    const unsigned char* adj_comm,
    double world_w,
    double world_h,
    double sensor_range,
    float decay,
    float alpha,
    float* grids // shape: (n_drones, rows * cols)
) {
    int cells = rows * cols;
    double cell_w = world_w / cols;
    double cell_h = world_h / rows;
    double r_sq = (sensor_range * 0.9) * (sensor_range * 0.9);

    for (int i = 0; i < n_drones; ++i) {
        if (!active[i]) continue;
        float* g = grids + i * cells;
        double px = pos[i * 2 + 0];
        double py = pos[i * 2 + 1];

        for (int r = 0; r < rows; ++r) {
            double cy = (r + 0.5) * cell_h;
            double dy2 = (cy - py) * (cy - py);
            for (int c = 0; c < cols; ++c) {
                int idx = r * cols + c;
                double cx = (c + 0.5) * cell_w;
                if ((cx - px) * (cx - px) + dy2 <= r_sq) {
                    g[idx] = 1.0f;
                } else {
                    g[idx] *= decay;
                }
            }
        }
    }

    for (int i = 0; i < n_drones; ++i) {
        if (!active[i]) continue;
        float* gi = grids + i * cells;
        for (int j = 0; j < n_drones; ++j) {
            if (i == j || !active[j] || !adj_comm[i * n_drones + j]) continue;
            const float* gj = grids + j * cells;
            for (int k = 0; k < cells; ++k) {
                if (gj[k] > gi[k]) {
                    gi[k] += alpha * (gj[k] - gi[k]);
                }
            }
        }
    }
}
