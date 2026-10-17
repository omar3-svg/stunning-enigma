"""Team Bama controller, v3: minimum-curvature racing line + friction-circle speed profile,
hardened with a data-driven stress test.

What changed from v1 (and why): we generated the organisers' imitation corpus
(github.com/catears124/ecocar-data-gen, 212 "brake + turn + obstacle" scenarios) and drove v1
through every scenario. v1 was clean on only 138/212: it hit cones when the free gap beside an
obstacle was narrower than its safety margin. v2 is clean on 209/212 with the same speed.
The three fixes:
  a. Fair gap sharing. When there is not enough room for the wanted clearances, v1 put the car
     at a point biased towards the boundary cones. v2 computes the real free gap and shares the
     spare room: more to the obstacle side on straights (obstacle cones are 0.6 m apart, boundary
     cones 5 m), more to the inside in corners (tracking errors go to the inside).
  b. Obstacle memory. The sensor drops an obstacle 3 m behind the car while the rear of the car
     is still next to it. v2 remembers each obstacle in course coordinates for 8 m after it, and
     chooses its pass side once (no left/right flip-flopping between steps).
  c. Larger clearance (1.4 m to the car edge) whenever the road is wide enough for it.

v3 adds robustness, found with the simulator's stress test (config/robustness.yaml perturbations):
  d. Actuator-delay compensation: when commands reach the wheels late, predict where the car will
     be (kinematic roll-out of the commands already queued) and steer from that pose.
     2-step delay: 0/8 -> 7/8 clean courses.
  e. Steering-bias estimation: compare the applied steering (obs.steer) with what we sent and
     subtract the difference. Steering bias: 5/8 -> 8/8 clean.
  f. Online grip estimation: if the car turns much less than the steering asks while cornering
     hard, the tyres are saturated; lower all grip budgets to the measured grip.
     20% less grip: 0/8 -> 4/8 clean. All three are inactive or no-cost in normal conditions.

Every step (no map, only obs.road):
  1. Racing line: sideways offsets from the centerline along the preview, chosen so the path is
     as straight as possible inside the allowed corridor (box-constrained least squares).
  2. Speed profile: v = sqrt(a_lat / curvature), a backward braking pass that shares grip between
     braking and cornering, and an assumed slow corner just past the end of what we can see.
  3. Control: pure pursuit on the racing line (rate-limited to the actuator) and a P speed
     controller clamped by the friction circle, with anti-spin pedal easing.

Depends only on numpy and autocross.api.
"""
from __future__ import annotations

import math

import numpy as np

from autocross.api import Command, Controller, Observation, TrackInfo, VehicleParams

G = 9.81


class TeamBama(Controller):
    def on_start(self, track: TrackInfo, vehicle: VehicleParams) -> None:
        super().on_start(track, vehicle)
        c, vp = self.config, vehicle
        # grip use (fractions of the published limits). Tested: higher values hit more cones.
        self.a_lat = float(c.get("lat_frac", 0.72)) * vp.mu * G
        self.a_dec = float(c.get("decel_frac", 0.45)) * vp.max_decel
        self.a_acc = vp.max_accel
        self.a_brake_max = float(c.get("brake_cmd_frac", 0.55)) * vp.max_decel
        self.grip_cmd = float(c.get("grip_cmd", 0.85))
        self.beta_lim = float(c.get("beta_lim", 0.05))
        self.v_max = float(c.get("v_max", 30.0))
        self.v_horizon = float(c.get("v_horizon", 9.0))      # assumed speed for an unseen corner
        # racing line
        self.margin = float(c.get("margin", 0.8)) + 0.5 * vp.width   # car centre to cone line
        self.ds = float(c.get("plan_ds", 2.0))
        self.w_prev = float(c.get("w_prev", 3.0))            # stickiness to previous plan
        self.w_center = float(c.get("w_center", 0.002))      # weak pull to centerline
        # obstacles and tight gaps
        self.obs_margin = float(c.get("obs_margin", 1.4))    # wanted clearance, car edge to obstacle
        self.obs_pre = float(c.get("obs_pre", 4.5))          # m before an obstacle that is blocked
        self.obs_post = float(c.get("obs_post", 4.5))        # m after it
        self.obs_share = float(c.get("obs_share", 0.6))      # spare room given to the obstacle side
        self.share_in = float(c.get("share_in", 0.6))        # spare room given to a corner's inside
        self.k_corner = float(c.get("k_corner", 1.0 / 30.0)) # curvature that counts as a corner
        # steering and speed control
        self.k_ld = float(c.get("lookahead_gain", 0.3))
        self.ld_min = float(c.get("lookahead_min", 4.0))
        self.ld_max = float(c.get("lookahead_max", 14.0))
        self.kp_v = float(c.get("kp_speed", 2.0))
        # state
        self.obs_mem = []            # remembered obstacles: [course station, lat_min, lat_max, side]
        self.prev_plan = None        # (s, d) of last plan, in last step's station
        self.prev_progress = None
        self.prev_steer = 0.0
        self._kc = None
        # robustness (stress-tested): actuator delay compensation and steering-bias estimation
        self.delay_comp = bool(c.get("delay_comp", True))
        self.bias_est = bool(c.get("bias_est", True))
        self.bias_gain = float(c.get("bias_gain", 0.15))
        self.bias = 0.0
        self.cmd_hist = []           # steering commands we sent, newest last
        # grip estimation: if the car stops turning as much as the steering asks (tyres saturated),
        # the real grip is lower than published: lower all grip budgets to what was measured
        self.grip_adapt = bool(c.get("grip_adapt", True))
        self.sat_ratio = float(c.get("sat_ratio", 0.68))     # yaw rate / kinematic yaw rate
        self.sat_ay = float(c.get("sat_ay", 4.5))            # only judge when cornering hard
        self.mu_hat = vp.mu
        self.beta_cut = float(c.get("beta_cut", 1.0))
        self._sat_n = 0
        self._prev_obs_steer = 0.0
        self._base = (self.a_lat, self.a_dec, self.a_brake_max)

    # ------------------------------------------------------------------ obstacles
    def _update_obstacles(self, obstacles, progress):
        """Store obstacles in course coordinates (station = progress + distance) so they are not
        forgotten when the sensor drops them. Same obstacle = same station and overlapping width."""
        for ob in obstacles:
            if not np.isfinite(ob.distance):
                continue
            st = progress + ob.distance
            for m in self.obs_mem:
                if abs(m[0] - st) < 1.0 and ob.lateral_min < m[2] + 0.5 and ob.lateral_max > m[1] - 0.5:
                    m[0], m[1], m[2] = st, ob.lateral_min, ob.lateral_max
                    break
            else:
                self.obs_mem.append([st, ob.lateral_min, ob.lateral_max, 0])
        self.obs_mem = [m for m in self.obs_mem if m[0] > progress - 8.0]

    # ------------------------------------------------------------------ racing line
    def _plan_offsets(self, s, wl, wr, lat0, progress, shift):
        n = len(s)
        hw = 0.5 * self.vehicle.width
        # physical corridor for the car centre (zero clearance) ...
        p_lo = -wr + hw
        p_hi = wl - hw
        # ... and the clearance we would like on each side
        m_lo = np.full(n, self.margin - hw)
        m_hi = np.full(n, self.margin - hw)
        wt_lo = np.full(n, 0.5)
        wt_hi = np.full(n, 0.5)
        for m in self.obs_mem:
            dist = m[0] - progress
            win = (s > dist - self.obs_pre) & (s < dist + self.obs_post)
            if not win.any():
                continue
            if m[3] == 0:   # choose the side with more room, once, then keep it
                room_left = np.min(wl[win]) - m[2]
                room_right = m[1] + np.min(wr[win])
                m[3] = 1 if room_left >= room_right else -1
            if m[3] > 0:    # pass on the left: the obstacle becomes the right-hand wall
                edge = m[2] + hw
                upd = win & (edge > p_lo)
                p_lo[upd] = edge
                m_lo[upd] = self.obs_margin
                wt_lo[upd] = self.obs_share
            else:           # pass on the right: the obstacle becomes the left-hand wall
                edge = m[1] - hw
                upd = win & (edge < p_hi)
                p_hi[upd] = edge
                m_hi[upd] = self.obs_margin
                wt_hi[upd] = self.obs_share
        lo = p_lo + m_lo
        hi = p_hi - m_hi
        bad = lo > hi
        if bad.any():
            # not enough room for the wanted clearances: place the car inside the real gap,
            # frac = share of the spare room left on the right-hand (lo) side
            wl_, wh_ = wt_lo[bad], wt_hi[bad]
            frac = np.where(wl_ != wh_, np.where(wl_ > wh_, wl_, 1.0 - wh_), 0.5)
            kb = self._kc[bad]
            corner = np.abs(kb) > self.k_corner          # right turn (k < 0): inside is the right
            frac = np.where(corner, np.where(kb < 0, self.share_in, 1.0 - self.share_in), frac)
            mid = np.where(p_hi[bad] > p_lo[bad], p_lo[bad] + frac * (p_hi[bad] - p_lo[bad]),
                           0.5 * (p_lo[bad] + p_hi[bad]))
            lo[bad] = hi[bad] = mid

        # quadratic objective on offsets d (path p = c + d * normal)
        H = self._H + self.w_center * np.eye(n)
        g = self._g.copy()
        # stickiness to previous plan in the near field (decays with distance)
        if self.prev_plan is not None:
            ps, pd = self.prev_plan
            dprev = np.interp(s + shift, ps, pd, left=pd[0], right=pd[-1])
            w = self.w_prev * np.exp(-s / 12.0)
            H = H + np.diag(w)
            g = g - w * dprev
        # first point is where the car is now (hard)
        lo[0] = hi[0] = lat0
        return self._box_qp(H, g, lo, hi)

    @staticmethod
    def _box_qp(H, g, lo, hi, iters=8):
        """min 0.5 d'Hd + g'd  s.t. lo <= d <= hi   (small active-set loop)."""
        n = len(g)
        fixed = np.zeros(n, bool)
        val = np.zeros(n)
        fixed[0] = True
        val[0] = lo[0]
        d = np.zeros(n)
        for _ in range(iters):
            free = ~fixed
            d[fixed] = val[fixed]
            rhs = -g[free] - H[np.ix_(free, fixed)] @ d[fixed]
            d[free] = np.linalg.solve(H[np.ix_(free, free)], rhs)
            viol_lo = free & (d < lo - 1e-6)
            viol_hi = free & (d > hi + 1e-6)
            if not (viol_lo.any() or viol_hi.any()):
                break
            fixed |= viol_lo | viol_hi
            val[viol_lo] = lo[viol_lo]
            val[viol_hi] = hi[viol_hi]
        return np.clip(d, lo, hi)

    def _build_qp(self, c, nrm):
        """Curvature proxy: sum of squared second differences of the path points."""
        n = len(c)
        D = np.zeros((n - 2, n))
        idx = np.arange(n - 2)
        D[idx, idx], D[idx, idx + 1], D[idx, idx + 2] = 1.0, -2.0, 1.0
        Ax = D * nrm[:, 0]
        Ay = D * nrm[:, 1]
        self._H = Ax.T @ Ax + Ay.T @ Ay
        self._g = Ax.T @ (D @ c[:, 0]) + Ay.T @ (D @ c[:, 1])

    # ------------------------------------------------------------------ speed profile
    def _speed_profile(self, s, path, v_end):
        n = len(s)
        kap = np.zeros(n)
        if n >= 3:
            seg_v = np.diff(path, axis=0)
            psi = np.unwrap(np.arctan2(seg_v[:, 1], seg_v[:, 0]))
            seg_len = np.maximum(np.linalg.norm(seg_v, axis=1), 1e-6)
            k_mid = np.diff(psi) / (0.5 * (seg_len[:-1] + seg_len[1:]))
            kap[1:-1] = np.abs(k_mid)
            kap[0], kap[-1] = kap[1], kap[-2]
            for _ in range(2):   # smooth out noise from the 2 m sampling
                kap = np.convolve(np.pad(kap, 1, mode="edge"), np.ones(3) / 3, "valid")
        v_lim = np.minimum(np.sqrt(self.a_lat / np.maximum(kap, 1e-4)), self.v_max)
        v = v_lim.copy()
        v[-1] = min(v[-1], v_end)
        seg = np.diff(s)
        for i in range(n - 2, -1, -1):   # braking pass with friction circle
            lat_use = min((v[i + 1] ** 2 * kap[i + 1]) / self.a_lat, 1.0)
            a_avail = self.a_dec * math.sqrt(max(1.0 - lat_use ** 2, 0.05))
            v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * a_avail * seg[i]))
        return v, kap

    def _update_grip(self, obs, v):
        vp = self.vehicle
        k_kin = v * math.tan(obs.steer * vp.max_steer) / vp.wheelbase
        steady = abs(obs.steer - self._prev_obs_steer) < 0.02
        self._prev_obs_steer = obs.steer
        sat = (v > 5.0 and abs(obs.ay) > self.sat_ay and abs(k_kin) > 0.1 and steady
               and obs.yaw_rate * k_kin > 0 and obs.yaw_rate / k_kin < self.sat_ratio)
        self._sat_n = self._sat_n + 1 if sat else 0
        if self._sat_n >= 3:
            a_meas = math.hypot(obs.ax, obs.ay)
            self.mu_hat = max(min(self.mu_hat, 1.02 * a_meas / G), 0.4 * vp.mu)
        else:
            self.mu_hat = min(vp.mu, self.mu_hat + 0.003 * vp.control_dt)
        sc = self.mu_hat / vp.mu
        self.a_lat, self.a_dec, self.a_brake_max = (b * sc for b in self._base)

    def _predict_pose(self, obs, v, queued):
        """Kinematic bicycle roll-out over the commands already waiting in the actuator."""
        vp = self.vehicle
        x = y = yaw = 0.0
        st = obs.steer
        dt = vp.control_dt
        rate = vp.max_steer_rate * dt
        # yaw response of the real car relative to the kinematic model (handles understeer)
        k_kin = v * math.tan(st * vp.max_steer) / vp.wheelbase
        ratio = float(np.clip(obs.yaw_rate / k_kin, 0.5, 1.2)) if abs(k_kin) > 0.05 else 1.0
        for cmd in queued:
            st = st + float(np.clip(cmd + self.bias - st, -rate, rate))
            r = ratio * v * math.tan(st * vp.max_steer) / vp.wheelbase
            yaw_mid = yaw + 0.5 * r * dt
            x += v * dt * math.cos(yaw_mid)
            y += v * dt * math.sin(yaw_mid)
            yaw += r * dt
        return x, y, yaw

    # ------------------------------------------------------------------ main loop
    def step(self, obs: Observation) -> Command:
        road, vp = obs.road, self.vehicle
        if road is None or not road.valid or len(road.preview_s) < 4:
            return Command(brake=0.6)
        v = max(obs.speed, 0.0)
        if self.grip_adapt:
            self._update_grip(obs, v)

        # resample preview at plan_ds spacing
        s_raw = np.asarray(road.preview_s, float)
        s = np.arange(0.0, s_raw[-1] + 1e-6, self.ds)
        if len(s) < 4:
            s = s_raw
        cl = np.column_stack([np.interp(s, s_raw, road.preview_centerline[:, 0]),
                              np.interp(s, s_raw, road.preview_centerline[:, 1])])
        wl = np.interp(s, s_raw, road.preview_width_left)
        wr = np.interp(s, s_raw, road.preview_width_right)
        self._kc = np.interp(s, s_raw, road.preview_curvature)
        t = np.gradient(cl, axis=0)
        t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
        nrm = np.column_stack([-t[:, 1], t[:, 0]])          # left normal

        # how far we moved along the course since last step (to shift the old plan)
        shift = 0.0 if self.prev_progress is None else max(obs.progress - self.prev_progress, 0.0)
        self.prev_progress = obs.progress

        self._build_qp(cl, nrm)
        self._update_obstacles(road.obstacles, obs.progress)
        d = self._plan_offsets(s, wl, wr, road.lateral_offset, obs.progress, shift)
        self.prev_plan = (s, d)
        path = cl + d[:, None] * nrm

        # end-of-sight assumption: unseen corner unless the finish is in view
        v_end = self.v_horizon
        if np.isfinite(road.finish_distance) and road.finish_distance < s[-1]:
            v_end = self.v_max
        v_prof, kap = self._speed_profile(s, path, v_end)

        # ---- steering: pure pursuit on the racing line
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        ld = float(np.clip(self.k_ld * v + 2.0, self.ld_min, self.ld_max))
        nd = int(vp.actuation_delay_steps)
        if self.delay_comp and nd > 0 and len(self.cmd_hist) >= nd:
            # the next nd commands are already queued in the actuator: predict where the car will
            # be when this command starts to act, and steer from that predicted pose
            px, py, pyaw = self._predict_pose(obs, v, self.cmd_hist[-nd:])
            c_, s_ = math.cos(pyaw), math.sin(pyaw)
            rel = np.column_stack([(path[:, 0] - px) * c_ + (path[:, 1] - py) * s_,
                                   -(path[:, 0] - px) * s_ + (path[:, 1] - py) * c_])
            i0 = int(np.argmin(np.hypot(rel[:, 0], rel[:, 1]) + 1e3 * (rel[:, 0] < -2.0)))
            a0 = arc[i0] + max(rel[i0, 0], 0.0)
            tgt = np.array([np.interp(a0 + ld, arc, rel[:, 0]), np.interp(a0 + ld, arc, rel[:, 1])])
        else:
            tgt = np.array([np.interp(ld, arc, path[:, 0]), np.interp(ld, arc, path[:, 1])])
        alpha = math.atan2(tgt[1], max(tgt[0], 0.1))
        delta = math.atan2(2.0 * vp.wheelbase * math.sin(alpha), max(math.hypot(*tgt), 1.0))
        steer = float(np.clip(delta / vp.max_steer, -1.0, 1.0))
        # steering bias: the applied steering (obs.steer) should equal what we sent nd+1 steps ago
        if self.bias_est and len(self.cmd_hist) >= nd + 2:
            sent = self.cmd_hist[-1 - nd]
            prev = self.cmd_hist[-2 - nd]
            if abs(obs.steer) < 0.95 and abs(sent - prev) < 0.02:   # actuator has settled
                self.bias += self.bias_gain * ((obs.steer - sent) - self.bias)
                self.bias = float(np.clip(self.bias, -0.15, 0.15))
        steer = float(np.clip(steer - self.bias, -1.0, 1.0))
        max_dsteer = 0.95 * vp.max_steer_rate * vp.control_dt
        steer = float(np.clip(steer, self.prev_steer - max_dsteer, self.prev_steer + max_dsteer))
        self.prev_steer = steer
        self.cmd_hist.append(steer)
        if len(self.cmd_hist) > 20:
            self.cmd_hist.pop(0)

        # ---- speed: target a little ahead (covers actuator lag)
        look = v * vp.control_dt * (2 + vp.actuation_delay_steps)
        v_tgt = float(np.interp(look, arc, v_prof))
        # target may rise only as fast as the car can accelerate (it jumps when a corner
        # leaves view); it may fall at once
        v_ref = getattr(self, "v_ref", v)
        v_tgt = min(v_tgt, max(v_ref, v) + self.a_acc * vp.control_dt)
        self.v_ref = v_tgt
        a_cmd = self.kp_v * (v_tgt - v)
        # friction circle: grip left for speeding up / braking after cornering needs
        a_tot = self.grip_cmd * self.mu_hat * G
        a_long_max = math.sqrt(max(a_tot ** 2 - obs.ay ** 2, (0.25 * a_tot) ** 2))
        a_cmd = float(np.clip(a_cmd, -min(a_long_max, self.a_brake_max), min(a_long_max, vp.max_accel)))
        # stability: if the car is sliding sideways, ease off the pedals (rear lets go under braking)
        beta = math.atan2(obs.vy, max(obs.vx, 1.0))
        if abs(beta) > self.beta_lim:
            a_cmd *= 0.4
            if abs(beta) > self.beta_cut:   # clearly sliding: no throttle at all
                a_cmd = min(a_cmd, 0.0)
        if a_cmd >= 0:
            lat_use = min(abs(obs.ay) / (vp.mu * G), 1.0)
            thr_cap = max(1.0 - lat_use ** 2, 0.15)   # no big throttle mid-corner
            throttle, brake = min(a_cmd / self.a_acc, thr_cap), 0.0
        else:
            throttle, brake = 0.0, min(-a_cmd / vp.max_decel, 1.0)

        return Command(steer=steer, throttle=float(throttle), brake=float(brake),
                       path=path, path_frame="vehicle", target_speed=v_tgt,
                       debug={"v_tgt": v_tgt, "lookahead": ld, "mu_hat": self.mu_hat, "bias": self.bias})
