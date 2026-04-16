import csv
import json
import math
import os
import time

import numpy as np
import fsds
import pathlib

# ============================================================
# CONFIG
# ============================================================
PATH_CSV = pathlib.Path(__file__).parent / "path.csv"
STATE_FILE = "dms_state.json"
VEHICLE_NAME = "FSCar"

CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# Vehicle / controller params
WHEELBASE_M = 1.5
MAX_STEER_RAD = 0.90
MAX_VELOCITY = 5.0
MIN_VELOCITY = 1.2

# Predictive controller params
LOOKAHEAD_MIN_DIST = 3.0
LOOKAHEAD_MAX_DIST = 18.0
LOOKAHEAD_SAMPLES = 24

PREDICTION_DT = 0.45
SATURATION_WEIGHT = 12.0
CTE_WEIGHT = 3.0
YAW_WEIGHT = 2.5
TARGET_CTE_WEIGHT = 2.5

# Path locking
LOCAL_SEARCH_BACK = 5
LOCAL_SEARCH_FWD = 80
RELOCK_DIST_THRESH = 4.0

# Steering smoothing / speed reduction
STEER_ALPHA = 0.55
VEL_LIMIT_FACTOR = 0.72

# PID speed control
KP = 1.2
KI = 0.02
KD = 0.04

# Path / stop logic
RESAMPLE_SPACING = 0.5
CLOSED_PATH_THRESH = 2.0

MIN_PROGRESS_FOR_WRAP = 0.80
STOP_NEAR_START_FRAC = 0.10
STOP_STABLE_COUNT = 15

GOAL_TOL = 1.5
GOAL_STOP_SPEED = 0.8


# ============================================================
# BASIC UTILS
# ============================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def wrap_angle(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


def quat_to_yaw(q):
    siny = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy = 1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2)
    return math.atan2(siny, cosy)


def euclid(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


# ============================================================
# CSV PATH LOADING
# ============================================================
def load_path_csv(path_file):
    pts = []
    with open(path_file, "r", newline="") as f:
        reader = csv.reader(f)
        first = next(reader, None)

        if first is None:
            raise ValueError("path.csv is empty")

        try:
            pts.append((float(first[0]), float(first[1])))
        except Exception:
            pass

        for row in reader:
            if len(row) < 2:
                continue
            try:
                pts.append((float(row[0]), float(row[1])))
            except Exception:
                continue

    if len(pts) < 2:
        raise ValueError("path.csv must contain at least 2 valid points")

    return np.array(pts, dtype=float)


def resample_path(path, ds=0.5):
    if len(path) < 2:
        return path.copy()

    seg_len = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = s[-1]

    if total < ds:
        return path.copy()

    s_new = np.arange(0.0, total, ds)
    if len(s_new) == 0 or (total - s_new[-1] > 1e-6):
        s_new = np.append(s_new, total)

    x_new = np.interp(s_new, s, path[:, 0])
    y_new = np.interp(s_new, s, path[:, 1])
    return np.column_stack((x_new, y_new))


def compute_headings(path):
    h = np.zeros(len(path))
    d = np.diff(path, axis=0)
    h[:-1] = np.arctan2(d[:, 1], d[:, 0])
    h[-1] = h[-2] if len(h) > 1 else 0.0
    return h


def compute_cumulative_s(path):
    if len(path) < 2:
        return np.array([0.0])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(seg)))


def is_closed_path(path):
    if len(path) < 3:
        return False
    return np.linalg.norm(path[0] - path[-1]) < CLOSED_PATH_THRESH


# ============================================================
# DMS STATE FILE READER
# ============================================================
def read_dms_state(retries=5, delay=0.005):
    if not os.path.exists(STATE_FILE):
        return None

    for _ in range(retries):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError, OSError):
            time.sleep(delay)

    return None


# ============================================================
# SPEED PID
# ============================================================
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self._i = 0.0
        self._prev_err = None

    def reset(self):
        self._i = 0.0
        self._prev_err = None

    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self._i += err * dt
        d = 0.0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        return self.kp * err + self.ki * self._i + self.kd * d


# ============================================================
# CONTROLLER CORE
# ============================================================
def front_axle_state(x, y, yaw, wheelbase):
    fx = x + wheelbase * math.cos(yaw)
    fy = y + wheelbase * math.sin(yaw)
    return fx, fy


def signed_cte_to_point(px, py, yaw, tx, ty):
    dx = tx - px
    dy = ty - py
    return dy * math.cos(yaw) - dx * math.sin(yaw)


def get_steering_to_point(px, py, yaw, tx, ty, wheelbase):
    dx = tx - px
    dy = ty - py

    lx = dx * math.cos(yaw) + dy * math.sin(yaw)
    ly = -dx * math.sin(yaw) + dy * math.cos(yaw)

    dist_sq = lx * lx + ly * ly
    if dist_sq < 0.01:
        return 0.0

    return math.atan2(2.0 * wheelbase * ly, dist_sq)


def predict_bicycle_state(x, y, yaw, v, steering, wheelbase, dt):
    if abs(steering) < 1e-4:
        return (
            x + v * math.cos(yaw) * dt,
            y + v * math.sin(yaw) * dt,
            yaw
        )

    tan_delta = math.tan(steering)
    r = wheelbase / tan_delta
    beta = (v / r) * dt

    cx_ = x - r * math.sin(yaw)
    cy_ = y + r * math.cos(yaw)
    yaw_new = yaw + beta

    return (
        cx_ + r * math.sin(yaw_new),
        cy_ - r * math.cos(yaw_new),
        yaw_new
    )


def full_relock_index(px, py, path_x, path_y):
    return int(np.argmin((path_x - px) ** 2 + (path_y - py) ** 2))


def local_relock_index(px, py, path_x, path_y, cur_idx, closed_loop):
    n = len(path_x)

    if closed_loop:
        cand = []
        for i in range(-LOCAL_SEARCH_BACK, LOCAL_SEARCH_FWD + 1):
            cand.append((cur_idx + i) % n)
        cand = np.array(cand, dtype=int)
        dx = path_x[cand] - px
        dy = path_y[cand] - py
        best_local = int(np.argmin(dx * dx + dy * dy))
        idx = int(cand[best_local])

        dist = math.hypot(path_x[idx] - px, path_y[idx] - py)
        if dist > RELOCK_DIST_THRESH:
            idx = full_relock_index(px, py, path_x, path_y)
        return idx
    else:
        start = max(0, cur_idx - LOCAL_SEARCH_BACK)
        end = min(n, cur_idx + LOCAL_SEARCH_FWD + 1)
        cand = np.arange(start, end)
        dx = path_x[cand] - px
        dy = path_y[cand] - py
        best_local = int(np.argmin(dx * dx + dy * dy))
        idx = int(cand[best_local])

        dist = math.hypot(path_x[idx] - px, path_y[idx] - py)
        if dist > RELOCK_DIST_THRESH:
            idx = full_relock_index(px, py, path_x, path_y)
        return idx


def nearest_path_error(px, py, pyaw, path_x, path_y, path_yaws, center_idx, closed_loop, radius=20):
    n = len(path_x)
    min_dist_sq = float("inf")
    match_idx = center_idx

    if closed_loop:
        for j in range(-radius, radius + 1):
            idx = (center_idx + j) % n
            dx = path_x[idx] - px
            dy = path_y[idx] - py
            d2 = dx * dx + dy * dy
            if d2 < min_dist_sq:
                min_dist_sq = d2
                match_idx = idx
    else:
        start = max(0, center_idx - radius)
        end = min(n, center_idx + radius + 1)
        for idx in range(start, end):
            dx = path_x[idx] - px
            dy = path_y[idx] - py
            d2 = dx * dx + dy * dy
            if d2 < min_dist_sq:
                min_dist_sq = d2
                match_idx = idx

    path_theta = path_yaws[match_idx]
    yaw_err = abs(wrap_angle(pyaw - path_theta))
    dist_err = math.sqrt(min_dist_sq)
    return dist_err, yaw_err, match_idx


def candidate_indices_by_distance(cur_idx, path_s, closed_loop, min_dist, max_dist, samples):
    n = len(path_s)
    if n < 2:
        return [cur_idx]

    s0 = path_s[cur_idx]
    total = path_s[-1]

    dists = np.linspace(min_dist, max_dist, samples)
    out = []

    for d in dists:
        target_s = s0 + d

        if closed_loop:
            if total <= 1e-6:
                idx = cur_idx
            else:
                target_s = target_s % total
                idx = int(np.argmin(np.abs(path_s - target_s)))
        else:
            target_s = min(target_s, total)
            idx = int(np.argmin(np.abs(path_s - target_s)))

        if idx not in out:
            out.append(idx)

    return out


# ============================================================
# FSDS HELPERS
# ============================================================
def get_vehicle_state(client):
    state = client.getCarState(VEHICLE_NAME)
    kin = state.kinematics_estimated

    x = kin.position.x_val
    y = kin.position.y_val
    yaw = quat_to_yaw(kin.orientation)
    speed = math.hypot(kin.linear_velocity.x_val, kin.linear_velocity.y_val)

    return x, y, yaw, speed


def stop_vehicle(client):
    ctrl = fsds.CarControls()
    ctrl.steering = 0.0
    ctrl.throttle = 0.0
    ctrl.brake = 1.0
    client.setCarControls(ctrl, VEHICLE_NAME)


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading path...")
    raw_path = load_path_csv(PATH_CSV)
    path = resample_path(raw_path, ds=RESAMPLE_SPACING)

    if len(path) > 3 and np.linalg.norm(path[0] - path[-1]) < 0.2:
        path = path[:-1]

    path_x = path[:, 0].copy()
    path_y = path[:, 1].copy()
    path_yaws = compute_headings(path)
    path_s = compute_cumulative_s(path)
    closed_loop = is_closed_path(path)

    print(f"Loaded {len(path_x)} path points. Closed loop: {closed_loop}")

    print("Connecting to FSDS...")
    client = fsds.FSDSClient()
    client.confirmConnection()
    client.enableApiControl(False, VEHICLE_NAME)
    print("FSDS connected. Waiting for DMS takeover request...")

    x0, y0, yaw0, _ = get_vehicle_state(client)

    err_forward = abs(wrap_angle(path_yaws[0] - yaw0))
    reverse_path = path[::-1].copy()
    reverse_yaws = compute_headings(reverse_path)
    err_reverse = abs(wrap_angle(reverse_yaws[0] - yaw0))

    if err_reverse < err_forward:
        path = reverse_path
        path_x = path[:, 0].copy()
        path_y = path[:, 1].copy()
        path_yaws = compute_headings(path)
        path_s = compute_cumulative_s(path)
        print("Path reversed to match spawn heading.")

    fx0, fy0 = front_axle_state(x0, y0, yaw0, WHEELBASE_M)
    cur_idx = full_relock_index(fx0, fy0, path_x, path_y)
    prev_idx = cur_idx

    speed_pid = PID(KP, KI, KD)
    filtered_steer = 0.0

    stable_finish_count = 0
    lap_armed = False
    lap_done = False
    in_auto = False
    last_auto_t = time.perf_counter()

    while True:
        loop_start = time.perf_counter()

        dms = read_dms_state()
        auto_request = bool(dms and dms.get("auto_engaged", False))
        manual_override = bool(dms and dms.get("manual_override", False))

        if manual_override:
            if in_auto:
                print("[CTRL] Manual override received. Returning control.")
            in_auto = False
            lap_armed = False
            lap_done = False
            stable_finish_count = 0
            speed_pid.reset()
            filtered_steer = 0.0
            stop_vehicle(client)
            client.enableApiControl(False, VEHICLE_NAME)

        if auto_request and not in_auto:
            print("[CTRL] DMS requested takeover. Enabling autonomous control.")
            x, y, yaw, _ = get_vehicle_state(client)
            fx, fy = front_axle_state(x, y, yaw, WHEELBASE_M)
            cur_idx = full_relock_index(fx, fy, path_x, path_y)
            prev_idx = cur_idx
            lap_armed = False
            lap_done = False
            stable_finish_count = 0
            speed_pid.reset()
            filtered_steer = 0.0
            last_auto_t = time.perf_counter()
            client.enableApiControl(True, VEHICLE_NAME)
            in_auto = True

        if not auto_request and in_auto:
            print("[CTRL] DMS no longer requests takeover. Returning manual control.")
            in_auto = False
            lap_armed = False
            lap_done = False
            stable_finish_count = 0
            speed_pid.reset()
            filtered_steer = 0.0
            stop_vehicle(client)
            client.enableApiControl(False, VEHICLE_NAME)

        if in_auto:
            now = time.perf_counter()
            dt = max(now - last_auto_t, 1e-3)
            last_auto_t = now

            cx, cy, yaw, speed = get_vehicle_state(client)
            fx, fy = front_axle_state(cx, cy, yaw, WHEELBASE_M)

            # Stronger path relock using front axle
            cur_idx = local_relock_index(fx, fy, path_x, path_y, cur_idx, closed_loop)
            current_path_dist = euclid(fx, fy, path_x[cur_idx], path_y[cur_idx])

            n = len(path_x)

            # Distance-based lookahead candidates
            candidates = candidate_indices_by_distance(
                cur_idx,
                path_s,
                closed_loop,
                LOOKAHEAD_MIN_DIST,
                LOOKAHEAD_MAX_DIST,
                LOOKAHEAD_SAMPLES
            )

            best_cost = float("inf")
            best_idx = candidates[0]
            best_steer_req = 0.0
            pred_v = float(np.clip(speed, 1.0, MAX_VELOCITY))

            for i in candidates:
                tx, ty = path_x[i], path_y[i]

                # Steering based on front axle, not CoM
                steer_req = get_steering_to_point(fx, fy, yaw, tx, ty, WHEELBASE_M)
                valid_steer = float(np.clip(steer_req, -MAX_STEER_RAD, MAX_STEER_RAD))

                px, py, pyaw = predict_bicycle_state(
                    cx, cy, yaw, pred_v, valid_steer, WHEELBASE_M, PREDICTION_DT
                )

                pfx, pfy = front_axle_state(px, py, pyaw, WHEELBASE_M)

                pred_dist_err, pred_yaw_err, match_idx = nearest_path_error(
                    pfx, pfy, pyaw, path_x, path_y, path_yaws, i, closed_loop, radius=20
                )

                target_cte = abs(signed_cte_to_point(fx, fy, yaw, tx, ty))

                cost = (
                    CTE_WEIGHT * pred_dist_err +
                    YAW_WEIGHT * pred_yaw_err +
                    TARGET_CTE_WEIGHT * target_cte
                )

                if abs(steer_req) > MAX_STEER_RAD:
                    cost += (abs(steer_req) - MAX_STEER_RAD) * SATURATION_WEIGHT

                # Small preference for forward progress
                if closed_loop:
                    progress_bonus = ((i - cur_idx) % n) * 0.01
                else:
                    progress_bonus = max(0, i - cur_idx) * 0.01
                cost -= progress_bonus

                if cost < best_cost:
                    best_cost = cost
                    best_idx = i
                    best_steer_req = steer_req

            raw_target_steer = float(np.clip(best_steer_req, -MAX_STEER_RAD, MAX_STEER_RAD))
            filtered_steer = STEER_ALPHA * raw_target_steer + (1.0 - STEER_ALPHA) * filtered_steer
            steering_norm = clamp(filtered_steer / MAX_STEER_RAD, -1.0, 1.0)

            # Speed scheduling: slower in strong steering and when far off path
            target_speed = MAX_VELOCITY * (1.0 - VEL_LIMIT_FACTOR * abs(steering_norm))
            if current_path_dist > 2.0:
                target_speed = min(target_speed, 2.5)
            if current_path_dist > 4.0:
                target_speed = min(target_speed, 1.5)

            target_speed = clamp(target_speed, MIN_VELOCITY, MAX_VELOCITY)

            if closed_loop:
                progress_ratio = cur_idx / max(1, n - 1)

                if progress_ratio > MIN_PROGRESS_FOR_WRAP:
                    lap_armed = True

                if lap_armed and prev_idx > int(0.85 * n) and cur_idx < int(0.15 * n):
                    lap_done = True

                if lap_done:
                    target_speed = min(target_speed, 2.0)
                    if cur_idx < int(STOP_NEAR_START_FRAC * n):
                        target_speed = 0.0
            else:
                progress_ratio = cur_idx / max(1, n - 1)
                dist_to_goal = euclid(cx, cy, path_x[-1], path_y[-1])

                if progress_ratio > 0.90:
                    target_speed = min(target_speed, 3.0)
                if progress_ratio > 0.97:
                    target_speed = min(target_speed, 1.5)
                if progress_ratio > 0.995 or dist_to_goal < 2.0:
                    target_speed = 0.0

            speed_cmd = speed_pid.update(target_speed - speed, dt=dt)
            throttle = clamp(speed_cmd, 0.0, 1.0)
            brake = clamp(-speed_cmd, 0.0, 1.0)

            ctrl = fsds.CarControls()
            ctrl.steering = float(steering_norm)
            ctrl.throttle = float(throttle)
            ctrl.brake = float(brake)
            client.setCarControls(ctrl, VEHICLE_NAME)

            risk_txt = dms.get("risk_score", 0) if dms else 0
            print(
                f"[AUTO] cur_idx={cur_idx}/{n-1} best={best_idx} "
                f"pos=({cx:.2f},{cy:.2f}) v={speed:.2f} vref={target_speed:.2f} "
                f"steer={ctrl.steering:.3f} path_err={current_path_dist:.2f} "
                f"cost={best_cost:.3f} risk={risk_txt}"
            )

            if closed_loop:
                if lap_done and speed < GOAL_STOP_SPEED and cur_idx < int(STOP_NEAR_START_FRAC * n):
                    stable_finish_count += 1
                else:
                    stable_finish_count = 0
            else:
                reached_last = cur_idx >= n - 3
                near_goal = euclid(cx, cy, path_x[-1], path_y[-1]) < GOAL_TOL
                slow = speed < GOAL_STOP_SPEED

                if reached_last and near_goal and slow:
                    stable_finish_count += 1
                else:
                    stable_finish_count = 0

            if stable_finish_count >= STOP_STABLE_COUNT:
                print("[CTRL] Path/lap complete. Holding brake, staying in AUTO until DMS reset.")
                stop_vehicle(client)

            prev_idx = cur_idx

        elapsed = time.perf_counter() - loop_start
        sleep_time = max(0.0, DT - elapsed)
        try:
            time.sleep(sleep_time)
        except KeyboardInterrupt:
            print("\n[CTRL] Interrupted by user.")
            break

    try:
        stop_vehicle(client)
        client.enableApiControl(False, VEHICLE_NAME)
    except Exception:
        pass


if __name__ == "__main__":
    main()