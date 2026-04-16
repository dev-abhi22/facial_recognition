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

WHEELBASE = 1.5
MAX_STEER_RAD = 0.50
MIN_SPEED = 1.5
MAX_SPEED = 4.0

STANLEY_K = 1.4
STANLEY_SOFT = 1.0

KP = 0.40
KI = 0.03
KD = 0.01

RESAMPLE_SPACING = 0.5
CLOSED_PATH_THRESH = 2.0

SEARCH_BEHIND = 5
SEARCH_AHEAD = 100

MIN_PROGRESS_FOR_WRAP = 0.80
STOP_NEAR_START_FRAC = 0.10
STOP_STABLE_COUNT = 15

GOAL_TOL = 1.5
GOAL_STOP_SPEED = 0.8


# ============================================================
# UTILS
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


def compute_curvature(path):
    n = len(path)
    kappa = np.zeros(n)

    for i in range(1, n - 1):
        p0 = path[i - 1]
        p1 = path[i]
        p2 = path[i + 1]

        a = np.linalg.norm(p1 - p0)
        b = np.linalg.norm(p2 - p1)
        c = np.linalg.norm(p2 - p0)

        if a < 1e-6 or b < 1e-6 or c < 1e-6:
            continue

        area2 = abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1]) -
            (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
        kappa[i] = area2 / (a * b * c)

    if n > 2:
        kappa[0] = kappa[1]
        kappa[-1] = kappa[-2]

    return kappa


def make_speed_profile(curvature, vmin=MIN_SPEED, vmax=MAX_SPEED, gain=18.0):
    v = vmax / (1.0 + gain * np.abs(curvature))
    return np.clip(v, vmin, vmax)


def is_closed_path(path):
    if len(path) < 3:
        return False
    return np.linalg.norm(path[0] - path[-1]) < CLOSED_PATH_THRESH


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
# CONTROLLERS
# ============================================================
class PID:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.i = 0.0
        self.prev_e = 0.0
        self.first = True

    def reset(self):
        self.i = 0.0
        self.prev_e = 0.0
        self.first = True

    def step(self, error, dt):
        dt = max(dt, 1e-3)
        self.i += error * dt
        d = 0.0 if self.first else (error - self.prev_e) / dt
        self.first = False
        self.prev_e = error
        return self.kp * error + self.ki * self.i + self.kd * d


class StanleyController:
    def __init__(self, k=1.4, soft=1.0):
        self.k = k
        self.soft = soft

    def control(self, x, y, yaw, speed, front_x, front_y, path, headings, last_idx, closed_loop):
        n = len(path)

        if closed_loop:
            candidate_indices = []
            for j in range(-SEARCH_BEHIND, SEARCH_AHEAD + 1):
                idx = (last_idx + j) % n
                candidate_indices.append(idx)
            candidate_indices = np.array(candidate_indices, dtype=int)
            local = path[candidate_indices]
            dx = local[:, 0] - front_x
            dy = local[:, 1] - front_y
            dist2 = dx * dx + dy * dy
            best_local = int(np.argmin(dist2))
            idx = int(candidate_indices[best_local])
        else:
            start = max(0, last_idx - SEARCH_BEHIND)
            end = min(n, last_idx + SEARCH_AHEAD)
            local = path[start:end]
            dx = local[:, 0] - front_x
            dy = local[:, 1] - front_y
            dist2 = dx * dx + dy * dy
            best_local = int(np.argmin(dist2))
            idx = start + best_local

        tx, ty = path[idx]
        path_yaw = headings[idx]

        heading_error = wrap_angle(path_yaw - yaw)

        vx = tx - front_x
        vy = ty - front_y
        cte = vy * math.cos(yaw) - vx * math.sin(yaw)

        steer = heading_error + math.atan2(self.k * cte, self.soft + speed)
        steer = clamp(steer, -MAX_STEER_RAD, MAX_STEER_RAD)

        return steer, idx, cte, heading_error


# ============================================================
# FSDS HELPERS
# ============================================================
def get_vehicle_state(client):
    state = client.getCarState(VEHICLE_NAME)
    kin = state.kinematics_estimated

    x = kin.position.x_val
    y = kin.position.y_val
    yaw = quat_to_yaw(kin.orientation)
    vx = kin.linear_velocity.x_val
    vy = kin.linear_velocity.y_val
    speed = math.hypot(vx, vy)

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
    headings = compute_headings(path)
    curvature = compute_curvature(path)
    speed_profile = make_speed_profile(curvature)
    closed_loop = is_closed_path(path)

    print(f"Loaded {len(path)} path points. Closed loop: {closed_loop}")

    print("Connecting to FSDS...")
    client = fsds.FSDSClient()
    client.confirmConnection()
    client.enableApiControl(False, VEHICLE_NAME)
    print("FSDS connected. Waiting for DMS takeover request...")

    x0, y0, yaw0, _ = get_vehicle_state(client)

    err_forward = abs(wrap_angle(headings[0] - yaw0))
    err_reverse = abs(wrap_angle(wrap_angle(headings[-1] + math.pi) - yaw0))

    if err_reverse < err_forward:
        path = path[::-1].copy()
        headings = compute_headings(path)
        curvature = compute_curvature(path)
        speed_profile = make_speed_profile(curvature)
        print("Path reversed to match spawn heading.")

    d0 = np.hypot(path[:, 0] - x0, path[:, 1] - y0)
    idx = int(np.argmin(d0))
    last_idx = idx
    prev_idx = idx

    lat = StanleyController(k=STANLEY_K, soft=STANLEY_SOFT)
    lon = PID(KP, KI, KD)

    stable_finish_count = 0
    lap_armed = False
    lap_done = False
    in_auto = False

    while True:
        t0 = time.perf_counter()

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
            lon.reset()
            stop_vehicle(client)
            client.enableApiControl(False, VEHICLE_NAME)

        if auto_request and not in_auto:
            print("[CTRL] DMS requested takeover. Enabling autonomous control.")
            x, y, yaw, _ = get_vehicle_state(client)
            d = np.hypot(path[:, 0] - x, path[:, 1] - y)
            idx = int(np.argmin(d))
            last_idx = idx
            prev_idx = idx
            lap_armed = False
            lap_done = False
            stable_finish_count = 0
            lon.reset()
            client.enableApiControl(True, VEHICLE_NAME)
            in_auto = True

        if not auto_request and in_auto:
            print("[CTRL] DMS no longer requests takeover. Returning manual control.")
            in_auto = False
            lap_armed = False
            lap_done = False
            stable_finish_count = 0
            lon.reset()
            stop_vehicle(client)
            client.enableApiControl(False, VEHICLE_NAME)

        if in_auto:
            x, y, yaw, speed = get_vehicle_state(client)

            front_x = x + WHEELBASE * math.cos(yaw)
            front_y = y + WHEELBASE * math.sin(yaw)

            steer, idx, cte, he = lat.control(
                x=x, y=y, yaw=yaw, speed=speed,
                front_x=front_x, front_y=front_y,
                path=path, headings=headings,
                last_idx=last_idx, closed_loop=closed_loop
            )

            n = len(path)

            if closed_loop:
                progress_ratio = idx / max(1, n - 1)

                if progress_ratio > MIN_PROGRESS_FOR_WRAP:
                    lap_armed = True

                if lap_armed and prev_idx > int(0.85 * n) and idx < int(0.15 * n):
                    lap_done = True

                last_idx = idx
                prev_idx = idx
            else:
                last_idx = max(last_idx, idx)
                prev_idx = idx
                progress_ratio = idx / max(1, n - 1)

            target_speed = speed_profile[idx]

            if closed_loop:
                if lap_done:
                    target_speed = min(target_speed, 2.0)
                    if idx < int(STOP_NEAR_START_FRAC * n):
                        target_speed = 0.0
            else:
                dist_to_goal = math.hypot(path[-1, 0] - x, path[-1, 1] - y)
                if progress_ratio > 0.90:
                    target_speed = min(target_speed, 3.0)
                if progress_ratio > 0.97:
                    target_speed = min(target_speed, 1.5)
                if progress_ratio > 0.995 or dist_to_goal < 2.0:
                    target_speed = 0.0

            speed_error = target_speed - speed
            u = lon.step(speed_error, DT)

            throttle = clamp(u, 0.0, 1.0)
            brake = clamp(-u, 0.0, 1.0)

            ctrl = fsds.CarControls()
            ctrl.steering = clamp(steer / MAX_STEER_RAD, -1.0, 1.0)
            ctrl.throttle = throttle
            ctrl.brake = brake
            client.setCarControls(ctrl, VEHICLE_NAME)

            risk_txt = dms.get("risk_score", 0) if dms else 0
            print(
                f"[AUTO] idx={idx}/{n-1} pos=({x:.2f},{y:.2f}) "
                f"v={speed:.2f} vref={target_speed:.2f} "
                f"steer={ctrl.steering:.3f} cte={cte:.3f} "
                f"he={math.degrees(he):.1f}deg risk={risk_txt}"
            )

            if closed_loop:
                if lap_done and speed < GOAL_STOP_SPEED and idx < int(STOP_NEAR_START_FRAC * n):
                    stable_finish_count += 1
                else:
                    stable_finish_count = 0
            else:
                reached_last = idx >= n - 3
                near_goal = math.hypot(path[-1, 0] - x, path[-1, 1] - y) < GOAL_TOL
                slow = speed < GOAL_STOP_SPEED

                if reached_last and near_goal and slow:
                    stable_finish_count += 1
                else:
                    stable_finish_count = 0

            if stable_finish_count >= STOP_STABLE_COUNT:
                print("[CTRL] Path/lap complete. Holding brake, staying in AUTO until DMS reset.")
                stop_vehicle(client)

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, DT - elapsed))


if __name__ == "__main__":
    main()