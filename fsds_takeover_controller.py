import fsds
import csv
import json
import math
import time
import pathlib


# ============================================================
# CONFIG
# ============================================================
BASE_DIR = pathlib.Path(__file__).parent
PATH_CSV = BASE_DIR / "path.csv"
STATE_FILE = BASE_DIR / "dms_state.json"
VEHICLE_NAME = "FSCar"

WHEELBASE = 1.5
TARGET_SPEED = 5.0
MAX_STEER_RAD = 0.4363

LD_MIN = 1.0
LD_K = 1.0

SPEED_KP = 0.4
SPEED_KI = 0.05
SPEED_KD = 0.1
BRAKE_THRESHOLD = 0.3
MAX_THROTTLE = 0.5

CONTROL_DT = 0.05
SEARCH_WINDOW = 20
LAP_THRESHOLD = 3.0


# ============================================================
# DMS STATE
# ============================================================
def read_dms_state(retries=5, delay=0.005):
    if not STATE_FILE.exists():
        return None

    for _ in range(retries):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, PermissionError, OSError):
            time.sleep(delay)

    return None


# ============================================================
# EXACT CONTROLLER HELPERS
# ============================================================
def load_path(filepath):
    path = []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        prev = None
        for row in reader:
            point = (float(row["x"]), float(row["y"]))
            if prev is None or point != prev:
                path.append(point)
                prev = point
    return path


def get_yaw_from_quaternion(q):
    x, y, z, w = q.x_val, q.y_val, q.z_val, q.w_val
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def get_speed(state):
    vx = state.kinematics_estimated.linear_velocity.x_val
    vy = state.kinematics_estimated.linear_velocity.y_val
    return math.hypot(vx, vy)


def update_closest_idx(path, car_x, car_y, last_idx, search_window=20):
    n = len(path)
    end = min(last_idx + search_window, n)
    best_dist = float("inf")
    best_idx = last_idx
    for i in range(last_idx, end):
        dx = path[i][0] - car_x
        dy = path[i][1] - car_y
        d = dx * dx + dy * dy
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def circle_segment_intersection(car_x, car_y, r, p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    fx = p1[0] - car_x
    fy = p1[1] - car_y
    a = dx * dx + dy * dy
    if a < 1e-10:
        return None
    b = 2.0 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        return None
    sqrt_disc = math.sqrt(discriminant)
    t2 = (-b + sqrt_disc) / (2.0 * a)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    for t in (t2, t1):
        if 0.0 <= t <= 1.0:
            return (p1[0] + t * dx, p1[1] + t * dy)
    return None


def find_lookahead_point(path, car_x, car_y, lookahead_dist, last_idx):
    n = len(path)
    for i in range(last_idx, n - 1):
        pt = circle_segment_intersection(car_x, car_y, lookahead_dist, path[i], path[i + 1])
        if pt is not None:
            return pt
    return path[-1]


def pure_pursuit_steering(car_x, car_y, yaw, target_x, target_y, lookahead_dist):
    angle_to_target = math.atan2(target_y - car_y, target_x - car_x)
    alpha = angle_to_target - yaw
    alpha = math.atan2(math.sin(alpha), math.cos(alpha))
    delta_rad = math.atan2(2.0 * WHEELBASE * math.sin(alpha), lookahead_dist)
    delta_rad = -delta_rad
    return max(-1.0, min(1.0, delta_rad / MAX_STEER_RAD))


class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = None

    def compute(self, current, target, dt):
        error = target - current
        self.integral += error * dt
        self.integral = max(-2.0, min(2.0, self.integral))
        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / dt
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative


def compute_throttle_brake(pid, current_speed, target_speed, dt):
    pid_output = pid.compute(current_speed, target_speed, dt)
    overshoot = current_speed - target_speed
    if overshoot > BRAKE_THRESHOLD:
        pid.integral = 0.0
        return 0.0, min(0.4, overshoot * 0.3)
    elif overshoot > 0:
        return 0.0, 0.0
    else:
        return max(0.05, min(MAX_THROTTLE, pid_output)), 0.0


def stop_vehicle(client):
    stop = fsds.CarControls()
    stop.throttle = 0.0
    stop.steering = 0.0
    stop.brake = 1.0
    client.setCarControls(stop, VEHICLE_NAME)


# ============================================================
# MAIN
# ============================================================
def main():
    path = load_path(PATH_CSV)
    if len(path) < 2:
        raise ValueError("path.csv must contain at least 2 valid points")

    client = fsds.FSDSClient()
    client.confirmConnection()
    client.enableApiControl(False, VEHICLE_NAME)

    print(f"[CTRL] Watching DMS file: {STATE_FILE}")
    print("[CTRL] Exact pure pursuit controller ready. Waiting for DMS takeover request...")

    pid = PIDController(SPEED_KP, SPEED_KI, SPEED_KD)
    last_idx = 0
    lap_complete_lock = False
    prev_time = time.time()
    n = len(path)
    in_auto = False

    try:
        while True:
            dms = read_dms_state()
            auto_request = bool(dms and dms.get("auto_engaged", False))
            manual_override = bool(dms and dms.get("manual_override", False))

            if dms is None:
                print(f"[CTRL] No DMS state file found at: {STATE_FILE}")
                time.sleep(0.5)
                continue

            if manual_override:
                if in_auto:
                    print("[CTRL] Manual override received. Returning control.")
                in_auto = False
                pid.reset()
                last_idx = 0
                lap_complete_lock = False
                stop_vehicle(client)
                client.enableApiControl(False, VEHICLE_NAME)
                time.sleep(CONTROL_DT)
                continue

            if auto_request and not in_auto:
                print("[CTRL] DMS requested takeover. Enabling autonomous control.")
                state = client.getCarState(VEHICLE_NAME)
                car_x = state.kinematics_estimated.position.x_val
                car_y = state.kinematics_estimated.position.y_val

                best_dist = float("inf")
                best_idx = 0
                for i, (px, py) in enumerate(path):
                    d = (px - car_x) ** 2 + (py - car_y) ** 2
                    if d < best_dist:
                        best_dist = d
                        best_idx = i

                last_idx = best_idx
                lap_complete_lock = False
                pid.reset()
                prev_time = time.time()

                client.enableApiControl(True, VEHICLE_NAME)
                in_auto = True

            if not auto_request and in_auto:
                print("[CTRL] DMS no longer requests takeover. Returning manual control.")
                in_auto = False
                pid.reset()
                stop_vehicle(client)
                client.enableApiControl(False, VEHICLE_NAME)
                time.sleep(CONTROL_DT)
                continue

            if in_auto:
                now = time.time()
                dt = max(now - prev_time, 1e-3)
                prev_time = now

                state = client.getCarState(VEHICLE_NAME)
                car_x = state.kinematics_estimated.position.x_val
                car_y = state.kinematics_estimated.position.y_val
                yaw = get_yaw_from_quaternion(state.kinematics_estimated.orientation)
                speed = get_speed(state)

                lookahead_dist = LD_MIN + LD_K * speed
                last_idx = update_closest_idx(path, car_x, car_y, last_idx, search_window=SEARCH_WINDOW)

                dist_to_start = math.hypot(path[0][0] - car_x, path[0][1] - car_y)
                if last_idx > int(n * 0.9) and dist_to_start < LAP_THRESHOLD:
                    if not lap_complete_lock:
                        last_idx = 0
                        lap_complete_lock = True
                else:
                    if dist_to_start > LAP_THRESHOLD * 2:
                        lap_complete_lock = False

                target_x, target_y = find_lookahead_point(path, car_x, car_y, lookahead_dist, last_idx)
                steering = pure_pursuit_steering(car_x, car_y, yaw, target_x, target_y, lookahead_dist)
                throttle, brake = compute_throttle_brake(pid, speed, TARGET_SPEED, dt)

                controls = fsds.CarControls()
                controls.throttle = throttle
                controls.steering = steering
                controls.brake = brake
                client.setCarControls(controls, VEHICLE_NAME)

                risk_txt = dms.get("risk_score", 0)
                print(
                    f"[AUTO] idx={last_idx}/{n-1} "
                    f"pos=({car_x:.2f},{car_y:.2f}) "
                    f"v={speed:.2f} "
                    f"ld={lookahead_dist:.2f} "
                    f"target=({target_x:.2f},{target_y:.2f}) "
                    f"steer={steering:.3f} thr={throttle:.3f} brk={brake:.3f} "
                    f"risk={risk_txt}"
                )
            else:
                print(
                    f"[CTRL] Waiting... auto={auto_request} "
                    f"manual_override={manual_override} "
                    f"risk={dms.get('risk_score', 0)}"
                )

            time.sleep(CONTROL_DT)

    except KeyboardInterrupt:
        print("[CTRL] Stopped by user.")
    finally:
        try:
            stop_vehicle(client)
            client.enableApiControl(False, VEHICLE_NAME)
        except Exception:
            pass


if __name__ == "__main__":
    main()