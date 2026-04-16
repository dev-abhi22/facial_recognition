import cv2
import math
import numpy as np
import keyboard
import time
from collections import deque
from cvzone.FaceMeshModule import FaceMeshDetector
import fsds
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ==========================================
# FSDS CONNECTION SETUP
# ==========================================
print("Connecting to FSDS...")
client = fsds.FSDSClient()
client.confirmConnection()
client.enableApiControl(False)
print("FSDS Connected Successfully!")

# --- CALIBRATE GLOBAL ORIGIN ---
GLOBAL_OFFSET_X = 0.0
GLOBAL_OFFSET_Y = 0.0
print("Calibrating track coordinate offsets...")
try:
    ref_state = client.getRefereeState()
    o_big_x, o_big_y = [], []
    for c in ref_state.cones:
        color = c.get('color', -1) if isinstance(c, dict) else getattr(c, 'color', -1)
        if color == 2:  # Big Orange
            x = c.get('x', c.get('x_val', 0.0)) if isinstance(c, dict) else getattr(c, 'x', getattr(c, 'x_val', 0.0))
            y = c.get('y', c.get('y_val', 0.0)) if isinstance(c, dict) else getattr(c, 'y', getattr(c, 'y_val', 0.0))
            o_big_x.append(x / 100.0)
            o_big_y.append(y / 100.0)
    if len(o_big_x) > 0:
        GLOBAL_OFFSET_X = sum(o_big_x) / len(o_big_x)
        GLOBAL_OFFSET_Y = sum(o_big_y) / len(o_big_y)
        print(f"Global Offset applied: X={GLOBAL_OFFSET_X:.2f}m, Y={GLOBAL_OFFSET_Y:.2f}m")
    else:
        print("No Start/Finish line found. Origin will remain at 0,0.")
except Exception as e:
    print(f"Warning: Failed to parse global offset: {e}")

# --- Camera Setup ---
cap      = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

# --- DMS Thresholds & Counters ---
MAR_THRESH     = 0.3
ROLL_THRESH    = 30
ALARM_FRAMES_1 = 150
ALARM_FRAMES_2 = 60
sleep_frames   = 0
yawn_frames    = 0
yawn_numbers   = 0
wobble_frames  = 0

# --- Risk Factor Configuration ---
WEIGHT_STEERING = 15
WEIGHT_SLEEP    = 10
WEIGHT_WOBBLE   = 5
WEIGHT_YAWN     = 2
RISK_THRESHOLD  = 500
risk_score      = 0
risk_timer      = 0
auto_engaged    = False

# --- Manual Steering Variables ---
TURN_SPEED    = 8
MAX_ANGLE     = 180
BUFFER_SIZE   = 45
angle_changes = deque(maxlen=BUFFER_SIZE)
current_angle = 0
prev_angle    = 0

# ==========================================
# CONTROL CONSTANTS
# ==========================================
WHEELBASE_M      = 1.5
MAX_STEER_RAD    = 0.7
MAX_VELOCITY     = 6.0
VEL_LIMIT_FACTOR = 0.6
STEER_ALPHA      = 0.7

# Lookahead: how many waypoints ahead to search for the best target.
# With ~0.5m spacing, 5 = 2.5m min, 80 = 40m max. Tune as needed.
LOOKAHEAD_MIN   = 5     # minimum waypoints ahead to consider (skip points behind/under car)
LOOKAHEAD_MAX   = 80    # maximum waypoints ahead to consider
LOOKAHEAD_STEP  = 1     # check every waypoint in range

PREDICTION_DT     = 0.3
SATURATION_WEIGHT = 15.0

# How far back/forward to scan when finding closest waypoint.
# Keep backward window TINY to prevent regression.
CLOSEST_SEARCH_BACK = 2
CLOSEST_SEARCH_FWD  = 60   # generous forward window so we don't lose track after curves


# ==========================================
# UTILITY FUNCTIONS
# ==========================================
class PID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self._i = 0.0
        self._prev_err = None

    def reset(self):
        self._i, self._prev_err = 0.0, None

    def update(self, err, dt):
        dt = max(dt, 1e-3)
        self._i += err * dt
        d = 0.0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err
        return max(0.0, min(1.0, self.kp * err + self.ki * self._i + self.kd * d))


def find_closest_index(car_x, car_y, xs, ys, cur_idx):
    """
    Finds closest path index using a forward-biased window.
    Never looks more than CLOSEST_SEARCH_BACK steps behind cur_idx.
    This prevents regression on circular paths.
    """
    N = len(xs)
    if N == 0:
        return 0

    best_idx    = cur_idx
    min_dist_sq = float('inf')

    total_range = CLOSEST_SEARCH_BACK + CLOSEST_SEARCH_FWD + 1
    for i in range(-CLOSEST_SEARCH_BACK, CLOSEST_SEARCH_FWD + 1):
        idx     = (cur_idx + i) % N
        dx      = xs[idx] - car_x
        dy      = ys[idx] - car_y
        dist_sq = dx * dx + dy * dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_idx    = idx

    # --- Monotonic guard ---
    # Only accept new_idx if it is genuinely ahead (or the same).
    # Modular forward distance: if it wraps, that's fine — we just crossed lap boundary.
    fwd_steps = (best_idx - cur_idx) % N
    if fwd_steps > N // 2:
        # Would be a large backward jump — reject and keep cur_idx.
        # Exception: if the distance to cur_idx is very large (car is lost),
        # allow a re-lock by doing a full path scan.
        dx_cur   = xs[cur_idx] - car_x
        dy_cur   = ys[cur_idx] - car_y
        if dx_cur * dx_cur + dy_cur * dy_cur > 25.0:  # >5m off current point → re-lock
            best_idx = int(np.argmin(
                (xs - car_x) ** 2 + (ys - car_y) ** 2
            ))
        else:
            best_idx = cur_idx

    return best_idx


def predict_bicycle_state(x, y, yaw, v, steering, wheelbase, dt):
    if abs(steering) < 1e-4:
        return x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw
    tan_delta = math.tan(steering)
    r         = wheelbase / tan_delta
    beta      = (v / r) * dt
    cx_       = x - r * math.sin(yaw)
    cy_       = y + r * math.cos(yaw)
    yaw_new   = yaw + beta
    return cx_ + r * math.sin(yaw_new), cy_ - r * math.cos(yaw_new), yaw_new


def get_steering_to_point(px, py, yaw, tx, ty, wheelbase):
    dx      = tx - px
    dy      = ty - py
    lx      =  dx * math.cos(yaw) + dy * math.sin(yaw)
    ly      = -dx * math.sin(yaw) + dy * math.cos(yaw)
    dist_sq = lx ** 2 + ly ** 2
    if dist_sq < 0.01:
        return 0.0
    return math.atan(wheelbase * 2.0 * ly / dist_sq)


def calculate_trajectory_cost(pred_x, pred_y, pred_yaw,
                               path_x, path_y, path_yaws, candidate_idx):
    search_radius = 20
    N = len(path_x)

    min_dist_sq = float('inf')
    match_idx   = candidate_idx

    for j in range(-5, search_radius):
        idx     = (candidate_idx + j) % N
        dx      = path_x[idx] - pred_x
        dy      = path_y[idx] - pred_y
        dist_sq = dx * dx + dy * dy
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            match_idx   = idx

    error_dist = math.sqrt(min_dist_sq)
    path_theta = path_yaws[match_idx]
    diff       = pred_yaw - path_theta
    error_yaw  = abs(math.atan2(math.sin(diff), math.cos(diff)))

    return 1.0 * error_dist + 2.5 * error_yaw, match_idx


# ==========================================
# FSDS HELPERS
# ==========================================
def get_vehicle_state():
    try:
        kin   = client.getCarState().kinematics_estimated
        x     = kin.position.x_val + GLOBAL_OFFSET_X
        y     = kin.position.y_val + GLOBAL_OFFSET_Y
        q     = kin.orientation
        siny  = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
        cosy  = 1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2)
        yaw   = math.atan2(siny, cosy)
        speed = math.hypot(kin.linear_velocity.x_val, kin.linear_velocity.y_val)
        return x, y, yaw, speed
    except Exception as e:
        print(f"[FSDS state error] {e}")
        return None


def get_cone_centerline(vehicle_pos=None, vehicle_yaw=None):
    """
    Builds a dense, interpolated centerline from cone positions.
    Returns (xs, ys) arrays with ~0.5m spacing.
    """
    try:
        cones = client.getRefereeState().cones

        def get_val(item, key):
            if isinstance(item, dict):
                return item.get(key, item.get(key + '_val', 0.0))
            return getattr(item, key, getattr(item, key + '_val', 0.0))

        left_pts  = np.array([(get_val(c, 'x') / 100.0, get_val(c, 'y') / 100.0)
                               for c in cones if get_val(c, 'color') in (0, 2)], dtype=float)
        right_pts = np.array([(get_val(c, 'x') / 100.0, get_val(c, 'y') / 100.0)
                               for c in cones if get_val(c, 'color') in (1, 3)], dtype=float)

        if len(left_pts) < 2 or len(right_pts) < 2:
            return None, None

        # Build midpoint centerline: for each left cone, find nearest right cone
        unsorted = []
        for lx, ly in left_pts:
            dists = np.hypot(right_pts[:, 0] - lx, right_pts[:, 1] - ly)
            ri    = int(np.argmin(dists))
            unsorted.append(((lx + right_pts[ri, 0]) / 2.0,
                              (ly + right_pts[ri, 1]) / 2.0))
        unsorted = np.array(unsorted)

        # Sort starting from the point closest to the vehicle
        ref_x     = vehicle_pos[0] if vehicle_pos is not None else 0.0
        ref_y     = vehicle_pos[1] if vehicle_pos is not None else 0.0
        start_idx = int(np.argmin(np.hypot(unsorted[:, 0] - ref_x,
                                           unsorted[:, 1] - ref_y)))

        ordered   = []
        current   = unsorted[start_idx]
        unvisited = list(unsorted)
        unvisited.pop(start_idx)
        ordered.append(current)
        while unvisited:
            arr   = np.array(unvisited)
            dists = np.hypot(arr[:, 0] - current[0], arr[:, 1] - current[1])
            ni    = int(np.argmin(dists))
            current = unvisited.pop(ni)
            ordered.append(current)

        pts = np.array(ordered)

        # Flip direction if it's pointing opposite to vehicle heading
        if vehicle_yaw is not None and len(pts) > 2:
            seg_yaw = math.atan2(pts[1, 1] - pts[0, 1],
                                 pts[1, 0] - pts[0, 0])
            diff = math.atan2(math.sin(seg_yaw - vehicle_yaw),
                              math.cos(seg_yaw - vehicle_yaw))
            if abs(diff) > math.pi / 2.0:
                pts = np.concatenate([pts[:1], pts[1:][::-1]])
                print("[PATH] Direction reversed to match vehicle heading.")

        # -------------------------------------------------------
        # INTERPOLATION: resample to ~0.5m spacing
        # This is the key fix — dense path means cur_idx advances
        # smoothly every frame instead of getting stuck.
        # -------------------------------------------------------
        if len(pts) > 3:
            cumlen = np.concatenate([[0.0],
                                     np.cumsum(np.hypot(np.diff(pts[:, 0]),
                                                        np.diff(pts[:, 1])))])
            total_len = cumlen[-1]
            spacing   = 0.5  # metres per waypoint
            n_pts     = max(50, int(total_len / spacing))
            t_new     = np.linspace(0.0, total_len, n_pts)

            fx  = interp1d(cumlen, pts[:, 0], kind='linear')
            fy  = interp1d(cumlen, pts[:, 1], kind='linear')
            pts = np.column_stack([fx(t_new), fy(t_new)])

        return pts[:, 0], pts[:, 1]

    except Exception as e:
        print(f"[Cone path error] {e}")
        return None, None


def preview_track_matplotlib():
    print("Fetching track data for preview...")
    try:
        state = get_vehicle_state()
        vpos  = (state[0], state[1]) if state else None
        vyaw  = state[2]             if state else None

        cones = client.getRefereeState().cones

        def get_val(item, key):
            if isinstance(item, dict):
                return item.get(key, item.get(key + '_val', 0.0))
            return getattr(item, key, getattr(item, key + '_val', 0.0))

        left_pts  = np.array([(get_val(c, 'x') / 100.0, get_val(c, 'y') / 100.0)
                               for c in cones if get_val(c, 'color') in (0, 2)], dtype=float)
        right_pts = np.array([(get_val(c, 'x') / 100.0, get_val(c, 'y') / 100.0)
                               for c in cones if get_val(c, 'color') in (1, 3)], dtype=float)
        xs, ys = get_cone_centerline(vehicle_pos=vpos, vehicle_yaw=vyaw)

        plt.figure(figsize=(10, 8))
        if len(left_pts) > 0:
            plt.scatter(left_pts[:, 0],  left_pts[:, 1],  c='blue',   label='Left (raw)',  s=40)
        if len(right_pts) > 0:
            plt.scatter(right_pts[:, 0], right_pts[:, 1], c='orange', label='Right (raw)', s=40)
        if xs is not None and len(xs) > 0:
            plt.plot(xs, ys, c='red', linewidth=2.5, label='Centerline (aligned)')
            plt.scatter(xs[0], ys[0], c='green', marker='*', s=300, label='Path Start', zorder=5)
            if len(xs) > 3:
                plt.annotate("", xy=(xs[2], ys[2]), xytext=(xs[0], ys[0]),
                             arrowprops=dict(arrowstyle='->', color='green', lw=2))
        if vpos:
            plt.scatter(vpos[0], vpos[1], c='cyan', marker='^', s=200,
                        label='Vehicle Position', zorder=6)
        plt.title("Track Preview – Coordinate Alignment Check (Meters)")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.axis('equal')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.show(block=False)
        plt.pause(2)
        plt.close()
    except Exception as e:
        print(f"[Matplotlib Preview Error] {e}")


# ==========================================
# AUTONOMOUS CONTROL STATE
# ==========================================
speed_pid         = PID(kp=3.2, ki=0.0, kd=0.0)
filtered_steer    = 0.0
cur_idx           = 0
prev_best_idx     = 0
last_auto_t       = time.perf_counter()
route_x_cached    = np.array([])
route_y_cached    = np.array([])
path_yaws_cached  = np.array([])
_path_initialized = False


def run_autonomous_control(filtered_steer, cur_idx, prev_best_idx,
                           last_auto_t,
                           route_x_cached, route_y_cached, path_yaws_cached,
                           _path_initialized):

    now         = time.perf_counter()
    dt          = max(now - last_auto_t, 1e-3)
    last_auto_t = now

    state = get_vehicle_state()
    if state is None:
        return (filtered_steer, cur_idx, prev_best_idx, last_auto_t,
                route_x_cached, route_y_cached, path_yaws_cached,
                _path_initialized)
    cx, cy, yaw, speed = state

    # --- Path initialisation (runs once) ---
    if not _path_initialized or len(route_x_cached) < 3:
        xs, ys = get_cone_centerline(vehicle_pos=(cx, cy), vehicle_yaw=yaw)
        if xs is None or len(xs) < 3:
            ctrl          = fsds.CarControls()
            ctrl.throttle = 0.0
            ctrl.brake    = 0.5
            ctrl.steering = 0.0
            client.setCarControls(ctrl)
            return (filtered_steer, cur_idx, prev_best_idx, last_auto_t,
                    route_x_cached, route_y_cached, path_yaws_cached,
                    _path_initialized)

        route_x_cached   = xs.copy()
        route_y_cached   = ys.copy()
        dx_arr           = np.gradient(route_x_cached)
        dy_arr           = np.gradient(route_y_cached)
        path_yaws_cached = np.arctan2(dy_arr, dx_arr)

        # Lock starting index to the waypoint closest to the car
        dists         = np.hypot(route_x_cached - cx, route_y_cached - cy)
        cur_idx       = int(np.argmin(dists))
        prev_best_idx = cur_idx
        _path_initialized = True
        print(f"[INIT] Path locked! Total waypoints: {len(route_x_cached)}")
        print(f"[INIT] Starting at idx={cur_idx}, "
              f"wp=({route_x_cached[cur_idx]:.1f},{route_y_cached[cur_idx]:.1f}), "
              f"car=({cx:.1f},{cy:.1f})")

    N = len(route_x_cached)

    # --- Step 1: Find the closest waypoint (forward-biased) ---
    cur_idx = find_closest_index(cx, cy, route_x_cached, route_y_cached, cur_idx)

    # --- Step 2: Optimisation over lookahead window ---
    best_cost      = float('inf')
    best_idx       = (cur_idx + LOOKAHEAD_MIN) % N
    best_steer_req = 0.0

    pred_v = float(np.clip(speed, 1.0, MAX_VELOCITY))

    for step in range(LOOKAHEAD_MIN, LOOKAHEAD_MAX + 1, LOOKAHEAD_STEP):
        i = (cur_idx + step) % N

        tx, ty    = route_x_cached[i], route_y_cached[i]
        steer_req = get_steering_to_point(cx, cy, yaw, tx, ty, WHEELBASE_M)
        valid_steer = float(np.clip(steer_req, -MAX_STEER_RAD, MAX_STEER_RAD))

        px, py, pyaw = predict_bicycle_state(
            cx, cy, yaw, pred_v, valid_steer, WHEELBASE_M, PREDICTION_DT)

        cost, _ = calculate_trajectory_cost(
            px, py, pyaw,
            route_x_cached, route_y_cached, path_yaws_cached, i)

        # Penalise saturated steering
        if abs(steer_req) > MAX_STEER_RAD:
            cost += (abs(steer_req) - MAX_STEER_RAD) * SATURATION_WEIGHT

        if cost < best_cost:
            best_cost      = cost
            best_idx       = i
            best_steer_req = steer_req

    prev_best_idx = best_idx

    raw_target_steer = float(np.clip(best_steer_req, -MAX_STEER_RAD, MAX_STEER_RAD))
    filtered_steer   = STEER_ALPHA * raw_target_steer + (1.0 - STEER_ALPHA) * filtered_steer
    steering_norm    = filtered_steer / MAX_STEER_RAD

    target_speed = MAX_VELOCITY * (1.0 - VEL_LIMIT_FACTOR * abs(steering_norm))
    target_speed = max(1.0, min(MAX_VELOCITY, target_speed))
    throttle     = speed_pid.update(target_speed - speed, dt=dt)

    ctrl          = fsds.CarControls()
    ctrl.steering = float(np.clip(steering_norm, -1.0, 1.0))
    ctrl.throttle = float(throttle)
    ctrl.brake    = 0.0
    client.setCarControls(ctrl)

    print(f"[DBG] cur_idx={cur_idx}/{N}  best_target={best_idx}  "
          f"pos=({cx:.1f},{cy:.1f})  spd={speed:.2f}  "
          f"yaw={math.degrees(yaw):.1f}°  steer={steering_norm:.3f}")

    return (filtered_steer, cur_idx, prev_best_idx, last_auto_t,
            route_x_cached, route_y_cached, path_yaws_cached,
            _path_initialized)


# ==========================================
# 1. BASELINE CALIBRATION LOOP
# ==========================================
print("Calibrating Driver Baseline... Please look straight ahead.")
preview_track_matplotlib()
ear_vals, pitch_vals, yaw_vals = [], [], []
while True:
    success, img = cap.read()
    if not success:
        break
    img, faces = detector.findFaceMesh(img, draw=True)
    if faces:
        face = faces[0]
        l_vert, _ = detector.findDistance(face[159], face[145])
        l_horz, _ = detector.findDistance(face[130], face[133])
        r_vert, _ = detector.findDistance(face[386], face[374])
        r_horz, _ = detector.findDistance(face[362], face[263])
        ear_vals.append(((l_vert / l_horz if l_horz else 0)
                       + (r_vert / r_horz if r_horz else 0)) / 2)
        nose, chin, forehead = face[1], face[152], face[10]
        nc, _ = detector.findDistance(nose, chin)
        nf, _ = detector.findDistance(nose, forehead)
        pitch_vals.append(nc / nf if nf else 0)
        nl, _ = detector.findDistance(nose, face[234])
        nr, _ = detector.findDistance(nose, face[454])
        yaw_vals.append(nl / nr if nr else 1)
        if len(pitch_vals) >= 30:
            break

    pct = int(100 * len(pitch_vals) / 30)
    bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
    cv2.putText(img, f"CALIBRATING  [{bar}]  {pct}%",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.imshow("Combined Advanced DMS", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

EAR_THRESH   = sum(ear_vals)   / max(len(ear_vals),   1)
PITCH_THRESH = sum(pitch_vals) / max(len(pitch_vals), 1)
YAW_base     = sum(yaw_vals)   / max(len(yaw_vals),   1)
YAW_MIN      = YAW_base * 0.2
YAW_MAX      = YAW_base * 1.8
print(f"Calibration Complete!  EAR={EAR_THRESH:.3f}  "
      f"PITCH={PITCH_THRESH:.3f}  YAW={YAW_base:.3f}")
print("Monitoring Active.")

avg_ear = pitch = mar = roll = yaw_face = 0.0
volatility = 0

# ==========================================
# 2. MAIN MONITORING & FSDS LOOP
# ==========================================
while True:
    success, img = cap.read()
    if not success:
        break
    h, w, _ = img.shape
    WHEEL_CENTER = (w - 80, h - 80)
    WHEEL_RADIUS = 60

    if keyboard.is_pressed('left'):  current_angle -= TURN_SPEED
    if keyboard.is_pressed('right'): current_angle += TURN_SPEED
    current_angle = max(-MAX_ANGLE, min(MAX_ANGLE, current_angle))
    delta = abs(current_angle - prev_angle)
    angle_changes.append(delta)
    prev_angle = current_angle
    volatility = sum(angle_changes)

    img, faces = detector.findFaceMesh(img, draw=True)
    current_frame_risk = 0

    if faces:
        face = faces[0]
        l_vert, _ = detector.findDistance(face[159], face[145])
        l_horz, _ = detector.findDistance(face[130], face[133])
        r_vert, _ = detector.findDistance(face[386], face[374])
        r_horz, _ = detector.findDistance(face[362], face[263])
        avg_ear    = ((l_vert / l_horz if l_horz else 0)
                    + (r_vert / r_horz if r_horz else 0)) / 2
        m_vert, _ = detector.findDistance(face[11],  face[16])
        m_horz, _ = detector.findDistance(face[78],  face[308])
        mar        = m_vert / m_horz if m_horz else 0
        nose, chin, forehead = face[1], face[152], face[10]
        nc, _ = detector.findDistance(nose, chin)
        nf, _ = detector.findDistance(nose, forehead)
        pitch = nc / nf if nf else 0
        nl, _ = detector.findDistance(nose, face[234])
        nr, _ = detector.findDistance(nose, face[454])
        yaw_face = nl / nr if nr else 1
        leo, reo = face[33], face[263]
        roll     = math.degrees(math.atan2(reo[1] - leo[1], reo[0] - leo[0]))

        if avg_ear < 0.8 * EAR_THRESH: sleep_frames += 1
        else:                           sleep_frames  = 0
        if mar > MAR_THRESH:            yawn_frames  += 1
        else:                           yawn_frames   = 0
        if (pitch < 0.67 * PITCH_THRESH or pitch > 3 * PITCH_THRESH) or \
           (yaw_face < YAW_MIN or yaw_face > YAW_MAX) or abs(roll) > ROLL_THRESH:
            wobble_frames += 1
        else:
            wobble_frames = max(0, wobble_frames - 2)

    warning_color   = (0, 255, 0)
    wheel_color     = (255, 255, 255)
    active_warnings = []

    if sleep_frames > ALARM_FRAMES_2:
        active_warnings.append("DANGER: DRIVER ASLEEP!")
        current_frame_risk += WEIGHT_SLEEP
        warning_color = (0, 0, 255)
    if wobble_frames > ALARM_FRAMES_1:
        active_warnings.append("WARNING: HEAD DISTRACTED!")
        current_frame_risk += WEIGHT_WOBBLE
        warning_color = (0, 165, 255)
    if yawn_frames > ALARM_FRAMES_2:
        yawn_numbers += 1
        yawn_frames   = 0
    if yawn_numbers > 5:
        active_warnings.append("WARNING: EXCESSIVE YAWNING!")
        current_frame_risk += WEIGHT_YAWN
        warning_color = (0, 255, 255)
    if volatility > 180:
        active_warnings.append("DANGER: ERRATIC STEERING!")
        current_frame_risk += WEIGHT_STEERING
        wheel_color = (0, 0, 255)

    if not auto_engaged:
        risk_score += current_frame_risk

    risk_timer += 1
    if risk_timer > 300:
        if not auto_engaged:
            risk_score = 0
        yawn_numbers = 0
        risk_timer   = 0

    if risk_score > RISK_THRESHOLD:
        auto_engaged = True

    if auto_engaged:
        cv2.rectangle(img, (20, 30), (640, 160), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, "AUTO-PILOT ENGAGED [DMS Triggered]",
                    (30, 80),  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(img, "Press 'M' to regain manual control",
                    (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        client.enableApiControl(True)

        (filtered_steer, cur_idx, prev_best_idx,
         last_auto_t, route_x_cached, route_y_cached,
         path_yaws_cached, _path_initialized) = run_autonomous_control(
            filtered_steer, cur_idx, prev_best_idx,
            last_auto_t, route_x_cached, route_y_cached,
            path_yaws_cached, _path_initialized)
    else:
        client.enableApiControl(False)
        cv2.putText(img, "MANUAL MODE",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # --- 'M' key: manual override / reset ---
    if keyboard.is_pressed('m'):
        auto_engaged      = False
        risk_score        = 0
        risk_timer        = 0
        filtered_steer    = 0.0
        _path_initialized = False
        speed_pid.reset()
        client.enableApiControl(False)
        print("[OVERRIDE] Manual control restored by driver.")

    # --- On-screen warnings ---
    for i, text in enumerate(active_warnings):
        cv2.putText(img, text, (20, 200 + i * 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 3)

    # --- Steering wheel visualisation ---
    cv2.circle(img, WHEEL_CENTER, WHEEL_RADIUS, wheel_color, 4)
    theta_rad = math.radians(current_angle - 90)
    end_x = int(WHEEL_CENTER[0] + WHEEL_RADIUS * math.cos(theta_rad))
    end_y = int(WHEEL_CENTER[1] + WHEEL_RADIUS * math.sin(theta_rad))
    cv2.line(img, WHEEL_CENTER, (end_x, end_y), wheel_color, 4)
    cv2.circle(img, WHEEL_CENTER, 10, wheel_color, cv2.FILLED)

    # --- HUD ---
    risk_color = (0,   0, 255) if risk_score > RISK_THRESHOLD * 0.75 else \
                 (0, 165, 255) if risk_score > RISK_THRESHOLD * 0.50 else (0, 255, 0)
    cv2.putText(img, f"Risk: {risk_score}/{RISK_THRESHOLD}",
                (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, risk_color, 2)
    vol_color = (0, 0, 255) if volatility > 180 else \
                (0, 255, 255) if volatility > 80 else (0, 255, 0)
    cv2.putText(img, f"Vol:{volatility}",
                (w - 140, h - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vol_color, 2)
    cv2.putText(img, f"idx:{cur_idx}",
                (w - 140, h - 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    cv2.putText(img,
                f"EAR:{avg_ear:.2f} MAR:{mar:.2f} P:{pitch:.2f} Y:{yaw_face:.2f} R:{int(roll)}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    cv2.imshow("Combined Advanced DMS", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Shutting down... Restoring manual control.")
client.enableApiControl(False)
cap.release()
cv2.destroyAllWindows()