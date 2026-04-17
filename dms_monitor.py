import cv2
import math
import json
import time
import os
import uuid
from collections import deque
import keyboard
from cvzone.FaceMeshModule import FaceMeshDetector
import pathlib

# ============================================================
# CONFIG (Core Mechanics - Unchanged)
# ============================================================
STATE_FILE = pathlib.Path(__file__).parent / "dms_state.json"

MAR_THRESH = 0.30
ROLL_THRESH = 30
ALARM_FRAMES_SLEEP = 60
ALARM_FRAMES_WOBBLE = 45
ALARM_FRAMES_YAWN = 20

WEIGHT_STEERING = 15
WEIGHT_SLEEP = 10
WEIGHT_WOBBLE = 3
WEIGHT_YAWN = 2

RISK_THRESHOLD = 700
RISK_RESET_FRAMES = 300

TURN_SPEED = 8
MAX_ANGLE = 180
BUFFER_SIZE = 45

# ============================================================
# STATE HELPERS
# ============================================================
def default_state():
    return {
        "timestamp": time.time(),
        "auto_engaged": False,
        "manual_override": False,
        "risk_score": 0,
        "sleep_frames": 0,
        "wobble_frames": 0,
        "yawn_count": 0,
        "volatility": 0,
        "ear": 0.0,
        "mar": 0.0,
        "pitch": 0.0,
        "yaw_face": 0.0,
        "roll": 0.0,
        "warnings": [],
        "calibrated": False
    }

def write_state(data, retries=10, delay=0.01):
    tmp = f"{STATE_FILE}.{uuid.uuid4().hex}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    for _ in range(retries):
        try:
            os.replace(tmp, STATE_FILE)
            return
        except (PermissionError, OSError):
            time.sleep(delay)
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"[DMS] Failed to write state: {e}")

# ============================================================
# INIT
# ============================================================
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

angle_changes = deque(maxlen=BUFFER_SIZE)
current_angle = 0
prev_angle = 0

sleep_frames = 0
yawn_frames = 0
yawn_numbers = 0
wobble_frames = 0
risk_score = 0
risk_timer = 0
auto_engaged = False

write_state(default_state())

# ============================================================
# CALIBRATION
# ============================================================
print("Calibrating driver baseline...")
ear_vals, pitch_vals, yaw_vals = [], [], []

while True:
    success, img = cap.read()
    if not success: continue
    img, faces = detector.findFaceMesh(img, draw=True)
    if faces:
        face = faces[0]
        l_vert, _ = detector.findDistance(face[159], face[145])
        l_horz, _ = detector.findDistance(face[130], face[133])
        r_vert, _ = detector.findDistance(face[386], face[374])
        r_horz, _ = detector.findDistance(face[362], face[263])
        ear = ((l_vert / l_horz if l_horz else 0.0) + (r_vert / r_horz if r_horz else 0.0)) / 2.0
        ear_vals.append(ear)
        nose, chin, forehead = face[1], face[152], face[10]
        nc, _ = detector.findDistance(nose, chin)
        nf, _ = detector.findDistance(nose, forehead)
        pitch = nc / nf if nf else 0.0
        pitch_vals.append(pitch)
        nl, _ = detector.findDistance(nose, face[234])
        nr, _ = detector.findDistance(nose, face[454])
        yaw_face = nl / nr if nr else 1.0
        yaw_vals.append(yaw_face)
        if len(pitch_vals) >= 30: break

    pct = int(100 * len(pitch_vals) / 30)
    cv2.rectangle(img, (50, 50), (450, 80), (50, 50, 50), -1)
    cv2.rectangle(img, (50, 50), (50 + int(4 * pct), 80), (0, 255, 255), -1)
    cv2.putText(img, f"CALIBRATING: {pct}%", (60, 72), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow("DMS Monitor", img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

EAR_THRESH = sum(ear_vals) / max(len(ear_vals), 1)
PITCH_THRESH = sum(pitch_vals) / max(len(pitch_vals), 1)
YAW_BASE = sum(yaw_vals) / max(len(yaw_vals), 1)
YAW_MIN, YAW_MAX = YAW_BASE * 0.5, YAW_BASE * 1.8

# ============================================================
# MAIN LOOP
# ============================================================
while True:
    success, img = cap.read()
    if not success: continue
    h, w, _ = img.shape
    
    # Logic: Steering
    if keyboard.is_pressed("left"): current_angle = -180
    elif keyboard.is_pressed("right"): current_angle = 180
    else: current_angle = 0 # Added centered return for cleaner UI
    
    delta = abs(current_angle - prev_angle)
    angle_changes.append(delta)
    prev_angle = current_angle
    volatility = sum(angle_changes)

    img, faces = detector.findFaceMesh(img, draw=True)
    current_frame_risk = 0
    warnings = []

    if faces:
        face = faces[0]
        # Landmarks & Calcs
        l_vert, _ = detector.findDistance(face[159], face[145])
        l_horz, _ = detector.findDistance(face[130], face[133])
        r_vert, _ = detector.findDistance(face[386], face[374])
        r_horz, _ = detector.findDistance(face[362], face[263])
        avg_ear = ((l_vert / l_horz if l_horz else 0.0) + (r_vert / r_horz if r_horz else 0.0)) / 2.0
        m_vert, _ = detector.findDistance(face[11], face[16])
        m_horz, _ = detector.findDistance(face[78], face[308])
        mar = m_vert / m_horz if m_horz else 0.0
        nose, chin, forehead = face[1], face[152], face[10]
        nc, _ = detector.findDistance(nose, chin)
        nf, _ = detector.findDistance(nose, forehead)
        pitch = nc / nf if nf else 0.0
        nl, _ = detector.findDistance(nose, face[234])
        nr, _ = detector.findDistance(nose, face[454])
        yaw_face = nl / nr if nr else 1.0
        leo, reo = face[33], face[263]
        roll = math.degrees(math.atan2(reo[1] - leo[1], reo[0] - leo[0]))

        # State detection logic
        if avg_ear < 0.9 * EAR_THRESH: sleep_frames += 1
        else: sleep_frames = max(0, sleep_frames - 3)
        if mar > MAR_THRESH: yawn_frames += 0
        else: yawn_frames = 0
        if yawn_frames > ALARM_FRAMES_YAWN:
            yawn_numbers += 0
            yawn_frames = 0
        if ((pitch < 0.67 * PITCH_THRESH or pitch > 3 * PITCH_THRESH) or 
            (yaw_face < YAW_MIN or yaw_face > YAW_MAX) or abs(roll) > ROLL_THRESH):
            wobble_frames += 1
        else: wobble_frames = max(0, wobble_frames - 2)
    else:
        avg_ear = mar = pitch = yaw_face = roll = 0.0

    # Risk Accumulation logic
    if sleep_frames > ALARM_FRAMES_SLEEP:
        warnings.append("DANGER: DRIVER ASLEEP")
        current_frame_risk += WEIGHT_SLEEP
    if wobble_frames > ALARM_FRAMES_WOBBLE:
        warnings.append("WARNING: HEAD DISTRACTED")
        current_frame_risk += WEIGHT_WOBBLE
    if yawn_numbers > 5:
        warnings.append("WARNING: EXCESSIVE YAWNING")
        current_frame_risk += WEIGHT_YAWN
    if volatility > 2000:
        warnings.append("DANGER: ERRATIC STEERING")
        current_frame_risk += WEIGHT_STEERING

    if not auto_engaged: risk_score += current_frame_risk
    risk_timer += 1
    if risk_timer > RISK_RESET_FRAMES:
        if not auto_engaged: risk_score = 0
        yawn_numbers = 0
        risk_timer = 0
    if risk_score > RISK_THRESHOLD: auto_engaged = True

    manual_override = False
    if keyboard.is_pressed("x"):
        auto_engaged, risk_score, risk_timer, yawn_numbers, sleep_frames, wobble_frames = False, 0, 0, 0, 0, 0
        manual_override = True

    # ============================================================
    # IMPROVED UI RENDERING
    # ============================================================
    ACCENT = (0, 0, 255) if auto_engaged else (0, 255, 0)
    
    # 1. Header Overlay
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (20, 20, 20), -1)
    img = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

    # 2. Mode & Risk Bar
    mode_str = "AUTOPILOT ACTIVE" if auto_engaged else "MANUAL MONITORING"
    cv2.putText(img, mode_str, (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.9, ACCENT, 2)
    
    # Risk Bar Background
    cv2.rectangle(img, (25, 55), (275, 70), (100, 100, 100), 1)
    risk_w = int((min(risk_score, RISK_THRESHOLD) / RISK_THRESHOLD) * 250)
    cv2.rectangle(img, (25, 55), (25 + risk_w, 70), ACCENT, -1)
    cv2.putText(img, f"RISK: {risk_score}/{RISK_THRESHOLD}", (285, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 3. Telemetry Panel (Left)
    panel_y = 130
    telemetry = [("EAR", avg_ear), ("MAR", mar), ("PITCH", pitch), ("YAW", yaw_face), ("ROLL", roll)]
    for label, val in telemetry:
        cv2.putText(img, f"{label}:", (25, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"{val:.2f}" if label != "ROLL" else f"{int(val)}", (85, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        panel_y += 25

    # 4. Warnings (Central Alert)
    for i, text in enumerate(warnings):
        tw, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2)[0]
        tx = (w // 2) - (tw // 2)
        cv2.rectangle(img, (tx - 10, 110 + i * 35), (tx + tw + 10, 140 + i * 35), (0, 0, 255), -1)
        cv2.putText(img, text, (tx, 132 + i * 35), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    # 5. Steering Wheel HUD
    wheel_center = (w - 80, h - 80)
    cv2.circle(img, wheel_center, 60, (255, 255, 255), 2)
    cv2.circle(img, wheel_center, 65, (50, 50, 50), 1)
    
    theta = math.radians(current_angle - 90)
    ex, ey = int(wheel_center[0] + 60 * math.cos(theta)), int(wheel_center[1] + 60 * math.sin(theta))
    cv2.line(img, wheel_center, (ex, ey), ACCENT, 4)
    cv2.circle(img, wheel_center, 8, ACCENT, -1)
    cv2.putText(img, f"VOL: {volatility}", (w - 150, h - 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # 6. Footer
    cv2.putText(img, "[X] EMERGENCY RESET  |  [Q] QUIT", (25, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    # State Sync
    state = default_state() # Update with current values
    state.update({"auto_engaged": auto_engaged, "risk_score": risk_score, "volatility": volatility, "warnings": warnings})
    write_state(state)

    cv2.imshow("DMS Monitor", img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()