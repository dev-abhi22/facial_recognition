import cv2
import math
import numpy as np
from collections import deque
import keyboard
from cvzone.FaceMeshModule import FaceMeshDetector

# --- Camera Setup ---
cap = cv2.VideoCapture(0)
detector = FaceMeshDetector(maxFaces=1)

# --- Face Thresholds & Counters ---
# EAR_THRESH = 0.3      
MAR_THRESH = 0.3     
# PITCH_THRESH = 0.5   
ROLL_THRESH = 30     
# YAW_MIN = 0.5          
# YAW_MAX = 2.0       
ALARM_FRAMES_1 = 150
ALARM_FRAMES_2 = 60
# 30 frames = 1 sec

sleep_frames = 0
yawn_frames = 0
yawn_numbers = 0 
wobble_frames = 0
timer = 0

# --- Steering Variables ---
TURN_SPEED = 8       
MAX_ANGLE = 180      
BUFFER_SIZE = 45 
angle_changes = deque(maxlen=BUFFER_SIZE)
current_angle = 0
prev_angle = 0
auto = False

ear = []
ya = []
pi = []
warnings = []

while True:
    success, img = cap.read()
    if not success:
        break

    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]
        
        l_vert, _ = detector.findDistance(face[159], face[145])
        l_horz, _ = detector.findDistance(face[130], face[133])
        left_ear = l_vert / l_horz if l_horz != 0 else 0
        
        r_vert, _ = detector.findDistance(face[386], face[374])
        r_horz, _ = detector.findDistance(face[362], face[263])
        right_ear = r_vert / r_horz if r_horz != 0 else 0
        
        avg_ear = (left_ear + right_ear) / 2

        ear.append(avg_ear)

        nose, chin, forehead = face[1], face[152], face[10]
        left_cheek, right_cheek = face[234], face[454]
        
        nose_chin, _ = detector.findDistance(nose, chin)
        nose_forehead, _ = detector.findDistance(nose, forehead)
        pitch = nose_chin / nose_forehead if nose_forehead != 0 else 0

        pi.append(pitch)


        
        nose_left, _ = detector.findDistance(nose, left_cheek)
        nose_right, _ = detector.findDistance(nose, right_cheek)
        yaw = nose_left / nose_right if nose_right != 0 else 1

        ya.append(yaw)

        if len(pi) > 30:
            break 

        cv2.imshow("Combined Advanced DMS", img)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
          break 
        

EAR_THRESH = sum(ear)/len(ear)
PITCH_THRESH = sum(pi)/len(pi)
YAW = sum(ya)/len(ya)
YAW_MIN = YAW*0.2      
YAW_MAX = YAW*1.8


while True:
    success, img = cap.read()
    if not success:
        break
        
    # Get image dimensions to place the steering wheel in the bottom right corner
    h, w, _ = img.shape
    WHEEL_CENTER = (w - 80, h - 80)
    WHEEL_RADIUS = 60

    # 1. Read Keyboard
    if keyboard.is_pressed('left'):
        current_angle -= TURN_SPEED
    if keyboard.is_pressed('right'):
        current_angle += TURN_SPEED
        
    current_angle = max(-MAX_ANGLE, min(MAX_ANGLE, current_angle))
    
    # 2. Calculate Volatility
    delta = abs(current_angle - prev_angle)
    angle_changes.append(delta)
    prev_angle = current_angle
    volatility = sum(angle_changes)

    img, faces = detector.findFaceMesh(img, draw=True)

    if faces:
        face = faces[0]
        
        # 1. EAR (Eyes)
        l_vert, _ = detector.findDistance(face[159], face[145])
        l_horz, _ = detector.findDistance(face[130], face[133])
        left_ear = l_vert / l_horz if l_horz != 0 else 0
        
        r_vert, _ = detector.findDistance(face[386], face[374])
        r_horz, _ = detector.findDistance(face[362], face[263])
        right_ear = r_vert / r_horz if r_horz != 0 else 0
        
        avg_ear = (left_ear + right_ear) / 2
        
        # 2. MAR (Mouth - Using Outer Lips 11 and 16)
        m_vert, _ = detector.findDistance(face[11], face[16])
        m_horz, _ = detector.findDistance(face[78], face[308])
        mar = m_vert / m_horz if m_horz != 0 else 0

        # 3. Head Pose
        nose, chin, forehead = face[1], face[152], face[10]
        left_cheek, right_cheek = face[234], face[454]
        
        nose_chin, _ = detector.findDistance(nose, chin)
        nose_forehead, _ = detector.findDistance(nose, forehead)
        pitch = nose_chin / nose_forehead if nose_forehead != 0 else 0
        
        nose_left, _ = detector.findDistance(nose, left_cheek)
        nose_right, _ = detector.findDistance(nose, right_cheek)
        yaw = nose_left / nose_right if nose_right != 0 else 1
        
        left_eye_outer, right_eye_outer = face[33], face[263]
        delta_y = right_eye_outer[1] - left_eye_outer[1]
        delta_x = right_eye_outer[0] - left_eye_outer[0]
        roll = math.degrees(math.atan2(delta_y, delta_x))

        # 4. Update Counters
        if avg_ear < 0.8*EAR_THRESH: sleep_frames += 1 #applied percentage logic, more reliable 
        else: sleep_frames = 0
            
        if mar > MAR_THRESH: yawn_frames += 1
        else: yawn_frames = 0

            # 3. Check Head Wobbling/Dropping (The Sticky Counter)
        if (pitch < 0.67*PITCH_THRESH or pitch > 3*PITCH_THRESH) or (yaw < YAW_MIN or yaw > YAW_MAX) or abs(roll) > ROLL_THRESH:
            wobble_frames += 1
        else:
            # Subtract 2 instead of resetting to 0
            wobble_frames -= 2
            # Make sure the counter never drops below 0
            if wobble_frames < 0:
                wobble_frames = 0

    warning_color = (0, 255, 0) # Default Green
    wheel_color = (255, 255, 255) # Default White
    
    # Check Warnings (We use a list so we can show multiple warnings at once)
    active_warnings = []
    
    
    if sleep_frames > ALARM_FRAMES_2:
        active_warnings.append("DANGER: DRIVER ASLEEP!")
        warning_color = (0, 0, 255)
        
    if wobble_frames > ALARM_FRAMES_1:
        active_warnings.append("WARNING: HEAD DISTRACTED!")
        warning_color = (0, 165, 255)

    if yawn_frames > ALARM_FRAMES_2:
        yawn_numbers += 1
        
    if yawn_numbers >(5):
        active_warnings.append("WARNING: YAWNING!")
        warning_color = (0, 255, 255)
        
    if volatility > 180:
        active_warnings.append("DANGER: ERRATIC STEERING!")
        wheel_color = (0, 0, 255) # Turn wheel red

    warnings.extend(active_warnings)
    if len(warnings) > 100:
        auto = True
    if auto == True:#auto pilot mode activated after multiple warnings
        cv2.putText(img, "AUTO-PILOT MODE", (60,80) ,cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    timer += 1
    if timer > 120:
        yawn_numbers = 0
        warnings.clear()
        timer = 0

    # Print active warnings on screen
    for i, text in enumerate(active_warnings):
        cv2.putText(img, text, (20, 200 + (i * 40)), cv2.FONT_HERSHEY_SIMPLEX, 1, warning_color, 3)

    # --- Draw the Steering Wheel (Bottom Right) ---
    cv2.circle(img, WHEEL_CENTER, WHEEL_RADIUS, wheel_color, 4)
    theta_rad = math.radians(current_angle - 90)
    end_x = int(WHEEL_CENTER[0] + WHEEL_RADIUS * math.cos(theta_rad))
    end_y = int(WHEEL_CENTER[1] + WHEEL_RADIUS * math.sin(theta_rad))
    cv2.line(img, WHEEL_CENTER, (end_x, end_y), wheel_color, 4)
    cv2.circle(img, WHEEL_CENTER, 10, wheel_color, cv2.FILLED)
    
    # Print Steering Telemetry above the wheel
    vol_color = (0, 255, 255) if volatility > 80 else (0, 255, 0)
    cv2.putText(img, f"Vol: {volatility}", (w - 140, h - 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vol_color, 2)

    # --- DEBUG DASHBOARD ---
    cv2.putText(img, f"Pitch: {pitch:.2f} | Yaw: {yaw:.2f} | Roll: {int(roll)} | EAR: {avg_ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show the final merged frame
    cv2.imshow("Combined Advanced DMS", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()