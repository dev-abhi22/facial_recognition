import cv2
import math
from cvzone.FaceMeshModule import FaceMeshDetector

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
cap = cv2.VideoCapture(0) # 0 for webcam, or "video.mp4"
detector = FaceMeshDetector(maxFaces=1)

# --- Frame Counters (The "Rolling Buffers") ---
# These count how many frames in a row a condition has been met
sleep_frames = 0
yawn_frames = 0
wobble_frames = 0

# --- Thresholds ---
# Tweak these numbers based on your specific camera setup
EAR_THRESH = 0.3     # Below this, eyes are closed
MAR_THRESH = 0.3      # Above this, mouth is yawning
PITCH_THRESH = 0.6     # Below this, head is dropped forward
ROLL_THRESH = 30       # Above this (degrees), head is tilted sideways
YAW_MIN = 0.5          # Below this, looking hard right
YAW_MAX = 2.0          # Above this, looking hard left

# How many frames before triggering the alarm? (Assuming ~30 FPS)
# 30 frames = 1 second
ALARM_FRAMES = 150 

# ==========================================
# 2. MAIN PROCESSING LOOP
# ==========================================
while True:
    success, img = cap.read()
    if not success:
        break
        
    img, faces = detector.findFaceMesh(img, draw=True)

    # Initialize a blank warning text for this frame
    warning_text = ""
    warning_color = (0, 255, 0) # Green (Safe)

    if faces:
        face = faces[0]
        
        # ------------------------------------------
        # A. EAR (Eye Aspect Ratio) - Sleep Detection
        # ------------------------------------------
        # Left Eye (Vertical: 159-145, Horizontal: 130-133)
        l_vert, _ = detector.findDistance(face[159], face[145])
        l_horz, _ = detector.findDistance(face[130], face[133])
        left_ear = l_vert / l_horz if l_horz != 0 else 0
        
        # Right Eye (Vertical: 386-374, Horizontal: 362-263)
        r_vert, _ = detector.findDistance(face[386], face[374])
        r_horz, _ = detector.findDistance(face[362], face[263])
        right_ear = r_vert / r_horz if r_horz != 0 else 0
        
        avg_ear = (left_ear + right_ear) / 2
        
       # ------------------------------------------
        # B. MAR (Mouth Aspect Ratio) - Yawn Detection
        # ------------------------------------------
        # Changed to Outer Lips! (Top: 11, Bottom: 16)
        m_vert, _ = detector.findDistance(face[11], face[16])
        m_horz, _ = detector.findDistance(face[78], face[308])
        mar = m_vert / m_horz if m_horz != 0 else 0

        # ------------------------------------------
        # C. Head Pose - Wobbling / Dropping
        # ------------------------------------------
        nose = face[1]
        chin = face[152]
        forehead = face[10]
        left_cheek = face[234]
        right_cheek = face[454]
        
        # Pitch (Nodding)
        nose_chin, _ = detector.findDistance(nose, chin)
        nose_forehead, _ = detector.findDistance(nose, forehead)
        pitch = nose_chin / nose_forehead if nose_forehead != 0 else 0
        
        # Yaw (Shaking / Looking away)
        nose_left, _ = detector.findDistance(nose, left_cheek)
        nose_right, _ = detector.findDistance(nose, right_cheek)
        yaw = nose_left / nose_right if nose_right != 0 else 1
        
        # Roll (Tilting)
        left_eye_outer = face[33]
        right_eye_outer = face[263]
        delta_y = right_eye_outer[1] - left_eye_outer[1]
        delta_x = right_eye_outer[0] - left_eye_outer[0]
        roll = math.degrees(math.atan2(delta_y, delta_x))

        # ==========================================
        # 3. THE LOGIC & BUFFERS
        # ==========================================
        
        # 1. Check Sleep (Eyes Closed)
        if avg_ear < EAR_THRESH:
            sleep_frames += 1
        else:
            sleep_frames = 0
            
        # 2. Check Yawning
        if mar > MAR_THRESH:
            yawn_frames += 1
        else:
            yawn_frames = 0
            
        # 3. Check Head Wobbling/Dropping (Pitch, Yaw, Roll)
        if pitch < PITCH_THRESH or (yaw < YAW_MIN or yaw > YAW_MAX) or abs(roll) > ROLL_THRESH:
            wobble_frames += 1
        else:
            wobble_frames = 0

        # ==========================================
        # 4. TRIGGER ALARMS (UI Updates)
        # ==========================================
        # If any counter exceeds our 30-frame limit, trigger the alarm
        if sleep_frames > ALARM_FRAMES:
            warning_text = "DANGER: DRIVER ASLEEP!"
            warning_color = (0, 0, 255) # Red
            
        elif wobble_frames > ALARM_FRAMES:
            warning_text = "WARNING: HEAD WOBBLING/DISTRACTED!"
            warning_color = (0, 165, 255) # Orange
            
        elif yawn_frames > ALARM_FRAMES:
            warning_text = "WARNING: DROWSINESS (YAWNING)!"
            warning_color = (0, 255, 255) # Yellow
            
        # Optional: Display live metrics for debugging
        cv2.putText(img, f"EAR: {avg_ear:.2f} | MAR: {mar:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Pitch: {pitch:.2f} | Yaw: {yaw:.2f} | Roll: {int(roll)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display the final warning text if one was triggered
    if warning_text != "":
        cv2.putText(img, warning_text, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.2, warning_color, 4)

    # Show the video feed
    cv2.imshow("Driver Monitoring System", img)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()