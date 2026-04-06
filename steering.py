import cv2
import numpy as np
import math
from collections import deque
import keyboard

# ==========================================
# 1. CONFIGURATION
# ==========================================
WIDTH, HEIGHT = 640, 480
CENTER = (WIDTH // 2, HEIGHT // 2)
WHEEL_RADIUS = 150

TURN_SPEED = 8       # How many degrees the wheel turns per frame when key is pressed
MAX_ANGLE = 180      # Lock the wheel from spinning infinitely

# The Volatility Buffer
# We will remember the wheel movement over the last 30 frames (approx 1 second)
BUFFER_SIZE = 45
angle_changes = deque(maxlen=BUFFER_SIZE)

# State Variables
current_angle = 0
prev_angle = 0

# ==========================================
# 2. MAIN LOOP
# ==========================================
while True:
    # Create a blank black canvas for our dashboard
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    
    # --- A. Handle Keyboard Input ---
    # waitKey(30) means the loop runs roughly at 30 Frames Per Second
    key = cv2.waitKey(30) & 0xFF
    
    if keyboard.is_pressed('left'):
        current_angle -= TURN_SPEED
    elif keyboard.is_pressed('right'):
        current_angle += TURN_SPEED
    elif keyboard.is_pressed('q'):
        break
        
    # Clamp the angle so it stays between -180 and 180 degrees
    current_angle = max(-MAX_ANGLE, min(MAX_ANGLE, current_angle))
    
    # --- B. Calculate Rate of Change (Volatility) ---
    # How much did the wheel move in this exact frame?
    delta = abs(current_angle - prev_angle)
    
    # Add this frame's movement to our rolling memory
    angle_changes.append(delta)
    prev_angle = current_angle
    
    # Sum up all the movements over the last second
    volatility = sum(angle_changes)
    
    # --- C. Warning Logic ---
    warning_text = ""
    color = (255, 255, 255) # White wheel (Normal)
    
    # If the driver has moved the wheel a total of 120+ degrees back and forth in 1 second
    if volatility > 180:
        warning_text = "DANGER: ERRATIC STEERING!"
        color = (0, 0, 255) # Red wheel (Warning)
        
    # --- D. Draw the Steering Wheel ---
    # Draw the outer rim
    cv2.circle(img, CENTER, WHEEL_RADIUS, color, 8)
    
    # Calculate the math to draw the center spoke (the line)
    # We subtract 90 degrees so an angle of '0' points straight UP
    theta_rad = math.radians(current_angle - 90)
    end_x = int(CENTER[0] + WHEEL_RADIUS * math.cos(theta_rad))
    end_y = int(CENTER[1] + WHEEL_RADIUS * math.sin(theta_rad))
    
    # Draw the center spoke
    cv2.line(img, CENTER, (end_x, end_y), color, 8)
    # Draw the center hub
    cv2.circle(img, CENTER, 20, color, cv2.FILLED)
    
    # --- E. Display the Dashboard Text ---
    cv2.putText(img, f"Angle: {current_angle} deg", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    # Display the volatility score. We use a dynamic color that turns red as it gets higher
    vol_color = (0, 255, 255) if volatility > 80 else (0, 255, 0)
    cv2.putText(img, f"Volatility: {volatility}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, vol_color, 2)
    
    cv2.putText(img, "Press 'A' (Left) / 'D' (Right) | 'Q' to Quit", (10, HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # If the warning triggered, print it in huge red letters
    if warning_text:
        cv2.putText(img, warning_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # Show the final dashboard
    cv2.imshow("Telemetry Simulator", img)

cv2.destroyAllWindows()