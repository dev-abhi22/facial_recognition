import csv
import math
import time
import pathlib

import keyboard
import fsds


# ============================================================
# CONFIG
# ============================================================
VEHICLE_NAME = "FSCar"
OUTPUT_CSV = pathlib.Path(__file__).parent / "recorded_path_test2.csv"

SAMPLE_PERIOD_SEC = 0.5
MIN_POINT_DISTANCE_M = 0.3


# ============================================================
# UTILS
# ============================================================
def quat_to_yaw(q):
    siny = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy = 1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2)
    return math.atan2(siny, cosy)


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def get_vehicle_state(client, vehicle_name):
    state = client.getCarState(vehicle_name)
    kin = state.kinematics_estimated

    x = kin.position.x_val
    y = kin.position.y_val
    yaw = quat_to_yaw(kin.orientation)
    speed = math.hypot(kin.linear_velocity.x_val, kin.linear_velocity.y_val)

    return x, y, yaw, speed


def save_points_to_csv(points, output_csv):
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(points)


# ============================================================
# MAIN
# ============================================================
def main():
    print("Connecting to FSDS...")
    client = fsds.FSDSClient()
    client.confirmConnection()

    # Keep manual control with you
    client.enableApiControl(False, VEHICLE_NAME)

    print("Connected.")
    print()
    print("Controls:")
    print("  s  -> start sampling")
    print("  e  -> stop sampling")
    print("  q  -> save and quit")
    print()

    sampling = False
    points = []
    last_sample_time = 0.0
    last_saved_x = None
    last_saved_y = None

    s_was_down = False
    e_was_down = False
    q_was_down = False

    try:
        while True:
            now = time.perf_counter()

            s_down = keyboard.is_pressed("s")
            e_down = keyboard.is_pressed("e")
            q_down = keyboard.is_pressed("q")

            # Start sampling
            if s_down and not s_was_down:
                sampling = True
                print("[REC] Sampling started.")

            # Stop sampling
            if e_down and not e_was_down:
                sampling = False
                print("[REC] Sampling stopped.")

            # Quit
            if q_down and not q_was_down:
                print("[REC] Quit requested.")
                break

            s_was_down = s_down
            e_was_down = e_down
            q_was_down = q_down

            if sampling:
                if now - last_sample_time >= SAMPLE_PERIOD_SEC:
                    x, y, yaw, speed = get_vehicle_state(client, VEHICLE_NAME)
                    last_sample_time = now

                    should_save = False

                    if last_saved_x is None or last_saved_y is None:
                        should_save = True
                    else:
                        d = distance(last_saved_x, last_saved_y, x, y)
                        if d >= MIN_POINT_DISTANCE_M:
                            should_save = True

                    if should_save:
                        points.append((x, y))
                        last_saved_x = x
                        last_saved_y = y
                        print(
                            f"[REC] Saved #{len(points):04d} "
                            f"x={x:.3f}, y={y:.3f}, speed={speed:.2f} m/s"
                        )
                    else:
                        d = distance(last_saved_x, last_saved_y, x, y)
                        print(
                            f"[REC] Skipped point "
                            f"(too close: {d:.3f} m < {MIN_POINT_DISTANCE_M:.3f} m)"
                        )

            time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n[REC] Interrupted by user.")

    finally:
        if points:
            save_points_to_csv(points, OUTPUT_CSV)
            print(f"[REC] Saved {len(points)} points to: {OUTPUT_CSV}")
        else:
            print("[REC] No points were recorded.")

        try:
            client.enableApiControl(False, VEHICLE_NAME)
        except Exception:
            pass


if __name__ == "__main__":
    main()