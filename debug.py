import csv
import math
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import fsds


# ============================================================
# CONFIG
# ============================================================
PATH_CSV = pathlib.Path(__file__).parent / "path.csv"
VEHICLE_NAME = "FSCar"
UPDATE_HZ = 10.0


# ============================================================
# UTILS
# ============================================================
def quat_to_yaw(q):
    siny = 2.0 * (q.w_val * q.z_val + q.x_val * q.y_val)
    cosy = 1.0 - 2.0 * (q.y_val ** 2 + q.z_val ** 2)
    return math.atan2(siny, cosy)


def load_path_csv(path_file):
    pts = []
    with open(path_file, "r", newline="") as f:
        reader = csv.reader(f)
        fire = next(reader, None)

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


def get_vehicle_state(client, vehicle_name):
    state = client.getCarState(vehicle_name)
    kin = state.kinematics_estimated

    x = kin.position.x_val
    y = kin.position.y_val
    yaw = quat_to_yaw(kin.orientation)
    speed = math.hypot(kin.linear_velocity.x_val, kin.linear_velocity.y_val)

    return x, y, yaw, speed


def nearest_path_index(path, x, y):
    d2 = (path[:, 0] - x) ** 2 + (path[:, 1] - y) ** 2
    return int(np.argmin(d2))


# ============================================================
# MAIN
# ============================================================
def main():
    print("Loading path...")
    path = load_path_csv(PATH_CSV)
    print(f"Loaded {len(path)} points from {PATH_CSV}")

    print("Connecting to FSDS...")
    client = fsds.FSDSClient()
    client.confirmConnection()
    print("Connected.")

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))

    while True:
        try:
            x, y, yaw, speed = get_vehicle_state(client, VEHICLE_NAME)

            idx = nearest_path_index(path, x, y)
            nx, ny = path[idx]
            err = math.hypot(nx - x, ny - y)

            ax.clear()

            # Path
            ax.plot(path[:, 0], path[:, 1], label="path.csv", linewidth=2)

            # Start / end
            ax.scatter(path[0, 0], path[0, 1], s=100, marker="o", label="Path Start")
            ax.scatter(path[-1, 0], path[-1, 1], s=100, marker="x", label="Path End")

            # Car position
            ax.scatter(x, y, s=120, marker="^", label="Car Position")

            # Nearest path point
            ax.scatter(nx, ny, s=100, marker="s", label="Nearest Path Point")

            # Line from car to nearest point
            ax.plot([x, nx], [y, ny], linestyle="--", linewidth=1)

            # Heading arrow
            arrow_len = 2.0
            ax.arrow(
                x,
                y,
                arrow_len * math.cos(yaw),
                arrow_len * math.sin(yaw),
                head_width=0.6,
                head_length=0.8,
                length_includes_head=True
            )

            ax.set_title(
                f"FSDS Path Debug\n"
                f"Car=({x:.2f}, {y:.2f})  "
                f"Nearest=({nx:.2f}, {ny:.2f})  "
                f"Err={err:.2f} m  "
                f"Speed={speed:.2f} m/s  "
                f"Idx={idx}"
            )
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.axis("equal")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend(loc="best")

            plt.pause(1.0 / UPDATE_HZ)

            print(
                f"car=({x:.2f},{y:.2f}) "
                f"nearest=({nx:.2f},{ny:.2f}) "
                f"err={err:.2f}m idx={idx} speed={speed:.2f}"
            )

        except KeyboardInterrupt:
            print("\nStopped by user.")
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()