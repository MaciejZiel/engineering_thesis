"""Teach named UR joint positions and save them to robot_positions.json."""

import json
import rtde_receive

ROBOT_IP = "10.20.3.136"
OUTPUT_FILE = "robot_positions.json"


def rad_to_deg_list(q):
    """Convert joint angles from radians to rounded degrees for readability."""
    return [round(v * 180.0 / 3.1415926535, 1) for v in q]


def main():
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    # Store both machine-friendly radians and human-readable degrees.
    positions = {}

    print("\n=== UR7e POSITION TEACHER ===")
    print("Move robot manually with freedrive or pendant.")
    print("Press ENTER to capture current joints.")
    print("Type q to finish.\n")

    while True:
        name = input(
            "Position name (example: home, waypoint, transfer): "
        ).strip()

        if name.lower() == "q":
            break

        input(f"Move robot to '{name}' and press ENTER to capture...")

        # Read the current joint configuration from the robot controller.
        joints = rtde_r.getActualQ()

        positions[name] = {
            "joints_rad": joints,
            "joints_deg": rad_to_deg_list(joints),
        }

        print(f"\nSaved position: {name}")
        print("Radians:")
        print(joints)

        print("Degrees:")
        print(rad_to_deg_list(joints))
        print()

    with open(OUTPUT_FILE, "w") as f:
        json.dump(positions, f, indent=4)

    print(f"\nSaved all positions to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
