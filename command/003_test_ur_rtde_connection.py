"""Verify RTDE communication and perform a small linear robot motion."""

import rtde_control
import rtde_receive

ROBOT_IP = "10.20.3.136"

try:
    print("Connecting to robot...")

    rtde_c = rtde_control.RTDEControlInterface(ROBOT_IP)
    rtde_r = rtde_receive.RTDEReceiveInterface(ROBOT_IP)

    print("Connected!")

    # Read the current robot state before sending any motion command.
    pose = rtde_r.getActualTCPPose()
    q = rtde_r.getActualQ()

    print("\nCurrent TCP pose:")
    print(pose)

    print("\nJoint positions:")
    print(q)

    # Move 5 cm upward in TCP Z to confirm control commands work.
    print("\nMoving slightly in Z...")

    target = pose.copy()
    target[2] += 0.05  # +5 cm

    rtde_c.moveL(target, 0.1, 0.1)

    print("Move done")

except Exception as e:
    print("ERROR:", e)

finally:
    try:
        rtde_c.disconnect()
    except Exception:
        pass
