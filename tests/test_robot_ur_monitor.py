import math
import unittest

from vision_robot_arm.robot.config import RobotConfig
from vision_robot_arm.robot.ur_dashboard import DashboardStatus
from vision_robot_arm.robot.ur_monitor import MonitorConnectionError, URMonitorBackend


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now


class FakeRtde:
    def __init__(self, sample=None, connects=True):
        self.sample = sample
        self.connects = connects
        self.last_error = "disabled"
        self.controller_version = (5, 22, 1, 42)
        self.closed = False

    def connect(self):
        return self.connects

    def read(self):
        sample, self.sample = self.sample, None
        return sample

    def close(self):
        self.closed = True


def sample():
    return {
        "actual_q": (0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0),
        "actual_qd": (0.0,) * 6,
        "actual_TCP_pose": (0.0,) * 6,
        "actual_TCP_speed": (0.0,) * 6,
        "robot_mode": 7,
        "safety_status": 2,
        "speed_scaling": 0.1,
        "target_speed_fraction": 0.1,
        "runtime_state": 2,
    }


class MonitorTests(unittest.TestCase):
    def config(self):
        return RobotConfig(backend="ur", right_host="10.0.0.2")

    def test_reports_actual_joint_state_without_motion_api(self):
        clock = Clock()
        rtde = FakeRtde(sample())
        backend = URMonitorBackend(
            self.config(),
            rtde_factory=lambda host, port: rtde,
            status_query=lambda host, port: DashboardStatus("RUNNING", "REDUCED", False),
            clock=clock,
        )

        state = backend.robot_state()

        self.assertAlmostEqual(state.arm("right").joints["shoulder"], -90.0)
        self.assertEqual(
            state.arm("right").joints, state.arm("right").targets
        )
        self.assertFalse(hasattr(backend, "send"))
        self.assertIn("MONITOR-ONLY", backend.status_lines()[0])

    def test_stale_feedback_is_not_presented_as_live(self):
        clock = Clock()
        rtde = FakeRtde(sample())
        backend = URMonitorBackend(
            self.config(),
            rtde_factory=lambda host, port: rtde,
            status_query=lambda host, port: None,
            clock=clock,
        )
        self.assertIsNotNone(backend.robot_state())
        clock.now += 0.6
        self.assertIsNone(backend.robot_state())

    def test_failed_rtde_connection_closes_and_fails(self):
        rtde = FakeRtde(connects=False)
        with self.assertRaises(MonitorConnectionError):
            URMonitorBackend(
                self.config(),
                rtde_factory=lambda host, port: rtde,
                status_query=lambda host, port: None,
            )
        self.assertTrue(rtde.closed)


if __name__ == "__main__":
    unittest.main()
