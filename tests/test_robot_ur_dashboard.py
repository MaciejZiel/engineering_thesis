import unittest

from vision_robot_arm.robot.ur_dashboard import DashboardStatus, query_status


class ScriptedDashboard:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.asked: list[str] = []
        self.closed = False

    def sendall(self, data: bytes) -> None:
        self.asked.append(data.decode("ascii").strip())

    def recv(self, size: int) -> bytes:
        if not self._replies:
            return b""
        return (self._replies.pop(0) + "\n").encode("ascii")

    def close(self) -> None:
        self.closed = True


def dashboard(replies: list[str]) -> tuple[ScriptedDashboard, object]:
    connection = ScriptedDashboard(replies)
    return connection, lambda host, port: connection


READY = [
    "Connected: Universal Robots Dashboard Server",
    "Robotmode: RUNNING",
    "Safetystatus: NORMAL",
    "true",
]


class QueryTests(unittest.TestCase):
    def test_reads_mode_safety_and_remote_control(self) -> None:
        connection, connector = dashboard(READY)

        status = query_status("10.0.0.2", connector=connector)

        self.assertEqual(status, DashboardStatus("RUNNING", "NORMAL", True))
        self.assertEqual(connection.asked, ["robotmode", "safetystatus", "is in remote control"])
        self.assertTrue(connection.closed)

    def test_local_control_is_detected(self) -> None:
        _, connector = dashboard(
            ["Connected", "Robotmode: RUNNING", "Safetystatus: NORMAL", "false"]
        )

        self.assertIs(query_status("10.0.0.2", connector=connector).remote_control, False)

    def test_unknown_remote_reply_stays_unknown(self) -> None:
        _, connector = dashboard(
            ["Connected", "Robotmode: IDLE", "Safetystatus: NORMAL", "Unsupported command"]
        )

        status = query_status("10.0.0.2", connector=connector)

        self.assertIsNone(status.remote_control)
        self.assertEqual(status.robot_mode, "IDLE")

    def test_unreachable_dashboard_returns_none(self) -> None:
        def refuse(host: str, port: int) -> None:
            raise OSError("refused")

        self.assertIsNone(query_status("10.0.0.2", connector=refuse))


class BlockingProblemTests(unittest.TestCase):
    def test_ready_robot_has_no_problem(self) -> None:
        self.assertIsNone(DashboardStatus("RUNNING", "NORMAL", True).blocking_problem())
        self.assertIsNone(DashboardStatus("RUNNING", "REDUCED", True).blocking_problem())

    def test_local_control_is_reported_first(self) -> None:
        problem = DashboardStatus("POWER_OFF", "NORMAL", False).blocking_problem()

        self.assertIn("Local control", problem)

    def test_powered_down_robot_is_reported(self) -> None:
        problem = DashboardStatus("POWER_OFF", "NORMAL", True).blocking_problem()

        self.assertIn("POWER_OFF", problem)
        self.assertIn("brakes", problem)

    def test_protective_stop_is_reported(self) -> None:
        problem = DashboardStatus("RUNNING", "PROTECTIVE_STOP", True).blocking_problem()

        self.assertIn("PROTECTIVE_STOP", problem)

    def test_unknown_status_does_not_block(self) -> None:
        self.assertIsNone(DashboardStatus().blocking_problem())

    def test_describe_is_readable(self) -> None:
        self.assertEqual(
            DashboardStatus("RUNNING", "NORMAL", True).describe(),
            "mode RUNNING, safety NORMAL, control remote",
        )


if __name__ == "__main__":
    unittest.main()
