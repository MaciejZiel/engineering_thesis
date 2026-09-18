import socket
import unittest

from vision_robot_arm.robot.robotiq_socket import (
    FLUSH_REQUEST,
    GripperError,
    RobotiqSocketGripper,
    percent_to_register,
)


class FakeDaemon:
    """The URCap's text server: SET -> ack, GET VAR -> 'VAR value'.

    With `lagging=True` it behaves like the controller in the lab: the reply to
    a request is only flushed when the next request arrives.
    """

    def __init__(
        self, status: int = 3, stray: bytes = b"", lagging: bool = False, bare_ack: bool = False
    ) -> None:
        self.registers = {"STA": status, "POS": 3, "OBJ": 3, "FLT": 0, "ACT": 1}
        self.sent: list[str] = []
        self.pending = stray  # bytes ready to be read
        self.held = b""  # a lagging daemon's not-yet-flushed reply
        self.lagging = lagging
        self.bare_ack = bare_ack  # the real server: `ack` without a newline
        self.timeout = None
        self.closed = False
        self.reject = False

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def sendall(self, data: bytes) -> None:
        for request in data.decode("ascii").split("\n"):
            if request:
                self._handle(request)

    def _handle(self, request: str) -> None:
        self.sent.append(request)
        if self.lagging:
            self.pending += self.held
            self.held = b""
        words = request.split()
        if words[0] == "SET":
            terminator = b"" if self.bare_ack else b"\n"
            if self.reject:
                reply = b"nack" + terminator
            else:
                for name, value in zip(words[1::2], words[2::2]):
                    self.registers[name] = int(value)
                    if name == "ACT" and int(value) == 1:
                        self.registers["STA"] = 3
                reply = b"ack" + terminator
        else:
            reply = f"{words[1]} {self.registers[words[1]]}\n".encode("ascii")
        if self.lagging:
            self.held = reply
        else:
            self.pending += reply

    def recv(self, size: int) -> bytes:
        if not self.pending:
            raise socket.timeout("timed out")
        chunk, self.pending = self.pending[:size], self.pending[size:]
        return chunk

    def close(self) -> None:
        self.closed = True

    @property
    def requests(self) -> list[str]:
        """What the driver actually asked for, without the flush follow-ups."""
        return self.sent[0::2]


def gripper(daemon: FakeDaemon) -> RobotiqSocketGripper:
    return RobotiqSocketGripper(
        "10.20.3.20", connector=lambda host, port, timeout: daemon, sleep=lambda _: None
    )


class RobotiqSocketGripperTests(unittest.TestCase):
    def test_every_request_carries_a_flush_follow_up(self) -> None:
        daemon = FakeDaemon()

        gripper(daemon).connect()

        self.assertEqual(daemon.sent, ["GET STA", FLUSH_REQUEST])
        self.assertEqual(daemon.timeout, 1.0)

    def test_the_lab_daemon_that_replies_one_request_late_still_works(self) -> None:
        daemon = FakeDaemon(lagging=True)
        hand = gripper(daemon)

        hand.connect()
        hand.command(True, 80, 50)
        status = hand.status()

        self.assertEqual(status["POS"], 255)
        self.assertEqual(daemon.requests[1], "SET POS 255 SPE 204 FOR 128 GTO 1")

    def test_the_lab_daemon_writes_ack_without_a_newline(self) -> None:
        # Exactly what the cell produced: 'ackSTA 3\n' on one read.
        daemon = FakeDaemon(lagging=True, bare_ack=True)
        hand = gripper(daemon)
        hand.connect()

        hand.command(True, 80, 50)  # must not be mistaken for a rejection

        self.assertEqual(daemon.registers["POS"], 255)
        self.assertEqual(hand.get("POS"), 255)

    def test_a_bare_nack_is_still_a_rejection(self) -> None:
        daemon = FakeDaemon(lagging=True, bare_ack=True)
        hand = gripper(daemon)
        hand.connect()
        daemon.reject = True

        with self.assertRaisesRegex(GripperError, "rejected"):
            hand.command(True, 80, 50)

    def test_connect_checks_activation_without_resetting_an_active_gripper(self) -> None:
        daemon = FakeDaemon(status=3)

        gripper(daemon).connect()

        self.assertEqual(daemon.requests, ["GET STA"])

    def test_connect_activates_an_inactive_gripper_and_waits_for_it(self) -> None:
        daemon = FakeDaemon(status=0)

        gripper(daemon).connect()

        self.assertEqual(daemon.requests[:2], ["GET STA", "SET ACT 1"])
        self.assertEqual(daemon.requests[-1], "GET STA")

    def test_a_close_command_is_one_set_line_with_scaled_registers(self) -> None:
        daemon = FakeDaemon()
        hand = gripper(daemon)
        hand.connect()

        hand.command(True, speed_percent=80, force_percent=50)
        hand.command(False, speed_percent=100, force_percent=0)

        self.assertEqual(daemon.requests[-2], "SET POS 255 SPE 204 FOR 128 GTO 1")
        self.assertEqual(daemon.requests[-1], "SET POS 0 SPE 255 FOR 0 GTO 1")
        self.assertEqual(daemon.registers["POS"], 0)

    def test_percent_scaling_is_clamped(self) -> None:
        self.assertEqual(percent_to_register(-5), 0)
        self.assertEqual(percent_to_register(50), 128)
        self.assertEqual(percent_to_register(250), 255)

    def test_late_and_concatenated_replies_are_sorted_out(self) -> None:
        # The daemon answered a previous GET late; its reply arrives first.
        daemon = FakeDaemon(status=3, stray=b"ACT 1\n")
        hand = gripper(daemon)
        hand.connect()

        self.assertEqual(hand.status(), {"STA": 3, "POS": 3, "OBJ": 3, "FLT": 0})

    def test_a_rejected_command_raises(self) -> None:
        daemon = FakeDaemon()
        hand = gripper(daemon)
        hand.connect()
        daemon.reject = True

        with self.assertRaisesRegex(GripperError, "rejected"):
            hand.command(True, 80, 50)

    def test_silence_raises_instead_of_hanging(self) -> None:
        daemon = FakeDaemon()
        hand = gripper(daemon)
        hand.connect()
        daemon.pending = b""

        def mute(data: bytes) -> None:
            daemon.sent.extend(line for line in data.decode().split("\n") if line)

        daemon.sendall = mute
        with self.assertRaisesRegex(GripperError, "no reply"):
            hand.command(True, 80, 50)

    def test_unreachable_host_is_a_gripper_error(self) -> None:
        def refuse(host: str, port: int, timeout: float):
            raise OSError("no route to host")

        hand = RobotiqSocketGripper("10.20.3.20", connector=refuse)
        with self.assertRaisesRegex(GripperError, "unreachable"):
            hand.connect()
        self.assertFalse(hand.connected)

    def test_close_releases_the_socket(self) -> None:
        daemon = FakeDaemon()
        hand = gripper(daemon)
        hand.connect()

        hand.close()

        self.assertTrue(daemon.closed)
        self.assertFalse(hand.connected)
        with self.assertRaisesRegex(GripperError, "not connected"):
            hand.command(True, 80, 50)


if __name__ == "__main__":
    unittest.main()
