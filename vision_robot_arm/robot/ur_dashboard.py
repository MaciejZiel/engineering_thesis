"""Dashboard server checks (port 29999) so a blocked robot fails with a readable message.

Without this, a robot left in Local mode or with the brakes engaged simply ignores the
URScript stream and the arm never moves, with nothing on screen explaining why.
"""

import socket
from dataclasses import dataclass
from typing import Any, Callable

DASHBOARD_PORT = 29999
DASHBOARD_TIMEOUT_S = 2.0
READY_ROBOT_MODE = "RUNNING"
READY_SAFETY_STATUS = ("NORMAL", "REDUCED")
ROBOT_MODE_NAMES = frozenset(
    {
        "NO_CONTROLLER",
        "DISCONNECTED",
        "CONFIRM_SAFETY",
        "BOOTING",
        "POWER_OFF",
        "POWER_ON",
        "IDLE",
        "BACKDRIVE",
        "RUNNING",
    }
)
SAFETY_STATUS_NAMES = frozenset(
    {
        "NORMAL",
        "REDUCED",
        "PROTECTIVE_STOP",
        "RECOVERY",
        "SAFEGUARD_STOP",
        "SYSTEM_EMERGENCY_STOP",
        "ROBOT_EMERGENCY_STOP",
        "VIOLATION",
        "FAULT",
        "AUTOMATIC_MODE_SAFEGUARD_STOP",
        "SYSTEM_THREE_POSITION_ENABLING_STOP",
    }
)

Connector = Callable[[str, int], Any]


@dataclass(frozen=True)
class DashboardStatus:
    robot_mode: str | None = None
    safety_status: str | None = None
    remote_control: bool | None = None

    def describe(self) -> str:
        mode = self.robot_mode or "unknown"
        safety = self.safety_status or "unknown"
        remote = {True: "remote", False: "local", None: "unknown"}[self.remote_control]
        return f"mode {mode}, safety {safety}, control {remote}"

    def blocking_problem(self) -> str | None:
        """What stops URScript from moving this robot, in the operator's words."""
        if self.remote_control is False:
            return "the robot is in Local control; switch the teach pendant to Remote Control"
        if self.robot_mode is not None and self.robot_mode != READY_ROBOT_MODE:
            return (
                f"the robot mode is {self.robot_mode}; power it on and release the brakes "
                "(robot mode must be RUNNING)"
            )
        if self.safety_status is not None and self.safety_status not in READY_SAFETY_STATUS:
            return f"the safety status is {self.safety_status}; clear it on the teach pendant"
        return None


def default_connector(host: str, port: int) -> Any:
    return socket.create_connection((host, port), timeout=DASHBOARD_TIMEOUT_S)


def query_status(
    host: str,
    port: int = DASHBOARD_PORT,
    connector: Connector = default_connector,
) -> DashboardStatus | None:
    """None when the dashboard cannot be reached; some installations firewall it off."""
    try:
        connection = connector(host, port)
    except OSError:
        return None
    try:
        _read_line(connection)
        mode = _value_of(_ask(connection, "robotmode"), ROBOT_MODE_NAMES)
        safety = _value_of(_ask(connection, "safetystatus"), SAFETY_STATUS_NAMES)
        remote = _ask(connection, "is in remote control").strip().lower()
    except OSError:
        return None
    finally:
        try:
            connection.close()
        except OSError:
            pass
    remote_control = {"true": True, "false": False}.get(remote)
    return DashboardStatus(robot_mode=mode, safety_status=safety, remote_control=remote_control)


def _ask(connection: Any, command: str) -> str:
    connection.sendall(f"{command}\n".encode("ascii"))
    return _read_line(connection)


def _read_line(connection: Any) -> str:
    data = b""
    while b"\n" not in data:
        chunk = connection.recv(1024)
        if not chunk:
            break
        data += chunk
    return data.decode("utf-8", "replace").strip()


def _value_of(reply: str, known: frozenset[str]) -> str | None:
    _, separator, value = reply.partition(":")
    text = (value if separator else reply).strip().upper().replace(" ", "_")
    return text if text in known else None
