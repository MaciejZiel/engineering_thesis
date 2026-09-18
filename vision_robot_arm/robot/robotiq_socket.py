"""Robotiq Hand-E over the URCap's own TCP server, without touching URScript.

The Robotiq URCap runs a small text server on the controller (port 63352) that
speaks ``SET VAR VALUE ...`` / ``GET VAR`` and answers ``ack`` / ``VAR VALUE``.
It is reachable from the network, so the gripper can be driven from this PC
directly. That matters because every program sent to the URScript port replaces
the one running: a gripper program issued next to a movej either stops the arm,
or is itself killed by the next motion command before it has set the GO bit. A
separate socket has neither problem, and a fist can act the instant it is seen
instead of at the next follow sample.

Register names follow the URCap: ACT activate, GTO go to position, STA status
(3 = active), POS position 0 open .. 255 closed, SPE speed, FOR force, OBJ object
detection, FLT fault.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Callable

ROBOTIQ_PORT = 63352
CONNECT_TIMEOUT_S = 2.0
REPLY_TIMEOUT_S = 1.0
# Measured on this cell's controller: the URCap server holds its reply to
# request N until request N+1 arrives. A lone `GET STA` therefore times out even
# though the daemon is healthy. Every request is sent together with this
# harmless follow-up in the same packet, which makes the real reply come out at
# once; the follow-up's own reply is a stray register line that the matching in
# `_exchange` skips.
FLUSH_REQUEST = "GET STA"
ACTIVATION_TIMEOUT_S = 10.0
ACTIVATION_POLL_S = 0.2
STATUS_ACTIVE = 3
POSITION_OPEN = 0
POSITION_CLOSED = 255
# How many reply lines to look through for the one we asked for. The daemon can
# answer a previous request late, so the next reply is not always ours.
MAX_STRAY_REPLIES = 8

Connector = Callable[[str, int, float], Any]


class GripperError(RuntimeError):
    """The gripper is unreachable, inactive or rejected a command."""


def default_connector(host: str, port: int, timeout_s: float) -> Any:
    return socket.create_connection((host, port), timeout=timeout_s)


def percent_to_register(percent: int) -> int:
    """0..100 % -> 0..255, rounded half up in exact integer arithmetic.

    Floating point would make 50 % land on 127: 50 * 2.55 is 127.49999... in
    binary, and Python's round() is half-to-even on top of that.
    """
    clamped = max(0, min(100, int(percent)))
    return (clamped * 255 + 50) // 100


def is_register_reply(line: str) -> bool:
    """`STA 3`, `POS 255` ... - the shape of a GET answer, as opposed to ack/nack."""
    words = line.split()
    return len(words) == 2 and words[0].isalpha() and words[0].isupper() and words[1].lstrip("-").isdigit()


class RobotiqSocketGripper:
    def __init__(
        self,
        host: str,
        port: int = ROBOTIQ_PORT,
        connector: Connector = default_connector,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.host = host
        self.port = port
        self._connector = connector
        self._sleep = sleep
        self._clock = clock
        self._socket: Any | None = None
        self._buffer = b""

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> None:
        """Open the socket and make sure the gripper is activated. Never resets it."""
        try:
            self._socket = self._connector(self.host, self.port, CONNECT_TIMEOUT_S)
            self._socket.settimeout(REPLY_TIMEOUT_S)
            # Requests are a few bytes each; do not let Nagle hold them back.
            setsockopt = getattr(self._socket, "setsockopt", None)
            if setsockopt is not None:
                setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError as error:
            self._socket = None
            raise GripperError(
                f"Robotiq gripper at {self.host}:{self.port} is unreachable: {error}"
            ) from error
        self.ensure_activated()

    def ensure_activated(self) -> None:
        """Activate only if needed; a reset would run the 5 s calibration cycle."""
        if self.get("STA") == STATUS_ACTIVE:
            return
        self.set(ACT=1)
        deadline = self._clock() + ACTIVATION_TIMEOUT_S
        while self._clock() < deadline:
            self._sleep(ACTIVATION_POLL_S)
            if self.get("STA") == STATUS_ACTIVE:
                return
        raise GripperError(
            f"Robotiq gripper did not report active (STA {STATUS_ACTIVE}) "
            f"within {ACTIVATION_TIMEOUT_S:g} s"
        )

    def command(self, closed: bool, speed_percent: int, force_percent: int) -> None:
        """One non-blocking move request; the gripper carries it out on its own."""
        self.set(
            POS=POSITION_CLOSED if closed else POSITION_OPEN,
            SPE=percent_to_register(speed_percent),
            FOR=percent_to_register(force_percent),
            GTO=1,
        )

    def status(self) -> dict[str, int]:
        return {name: self.get(name) for name in ("STA", "POS", "OBJ", "FLT")}

    def set(self, **variables: int) -> None:
        request = "SET " + " ".join(f"{name} {int(value)}" for name, value in variables.items())
        # A late GET answer may arrive first; the SET's own reply is whatever
        # does not look like one - normally `ack`.
        reply = self._exchange(request, lambda line: not is_register_reply(line))
        if reply.lower() != "ack":
            raise GripperError(f"Robotiq rejected '{request}': {reply!r}")

    def get(self, variable: str) -> int:
        reply = self._exchange(f"GET {variable}", lambda line: line.upper().startswith(f"{variable} "))
        try:
            return int(reply.split()[1])
        except (IndexError, ValueError) as error:
            raise GripperError(f"Robotiq answered '{reply}' to GET {variable}") from error

    def close(self) -> None:
        sock, self._socket = self._socket, None
        self._buffer = b""
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _exchange(self, request: str, accept: Callable[[str], bool]) -> str:
        """Send one request and return the first reply line meant for it."""
        if self._socket is None:
            raise GripperError("Robotiq gripper is not connected")
        try:
            self._socket.sendall(f"{request}\n{FLUSH_REQUEST}\n".encode("ascii"))
        except OSError as error:
            raise GripperError(f"could not send '{request}' to the Robotiq gripper: {error}") from error
        for _ in range(MAX_STRAY_REPLIES):
            line = self._read_line()
            if line and accept(line):
                return line
        raise GripperError(f"no matching reply to '{request}' from the Robotiq gripper")

    def _read_line(self) -> str:
        """Next reply. GET answers end in a newline; `ack`/`nack` do not.

        Measured on the cell: the server writes `ack` with no terminator, so
        with the flush follow-up it arrives glued to the next reply as
        `ackSTA 3\\n`. The bare tokens are therefore recognised on their own.
        """
        assert self._socket is not None
        while True:
            lowered = self._buffer.lower()
            for token in (b"nack", b"ack"):
                if lowered.startswith(token):
                    self._buffer = self._buffer[len(token):]
                    return token.decode("ascii")
            if b"\n" in self._buffer:
                line, _, self._buffer = self._buffer.partition(b"\n")
                return line.decode("ascii", "replace").strip()
            try:
                chunk = self._socket.recv(256)
            except OSError as error:
                raise GripperError(f"no reply from the Robotiq gripper: {error}") from error
            if not chunk:
                raise GripperError("the Robotiq gripper closed the connection")
            self._buffer += chunk
