"""Minimal RTDE client: read what the UR7e is actually doing, not what we asked for.

The Real-Time Data Exchange interface (port 30004) names the fields it sends, so the
layout never has to be guessed. Only the output half of the protocol is implemented:
the robot is commanded through URScript, this is feedback only.
"""

import socket
import struct
from typing import Any, Callable

RTDE_PORT = 30004
PROTOCOL_VERSION = 2
DEFAULT_FREQUENCY_HZ = 125.0
DEFAULT_VARIABLES = ("actual_q", "robot_mode", "safety_status")
HANDSHAKE_TIMEOUT_S = 2.0
HEADER = struct.Struct(">HB")

REQUEST_PROTOCOL_VERSION = 86
TEXT_MESSAGE = 77
CONTROL_PACKAGE_SETUP_OUTPUTS = 79
CONTROL_PACKAGE_START = 83
CONTROL_PACKAGE_PAUSE = 80
DATA_PACKAGE = 85

TYPE_SIZES = {
    "BOOL": 1,
    "UINT8": 1,
    "INT32": 4,
    "UINT32": 4,
    "UINT64": 8,
    "DOUBLE": 8,
    "VECTOR3D": 24,
    "VECTOR6D": 48,
    "VECTOR6INT32": 24,
    "VECTOR6UINT32": 24,
}

Connector = Callable[[str, int], Any]


class RtdeError(RuntimeError):
    """The controller refused the handshake or sent something unusable."""


def encode_packet(packet_type: int, payload: bytes = b"") -> bytes:
    return HEADER.pack(HEADER.size + len(payload), packet_type) + payload


def decode_packets(buffer: bytes) -> tuple[list[tuple[int, bytes]], bytes]:
    packets: list[tuple[int, bytes]] = []
    while len(buffer) >= HEADER.size:
        size, packet_type = HEADER.unpack_from(buffer)
        if size < HEADER.size or len(buffer) < size:
            break
        packets.append((packet_type, buffer[HEADER.size:size]))
        buffer = buffer[size:]
    return packets, buffer


def decode_values(payload: bytes, recipe: tuple[tuple[str, str], ...]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    offset = 0
    for name, type_name in recipe:
        size = TYPE_SIZES.get(type_name)
        if size is None or offset + size > len(payload):
            break
        chunk = payload[offset:offset + size]
        offset += size
        if type_name == "VECTOR6D":
            values[name] = struct.unpack(">6d", chunk)
        elif type_name == "VECTOR3D":
            values[name] = struct.unpack(">3d", chunk)
        elif type_name in ("VECTOR6INT32", "VECTOR6UINT32"):
            values[name] = struct.unpack(">6i" if type_name.endswith("INT32") else ">6I", chunk)
        elif type_name == "DOUBLE":
            values[name] = struct.unpack(">d", chunk)[0]
        elif type_name == "INT32":
            values[name] = struct.unpack(">i", chunk)[0]
        elif type_name == "UINT32":
            values[name] = struct.unpack(">I", chunk)[0]
        elif type_name == "UINT64":
            values[name] = struct.unpack(">Q", chunk)[0]
        elif type_name in ("BOOL", "UINT8"):
            values[name] = struct.unpack(">B", chunk)[0]
    return values


def default_connector(host: str, port: int) -> Any:
    return socket.create_connection((host, port), timeout=HANDSHAKE_TIMEOUT_S)


class RtdeClient:
    def __init__(
        self,
        host: str,
        port: int = RTDE_PORT,
        variables: tuple[str, ...] = DEFAULT_VARIABLES,
        frequency_hz: float = DEFAULT_FREQUENCY_HZ,
        connector: Connector = default_connector,
    ) -> None:
        self.host = host
        self.port = port
        self._variables = variables
        self._frequency_hz = frequency_hz
        self._connector = connector
        self._socket: Any | None = None
        self._recipe: tuple[tuple[str, str], ...] = ()
        self._buffer = b""
        self.last_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._socket is not None

    def connect(self) -> bool:
        """Negotiate the output recipe. Feedback is optional, so failures are reported, not raised."""
        try:
            self._socket = self._connector(self.host, self.port)
            self._handshake()
        except (OSError, RtdeError, struct.error) as error:
            self.last_error = str(error)
            self.close()
            return False
        self.last_error = None
        return True

    def read(self) -> dict[str, Any] | None:
        """Return the newest sample, or None when the robot has not sent a full package yet."""
        if self._socket is None:
            return None
        try:
            while True:
                chunk = self._socket.recv(4096)
                if not chunk:
                    self.last_error = "connection closed by the controller"
                    self.close()
                    return None
                self._buffer += chunk
                if len(chunk) < 4096:
                    break
        except (BlockingIOError, TimeoutError):
            pass
        except OSError as error:
            self.last_error = str(error)
            self.close()
            return None

        packets, self._buffer = decode_packets(self._buffer)
        sample = None
        for packet_type, payload in packets:
            if packet_type == DATA_PACKAGE and payload:
                sample = decode_values(payload[1:], self._recipe)
        return sample

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            except OSError:
                pass
        self._socket = None
        self._buffer = b""

    def _handshake(self) -> None:
        version = self._exchange(
            REQUEST_PROTOCOL_VERSION, struct.pack(">H", PROTOCOL_VERSION), REQUEST_PROTOCOL_VERSION
        )
        if not version or version[0] != 1:
            raise RtdeError(f"controller rejected RTDE protocol version {PROTOCOL_VERSION}")

        payload = struct.pack(">d", self._frequency_hz) + ",".join(self._variables).encode("utf-8")
        reply = self._exchange(CONTROL_PACKAGE_SETUP_OUTPUTS, payload, CONTROL_PACKAGE_SETUP_OUTPUTS)
        types = reply[1:].decode("utf-8", "replace").split(",") if reply else []
        if not types or len(types) != len(self._variables):
            raise RtdeError("controller did not describe the requested outputs")
        missing = [name for name, kind in zip(self._variables, types) if kind == "NOT_FOUND"]
        if missing:
            raise RtdeError(f"controller does not provide: {', '.join(missing)}")
        self._recipe = tuple(zip(self._variables, types))

        started = self._exchange(CONTROL_PACKAGE_START, b"", CONTROL_PACKAGE_START)
        if not started or started[0] != 1:
            raise RtdeError("controller refused to start the RTDE stream")
        self._socket.settimeout(0.0)

    def _exchange(self, packet_type: int, payload: bytes, expected: int) -> bytes:
        self._socket.settimeout(HANDSHAKE_TIMEOUT_S)
        self._socket.sendall(encode_packet(packet_type, payload))
        while True:
            packets, self._buffer = decode_packets(self._buffer)
            for reply_type, reply in packets:
                if reply_type == expected:
                    return reply
                if reply_type == TEXT_MESSAGE:
                    self.last_error = reply[1:].decode("utf-8", "replace")
            chunk = self._socket.recv(4096)
            if not chunk:
                raise RtdeError("connection closed during the RTDE handshake")
            self._buffer += chunk
