import time
import unittest

from vision_robot_arm.web.bridge import WebBridge


class WebBridgeTests(unittest.TestCase):
    def ready(self):
        bridge = WebBridge()
        bridge.state.update(status="running", detected=True)
        return bridge

    def test_demo_cannot_enqueue_actions(self):
        bridge = WebBridge(demo=True)
        with self.assertRaisesRegex(ValueError, "read-only"):
            bridge.command("record")
        self.assertEqual(bridge.poll_key(), 255)

    def test_existing_actions_are_delivered_once(self):
        bridge = self.ready()
        bridge.command("calibrate")
        self.assertEqual(bridge.poll_key(), ord("c"))
        self.assertEqual(bridge.poll_key(), 255)

    def test_stale_queued_actions_are_discarded(self):
        bridge = self.ready()
        bridge.commands.append((time.monotonic() - 2, ord("r")))
        self.assertEqual(bridge.poll_key(), 255)

    def test_stale_engine_refuses_actions(self):
        bridge = self.ready()
        bridge.updated -= 3
        with self.assertRaisesRegex(ValueError, "unavailable"):
            bridge.command("record")

    def test_no_pose_refuses_calibration_but_allows_recording(self):
        bridge = self.ready()
        bridge.state["detected"] = False
        with self.assertRaisesRegex(ValueError, "camera"):
            bridge.command("calibrate")
        bridge.command("record")
        self.assertEqual(bridge.poll_key(), ord("r"))

    def test_no_hardware_commands_are_exposed(self):
        bridge = self.ready()
        for action in ("arm", "start", "jog", "movej", "servoj"):
            with self.assertRaisesRegex(ValueError, "Unknown"):
                bridge.command(action)

    def test_error_clears_pending_commands(self):
        bridge = self.ready()
        bridge.command("record")
        bridge.fail("Camera lost")
        self.assertEqual(bridge.poll_key(), 255)
        self.assertEqual(bridge.snapshot()["status"], "error")


try:
    from fastapi.testclient import TestClient
    from vision_robot_arm.web.server import create_app
except ImportError:
    TestClient = None


@unittest.skipIf(TestClient is None, "Optional web dependencies are not installed")
class WebServerTests(unittest.TestCase):
    def setUp(self):
        self.bridge = WebBridge()
        self.bridge.state.update(status="running", detected=True)
        self.client = TestClient(create_app(self.bridge))

    def test_command_requires_session_token(self):
        self.assertEqual(self.client.post("/api/v1/commands/record").status_code, 403)

    def test_cross_origin_command_is_rejected(self):
        result = self.client.post("/api/v1/commands/record", headers={
            "origin": "https://untrusted.example", "x-session-token": self.bridge.token})
        self.assertEqual(result.status_code, 403)

    def test_allowed_command_reaches_existing_loop(self):
        result = self.client.post("/api/v1/commands/record", headers={"x-session-token": self.bridge.token})
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.bridge.poll_key(), ord("r"))

    def test_rebinding_host_is_rejected(self):
        self.assertEqual(self.client.get("/api/v1/state", headers={"host": "untrusted.example"}).status_code, 400)

    def test_snapshot_contains_version_and_age(self):
        result = self.client.get("/api/v1/state").json()
        self.assertEqual(result["schema_version"], 1)
        self.assertGreaterEqual(result["age_ms"], 0)

    def test_websocket_delivers_latest_snapshot(self):
        with self.client.websocket_connect("/api/v1/telemetry") as socket:
            self.assertEqual(socket.receive_json()["status"], "running")

    def test_frame_is_not_cached(self):
        self.bridge.jpeg = b"jpeg"
        result = self.client.get("/api/v1/frame")
        self.assertEqual(result.headers["cache-control"], "no-store")
