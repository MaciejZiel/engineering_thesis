"""Exercise the sequence runner through the real backend with fake RTDE/sockets."""

import math
import unittest
from unittest.mock import Mock

from tests.test_robot_ur import FakeClock, FakeConnector, FakeRtdeFactory, ready_status
from vision_robot_arm.robot.manual_test import ManualArmTestSession, ManualTestSettings, SequenceStep
from vision_robot_arm.robot.targets import JOINT_NAMES
from vision_robot_arm.robot.ur_backend import URBackend


class ManualSequenceTests(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.connector = FakeConnector()
        self.rtde = FakeRtdeFactory()
        self.backend = None

        def control(config):
            self.backend = URBackend(config, connector=self.connector, clock=self.clock,
                                     rtde_factory=self.rtde, status_query=ready_status,
                                     require_feedback=True)
            self.rtde.clients["robot"].sample = {
                "actual_q": (0, math.radians(-90), math.radians(20), math.radians(-80), 0, 0),
                "actual_qd": (0,) * 6,
                "robot_mode": 7,
                "safety_status": 1,
            }
            return self.backend

        self.session = ManualArmTestSession(lambda config: Mock(), control, clock=self.clock)
        self.session.connect_monitor(ManualTestSettings("robot"))
        self.session.prepare_control()
        self.session.arm()

    def tearDown(self):
        self.session.close()

    def tick_following_setpoint(self):
        points = self.backend._arms["right"]._setpoints.joints
        self.rtde.clients["robot"].sample["actual_q"] = tuple(math.radians(points[j]) for j in JOINT_NAMES)
        self.clock.now += .051
        self.session.tick()

    def test_plan_is_relative_and_validated_before_any_command(self):
        plan = self.backend.plan_joint_sequence((("shoulder", 5, 10), ("elbow", -15, 5), ("shoulder", -2, 10)))
        self.assertEqual(plan, ({"shoulder": -85}, {"elbow": 5}, {"shoulder": -87}))
        self.assertEqual(self.connector.sockets["robot"].sent, [])
        with self.assertRaisesRegex(ValueError, "Step 2"):
            self.session.start_sequence([SequenceStep("shoulder", 50), SequenceStep("shoulder", 50)])
        self.assertFalse(self.session.sequence_running)
        self.assertEqual(self.connector.sockets["robot"].sent, [])

    def test_steps_run_once_in_order_and_wait_for_actual_feedback(self):
        queue = [SequenceStep("shoulder", 3, 5), SequenceStep("elbow", -2, 2)]
        self.session.start_sequence(queue)
        queue.clear()  # Editing the caller's list cannot change a running sequence.
        self.assertEqual(self.session.sequence_index, 0)
        self.assertFalse(self.session.can_jog)
        # Setpoints moving alone must never advance the queue.
        for _ in range(4):
            self.clock.now += .051
            self.session.tick()
        self.assertEqual(self.session.sequence_index, 0)
        for _ in range(300):
            self.tick_following_setpoint()
            if not self.session.sequence_running:
                break
        self.assertEqual(self.session.sequence_state, "completed")
        self.assertEqual(self.session.sequence_index, 2)
        pose = self.backend._arms["right"].feedback_joints
        self.assertAlmostEqual(pose["shoulder"], -87, delta=.3)
        self.assertAlmostEqual(pose["elbow"], 18, delta=.3)
        self.assertEqual(pose["base"], 0)
        before = len(self.connector.sockets["robot"].sent)
        self.session.tick()
        self.assertEqual(before, len(self.connector.sockets["robot"].sent))

    def test_stop_cancels_pending_steps_without_auto_resume(self):
        self.session.start_sequence([SequenceStep("shoulder", 30), SequenceStep("elbow", -15)])
        self.tick_following_setpoint()
        self.session.stop_sequence()
        before = len(self.connector.sockets["robot"].sent)
        for _ in range(3):
            self.tick_following_setpoint()
        self.assertEqual(self.session.sequence_state, "stopped")
        self.assertEqual(self.session.sequence_index, 0)
        self.assertEqual(before, len(self.connector.sockets["robot"].sent))
        self.assertTrue(self.connector.sockets["robot"].sent[-1].startswith(b"stopj("))

    def test_expired_watchdog_cannot_be_revived_or_counted_as_completion(self):
        self.session.start_sequence([SequenceStep("shoulder", 30), SequenceStep("elbow", -15)])
        self.clock.now += .2
        self.session.tick()
        self.assertEqual(self.session.sequence_state, "stopped")
        self.assertEqual(self.session.sequence_index, 0)
        self.assertEqual(self.connector.sockets["robot"].commands(b"servoj("), [])

    def test_manual_motion_is_blocked_while_sequence_runs(self):
        self.session.start_sequence([SequenceStep("shoulder", 30)])
        for action in (lambda: self.session.begin_jog(1),
                       lambda: self.session.begin_multi_jog({"elbow": 5}),
                       lambda: self.session.begin_positions({"elbow": 20}, {"elbow": 5}),
                       lambda: self.session.start_sequence([SequenceStep("elbow", 2)])):
            with self.assertRaisesRegex(RuntimeError, "Stop the sequence"):
                action()

    def test_feedback_failure_cancels_the_queue(self):
        self.session.start_sequence([SequenceStep("shoulder", 30), SequenceStep("elbow", -15)])
        self.rtde.clients["robot"].sample["safety_status"] = 3
        self.clock.now += .051
        self.session.tick()
        self.assertEqual(self.session.sequence_state, "fault")
        self.assertEqual(self.session.phase, "fault")
        self.assertTrue(self.connector.sockets["robot"].closed)

    def test_stuck_robot_times_out_instead_of_starting_next_step(self):
        self.session.start_sequence([SequenceStep("shoulder", 1), SequenceStep("elbow", -15)])
        self.clock.now = self.session._sequence_deadline + .01
        self.session.tick()
        self.assertEqual(self.session.sequence_state, "stopped")
        self.assertIn("timed out", self.session.sequence_message)
        self.assertEqual(self.session.sequence_index, 0)

    def test_disconnect_cancels_pending_steps(self):
        self.session.start_sequence([SequenceStep("shoulder", 1), SequenceStep("elbow", -15)])
        self.session.stop_and_disconnect()
        self.assertEqual(self.session.sequence_state, "stopped")
        self.assertEqual(self.session.phase, "disconnected")

    def test_invalid_inputs_are_rejected_before_connection(self):
        for joint, delta, speed in (("other", 30, 5), ("elbow", float("nan"), 5),
                                    ("elbow", float("inf"), 5), ("elbow", 0, 5),
                                    ("elbow", 30, 31), ("elbow", 30, -1)):
            with self.assertRaises(ValueError):
                SequenceStep(joint, delta, speed)
