import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from vision_robot_arm.core.pose_state import PoseState
from vision_robot_arm.vision.recording import CsvPoseRecorder


class RecordingFailureTests(unittest.TestCase):
    def test_repeated_sessions_never_overwrite_previous_files(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = CsvPoseRecorder(Path(directory))
            first = recorder.start({})
            recorder.stop()
            content = first.read_bytes()
            second = recorder.start({})
            recorder.stop()
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), content)

    def test_failed_start_is_reported_without_remaining_active(self):
        recorder = CsvPoseRecorder(Path("unused"))
        with patch.object(Path, "mkdir", side_effect=PermissionError("denied")):
            self.assertEqual(recorder.toggle({}), (False, None))
        self.assertFalse(recorder.is_recording)
        self.assertIn("denied", recorder.last_error)

    def test_failed_write_stops_recording_and_closes_file(self):
        recorder = CsvPoseRecorder(Path("unused"))
        file = Mock()
        recorder._file = file
        recorder._writer = Mock()
        recorder._writer.writerow.side_effect = OSError("disk full")
        state = PoseState(0, [], None, {}, {}, {}, (), False)
        recorder.write_state(state, {})
        self.assertFalse(recorder.is_recording)
        self.assertIn("disk full", recorder.last_error)
        file.close.assert_called_once()

    def test_stop_closes_and_clears_state_even_when_flush_fails(self):
        recorder = CsvPoseRecorder(Path("unused"))
        file = Mock()
        file.flush.side_effect = OSError("disk full")
        recorder._file, recorder._writer = file, Mock()
        with self.assertRaises(OSError):
            recorder.stop()
        file.close.assert_called_once()
        self.assertFalse(recorder.is_recording)
