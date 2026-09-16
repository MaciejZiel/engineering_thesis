# Vision Robot Arm

Prototype for recognizing a person from a webcam, drawing a custom stick
figure, and printing joint data that can later feed a robot arm controller.

## Status

Current version:

- opens a webcam with OpenCV
- detects a single human pose with MediaPipe Pose Landmarker
- draws a custom skeleton with a horizontal hip line
- smooths landmarks and joint angles
- can calibrate a neutral pose
- records pose data to CSV
- recognizes simple rule-based gestures
- maps pose state to debug robot commands
- prints joint angles, raw pose landmarks, or both
- lets you switch data modes while the camera is running

## Setup

Use Python 3.12 or 3.13. Create a virtual environment and install the runtime
dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

For development (tests) and the optional serial robot backend install the
project with its extras instead:

```powershell
.venv\Scripts\python -m pip install -e .[dev,serial]
```

## Run

```powershell
python main.py
```

If the default camera is not correct:

```powershell
python main.py --camera 1
```

To process a saved video instead of the webcam:

```powershell
python main.py --video sample.mp4
```

To loop a video while testing:

```powershell
python main.py --video sample.mp4 --loop-video
```

Useful options:

```powershell
python main.py --camera 0 --width 1280 --height 720 --print-interval 0.5
```

For stronger smoothing, lower the alpha value:

```powershell
python main.py --smoothing-alpha 0.2
```

The default pose model is stored at:

```text
models/pose_landmarker_lite.task
```

You can use another MediaPipe Pose Landmarker model with:

```powershell
python main.py --model path\to\pose_landmarker.task
```

## Project Structure

The package is split into three areas so two people can work in parallel:
`core` is the shared contract, `vision` turns camera frames into a pose state,
and `robot` turns the pose state into robot commands.

```text
main.py                    # thin entrypoint
vision_robot_arm/
  app.py                   # camera loop and module wiring (shared)
  cli.py                   # command-line arguments (shared)
  core/                    # shared contract between vision and robot
    config.py              # runtime settings and defaults
    pose_state.py          # shared pose data object
    runtime.py             # optional dependency loading
  vision/                  # camera, pose detection, gestures, drawing
    calibration.py         # neutral pose calibration
    drawing.py             # custom stick figure and overlay
    gestures.py            # simple rule-based gesture detection
    landmarks.py           # landmark lookup and visibility checks
    metrics.py             # joint angle calculations
    output.py              # console printing modes
    pose_tracker.py        # MediaPipe Pose Landmarker wrapper
    recording.py           # CSV pose recordings
    smoothing.py           # low-pass filters
    state_builder.py       # raw detections -> smoothed pose state
  robot/                   # robot arm control
    backend.py             # RobotBackend protocol and debug backend
    config.py              # RobotConfig and joint limits
    controller.py          # RobotController glue between mapper and backend
    factory.py             # backend selection from --robot-backend
    mapping.py             # PoseState -> JointTargets
    targets.py             # JointTargets value object
models/
  pose_landmarker_lite.task
tests/
  test_cli.py
  test_core_config.py
  test_robot_factory.py
  test_robot_mapping.py
  test_vision_metrics.py
  test_vision_smoothing.py
```

Test files follow the `test_<area>_<topic>.py` convention so that each area
owns its own tests. See `docs/ARCHITECTURE.md` for the data flow and import
rules, and `docs/OWNERSHIP.md` for who owns which area and how we commit.

## Controls

- `1`: print joint angles
- `2`: print raw landmarks
- `3`: print angles and raw landmarks
- `c`: calibrate the current pose as neutral
- `r`: start/stop CSV recording
- `q` or `Esc`: quit

## Tests

```powershell
python -m unittest discover -s tests
```

With the `dev` extra installed you can also run:

```powershell
python -m pytest
```

## Printed Data

Angle mode prints values such as elbows, shoulders, hips, knees, and ankles.
The angle is calculated from three body points, with the middle point as the
joint. After calibration, angle mode also prints relative angle offsets from
the calibrated neutral pose.

Landmark mode prints MediaPipe's 33 pose points with normalized image
coordinates (`x`, `y`, `z`) and `visibility`. When available, world coordinates
are printed too.

## CSV Recordings

Press `r` while the app is running to start or stop recording. Files are saved
under `recordings/` by default and include timestamps, smoothed angles, relative
angles after calibration, detected gestures, image landmarks, visibility, and
world coordinates.

```powershell
python main.py --recording-dir recordings
```

## Gestures

The first gesture layer is rule-based and prints/records:

- `left_hand_up`, `right_hand_up`, `both_hands_up`
- `left_arm_side`, `right_arm_side`
- `left_elbow_bent`, `right_elbow_bent`

## Robot Backends

The robot side turns every `PoseState` into `JointTargets` (named joint angles
in degrees, a gripper command and a lift-mode flag) and hands them to a backend
selected with `--robot-backend`:

| Backend  | What it does                                                    |
| -------- | --------------------------------------------------------------- |
| `none`   | default, robot side disabled                                    |
| `debug`  | prints the mapped targets to the console                        |
| `sim`    | simulated arm (planned)                                         |
| `serial` | sends targets to hardware over a serial port (planned)          |

```powershell
python main.py --robot-backend debug
```

`--robot-debug` still works as a deprecated alias for `--robot-backend debug`.

Current mapping (`vision_robot_arm/robot/mapping.py`):

- right shoulder angle -> `shoulder`
- right elbow angle -> `elbow`
- `right_hand_up` -> `lift_mode=on`
- `right_elbow_bent` -> `gripper=close`
- `right_arm_side` -> `gripper=open`

Joint angles are clamped to `--robot-shoulder-range` / `--robot-elbow-range`
(default `0 180`) and changes smaller than `--robot-deadband` degrees (default
`1.5`) are ignored to suppress jitter. When neither gripper gesture is active
the gripper keeps its previous state.
