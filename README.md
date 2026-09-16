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
.venv\Scripts\python main.py
```

Or, with the virtual environment activated (`.venv\Scripts\Activate.ps1`):

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

Frames are shown at 1920x1080 by default. The camera is asked for that size
and frames of another size are resized, so the overlay always has the same
proportions. Pick a smaller size for slow machines or `0 0` to keep the
camera's native size:

```powershell
python main.py --width 1280 --height 720
python main.py --width 0 --height 0
```

The camera image is mirrored like a selfie view, so your right arm appears on
the right side of the window. Detection still runs on the original image, so
`right_*` angles and gestures always mean your own right side. Turn mirroring
off with:

```powershell
python main.py --no-mirror
```

Other useful options:

```powershell
python main.py --camera 0 --print-interval 0.5
```

For stronger smoothing, lower the alpha value:

```powershell
python main.py --smoothing-alpha 0.2
```

The default models are stored at:

```text
models/pose_landmarker_lite.task
models/hand_landmarker.task
```

You can use another MediaPipe Pose Landmarker model with:

```powershell
python main.py --model path\to\pose_landmarker.task
```

Hand tracking (open hand / fist for the gripper) uses the MediaPipe Hand
Landmarker. Point `--hand-model` at another `.task` file or disable it with
`--no-hands` on slow machines.

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
    hud.py                 # translucent panels and text used by both areas
    pose_state.py          # shared pose data object
    runtime.py             # optional dependency loading
  vision/                  # camera, pose detection, gestures, drawing
    calibration.py         # neutral pose calibration
    drawing.py             # custom stick figure and overlay
    gestures.py            # simple rule-based body gesture detection
    hand_gestures.py       # open hand / fist detection and left-right matching
    hand_tracker.py        # MediaPipe Hand Landmarker wrapper
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
    serial_backend.py      # serial line protocol and pyserial backend
    simulation.py          # simulated two-arm robot backend
    targets.py             # JointTargets, ArmState and RobotState value objects
    visualization.py       # robot simulation window drawn in test mode
models/
  hand_landmarker.task
  pose_landmarker_lite.task
tests/
  test_cli.py
  test_core_config.py
  test_robot_factory.py
  test_robot_mapping.py
  test_robot_serial.py
  test_robot_simulation.py
  test_robot_visualization.py
  test_vision_drawing.py
  test_vision_hand_gestures.py
  test_vision_metrics.py
  test_vision_smoothing.py
```

Test files follow the `test_<area>_<topic>.py` convention so that each area
owns its own tests. See `docs/ARCHITECTURE.md` for the data flow and import
rules, and `docs/OWNERSHIP.md` for who owns which area and how we commit.

## Test Mode

Test mode is the quickest way to see what the robot side receives. It draws
the current shoulder, elbow and wrist angles next to the joints on the camera
image, prints the robot state to the console, and opens a second window,
"Robot Simulation", with two schematic three-link arms. The left panel is
the robot arm driven by your left arm, the right panel by your right arm
(mirrored like the camera view). The grey arm is the mapped target, the green
arm is where the simulated robot currently is, the blue jaws show the gripper.
The top-left panel of the camera window shows detection, robot backend,
calibration and recording state plus detected gestures; the key hints sit at
the bottom.

```powershell
python main.py --test-mode
```

Without an explicit `--robot-backend` the simulated robot is used. Combine it
with any other backend to inspect what is being sent:

```powershell
python main.py --test-mode --robot-backend serial --robot-port COM3
```

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

The second layer uses the MediaPipe Hand Landmarker. Each detected hand is
matched to the nearest pose wrist, so the names always refer to your own
left or right hand. A hand counts as open when at least three fingers are
extended (fingertip farther from the wrist than the middle knuckle) and as a
fist when none are:

- `left_hand_open`, `right_hand_open`
- `left_fist`, `right_fist`

## Robot Backends

The robot side turns every `PoseState` into `JointTargets`: for each of the
two robot arms (`right`, `left`) the shoulder, elbow and wrist angles in
degrees plus a gripper command, and one lift-mode flag. Your right arm drives
the right robot arm and your left arm the left one. The targets go to a
backend selected with `--robot-backend`:

| Backend  | What it does                                                    |
| -------- | --------------------------------------------------------------- |
| `none`   | default, robot side disabled                                    |
| `debug`  | prints the mapped targets to the console                        |
| `sim`    | two simulated arms with speed limit, shown in the test window   |
| `serial` | sends targets to hardware over a serial port (pyserial)         |

```powershell
python main.py --robot-backend debug
```

`--robot-debug` still works as a deprecated alias for `--robot-backend debug`.

Each simulated arm starts at `--robot-home` degrees (default `90`) and moves
toward the mapped targets at most `--robot-max-speed` degrees per second
(default `90`). With `--test-mode` the current and target angles are drawn in
the simulation window, so you can test the mapping without hardware:

```powershell
python main.py --robot-backend sim --robot-max-speed 60
```

### Serial protocol

The serial backend needs the `serial` extra (`pip install -e .[serial]`) and a
port:

```powershell
python main.py --robot-backend serial --robot-port COM3 --robot-baud 115200
```

Each frame is one ASCII line, at most every `--robot-send-interval` seconds
(default `0.05`):

```text
RS:90.0;RE:45.0;RW:120.0;RG:1;LS:30.0;LE:170.0;LW:90.0;LG:0;L:0
```

Every field except `L` starts with the arm, `R` (right) or `L` (left):

| Field | Meaning                                       |
| ----- | --------------------------------------------- |
| `?S`  | shoulder angle in degrees                     |
| `?E`  | elbow angle in degrees                        |
| `?W`  | wrist angle in degrees                        |
| `?G`  | gripper, `1` close / `0` open, omitted = hold  |
| `L`   | lift mode, `1` on / `0` off                    |

Joints without a reliable angle in the current frame are omitted. The wire
format lives in `encode_targets` in `vision_robot_arm/robot/serial_backend.py`
and is expected to change once the hardware is chosen.

Current mapping (`vision_robot_arm/robot/mapping.py`), applied to each arm:

- shoulder angle (elbow-shoulder-hip) -> `shoulder`
- elbow angle (shoulder-elbow-wrist) -> `elbow`
- wrist angle (elbow-wrist-index finger) -> `wrist`
- `<side>_fist` -> `gripper=close`
- `<side>_hand_open` -> `gripper=open`
- `right_hand_up` -> `lift_mode=on`

Joint angles are clamped to `--robot-shoulder-range`, `--robot-elbow-range`
and `--robot-wrist-range`
(default `0 180`) and changes smaller than `--robot-deadband` degrees (default
`1.5`) are ignored to suppress jitter. When neither gripper gesture is active
the gripper keeps its previous state.
