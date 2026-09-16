# Vision Robot Arm

Prototype for recognizing a person from a webcam, drawing a custom stick
figure, and turning the joint data into commands for two Universal Robots
UR7e cobots (the PJA Arm Robotics lab setup: two UR7e, UR AI Accelerator with
NVIDIA Jetson AGX Orin and an Orbbec Gemini 335Lg 3D camera).

## Status

Current version:

- presents the camera, system status, controls and dual-arm digital twin in one dashboard window
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

Use Python 3.12 (the tested Windows version is 3.12.10). Create a virtual environment and install the runtime
dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-lock.txt
```

The lock file is the tested, reproducible Windows environment and already
contains the test runner and optional serial dependency. Install the project
itself in editable mode without changing those versions:

```powershell
.venv\Scripts\python -m pip install --no-deps -e .
```

`requirements.txt` and the extras in `pyproject.toml` remain available for
dependency maintenance, but both collaborators should use the lock file for
normal development and demonstrations.

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

## Dashboard UI

The application uses one resizable OpenCV window named **Motion Twin**. The
camera feed occupies the main area. A sidebar contains the embedded digital
twin of both UR7e arms, tracking/calibration/recording state, robot connection
state and recognized gestures. The bottom bar contains clickable controls, so
the application can be operated with either the mouse or keyboard.

The visual design follows the lab setup described by the PJATK ARM Robotics
program: two UR7e cobots, an NVIDIA Jetson Orin AGX module and an Orbbec Gemini
335Lg 3D camera. Orange indicates tracked human arms and the current robot
position; grey indicates robot targets.

Press `f` or click **FULLSCREEN** for presentation mode. Closing the window,
pressing `q`/`Esc`, or clicking **QUIT** shuts down the trackers, recorder,
camera and robot backend cleanly.

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
    serial_backend.py      # generic serial line protocol (pyserial)
    simulation.py          # two simulated UR7e arms
    targets.py             # UR joint names, JointTargets, ArmState, RobotState
    ur_backend.py          # URScript servoj over TCP to the UR7e controllers
    visualization.py       # robot digital twin embedded in the dashboard
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
  test_robot_ur.py
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
image and prints the robot state to the console. The dashboard's embedded
digital-twin panel contains two schematic UR7e arms (shoulder, elbow, wrist 1;
base, wrist 2 and wrist 3 are held at the home pose). The left panel is the
cobot driven by your left arm and the right panel by your right arm (mirrored
like the camera view). The grey arm is the mapped target and the orange arm is
where the simulated robot currently is.
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
- `f`: toggle fullscreen presentation mode
- `q` or `Esc`: quit

The same output-mode, calibration, recording, fullscreen and quit actions are
available as buttons in the bottom bar.

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

The target hardware is two **Universal Robots UR7e** cobots (6 joints: base,
shoulder, elbow, wrist 1, wrist 2, wrist 3; payload 7.5 kg, reach 850 mm,
joint range +-360 deg except elbow +-160 deg, max joint speed 180 deg/s).
The robot side turns every `PoseState` into `JointTargets`: for each of the
two cobots (`right`, `left`) the UR `shoulder`, `elbow` and `wrist_1` joint
angles in degrees plus a gripper command, and one lift-mode flag. `base`,
`wrist_2` and `wrist_3` are held at the UR home pose
`[0, -90, 0, -90, 0, 0]`. Your right arm drives the right cobot and your left
arm the left one. The targets go to a backend selected with
`--robot-backend`:

| Backend  | What it does                                                    |
| -------- | --------------------------------------------------------------- |
| `none`   | default, robot side disabled                                    |
| `debug`  | prints the mapped targets to the console                        |
| `sim`    | two simulated UR7e arms with speed limit, shown in the test window |
| `ur`     | real UR7e cobots: URScript `servoj` over TCP (ports 30001/30002) |
| `serial` | generic serial line protocol for a microcontroller (pyserial)   |

```powershell
python main.py --robot-backend debug
```

`--robot-debug` still works as a deprecated alias for `--robot-backend debug`.

Each simulated arm starts at the UR home pose and moves toward the mapped
targets at most `--robot-max-speed` degrees per second (default `60`, UR7e
hardware limit `180`). With `--test-mode` the current and target angles are
the embedded digital-twin panel, so you can test the mapping without hardware:

```powershell
python main.py --robot-backend sim --robot-max-speed 60
```

### UR7e over URScript

Put each cobot in **Remote Control** mode on the teach pendant, give the
PC a route to the controllers, then:

```powershell
python main.py --robot-backend ur --robot-right-host 192.168.1.10 --robot-left-host 192.168.1.11
```

The backend opens one TCP connection per cobot to the URScript secondary
interface (`--robot-ur-port`, default `30002`; `30001` is the primary
interface) and streams one line every `--robot-send-interval` seconds:

```text
servoj([0.0000, -0.7854, 1.5708, -1.5708, 0.0000, 0.0000], 0, 0, 0.050, 0.100, 300)
```

The six values are the joint angles in radians in UR order (base, shoulder,
elbow, wrist 1, wrist 2, wrist 3). The remaining parameters are `a` and `v`
(ignored by `servoj`), `t` (`--robot-send-interval`), `lookahead_time`
(`--robot-servo-lookahead`, default `0.1`) and `gain` (`--robot-servo-gain`,
default `300`). The gripper is driven through tool digital output 0
(`set_tool_digital_out(0, True)` = close); swap `encode_gripper` in
`vision_robot_arm/robot/ur_backend.py` for the URCap call of the gripper the
lab mounts (for example Robotiq). Start with a low `--robot-max-speed` and
narrow joint ranges when testing on the real cobots.

Body angle to UR joint mapping (`vision_robot_arm/robot/config.py`,
`JointMapping`):

| Body angle (deg)                        | UR joint  | Formula        | Default range |
| --------------------------------------- | --------- | -------------- | ------------- |
| shoulder (elbow-shoulder-hip), 0 = down | `shoulder`| body - 180     | -180 .. 0     |
| elbow (shoulder-elbow-wrist), 180 = straight | `elbow` | 180 - body   | -160 .. 160   |
| wrist (elbow-wrist-index), 180 = straight | `wrist_1` | 180 - body   | -180 .. 180   |

So an arm held horizontally with a straight elbow gives the UR home pose
`shoulder=-90, elbow=0`. Offsets and signs are constants in `config.py`;
the ranges are the `--robot-*-range` flags.

### Serial protocol

The serial backend is a generic fallback for a microcontroller-driven arm. It
needs the `serial` extra (`pip install -e .[serial]`) and a port:

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
| `?W`  | wrist 1 angle in degrees                      |
| `?G`  | gripper, `1` close / `0` open, omitted = hold  |
| `L`   | lift mode, `1` on / `0` off                    |

Joints without a reliable angle in the current frame are omitted. The wire
format lives in `encode_targets` in `vision_robot_arm/robot/serial_backend.py`
and is expected to change once the hardware is chosen.

Current mapping (`vision_robot_arm/robot/mapping.py`), applied to each arm:

- shoulder angle (elbow-shoulder-hip) -> UR `shoulder`
- elbow angle (shoulder-elbow-wrist) -> UR `elbow`
- wrist angle (elbow-wrist-index finger) -> UR `wrist_1`
- `<side>_fist` -> `gripper=close`
- `<side>_hand_open` -> `gripper=open`
- `right_hand_up` -> `lift_mode=on`

UR joint angles are clamped to `--robot-shoulder-range` (default `-180 0`),
`--robot-elbow-range` (default `-160 160`) and `--robot-wrist-range` (default
`-180 180`) and changes smaller than `--robot-deadband` degrees (default
`1.5`) are ignored to suppress jitter. When neither gripper gesture is active
the gripper keeps its previous state.
