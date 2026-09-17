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

To list cameras or measure capture delivery without loading MediaPipe models or opening a window:

```powershell
python main.py --list-cameras
python main.py --diagnose-camera 0 --width 1920 --height 1080 --camera-format mjpg
```

Diagnostics honor `--camera-backend`, `--camera-format`, `--width`, `--height`,
and `--fps`. Use `--diagnostic-warmup 2 --diagnostic-seconds 5` to adjust the
warmup and measurement windows. Each invocation measures one requested mode;
driver-reported settings and delivered frame dimensions are shown separately.
No images are stored, exposure is not changed, and robot connections are not
opened. Read/interval p95 timings describe host delivery, **not** sensor-to-display
latency. A low FPS result is a valid measurement, not proof of a particular cause.
Native driver reads can block beyond the measurement window; it is not a hard
process timeout.

Capture preserves the driver's default buffer count. Forcing a single V4L2
buffer reduced the tested USB camera from approximately 29.6 to 14.8 FPS at
1080p MJPG. Removing that override restored 29.6 FPS in the application's capture
path. This device-specific result does not establish end-to-end UI latency or
performance on other cameras.

To process a saved video instead of the webcam:

```powershell
python main.py --video sample.mp4
```

To loop a video while testing:

```powershell
python main.py --video sample.mp4 --loop-video
```

The camera is asked for 1920x1080 by default. A smaller or differently shaped
camera image keeps its native resolution and aspect ratio, so 4:3 cameras are
never stretched to 16:9. Frames larger than the configured bounds are reduced
proportionally. Pick a smaller processing limit for slow machines or `0 0` to
keep every source at its native size:

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

The filter reads how fast a joint is actually moving, averaged over several frames.
Landmark noise averages to nothing, so a joint you hold still is held still; a joint
that is sweeping gets the full response set by the alpha above. Measured against 1.5
degrees of landmark noise, a still joint settles from 0.73 to 0.29 degrees of jitter
and stops being re-commanded to the robot at all, while the lag at 60 degrees per
second is unchanged. Tracked hands are smoothed the same way before they are drawn
or measured.

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
`--no-hands` on slow machines. With `--no-hands` there are no gripper gestures,
and the wrist angle falls back to the pose model (elbow-wrist-index), which is
unsigned: the robot wrist can then only bend one way, and `angle_*_wrist` in a
recording means something different from a run with hand tracking on.

### 3D tracking data

Robot-driving joint angles require MediaPipe world landmarks in the live
application. Shoulders and elbows use pose-world 3D vectors; wrist flexion uses
the metric hand-world direction and its palm hinge axis. There is no silent 2D
fallback for control when world output is missing. Image landmarks remain in use
for visibility checks, hand-to-arm association and the camera overlay.

Hand world landmarks have a hand-local origin. For recording and future 3D
visualization, each hand is translated so its wrist coincides with the pose-world
wrist. The stored frame is named `pose_wrist_anchored_m`; it preserves the metric
hand shape without pretending that the hand model provides an independent global
position. CSV recordings include image and anchored-world coordinates for all 21
landmarks on both hands plus an angle-source column. A held wrist measurement is
explicitly marked `hand_world_3d_held`.

Pose-world coordinates are body-relative model estimates, not camera-space depth.
They support 3D joint orientation, including motion toward and away from the
camera, but cannot measure the operator's absolute distance from the robots. That
later requires registered depth from the target RGB-D camera and calibration into
the robot/table coordinate system.

## Dashboard UI

### Hardware control lifecycle

Calibration (**C**) now requires stable measurements over at least 600 ms in an
800 ms window, with all three mapped angles for each configured arm (both arms
in preview). The captured human pose maps to the robot's configured home;
subsequent motion uses signed angular offsets. Calibration changes pause hardware
control. **K** saves the profile to `recordings/calibration.json`, **L** loads it,
and **X** resets calibration. Loading never automatically enables motion. Profiles
must be recaptured when the camera/operator setup changes. Without calibration,
the existing absolute angle mapping is retained.

Selecting the UR backend in the application no longer connects or moves an arm
at startup. It defaults to **monitor** mode, where the backend has no motion
methods and only reads RTDE/dashboard telemetry. In tracking mode the hardware
button (keyboard **H**) advances one explicit step at a time:
**Connect → capture the stationary current pose (no motion) → enable control**.
There is no automatic trip to a hard-coded home position. **P** or the red
**Stop motion** button pauses motion; resuming requires another explicit action.
Missing any mapped joint on a configured arm pauses the session. Returning into
view does not automatically resume control. A fault closes both connections and
requires restarting the session. Interactive UR motion requires fresh RTDE
position and velocity feedback; it does not silently continue open-loop.

These software checks are not a hardware emergency stop or collision avoidance.
Do not enable motion without the laboratory's approved workspace and procedures.
The lower-level backend retains its legacy home method for tests and API
compatibility; the interactive application does not call it.

The application uses one resizable OpenCV window named **Motion Twin**. The
camera feed occupies the main area. A graphite sidebar contains a compact
schematic of both UR7e arms and the session's tracking, calibration, recording
and gesture state. Backend selection is labelled explicitly: **Simulation**,
**Preview only**, or the configured output transport. These labels do not
claim a verified hardware connection or measured robot feedback.

The bottom bar prioritizes calibration and recording. Controls respond to hover,
press and keyboard focus; use **Tab** and **Enter** or **Space** to activate
them. Calibration is disabled until a pose supplies at least one usable angle.
Click **Details** in the session panel (or press **d**) for backend diagnostics.
The **Console** control changes terminal output, not the camera display.

Lato fonts are bundled under the SIL Open Font License and rendered with Pillow,
which is already included in `requirements-lock.txt` and is now an explicit
runtime dependency. No system font installation is needed. Pillow caches text
masks and blends only their bounds so the video frame is not converted per label.

On Windows the process enables per-monitor DPI awareness and reads the usable
pixel area of the primary display. The initial dashboard uses 90% of that area
and its render surface follows the real OpenCV viewport after resizing or
entering fullscreen. UI text and line work are therefore rendered directly at
the target resolution instead of being stretched from a fixed 1920x1080
bitmap. The responsive layout is intended for HD/FHD, QHD, ultrawide and 4K
displays; low-resolution camera frames are enlarged with bilinear interpolation
and never have their aspect ratio changed.

The visual design follows the lab setup described by the PJATK ARM Robotics
program: two UR7e cobots, an NVIDIA Jetson Orin AGX module and an Orbbec Gemini
335Lg 3D camera. A muted orange highlights the human arms and simulated or
commanded robot pose; grey indicates targets. The preview is a schematic,
not a physical robot feedback display.

Press `f` or click **FULLSCREEN** for presentation mode. Closing the window,
pressing `q`/`Esc`, or clicking **QUIT** shuts down the trackers, recorder,
camera and robot backend cleanly.

Generate camera-free previews for visual review (sample data, no robot connection):

```bash
python scripts/preview_dashboard.py
python scripts/preview_dashboard.py --size 1440 900
```

The script saves tracking, waiting, recording and diagnostics states under
`artifacts/ui-review/` (git-ignored).

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
    arm_pose.py            # arm elevation that works without visible hips
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
    ur_dashboard.py        # dashboard readiness check (port 29999)
    ur_rtde.py             # RTDE feedback client (port 30004)
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
  test_robot_ur_dashboard.py
  test_robot_ur_rtde.py
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
angles in degrees plus a gripper command, and one lift-mode flag. In simulation,
`base`, `wrist_2` and `wrist_3` begin at the preview home pose. On hardware they
remain at the feedback-confirmed pose captured before control is enabled. Your
right arm drives the right cobot and your left
arm the left one. The targets go to a backend selected with
`--robot-backend`:

| Backend  | What it does                                                    |
| -------- | --------------------------------------------------------------- |
| `none`   | default, robot side disabled                                    |
| `debug`  | prints the mapped targets to the console                        |
| `sim`    | two simulated UR7e arms with speed limit, shown in the test window |
| `ur`     | real UR7e cobots: ramped `servoj` over TCP plus RTDE feedback   |
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

### Safe first UR7e connection

Do not begin with two-arm tracking. Verify the physical installation, configured
safety planes/joint limits and accessible teach-pendant emergency stop according
to the laboratory procedure first. The checks below are application safeguards,
not safety-rated robot functions.

Start with the read-only monitor. This opens dashboard and RTDE connections but
cannot send `movej`, `servoj`, `stopj` or tool-output commands:

```powershell
python main.py --robot-backend ur --robot-operation monitor --robot-right-host 192.168.1.10
```

Press **H** once to connect. Confirm in **Details** that the expected controller
IP, software version, joint positions, TCP speed, joint speed, speed scaling and
safety state are shown. Repeat separately for the other robot. Monitor mode is
the default when `--robot-operation` is omitted.

Only after the read-only check succeeds, test one joint on exactly one robot.
Put that controller in **Remote Control** and **REDUCED** safety mode, clear the
workspace, select a low pendant speed slider and run:

```powershell
python main.py --robot-backend ur --robot-operation commissioning `
  --robot-right-host 192.168.1.10 `
  --robot-commissioning-joint shoulder `
  --robot-commissioning-speed 2 `
  --robot-commissioning-excursion 2
```

Press **H** once to connect and a second time to capture the stationary current
pose. Neither action commands motion. Then hold the on-screen `−`/`+` button or
the **[**/**]** key to jog. Releasing it stops refresh; a 150 ms watchdog sends
`stopj`. The default is limited to 2 deg/s and ±2 degrees from the captured
origin. Hard validation prevents commissioning above 5 deg/s or ±5 degrees and
prevents specifying two robot hosts. Press **P**, **H**, or **Stop motion** to
disarm commissioning.

If direction, joint identity, feedback, stopping or visualization is wrong, stop
there and fix it before testing the next joint. Repeat with `base`, `elbow`,
`wrist_1`, `wrist_2` and `wrist_3`, one at a time.

### Vision tracking over URScript

Full tracking is selected explicitly:

```powershell
python main.py --robot-backend ur --robot-operation tracking `
  --robot-right-host 192.168.1.10 --robot-left-host 192.168.1.11 `
  --robot-max-speed 5
```

This mode is not the first hardware test. It still needs validation of the lab's
actual base transforms, workspace/collision constraints and final real-time
command transport before it should control two physical arms around a shared
table. Start with simulation and recorded motion, then one physical arm at low
speed, then two arms only under the approved lab procedure.

The hardware session performs these checks in order:

1. **Readiness check** on the dashboard server (`--robot-dashboard-port`,
   default `29999`). Local control, a robot mode other than `RUNNING` or a
   safety stop end the run with a message naming the problem instead of a
   silently motionless arm. An unreachable dashboard also ends the run.
2. **Feedback capture**: fresh RTDE position and velocity are required and every
   configured arm must be stationary. The actual joint pose becomes the initial
   setpoint without sending a motion command.
3. **Explicit enable**: another operator action is required before the first
   mapped target can be sent.
4. **Streaming**: one line per cobot every `--robot-send-interval` seconds:

```text
servoj([0.0000, -0.7854, 1.5708, -1.5708, 0.0000, 0.0000], 0, 0, 0.050, 0.100, 300)
```

The six values are the joint angles in radians in UR order (base, shoulder,
elbow, wrist 1, wrist 2, wrist 3). The remaining parameters are `a` and `v`
(ignored by `servoj`), `t` (`--robot-send-interval`), `lookahead_time`
(`--robot-servo-lookahead`, default `0.1`) and `gain` (`--robot-servo-gain`,
default `300`).

**The streamed pose is a ramped setpoint, not the raw mapped angle.** Each arm
runs the same motion model as the simulator, so the commanded pose moves toward
your arm at most `--robot-max-speed` degrees per second (default `60`, UR7e
limit `180`). A tracking glitch therefore cannot ask the controller for a jump
of tens of degrees.

**Feedback**: the backend opens an RTDE connection (`--robot-rtde-port`,
default `30004`) and reads joint position/velocity, TCP pose/velocity, robot and
safety state, speed scaling and runtime state. The arm preview and status lines
then show where the robot actually is; the grey ghost stays the commanded target.
Monitor mode remains read-only. Commissioning and tracking refuse to start when
feedback or dashboard preflight is disabled.

**Shutdown** sends `stopj`, so closing the window decelerates the arms instead
of leaving the last `servoj` running.

The gripper is driven through a tool digital output
(`--robot-tool-output`, default `0`; `set_tool_digital_out(0, True)` = close).
Swap `encode_gripper` in `vision_robot_arm/robot/ur_backend.py` for the URCap
call of the gripper the lab mounts (for example Robotiq). Do not test the gripper
until its electrical interface and safe output state have been verified.

Protocol and safety references: the official UR documentation describes
[RTDE](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html),
the [primary/secondary interfaces](https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/primary-secondary-guide.html),
[`servoj`](https://www.universal-robots.com/manuals/EN/HTML/SW5_24/Content/prod-scriptmanual/all_scripts/servoj_qavt0-008lookahead_time.htm),
and the controller's [safety parameter set](https://www.universal-robots.com/manuals/EN/HTML/SW10_6/Content/prod-usr-man/hardware/arm-e-Series/UR5e/H_g5_sections/safetyFunctionsAndinterfaces/safety_parameter_set.htm).

Body angle to UR joint mapping (`vision_robot_arm/robot/config.py`,
`JointMapping`):

| Body angle (deg)                        | UR joint  | Formula        | Default range |
| --------------------------------------- | --------- | -------------- | ------------- |
| shoulder elevation, 0 = down, 180 = up | `shoulder`| body - 180     | -180 .. 0     |
| elbow (shoulder-elbow-wrist), 180 = straight | `elbow` | 180 - body   | -160 .. 160   |
| wrist (3D forearm vs hand, signed), 180 = straight | `wrist_1` | 180 - body | -180 .. 180 |

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

- shoulder elevation (shoulder-to-elbow against the torso, or against image
  vertical when the hips are out of frame): 0 = arm down, 90 = horizontal,
  180 = raised -> UR `shoulder`
- elbow angle (shoulder-elbow-wrist) -> UR `elbow`
- wrist angle: signed angle in the image plane between the forearm (pose elbow to
  wrist) and the hand (hand-tracker wrist to middle knuckle); 180 = straight,
  below 180 = bent one way, above 180 = the other; held for 0.5 s when the hand
  tracker drops a frame -> UR `wrist_1`. A hand pointing straight at the camera
  is left unmeasured, because its projection is too short to have a direction.
  MediaPipe depth is deliberately not used here: the pose model and the hand
  model measure z from different origins, so mixing them moved the reported
  angle by tens of degrees with no real motion.
- `<side>_fist` -> `gripper=close`
- `<side>_hand_open` -> `gripper=open`
- `right_hand_up` -> `lift_mode=on`

Lift mode is a flag, not a motion: it is shown on the dashboard and sent in the
serial frame as `L:`, but the `ur` backend does not act on it. It is the hook for
whatever the lab decides it should do (a second tool output, a URCap call), and
nothing moves because of it today.

UR joint angles are clamped to `--robot-shoulder-range` (default `-180 0`),
`--robot-elbow-range` (default `-160 160`) and `--robot-wrist-range` (default
`-180 180`) and changes smaller than `--robot-deadband` degrees (default
`1.5`) are ignored to suppress jitter. When neither gripper gesture is active
the gripper keeps its previous state.
