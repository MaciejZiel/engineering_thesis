# Architecture

## Data flow

```text
camera / video file
      |
      v
PoseTracker            vision/pose_tracker.py   MediaPipe Pose Landmarker wrapper
      |  raw landmarks
      v
PoseStateBuilder       vision/state_builder.py  smoothing, angles, calibration, gestures
      |  PoseState (core/pose_state.py)
      +-----------------------------+-----------------------------+
      v                             v                             v
console / CSV                 stick figure + overlay          RobotController
vision/output.py              vision/drawing.py               robot/controller.py
vision/recording.py
```

`app.py` is the composition root. It opens the video source, builds the objects
above and runs the frame loop. `cli.py` parses command-line flags into a frozen
`AppConfig` and hands it to `run_app`.

## Layers and import rules

```text
core    -> (stdlib only, plus robot/config.py for RobotConfig)
vision  -> core
robot   -> core
app.py  -> core, vision, robot
cli.py  -> core, robot/config.py, app
```

- `core` is the shared contract: `AppConfig`, `PoseState`, `LandmarkPoint`,
  the optional dependency loader and `hud.py` with the translucent panel and
  text primitives that both the vision overlay and the robot panel use. Text is
  drawn in a single pass on a translucent background; the two-pass outline
  technique renders misaligned on OpenCV 5. Sizes scale with frame height
  (`ui_scale`, 720 px = 1.0). Changing anything in `core` affects both the
  vision and the robot side, so agree on it first.
- `vision` never imports from `robot` and `robot` never imports from `vision`.
  The only thing they share is `PoseState`.
- Third-party libraries (OpenCV, MediaPipe) are loaded once in
  `core/runtime.py` and passed into functions as arguments. Pure-logic modules
  therefore import and test without the heavy dependencies installed.
- User-facing errors are raised as `SystemExit` with a message that names the
  command-line flag to fix.

## PoseState contract

`PoseState` is a frozen dataclass produced once per frame:

| Field             | Meaning                                                    |
| ----------------- | ---------------------------------------------------------- |
| `timestamp_ms`    | frame timestamp, strictly increasing                       |
| `landmarks`       | 33 smoothed image landmarks (normalized x, y, z, visibility) |
| `world_landmarks` | smoothed world landmarks in metres, or `None`              |
| `raw_angles`      | joint angles before smoothing                              |
| `angles`          | smoothed joint angles in degrees, `None` when unreliable   |
| `relative_angles` | `angles` minus the calibrated neutral pose                 |
| `gestures`        | tuple of rule-based gesture names                          |
| `calibrated`      | whether a neutral pose has been captured                   |

Angle and gesture names are lowercase snake case (`right_elbow`,
`right_hand_up`). Landmark keys are the uppercase MediaPipe names
(`RIGHT_ELBOW`).

## Robot side

```text
PoseState -> RobotMapper -> JointTargets -> RobotBackend
             robot/mapping.py  robot/targets.py  robot/backend.py
```

- Target hardware: two Universal Robots UR7e cobots. `robot/targets.py` uses
  the UR joint names (`base`, `shoulder`, `elbow`, `wrist_1`, `wrist_2`,
  `wrist_3`); only shoulder, elbow and wrist_1 are driven by the body, the
  rest stay at the UR home pose `[0, -90, 0, -90, 0, 0]` deg.
- `RobotMapper` (`robot/mapping.py`) converts body angles into UR joint angles
  through `JointMapping` (offset, sign, limit; defaults in `robot/config.py`),
  clamps them to UR7e ranges and applies a dead-band so tiny changes do not
  reach the hardware. It knows nothing about transport.
- `JointTargets` (`robot/targets.py`) is the robot-side value object: per arm
  the UR joint name to angle in degrees and an optional gripper command
  (`None` means keep the previous state), plus a lift-mode flag.
- `RobotBackend` (`robot/backend.py`) is the transport protocol:
  `send(targets)`, `status_lines()` for the overlay and `close()`. Backends:
  `DebugBackend` prints, `SimulationBackend` (`robot/simulation.py`) keeps two
  in-memory arms that move toward the targets with a speed limit,
  `URBackend` (`robot/ur_backend.py`) opens one TCP socket per cobot to the
  URScript interface (port 30002) and streams `servoj([...6 radians...], 0, 0,
  t, lookahead_time, gain)` lines, plus `set_tool_digital_out` for the gripper;
  `SerialBackend` (`robot/serial_backend.py`) is a generic fallback that
  writes one ASCII line per frame over pyserial. pyserial is an optional
  dependency loaded the same way as OpenCV and MediaPipe: missing module means
  a `SystemExit` with an install hint, never an import error at startup.
- The overlay hook: `draw_overlay(..., status_lines=...)` in
  `vision/drawing.py` appends whatever the active backend reports, so the
  robot side can show state on screen without touching drawing code.
- Every backend also exposes `robot_state() -> RobotState | None`: per arm
  (`right`, `left`) the current and target shoulder/elbow/wrist angles and
  the gripper, plus the lift-mode flag. `robot/visualization.py` renders it
  into the separate simulation window (two three-link arms) when
  `--test-mode` is on; `vision/drawing.py` labels the body joints with their
  angles in the same mode.
- Hand gestures: `vision/hand_tracker.py` wraps the MediaPipe Hand
  Landmarker, `vision/hand_gestures.py` matches each hand to the nearest pose
  wrist and classifies it as open or fist. The names (`right_fist`,
  `left_hand_open`, ...) are merged into `PoseState.gestures` by
  `PoseStateBuilder.build(extra_gestures=...)`, so the robot side sees them
  like any other gesture.
- `MappedRobotController` (`robot/controller.py`) glues a mapper to a backend
  and is what `app.py` talks to through the `RobotController` protocol.
- `create_robot_controller(RobotConfig)` (`robot/factory.py`) chooses the
  backend from `--robot-backend`.
- `RobotConfig` (`robot/config.py`) holds all robot settings and is embedded
  in `AppConfig` as `config.robot`. This is the one place where `core`
  imports from `robot`; `robot/config.py` depends on the standard library only.
