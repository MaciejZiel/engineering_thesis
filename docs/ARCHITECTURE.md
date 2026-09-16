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

- `core` is the shared contract: `AppConfig`, `PoseState`, `LandmarkPoint` and
  the optional dependency loader. Changing anything in `core` affects both the
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

- `RobotMapper` (`robot/mapping.py`) picks the source angles and gestures,
  clamps them to the configured `JointLimit`s and applies a dead-band so tiny
  changes do not reach the hardware. It knows nothing about transport.
- `JointTargets` (`robot/targets.py`) is the robot-side value object: joint
  name to angle in degrees, an optional gripper command (`None` means keep
  the previous state) and a lift-mode flag.
- `RobotBackend` (`robot/backend.py`) is the transport protocol:
  `send(targets)`, `status_lines()` for the overlay and `close()`. Backends:
  `DebugBackend` prints, further backends (simulation, serial) plug in here.
- `MappedRobotController` (`robot/controller.py`) glues a mapper to a backend
  and is what `app.py` talks to through the `RobotController` protocol.
- `create_robot_controller(RobotConfig)` (`robot/factory.py`) chooses the
  backend from `--robot-backend`.
- `RobotConfig` (`robot/config.py`) holds all robot settings and is embedded
  in `AppConfig` as `config.robot`. This is the one place where `core`
  imports from `robot`; `robot/config.py` depends on the standard library only.
