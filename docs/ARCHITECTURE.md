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
core    -> (stdlib only)
vision  -> core
robot   -> core
app.py  -> core, vision, robot
cli.py  -> core, app
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

The robot side consumes `PoseState` through the `RobotController` protocol
(`update(state)`, `close()`). The default controller does nothing; the debug
controller prints mapped commands. Real hardware plugs in behind the same
protocol without touching the vision side.
