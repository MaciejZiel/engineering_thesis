# Vision Robot Arm

Prototype for recognizing a person from a webcam, drawing a custom stick
figure, and printing joint data that can later feed a robot arm controller.

## Status

Current version:

- opens a webcam with OpenCV
- detects a single human pose with MediaPipe Pose Landmarker
- draws a custom skeleton with a horizontal hip line
- prints joint angles, raw pose landmarks, or both
- lets you switch data modes while the camera is running

## Setup

Use Python 3.12 from this project environment.

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

If the default camera is not correct:

```powershell
python main.py --camera 1
```

Useful options:

```powershell
python main.py --camera 0 --width 1280 --height 720 --print-interval 0.5
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

```text
main.py                  # thin entrypoint
vision_robot_arm/
  app.py                 # camera loop and module wiring
  cli.py                 # command-line arguments
  config.py              # runtime settings and defaults
  drawing.py             # custom stick figure and overlay
  landmarks.py           # landmark lookup and visibility checks
  metrics.py             # joint angle calculations
  output.py              # console printing modes
  pose_tracker.py        # MediaPipe Pose Landmarker wrapper
  runtime.py             # optional dependency loading
models/
  pose_landmarker_lite.task
tests/
  test_metrics.py
```

## Controls

- `1`: print joint angles
- `2`: print raw landmarks
- `3`: print angles and raw landmarks
- `q` or `Esc`: quit

## Tests

```powershell
python -m unittest discover -s tests
```

## Printed Data

Angle mode prints values such as elbows, shoulders, hips, knees, and ankles.
The angle is calculated from three body points, with the middle point as the
joint.

Landmark mode prints MediaPipe's 33 pose points with normalized image
coordinates (`x`, `y`, `z`) and `visibility`. When available, world coordinates
are printed too.
