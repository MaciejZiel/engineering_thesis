# Ownership and workflow

Two people work on this repository at the same time and commit directly to
`main`. The package is split so that day-to-day work rarely touches the same
file.

## Who owns what

| Path                            | Area   | Rule                                                        |
| ------------------------------- | ------ | ----------------------------------------------------------- |
| `vision_robot_arm/vision/`      | vision | camera, pose detection, smoothing, gestures, drawing, CSV   |
| `tests/test_vision_*.py`        | vision |                                                             |
| `vision_robot_arm/robot/`       | robot  | pose to joint mapping, robot backends, hardware protocol    |
| `tests/test_robot_*.py`         | robot  |                                                             |
| `vision_robot_arm/core/`        | shared | contract between the areas, change only after agreeing      |
| `vision_robot_arm/app.py`       | shared | wiring only, keep edits small                               |
| `vision_robot_arm/cli.py`       | shared | add flags for your own area, keep existing flags working    |
| `README.md`, `docs/`            | shared | update the section for the area you changed                 |
| `requirements.txt`, `pyproject` | shared | mention dependency changes to the other person              |

If you must edit a file owned by the other area (for example a small hook in
`drawing.py`), keep the change minimal, put it in its own commit and tell the
other person the same day.

## Workflow

1. Pull before you start and before every push:

   ```powershell
   git pull --rebase
   ```

2. Run the tests before every push:

   ```powershell
   python -m unittest discover -s tests
   ```

3. One area per commit. Imperative subject line, as in the existing history:
   `Add simulation robot backend`, `Fix angle smoother alpha`.

4. Push small commits often. Never rewrite history that is already on
   `origin/main`.

5. A change to `core/` or to a file owned by the other area is announced before
   it is pushed, so the other person can commit or stash their work first.
