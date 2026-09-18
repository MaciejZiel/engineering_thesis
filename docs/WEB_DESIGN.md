# Motion Twin web interface

## Product truth

Motion Twin is a local laboratory tool for a single operator and two UR7e arms.
Python owns acquisition, tracking, calibration, recording, and robot logic.
Body coordinates are metres: X right, Y forward, Z up. MediaPipe depth is an
estimate. Simulation is always distinguished from physical feedback.
The web launcher's initial integration is simulation only; hardware operation
remains in the existing desktop entry point until separately commissioned.

## Design direction

A quiet graphite instrument workspace, with a warm copper accent and a large
spatial canvas. Borrow desktop density and precise grouping from professional
editors. Use a narrow navigation rail, compact title bar, split workspace, and
an inspector; avoid repeated metric cards. Robot geometry is the focal point.

Tokens live in frontend/src/styles.css. System typography avoids network font
dependencies; numbers use tabular figures. Lucide is the sole icon family.
Controls use 6px corners, surfaces use 8px corners, and panel boundaries are
single-pixel rules. Green denotes available tracking, amber simulation or
attention, and red recording/errors. Labels always accompany status colour.

Workspace, calibration, session recording, diagnostics, and view preferences
share navigation, buttons, status treatment, and empty states. Native dialogs
provide focus containment and Escape dismissal. Keyboard shortcuts are ignored
while editing fields. Reduced-motion preferences suppress transitions.

## Audit and preservation

The existing OpenCV interface fits the camera well but gives little space to 3D;
many calibration actions are keyboard-only. Preserve the tracker, kinematics,
hand gestures, recording, and calibration calls. Introduce an optional web
presentation boundary, with bounded state publication, rather than duplicating
tracking. The old entry point remains available during migration.

## Verification

Inspect actual browser rendering at 1440×900, 1280×720, and 1024×768, plus narrow
windows. Exercise navigation, calibration/recording dialogs, display preferences,
3D presets, keyboard focus, disconnection, and no-pose states. Demonstration data
must be labelled and cannot execute calibration or recording commands.
