# Web Frontend Migration Plan

## 1. Purpose

This document describes how to replace the OpenCV-rendered operator interface
with a production-quality local web frontend while keeping camera processing,
pose tracking, robot mapping, safety checks, recording, and UR communication in
Python.

The migration must be incremental. The existing desktop application remains a
working fallback until the web interface has passed the same functional and
safety checks. No phase is allowed to make robot motion depend on browser timing
or browser availability.

## 2. Product goals

The finished application should provide:

- a clean, dark operator dashboard suitable for a laboratory demonstration;
- a low-latency camera preview with the tracked skeleton overlay;
- an interactive 3D digital twin of the operator and both UR7e arms;
- clear distinction between measured robot state, commanded targets, and
  simulated state;
- visible tracking quality, latency, connection, operation, and safety status;
- guided body/skeleton calibration and saved operator profiles;
- simulation, monitor, commissioning, and tracking workflows;
- configuration of camera, tracking, workspace, and robot limits;
- session recording, playback, and useful diagnostics;
- graceful recovery from camera, tracking, browser, network, and robot failures;
- operation on the target laboratory computer without an Internet connection.

The first release is a local application. Remote access from another computer
or tablet is a later feature and must not be enabled by default.

## 3. Non-goals

- Rewriting pose tracking or robot control in JavaScript.
- Connecting the browser directly to a camera device or a UR controller.
- Making the frontend part of the real-time robot control loop.
- Allowing the frontend to bypass backend safety validation.
- Removing the current dashboard before the replacement is verified.
- Solving absolute depth from a single RGB camera in the UI. The frontend can
  show the estimate and its confidence, but cannot make it physically accurate.

## 4. Recommended technology

### Backend

- Existing Python tracking and robot packages remain the source of truth.
- FastAPI exposes the local HTTP API and WebSocket streams.
- Pydantic models define and validate API messages at the boundary.
- OpenCV/MediaPipe remain responsible for capture and inference.
- The existing robot backends remain responsible for simulation and UR RTDE.

### Frontend

- React with TypeScript.
- Vite for development and production builds.
- Three.js through React Three Fiber for the digital twin.
- A small, explicit state layer; start with React state and selectors and add a
  dedicated store only when data flow justifies it.
- CSS variables and component-level CSS for the visual system. A component
  library may be used selectively, but the operator interface should not look
  like a generic administration template.
- Vitest and Testing Library for component and protocol tests.
- Playwright for critical operator journeys and screenshots.

Pin frontend and backend dependency versions. Commit the lock file and make the
production build reproducible.

## 5. Target architecture

```text
camera / video
      |
      v
Python capture and tracking loop
      |
      +--> PoseState --> RobotMapper --> safety gate --> RobotBackend --> UR7e
      |                                       |
      |                                       +--> RobotState / safety state
      |
      +--> immutable UI snapshot
                    |
                    v
          local FastAPI application
             |                |
          HTTP API        WebSocket streams
             |                |
             +-------+--------+
                     v
           React operator frontend
             |              |
       camera canvas    Three.js digital twin
```

The Python domain objects must not be serialized directly. A dedicated adapter
converts them into versioned API models. This prevents an internal refactor from
silently breaking the frontend.

The backend owns all mutable operational state. Reloading the page must restore
the current status from a fresh backend snapshot rather than resetting the
tracker or robot controller.

## 6. Process and threading model

Initially, run the tracking loop and web server in one Python process with
separate, bounded execution contexts:

- capture/inference produces the newest immutable snapshot;
- robot communication owns its existing worker/control lifecycle;
- the API reads the latest snapshot without blocking capture;
- slow or disconnected clients never create an unbounded queue;
- each WebSocket client receives the newest state and may skip stale frames.

Prefer a latest-value buffer for high-rate telemetry. Do not enqueue every pose
frame when the browser cannot keep up. Configuration changes and operator
commands are low-rate reliable events and use a separate command path.

Splitting the API and tracking engine into separate processes is an optional
later optimization. It should only be introduced after profiling demonstrates a
need, because it adds deployment and lifecycle complexity.

## 7. API contract

Version the API from the start under `/api/v1` and include `schema_version` in
WebSocket envelopes.

### HTTP endpoints

The initial read-only API should include:

- `GET /api/v1/health` -- process readiness and build version;
- `GET /api/v1/snapshot` -- latest complete UI state;
- `GET /api/v1/config` -- public, non-secret effective configuration;
- `GET /api/v1/profiles` -- available calibration/operator profiles;
- `GET /api/v1/recordings` -- recorded sessions and metadata.

Later command endpoints should express intent rather than raw robot protocol:

- `POST /api/v1/calibration/pose/start`;
- `POST /api/v1/calibration/skeleton/start`;
- `POST /api/v1/calibration/cancel`;
- `POST /api/v1/recording/start` and `/stop`;
- `POST /api/v1/operation/arm`;
- `POST /api/v1/operation/start`;
- `POST /api/v1/operation/stop`;
- `POST /api/v1/config/validate` and, later, `/apply`.

Every command returns a command identifier and the backend decision. Commands
that are rejected include a stable reason code plus a human-readable message.
Robot motion must never be exposed as a generic endpoint accepting arbitrary
joint angles from the browser.

### Telemetry WebSocket

Use one initial endpoint, `/api/v1/ws/telemetry`, with typed envelopes:

```json
{
  "schema_version": 1,
  "sequence": 1842,
  "server_time_ms": 1758156000123,
  "type": "state",
  "payload": {}
}
```

The state payload should contain only data needed for rendering:

- frame identifier and capture/inference timestamps;
- display FPS, inference FPS, and end-to-end age;
- normalized 2D landmarks and body-relative 3D landmarks;
- landmark visibility/confidence;
- hand landmarks and gestures;
- calculated joint angles and measurement sources;
- calibration state and progress;
- mapped Cartesian targets;
- per-arm measured joints, target joints, TCP, connection freshness, and source;
- operation mode, safety state, warnings, and blocking reasons;
- recording state.

Use stable string enums and explicit units in field names or schema descriptions.
Coordinates must declare their frame (`image`, `body`, `robot_base`) and units.
Never send an unlabeled `x/y/z` tuple across the boundary.

### Protocol rules

- Generate TypeScript types from the backend schema or validate shared fixture
  files in both languages.
- Additive optional fields are backward-compatible within one schema version.
- Removing, renaming, or changing the meaning of a field requires a new version.
- Unknown message types must be ignored safely by older clients.
- Every stream carries monotonic sequence numbers so the UI can detect gaps.
- Stale robot and tracking data are represented explicitly, never as fresh data
  with repeated values.

## 8. Camera delivery

Do not send full-resolution base64 JPEG images inside telemetry JSON.

Implement camera delivery in stages:

1. Start with an MJPEG endpoint because it is simple and robust for a local
   read-only preview.
2. Measure capture-to-display latency and CPU usage at 720p and 1080p.
3. If MJPEG is too expensive or latent, move video to WebRTC while keeping pose
   and robot telemetry on WebSocket.

The overlay should preferably be rendered in the browser on a canvas above the
video. This avoids encoding a second annotated video, permits toggling layers,
and keeps labels sharp. Each pose update must reference the source frame ID so
the frontend can avoid drawing a new skeleton over an unrelated old frame.

Provide separate quality settings for processing resolution and preview
resolution. Preview quality must never silently reduce tracking quality.

## 9. 3D digital twin

The Three.js scene should contain:

- two UR7e models loaded from reviewed local assets;
- current robot pose and a visually distinct target pose;
- TCP position and target marker;
- tracked torso, arms, and hands;
- robot bases, table/workspace boundaries, floor grid, and coordinate axes;
- optional safety/workspace volumes;
- labels shown on selection rather than permanently cluttering the scene.

Required camera presets:

- perspective;
- front;
- top;
- left/right side;
- reset/focus selection.

The user may orbit and zoom in perspective mode. Diagnostic projections should
use deterministic cameras so screenshots remain comparable.

Coordinate transformations stay in Python where possible. The API provides
points in documented frames; the frontend only applies the final conversion to
the Three.js convention. Add golden fixtures for neutral pose, left/right reach,
forward/back reach, and vertical reach to prevent axis or mirroring regressions.

Mirroring is a presentation option. It must never change the semantic meaning
of `left`, `right`, or robot coordinates.

## 10. User experience and information architecture

### Main operator screen

- Top status bar: application state, camera, tracking, both robots, mode, FPS,
  latency, and the most important warning.
- Main left panel: camera preview and optional 2D overlay.
- Main right panel: interactive 3D digital twin.
- Compact telemetry strip: left/right hand XYZ, joint confidence, target age,
  and measured-versus-commanded state.
- Bottom action bar: simulation, calibration, recording, arm/start/stop controls.
- Expandable diagnostics drawer: raw values, logs, coordinate frames, and
  performance timings.

### Supporting screens

- Guided setup and camera selection.
- Skeleton and neutral-pose calibration wizard.
- Robot connection and read-only monitor screen.
- Limits/workspace configuration with validation and change summary.
- Recordings and playback.
- Diagnostics and exportable support bundle.

### Visual language

- Graphite background and restrained panels.
- One primary accent; reserve red, amber, and green for operational meaning.
- Consistent spacing, typography, radii, and elevation tokens.
- Minimum readable font sizes for the expected laboratory display.
- No status represented by colour alone.
- Motion is functional and subtle; disable it when reduced motion is requested.
- Controls unavailable in the current mode explain why they are disabled.

## 11. Safety boundary

The frontend is untrusted from the robot controller's perspective. Every action
is revalidated in Python.

Before accepting any motion-related request, the backend verifies at least:

- requested operation mode permits motion;
- both relevant robot connections and feedback are fresh;
- robot and safety modes allow the operation;
- tracking and calibration are valid and recent;
- targets are finite and inside joint and Cartesian limits;
- velocity and acceleration constraints are satisfied;
- workspace restrictions and collision policy are satisfied;
- no stop, fault, or protective-stop state is active;
- an explicit arming step has completed.

Required failure behavior:

- loss of camera/tracking: hold or controlled stop according to the approved
  robot policy;
- stale RTDE feedback: stop producing motion commands;
- backend/frontend WebSocket loss: browser shows disconnected immediately;
  backend behavior must not depend solely on the browser connection;
- backend shutdown: controller executes the existing controlled-stop path;
- invalid frontend data: reject, log, and remain in the current safe state.

The browser STOP control is useful but is not a replacement for the physical
emergency stop. Its wording and visual treatment must not imply otherwise.

Bind the first version to loopback (`127.0.0.1`) only. Before any LAN access,
add authentication, authorization, TLS or a trusted reverse proxy, origin
validation, and a documented network threat model.

## 12. Migration phases

Each phase should be delivered through small English-language commits. A phase
is complete only when its tests and acceptance gate pass.

### Phase 0 -- Baseline and decisions

Deliverables:

- record current startup time, camera FPS, inference FPS, latency, and CPU usage;
- capture reference screenshots and short test recordings;
- inventory all current controls, status lines, CLI options, and workflows;
- document coordinate frames and mirroring in one canonical place;
- correct any stale architecture documentation before building against it;
- create architecture decision records for the frontend stack and transport.

Acceptance gate: the current application and all tests still pass, and the team
agrees on the API vocabulary and coordinate conventions.

### Phase 1 -- Extract an application service

Deliverables:

- move orchestration out of the OpenCV dashboard loop into a testable service;
- expose immutable latest-state snapshots;
- represent lifecycle, calibration, recording, and operation as explicit state;
- keep the existing dashboard consuming the same service;
- add lifecycle and concurrency tests.

Acceptance gate: the old UI behaves as before and no web dependency is required
to run it.

### Phase 2 -- Read-only API

Deliverables:

- add FastAPI behind an explicit CLI option;
- implement health, snapshot, configuration, and telemetry endpoints;
- define versioned Pydantic schemas and sample fixtures;
- add bounded latest-value broadcasting and disconnect handling;
- bind to loopback by default;
- add API and WebSocket tests.

Acceptance gate: a test client can observe a full simulated session without
affecting tracking or robot state.

### Phase 3 -- Frontend foundation

Deliverables:

- create the React/TypeScript application and production build;
- establish design tokens, layout shell, routing, error boundary, and connection
  state;
- show system status and live telemetry from simulation;
- validate incoming messages rather than assuming their shape;
- add component tests, protocol fixtures, linting, and CI jobs.

Acceptance gate: reload, reconnect, backend restart, and malformed message tests
all produce understandable states without a blank screen.

### Phase 4 -- Camera and tracking UI

Deliverables:

- add the camera stream;
- render synchronized landmarks and confidence on a canvas overlay;
- expose layer toggles for raw/smoothed skeleton, hands, labels, and diagnostics;
- show frame age, inference FPS, dropped/gapped updates, and tracking quality;
- preserve source aspect ratio at all window sizes.

Acceptance gate: camera and overlay remain aligned at supported resolutions and
the measured latency is no worse than the agreed baseline budget.

### Phase 5 -- Three.js digital twin

Deliverables:

- render operator and both robot arms from shared golden fixtures;
- add current/target distinction, TCP markers, grid, axes, and workspace bounds;
- add perspective/front/top/side presets plus orbit controls;
- implement mirroring as presentation only;
- profile and cap rendering work when the tab is not visible.

Acceptance gate: automated screenshots and numeric fixture checks confirm axes,
left/right assignment, units, joint poses, and forward/back direction.

### Phase 6 -- Calibration and recording workflows

Deliverables:

- implement guided skeleton and neutral-pose calibration;
- show pose instructions, progress, quality problems, success, and cancellation;
- manage profiles without exposing arbitrary filesystem paths;
- implement recording controls and session metadata;
- add playback using the same UI state contract as live mode.

Acceptance gate: a new operator can complete calibration without terminal input,
and cancelling or losing tracking leaves the backend in a known state.

### Phase 7 -- Robot monitor mode

Deliverables:

- configure both robot addresses and test connectivity;
- display RTDE freshness, measured joints/TCP, runtime, and safety status;
- clearly label `SIMULATED`, `TARGET`, `MEASURED`, and `STALE` data;
- expose connection failures without retry storms;
- export useful diagnostics for failed laboratory connections.

Acceptance gate: run against both UR7e controllers in read-only mode. No motion
command path is reachable from the frontend.

### Phase 8 -- Commissioning controls

Deliverables:

- add explicit arming and hold-to-run behavior where appropriate;
- add validated low-speed commissioning actions;
- show backend rejection reasons and required recovery steps;
- implement command IDs, idempotency where needed, and an audit log;
- add watchdog and stale-command tests.

Acceptance gate: perform the written laboratory checklist at minimum configured
speed with one arm at a time, then both arms, with an observer at the physical
emergency stop.

### Phase 9 -- Tracking operation

Deliverables:

- expose tracking start/stop through intent-based commands;
- show preflight requirements and live limiting/clamping indicators;
- display tracking loss, target age, and robot-following error prominently;
- add controlled recovery rather than automatic unexpected restart;
- run recorded-session, simulation, and hardware test matrices.

Acceptance gate: all safety scenarios pass and hardware behavior matches the
approved direction, scale, speed, workspace, and stop criteria.

### Phase 10 -- Packaging and retirement

Deliverables:

- serve the built frontend from the Python application;
- provide one command/shortcut that starts backend and browser in kiosk mode;
- add clean shutdown, single-instance behavior, logs, and version information;
- document offline installation and recovery;
- retain the legacy dashboard behind a fallback flag for at least one validated
  release;
- remove legacy rendering only after feature parity and sign-off.

Acceptance gate: a clean target machine can install and run the application
offline using the documented procedure.

## 13. Testing strategy

### Python

- Unit tests for domain services and schema adapters.
- API and WebSocket tests without camera or robot hardware.
- Contract tests using frozen JSON fixtures.
- Concurrency tests for slow clients, reconnects, and shutdown.
- Safety-state tests for every rejected command and stale-data transition.

### Frontend

- Unit tests for transforms, selectors, formatting, and validation.
- Component tests for all loading, empty, stale, warning, and error states.
- Three.js numeric tests independent of screenshots.
- Screenshot tests for supported dashboard sizes.
- End-to-end tests for reconnect, calibration, recording, simulation, and monitor
  workflows.

### Integration and hardware

- Recorded pose sessions for repeatable tracking regressions.
- Simulation tests for target mapping and digital-twin agreement.
- Network interruption tests for camera, frontend, and each robot separately.
- One-arm commissioning before dual-arm commissioning.
- Written hardware checklist with date, software commit, configuration profile,
  observers, and results.

CI must build both applications, run both test suites, validate the generated
API contract, and produce the frontend bundle. Hardware tests remain a separate
manual or laboratory pipeline.

## 14. Performance budgets

Establish measured budgets during Phase 0. Initial targets to validate rather
than assume:

- camera delivery remains at the configured 30 FPS when supported by hardware;
- pose telemetry reaches the UI at 20--30 updates per second;
- robot/safety telemetry is displayed with an explicit age;
- the 3D view maintains 60 FPS on the target machine where practical;
- frontend work does not materially reduce Python inference FPS;
- memory remains bounded during long sessions and repeated reconnects;
- leaving the dashboard open for one hour does not grow queues or event
  listeners.

Measure capture, inference, serialization, transport, browser receipt, and render
times separately. A single averaged FPS number is not sufficient.

## 15. Configuration and persistence

- Keep secrets and machine-specific addresses out of frontend bundles.
- Backend configuration is authoritative and validated atomically.
- Separate factory defaults, machine configuration, operator profiles, and
  session data.
- Configuration writes use schema versions and atomic replacement.
- Show a diff and validation result before applying safety-relevant changes.
- Record the effective configuration and software commit with every session.
- Do not allow the browser to submit arbitrary output paths.

## 16. Observability

Use structured backend logs with timestamps, severity, subsystem, session ID,
command ID, and stable event codes. The UI should show concise operator messages
and keep technical details in the diagnostics drawer.

Record metrics for:

- capture and inference timing;
- dropped frames and telemetry gaps;
- tracking confidence and loss duration;
- WebSocket clients, reconnects, and send lag;
- RTDE feedback age and connection transitions;
- command acceptance/rejection;
- target limiting and following error.

A support bundle should include logs, public configuration, versions, recent
metrics, and optional recorded state, but no video unless the operator explicitly
chooses to include it.

## 17. Deployment

Development runs the Vite server and Python backend separately. Production uses
a static frontend build served locally by Python or a minimal local web server.
The application opens the local URL automatically and must work offline.

Do not introduce Electron initially. A normal browser or kiosk window keeps the
runtime smaller. Consider Tauri or another desktop shell only if later
requirements demand window management, auto-update, or OS integration that the
local web application cannot provide cleanly.

## 18. Definition of done

The web migration is complete when:

- one documented command starts the complete local application;
- the interface covers all approved current workflows without terminal input;
- simulation, measured state, targets, stale data, and faults are unmistakable;
- camera, tracking, and 3D views remain synchronized within the agreed budget;
- browser failure cannot bypass or disable backend safety behavior;
- both robot connections can be monitored and commissioned through the written
  laboratory procedure;
- automated Python, frontend, contract, and end-to-end tests pass;
- the application installs and operates offline on the target machine;
- the team has completed and signed off the hardware safety checklist;
- the legacy dashboard has remained available through at least one validated
  web release and can then be removed in a separate commit.

## 19. Recommended first implementation slice

The first coding slice should stop after read-only visibility:

1. Define `UiSnapshotV1` and its coordinate/unit conventions.
2. Extract a latest-state publisher from the current application loop.
3. Add `/api/v1/health`, `/api/v1/snapshot`, and read-only telemetry WebSocket.
4. Create the React shell with connection and stale-data states.
5. Display FPS, tracking state, body XYZ, and simulated robot state.
6. Add shared contract fixtures and CI checks.

This slice proves the architecture without touching robot commands, camera
encoding, or the current operator workflow. Camera streaming and Three.js should
start only after this boundary is stable.
