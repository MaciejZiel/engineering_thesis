# Production-readiness work plan

Work in small English-language commits on main. Run the complete unit suite
after each coherent change. Do not operate physical robots during development.
Completed code is not evidence of hardware safety or validated tracking accuracy.

## 1. Control lifecycle and stale data (audit items 1–8)

- [x] Fail closed on homing timeout when RTDE is configured; stop both arms.
- [x] Separate connection, homing, arming, pause and explicit resume in interactive UR sessions.
- [ ] Expire per-arm/per-joint targets and clear them on tracking loss.
- [x] Reject stale feedback and react to controller fault states before streaming.
- [ ] Define dual-arm workspace constraints with the hardware owner.

## 2. Tracking and calibration (items 9–22)

- [ ] Reconcile image wrist refinement with world-space elbow measurements.
- [ ] Validate depth motion, torso reference transitions and hand assignment.
- [x] Normalize hand smoothing by elapsed time.
- [ ] Lock operator identity; expose independent hand confidence thresholds.
- [ ] Distinguish measured, held and missing joint values.
- [x] Define calibration semantics, stable capture, persistence and reset.
- [ ] Document supported axes and informational lift mode.

## 3. Reliability and recording (items 31–37)

- [x] Prevent recording filename collisions and isolate recording failures.
- [ ] Bound recording I/O and profile the synchronous pipeline.
- [ ] Correct video pacing and define camera recovery behavior.
- [x] Repair debug backend helper methods.

## 4. Operational UI (items 23–30)

- [ ] Per-arm quality and lifecycle status, source-of-position labels.
- [ ] Visible actionable errors and recording status.
- [ ] Settings and latency diagnostics.

## 5. Verification and delivery (items 38–44)

- [ ] Recorded-video regression fixtures and measurable quality thresholds.
- [ ] End-to-end and fault-injection tests.
- [ ] CI and clean-install verification.
- [ ] Session metadata and reproducibility instructions.

## External validation / decisions required

- Physical mounting transforms, tool geometry, workspace obstacles and separation
  are required before collision checking can be implemented meaningfully.
- Additional controlled robot axes need an agreed mapping and hardware validation.
- Camera recordings and measured motion-to-command latency are required to claim
  tracking accuracy. Synthetic tests alone are insufficient.
- Hardware acceptance must be performed by the operators under the laboratory's
  procedures; an application pause is not a hardware emergency stop.

## Handoff — paused at the user's request, 2026-09-17

The full 44-item audit is NOT complete. Stop here and discuss the user's next
priority before continuing. All implementation commits use English messages.

### Implemented and checked

- Homing timeout/no confirmation blocks streaming and stops both arms.
- Position feedback expires after 500 ms, including a silently connected socket.
  Observed controller faults block streaming and latch a session fault.
- Interactive UR startup is motion-free: connect, home, enable, pause and resume
  are separate operator actions. Missing any mapped joint pauses control and
  recovery does not auto-resume. RTDE is required in this interactive path.
- Pausing clears UR accumulated targets and resets setpoints from available
  feedback. This is not a general per-joint expiry implementation for all backends.
- Debug backend state helpers now delegate to their tracker.
- CSV names are collision-resistant and exclusively created. Start/write failures
  are isolated; shutdown closes the file even if flushing fails. Flushes are
  periodic rather than per-frame. Recording remains synchronous.
- Hand/body landmark smoothing uses elapsed frame time. Independent hand
  detection/presence confidence options no longer have a hidden 0.4 cap.
- Video waits subtract processing time and reject non-finite reported FPS.
- Calibration uses a stable sample window, maps offsets around configured robot
  home, supports save/load/reset, and pauses hardware when changed. Keyboard:
  C capture, K save, L load, X reset; profiles live in the recording directory.
- UI has a hardware lifecycle action and a visible error/message line.
- Added Linux/Windows Python 3.12/3.13 CI configuration. Remote CI has not yet
  been observed. A clean Linux Python 3.12 environment installed the lock file
  and editable package successfully; pip check passed (277 tests at that point).
- Final local suite: 281 tests pass. No physical robot or live-camera acceptance
  testing was performed. Existing project .venv was not replaced.

### Still outstanding / partial

- Audit 3–4: per-joint ages and expiration across all backends, held-vs-fresh
  measurement provenance, and independent watchdog behavior if the main loop
  blocks. Current interactive UR behavior is a conservative whole-session pause.
- Audit 8: dual-arm collision/workspace constraints need mounting/tool geometry.
- Audit 9–11, 13–14, 16–17: reconcile 2D wrist refinement with 3D elbows; depth
  motion and torso-reference transitions; temporal hand/operator identity;
  evaluate hand world coordinates; label held wrist values.
- Audit 21–22: decide additional controlled axes and lift-mode behavior. Existing
  implementation remains three mapped axes per arm plus gripper.
- Audit 23–30: per-joint UI status, arm-specific quality, measured/commanded pose
  provenance, full settings, latency diagnostics, detailed recording status.
  Current UI additions cover lifecycle actions and error messages only.
- Audit 33–36: bounded/asynchronous recording, pipeline profiling/threading,
  robust source-clock playback pacing and camera reconnection. Current pacing
  subtracts processing duration but is not a complete media scheduler.
- Audit 38–41: real-video fixtures, quantitative tracking thresholds, broader
  end-to-end/fault tests and laboratory acceptance. Added unit/fault tests do
  not establish safety certification or tracking accuracy.
- Audit 42–44: observe remote CI results, verify Windows/Python 3.13/Jetson
  installations, record session/model/configuration metadata and replay it.
- Calibration profiles are versioned but do not yet bind to camera, model or
  mapping configuration. Recapture after setup changes. UI settings and dedicated
  profile controls beyond keyboard shortcuts are still outstanding.
- Legacy direct URBackend callers retain auto-home by default; the interactive
  application explicitly disables it. Revisit that API compatibility choice.
- Preflight availability, continuous feedback monitoring outside send(), and
  pause/setpoint timing during physical deceleration still need hardware review.

### Validation environment retained

A separate `/tmp/engineering-thesis-verify.*` virtual environment was created
for clean-install checks. It can be removed after verification is no longer
needed; no user environment was deleted or overwritten.
