# Production-readiness work plan

Work in small English-language commits on main. Run the complete unit suite
after each coherent change. Do not operate physical robots during development.
Completed code is not evidence of hardware safety or validated tracking accuracy.

## 1. Control lifecycle and stale data (audit items 1–8)

- [x] Fail closed on homing timeout when RTDE is configured; stop both arms.
- [ ] Separate connection, homing, arming, pause and explicit resume.
- [ ] Expire per-arm/per-joint targets and clear them on tracking loss.
- [ ] Reject stale feedback and react to controller fault states.
- [ ] Define dual-arm workspace constraints with the hardware owner.

## 2. Tracking and calibration (items 9–22)

- [ ] Reconcile image wrist refinement with world-space elbow measurements.
- [ ] Validate depth motion, torso reference transitions and hand assignment.
- [ ] Normalize hand smoothing by elapsed time.
- [ ] Lock operator identity; expose independent hand confidence thresholds.
- [ ] Distinguish measured, held and missing joint values.
- [ ] Define calibration semantics, stable capture, persistence and reset.
- [ ] Document supported axes and informational lift mode.

## 3. Reliability and recording (items 31–37)

- [ ] Prevent recording filename collisions and isolate recording failures.
- [ ] Bound recording I/O and profile the synchronous pipeline.
- [ ] Correct video pacing and define camera recovery behavior.
- [ ] Repair debug backend helper methods.

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
