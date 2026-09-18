# Visual and interaction quality gates

Use this reference for final verification, polish, or a review-only task.

## Calibrate the check

Review the surfaces and states changed by the task. Use the repository's existing build, lint, unit, accessibility, and browser tooling. Broaden coverage only when failures, shared components, or cross-route effects justify it.

## Inspect rendered pixels

Capture the actual interface at representative widths. A useful default is one wide desktop and one narrow mobile viewport; add tablet, short-height, or ultrawide coverage when the composition depends on it. Exercise the primary interaction before capturing stateful screens.

Check:

- clear first-glance hierarchy and a visible primary task;
- consistent alignment, spacing rhythm, type scale, color roles, geometry, and icon treatment;
- intended asset loading, crop, focal point, and resolution;
- text wrapping, long words, localization growth, zoom, and no incoherent overlap;
- stable controls with hover, focus, active, disabled, selected, loading, error, empty, and success states as relevant;
- navigation, dialogs, menus, forms, scrolling, and keyboard use;
- reduced motion, contrast, accessible names, semantic structure, and focus order;
- no unintended horizontal scroll, layout shift, blank canvas, console error, or failed request caused by the change.

Do not mistake a passing build for a passing interface.

## Critique by impact

Rank issues by user impact:

1. Broken flow, missing content, inaccessible control, unreadable contrast, overlap, or viewport failure.
2. Misleading hierarchy, inconsistent state, poor responsive adaptation, or visual direction that conflicts with the brief.
3. Typography, spacing, asset, motion, or copy polish.

Fix high-impact issues before decorative refinements. Ground critique in a visible element, user task, or measurable behavior rather than vague taste language.

## Iterate to completion

After each material fix, rerun only the affected checks and recapture the relevant viewport. Continue while another pass has a clear, observable improvement. Stop when the requested surfaces work, the chosen direction is coherent, and no material visual or interaction defect remains.

For review-only requests, do not edit. Return a concise, prioritized set of findings with the affected surface and a concrete recommendation.
