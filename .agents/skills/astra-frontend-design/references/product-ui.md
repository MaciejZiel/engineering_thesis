# Product and operational UI

Use this reference for SaaS applications, dashboards, admin tools, editors, settings, data-heavy screens, and repeated workflows.

## Optimize for the work

Identify the user's primary task, the decisions they make, the information they compare, and the actions they repeat. Let those facts determine hierarchy and density. Operational tools should usually feel calm, direct, and predictable; do not give them landing-page composition, oversized promotional type, or decorative card grids.

Make the useful product state the first screen. Navigation, filters, search, selection, bulk actions, status, and detail views should work together as a coherent workflow rather than as isolated showcase components.

## Choose controls by behavior

- Use buttons for commands, links for navigation, tabs for peer views, and menus for secondary option sets.
- Use checkboxes or switches for independent binary state, radio groups or segmented controls for exclusive choices, and inputs or steppers for precise numeric values.
- Use familiar icons for common actions and pair unfamiliar icons with accessible names and tooltips. Stay within the project's icon family.
- Keep labels persistent. Do not use placeholder text as the only form label.

## Design the information hierarchy

Place the primary object and its current state first. Keep frequent actions near their target. Use alignment, spacing, typography, and separators before adding containers. Cards are appropriate for repeated independent objects or genuinely framed tools; avoid nesting cards or floating every section.

For data-heavy views:

- align comparable values and use tabular numerals where comparison benefits;
- preserve units, signs, precision, timestamps, and time zones;
- make sorting, filtering, pagination or virtualization, selection, and row actions explicit;
- avoid encoding status by color alone;
- keep dense desktop workflows usable without pretending the same table can simply shrink onto mobile.

On narrow screens, preserve the task rather than the desktop geometry. Prioritize columns, provide a detail drill-in, collapse secondary controls, or switch to a list when needed.

## Complete the state model

Implement the states a user can actually encounter:

- initial loading and incremental loading;
- no data, no search results, and no permission;
- validation, recoverable failure, and service failure;
- optimistic or pending actions, success confirmation, undo where appropriate;
- disabled controls with a discoverable reason;
- long content, localization growth, zoom, and interrupted interactions.

Loading placeholders should resemble the final structure. Empty states should explain how to proceed. Errors should say what happened and what action is available.

## Protect accessibility and speed

Use native semantics first. Preserve logical focus order, visible focus, full keyboard operation, accessible names, and announcements for meaningful asynchronous changes. Respect reduced motion and adequate target sizes. Avoid state or animation patterns that rerender large trees on every pointer or scroll update.

Verify the common workflow end to end, not only the static successful state.
