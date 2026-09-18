# Motion and interaction

Use this reference when motion, scrolling, gestures, or interaction choreography is a meaningful part of the experience.

## Give motion a job

Each animation should communicate continuity, reveal hierarchy, acknowledge input, explain a state change, or create one deliberate narrative moment. Prefer a coherent sequence over many unrelated fade-and-rise effects. Interaction feedback may be subtle and frequent; autonomous motion should be rarer.

Match intensity to the context. A trading or admin tool benefits from fast state feedback and stable geometry. A portfolio or campaign page can support richer choreography when it remains legible and controllable.

## Choose the lightest capable mechanism

- CSS transitions and keyframes for simple state changes.
- The project's existing motion library for component choreography, layout transitions, and gestures.
- A timeline library for complex scroll narratives only when its control model is actually needed.
- Canvas or WebGL for a true scene or simulation, not for decorative complexity that CSS can express.

Check existing dependencies before adding a library. Keep animation logic in focused client-side components in server-rendered applications. Clean up observers, timelines, animation frames, and event handlers.

## Preserve stability and performance

Prefer transform and opacity for frequent animation. Avoid layout shift, accidental scroll traps, pointer-following state that rerenders a large component tree, and competing libraries controlling the same property. Define stable dimensions for elements whose content or state changes.

Make hover enhancements supplementary because touch and keyboard users may never see them. Provide visible focus and active feedback. Keep important controls available during animation.

## Respect user control

Honor `prefers-reduced-motion` with a meaningful low-motion alternative, not merely a slower version. Avoid essential information that appears only through motion. Pause or expose controls for long-running ambient media where appropriate.

## Verify the moving result

Test the initial frame, transition, final state, interruption, repeated activation, narrow viewport, and reduced-motion mode. Check that text remains readable, focus lands correctly, hit targets do not move away, and the page stays responsive during the effect.
