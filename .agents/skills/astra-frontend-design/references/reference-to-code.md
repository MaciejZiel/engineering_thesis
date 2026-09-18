# Reference-led implementation

Use this reference when the user supplies screenshots, Figma frames, images, a design file, or named sites as visual direction.

## Determine the fidelity target

Distinguish between:

- exact implementation of a user-owned design;
- close reproduction of an existing project state;
- inspiration from a third-party reference;
- extraction of a visual grammar for a new product.

Follow exact dimensions and assets for user-owned designs when practical. For third-party inspiration, reproduce the underlying principles and interaction qualities without copying protected brand assets, logos, copy, or a signature composition verbatim.

## Extract the visual grammar

Inspect the source rather than relying on a style label. Derive:

- canvas and container geometry;
- grid, alignment, whitespace, and responsive collapse;
- type families, scale, weight, line height, and measure;
- semantic colors, contrast, borders, radii, shadows, and material treatment;
- asset cropping, focal points, and aspect ratios;
- control states, transition timing, scroll behavior, and layering;
- recurring components and the exceptions that create emphasis.

Separate observed facts from inference. If a font or asset is unavailable, choose a substitute with similar measurable characteristics and note the substitution.

## Implement structure before effects

Match content order, proportions, line breaks, and responsive behavior before tuning shadows or animation. Use the project's components and tokens where they can express the reference faithfully. Do not add a new UI library merely because the reference resembles it.

Preserve real content and behavior from the target project. A reference can guide presentation; it does not authorize removing functionality or replacing product facts.

## Compare rendered output

Capture the implementation at the reference viewport and at a narrow viewport. Compare silhouette, major alignment lines, whitespace distribution, typography, asset framing, and visual emphasis. Fix the largest perceptual mismatch first. Repeat until remaining differences are intentional or caused by documented constraints.
