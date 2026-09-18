# Astra frontend prompt recipes

These prompts are deliberately compact. Replace bracketed fields and attach screenshots or references when available. The skill provides the reusable judgment; the task prompt provides product-specific truth.

## Optional cost-aware delegation

> Use `$astra-frontend-design` with `$astra-efficient-orchestration` if delegation is permitted. For this bounded, browser-testable build, have one Astra worker own the entire visual implementation in a continuing thread from my original brief and references. Let a small coordinator run the app, inspect desktop/mobile screenshots and relevant controls, and return only concrete issues and evidence for focused fixes. Do not have the coordinator prescribe layout or code architecture. Keep required checks assigned; use direct Astra instead if the result cannot be evaluated reliably.

## Build a page or app surface

> Use `$astra-frontend-design`. Build [surface] for [audience] whose primary task is [job]. The visual character should feel [three concrete adjectives], grounded in [subject-specific material or reference]. Preserve [constraints]. Treat this as an implementation task: inspect the repository, implement the real workflow, run it, review desktop and mobile screenshots, and fix material visual or interaction issues before handing it back. Make reasonable assumptions and ask only if a missing answer would change the product or brand direction.

## High-concept landing page or portfolio

> Use `$astra-frontend-design` in expressive-site mode. Create a distinct visual thesis for [product/person/place] based on [subject matter], with one memorable first-viewport idea and disciplined supporting sections. Avoid interchangeable template composition and decorative effects without a purpose. Use real or clearly temporary assets, implement responsive behavior and purposeful motion, then inspect the rendered page at desktop and mobile sizes and refine it to completion.

## Dense product UI

> Use `$astra-frontend-design` in product-UI mode. Build [dashboard/tool/workflow] for [role]. Optimize for [decisions/actions], [density], and repeated daily use. Make navigation, filters, states, and keyboard behavior complete. Use the existing design system and data conventions. Run the primary workflow, inspect wide and narrow layouts, and fix hierarchy, overflow, accessibility, and interaction defects.

## Redesign an existing interface

> Use `$astra-frontend-design` in [preserve/evolve/overhaul] redesign mode. Inspect the running interface and code before editing. Preserve [routes, behavior, content, brand, analytics, accessibility, or SEO constraints]. Improve [specific goals]. Capture comparable before-and-after views, implement the redesign consistently across the requested surfaces, and correct visual or functional regressions before finishing.

## Implement a screenshot or Figma design

> Use `$astra-frontend-design` in reference-led mode. Implement the attached [screenshot/frame] with [exact/close/inspired] fidelity inside the existing stack. Extract its layout, type, color, asset framing, states, and responsive logic rather than relying on a style label. Preserve the product's real content and behavior. Compare the rendered output at the reference viewport and on mobile, then fix the largest perceptual mismatches.

## Polish a working frontend

> Use `$astra-frontend-design` to polish [route/component]. Keep behavior and product identity intact. Inspect it in the browser, identify the few highest-impact problems in hierarchy, typography, spacing, responsiveness, content fit, states, and motion, implement the fixes, and repeat the relevant screenshots and checks until another pass would not materially improve the result.

## Audit without editing

> Use `$astra-frontend-design` for a review only. Inspect [surface] at representative desktop and mobile widths and exercise its primary workflow. Return prioritized findings tied to concrete elements and user impact across visual hierarchy, responsive behavior, accessibility, states, content, and performance. Do not change files.

## Establish durable project design context

> Use `$astra-frontend-design` to inspect this product and create or update `PRODUCT.md` and `DESIGN.md`. Keep durable product truth separate from visual direction. Record only facts and decisions that future frontend work needs: audience, jobs, constraints, content voice, tokens, typography, layout, components, motion, responsive rules, references, and explicit exceptions. Preserve existing documents and note unresolved choices rather than inventing brand facts.
