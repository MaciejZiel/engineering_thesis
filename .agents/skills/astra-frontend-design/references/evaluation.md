# Evaluate frontend skill quality

Use this reference when comparing this skill with another model, skill, or revision. The goal is to measure rendered outcomes, not prompt compliance or the elegance of the generated source alone.

## Comparison setup

Use the same clean starter, task brief, viewport sizes, time limit, tool access, and dependency policy for each condition. Recommended conditions:

1. GPT-6 Astra without a frontend skill.
2. GPT-6 Astra with `$astra-frontend-design`.
3. Claude Fable without an added frontend skill.
4. Claude Fable with its normal frontend-design guidance, if that is the user's real alternative.

Keep model and skill names hidden from reviewers. Randomize screenshot order. Run more than one task because a single aesthetic can flatter one model.

The reusable cases are in [evals.json](../assets/evals.json). Replace fictional names when desired, but keep the workload mix and acceptance criteria comparable.

## Required evidence

For each run, retain:

- final desktop and mobile screenshots at identical viewport sizes;
- a short screen recording for interaction-heavy tasks;
- build, lint, console, and primary-flow results;
- wall time, model effort, tool calls, and any human intervention;
- the final code diff and dependency changes.

Do not grade a screenshot if the implementation failed to run. Record functional failure separately instead of awarding aesthetic points to a static mock.

## Blind scoring rubric

Score each category from 1 to 5, then apply the weight.

| Category | Weight | What to judge |
| --- | ---: | --- |
| Product and audience fit | 20 | The surface serves the stated job and feels native to its subject. |
| Distinctiveness and coherence | 20 | A specific visual thesis is carried consistently without template drift. |
| Hierarchy and composition | 15 | Attention, grouping, rhythm, and navigation are immediately clear. |
| Responsive quality | 15 | Mobile is intentionally recomposed; text, controls, and assets fit. |
| Functional completeness | 10 | Expected controls, flows, and relevant states work. |
| Typography, color, and assets | 10 | Choices are legible, intentional, harmonious, and well framed. |
| Accessibility and interaction quality | 10 | Keyboard, focus, contrast, motion preferences, feedback, and stability are sound. |

Require reviewers to cite a visible element or interaction for each score below 3 or above 4. Track factual defects separately: overlap, clipping, broken control, missing asset, console error, or failed route.

## Decision rule

Treat the skill as an improvement only when Astra with the skill beats Astra without it across the median task and does not increase functional failures. Treat it as competitive with Fable only when the weighted score is within the reviewer's predeclared equivalence margin and the completion rate is comparable.

A practical initial margin is five points on a 100-point scale, but choose it before seeing results. Inspect category-level gaps: a tied total can hide strong aesthetics and weak product UX, or the reverse.

Use observed failures to revise the narrowest relevant instruction or reference. Do not add a universal rule to fix one unusual brief.
