# Research basis and adaptation decisions

## Agent-routing note (2026-09-13)

The user's attached [“Idiot Boss” article](https://x.com/anshuc/article/2098811738674147520) compares two visual builds and reports that delegating all implementation to Luna reduced quality, while one continuing Astra implementation worker under a smaller browser-testing coordinator came closer to direct Astra at lower API-equivalent cost. This is limited visual-task evidence, not a verified ChatGPT quota conversion or universal frontend quality claim. Accordingly, this skill keeps product and visual judgment with the capable implementer; a smaller coordinator supplies the brief and observational QA, when delegation is allowed.

This skill is an original synthesis for GPT-6 Astra. It paraphrases general methods and links to source material; it does not reproduce third-party skill text wholesale.

## Primary model guidance

- [OpenAI: Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra) - short discriminating descriptions, progressive disclosure, contextual references, calibrated testing, and explicit completion criteria.
- [OpenAI: GPT-6 Astra model guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra) - Astra's sensitivity to instructions, tendency to ask material questions, need for clear follow-through, and verification calibration.
- [OpenAI: Frontend prompt instructions](https://developers.openai.com/api/docs/guides/frontend-prompt) - domain-aware UI, functional states, appropriate controls, assets, responsive stability, and screenshot verification. The published prompt targets GPT-5.5, so this skill keeps the durable criteria while removing brittle universal styling rules.

## Public frontend skill patterns reviewed

- [Anthropic frontend-design](https://github.com/anthropics/skills/tree/main/skills/frontend-design) - subject-matter grounding, one deliberate visual point of view, compact tokens, restraint, and screenshot-based self-critique.
- [Taste Skill](https://github.com/Leonxlnx/taste-skill) - explicit brief inference, anti-default checks, redesign awareness, motion and density calibration, and preflight review. This Astra version replaces its many fixed bans and numeric presets with contextual judgment.
- [UI UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill) - domain and platform matching, palette and type selection, state coverage, responsive checks, and design-system generation. This skill avoids loading a large style catalog when the brief already supplies direction.
- [Impeccable](https://github.com/pbakaus/impeccable) - separation of durable product truth from visual direction, focused modes for shaping, critique, audit, hardening, and polish, plus live browser iteration. This skill expresses those modes through progressive references rather than a command framework or runtime dependency.
- [Vercel agent skills](https://github.com/vercel-labs/agent-skills) - focused React performance and web-interface review guidance. This skill defers to a project's existing engineering standards and keeps its own performance rules at the visible-experience level.

## Standards and evidence

- [WCAG 2.2](https://www.w3.org/TR/WCAG22/) and [WAI-ARIA Authoring Practices](https://www.w3.org/WAI/ARIA/apg/) inform the accessibility floor.
- [web.dev Core Web Vitals](https://web.dev/articles/vitals) informs user-visible performance checks.
- Duan et al., [Visual Prompting with Iterative Refinement for Design Critique Generation](https://arxiv.org/abs/2412.16829), supports visually grounded, iterative critique instead of text-only review.

## Why one skill

Several popular frontend skills overlap in their activation descriptions and repeat similar anti-pattern lists. Loading them together can produce conflicting constraints and spend context on guidance unrelated to the current surface. Astra is capable enough to apply nuanced criteria without a long universal recipe, but benefits from a clear definition of completion. The resulting architecture is one concise entrypoint with references selected by task mode and a mandatory render-inspect-revise loop for implementation work.
