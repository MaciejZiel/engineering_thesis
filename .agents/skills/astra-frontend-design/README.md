# Astra Frontend Design

A production-focused **frontend design Agent Skill** for **GPT-6 Astra** and **OpenAI Codex**.

It helps Astra turn product briefs, screenshots, existing interfaces, and rough ideas into distinctive responsive frontends with stronger art direction, interaction design, implementation discipline, and visual QA.

## Why this skill exists

Frontend generation often fails in predictable ways: generic landing-page composition, weak hierarchy, arbitrary gradients, repetitive cards, desktop-only polish, and code that looks plausible without being visually verified.

Astra Frontend Design adds a compact decision system for choosing the right workflow, establishing a coherent visual direction, translating references into reusable design rules, and checking the implemented result at realistic viewport sizes.

## What it provides

- Greenfield product and landing-page direction
- Product UI and application-shell guidance
- Existing-interface redesign without accidental product drift
- Screenshot and visual-reference reconstruction
- Responsive layout, typography, color, and spatial-system rules
- Motion and interaction guidance with reduced-motion support
- Accessibility, code-quality, and visual-QA gates
- Product and design brief templates
- Prompt recipes and a blinded evaluation suite

## Install

Using the Agent Skills installer:

```bash
npx skills@latest add Enixes/astra-frontend-design
```

Or clone it directly into the Codex skills directory:

```bash
git clone https://github.com/Enixes/astra-frontend-design.git \
  ~/.codex/skills/astra-frontend-design
```

Start a fresh Codex session after installation so the skill catalog refreshes.

## Quick start

### Build a frontend

```text
Use $astra-frontend-design to build a responsive product landing page. Establish a specific art direction, implement the complete experience, and visually verify desktop and mobile before finishing.
```

### Redesign an existing interface

```text
Use $astra-frontend-design to redesign this interface without changing its product behavior. Preserve working flows, improve hierarchy and visual character, and compare the result against the current UI.
```

### Rebuild from a reference

```text
Use $astra-frontend-design to translate these screenshots into a responsive implementation. Infer the underlying layout and design system instead of copying isolated pixels.
```

## Workflow routing

| Request | Guidance loaded |
|---|---|
| New marketing site or greenfield experience | Art direction and frontend quality gates |
| Dashboard, editor, settings, or internal tool | Product UI guidance |
| Existing product redesign | Redesign constraints and behavior preservation |
| Screenshot or visual reference | Reference-to-code workflow |
| Animated or scroll-responsive experience | Motion and interaction guidance |

Detailed references load only when the request needs them, keeping the skill's default context footprint small.

## Quality standard

The skill expects a finished frontend to have:

- A deliberate visual premise rather than a generic template
- Clear hierarchy and consistent type, spacing, color, and component rules
- Real responsive behavior instead of a compressed desktop composition
- Appropriate semantic HTML, keyboard support, contrast, and reduced-motion handling
- Meaningful interaction states and motion that explain change
- Visual inspection of the rendered result, followed by targeted correction

It does not claim automatic parity with another model or design tool. The included evaluation suite is intended for blinded comparisons using identical briefs, viewports, and scoring criteria.

## Repository structure

```text
.
├── SKILL.md
├── agents/openai.yaml
├── assets/
│   ├── DESIGN.template.md
│   ├── PRODUCT.template.md
│   └── evals.json
└── references/
    ├── art-direction.md
    ├── evaluation.md
    ├── motion-and-interaction.md
    ├── product-ui.md
    ├── prompt-recipes.md
    ├── quality-gates.md
    ├── redesign.md
    ├── reference-to-code.md
    └── research-basis.md
```

## Efficient multi-agent use

Pair it with [Astra Efficient Orchestration](https://github.com/Enixes/astra-efficient-orchestration) when the user wants a cost-aware route and delegation is permitted. For a bounded, browser-testable frontend, a small coordinator can pass the original brief to one Astra implementation worker, inspect the running result, and send screenshots with observed defects back to the same worker. Do not ask a weaker worker to implement the entire quality-critical UI or let a coordinator prescribe aesthetic/technical choices. Prefer direct Astra for quality-first or hard-to-evaluate work, and Astra-led planning for ambitious projects.

The [user-provided experiment](https://x.com/anshuc/article/2098811738674147520) supports trying that route on visual builds; its API-equivalent cost comparisons are not measured Plus/Pro quota savings or proof of general parity.

## References

- [GPT-6 Astra model guidance](https://developers.openai.com/api/docs/guides/latest-model?model=gpt-6-astra)
- [Rethinking skills and prompts for GPT-6 Astra](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra)
- [Agent Skills overview](https://agentskills.io/)
