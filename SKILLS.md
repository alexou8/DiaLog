# SKILLS.md

Agent skills available in this repository, when to use each, and which take
precedence when several apply.

Skills are vendored under `.claude/skills/` and committed, so every contributor
and every agent session gets the same behaviour with no install step. The
per-skill provenance table and the rules for adding or updating one live in
[`.claude/skills/README.md`](.claude/skills/README.md) — this file covers
_when to reach for which_, and does not duplicate the skills' own
`SKILL.md` definitions.

## Available skills

| Skill                         | Use it when                                                                                                    | Status for that task         |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------- | ---------------------------- |
| `accessibility`               | Auditing or changing anything a user sees. WCAG 2.2 conformance, screen-reader behaviour, keyboard navigation. | **Mandatory**                |
| `web-design-guidelines`       | Reviewing UI code before calling it done.                                                                      | **Mandatory** for UI changes |
| `frontend-design`             | Building new UI or reshaping existing UI — aesthetic direction and typography.                                 | Recommended                  |
| `ui-ux-pro-max`               | You need concrete reference data: palettes, font pairings, chart types, layout and motion patterns.            | Optional                     |
| `vercel-react-best-practices` | React 19 / Next.js 15 performance work — Server vs Client Components, data fetching, bundle size.              | Recommended                  |
| `prisma-cli`                  | Migrations, `generate`, seeding, studio.                                                                       | Recommended                  |
| `find-skills`                 | Before hand-rolling a capability that probably already exists as a skill.                                      | Optional                     |

The `codex@openai-codex` plugin (declared in `.claude/settings.json`) ships
three more — `codex-cli-runtime`, `codex-result-handling`, and
`gpt-5-4-prompting`. They are not vendored under `.claude/skills/`: they arrive
with the plugin and are updated by updating it. They govern how Claude talks to
Codex, not how DiaLog is built, so they sit outside the table above and below
everything in the precedence order.

## Precedence

When several skills apply to one task, resolve in this order:

1. **`CLAUDE.md` and `AGENTS.md` win over any skill.** A skill is general
   advice; the invariants in this repository encode specific past bugs and
   safety decisions. Where they conflict, the repository is right.
2. **`accessibility` outranks the other UI skills.** This is an
   accessibility-first product with an axe-core e2e suite. If
   `frontend-design` or `ui-ux-pro-max` suggests something that costs contrast,
   focus visibility, or a non-colour signal, the accessibility rule wins.
3. **`web-design-guidelines` is a review gate, not a design source.** Run it
   after the change, before declaring done.
4. **`vercel-react-best-practices` yields to correctness and security.** Do not
   take a performance suggestion that moves an authorization check to the
   client or widens what reaches the AI layer.

## How they combine on a typical UI change

`frontend-design` (or `ui-ux-pro-max` for reference data) to make the change →
`accessibility` to audit it → `web-design-guidelines` to review the code →
then the verification commands in [AGENTS.md](AGENTS.md).

## Notes

- The skills are third-party content, excluded from ESLint and Prettier (see
  `eslint.config.mjs` and `.prettierignore`) because they are not project
  source. Do not hand-edit them; update them through the `skills` CLI as
  described in `.claude/skills/README.md`.
- Skills run with full agent permissions. Read a skill's `SKILL.md` and any
  `scripts/` before committing it.
- `skills-lock.json` at the repository root pins what is installed.
