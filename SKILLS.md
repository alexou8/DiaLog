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

| Skill                              | Use it when                                                                                                    | Status for that task                      |
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| `accessibility`                    | Auditing or changing anything a user sees. WCAG 2.2 conformance, screen-reader behaviour, keyboard navigation. | **Mandatory**                             |
| `web-design-guidelines`            | Reviewing UI code before calling it done.                                                                      | **Mandatory** for UI changes              |
| `frontend-design`                  | Building new UI or reshaping existing UI — aesthetic direction and typography.                                 | Recommended                               |
| `ui-ux-pro-max`                    | You need concrete reference data: palettes, font pairings, chart types, layout and motion patterns.            | Optional                                  |
| `vercel-react-best-practices`      | React 19 / Next.js 15 performance work — Server vs Client Components, data fetching, bundle size.              | Recommended                               |
| `prisma-cli`                       | Migrations, `generate`, seeding, studio.                                                                       | Recommended                               |
| `security-and-hardening`           | Any change to auth, sessions, API/route handlers, input validation, or import/parse code; dependency changes.  | **Mandatory** for those                   |
| `playwright-best-practices`        | Writing or changing anything under `tests/e2e/`, or diagnosing an e2e failure.                                 | **Mandatory** for e2e changes             |
| `supabase-postgres-best-practices` | Before changing `prisma/schema.prisma`, adding an index or migration, or writing raw SQL.                      | **Mandatory** before schema/index changes |
| `shadcn`                           | Composing or adding components under `components/ui/` (forms, dialogs, tables); `components.json` is present.  | Recommended                               |
| `find-skills`                      | Before hand-rolling a capability that probably already exists as a skill.                                      | Optional                                  |

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
5. **`security-and-hardening` is a floor, and `AGENTS.md` still wins.** It is
   generic OWASP advice; where it is silent or looser than an invariant
   (ownership scoping, the AI privacy boundary, logging no health values), the
   invariant applies. It never licenses weakening a boundary to simplify code.
6. **`playwright-best-practices` yields to the repository's own e2e rules.**
   `workers: 1`, no retries, the shared-account pool, `countRecordRows()` and
   the `locator.count()` ban are deliberate (see `AGENTS.md`); do not take a
   generic suggestion to parallelise, retry or `waitForTimeout`.
7. **`supabase-postgres-best-practices` is advice about Postgres, not about
   this schema.** Several rules are Supabase-specific (RLS, `auth.uid()`, Edge
   Functions) and do not apply: authorization here is `requireUser()` plus
   owner-scoped queries, not row-level security. Prisma migrations remain the
   only schema-change mechanism, and an applied migration is never edited.
8. **`shadcn` yields to `accessibility`, and its CLI is never run implicitly.**
   Components are already vendored in `components/ui/`; compose them rather than
   re-adding. `npx shadcn@latest` fetches unpinned code — run it by hand, and
   read the diff.

## How they combine on other kinds of change

- **Auth, API route, or import/parse change:** `security-and-hardening` before
  writing code and again when reviewing it, then the invariants in
  [AGENTS.md](AGENTS.md).
- **Schema or index change:** `supabase-postgres-best-practices` and
  `prisma-cli`, plus the destructive-change rules in `AGENTS.md`.
- **e2e change:** `playwright-best-practices`, bounded by the e2e rules in
  `AGENTS.md`.

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
- **One vendored skill is patched.** `shadcn`'s `SKILL.md` had its load-time
  `!npx shadcn@latest info --json` execution and its `allowed-tools`
  pre-approval removed when it was vendored. That is a deliberate exception to
  "do not hand-edit", and must be re-applied if the skill is updated. Details in
  [`.claude/skills/README.md`](.claude/skills/README.md).
- Do not run `npx skills update` / `npx skills check` here: they replace the
  committed copies with symlinks. Re-add one skill with `--copy` instead.
