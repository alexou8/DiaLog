# Vendored agent skills

These skills are checked in so every contributor and every agent session gets the
same behaviour without a separate install step. They are third-party content
managed by the [`skills`](https://skills.sh) CLI — they are excluded from ESLint
and Prettier (see `eslint.config.mjs` and `.prettierignore`) because they are not
project source.

| Skill | Source | Why it is here |
| --- | --- | --- |
| `find-skills` | `vercel-labs/skills` | Discover and install new skills. Run it before hand-rolling a capability that probably already exists. |
| `web-design-guidelines` | `vercel-labs/agent-skills` | Reviews UI code against the Web Interface Guidelines. |
| `vercel-react-best-practices` | `vercel-labs/agent-skills` | React 19 / Next.js 15 performance patterns — matches this app's stack exactly. |
| `frontend-design` | `anthropics/skills` | Aesthetic direction for new UI, so screens don't land on templated defaults. |
| `ui-ux-pro-max` | `nextlevelbuilder/ui-ux-pro-max-skill` | Searchable local data for palettes, typography, motion, charts and layout. |
| `accessibility` | `addyosmani/web-quality-skills` | WCAG 2.2 auditing. DiaLog is an accessibility-first product with an axe-core e2e suite, so this is load-bearing, not optional. |
| `prisma-cli` | `prisma/skills` | Prisma CLI reference — this repo drives migrations, generate and seeding through it constantly. |
| `shadcn` | `shadcn-ui/ui` | The official shadcn/ui skill (369K installs, 125K stars). `components.json` is committed and `components/ui/` is built from shadcn primitives over Radix, so composition, forms, styling and icon rules apply directly. **Patched when vendored — see below.** |
| `playwright-best-practices` | `currents-dev/playwright-best-practices-skill` | Playwright e2e practice: flaky tests, auth, page objects, axe-core, offline/service workers, CI. Matches the Playwright + axe suite. Below the 1K-star bar (92K installs, ~395 stars, vendor Currents.dev) but accepted: 63 Markdown files and no executables. |
| `supabase-postgres-best-practices` | `supabase/agent-skills` | Generic Postgres schema, index, query, pooling and migration rules ("Postgres anywhere" — most of it is not Supabase-specific). Complements `prisma-cli`, which covers the CLI only. |
| `security-and-hardening` | `addyosmani/agent-skills` | OWASP-oriented review: trust boundaries, input validation, authN/Z, headers, secrets, dependency audit. Nothing else vendored covers security, and this app holds personal health data. |

## Patched when vendored

Vendored skills are third-party content and are not hand-edited. There is one
deliberate, recorded exception:

- **`shadcn`** — `SKILL.md` was patched. Upstream ran `npx shadcn@latest info
  --json` at skill-load time through a `!` shell directive, and pre-approved
  every `shadcn@latest` subcommand through `allowed-tools`. Both were removed:
  this repository handles health data and must not execute an unpinned npm
  package implicitly, or pre-approve one, on behalf of a third party. The patch
  leaves an HTML comment in `SKILL.md` saying so, and the instruction to run the
  command by hand. **Re-apply it after every update of this skill**, and expect
  `skills-lock.json`'s hash for it to differ from a fresh upstream copy.

Every other skill is what the CLI installed (line endings aside).

## Updating

Update one skill at a time by re-adding it with `--copy`:

```bash
npx skills add <owner/repo@skill> --agent claude-code --copy -y
```

Do **not** run `npx skills update` or `npx skills check` in this repository.
Both replace the copied skill directories with symlinks into `.agents/skills`
(absolute `C:\` paths on Windows) and rewrite `skills-lock.json`, which breaks
the `--copy` convention below; the symlinks do not work in a fresh clone or in
CI. If it has already happened, `git checkout -- .claude/skills
skills-lock.json` undoes it. Vendored files are CRLF on Windows, so a re-add
also changes the recorded hash without changing the content.

## Adding one

```bash
npx skills add <owner/repo@skill> --agent claude-code --copy -y
```

Always use `--copy` (not the default symlink) so the skill is committed and
works in CI and in fresh clones. Then add a row to the table above.

Before adding, apply the quality bar from `find-skills`: prefer 1K+ installs and
a reputable source (`vercel-labs`, `anthropics`, `prisma`, `microsoft`); treat
anything under 100 installs with suspicion. Skills run with full agent
permissions — read `SKILL.md` and any `scripts/` before committing.
