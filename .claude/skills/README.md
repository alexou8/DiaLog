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

## Updating

```bash
npx skills update
```

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
