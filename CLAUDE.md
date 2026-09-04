# CLAUDE.md

Claude Code-specific guidance for this repository.

## Read AGENTS.md first

**[AGENTS.md](AGENTS.md) is the engineering contract** — commands, repository
layout, invariants, CI policy, testing tiers, verification, security and
data-handling rules, and style. It is tool-neutral and self-contained, so every
agent working here (Claude Code, Codex, Cursor, Copilot) follows the same rules
from the same file.

This file deliberately does **not** restate the contract. Duplicating those
facts across two files guarantees they drift, and a stale invariant is worse
than no invariant. What follows is only the part specific to Claude Code:
Codex delegation, skills, subagent orchestration, and model selection.

The one deliberate exception is a task brief. Copying the applicable
invariants into a brief for an agent that starts cold is not documentation
duplication — a brief is ephemeral and is discarded with the task, so it
cannot drift out of sync with `AGENTS.md` the way a second checked-in copy
would.

DiaLog is an accessible glucose and metabolic health tracking app handling
personal health data. If you read nothing else in AGENTS.md, read its
**Invariants** section before changing code.

## Skills

Vendored under `.claude/skills/` and committed, so every session and every
contributor gets the same behaviour with no install step. See
[SKILLS.md](SKILLS.md) for when to use each and how they take precedence, and
`.claude/skills/README.md` for provenance and the quality bar for adding more.

Reach for them rather than improvising:

- `frontend-design`, `ui-ux-pro-max` — building or reshaping UI.
- `web-design-guidelines` — reviewing UI code before you call it done.
- `accessibility` — WCAG 2.2 auditing. This is an accessibility-first product
  with an axe-core e2e suite; treat this skill as part of the definition of
  done for any UI change, not an optional extra.
- `vercel-react-best-practices` — React 19 / Next.js 15 performance work.
- `prisma-cli` — migrations, generate, seeding, studio.
- `find-skills` — before hand-rolling a capability that probably already exists
  as a skill.

The Codex plugin also ships skills (`codex-cli-runtime`, `codex-result-handling`,
`gpt-5-4-prompting`). They arrive with the plugin rather than being vendored
under `.claude/skills/`, and they govern how to talk to Codex, not how to build
DiaLog. The precedence rules in [SKILLS.md](SKILLS.md) still apply: `AGENTS.md`
outranks all of them.

## Agent workflow

### The operating model

> **Claude orchestrates and maintains the engineering conversation. Explore
> investigates. Codex implements. Claude reviews and responds. Codex resumes
> and corrects. Claude verifies.**

Claude Code is the architect, planner, reviewer, and final engineering decision
maker. OpenAI Codex — invoked through the official
[`openai/codex-plugin-cc`](https://github.com/openai/codex-plugin-cc) plugin —
is the primary implementation engineer for substantive code changes.

The two are not one-shot agents handing a baton over a wall. For anything beyond
a trivial change the loop is:

```text
Claude investigates + plans
  → /codex:rescue  (bounded implementation contract)
    → Codex investigates, implements, tests, reports findings
  → Claude reads the actual diff, not just the report
    → accepts, or sends focused corrections
      → /codex:rescue --resume  (same Codex thread, same context)
        → Codex fixes, reruns verification, reports exact output
  → Claude verifies, and adjudicates anything Codex pushed back on
```

That loop repeats until the acceptance criteria are genuinely met. Claude makes
the final call; Codex is expected to surface implementation realities Claude
missed rather than implement a plan it can see is wrong.

### Setup and prerequisites

The plugin is declared in `.claude/settings.json`, so a session picks it up
automatically. What is not in the repository is the machine-level state:

```bash
npm install -g @openai/codex   # if the codex CLI is missing
codex login                    # or: codex login --device-auth  (remote/headless)
codex login status             # verify
```

Then `/codex:setup` inside Claude Code. Do **not** pass
`--enable-review-gate`: review is a decision Claude makes per change, not an
automatic loop that spends usage on every stop. Enable it only if asked.

If `codex login status` reports "Not logged in", stop and tell the user — Codex
delegation cannot work, and silently falling back to implementing everything in
Claude is a worse outcome than saying so.

### Commands

Only these exist. Do not invent others; check `/codex:status` output and the
installed plugin version before documenting anything new.

| Command                     | Use it for                                                                      |
| --------------------------- | ------------------------------------------------------------------------------- |
| `/codex:rescue`             | The primary implementation command. Send a bounded contract (below).            |
| `--background`              | Long work. Claude keeps investigating adjacent, non-overlapping code meanwhile. |
| `--resume`                  | Corrections, follow-ups, review findings, test failures — same Codex thread.    |
| `--fresh`                   | A genuinely new task where prior context would mislead. Not the default.        |
| `--model` / `--effort`      | Match cost to risk. See the effort table below.                                 |
| `/codex:status`             | Job state for background work.                                                  |
| `/codex:result`             | Retrieve completed output. A claim, not proof.                                  |
| `/codex:cancel`             | Stop a job that is no longer wanted.                                            |
| `/codex:review`             | Read-only review of the change (`--base main`, `--background`).                 |
| `/codex:adversarial-review` | Pressure-test a high-risk design with an explicit risk focus.                   |
| `/codex:transfer`           | Move the session into a persistent Codex thread. Rarely; not the default.       |

Prefer `--resume` over `--fresh` whenever the existing thread already
understands the task, the files, the implementation choices, the errors it hit,
and the test context. Starting cold throws all of that away.

### Picking who does the work

Match the worker to the shape of the work, not to how important it feels:

| Work                                                                                                                                    | Worker                           | Notes                                                                                    |
| --------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ---------------------------------------------------------------------------------------- |
| Planning, architecture, requirements, security reasoning, domain decisions, adjudicating conflicts, reviewing diffs, final verification | **Claude orchestrator (Opus)**   | Never delegate the decision. Keep these in the top-level session.                        |
| Substantive implementation: features, bug fixes, refactors, migrations, test suites, debugging, connectors                              | **Codex** via `/codex:rescue`    | The default implementation path.                                                         |
| Broad read-only fan-out — tracing a flow, finding references, locating tests, mapping architecture                                      | **`Explore` agent**              | Returns the conclusion instead of dumping file contents into the orchestrator's context. |
| Mechanical, verifiable, low-judgement work — inventories, "what does this file export", "does this pattern appear anywhere else"        | **Haiku subagent**               | Cheap and fast. No design judgement required.                                            |
| A scoped change where round-tripping to Codex costs more than doing it — a one-file edit, a docs fix                                    | **Claude, or a Sonnet subagent** | Delegation has overhead. Do not perform the ceremony for a two-line change.              |

Effort follows the same logic — spend it where a wrong answer is expensive:

- **High** — auth, session handling, authorization checks, glucose units, dedupe
  logic, migrations, concurrency. A silent bug there is a safety problem.
- **Medium** — ordinary feature and fix work, test authoring, CI.
- **Low** — lookups, inventories, formatting, mechanical repairs.

Do not reach for the most expensive Codex configuration by default.

### Writing a Codex implementation contract

Codex starts cold. "Improve this feature" or "fix everything" is not a task.
Every contract carries the facts Claude already established, so Codex does not
re-derive them:

1. **Task** — one implementation objective, and why it exists.
2. **Context** — paths, function names, the failing command and its exact
   output, the constraint you discovered. Paste them.
3. **Scope** — the files it may touch, and the files it must not.
4. **Existing architecture** — the services, patterns, and abstractions already
   in play, so Codex extends them instead of inventing a parallel one.
5. **Required invariants** — restate the applicable rules from `AGENTS.md`
   verbatim. Do not assume Codex will read the whole file.
6. **Failure paths** — only the ones that apply: invalid input, unauthenticated
   user, another user's resource, missing resource, empty state, dependency
   failure, partial failure, concurrent modification, malformed import,
   unavailable provider.
7. **Acceptance criteria** — a checklist that can actually be evaluated.
8. **Verification** — the exact commands to run before reporting back
   (`AGENTS.md` has them; narrowest check first, full suite before handing
   back).
9. **Communication** — if the plan conflicts with the repository, would violate
   an invariant, rests on a wrong assumption, duplicates an existing
   abstraction, or hides a security problem: report it, do not silently work
   around it.
10. **Restrictions** — do not commit, push, weaken tests, skip failing tests,
    bypass CI checks, widen scope without saying so, log health values or user
    free text, or remove owner scoping.
11. **Return to Claude** — implementation summary, files changed, design
    decisions, unexpected findings, assumptions, verification commands and
    their exact output, unresolved risks, and questions for Claude to resolve.
    "Done." is not a report.

For UI work, put the conclusions from `accessibility`, `frontend-design`, and
`web-design-guidelines` into the contract. Claude reads the skills; Codex should
not have to rediscover them.

### Reviewing what comes back

**Codex output is a claim. The diff is the evidence.** Before accepting
anything:

```bash
git status
git diff
```

Then read the changed files. Evaluate correctness, architecture, ownership
boundaries and the `AGENTS.md` invariants, security, test coverage,
accessibility, failure paths, and scope creep. Run the verification command
yourself where practical — a report saying "tests pass" without the output is
not evidence, and unit tests passing while an integration test fails is new
input for Codex, not a reason for Claude to take over.

When something is wrong, do not silently reimplement the feature in Claude.
Send focused feedback through `--resume`: what is wrong, the mechanism, which
files to touch, what to re-run, and what not to change. "Fix the UI" is not
feedback.

Use `/codex:review` (or `/codex:adversarial-review` with a named risk, for auth,
health data, deletion, AI safety, or migrations) as a second channel on
substantial work. Adjudicate the findings — reject false positives explicitly,
convert accepted ones into a `--resume` fix task. Do not implement every
suggestion reflexively.

### When Codex pushes back

Codex is expected to challenge a plan that conflicts with the repository, would
violate an invariant, is impossible with the current architecture, overlooks a
failure path, or duplicates an existing abstraction. When it does, inspect the
evidence and decide. The hierarchy is:

```text
repository evidence → AGENTS.md invariants → tests/CI → documented architecture → engineering reasoning
```

Neither agent wins by confidence. Claude decides after weighing that evidence —
and does not require Codex to implement a plan that is demonstrably wrong just
because Claude proposed it first.

### CI failures

Claude owns diagnosis: read the exact failure, work out the mechanism, then
send Codex a bounded reproduction-and-fix task. If it is still red, the exact
new failure goes back into the _same_ Codex context via `--resume`. Never
accept a skipped test, weakened assertion, arbitrary sleep, `continue-on-error`,
`|| true`, a deleted check, or looser lint/type settings as a fix — see the CI
section of `AGENTS.md`.

### Parallelism and conflicts

Two workers editing the same file will clobber each other. Partition by
directory or file before dispatching, and never run two implementation workers
against overlapping paths — merge them into one contract, serialize them, or
give each its own git worktree. Read-only `Explore` agents can always run
alongside an active Codex job. Claude owns the coordination.

Use worktrees when they materially improve isolation, not for trivial changes.

### Git ownership

Claude owns repository integration. Codex edits code, adds tests, runs
verification, and reports — it does not commit, push, merge, force-push,
rewrite history, or open PRs. Every contract says so explicitly.

Before any agent work, check `git status` and `git diff`. If the user has
uncommitted changes, preserve them and keep them mentally separate from agent
changes in review. Never run a destructive git command just to get a clean
workspace.

### Work state

Use GitHub Issues for durable, user-visible work and `git diff` plus tests for
implementation truth. Do not stand up a parallel task system — a `TASKS.md`, a
`PLAN.md`, and an issue tracker describing the same work will disagree within a
week.

## Token efficiency

AGENTS.md carries the general rules. The one that matters most when
orchestrating: **delegate wide reads.** If answering a question means opening
more than a handful of files, that is an `Explore` agent's job, not the
orchestrator's.
