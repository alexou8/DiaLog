# CLAUDE.md

Claude Code-specific guidance for this repository.

## Read AGENTS.md first

**[AGENTS.md](AGENTS.md) is the engineering contract** — commands, repository
layout, invariants, CI policy, testing tiers, verification, security and
data-handling rules, and style. It is tool-neutral and self-contained, so every
agent working here (Claude Code, Codex, Cursor, Copilot) follows the same rules
from the same file.

This file deliberately does **not** repeat any of it. Duplicating those facts
across two files guarantees they drift, and a stale invariant is worse than no
invariant. What follows is only the part that is specific to Claude Code:
skills, subagent orchestration, and model selection.

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

## Agent workflow

### Orchestrate, don't do everything yourself

The default posture for anything beyond a one-file change is: **the top-level
session plans and reviews, subagents execute.** The orchestrator's context is
the scarcest resource in the session — every file a subagent reads is a file the
orchestrator never has to hold.

Plan first, in the orchestrator. Read only enough to decide _what_ needs to
happen and _which files are involved_, then hand each unit of work to an agent
with a brief that already contains the facts you established, so the subagent
never re-derives them.

### Picking a model and effort level

Match the model to the shape of the work, not to how important it feels:

| Work                                                                                                                                                                               | Model                       | Notes                                                                            |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | -------------------------------------------------------------------------------- |
| Planning, architecture, cross-cutting refactors, security-sensitive changes (`lib/auth/**`, `middleware.ts`), final review                                                         | **Opus** — the orchestrator | Keep these in the top-level session.                                             |
| A scoped feature, a bug fix inside one module, a new import connector, a CI/config change, writing a test suite                                                                    | **Sonnet** subagent         | The workhorse. Most delegated work lands here.                                   |
| Mechanical, verifiable, low-judgement work — renames, adding a fixture, mapping one file's exports, running a command and reporting output, grepping for a pattern across the tree | **Haiku** subagent          | Cheap and fast. If a task needs no design judgement, it should not be on Sonnet. |

Effort follows the same logic. Spend it where a wrong answer is expensive:

- **High effort** — anything touching auth, session handling, authorization
  checks, glucose-value units, or dedupe logic. A silent bug there is a
  correctness or safety problem, not a cosmetic one.
- **Medium effort** — ordinary feature and fix work, test authoring, CI.
- **Low effort** — lookups, inventories, formatting, "what does this file
  export", "does this pattern appear anywhere else".

Use the `Explore` agent for broad read-only fan-out (searching many files,
"where is X handled") — it returns the conclusion instead of dumping file
contents into the orchestrator's context. Use `Plan` when the shape of a change
is genuinely unclear before committing to it.

### Writing a good subagent brief

A subagent starts cold. A brief that omits context guarantees the subagent
re-reads what you already read, which costs more than writing the brief. Every
brief should carry:

1. **Exact scope** — the files it may touch, and the files it must not.
2. **Facts you already established** — paths, function names, the failing
   command and its output, the constraint you discovered. Paste them; do not
   make the agent go find them.
3. **The invariants from AGENTS.md** that apply, restated. Do not assume it will
   read that file.
4. **Verification** — the exact command it must run before reporting done
   (`npm run typecheck`, `npm test`, the specific vitest file).
5. **"Do not commit or push."** The orchestrator owns the git history so that
   one reviewer sees one coherent diff.
6. **What to report back** — a short structured summary, not a transcript.

Run independent subagents in parallel in a single message. Serialize only where
one genuinely needs another's output.

### Parallelism and conflicts

Two agents editing the same file will clobber each other. Partition by
directory or by file before dispatching. When two units of work genuinely
overlap, either merge them into one brief or run them in sequence. For work
that needs to build and test in isolation, give the agent its own worktree.

### Reviewing what comes back

Subagent output is a claim, not a result. Before accepting it: read the diff,
run the verification command yourself, and check it did not widen scope. A
subagent reporting "all tests pass" without the output is not evidence. If the
work is wrong, send a correction to the same agent — it still has its context —
rather than spawning a fresh one.

## Token efficiency

AGENTS.md carries the general rules. The one that matters most when
orchestrating: **delegate wide reads.** If answering a question means opening
more than a handful of files, that is an `Explore` agent's job, not the
orchestrator's.
