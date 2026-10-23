# Agent Commandments

If you are an AI agent, follow all of these rules while working on this project.

## Correctness & Honesty

**Commandment #1 — No fake fallbacks.**
Avoid hard-coded fallback solutions and instead prioritise traditional error handling in line with industry practice.

**Commandment #2 — No magic.**
Never invent 'magic' solutions. If you don't know whether an API, library, or method exists, verify it — don't guess and present it as fact.

**Commandment #3 — Structure over monoliths.**
Ensure code is organized and structured, avoiding single-file monoliths. Spread functionality across smaller functions or files. Enforce a policy of 'one language per file', with the sole exception of SQL should it be required.

**Commandment #4 — Never fake success.**
Never mark a task complete or tests as passing without actually running them and observing the output.

**Commandment #5 — Don't cheat the tests.**
Never delete, skip, or weaken a test just to make it pass. Fix the underlying code, or flag the test as wrong and explain why.

**Commandment #6 — No silent failures.**
Don't swallow exceptions with empty `catch` blocks or bare `except: pass`. Log, handle, or propagate errors explicitly.

## Scope Discipline

**Commandment #7 — Stay in scope.**
Only change what was asked. No unrelated refactors, renames, or "improvements" bundled into the same change.

**Commandment #8 — No surprise dependencies.**
Don't introduce a new library or dependency without flagging it first, even if it solves the problem faster.

**Commandment #9 — Match the house style.**
Follow the existing code style and conventions of the file or repo rather than imposing your own.

## Safety

**Commandment #10 — Protect secrets.**
Never commit secrets, API keys, or credentials. Flag them if found already present in the codebase.

**Commandment #11 — No destructive surprises.**
Never run destructive commands (`rm -rf`, force-push, `DROP TABLE`, migrations, etc.) without explicit confirmation first.

**Commandment #12 — Declare your blast radius.**
Never modify files outside the stated task scope without calling it out before doing so.

## Process

**Commandment #13 — Explain before acting.**
For any non-trivial change, briefly explain *why* before diving into *what*.

**Commandment #14 — Stop, don't spiral.**
If stuck after several attempts, stop and ask for guidance rather than escalating into hacky workarounds.

**Commandment #15 — Smallest correct diff.**
Prefer the smallest correct change over a rewrite, unless a rewrite was explicitly requested.

**Commandment #16 — Leave it clean.**
No dead code, commented-out blocks, or debug print statements in the final version.

**Commandment #17 — No placeholder pretending.**
Don't present placeholder or TODO logic as if it were finished, working code.

## Documentation

**Commandment #18 — Keep docs in sync.**
Update relevant docs, comments, and READMEs when behavior changes — don't let them drift out of date.
