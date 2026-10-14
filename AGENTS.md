# AGENTS.md - Ultimate MCP Client

> Standard agent guidelines for AI coding assistants.
> For comprehensive architecture and theory, see [AGENT.md](AGENT.md).

## RULE 0.5 - SUITE-WIDE RULES LIVE IN /data/projects/AGENTS.md

The suite-wide rules in **`/data/projects/AGENTS.md`** bind you here too. Read it. Two sections
are load-bearing for perf work and are NOT duplicated below, so they cannot drift out of sync:

- **`## Named Reward-Hacking Patterns (ALL FORBIDDEN)`** — 12 named patterns, several already
  observed in this suite: gate self-weakening (and the exact price of a legitimate gate fix),
  proof-class inflation, golden regeneration reflex, commit-stream pumping, tautological tests,
  easy-lever cherry-picking, close-pump abuse, scope-splitting, spec-editing as progress,
  conformance metastasis, dependency smuggling, bench-path hardcoding.
- **`### Work-Graph Discipline`** — JSONL is truth and `beads.db` is disposable, `br sync
  --import-only` after every pull, single-writer on graph structure, closure on cited evidence
  with blocker beads gated on their named probe, `br dep cycles` stays empty.

The three that most often decide whether a number here is real: a **self-speedup is
MAINTENANCE, not a win** — a win needs the incumbent live in the SAME invocation; **never
weaken a gate to land a change**, and if a gate is genuinely defective, meet the evidence
standard and publish the win/lose split of what the fix admits; and **reporting a loss is a
success** — one line, revert, next lever, no retraction narrative.

---

## RULE 1 - NO FILE DELETION
Do not delete any file or directory unless the user explicitly gives the exact command in this session.

## IRREVERSIBLE GIT & FILESYSTEM ACTIONS
Never run destructive commands (e.g., `git reset --hard`, `git clean -fd`, `rm -rf`) without explicit user approval and the exact command in the same message.

---

## Quick Reference
- Package manager: `uv` (do not use pip).
- Run CLI: `mcpclient run --interactive`
- Run Web UI: `mcpclient run --webui`
- Config: `.mcpclient_config/config.yaml` (edit via `mcpclient config --edit`).

## Safety First
- Keep stdout clean when stdio servers are active; log to stderr.
- Avoid leaking API keys; use `.env` and environment variables.
- Respect rate limits and timeouts when calling servers/tools.

## Workflow Overview
- Read [README.md](README.md) for usage and commands.
- Use `mcpclient` commands for server discovery and management.
- See `.cursor/rules/` for architecture notes and class-level guidance.

## Quality Gates
- Install deps: `uv sync` (or `uv sync --all-extras`).
- Lint/format: `uv run lint` or `ruff check . && ruff format .`
- Type check: `uv run typecheck` or `mypy mcpclient.py`

## Project-Specific Notes
- Core logic lives in `mcp_client.py` (CLI + Web UI + server management).
- AML logic in `agent_master_loop.py`.
- STDIO safety wrappers are critical; avoid direct writes to stdout.

For any web requests you must make with curl or otherwise, identify yourself honestly: use a User-Agent that names this project, e.g. "ultimate_mcp_client-agent (+https://github.com/Dicklesworthstone/ultimate_mcp_client)". Never borrow another product's or company's crawler or client User-Agent, and never spoof a client to get past a block.
