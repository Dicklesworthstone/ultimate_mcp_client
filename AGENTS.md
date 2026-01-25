# AGENTS.md - Ultimate MCP Client

> Standard agent guidelines for AI coding assistants.
> For comprehensive architecture and theory, see [AGENT.md](AGENT.md).

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

