# Ultimate MCP Client - AGENTS.md Research Summary

## AGENT.md Structure (line references)
- Lines 1-107: Core narrative sections I-VII covering UMS, AML, innovations, literature comparison, and conclusion.
  - I. Introduction (lines 3-10)
  - II. UMS details: memory hierarchy, schema, operations (lines 12-49)
  - III. AML orchestration cycle and planning (lines 51-76)
  - IV. Innovations (lines 78-91)
  - V. Literature comparison (lines 93-101)
  - VI. Significance/publishability (lines 102-105)
  - VII. Conclusion (lines 107-109)
- Lines 111-245: Walkthrough example of a financial analysis task (Phase 1-5, loops 1-K).
- Lines 246-1037: Second, more granular simulation of the same task with detailed loop steps and database-state narration.

## Key Takeaways from AGENT.md
- AGENT.md is a deep architecture and research narrative (UMS + AML cognitive system).
- It does not include repo-specific safety rules, dev workflow steps, or quality gates.
- It is long-form and not a quick-reference entry point for agents.

## .cursor/rules Summary (short)
- 00-package-management-with-uv.mdc: use `uv`; key commands `uv lock --upgrade`, `uv sync --all-extras`.
- 01-project-overview.mdc: core files (`mcp_client.py`, `mcp_client_multi.py`, `agent_master_loop.py`) and docs.
- 02-mcp-client-architecture.mdc: key classes (MCPClient, ServerManager, RobustStdioSession, ConversationGraph, ToolCache); CLI + Web UI stack; stdio safety wrappers.
- 03-agent-master-loop.mdc: AgentState, AgentMasterLoop, planning/execution, memory/meta-cognition, error handling.
- 04-server-discovery-integration.mdc: discovery methods (filesystem, mDNS, port scan, registries, Claude Desktop), stdio safety.
- 05-conversation-management.mdc: branching conversation graph, persistence, CLI commands.
- 06-tool-integration.mdc: tool discovery/routing/execution, caching, dependency tracking.
- 07-web-interface.mdc: FastAPI + WebSockets backend, Alpine/Tailwind frontend, features.
- 08-observability-monitoring.mdc: OpenTelemetry, dashboards, logging, debug features.

## Recommended Content for AGENTS.md (quick reference)
- Short pointer to AGENT.md for the deep architecture narrative.
- Safety and stdio guidance (avoid stdout pollution when using stdio servers; prefer stderr for logs).
- Dev workflow: how to run CLI and Web UI, where config/env lives.
- Quality gates: `uv sync`, `pytest` (unit/integration), optional `ruff`/`mypy` if configured.
- Key entry points and files: `mcp_client.py`, `agent_master_loop.py`, `mcp_client_ui.html`.
- Explicit mention that uv is the required package manager.

