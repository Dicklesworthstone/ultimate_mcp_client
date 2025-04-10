# 🧠 Ultimate MCP Client

<div align="center">

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Protocol](https://img.shields.io/badge/Protocol-MCP-purple.svg)](https://github.com/mpctechdebt/mcp)
<!-- Optional: Add build status, code coverage badges later -->

A comprehensive, asynchronous client for the **Model Context Protocol (MCP)**. It bridges the gap between powerful AI models like Anthropic's Claude and a universe of external tools, local/remote servers, and contextual data sources, enabling complex, stateful interactions.

![Web UI Screenshot](https://github.com/Dicklesworthstone/ultimate_mcp_client/blob/main/banner.webp) 
<!-- Consider updating screenshot if banner.webp is just an illustration -->

</div>

> Built by Jeffrey Emanuel

---

## 🎯 Purpose & Motivation

The Model Context Protocol (MCP) standardizes how AI models interact with external capabilities (tools, resources, prompts). This client aims to be the **ultimate interface** for leveraging MCP, providing:

1.  **Robust Connectivity:** Reliably connect to diverse MCP servers (STDIO, SSE) with built-in resilience.
2.  **Rich User Experience:** Offer both a powerful interactive CLI and a modern, reactive Web UI.
3.  **Advanced State Management:** Go beyond simple chat history with forkable conversation graphs and smart context optimization.
4.  **Developer Introspection:** Provide observability via OpenTelemetry and dashboards for monitoring and debugging.
5.  **Seamless Integration:** Easily discover and integrate local, network, and registry-based MCP servers.

---

## 🔌 Key Features

- **Dual Interfaces: Web UI & CLI**
    - **Web UI:** Beautiful, reactive interface built with Alpine.js, DaisyUI, and Tailwind CSS. Features real-time chat streaming, server/tool management, visual conversation branching, settings configuration, theme switching, and direct tool execution modals.
    - **CLI:** Feature-rich interactive shell (`/commands`, autocompletion, Markdown) and batch-mode operation via Typer. Includes a live TUI dashboard.

- **Robust Server Connectivity & Management**
    - Supports `stdio` and `sse` (HTTP Server-Sent Events) MCP servers.
    - **Advanced STDIO Handling:** Features a custom `RobustStdioSession` to gracefully handle noisy or non-compliant `stdio` servers, preventing protocol corruption. Includes critical safety mechanisms (`StdioProtectionWrapper`, `safe_stdout`, `get_safe_console`) to protect `stdio` streams from accidental output pollution – a significant engineering challenge for reliable `stdio` communication.
    - **Resilience:** Automatic retries with exponential backoff and circuit breakers for failing servers. Background health monitoring (`ServerMonitor`).
    - **Process Management:** Handles starting, stopping, and monitoring `stdio` server processes.

- **Intelligent Server Discovery**
    - Auto-discovers local `stdio` servers (Python/JS scripts) in configured paths.
    - **mDNS Discovery:** Real-time discovery and notification of MCP servers on the local network (`_mcp._tcp.local.`). Interactive commands (`/discover`) for managing LAN servers.
    - **Registry Integration:** Connects to remote MCP registries to find and add shared servers.
    - **Claude Desktop Import:** Automatically detects `claude_desktop_config.json`, intelligently adapts `wsl.exe` commands and Windows paths for seamless execution within the Linux/WSL environment.

- **Powerful AI Integration & Streaming**
    - Deep integration with Claude models via the `anthropic` SDK, supporting multi-turn tool use.
    - **Real-time Streaming:** Streams AI responses and tool status updates via WebSockets (Web UI) and live TUI rendering (CLI). Handles complex streaming scenarios, including partial JSON input accumulation for tool calls.
    - **Intelligent Tool Routing:** Directs tool calls to the correct originating server based on loaded capabilities.
    - **Direct Tool Execution:** Run specific tools with custom parameters via `/tool` command or Web UI modal.

- **Advanced Conversation Management**
    - **Branching:** Forkable conversation graphs (`ConversationGraph`) allow exploring different interaction paths. Visually represented in the Web UI.
    - **Persistence:** Conversation graphs are saved to JSON, preserving branches and state across sessions.
    - **Context Optimization:** Automatic or manual summarization of long conversation histories using a specified AI model to stay within context limits (`/optimize`).
    - **Dynamic Prompts:** Inject pre-defined prompt templates from servers into the conversation context (`/prompt`).
    - **Import/Export:** Easily save and load conversation branches in a portable JSON format (`/export`, `/import`).

- **Observability & Monitoring**
    - **OpenTelemetry:** Integrated metrics (counters, histograms) and tracing (spans) for monitoring client and server performance. Console exporters available for debugging.
    - **Live Dashboards:** Real-time monitoring TUI (`/dashboard`) and Web UI visualizations showing server health, tool usage, and client stats.

- **Smart Caching**
    - Optional disk (`diskcache`) and in-memory caching for tool results.
    - Configurable Time-To-Live (TTL) per tool category.
    - **Dependency Tracking:** Invalidate related caches automatically when a dependency's data changes (e.g., update `weather:current` invalidates `weather:forecast`). Viewable via `/cache dependencies`.

---

## 🚀 Quickstart

### Install Dependencies

> **Requires Python 3.13+**

First, install [uv](https://github.com/astral-sh/uv) if you don't have it already:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex" 
```

Then clone the repository, set up a virtual environment using Python 3.13+, and install packages:

```bash
git clone https://github.com/Dicklesworthstone/ultimate_mcp_client
cd ultimate_mcp_client
# Create venv using uv (recommended)
uv venv --python 3.13 
# Or using standard venv
# python3.13 -m venv .venv 

# Activate environment
source .venv/bin/activate # Linux/macOS
# .venv\Scripts\activate # Windows

# Install dependencies using uv (fastest)
uv sync --all-extras 
# Or using pip
# pip install -e ".[all]" # Or just pip install -e . if you only need core deps
```

### Configure API Key

Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..." 
# Or add it to a .env file in the project root
```

Alternatively, set it later using the `/config api-key ...` command in the interactive CLI or via the Web UI settings.

### Launch the Web UI (Recommended)

```bash
mcpclient run --webui
```

Then open your browser to `http://127.0.0.1:8017` (or the configured host/port).

You can customize the host and port:

```bash
mcpclient run --webui --host 0.0.0.0 --port 8080
```

### Run Interactive CLI Mode

```bash
mcpclient run --interactive
```

### Run a One-Off Query

```bash
mcpclient run --query "What's the weather in New York?"
```

### Show the CLI Dashboard

```bash
mcpclient run --dashboard
```

### Import and Export Conversations

```bash
# Export current conversation branch
mcpclient export --output my_conversation.json

# Export specific conversation branch by ID (first 8 chars often suffice)
mcpclient export --id 12345678 --output specific_branch.json

# Import a conversation file (creates a new branch under the current node)
mcpclient import-conv my_conversation.json
```

---

## 🌐 Web UI Features

The web UI (`mcpclient run --webui`) provides a modern, user-friendly interface:

-   **Real-time Chat:** Streamed responses from Claude, Markdown rendering, code highlighting with copy buttons.
-   **Tool Interaction:** Clear display of tool calls and results within the chat flow. Direct tool execution modal for testing.
-   **Server Management:** Add/remove servers (STDIO/SSE), connect/disconnect, enable/disable, view status and discovered tools.
-   **Discovery:** Buttons to trigger local, registry, and mDNS discovery; list discovered servers and add them easily.
-   **Conversation Branching:** Interactive tree view of the conversation graph; checkout different branches, fork new ones.
-   **Context Management:** Clear context, trigger optimization/summarization.
-   **Import/Export:** Buttons to export the current branch or import a previously exported JSON file.
-   **Settings:** Configure API key, default model, temperature, feature toggles (streaming, caching, discovery).
-   **Theme Switching:** Choose from various DaisyUI themes with automatic light/dark mode support for code highlighting.
-   **Status Indicators:** Real-time WebSocket connection status, server/tool counts.

---

## 🔌 API Server

When running with `--webui`, a FastAPI server provides programmatic access:

```
GET    /api/status                     - Client overview (model, servers, tools, history count)
GET    /api/config                     - Get current (non-sensitive) configuration
PUT    /api/config                     - Update configuration settings
GET    /api/servers                    - List all configured servers with status/health
POST   /api/servers                    - Add a new server configuration
DELETE /api/servers/{server_name}    - Remove a server configuration
POST   /api/servers/{server_name}/connect    - Connect to a specific server
POST   /api/servers/{server_name}/disconnect - Disconnect from a specific server
PUT    /api/servers/{server_name}/enable     - Enable/disable a server (connects/disconnects if needed)
GET    /api/tools                      - List all available tools from connected servers
GET    /api/resources                  - List all available resources
GET    /api/prompts                    - List all available prompts
GET    /api/conversation               - Get current conversation state (messages, current node, node graph)
POST   /api/conversation/fork          - Create a fork from the current conversation node
POST   /api/conversation/checkout      - Switch the current context to a different conversation node/branch
POST   /api/conversation/clear         - Clear messages on the current node and switch to root
POST   /api/conversation/optimize      - Trigger context summarization for the current node
POST   /api/tool/execute               - Execute a specific tool with given parameters
WS     /ws/chat                        - WebSocket endpoint for streaming chat and status updates
```

---

## ⚙️ Commands

### CLI Options

Run `mcpclient --help` or `mcpclient [COMMAND] --help` for details.

### Interactive Shell Commands (`mcpclient run --interactive`)

Type `/` followed by a command:

```text
/help         Show this help message  
/exit, /quit  Exit the client
/config       Manage configuration (api-key, model, etc.)
/servers      Manage MCP servers (list, add, remove, connect, disconnect, enable, disable, status)
/discover     Discover/manage LAN servers (list, connect, refresh, auto on|off)
/tools        List available tools (optionally filter by server)
/tool         Directly execute a tool: /tool <tool_name> '{"param": "value"}'
/resources    List available resources (optionally filter by server)
/prompts      List available prompt templates (optionally filter by server)
/prompt       Apply a prompt template to the current conversation context
/model        View or change the current AI model
/fork         Create a new branch from the current conversation point: /fork [Branch Name]
/branch       Manage branches (list, checkout <node_id>)
/export       Export current branch: /export [--id <node_id>] [--output <file.json>]
/import       Import conversation file: /import <file.json>
/history      View recent conversation history (optionally specify number: /history 10)
/cache        Manage tool cache (list, clear [--all|tool_name], clean, dependencies [tool_name])
/dashboard    Show the live Textual User Interface (TUI) dashboard
/optimize     Summarize current conversation context: /optimize [--model <model>] [--tokens <num>]
/reload       Disconnect, reload capabilities, and reconnect to enabled servers
/clear        Clear messages in the current branch and reset to root
```

---

## 🏗️ Architecture & Engineering Highlights

This client employs several techniques to provide a robust and feature-rich experience:

-   **Asynchronous Core:** Built entirely on Python's `asyncio` for efficient handling of network I/O, subprocess communication, and concurrent operations.
-   **Component-Based Design:** While monolithic for deployment ease, it internally separates concerns:
    -   `MCPClient`: Overall application orchestrator.
    -   `ServerManager`: Handles server lifecycle, discovery, capability aggregation, and process management using `AsyncExitStack` for reliable resource cleanup.
    -   `RobustStdioSession`: **(Key Engineering Effort)** A custom MCP `ClientSession` designed to handle unreliable `stdio` servers. It filters noisy output, directly resolves response futures, manages background reader tasks, and handles process termination gracefully. This is crucial for stability when interacting with diverse `stdio`-based tools.
    -   **STDIO Safety:** Global `StdioProtectionWrapper` on `sys.stdout`, `safe_stdout()` context manager, and `get_safe_console()` function work together to prevent accidental output pollution that could corrupt the `stdio` communication channel with servers. This allows safe coexistence of multiple `stdio` servers and user output (redirected to `stderr` when needed).
    -   `ConversationGraph`: Manages the branching conversation structure, persisted as JSON.
    -   `ToolCache`: Implements caching logic using `diskcache` and in-memory storage, including TTLs and dependency invalidation.
    -   `ServerRegistry` / `ServerMonitor`: Handle discovery protocols (mDNS/Zeroconf, remote registries) and background health checks.
-   **Dual Interface:**
    -   **Web Backend:** Uses `FastAPI` for the REST API, `uvicorn` for serving, and `websockets` for real-time communication. A lifespan manager ensures proper setup/teardown of the `MCPClient`.
    -   **Web Frontend:** Leverages `Alpine.js` for reactivity, `Tailwind CSS` + `DaisyUI` for styling, `Marked.js` + `highlight.js` + `DOMPurify` for rendering Markdown/code safely, and `Tippy.js` for tooltips.
    -   **CLI/TUI:** Uses `Typer` for command parsing and `Rich` for formatted output, tables, progress bars, Markdown rendering, and the live TUI dashboard. Includes careful management (`_run_with_progress`) to avoid issues with nested `Rich Live` displays.
-   **Resilience:** Employs decorators (`@retry_with_circuit_breaker`, `@with_tool_error_handling`) and structured exception handling throughout.
-   **Observability:** Integrates `OpenTelemetry` for metrics and tracing, providing insights into performance and behavior.
-   **Configuration:** Flexible configuration via YAML file (`config.yaml`), environment variables (`ANTHROPIC_API_KEY`), and interactive commands.

---

## 🔄 Smart Cache Dependency Tracking

*(Existing content is good, no changes needed here unless you want to add more detail)*

The Smart Cache Dependency system allows tools to declare dependencies on other tools:

- When a tool's cache is invalidated, all dependent tools are automatically invalidated
- Dependencies are registered when servers declare tool relationships
- View the dependency graph with `/cache dependencies`
- Improves data consistency by ensuring related tools use fresh data

Example dependency flow:
```
weather:current → weather:forecast → travel:recommendations
```
If the current weather data is updated, both the forecast and travel recommendations caches are automatically invalidated.

---

## 🔍 Tool & Server Discovery

- **Configured Paths:** Searches common locations for local `stdio` server scripts (e.g., `.py`, `.js`):
    - `~/.config/mcpclient/servers` (or project `.mcpclient_config/servers`)
    - `~/mcp-servers`
    - `~/modelcontextprotocol/servers`
- **Claude Desktop Config:** If `claude_desktop_config.json` is present in the project root:
    - Imports server configurations.
    - **Intelligently Adapts Commands:** Converts `wsl.exe ... bash -c "command"` into direct Linux shell execution (`/bin/bash -c "command"`).
    - Adapts Windows-style paths (`C:\...`) in arguments to Linux/WSL paths (`/mnt/c/...`) for non-WSL commands.
- **Remote Registries:** Connects to URLs defined in `config.yaml` (or `REGISTRY_URLS` default) to discover public/shared servers.
- **Local Network (mDNS/Zeroconf):**
    - Listens for `_mcp._tcp.local.` services on the LAN.
    - Provides real-time notifications in the interactive CLI when new servers appear.
    - `/discover list`: View details of discovered LAN servers.
    - `/discover connect NAME`: Add a discovered server to config and connect.
    - `/discover refresh`: Manually re-scan the network.
    - `/discover auto [on|off]`: Toggle continuous background mDNS scanning.

---

## 📡 Telemetry + Debugging

- **OpenTelemetry:** Generates traces for operations like query processing and server connections, and metrics like request counts and latencies. Configure exporters as needed (console exporter available but noisy).
- **Dashboards:**
    - CLI: `mcpclient run --dashboard` for a live TUI view.
    - Web UI: Provides visual server status and health indicators.
- **Logging:** Uses Python's `logging` with `RichHandler`. Log level controlled by `--verbose`/`-v` flag. Verbose mode also enables detailed `stdio` session logging (`USE_VERBOSE_SESSION_LOGGING`).
- **Error Tracebacks:** Set environment variable `MCP_CLIENT_DEBUG=1` to show full Python tracebacks on unexpected errors in the CLI.
- **STDIO Logs:** `stdio` server `stderr` is captured to `.mcpclient_config/<server_name>_stderr.log`.

---

## 📦 Configuration

- **Primary File:** `~/.config/mcpclient/config.yaml` (or `.mcpclient_config/config.yaml` in the project root if the home directory path is unavailable).
- **Environment Variables:** `ANTHROPIC_API_KEY` overrides the key in the config file.
- **Interactive Commands:** Use `/config` subcommands (e.g., `/config api-key ...`, `/config model ...`) to modify settings, which are saved to the YAML file.
- **Web UI Settings:** Changes made in the Web UI Settings tab are persisted via the API to the YAML file.

**View Current Config:**

```bash
mcpclient config --show 
# OR in interactive mode:
/config 
```

**Edit Config File Manually:**

```bash
mcpclient config --edit 
# (Uses EDITOR environment variable)
```

---

## 🧪 Development Notes

- **Core:** Python 3.13+, `asyncio`
- **CLI:** `Typer`, `Rich`
- **Web:** `FastAPI`, `Uvicorn`, `WebSockets`, `Alpine.js`, `Tailwind CSS`, `DaisyUI`
- **MCP:** `mcp` SDK
- **AI:** `anthropic` SDK
- **Observability:** `opentelemetry-sdk`, `opentelemetry-api`
- **Utilities:** `httpx`, `PyYAML`, `python-dotenv`, `psutil`, `aiofiles`, `diskcache`, `tiktoken`, `zeroconf`
- **Linting/Formatting:** `ruff` (`uv run lint` or `ruff check . && ruff format .`)
- **Type Checking:** `mypy` (`uv run typecheck` or `mypy mcpclient.py`)

The project is structured as a single primary Python file (`mcpclient.py`) for easier introspection and potential bundling, though modularity is maintained internally through classes. The Web UI (`mcp_client_ui.html`) is a self-contained HTML file using modern frontend libraries.

---

## 📝 License

MIT License.