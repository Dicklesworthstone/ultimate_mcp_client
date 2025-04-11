# 🧠 Ultimate MCP Client

<div align="center">

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Protocol](https://img.shields.io/badge/Protocol-MCP-purple.svg)](https://github.com/mpctechdebt/mcp)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A comprehensive, asynchronous client for the **Model Context Protocol (MCP)**. It bridges the gap between powerful AI models like Anthropic's Claude and a universe of external tools, local/remote servers, and contextual data sources, enabling complex, stateful interactions.

![Web UI Screenshot](https://github.com/Dicklesworthstone/ultimate_mcp_client/blob/main/banner.webp)

</div>

> Built by Jeffrey Emanuel

---

## 🎯 Purpose & Motivation

The Model Context Protocol (MCP) standardizes how AI models interact with external capabilities (tools, resources, prompts). This client aims to be the **ultimate interface** for leveraging MCP, providing:

1.  **Robust Connectivity:** Reliably connect to diverse MCP servers (`stdio`, `sse`) with built-in resilience, advanced error handling, and intelligent discovery.
2.  **Rich User Experience:** Offer both a powerful interactive CLI and a modern, reactive Web UI, ensuring usability for diverse workflows.
3.  **Advanced State Management:** Go beyond simple chat history with forkable conversation graphs, persistent state across sessions, and smart context optimization techniques.
4.  **Developer Introspection:** Provide deep observability via OpenTelemetry metrics and traces, alongside live dashboards for monitoring client and server health and performance.
5.  **Seamless Integration:** Easily discover and integrate local filesystem scripts, local network (mDNS) servers, local port scan results, remote registry entries, and even configurations from existing tools like the Claude Desktop app.

**This project tackles significant engineering challenges**, especially around reliable `stdio` communication, asynchronous state management, and providing a consistent experience across both CLI and Web interfaces, to deliver a truly comprehensive MCP client solution.

---

## 🔌 Key Features

-   **Dual Interfaces: Web UI & CLI**
    -   **Web UI:** Beautiful, reactive interface built with **Alpine.js**, **DaisyUI**, and **Tailwind CSS**. Features real-time chat streaming via WebSockets, server/tool management modals, **visual conversation branching view**, settings configuration, **multiple theme switching** (respecting light/dark modes for code highlighting), and direct tool execution capabilities.
    -   **CLI:** Feature-rich interactive shell (`/commands`, autocompletion, **Rich** Markdown rendering) and batch-mode operation via **Typer**. Includes a live **Textual User Interface (TUI) dashboard** for monitoring.

-   **Robust Server Connectivity & Management**
    -   Supports `stdio` and `sse` (HTTP Server-Sent Events) MCP servers.
    -   **Advanced STDIO Handling (Key Engineering Effort):** Features a custom `RobustStdioSession` designed to gracefully handle potentially noisy or non-compliant `stdio` servers. It includes:
        -   *Noise Filtering:* Ignores non-JSON-RPC output that could corrupt the protocol.
        -   *Direct Future Resolution:* Avoids complex queueing for faster response handling.
        -   *Process Lifecycle Management:* Reliably starts, monitors, terminates, and kills `stdio` server processes.
        -   **Critical STDIO Safety:** A multi-layered system (`StdioProtectionWrapper`, `safe_stdout` context manager, `get_safe_console()` function) prevents accidental writes to `sys.stdout` from corrupting the `stdio` channel, ensuring safe coexistence of multiple `stdio` servers and user output (redirected to `stderr` when necessary). This was crucial for stability.
    -   **Resilience:** Automatic connection retries with exponential backoff and circuit breakers for failing servers (`@retry_with_circuit_breaker`). Background health monitoring (`ServerMonitor`) checks server responsiveness.

-   **Intelligent Server Discovery**
    -   Auto-discovers local `stdio` servers (Python/JS scripts) in configured filesystem paths.
    -   **mDNS Discovery (Zeroconf):** Real-time discovery and notification of MCP servers (`_mcp._tcp.local.`) on the local network. Interactive commands (`/discover list`, `/discover connect`, etc.) for managing LAN servers.
    -   **Local Port Scanning:** Actively scans a configurable range of local ports (e.g., 8000-9000) for MCP servers by attempting an `initialize` handshake. Useful for servers not advertising via mDNS. Configurable via `/config port-scan ...` commands.
    -   **Registry Integration:** Connects to remote MCP registries (specified in config) to find and add shared servers.
    -   **Claude Desktop Import:** Automatically detects `claude_desktop_config.json` in the project root. **Intelligently adapts configurations:**
        -   Remaps `wsl.exe ... bash -c "cmd"` calls to direct Linux shell execution (`/bin/bash -c "cmd"`).
        -   Converts Windows-style paths (`C:\...`) in arguments to their Linux/WSL equivalents (`/mnt/c/...`) using `adapt_path_for_platform` for seamless integration.

-   **Powerful AI Integration & Streaming**
    -   Deep integration with Claude models via the official `anthropic` SDK, supporting multi-turn tool use scenarios.
    -   **Real-time Streaming:** Streams AI responses and tool status updates via WebSockets (Web UI) and live `Rich` rendering (CLI). Handles complex streaming events, including **partial JSON input accumulation (`input_json_delta`)** for tools requiring structured input.
    -   **Intelligent Tool Routing:** Directs tool calls to the correct originating server based on loaded capabilities, using sanitized names for API compatibility while tracking original names internally.
    -   **Direct Tool Execution:** Run specific tools with custom JSON parameters via the `/tool` command (CLI) or a dedicated modal (Web UI).

-   **Advanced Conversation Management**
    -   **Branching:** Forkable conversation graphs (`ConversationGraph`) allow exploring different interaction paths without losing history. Visually represented and navigable in the Web UI.
    -   **Persistence:** Conversation graphs (including all branches and messages) are automatically saved to JSON files in the config directory, preserving state across sessions.
    -   **Context Optimization:** Automatic or manual summarization (`/optimize`) of long conversation histories using a specified AI model (configurable) to stay within context limits.
    -   **Dynamic Prompts:** Inject pre-defined prompt templates obtained from servers into the current conversation context using the `/prompt` command.
    -   **Import/Export:** Easily save (`/export`) and load (`/import`) entire conversation branches in a portable JSON format for sharing or backup.

-   **Observability & Monitoring**
    -   **OpenTelemetry:** Integrated metrics (counters, histograms using `opentelemetry-sdk`) and tracing (spans) for monitoring client operations, server requests, and tool execution performance. Console exporters can be enabled for debugging.
    -   **Live Dashboards:**
        -   **CLI:** Real-time TUI dashboard (`/dashboard`) built with `Rich`, showing server health, tool usage stats, and client info.
        -   **Web UI:** Dynamically updates server status, health indicators, and capability counts.

-   **Smart Caching**
    -   Optional disk (`diskcache`) and in-memory caching for tool results to improve speed and reduce costs.
    -   Configurable Time-To-Live (TTL) per tool category (e.g., `weather`, `filesystem`).
    -   **Dependency Tracking:** Define relationships between tools. Invalidating one tool's cache (e.g., `stock:lookup`) can automatically invalidate dependent caches (e.g., `stock:analyze`). View the graph via `/cache dependencies`.

---

## 📸 Screenshots

A glimpse into the Ultimate MCP Client's interfaces:

<br/>

<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="50%">
        <img src="https://github.com/Dicklesworthstone/ultimate_mcp_client/blob/main/screenshots/terminal_example_01.webp?raw=true" alt="CLI Interactive Mode showing tool execution and streaming" width="95%">
        <br/>
        <p align="center"><small><em>Interactive CLI: Streaming response with tool call/result.</em></small></p>
      </td>
      <td align="center" valign="top" width="50%">
        <img src="https://github.com/Dicklesworthstone/ultimate_mcp_client/blob/main/screenshots/terminal_example_02.webp?raw=true" alt="CLI TUI Dashboard showing server status" width="95%">
        <br/>
        <p align="center"><small><em>Live TUI Dashboard: Real-time server & tool monitoring (<code>/dashboard</code>).</em></small></p>
      </td>
    </tr>
    <tr>
      <td align="center" valign="top" width="33%">
        <img src="https://github.com/Dicklesworthstone/ultimate_mcp_client/blob/main/screenshots/webui_example_01.webp?raw=true" alt="Web UI Chat Interface" width="95%">
        <br/>
        <p align="center"><small><em>Web UI: Main chat interface showing messages and tool interactions.</em></small></p>
      </td>
      <td align="center" valign="top" width="33%">
        <img src="https://github.com/Dicklesworthstone/ultimate_mcp_client/blob/main/screenshots/webui_example_02.webp?raw=true" alt="Web UI Server Management Tab" width="95%">
        <br/>
        <p align="center"><small><em>Web UI: Server management tab with connection status and controls.</em></small></p>
      </td>
      <td align="center" valign="top" width="33%">
        <img src="https://github.com/Dicklesworthstone/ultimate_mcp_client/blob/main/screenshots/webui_example_03.webp?raw=true" alt="Web UI Conversation Branching View" width="95%">
        <br/>
        <p align="center"><small><em>Web UI: Conversation tab showing the branching graph structure.</em></small></p>
      </td>
    </tr>
  </tbody>
</table>

> **Note:** Some screenshots feature tools like `llm_gateway:generate_completion`. These are provided by the [LLM Gateway MCP Server](https://github.com/Dicklesworthstone/llm_gateway_mcp_server), another project by the same author. This server acts as an MCP-native gateway, enabling advanced agents (like Claude, used by *this* client) to intelligently delegate tasks to various other LLMs (e.g., Gemini, GPT-4o-mini), often optimizing for cost and performance.

<br/>

---

## 🚀 Quickstart

### Install Dependencies

> **Requires Python 3.13+**

First, install [uv](https://github.com/astral-sh/uv) (the recommended fast Python package installer):

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
# .venv\Scripts\activate # Windows Powershell
# .\venv\Scripts\activate.bat # Windows CMD

# Install dependencies using uv (fastest)
uv sync --all-extras
# Or using pip (slower)
# pip install -e . # Installs only core dependencies
```

### Configure API Key

Set your Anthropic API key as an environment variable:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# Or add ANTHROPIC_API_KEY="sk-ant-..." to a .env file in the project root
```

Alternatively, set it later using the `/config api-key ...` command in the interactive CLI or via the Web UI settings panel.

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

### Configure Port Scanning

```bash
# Enable port scanning (if disabled)
mcpclient config port-scan enable true

# Set the range (example)
mcpclient config port-scan range 8000 8999
```

### Import and Export Conversations

```bash
# Export current conversation branch to a default filename
mcpclient export

# Export current branch to a specific file
mcpclient export --output my_conversation.json

# Export specific conversation branch by ID (first 8 chars often suffice)
mcpclient export --id 12345678 --output specific_branch.json

# Import a conversation file (creates a new branch under the current node)
mcpclient import-conv my_conversation.json
```

---

## 🌐 Web UI Features

The web UI (`mcpclient run --webui`) provides a modern, user-friendly interface built with Alpine.js, Tailwind CSS, and DaisyUI:

-   **Real-time Chat:** Streamed responses from Claude via WebSockets, rich Markdown rendering (via `Marked.js`), code block syntax highlighting (via `highlight.js`) with copy buttons, and clear display of system messages and status updates.
-   **Tool Interaction:** Visual separation and display of tool calls (arguments) and their results (success or error) directly within the chat flow. A modal allows direct execution of any available tool with custom JSON parameters for testing and debugging.
-   **Server Management:** A dedicated sidebar tab lists configured servers. Users can:
    -   Add new servers (STDIO/SSE) via a modal form.
    -   Connect/disconnect from servers with status indicators (loading, connected, disconnected, error).
    -   Enable/disable servers (automatically disconnects if disabled).
    -   Remove server configurations.
    -   View basic health/status and the number of tools provided by each server.
-   **Discovery:** Buttons in the Servers tab trigger discovery scans (local filesystem, remote registry, mDNS). Discovered servers are listed, allowing users to add them to the configuration with one click.
-   **Conversation Branching:** An interactive tree view in the Conversation tab visualizes the `ConversationGraph`. Users can click to checkout different branches (updating the chat view) or fork the current point to create new branches.
-   **Context Management:** Buttons in the Conversation tab allow users to clear the current branch's messages (resetting to parent/root state) or trigger context optimization/summarization via the API.
-   **Import/Export:** Buttons in the Conversation tab allow exporting the current branch to a JSON file or importing a previously exported JSON file via file selection (uploading to the backend).
-   **Settings:** A dedicated sidebar tab allows users to view and modify key configuration options (API Key, Default Model, Max Tokens, Temperature, feature toggles like Streaming, Caching, Auto-Discovery, mDNS) which are persisted to the server's config file.
-   **Theme Switching:** A dropdown menu allows selecting from a wide range of DaisyUI themes. The UI adapts instantly, and code highlighting automatically switches between light/dark styles based on the selected theme.
-   **Status Indicators:** Real-time WebSocket connection status icon, overall server connection count, and total available tool count are displayed in the header. Transient status messages (e.g., "Executing tool...") appear below the chat input.

---

## 🔌 API Server

When running with `--webui`, a FastAPI server provides programmatic access:

```
GET    /api/status                     - Client overview (model, servers, tools, history count, current node)
GET    /api/config                     - Get current (non-sensitive) configuration
PUT    /api/config                     - Update configuration settings (e.g., { "temperature": 0.8 })
GET    /api/servers                    - List all configured servers with status, health, tools
POST   /api/servers                    - Add a new server configuration (Requires ServerAddRequest model)
DELETE /api/servers/{server_name}    - Remove a server configuration
POST   /api/servers/{server_name}/connect    - Connect to a specific server
POST   /api/servers/{server_name}/disconnect - Disconnect from a specific server
PUT    /api/servers/{server_name}/enable?enabled={true|false} - Enable/disable a server
GET    /api/tools                      - List all available tools from connected servers
GET    /api/resources                  - List all available resources
GET    /api/prompts                    - List all available prompts
GET    /api/conversation               - Get current conversation state (messages, current node ID/name, full node graph)
POST   /api/conversation/fork          - Create a fork (optionally named) from the current conversation node
POST   /api/conversation/checkout      - Switch the current context to a different conversation node/branch (by ID)
POST   /api/conversation/clear         - Clear messages on the current node and switch to root
POST   /api/conversation/optimize      - Trigger context summarization for the current node (optional model/token args)
POST   /api/tool/execute               - Execute a specific tool with given parameters (Requires ToolExecuteRequest model)
WS     /ws/chat                        - WebSocket endpoint for streaming chat (`query`, `command`), receiving updates (`text_chunk`, `status`, `query_complete`, `error`, `state_update`).
```

*(Note: The actual request/response models like `ServerAddRequest`, `ToolExecuteRequest` are defined in the Python code and used by FastAPI for validation.)*

---

## ⚙️ Commands

### CLI Options

Run `mcpclient --help` or `mcpclient [COMMAND] --help` for details.

### Interactive Shell Commands (`mcpclient run --interactive`)

Type `/` followed by a command:

```text
/help         Show this help message
/exit, /quit  Exit the client
/config       Manage configuration (api-key, model, max-tokens, history-size, auto-discover, discovery-path, port-scan)
/servers      Manage MCP servers (list, add, remove, connect, disconnect, enable, disable, status)
/discover     Discover/manage LAN servers (list, connect <NAME>, refresh, auto <on|off>)
/tools        List available tools (optionally filter by server: /tools <server_name>)
/tool         Directly execute a tool: /tool <tool_name> '{"param": "value"}'
/resources    List available resources (optionally filter by server: /resources <server_name>)
/prompts      List available prompt templates (optionally filter by server: /prompts <server_name>)
/prompt       Apply a prompt template to the current conversation context: /prompt <prompt_name>
/model        View or change the current AI model: /model [<model_name>]
/fork         Create a new branch from the current conversation point: /fork [Optional Branch Name]
/branch       Manage branches (list, checkout <node_id_prefix>)
/export       Export current branch: /export [--id <node_id>] [--output <file.json>]
/import       Import conversation file: /import <file.json>
/history      View recent conversation history (optionally specify number: /history 10)
/cache        Manage tool cache (list, clear [--all|tool_name], clean, dependencies [tool_name])
/dashboard    Show the live Textual User Interface (TUI) dashboard (requires separate run)
/optimize     Summarize current conversation context: /optimize [--model <model>] [--tokens <num>]
/reload       Disconnect, reload capabilities, and reconnect to enabled servers
/clear        Clear messages in the current branch and optionally reset to root
```

---

## 🏗️ Architecture & Engineering Highlights

This client employs several techniques to provide a robust and feature-rich experience:

-   **Asynchronous Core:** Built entirely on Python's `asyncio` for efficient handling of network I/O (HTTP, WebSockets, SSE), subprocess communication (`stdio`), filesystem operations (`aiofiles`), and concurrent background tasks (monitoring, discovery, port scanning).
-   **Component-Based Design:** While packaged primarily as a single script for ease of use, it internally separates concerns into distinct classes:
    -   `MCPClient`: The main application class, orchestrating UI loops, command handling, and core logic.
    -   `ServerManager`: Handles server configuration, lifecycle (connecting, disconnecting, restarting), discovery mechanisms, capability aggregation (tools, resources, prompts), and manages server processes/sessions. Uses `AsyncExitStack` for reliable resource cleanup.
    -   **`RobustStdioSession` (Key Engineering Effort):** A custom implementation of the `mcp.ClientSession` tailored specifically for `stdio` servers. It includes logic to:
        -   Filter non-JSON-RPC output from the server's `stdout` to prevent protocol errors.
        -   Handle responses by directly resolving `asyncio.Future` objects associated with request IDs, offering a potentially more performant alternative to queue-based approaches.
        -   Manage background tasks for reading `stdout` and writing captured `stderr` to log files asynchronously.
        -   Gracefully handle process termination and cleanup.
    -   **STDIO Safety Mechanisms (Crucial):** A multi-layered defense against accidental `stdout` pollution, which is fatal for `stdio`-based protocols:
        -   `StdioProtectionWrapper`: Globally wraps `sys.stdout` to intercept writes, redirecting them to `stderr` if any `stdio` server is active.
        -   `safe_stdout()`: A context manager used during critical operations (like server connection) to temporarily redirect `stdout` to `stderr`.
        -   `get_safe_console()` / `safe_print()`: Utility functions ensuring that UI output (via `Rich`) uses the correct stream (`stdout` or `stderr`) based on active `stdio` servers.
        -   These measures allow the client UI and logging to function correctly even when communicating with sensitive `stdio` servers.
    -   `ConversationGraph`: Manages the non-linear, branching conversation structure using `ConversationNode` objects. Handles persistence to/from JSON.
    -   `ToolCache`: Implements caching logic using `diskcache` for persistence and an in-memory layer for speed. Includes TTL management and dependency invalidation.
    -   `ServerRegistry` / `ServerMonitor`: Handle mDNS/Zeroconf discovery/registration and background server health checks with recovery attempts.
-   **Dual Interface Implementation:**
    -   **Web Backend:** Uses `FastAPI` for a clean REST API structure, `uvicorn` for the ASGI server, and `websockets` for bidirectional real-time chat. A FastAPI `lifespan` context manager ensures the `MCPClient` is properly initialized on startup and cleaned up on shutdown. Dependency injection provides endpoint access to the client instance.
    -   **Web Frontend:** Leverages `Alpine.js` for lightweight reactivity and component logic directly in the HTML. `Tailwind CSS` and `DaisyUI` provide styling and pre-built components. `Marked.js`, `highlight.js`, and `DOMPurify` handle secure and attractive rendering of Markdown and code. `Tippy.js` provides tooltips.
    -   **CLI/TUI:** Uses `Typer` for defining the command-line interface and parsing arguments. `Rich` is heavily used for formatted console output, tables, progress bars, Markdown rendering, syntax highlighting, and the live TUI dashboard. Careful management (`_run_with_progress`, `_run_with_simple_progress` helpers) prevents issues with nested `Rich Live` displays during complex operations.
-   **Resilience & Error Handling:** Employs decorators (`@retry_with_circuit_breaker`, `@with_tool_error_handling`) for common patterns like retries and standardized error reporting during tool execution. Structured `try...except...finally` blocks are used throughout for robustness.
-   **Observability:** Integrates `OpenTelemetry` for structured metrics (request counters, latency histograms, tool executions) and distributed tracing (spans track operations like query processing, server connections, tool calls). This aids in performance analysis and debugging.
-   **Configuration:** Flexible configuration system using a `config.yaml` file, environment variables (especially for sensitive keys like `ANTHROPIC_API_KEY`), and interactive commands or Web UI settings that persist changes back to the YAML file.

---

## 🔄 Smart Cache Dependency Tracking

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

This client offers multiple ways to find and integrate MCP servers:

-   **Configured Paths:** Searches directories specified in `config.yaml` (under `discovery_paths`) for local `stdio` server scripts (e.g., `.py`, `.js`). Defaults include:
    -   `.mcpclient_config/servers` (in project root)
    -   `~/mcp-servers`
    -   `~/modelcontextprotocol/servers`
-   **Claude Desktop Config (`claude_desktop_config.json`):** If this file exists in the project root, the client automatically imports server definitions from it during setup.
    -   **Intelligent Command Adaptation:**
        -   Detects `wsl.exe ... <shell> -c "command ..."` patterns.
        -   Extracts the Linux `<shell>` (e.g., `bash`, `sh`) and the `"command ..."`.
        -   Remaps the configuration to execute the command directly using the identified Linux shell (e.g., `/bin/bash -c "command ..."`), bypassing `wsl.exe`.
        -   For other command types (e.g., `npx ...`), it uses `adapt_path_for_platform` to scan arguments for Windows-style paths (`C:\Users\...`) and converts them to their `/mnt/c/Users/...` equivalents, ensuring compatibility when running the client in WSL/Linux.
-   **Remote Registries:** Connects to MCP registry server URLs defined in `config.yaml` (`registry_urls`) to discover publicly available or shared servers (typically SSE).
-   **Local Network (mDNS/Zeroconf):**
    -   Uses the `zeroconf` library to listen for services advertised under `_mcp._tcp.local.` on the local network.
    -   Provides real-time notifications in the interactive CLI when new servers appear or disappear.
    -   The `/discover` command suite allows managing these discovered servers:
        -   `/discover list`: View details of currently visible LAN servers.
        -   `/discover connect <NAME>`: Add a discovered server to the configuration file and attempt to connect.
        -   `/discover refresh`: Manually trigger a re-scan of the network.
        -   `/discover auto [on|off]`: Toggle continuous background mDNS scanning (requires `enable_local_discovery: true` in config).
  -   **Local Port Scanning:**
      -   If enabled (`enable_port_scanning: true` in `config.yaml` or via `/config port-scan enable true`), actively scans a range of ports on specified local IP addresses during startup discovery.
      -   Attempts a basic MCP `initialize` handshake on each port to detect responsive MCP servers (primarily SSE).
      -   Useful for finding servers that don't use mDNS advertisement.
      -   The range, target IPs, concurrency, and timeout are configurable via `config.yaml` or the `/config port-scan ...` commands.
      -   Found servers are presented alongside other discovered servers for optional addition to the configuration.
---

## 📡 Telemetry + Debugging

-   **OpenTelemetry:** Generates traces and metrics for key operations. By default, data is collected but not exported. Set `use_console_exporter = True` near the top of `mcpclient.py` to enable noisy console output for debugging traces and metrics. For production, configure appropriate OTel exporters (e.g., Jaeger, Prometheus).
-   **Dashboards:**
    -   **CLI:** Run `mcpclient run --dashboard` for a live TUI monitoring view powered by `Rich`. Shows server status, health, connection state, basic tool usage stats, and client info. Refreshes periodically.
    -   **Web UI:** The "Servers" tab provides visual server status (connection, health via icons), and the header shows overall counts.
-   **Logging:** Uses Python's standard `logging` configured with `RichHandler` for pretty console output (to `stderr` to avoid `stdio` conflicts).
    -   Use the `--verbose` or `-v` flag to increase log level to DEBUG for detailed internal information.
    -   Verbose mode also enables detailed `stdio` session logging (`USE_VERBOSE_SESSION_LOGGING = True`) to see raw JSON-RPC messages (useful for debugging MCP servers).
-   **Error Tracebacks:** Set the environment variable `MCP_CLIENT_DEBUG=1` before running to make the CLI print full Python tracebacks for unexpected errors, aiding in debugging client-side issues.
-   **STDIO Server Logs:** The `stderr` output from `stdio` servers is captured asynchronously and written to log files in the configuration directory (e.g., `.mcpclient_config/<server_name>_stderr.log`), crucial for diagnosing server-side problems.

---

## 📦 Configuration

-   **Primary File:** Configuration is loaded from and saved to `.mcpclient_config/config.yaml` located in the project's root directory.
-   **Environment Variables:** `ANTHROPIC_API_KEY` is the primary way to provide the API key and overrides any key stored in the config file. `EDITOR` is used by `/config edit`. `MCP_CLIENT_DEBUG=1` enables tracebacks.
-   **Interactive CLI:** The `/config` command allows viewing and modifying settings like `api-key`, `model`, `max-tokens`, `discovery-path`, port scanning parameters, etc. Changes are saved back to `config.yaml`.
-   **Web UI Settings:** The "Settings" tab in the Web UI provides controls for common configuration options. Changes made here are sent via the API and saved to `config.yaml`.

**View Current Config:**

```bash
mcpclient config --show
# OR in interactive mode:
/config
```

**Edit Config File Manually:**

```bash
mcpclient config --edit
# (This will open .mcpclient_config/config.yaml in the editor specified by your $EDITOR environment variable)
```

---

## 🧪 Development Notes

-   **Core:** Python 3.13+, `asyncio`
-   **CLI:** `Typer`, `Rich`
-   **Web:** `FastAPI`, `Uvicorn`, `WebSockets`, `Alpine.js`, `Tailwind CSS`, `DaisyUI`
-   **MCP:** `mcp` SDK (`mcp>=1.0.0`)
-   **AI:** `anthropic` SDK (`anthropic>=0.15.0`)
-   **Observability:** `opentelemetry-sdk`, `opentelemetry-api`, `opentelemetry-instrumentation`
-   **Utilities:** `httpx`, `PyYAML`, `python-dotenv`, `psutil`, `aiofiles`, `diskcache`, `tiktoken`, `zeroconf`, `colorama`
-   **Linting/Formatting:** `ruff` is configured in `pyproject.toml`. Use `uv run lint` or `ruff check . && ruff format .`.
-   **Type Checking:** `mypy` is configured in `pyproject.toml`. Use `uv run typecheck` or `mypy mcpclient.py`.

The project is primarily structured within `mcpclient.py` for easier distribution and introspection, although internal class-based modularity is maintained. The Web UI is served from the self-contained `mcp_client_ui.html` file, utilizing CDN-hosted libraries for simplicity. Key complex logic, such as the robust `stdio` handling and asynchronous management, resides within dedicated classes like `RobustStdioSession` and `ServerManager`.

---

## 📝 License

MIT License. Refer to standard MIT terms.