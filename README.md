# 🧠 Ultimate MCP Client

A comprehensive command-line client for the **Model Context Protocol (MCP)** that connects AI models like Claude to external tools, local and remote servers, and contextual data sources.

> Built by Jeffrey Emanuel — MIT Licensed — Python 3.13+

---

## 🔌 Key Features

- **Server Discovery**  
  - Auto-discovers local `stdio` servers (Python/JS scripts)
  - mDNS-based discovery of LAN servers
  - Integration with public MCP registries

- **Streaming AI with Tools**  
  - Claude integration (via Anthropic SDK) with full support for tool use  
  - Intelligent routing of tool calls to the correct server

- **Advanced Conversation Management**  
  - Branching/forkable conversation graphs  
  - Persistent history across sessions  
  - Per-branch model tracking

- **Observability**  
  - OpenTelemetry metrics and spans  
  - Live TUI dashboard showing server health and tool usage

- **Smart Caching**  
  - Optional disk and in-memory caching for tool results  
  - Per-tool TTLs and runtime cache invalidation

- **Rich CLI UX**  
  - Interactive shell with `/commands`, autocompletion, and Markdown rendering  
  - Typer-powered CLI for batch mode, dashboards, and introspection

---

## 🚀 Quickstart

### Install Dependencies

> Requires Python 3.13+

First, install uv if you don't have it already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone, setup a virtual environment, and install packages using uv:

```bash
git clone https://github.com/Dicklesworthstone/ultimate_mcp_client
cd ultimate_mcp_client
uv venv --python 3.13 && source .venv/bin/activate
uv sync --all-extras
```

Or, you can run it using uv's self-contained script functionality:

```bash
uv run mcp_client.py
```


### Run Interactive Mode

```bash
mcpclient run --interactive
```

### Run a One-Off Query

```bash
mcpclient run --query "What’s the weather in New York?"
```

### Show the Dashboard

```bash
mcpclient run --dashboard
```

---

## ⚙️ Commands

### CLI Options

```bash
mcpclient run --help
```

### Interactive Shell Commands

```text
/help         Show available commands  
/servers      Manage MCP servers (list, connect, add, etc.)  
/tools        List or inspect tools  
/resources    List available resources  
/prompts      List available prompt templates  
/model        Change Claude model  
/fork         Create a conversation branch  
/branch       List or switch between branches  
/cache        Manage tool result cache  
/dashboard    Open real-time monitoring dashboard  
```

---

## 🏗️ Architecture Overview

```
Claude ↔ MCPClient ↔ Tool Registry + Conversation Graph
               ↘
           ServerManager ↔ [MCP Servers]
                  ↘
               ToolCache
```

---

## 🔍 Tool & Server Discovery

- Searches in:
  - `~/.config/mcpclient/servers`
  - `~/mcp-servers`
  - `~/modelcontextprotocol/servers`
- Auto-adds servers found via:
  - Registry (e.g., `https://registry.modelcontextprotocol.io`)
  - Zeroconf (`_mcp._tcp.local`) on LAN

---

## 📡 Telemetry + Debugging

- [x] OpenTelemetry-compatible metrics and tracing
- [x] Export spans and counters to console
- [x] CLI dashboard (TUI) with live health stats

Set `MCP_CLIENT_DEBUG=1` to enable tracebacks on errors.

---

## 📦 Configuration

Run:

```bash
mcpclient config --show
```

To set API key:

```bash
/config api-key sk-ant-xxx
```

To change model:

```bash
/config model claude-3-7-sonnet-20250219
```

All settings are stored in:

```bash
~/.config/mcpclient/config.yaml
```

---

## 🧪 Development Notes

- Project is monolithic by design for ease of deployment and introspection.
- Linting: `ruff`
- Type checks: `mypy`
- Optional: OpenTelemetry for spans, counters, histograms

---

## 📝 License

MIT License. See `LICENSE` file.

---

## 👤 Author

Jeffrey Emanuel  
<jeffrey.emanuel@gmail.com>
