# 🧠 Ultimate MCP Client

A comprehensive command-line client for the **Model Context Protocol (MCP)** that connects AI models like Claude to external tools, local and remote servers, and contextual data sources.

> Built by Jeffrey Emanuel — MIT Licensed — Python 3.13+

---

## 🔌 Key Features

- **Server Discovery**  
  - Auto-discovers local `stdio` servers (Python/JS scripts)
  - mDNS-based discovery of LAN servers with real-time notifications
  - Interactive commands for exploring and connecting to local network servers
  - Integration with public MCP registries
  - Re-use your existing json settings file from Claude Desktop, for easy import of existing server configurations

- **Streaming AI with Tools**  
  - Claude integration (via Anthropic SDK) with full support for tool use  
  - Intelligent routing of tool calls to the correct server
  - Direct tool execution with custom parameters

- **Advanced Conversation Management**  
  - Branching/forkable conversation graphs  
  - Persistent history across sessions  
  - Per-branch model tracking
  - Dynamic contextual prompt injection
  - Automatic context optimization through summarization
  - Import/export conversations to portable JSON files

- **Observability**  
  - OpenTelemetry metrics and spans  
  - Live TUI dashboard showing server health and tool usage

- **Smart Caching**  
  - Optional disk and in-memory caching for tool results  
  - Per-tool TTLs and runtime cache invalidation
  - Dependency graph for automatic invalidation of related caches

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

### Claude Desktop Integration

If you're already using Claude Desktop, you can easily import your existing MCP server configurations:

1. Copy your `claude_desktop_config.json` file to the project root directory
2. Start the client with any command (e.g., `mcpclient run --interactive`)
3. The client will automatically detect the file, import the server configurations, and display a summary

Example Claude Desktop config structure:
```json
{
  "mcpServers": {
    "llm_gateway": {
      "command": "wsl.exe",
      "args": [
        "bash",
        "-c",
        "cd /home/user/llm_gateway && python -m server run"
      ]
    },
    "filesystem": {
      "command": "npx",
      "args": [
        "-y",
        "@modelcontextprotocol/server-filesystem",
        "C:\\Users\\user\\Documents"
      ]
    }
  }
}
```

### Run Interactive Mode

```bash
mcpclient run --interactive
```

### Run a One-Off Query

```bash
mcpclient run --query "What's the weather in New York?"
```

### Show the Dashboard

```bash
mcpclient run --dashboard
```

### Import and Export Conversations

```bash
# Export current conversation
mcpclient export --output my_conversation.json

# Export specific conversation by ID
mcpclient export --id 12345678-abcd-1234-5678-abcdef123456 --output my_conversation.json

# Import a conversation
mcpclient import-conv my_conversation.json
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
/discover     Discover and connect to MCP servers on local network
/tools        List or inspect tools  
/tool         Directly execute a tool with custom parameters
/resources    List available resources  
/prompts      List available prompt templates  
/prompt       Apply a prompt template to the current conversation
/model        Change Claude model  
/fork         Create a conversation branch  
/branch       List or switch between branches  
/export       Export conversation to a file
/import       Import conversation from a file
/cache        Manage tool result cache and dependencies  (list, clear, clean, dependencies)
/dashboard    Open real-time monitoring dashboard  
/optimize     Optimize conversation context through summarization
/clear        Clear the conversation context
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

- Searches in:
  - `~/.config/mcpclient/servers`
  - `~/mcp-servers`
  - `~/modelcontextprotocol/servers`
- Auto-adds servers found via:
  - Registry (e.g., `https://registry.modelcontextprotocol.io`)
  - Zeroconf (`_mcp._tcp.local`) on LAN
- Continuous local network discovery:
  - Real-time notifications of new MCP servers on your network
  - `/discover list` - View all discovered servers with their details
  - `/discover connect NAME` - Connect to a specific discovered server
  - `/discover refresh` - Force a refresh of the discovery process
  - `/discover auto [on|off]` - Toggle automatic discovery

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
- OpenTelemetry for spans, counters, histograms

---

## 📝 License

MIT License.

