#!/usr/bin/env python3

# /// script
# dependencies = [
#     "anthropic>=0.21.3",
#     "openai>=1.10.0",
#     "mcp>=1.0.0",
#     "typer>=0.9.0",
#     "rich>=13.6.0",
#     "httpx>=0.25.0",
#     "pyyaml>=6.0.1",
#     "python-dotenv>=1.0.0",
#     "colorama>=0.4.6",
#     "psutil>=5.9.5",
#     "zeroconf>=0.39.0",
#     "diskcache>=5.6.1",
#     "typing-extensions>=4.8.0",
#     "opentelemetry-api>=1.19.0",
#     "opentelemetry-sdk>=1.19.0",
#     "opentelemetry-instrumentation>=0.41b0",
#     "asyncio>=3.4.3",
#     "aiofiles>=23.2.0",
#     "tiktoken>=0.5.1", # Keep for token counting estimation
#     "fastapi>=0.104.0",
#     "uvicorn[standard]>=0.24.0",
#     "websockets>=11.0",
#     "python-multipart>=0.0.6"
# ]
# ///

"""
Ultimate MCP Client - Multi-Provider Edition
===========================================

A comprehensive client for the Model Context Protocol (MCP) that connects AI models
from various providers (Anthropic, OpenAI, Gemini, Grok, DeepSeek, Mistral, Groq, Cerebras)
with external tools, servers, and data sources.

Key Features:
------------
- Multi-Provider Support: Seamlessly switch between models from different AI providers.
- Web UI: Modern reactive interface with DaisyUI/Tailwind styling
- API Server: Full REST API for programmatic access with FastAPI
- WebSocket Support: Real-time streaming AI responses in both CLI and Web UI
- Server Management: Discover, connect to, and monitor MCP servers
- Tool Integration: Execute tools from multiple servers with intelligent routing
- Streaming: Real-time streaming responses with tool execution
- Caching: Smart caching of tool results with configurable TTLs
- Conversation Branches: Create and manage conversation forks and branches
- Conversation Import/Export: Save and share conversations with easy portable JSON format
- Health Dashboard: Real-time monitoring of servers and tool performance in CLI and Web UI
- Observability: Comprehensive metrics and tracing
- Registry Integration: Connect to remote registries to discover servers
- Local Discovery (mDNS): Discover MCP servers on your local network via mDNS/Zeroconf.
- Local Port Scanning: Actively scan a configurable range of local ports to find MCP servers.

Usage:
------
# Interactive CLI mode
python mcp_client_multi.py run --interactive

# Launch Web UI
python mcp_client_multi.py run --webui

# Single query (specify provider via model name if needed)
python mcp_client_multi.py run --query "Explain quantum entanglement" --model gemini-2.0-flash-latest
python mcp_client_multi.py run --query "Write a python function" --model gpt-4.1

# Show dashboard
python mcp_client_multi.py run --dashboard

# Server management
python mcp_client_multi.py servers --list

# Conversation import/export
python mcp_client_multi.py export --id [CONVERSATION_ID] --output [FILE_PATH]
python mcp_client_multi.py import-conv [FILE_PATH]

# Configuration
python mcp_client_multi.py config --show
# Example: Set OpenAI API Key
python mcp_client_multi.py config api-key openai YOUR_OPENAI_KEY
# Example: Set default model to Gemini Flash
python mcp_client_multi.py config model gemini-2.0-flash-latest

Command Reference:
-----------------
Interactive mode commands:
- /help - Show available commands
- /servers - Manage MCP servers (list, add, connect, etc.)
- /tools - List and inspect available tools
- /tool - Directly execute a tool with custom parameters
- /resources - List available resources
- /prompts - List available prompts
- /prompt - Apply a prompt template to the current conversation
- /model [MODEL_NAME] - Change AI model (e.g., /model gpt-4o, /model claude-3-sonnet-20240229)
- /fork - Create a conversation branch
- /branch - Manage conversation branches
- /export - Export conversation to a file
- /import - Import conversation from a file
- /cache - Manage tool caching
- /dashboard - Open health monitoring dashboard
- /monitor - Control server monitoring
- /registry - Manage server registry connections
- /discover - Discover and connect to MCP servers on local network (via mDNS and Port Scanning)
- /optimize - Optimize conversation context through summarization
- /clear - Clear the conversation context
- /config - Manage client configuration (API keys, models, discovery methods, etc.)
    - /config api-key [PROVIDER] [KEY] - Set API key (e.g., /config api-key openai sk-...)
    - /config base-url [PROVIDER] [URL] - Set base URL (e.g., /config base-url deepseek http://host:port)
    - /config model [NAME] - Set default AI model
    - /config max-tokens [NUMBER] - Set default max tokens for generation
    - [...] (other config options)

Web UI Features:
--------------
- Server Management: Add, remove, connect, and manage MCP servers
- Conversation Interface: Chat with models with streaming responses
- Tool Execution: View and interact with tool calls and results in real-time
- Branch Management: Visual conversation tree with fork/switch capabilities
- Settings Panel: Configure API keys, models, and parameters
- Theme Customization: Multiple built-in themes with light/dark mode

API Endpoints:
------------
- GET /api/status - Get client status
- GET/PUT /api/config - Get or update configuration
- GET /api/models - List available models by provider
- GET/POST/DELETE /api/servers/... - Manage servers
- GET /api/tools - List available tools
- POST /api/tool/execute - Execute a tool directly
- WS /ws/chat - WebSocket for chat communication

Author: Jeffrey Emanuel (Original), Adapted by AI
License: MIT
Version: 2.0.0 (Multi-Provider)
"""

import asyncio
import atexit
import copy
import dataclasses
import functools
import hashlib
import io
import ipaddress
import json
import logging
import os
import platform
import random
import re
import readline
import shlex
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager, redirect_stdout, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Callable,
    Coroutine,
    Dict,
    List,
    NotRequired,
    Optional,
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
)
from urllib.parse import urlparse

# Other imports
import aiofiles

# === Provider SDK Imports ===
import anthropic
import anyio
import colorama
import diskcache
import httpx
import openai
import psutil
import tiktoken
import typer
import uvicorn
import yaml
from anthropic import AsyncAnthropic, AsyncMessageStream
from anthropic.types import (
    ContentBlockDeltaEvent,
    MessageStreamEvent,
)
from decouple import Config as DecoupleConfig
from decouple import Csv, RepositoryEnv
from dotenv import dotenv_values, find_dotenv, load_dotenv, set_key
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.shared.exceptions import McpError
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    Tool,
)
from mcp.types import (
    InitializeResult as MCPInitializeResult,  # Alias to avoid confusion with provider results
)
from mcp.types import Prompt as McpPromptType
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIError as OpenAIAPIError
from openai import (
    AsyncOpenAI,  # For OpenAI, Grok, DeepSeek, Mistral, Groq, Cerebras, Gemini
    AsyncStream,
)
from openai import AuthenticationError as OpenAIAuthenticationError
from openai.types.chat import ChatCompletionChunk
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pydantic import AnyUrl, BaseModel, Field, ValidationError
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Confirm, Prompt
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.tree import Tree
from starlette.responses import FileResponse
from typing_extensions import Annotated, Literal, TypeAlias
from zeroconf import EventLoopBlocked, NonUniqueNameException, ServiceBrowser, ServiceInfo, Zeroconf

decouple_config = DecoupleConfig(RepositoryEnv('.env'))

# =============================================================================
# Constants Integration (Copied from user input)
# =============================================================================

class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    GEMINI = "gemini"
    OPENROUTER = "openrouter"
    GROK = "grok"
    MISTRAL = "mistral" 
    GROQ = "groq"
    CEREBRAS = "cerebras"

class TaskType(str, Enum): # Keep as is
    COMPLETION = "completion"
    CHAT = "chat"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    GENERATION = "generation"
    ANALYSIS = "analysis"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    QA = "qa"
    DATABASE = "database"
    QUERY = "query"
    BROWSER = "browser"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    DOCUMENT_PROCESSING = "document_processing"
    DOCUMENT = "document"

class LogLevel(str, Enum): # Keep as is
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Cost estimates (Copied and slightly adjusted for consistency)
COST_PER_MILLION_TOKENS: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    
    # Claude models
    "claude-3-7-sonnet-20250219": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},

    # DeepSeek models
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    
    # Gemini models
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.35, "output": 1.05},
    "gemini-2.0-flash-thinking-exp-01-21": {"input": 0.0, "output": 0.0},
    "gemini-2.5-pro-exp-03-25": {"input": 1.25, "output": 10.0},

    # OpenRouter models
    "mistralai/mistral-nemo": {"input": 0.035, "output": 0.08},
    
    # Grok models (based on the provided documentation)
    "grok-3-latest": {"input": 3.0, "output": 15.0},
    "grok-3-fast-latest": {"input": 5.0, "output": 25.0},
    "grok-3-mini-latest": {"input": 0.30, "output": 0.50},
    "grok-3-mini-fast-latest": {"input": 0.60, "output": 4.0},

    # Mistral models
    "mistral-large-latest": {"input": 0.035, "output": 0.08},

    # Groq models
    "llama-3.3-70b-versatile": {"input": 0.0001, "output": 0.0001},

    # Cerebras models
    "llama-4-scout-17b-16e-instruct": {"input": 0.0001, "output": 0.0001},
}

# Default models by provider (Using Provider Enum values as keys)
DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4.1-mini",
    Provider.ANTHROPIC: "claude-3-5-haiku-20241022",
    Provider.DEEPSEEK: "deepseek-chat",
    Provider.GEMINI: "gemini-2.5-pro-exp-03-25",
    Provider.OPENROUTER: "mistralai/mistral-nemo",
    Provider.GROK: "grok-3-latest",
    Provider.MISTRAL: "mistral-large-latest",
    Provider.GROQ: "llama-3.3-70b-versatile",
    Provider.CEREBRAS: "llama-4-scout-17b-16e-instruct",
}

# Emoji mapping
EMOJI_MAP = {
    "start": "🚀", "success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️",
    "debug": "🔍", "critical": "🔥", "server": "🖥️", "cache": "💾", "provider": "🔌",
    "request": "📤", "response": "📥", "processing": "⚙️", "model": "🧠", "config": "🔧",
    "token": "🔢", "cost": "💰", "time": "⏱️", "tool": "🛠️", "cancel": "🛑",
    "database": "🗄️", "browser": "🌐", "completion": "✍️", "chat": "💬",
    "summarization": "📝", "extraction": "💡", "generation": "🎨", "analysis": "📊",
    "classification": "🏷️", "query": "❓", "download": "⬇️", "upload": "⬆️",
    "document_processing": "📄", "document": "📄", "translation": "🔄", "qa": "❓",
    "history": "📜", "search": "🔎", "port": "🔌", "package": "📦", "resource": "📚",
    "prompt": "💬", "trident_emblem": "🔱", "desktop_computer": "🖥️", "gear": "⚙️",
    "scroll": "📜", "magnifying_glass_tilted_right": "🔎", "electric_plug": "🔌",
    "party_popper": "🎉", "collision": "💥", "robot": "🤖", "water_wave": "🌊",
    "green_circle": "🟢", "red_circle": "🔴", "white_check_mark": "✅", "cross_mark": "❌",
    "question_mark": "❓", "cached": "📦",
    Provider.OPENAI.value: "🟢", Provider.ANTHROPIC.value: "🟣", Provider.DEEPSEEK.value: "🐋",
    Provider.GEMINI.value: "♊", Provider.GROK.value: "⚡",
    Provider.MISTRAL.value: "🌫️", Provider.GROQ.value: "🚅",
    Provider.CEREBRAS.value: "🧠", Provider.OPENROUTER.value: "🔄",
    "status_healthy": "✅", "status_degraded": "⚠️", "status_error": "❌", "status_unknown": "❓",
}

# Add Emojis for status.healthy etc if needed, or map directly in Rich styles
# Example mapping for status consistency:
EMOJI_MAP["status_healthy"] = EMOJI_MAP["white_check_mark"]
EMOJI_MAP["status_degraded"] = EMOJI_MAP["warning"]
EMOJI_MAP["status_error"] = EMOJI_MAP["cross_mark"]
EMOJI_MAP["status_unknown"] = EMOJI_MAP["question_mark"]

# =============================================================================
# Model -> Provider Mapping
# =============================================================================
# Maps known model identifiers (or prefixes) to their provider's enum value string.
MODEL_PROVIDER_MAP: Dict[str, str] = {}

# Logic to infer provider from model name (simplified version for static generation)
def _infer_provider(model_name: str) -> Optional[str]:
    # ... (keep infer provider function) ...
    lname = model_name.lower()
    if lname.startswith("openai/"): return Provider.OPENAI.value
    if lname.startswith("anthropic/"): return Provider.ANTHROPIC.value
    if lname.startswith("google/") or lname.startswith("gemini/"): return Provider.GEMINI.value
    if lname.startswith("grok/"): return Provider.GROK.value
    if lname.startswith("deepseek/"): return Provider.DEEPSEEK.value
    if lname.startswith("mistralai/"): return Provider.MISTRAL.value
    if lname.startswith("groq/"): return Provider.GROQ.value
    if lname.startswith("cerebras/"): return Provider.CEREBRAS.value
    if lname.startswith("openrouter/"): return Provider.OPENROUTER.value # Add openrouter prefix
    if lname.startswith("gpt-") or lname.startswith("o1-") or lname.startswith("o3-"): return Provider.OPENAI.value
    if lname.startswith("claude-"): return Provider.ANTHROPIC.value
    if lname.startswith("gemini-"): return Provider.GEMINI.value
    if lname.startswith("grok-"): return Provider.GROK.value
    if lname.startswith("deepseek-"): return Provider.DEEPSEEK.value
    if lname.startswith("mistral-"): return Provider.MISTRAL.value
    if lname.startswith("groq-"): return Provider.GROQ.value
    if lname.startswith("cerebras-"): return Provider.CEREBRAS.value
    return None # Cannot infer provider for this model

# Populate the map
for model_key in COST_PER_MILLION_TOKENS.keys():
    inferred_provider = _infer_provider(model_key)
    if inferred_provider:
        MODEL_PROVIDER_MAP[model_key] = inferred_provider
    else:
        pass # Skip adding models we cannot map

# =============================================================================
# Type Aliases & Canonical Internal Format
# =============================================================================
# Define the internal canonical message format (based on Anthropic's structure)
# We use this format internally and convert to provider-specific formats as needed.

class TextContentBlock(TypedDict):
    type: Literal["text"]
    text: str

class ToolUseContentBlock(TypedDict):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]

class ToolResultContentBlock(TypedDict):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict]] # Content can be simple string or richer structure
    # Optional fields for provider-specific needs or errors
    is_error: NotRequired[bool]
    # Used internally to differentiate from regular user message after tool execution
    _is_tool_result: NotRequired[bool]

# Define the content type alias
InternalContent: TypeAlias = Union[str, List[Union[TextContentBlock, ToolUseContentBlock, ToolResultContentBlock]]]

# Define the main message structure
class InternalMessage(TypedDict):
    role: Literal["user", "assistant", "system"] # Keep roles simple internally
    content: InternalContent
    # Optional: Add fields used during processing, like tool_use_id for linking
    # tool_use_id: NotRequired[str] # Example if needed for processing state

InternalMessageList = List[InternalMessage] # Represents a list of messages
ContentDict = Dict[str, Any]
PartDict = Dict[str, Any]
FunctionResponseDict = Dict[str, Any]
FunctionCallDict = Dict[str, Any]
# =============================================================================


# Global flag for verbose logging (can be set by --verbose)
USE_VERBOSE_SESSION_LOGGING = False

# --- Set up Typer app ---
app = typer.Typer(
    help="🔌 Ultimate MCP Client - Multi-Provider Edition",
    context_settings={"help_option_names": ["--help", "-h"]}
)

@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """Show help if no command is provided."""
    if ctx.invoked_subcommand is None:
        print(ctx.get_help())
        raise typer.Exit()


class StdioProtectionWrapper:
    """Wrapper that prevents accidental writes to stdout when stdio servers are active."""
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.active_stdio_servers = False
        self._buffer = []

    def update_stdio_status(self):
        """Check if we have any active stdio servers"""
        try:
            if hasattr(app, "mcp_client") and app.mcp_client and hasattr(app.mcp_client, "server_manager"):
                for name, server in app.mcp_client.server_manager.config.servers.items():
                    if server.type == ServerType.STDIO and name in app.mcp_client.server_manager.active_sessions:
                        self.active_stdio_servers = True
                        return
                self.active_stdio_servers = False
        except (NameError, AttributeError):
            self.active_stdio_servers = False

    def write(self, text):
        """Intercept writes to stdout"""
        self.update_stdio_status()
        if self.active_stdio_servers:
            sys.stderr.write(text)
            if text.strip() and text != "\n":
                self._buffer.append(text)
                if len(self._buffer) > 100: self._buffer.pop(0)
        else:
            self.original_stdout.write(text)

    def flush(self):
        if not self.active_stdio_servers: self.original_stdout.flush()
        else: sys.stderr.flush()

    def isatty(self): return self.original_stdout.isatty()
    def fileno(self): return self.original_stdout.fileno()
    def readable(self): return self.original_stdout.readable()
    def writable(self): return self.original_stdout.writable()

# Apply the protection wrapper
sys.stdout = StdioProtectionWrapper(sys.stdout)

# --- Rich Theme and Consoles ---
custom_theme = Theme({
    "info": "cyan", "success": "green bold", "warning": "yellow bold", "error": "red bold",
    "server": "blue", "tool": "magenta", "resource": "cyan", "prompt": "yellow",
    "model": "bright_blue", "dashboard.title": "white on blue", "dashboard.border": "blue",
    "status.healthy": "green", "status.degraded": "yellow", "status.error": "red",
    "metric.good": "green", "metric.warn": "yellow", "metric.bad": "red",
})
console = Console(theme=custom_theme)
stderr_console = Console(theme=custom_theme, stderr=True, highlight=False)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, console=stderr_console)]
)
log = logging.getLogger("mcpclient_multi") # Use a unique logger name

# --- Signal Handling (force_exit_handler, atexit_handler, sigint_handler) ---
def force_exit_handler(is_force=False):
    print("\nForcing exit and cleaning up resources...")
    if is_force:
        print("Emergency shutdown initiated!")
        if 'app' in globals() and hasattr(app, 'mcp_client'):
            if hasattr(app.mcp_client, 'server_manager'):
                for name, process in app.mcp_client.server_manager.processes.items():
                    try:
                        if process.returncode is None:
                            print(f"Force killing process {name} (PID {process.pid})")
                            process.kill()
                    except Exception: pass
        os._exit(1)
    sys.exit(1)

def atexit_handler():
    print("\nShutting down and cleaning resources...")
    # Ensure cleanup happens via MCPClient.close() called in main_async finally block

# Signal handler for SIGINT (Ctrl+C)
def sigint_handler(signum, frame):
    print("\nCtrl+C detected.")
    client_instance = getattr(app, 'mcp_client', None)
    active_query_task = getattr(client_instance, 'current_query_task', None) if client_instance else None

    if active_query_task and not active_query_task.done():
        print("Attempting to abort current request... (Press Ctrl+C again to force exit)")
        try: active_query_task.cancel()
        except Exception as e: print(f"Error trying to cancel task: {e}")
        return # Don't increment counter or exit yet

    sigint_handler.counter += 1
    if sigint_handler.counter >= 2:
        print("Multiple interrupts detected. Forcing immediate exit!")
        force_exit_handler(is_force=True)

    print("Shutting down...")
    try: sys.exit(1) # Triggers atexit
    except SystemExit: pass
    except Exception as e:
        print(f"Error during clean shutdown attempt: {e}. Forcing exit!")
        force_exit_handler(is_force=True)

sigint_handler.counter = 0
signal.signal(signal.SIGINT, sigint_handler)
atexit.register(atexit_handler)


# --- Pydantic Models (ServerAddRequest, ConfigUpdateRequest, etc. - Update ConfigUpdateRequest) ---
class WebSocketMessage(BaseModel):
    type: str
    payload: Any = None

class ServerType(Enum):
    STDIO = "stdio"
    SSE = "sse"

class ServerAddRequest(BaseModel):
    name: str
    type: ServerType
    path: str
    argsString: Optional[str] = ""

# Model for the GET /api/config response (excluding sensitive data)
class ConfigGetResponse(BaseModel):
    default_model: str = Field(..., alias="defaultModel")
    default_max_tokens: int = Field(..., alias="defaultMaxTokens")
    history_size: int = Field(..., alias="historySize")
    auto_discover: bool = Field(..., alias="autoDiscover")
    discovery_paths: List[str] = Field(..., alias="discoveryPaths")
    enable_streaming: bool = Field(..., alias="enableStreaming")
    enable_caching: bool = Field(..., alias="enableCaching")
    enable_metrics: bool = Field(..., alias="enableMetrics")
    enable_registry: bool = Field(..., alias="enableRegistry")
    enable_local_discovery: bool = Field(..., alias="enableLocalDiscovery")
    temperature: float
    cache_ttl_mapping: Dict[str, int] = Field(..., alias="cacheTtlMapping")
    conversation_graphs_dir: str = Field(..., alias="conversationGraphsDir")
    registry_urls: List[str] = Field(..., alias="registryUrls")
    dashboard_refresh_rate: float = Field(..., alias="dashboardRefreshRate")
    summarization_model: str = Field(..., alias="summarizationModel")
    use_auto_summarization: bool = Field(..., alias="useAutoSummarization")
    auto_summarize_threshold: int = Field(..., alias="autoSummarizeThreshold")
    max_summarized_tokens: int = Field(..., alias="maxSummarizedTokens")
    enable_port_scanning: bool = Field(..., alias="enablePortScanning")
    port_scan_range_start: int = Field(..., alias="portScanRangeStart")
    port_scan_range_end: int = Field(..., alias="portScanRangeEnd")
    port_scan_concurrency: int = Field(..., alias="portScanConcurrency")
    port_scan_timeout: float = Field(..., alias="portScanTimeout")
    port_scan_targets: List[str] = Field(..., alias="portScanTargets")

    # Provider Base URLs (safe to return)
    openai_base_url: Optional[str] = Field(None, alias="openaiBaseUrl")
    gemini_base_url: Optional[str] = Field(None, alias="geminiBaseUrl")
    grok_base_url: Optional[str] = Field(None, alias="grokBaseUrl")
    deepseek_base_url: Optional[str] = Field(None, alias="deepseekBaseUrl")
    mistral_base_url: Optional[str] = Field(None, alias="mistralBaseUrl")
    groq_base_url: Optional[str] = Field(None, alias="groqBaseUrl")
    cerebras_base_url: Optional[str] = Field(None, alias="cerebrasBaseUrl")
    openrouter_base_url: Optional[str] = Field(None, alias="openrouterBaseUrl")
    # Anthropic base URL is usually not configurable via SDK, so omit unless needed

    class Config:
        populate_by_name = True # Allow using aliases in responses

class ConfigUpdateRequest(BaseModel):
    # --- Provider API Keys (Optional) ---
    # Note: Setting these via API only affects the current session.
    # For persistence, edit .env or environment variables.
    anthropic_api_key: Optional[str] = Field(None, alias="anthropicApiKey")
    openai_api_key: Optional[str] = Field(None, alias="openaiApiKey")
    gemini_api_key: Optional[str] = Field(None, alias="geminiApiKey")
    grok_api_key: Optional[str] = Field(None, alias="grokApiKey")
    deepseek_api_key: Optional[str] = Field(None, alias="deepseekApiKey")
    mistral_api_key: Optional[str] = Field(None, alias="mistralApiKey")
    groq_api_key: Optional[str] = Field(None, alias="groqApiKey")
    cerebras_api_key: Optional[str] = Field(None, alias="cerebrasApiKey")
    openrouter_api_key: Optional[str] = Field(None, alias="openrouterApiKey")

    # --- Provider Base URLs (Optional) ---
    # Note: Setting these via API only affects the current session.
    openai_base_url: Optional[str] = Field(None, alias="openaiBaseUrl")
    gemini_base_url: Optional[str] = Field(None, alias="geminiBaseUrl")
    grok_base_url: Optional[str] = Field(None, alias="grokBaseUrl")
    deepseek_base_url: Optional[str] = Field(None, alias="deepseekBaseUrl")
    mistral_base_url: Optional[str] = Field(None, alias="mistralBaseUrl")
    groq_base_url: Optional[str] = Field(None, alias="groqBaseUrl")
    cerebras_base_url: Optional[str] = Field(None, alias="cerebrasBaseUrl")
    openrouter_base_url: Optional[str] = Field(None, alias="openrouterBaseUrl")

    # --- General Settings (Optional) ---
    # Note: Setting these via API only affects the current session.
    default_model: Optional[str] = Field(None, alias="defaultModel")
    default_max_tokens: Optional[int] = Field(None, alias="defaultMaxTokens")
    history_size: Optional[int] = Field(None, alias="historySize")
    auto_discover: Optional[bool] = Field(None, alias="autoDiscover")
    discovery_paths: Optional[List[str]] = Field(None, alias="discoveryPaths")
    enable_streaming: Optional[bool] = Field(None, alias="enableStreaming")
    enable_caching: Optional[bool] = Field(None, alias="enableCaching")
    enable_metrics: Optional[bool] = Field(None, alias="enableMetrics")
    enable_registry: Optional[bool] = Field(None, alias="enableRegistry")
    enable_local_discovery: Optional[bool] = Field(None, alias="enableLocalDiscovery")
    temperature: Optional[float] = None
    registry_urls: Optional[List[str]] = Field(None, alias="registryUrls")
    dashboard_refresh_rate: Optional[float] = Field(None, alias="dashboardRefreshRate")
    summarization_model: Optional[str] = Field(None, alias="summarizationModel")
    use_auto_summarization: Optional[bool] = Field(None, alias="useAutoSummarization")
    auto_summarize_threshold: Optional[int] = Field(None, alias="autoSummarizeThreshold")
    max_summarized_tokens: Optional[int] = Field(None, alias="maxSummarizedTokens")
    enable_port_scanning: Optional[bool] = Field(None, alias="enablePortScanning")
    port_scan_range_start: Optional[int] = Field(None, alias="portScanRangeStart")
    port_scan_range_end: Optional[int] = Field(None, alias="portScanRangeEnd")
    port_scan_concurrency: Optional[int] = Field(None, alias="portScanConcurrency")
    port_scan_timeout: Optional[float] = Field(None, alias="portScanTimeout")
    port_scan_targets: Optional[List[str]] = Field(None, alias="portScanTargets")

    # --- Complex Settings (Optional - Updates WILL BE SAVED to YAML) ---
    cache_ttl_mapping: Optional[Dict[str, int]] = Field(None, alias="cacheTtlMapping")
    # Note: 'servers' are managed via dedicated /api/servers endpoints, not this general config update.

    class Config:
        populate_by_name = True # Allow using aliases in requests
        extra = 'ignore' # Ignore extra fields in the request

class ToolExecuteRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any]

class GraphNodeData(BaseModel):
    id: str
    name: str
    parent_id: Optional[str] = Field(None, alias="parentId") # Use alias for JS
    model: Optional[str] = None # Model used for this node/branch
    created_at: str = Field(..., alias="createdAt")
    modified_at: str = Field(..., alias="modifiedAt")
    message_count: int = Field(..., alias="messageCount")

    class Config:
        populate_by_name = True # Allow population by alias

class NodeRenameRequest(BaseModel):
    new_name: str = Field(..., min_length=1)
    
class ForkRequest(BaseModel): name: Optional[str] = None
class CheckoutRequest(BaseModel): node_id: str
class OptimizeRequest(BaseModel):
    model: Optional[str] = None
    target_tokens: Optional[int] = None
class ApplyPromptRequest(BaseModel): prompt_name: str

class DiscoveredServer(BaseModel):
    name: str
    type: str
    path_or_url: str
    source: str
    description: Optional[str] = None
    version: Optional[str] = None
    categories: List[str] = []
    is_configured: bool = False

class ChatHistoryResponse(BaseModel):
    query: str
    response: str
    model: str
    timestamp: str
    server_names: List[str] = Field(default_factory=list)
    tools_used: List[str] = Field(default_factory=list)
    conversation_id: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    cached: bool = False
    streamed: bool = False

@dataclass
class CacheEntry:
    result: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    tool_name: str = ""
    parameters_hash: str = ""

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        # Ensure comparison is timezone-aware if necessary, assuming naive for now
        return datetime.now() > self.expires_at

class CacheEntryDetail(BaseModel):
    key: str
    tool_name: str
    created_at: datetime
    expires_at: Optional[datetime] = None
        
class CacheDependencyInfo(BaseModel):
    dependencies: Dict[str, List[str]]

class ServerDetail(BaseModel):
    name: str
    type: ServerType
    path: str
    args: List[str]
    enabled: bool
    auto_start: bool
    description: str
    trusted: bool
    categories: List[str]
    version: Optional[str] = None
    rating: float
    retry_count: int
    timeout: float
    registry_url: Optional[str] = None
    capabilities: Dict[str, bool]
    is_connected: bool
    metrics: Dict[str, Any]
    process_info: Optional[Dict[str, Any]] = None

class DashboardData(BaseModel):
    client_info: Dict[str, Any]
    servers: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]

# --- FastAPI app placeholder ---
web_app: Optional[FastAPI] = None

# --- Path Adaptation Helper (adapt_path_for_platform) ---
# Helper function to adapt paths for different platforms, used for processing Claude Desktop JSON config file
def adapt_path_for_platform(command: str, args: List[str]) -> Tuple[str, List[str]]:
    """
    ALWAYS assumes running on Linux/WSL.
    Converts Windows-style paths (e.g., 'C:\\Users\\...') found in the command
    or arguments to their corresponding /mnt/ drive equivalents
    (e.g., '/mnt/c/Users/...').
    """

    # --- Added Debug Logging ---
    log.debug(f"adapt_path_for_platform: Initial input - command='{command}', args={args}")
    # --- End Added Debug Logging ---

    def convert_windows_path_to_linux(path_str: str) -> str:
        """
        Directly converts 'DRIVE:\\path' or 'DRIVE:/path' to '/mnt/drive/path'.
        Handles drive letters C-Z, case-insensitive.
        Replaces backslashes (represented as '\\' in Python strings) with forward slashes.
        """
        log.debug(f"convert_windows_path_to_linux: Checking path string: {repr(path_str)}")

        # Check for DRIVE:\ or DRIVE:/ pattern (case-insensitive)
        # Note: It checks the actual string value, which might be 'C:\\Users\\...'
        if isinstance(path_str, str) and len(path_str) > 2 and path_str[1] == ':' and path_str[2] in ['\\', '/'] and path_str[0].isalpha():
            try: # Added try-except for robustness during conversion
                drive_letter = path_str[0].lower()
                # path_str[2:] correctly gets the part after 'C:'
                # .replace("\\", "/") correctly handles the single literal backslash in the Python string
                rest_of_path = path_str[3:].replace("\\", "/") # Use index 3 to skip ':\' or ':/'
                # Ensure rest_of_path doesn't start with / after C: (redundant if using index 3, but safe)
                # if rest_of_path.startswith('/'):
                #     rest_of_path = rest_of_path[1:]
                linux_path = f"/mnt/{drive_letter}/{rest_of_path}"
                # Use logger configured elsewhere in your script
                log.debug(f"Converted Windows path '{path_str}' to Linux path '{linux_path}'")
                return linux_path
            except Exception as e:
                log.error(f"Error during path conversion for '{path_str}': {e}", exc_info=True)
                # Return original path on conversion error
                return path_str
        # If it doesn't look like a Windows path, return it unchanged
        log.debug(f"convert_windows_path_to_linux: Path '{path_str}' did not match Windows pattern or wasn't converted.")
        return path_str

    # Apply conversion to the command string itself only if it looks like a path
    # Check if the command itself looks like a potential path that needs conversion
    # (e.g., "C:\path\to\executable.exe" vs just "npx")
    # A simple check: does it contain ':' and '\' or '/'? More robust checks could be added.
    adapted_command = command # Default to original command
    if isinstance(command, str) and ':' in command and ('\\' in command or '/' in command):
         log.debug(f"Attempting conversion for command part: '{command}'")
         adapted_command = convert_windows_path_to_linux(command)
    else:
         log.debug(f"Command part '{command}' likely not a path, skipping conversion.")

    # Apply conversion to each argument if it's a string
    adapted_args = []
    for i, arg in enumerate(args):
        # Make sure we only try to convert strings
        if isinstance(arg, str):
            # --- Added Debug Logging for Arg ---
            log.debug(f"adapt_path_for_platform: Processing arg {i}: {repr(arg)}")
            # --- End Added Debug Logging ---
            converted_arg = convert_windows_path_to_linux(arg)
            adapted_args.append(converted_arg)
        else:
            # --- Added Debug Logging for Non-String Arg ---
            log.debug(f"adapt_path_for_platform: Skipping non-string arg {i}: {repr(arg)}")
            # --- End Added Debug Logging ---
            adapted_args.append(arg) # Keep non-string args (like numbers, bools) as is

    # Log if changes were made (using DEBUG level)
    if adapted_command != command or adapted_args != args:
        log.debug(f"Path adaptation final result: command='{adapted_command}', args={adapted_args}")
    else:
        log.debug("Path adaptation: No changes made to command or arguments.")

    return adapted_command, adapted_args

# --- Stdio Safety (get_safe_console, safe_stdout, verify_no_stdout_pollution) ---
@contextmanager
def safe_stdout():
    has_stdio_servers = False
    try:
        if hasattr(app, "mcp_client") and app.mcp_client and hasattr(app.mcp_client, "server_manager"):
            for name, server in app.mcp_client.server_manager.config.servers.items():
                if server.type == ServerType.STDIO and name in app.mcp_client.server_manager.active_sessions:
                    has_stdio_servers = True
                    break
    except (NameError, AttributeError): pass
    if has_stdio_servers:
        with redirect_stdout(sys.stderr): yield
    else: yield

def get_safe_console():
    has_stdio_servers = False
    try:
        if hasattr(app, "mcp_client") and app.mcp_client and hasattr(app.mcp_client, "server_manager"):
            for name, server in app.mcp_client.server_manager.config.servers.items():
                if server.type == ServerType.STDIO and name in app.mcp_client.server_manager.active_sessions:
                    has_stdio_servers = True
                    # Optional: Add back warning logic if needed
                    break
    except (NameError, AttributeError): pass
    return stderr_console if has_stdio_servers else console

def verify_no_stdout_pollution():
    original_stdout = sys.stdout
    test_buffer = io.StringIO()
    sys.stdout = test_buffer
    try:
        test_buffer.write("TEST_STDOUT_POLLUTION_VERIFICATION")
        captured = test_buffer.getvalue()  # noqa: F841
        if isinstance(original_stdout, StdioProtectionWrapper):
            original_stdout.update_stdio_status()
            # Simplified check: If wrapper exists, assume protection is attempted
            return True
        else:
            sys.stderr.write("\n[CRITICAL] STDOUT IS NOT PROPERLY WRAPPED WITH StdioProtectionWrapper\n")
            log.critical("STDOUT IS NOT PROPERLY WRAPPED WITH StdioProtectionWrapper")
            return False
    finally:
        sys.stdout = original_stdout

# --- Directory Constants ---
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_DIR = PROJECT_ROOT / ".mcpclient_multi_config" # New config dir name
CONFIG_FILE = CONFIG_DIR / "config.yaml"
HISTORY_FILE = CONFIG_DIR / "history.json"
SERVER_DIR = CONFIG_DIR / "servers"
CACHE_DIR = CONFIG_DIR / "cache"
REGISTRY_DIR = CONFIG_DIR / "registry"
MAX_HISTORY_ENTRIES = 300
REGISTRY_URLS = []

# Create directories
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SERVER_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# --- OpenTelemetry Initialization ---
trace_provider = TracerProvider()
use_console_exporter = False
if use_console_exporter:
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    trace_provider.add_span_processor(span_processor)
trace.set_tracer_provider(trace_provider)

try:
    if use_console_exporter:
        reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
        meter_provider = MeterProvider(metric_readers=[reader])
    else:
        meter_provider = MeterProvider()
    metrics.set_meter_provider(meter_provider)
except (ImportError, AttributeError):
    log.warning("OpenTelemetry metrics API initialization failed.")
    meter_provider = MeterProvider() # Dummy
    metrics.set_meter_provider(meter_provider)

tracer = trace.get_tracer("mcpclient_multi")
meter = metrics.get_meter("mcpclient_multi")

try:
    request_counter = meter.create_counter("mcp_requests", description="Number of MCP requests", unit="1")
    latency_histogram = meter.create_histogram("mcp_latency", description="Latency of MCP requests", unit="ms")
    tool_execution_counter = meter.create_counter("tool_executions", description="Number of tool executions", unit="1")
except Exception as e:
    log.warning(f"Failed to create metrics instruments: {e}")
    request_counter, latency_histogram, tool_execution_counter = None, None, None

# --- ServerStatus Enum ---
class ServerStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    UNKNOWN = "unknown"

# --- ServerVersion Class ---
@dataclass
class ServerVersion:
    major: int
    minor: int
    patch: int

    @classmethod
    def from_string(cls, version_str: str) -> "ServerVersion":
        parts = version_str.split(".") + ["0"] * 3
        return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]))
    def __str__(self) -> str: 
        return f"{self.major}.{self.minor}.{self.patch}"
    def is_compatible_with(self, other: "ServerVersion") -> bool: 
        return self.major == other.major

# --- ServerMetrics Class ---
@dataclass
class ServerMetrics:
    uptime: float = 0.0
    request_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    last_checked: datetime = field(default_factory=datetime.now)
    status: ServerStatus = ServerStatus.UNKNOWN
    response_times: List[float] = field(default_factory=list)
    error_rate: float = 0.0
    def update_response_time(self, response_time: float) -> None:
        self.response_times.append(response_time)
        if len(self.response_times) > 100: self.response_times = self.response_times[-100:]
        self.avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.0
    def update_status(self) -> None:
        self.error_rate = self.error_count / max(1, self.request_count)
        if self.error_rate > 0.5 or self.avg_response_time > 10.0: self.status = ServerStatus.ERROR
        elif self.error_rate > 0.1 or self.avg_response_time > 5.0: self.status = ServerStatus.DEGRADED
        elif self.request_count > 0: self.status = ServerStatus.HEALTHY # Only healthy if requests made
        else: self.status = ServerStatus.UNKNOWN

# --- ServerConfig Class ---
@dataclass
class ServerConfig:
    name: str
    type: ServerType
    path: str
    args: List[str] = field(default_factory=list)
    enabled: bool = True
    auto_start: bool = True
    description: str = ""
    trusted: bool = False
    categories: List[str] = field(default_factory=list)
    version: Optional[ServerVersion] = None
    rating: float = 5.0
    retry_count: int = 3
    timeout: float = 250.0
    metrics: ServerMetrics = field(default_factory=ServerMetrics)
    registry_url: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {"max_attempts": 3, "backoff_factor": 0.5, "timeout_increment": 5})
    capabilities: Dict[str, bool] = field(default_factory=lambda: {"tools": True, "resources": True, "prompts": True})

# --- MCPTool, MCPResource, MCPPrompt Classes ---
@dataclass
class MCPTool:
    name: str
    description: str
    server_name: str
    input_schema: Dict[str, Any]
    original_tool: Tool
    call_count: int = 0
    avg_execution_time: float = 0.0
    execution_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_used: datetime = field(default_factory=datetime.now)
    def update_execution_time(self, time_ms: float) -> None:
        self.execution_times.append(time_ms)
        if self.execution_times: self.avg_execution_time = sum(self.execution_times) / len(self.execution_times)
        self.call_count += 1
        self.last_used = datetime.now()

@dataclass
class MCPResource:
    name: str
    description: str
    server_name: str
    template: str
    original_resource: Resource
    call_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)

@dataclass
class MCPPrompt:
    name: str
    description: str
    server_name: str
    template: str
    original_prompt: McpPromptType
    call_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)

# --- ConversationNode Class (Using Type Alias) ---
@dataclass
class ConversationNode:
    id: str
    messages: InternalMessageList = field(default_factory=list) # Use type alias
    parent: Optional["ConversationNode"] = None
    children: List["ConversationNode"] = field(default_factory=list)
    name: str = "Root"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    model: str = "" # Store model name used for the node

    def add_message(self, message: InternalMessage) -> None: # Use type alias
        self.messages.append(message)
        self.modified_at = datetime.now()

    def add_child(self, child: "ConversationNode") -> None: self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "messages": self.messages, # Assumes messages are dicts
                "parent_id": self.parent.id if self.parent else None,
                "children_ids": [child.id for child in self.children],
                "created_at": self.created_at.isoformat(),
                "modified_at": self.modified_at.isoformat(), "model": self.model }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationNode":
        return cls(id=data["id"], messages=data["messages"], name=data["name"],
                   created_at=datetime.fromisoformat(data["created_at"]),
                   modified_at=datetime.fromisoformat(data["modified_at"]),
                   model=data.get("model", "") ) # Load model if present

# --- ChatHistory Class ---
@dataclass
class ChatHistory:
    query: str
    response: str
    model: str
    timestamp: str
    server_names: List[str]
    tools_used: List[str] = field(default_factory=list)
    conversation_id: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: int = 0
    cached: bool = False
    streamed: bool = False

class ServerRegistry:
    """
    Manages discovery and interaction with remote registries and local network servers.
    """
    def __init__(self, registry_urls=None):
        """
        Initializes the ServerRegistry.

        Args:
            registry_urls (Optional[List[str]]): List of remote registry URLs.
        """
        self.registry_urls: List[str] = registry_urls or REGISTRY_URLS
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.local_ratings: Dict[str, float] = {}
        self.http_client: httpx.AsyncClient = httpx.AsyncClient(timeout=30.0)
        self.zeroconf: Optional[Zeroconf] = None
        self.browser: Optional[ServiceBrowser] = None
        self.discovered_servers: Dict[str, Dict[str, Any]] = {}
        # Assuming log is defined globally or passed in
        self.log = logging.getLogger("mcpclient_multi.ServerRegistry")

    async def discover_remote_servers(self, categories=None, min_rating=0.0, max_results=50) -> List[Dict[str, Any]]:
        """
        Discover servers from configured remote registries.

        Args:
            categories (Optional[List[str]]): Filter servers by category.
            min_rating (float): Minimum rating filter.
            max_results (int): Maximum results per registry.

        Returns:
            List[Dict[str, Any]]: A list of discovered server dictionaries.
        """
        all_servers: List[Dict[str, Any]] = []
        if not self.registry_urls:
            self.log.info("No registry URLs configured, skipping remote discovery.")
            return all_servers

        for registry_url in self.registry_urls:
            params: Dict[str, Any] = {"max_results": max_results}
            if categories:
                params["categories"] = ",".join(categories)
            if min_rating > 0.0: # Only add if > 0
                params["min_rating"] = min_rating

            try:
                response = await self.http_client.get(f"{registry_url}/servers", params=params, timeout=5.0)
                if response.status_code == 200:
                    try:
                        servers_data = response.json()
                        servers = servers_data.get("servers", [])
                        for server in servers:
                            server["registry_url"] = registry_url # Add source registry URL
                            all_servers.append(server)
                    except json.JSONDecodeError:
                        self.log.warning(f"Invalid JSON from registry {registry_url}")
                else:
                    self.log.warning(f"Failed to get servers from {registry_url}: Status {response.status_code}")
            except httpx.TimeoutException:
                self.log.warning(f"Timeout connecting to registry {registry_url}")
            except httpx.RequestError as e:
                 self.log.error(f"Network error querying registry {registry_url}: {e}")
            except Exception as e:
                self.log.error(f"Unexpected error querying registry {registry_url}: {e}", exc_info=True) # Include traceback for unexpected errors
        return all_servers

    async def get_server_details(self, server_id: str, registry_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a specific server from registries.

        Args:
            server_id (str): The unique identifier of the server.
            registry_url (Optional[str]): Specific registry URL to query, otherwise checks all known registries.

        Returns:
            Optional[Dict[str, Any]]: Server details dictionary or None if not found.
        """
        urls_to_try: List[str] = [registry_url] if registry_url else self.registry_urls
        if not urls_to_try:
            self.log.warning("No registry URLs configured to get server details.")
            return None

        for url in urls_to_try:
            details_url = f"{url}/servers/{server_id}"
            try:
                response = await self.http_client.get(details_url)
                response.raise_for_status() # Raises HTTPStatusError for 4xx/5xx
                server_details = response.json()
                server_details["registry_url"] = url # Add source
                return server_details
            except httpx.RequestError as e:
                # Log network errors but continue trying other registries
                self.log.debug(f"Network error getting details from {url}: {e}")
            except httpx.HTTPStatusError as e:
                # Log HTTP errors (like 404 Not Found) but continue
                self.log.debug(f"HTTP error getting details from {url}: {e.response.status_code}")
            except json.JSONDecodeError:
                self.log.warning(f"Invalid JSON response getting details from {url}")
            except Exception as e:
                # Log unexpected errors but continue
                self.log.error(f"Unexpected error getting details from {url}: {e}", exc_info=True)

        self.log.warning(f"Could not get details for server '{server_id}' from any configured registry.")
        return None

    def start_local_discovery(self):
        """
        Starts discovering MCP servers on the local network using mDNS/Zeroconf.
        """
        # Check if already running
        if self.browser is not None:
            self.log.info("Local discovery already running.")
            return

        try:
            # Ensure zeroconf is imported and available
            _ = ServiceBrowser # Check if imported
            _ = Zeroconf
        except NameError:
             log.warning("Zeroconf library not available. Install 'zeroconf'. Local discovery disabled.")
             self.zeroconf = None
             self.browser = None
             return

        try:
            # Nested class for the listener
            class MCPServiceListener:
                def __init__(self, registry_instance: 'ServerRegistry'):
                    self.registry = registry_instance
                    self.log = registry_instance.log # Use parent's logger

                def add_service(self, zeroconf_obj: Zeroconf, service_type: str, name: str):
                    """Callback when a service is added or updated."""
                    self.log.debug(f"mDNS Add Service Triggered: type={service_type}, name={name}")
                    info: Optional[ServiceInfo] = None
                    try:
                        # Use timeout for get_service_info
                        info = zeroconf_obj.get_service_info(service_type, name, timeout=1000) # 1 second timeout
                    except EventLoopBlocked:
                        # This can happen, log and return, might get info later
                        self.log.warning(f"Zeroconf event loop blocked getting info for {name}, will retry later.")
                        return
                    except Exception as e:
                        # Log other errors during info retrieval
                        self.log.error(f"Error getting Zeroconf service info for {name}: {e}", exc_info=True)
                        return

                    if not info:
                        self.log.debug(f"No service info returned for {name} after query.")
                        return

                    # Process valid ServiceInfo
                    try:
                        server_name = name.replace("._mcp._tcp.local.", "")
                        host = socket.inet_ntoa(info.addresses[0]) if info.addresses else "localhost"
                        port = info.port if info.port is not None else 0

                        props: Dict[str, str] = {}
                        if info.properties:
                            for k_bytes, v_bytes in info.properties.items():
                                try:
                                    key = k_bytes.decode('utf-8')
                                    value = v_bytes.decode('utf-8')
                                    props[key] = value
                                except UnicodeDecodeError:
                                    self.log.warning(f"Skipping non-UTF8 property key/value for {name}")
                                    continue

                        # Extract details from properties
                        server_protocol = props.get("type", "sse").lower() # Default to sse
                        version_str = props.get("version")
                        version_obj = None
                        if version_str:
                             try:
                                 version_obj = ServerVersion.from_string(version_str) # Use the dataclass if defined
                             except ValueError:
                                 self.log.warning(f"Invalid version string '{version_str}' from mDNS server {name}")

                        categories_str = props.get("categories", "")
                        categories = categories_str.split(",") if categories_str else []
                        description = props.get("description", f"mDNS discovered server at {host}:{port}")

                        # Build the discovered server dictionary
                        server_data = {
                            "name": server_name,
                            "host": host,
                            "port": port,
                            "type": server_protocol, # 'sse' or 'stdio' etc.
                            "url": f"http://{host}:{port}" if server_protocol == "sse" else f"mDNS:{server_name}", # Adjust URL based on type
                            "properties": props,
                            "version": version_obj, # Store the object or None
                            "categories": categories,
                            "description": description,
                            "discovered_via": "mdns"
                        }

                        # Store or update the discovered server
                        self.registry.discovered_servers[server_name] = server_data
                        self.log.info(f"Discovered/Updated local MCP server via mDNS: {server_name} at {host}:{port} ({description})")

                    except Exception as process_err:
                        # Catch errors during processing of the ServiceInfo
                        self.log.error(f"Error processing mDNS service info for {name}: {process_err}", exc_info=True)


                def remove_service(self, zeroconf_obj: Zeroconf, service_type: str, name: str):
                    """Callback when a service is removed."""
                    self.log.debug(f"mDNS Remove Service Triggered: type={service_type}, name={name}")
                    server_name = name.replace("._mcp._tcp.local.", "")
                    if server_name in self.registry.discovered_servers:
                        del self.registry.discovered_servers[server_name]
                        self.log.info(f"Removed local MCP server via mDNS: {server_name}")

                def update_service(self, zeroconf_obj: Zeroconf, service_type: str, name: str):
                    """Callback when a service is updated (often triggers add_service)."""
                    self.log.debug(f"mDNS Update Service Triggered: type={service_type}, name={name}")
                    # Re-call add_service to refresh the information
                    self.add_service(zeroconf_obj, service_type, name)

            # Initialize Zeroconf only if not already done
            if self.zeroconf is None:
                self.log.info("Initializing Zeroconf instance.")
                self.zeroconf = Zeroconf()

            listener = MCPServiceListener(self)
            self.log.info("Starting Zeroconf ServiceBrowser for _mcp._tcp.local.")
            # Create the browser instance
            self.browser = ServiceBrowser(self.zeroconf, "_mcp._tcp.local.", listener)
            self.log.info("Local MCP server discovery started.")

        except OSError as e:
            # Handle network-related errors during startup
            self.log.error(f"Error starting local discovery (network issue?): {e}")
            self.zeroconf = None # Ensure reset on error
            self.browser = None
        except Exception as e:
            # Catch other potential setup errors
            self.log.error(f"Unexpected error during Zeroconf setup: {e}", exc_info=True)
            self.zeroconf = None # Ensure reset on error
            self.browser = None

    def stop_local_discovery(self):
        """Stops local mDNS discovery."""
        if self.browser:
            self.log.info("Stopping Zeroconf ServiceBrowser.")
            # Note: ServiceBrowser doesn't have an explicit stop, closing Zeroconf handles it.
            self.browser = None # Clear the browser reference
        if self.zeroconf:
            self.log.info("Closing Zeroconf instance.")
            try:
                self.zeroconf.close()
            except Exception as e:
                self.log.error(f"Error closing Zeroconf: {e}", exc_info=True)
            finally:
                 self.zeroconf = None # Ensure zeroconf is None after close attempt
        self.log.info("Local discovery stopped.")

    async def rate_server(self, server_id: str, rating: float) -> bool:
        """
        Submit a rating for a discovered server to its registry.

        Args:
            server_id (str): The ID of the server to rate.
            rating (float): The rating value (e.g., 1.0 to 5.0).

        Returns:
            bool: True if rating was submitted successfully, False otherwise.
        """
        # Store locally first
        self.local_ratings[server_id] = rating

        # Find the server and its registry URL
        # Check self.servers (configured) and self.discovered_servers (via registry/mDNS)
        server_info = self.servers.get(server_id) # Check configured first
        if not server_info and server_id in self.discovered_servers:
             server_info = self.discovered_servers[server_id]

        registry_url_to_use = None
        if server_info and isinstance(server_info, dict): # Check if it's a dict before accessing keys
             registry_url_to_use = server_info.get("registry_url")
        elif hasattr(server_info, 'registry_url'): # Handle dataclass case (ServerConfig)
            registry_url_to_use = server_info.registry_url

        if not registry_url_to_use:
            self.log.warning(f"Cannot rate server '{server_id}': Registry URL unknown.")
            return False

        rate_url = f"{registry_url_to_use}/servers/{server_id}/rate"
        payload = {"rating": rating}

        try:
            response = await self.http_client.post(rate_url, json=payload)
            response.raise_for_status() # Check for HTTP errors
            if response.status_code == 200:
                self.log.info(f"Successfully submitted rating ({rating}) for server {server_id} to {registry_url_to_use}")
                return True
            else:
                # This case might not be hit due to raise_for_status, but good practice
                self.log.warning(f"Rating submission for {server_id} returned status {response.status_code}")
                return False
        except httpx.RequestError as e:
            self.log.error(f"Network error rating server {server_id} at {rate_url}: {e}")
        except httpx.HTTPStatusError as e:
            self.log.error(f"HTTP error rating server {server_id}: Status {e.response.status_code} from {rate_url}")
        except Exception as e:
            self.log.error(f"Unexpected error rating server {server_id}: {e}", exc_info=True)

        return False

    async def close(self):
        """Clean up resources: stop discovery and close HTTP client."""
        self.log.info("Closing ServerRegistry resources.")
        self.stop_local_discovery() # Ensure mDNS stops
        try:
            await self.http_client.aclose()
            self.log.debug("HTTP client closed.")
        except Exception as e:
            self.log.error(f"Error closing HTTP client: {e}", exc_info=True)
            
class ToolCache:
    def __init__(self, cache_dir=CACHE_DIR, custom_ttl_mapping=None):
        self.cache_dir = Path(cache_dir)
        self.memory_cache: Dict[str, CacheEntry] = {}
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.disk_cache = diskcache.Cache(str(self.cache_dir / "tool_results"))
        except Exception as e:
            log.error(f"Failed to initialize disk cache at {self.cache_dir}: {e}. Disk caching disabled.", exc_info=True)
            self.disk_cache = None # Disable disk cache if init fails

        self.ttl_mapping = {"weather": 1800, "filesystem": 300, "search": 86400, "database": 300}
        if custom_ttl_mapping:
            self.ttl_mapping.update(custom_ttl_mapping)
        self.dependency_graph: Dict[str, Set[str]] = {}

    def add_dependency(self, tool_name, depends_on):
        self.dependency_graph.setdefault(tool_name, set()).add(depends_on)

    def invalidate_related(self, tool_name):
        affected, stack = set(), [tool_name]
        while stack:
            current = stack.pop()
            # Avoid infinite loops for cyclic dependencies (though unlikely here)
            if current in affected:
                continue
            affected.add(current)
            for dependent, dependencies in self.dependency_graph.items():
                if current in dependencies and dependent not in affected:
                    stack.append(dependent)

        # Remove the originating tool itself, only invalidate dependents
        if tool_name in affected:
            affected.remove(tool_name)

        if affected:
             log.info(f"Invalidating related caches for tools dependent on '{tool_name}': {affected}")
             for tool in affected:
                 # Call invalidate for the tool name, which handles both memory and disk
                 self.invalidate(tool_name=tool)
                 # Logging moved inside the loop within invalidate(tool_name=...) for clarity

    def get_ttl(self, tool_name):
        for category, ttl in self.ttl_mapping.items():
            if category in tool_name.lower():
                return ttl
        return 3600 # Default 1 hour

    def generate_key(self, tool_name, params):
        # Ensure params are serializable, handle potential errors during dump
        try:
            # Use default=str as a fallback for basic non-serializable types
            params_str = json.dumps(params, sort_keys=True, default=str)
            params_hash = hashlib.sha256(params_str.encode()).hexdigest()
            return f"{tool_name}:{params_hash}"
        except TypeError as e:
            # Catch TypeError during json.dumps if default=str doesn't handle it
            log.warning(f"Could not generate cache key for tool '{tool_name}': Params not JSON serializable - {e}")
            raise # Re-raise the TypeError so caller knows key generation failed

    def get(self, tool_name, params):
        try:
            key = self.generate_key(tool_name, params)
        except TypeError:
            # Logged in generate_key, just return None
            return None

        # Check memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                log.debug(f"Cache HIT (Memory): {key}")
                return entry.result
            else:
                log.debug(f"Cache STALE (Memory): {key}")
                del self.memory_cache[key] # Remove expired from memory

        # Check disk cache if enabled
        if self.disk_cache:
            try:
                # Check existence *before* getting to potentially avoid errors
                if key in self.disk_cache:
                    entry = self.disk_cache.get(key) # Use get for safer retrieval
                    if isinstance(entry, CacheEntry) and not entry.is_expired():
                        log.debug(f"Cache HIT (Disk): {key}")
                        self.memory_cache[key] = entry # Promote to memory
                        return entry.result
                    else:
                        # Entry is expired or invalid type
                        log.debug(f"Cache STALE/INVALID (Disk): {key}")
                        # Safely delete expired/invalid entry
                        with suppress(KeyError, Exception): # Suppress errors during delete
                             del self.disk_cache[key]
            except (OSError, EOFError, diskcache.Timeout, Exception) as e:
                # Handle potential errors during disk cache access
                log.warning(f"Disk cache GET error for key '{key}': {e}")
                # Optionally try to delete potentially corrupted key
                with suppress(KeyError, Exception):
                     del self.disk_cache[key]

        log.debug(f"Cache MISS: {key}")
        return None

    def set(self, tool_name, params, result, ttl=None):
        try:
            key = self.generate_key(tool_name, params)
        except TypeError:
            # Logged in generate_key, cannot cache
            return

        if ttl is None:
            ttl = self.get_ttl(tool_name)

        expires_at = (datetime.now() + timedelta(seconds=ttl)) if ttl >= 0 else None # Allow ttl=0 for non-expiring? Changed to >=0
        if ttl < 0:
             log.warning(f"Negative TTL ({ttl}) provided for caching tool '{tool_name}'. Cache entry will not expire.")
             expires_at = None # Treat negative TTL as non-expiring

        entry = CacheEntry(result=result, created_at=datetime.now(), expires_at=expires_at,
                           tool_name=tool_name, parameters_hash=key.split(":")[-1])

        # Set memory cache
        self.memory_cache[key] = entry
        log.debug(f"Cache SET (Memory): {key} (TTL: {ttl}s)")

        # Set disk cache if enabled
        if self.disk_cache:
            try:
                self.disk_cache.set(key, entry, expire=ttl if ttl >= 0 else None) # Use diskcache expire param
                log.debug(f"Cache SET (Disk): {key} (TTL: {ttl}s)")
            except Exception as e:
                log.warning(f"Disk cache SET error for key '{key}': {e}")

    # --- invalidate Method (Synchronous Fix) ---
    def invalidate(self, tool_name=None, params=None):
        if tool_name and params:
            try:
                key = self.generate_key(tool_name, params)
            except TypeError:
                key = None # Cannot invalidate if key cannot be generated
            if key:
                log.debug(f"Invalidating specific cache key: {key}")
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if self.disk_cache:
                    # Use synchronous suppress for KeyError and potentially others
                    with suppress(KeyError, OSError, EOFError, diskcache.Timeout, Exception):
                         if key in self.disk_cache: # Check before deleting
                              del self.disk_cache[key]
        elif tool_name:
            log.info(f"Invalidating all cache entries for tool: {tool_name}")
            prefix = f"{tool_name}:"
            # Invalidate memory
            keys_to_remove_mem = [k for k in self.memory_cache if k.startswith(prefix)]
            for key in keys_to_remove_mem:
                del self.memory_cache[key]
            log.debug(f"Removed {len(keys_to_remove_mem)} entries from memory cache for {tool_name}.")
            # Invalidate disk
            if self.disk_cache:
                removed_disk_count = 0
                try:
                    # Iterate safely, collect keys first if needed, or use prefix deletion if available
                    # Simple iteration (might be slow for large caches)
                    keys_to_remove_disk = []
                    # Safely iterate keys
                    with suppress(Exception): # Suppress errors during iteration itself
                        for key in self.disk_cache.iterkeys(): # iterkeys is sync
                            if key.startswith(prefix):
                                keys_to_remove_disk.append(key)

                    for key in keys_to_remove_disk:
                         with suppress(KeyError, OSError, EOFError, diskcache.Timeout, Exception):
                              del self.disk_cache[key]
                              removed_disk_count += 1
                    log.debug(f"Removed {removed_disk_count} entries from disk cache for {tool_name}.")
                except Exception as e:
                    log.warning(f"Error during disk cache invalidation for tool '{tool_name}': {e}")
            # Invalidate related tools AFTER invalidating the current one
            self.invalidate_related(tool_name)
        else:
            log.info("Invalidating ALL cache entries.")
            self.memory_cache.clear()
            if self.disk_cache:
                try:
                    self.disk_cache.clear()
                except Exception as e:
                    log.error(f"Failed to clear disk cache: {e}")

    # --- clean Method (Synchronous Fix) ---
    def clean(self):
        """Remove expired entries from memory and disk caches."""
        log.debug("Running cache cleaning process...")
        # Clean memory cache
        mem_keys_before = set(self.memory_cache.keys())
        for key in list(self.memory_cache.keys()): # Iterate over a copy
            if self.memory_cache[key].is_expired():
                del self.memory_cache[key]
        mem_removed = len(mem_keys_before) - len(self.memory_cache)
        if mem_removed > 0:
            log.info(f"Removed {mem_removed} expired entries from memory cache.")

        # Clean disk cache if enabled
        if self.disk_cache:
            try:
                # diskcache's expire() method handles removing expired items efficiently
                # It returns the number of items removed.
                disk_removed = self.disk_cache.expire()
                if disk_removed > 0:
                     log.info(f"Removed {disk_removed} expired entries from disk cache.")
            except Exception as e:
                log.error(f"Error during disk cache expire: {e}", exc_info=True)
        log.debug("Cache cleaning finished.")


    def close(self):
        if self.disk_cache:
            try:
                self.disk_cache.close()
            except Exception as e:
                log.error(f"Error closing disk cache: {e}")

# --- ConversationGraph Class ---
class ConversationGraph:
    def __init__(self): 
        self.nodes: Dict[str, ConversationNode] = {}
        self.root = ConversationNode(id="root", name="Root")
        self.current_node = self.root
        self.nodes[self.root.id] = self.root

    def add_node(self, node: ConversationNode): 
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[ConversationNode]: 
        return self.nodes.get(node_id)
    
    def create_fork(self, name: Optional[str] = None) -> ConversationNode:
        fork_id=str(uuid.uuid4())
        fork_name = name or f"Fork {len(self.current_node.children) + 1}"
        new_node = ConversationNode(id=fork_id, name=fork_name, parent=self.current_node, messages=self.current_node.messages.copy(), model=self.current_node.model)
        self.current_node.add_child(new_node)
        self.add_node(new_node)
        return new_node
    
    def set_current_node(self, node_id: str) -> bool:
        if node_id in self.nodes: 
            self.current_node = self.nodes[node_id]
            return True
        return False
    
    def get_path_to_root(self, node: Optional[ConversationNode] = None) -> List[ConversationNode]:
        node = node or self.current_node
        path = [node]
        current = node
        while current.parent: 
            path.append(current.parent)
            current = current.parent
        return list(reversed(path))
    
    async def save(self, file_path: str):
        data = {"current_node_id": self.current_node.id, "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()}}
        try: 
            async with aiofiles.open(file_path, 'w') as f: 
                await f.write(json.dumps(data, indent=2))
        except (IOError, TypeError) as e: 
            log.error(f"Could not save graph {file_path}: {e}")

    @classmethod
    async def load(cls, file_path: str) -> "ConversationGraph":
        file_path_obj = Path(file_path)
        try:
            async with aiofiles.open(file_path_obj, 'r') as f: 
                content = await f.read()
            if not content.strip(): 
                log.warning(f"Graph file empty: {file_path}.")
                return cls()
            data = json.loads(content)
            graph = cls()
            for node_id, node_data in data["nodes"].items(): 
                node = ConversationNode.from_dict(node_data)
                graph.nodes[node_id] = node
            for node_id, node_data in data["nodes"].items():
                node = graph.nodes[node_id]
                parent_id = node_data.get("parent_id")
                if parent_id and parent_id in graph.nodes: 
                    node.parent = graph.nodes[parent_id]
                for child_id in node_data.get("children_ids", []):
                    if child_id in graph.nodes: 
                        child = graph.nodes[child_id]
                    if child not in node.children: 
                        node.children.append(child)
            current_node_id = data.get("current_node_id", "root")
            if current_node_id in graph.nodes: 
                graph.current_node = graph.nodes[current_node_id]
            else: 
                log.warning(f"Saved node_id '{current_node_id}' missing.")
                graph.current_node = graph.root
            log.info(f"Loaded graph from {file_path}")
            return graph
        except FileNotFoundError: 
            log.info(f"Graph file missing: {file_path}.")
            return cls()
        except (IOError, json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
            log.warning(f"Failed load/parse graph {file_path}: {e}.", exc_info=False)
            log.debug("Traceback:", exc_info=True)
            try:
                if file_path_obj.exists(): 
                    backup_path = file_path_obj.with_suffix(f".json.corrupted.{int(time.time())}")
                    os.rename(file_path_obj, backup_path)
                    log.info(f"Backed up corrupted file: {backup_path}")
            except Exception as backup_err: 
                log.error(f"Failed backup graph {file_path}: {backup_err}", exc_info=True)
            return cls()
        except Exception: 
            log.error(f"Unexpected error loading graph {file_path}.", exc_info=True)
            return cls()


# =============================================================================
# Configuration Class (REVISED IMPLEMENTATION)
# =============================================================================

# Helper Mappings for Config Env Var Overrides (REFINED)
PROVIDER_ENV_VAR_MAP = {
    Provider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
    Provider.OPENAI.value: "OPENAI_API_KEY",
    Provider.GEMINI.value: "GEMINI_API_KEY", # Use GEMINI_API_KEY
    Provider.GROK.value: "GROK_API_KEY",
    Provider.DEEPSEEK.value: "DEEPSEEK_API_KEY",
    Provider.MISTRAL.value: "MISTRAL_API_KEY",
    Provider.GROQ.value: "GROQ_API_KEY",
    Provider.CEREBRAS.value: "CEREBRAS_API_KEY",
    Provider.OPENROUTER.value: "OPENROUTER_API_KEY"
}

PROVIDER_CONFIG_KEY_ATTR_MAP = {
    Provider.ANTHROPIC.value: "anthropic_api_key",
    Provider.OPENAI.value: "openai_api_key",
    Provider.GEMINI.value: "gemini_api_key",
    Provider.GROK.value: "grok_api_key",
    Provider.DEEPSEEK.value: "deepseek_api_key",
    Provider.MISTRAL.value: "mistral_api_key",
    Provider.GROQ.value: "groq_api_key",
    Provider.CEREBRAS.value: "cerebras_api_key",
    Provider.OPENROUTER.value: "openrouter_api_key"
}

PROVIDER_CONFIG_URL_ATTR_MAP = {
    Provider.OPENAI.value: "openai_base_url",
    Provider.GEMINI.value: "gemini_base_url",
    Provider.GROK.value: "grok_base_url",
    Provider.DEEPSEEK.value: "deepseek_base_url",
    Provider.MISTRAL.value: "mistral_base_url",
    Provider.GROQ.value: "groq_base_url",
    Provider.CEREBRAS.value: "cerebras_base_url",
    Provider.OPENROUTER.value: "openrouter_base_url"
}

PROVIDER_ENV_URL_MAP = {
    Provider.OPENAI.value: "OPENAI_BASE_URL",
    Provider.GEMINI.value: "GEMINI_BASE_URL",
    Provider.GROK.value: "GROK_BASE_URL",
    Provider.DEEPSEEK.value: "DEEPSEEK_BASE_URL",
    Provider.MISTRAL.value: "MISTRAL_BASE_URL",
    Provider.GROQ.value: "GROQ_BASE_URL",
    Provider.CEREBRAS.value: "CEREBRAS_BASE_URL",
    Provider.OPENROUTER.value: "OPENROUTER_BASE_URL"
}

# Mapping for simple settings <-> Env Var Names
SIMPLE_SETTINGS_ENV_MAP = {
    "DEFAULT_MODEL": "default_model",
    "DEFAULT_MAX_TOKENS": "default_max_tokens",
    "HISTORY_SIZE": "history_size",
    "AUTO_DISCOVER": "auto_discover",
    "DISCOVERY_PATHS": "discovery_paths",
    "ENABLE_STREAMING": "enable_streaming",
    "ENABLE_CACHING": "enable_caching",
    "ENABLE_METRICS": "enable_metrics",
    "ENABLE_REGISTRY": "enable_registry",
    "ENABLE_LOCAL_DISCOVERY": "enable_local_discovery",
    "TEMPERATURE": "temperature",
    "CONVERSATION_GRAPHS_DIR": "conversation_graphs_dir",
    "REGISTRY_URLS": "registry_urls",
    "DASHBOARD_REFRESH_RATE": "dashboard_refresh_rate",
    "SUMMARIZATION_MODEL": "summarization_model",
    "USE_AUTO_SUMMARIZATION": "use_auto_summarization",
    "AUTO_SUMMARIZE_THRESHOLD": "auto_summarize_threshold",
    "MAX_SUMMARIZED_TOKENS": "max_summarized_tokens",
    "ENABLE_PORT_SCANNING": "enable_port_scanning",
    "PORT_SCAN_RANGE_START": "port_scan_range_start",
    "PORT_SCAN_RANGE_END": "port_scan_range_end",
    "PORT_SCAN_CONCURRENCY": "port_scan_concurrency",
    "PORT_SCAN_TIMEOUT": "port_scan_timeout",
    "PORT_SCAN_TARGETS": "port_scan_targets",
}

class Config:
    def __init__(self):
        """
        Initializes configuration by setting defaults, loading YAML,
        and applying environment variable overrides for simple settings.
        """
        log.debug("Initializing Config object...")
        self._set_defaults()
        self.dotenv_path = find_dotenv(raise_error_if_not_found=False, usecwd=True)
        if self.dotenv_path:
            log.info(f"Located .env file at: {self.dotenv_path}")
            load_dotenv(dotenv_path=self.dotenv_path, override=True) # Load .env into os.environ
            log.debug(f"Loaded environment variables from {self.dotenv_path}")
        else:
            # Decide where a .env file *should* go if created later
            self.dotenv_path = str(Path.cwd() / ".env") # Default to current working dir
            log.info(f"No .env file found. Will use default path if saving: {self.dotenv_path}")
        self.load_from_yaml()
        self._apply_env_overrides()
        Path(self.conversation_graphs_dir).mkdir(parents=True, exist_ok=True)
        log.info(f"Configuration initialized. Default model: {self.default_model}")
        log.debug(f"Config values after init: {self.__dict__}")

    def _set_defaults(self):
        """Sets hardcoded default values for all config attributes."""
        # API Keys & URLs
        self.anthropic_api_key: Optional[str] = None
        self.openai_api_key: Optional[str] = None
        self.gemini_api_key: Optional[str] = None
        self.grok_api_key: Optional[str] = None
        self.deepseek_api_key: Optional[str] = None
        self.mistral_api_key: Optional[str] = None
        self.groq_api_key: Optional[str] = None
        self.cerebras_api_key: Optional[str] = None
        self.openrouter_api_key: Optional[str] = None

        self.openai_base_url: Optional[str] = None
        self.gemini_base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/" # Special URL for openai-compatible API
        self.grok_base_url: Optional[str] = "https://api.x.ai/v1"
        self.deepseek_base_url: Optional[str] = "https://api.deepseek.com/v1"
        self.mistral_base_url: Optional[str] = "https://api.mistral.ai/v1"
        self.groq_base_url: Optional[str] = "https://api.groq.com/openai/v1"
        self.cerebras_base_url: Optional[str] = "https://api.cerebras.ai/v1"
        self.openrouter_base_url: Optional[str] = "https://openrouter.ai/api/v1"

        # Default model
        self.default_model: str = DEFAULT_MODELS.get(Provider.OPENAI.value, "gpt-4.1-mini")

        # Other settings
        self.default_max_tokens: int = 8000
        self.history_size: int = MAX_HISTORY_ENTRIES
        self.auto_discover: bool = True
        default_discovery_paths: List[str] = [
            str(SERVER_DIR),
            os.path.expanduser("~/mcp-servers"),
            os.path.expanduser("~/modelcontextprotocol/servers")
        ]
        self.discovery_paths: List[str] = default_discovery_paths
        self.enable_streaming: bool = True
        self.enable_caching: bool = True
        self.enable_metrics: bool = True
        self.enable_registry: bool = True
        self.enable_local_discovery: bool = True
        self.temperature: float = 0.7
        self.conversation_graphs_dir: str = str(CONFIG_DIR / "conversations")
        self.registry_urls: List[str] = REGISTRY_URLS.copy()
        self.dashboard_refresh_rate: float = 2.0
        self.summarization_model: str = DEFAULT_MODELS.get(Provider.ANTHROPIC.value, "claude-3-haiku-20240307")
        self.use_auto_summarization: bool = False
        self.auto_summarize_threshold: int = 100000
        self.max_summarized_tokens: int = 1500
        self.enable_port_scanning: bool = True
        self.port_scan_range_start: int = 8000
        self.port_scan_range_end: int = 9000
        self.port_scan_concurrency: int = 50
        self.port_scan_timeout: float = 4.5
        self.port_scan_targets: List[str] = ["127.0.0.1"]

        # Complex structures
        self.servers: Dict[str, ServerConfig] = {}
        self.cache_ttl_mapping: Dict[str, int] = {}

    def _apply_env_overrides(self):
        """Overrides simple config values with environment variables if they are set."""
        log.debug("Applying environment variable overrides...")
        updated_vars = []

        # --- Override API Keys ---
        for provider_value, attr_name in PROVIDER_CONFIG_KEY_ATTR_MAP.items():
            env_var_name = PROVIDER_ENV_VAR_MAP.get(provider_value)
            if env_var_name:
                env_value = os.getenv(env_var_name)
                if env_value is not None:
                    setattr(self, attr_name, env_value)
                    updated_vars.append(f"{attr_name} (from {env_var_name})")

        # --- Override Base URLs ---
        for provider_value, attr_name in PROVIDER_CONFIG_URL_ATTR_MAP.items():
            env_var_name = PROVIDER_ENV_URL_MAP.get(provider_value)
            if env_var_name:
                env_value = os.getenv(env_var_name)
                if env_value is not None:
                    setattr(self, attr_name, env_value)
                    updated_vars.append(f"{attr_name} (from {env_var_name})")

        # --- Override Other Simple Settings ---
        for env_var, attr_name in SIMPLE_SETTINGS_ENV_MAP.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                target_attr = getattr(self, attr_name, None)
                target_type = type(target_attr) if target_attr is not None else str
                try:
                    parsed_value: Any
                    if target_type is bool:
                        parsed_value = env_value.lower() in ('true', '1', 't', 'yes', 'y')
                    elif target_type is int:
                        parsed_value = int(env_value)
                    elif target_type is float:
                        parsed_value = float(env_value)

                    elif target_type is list:
                        try:
                            # Assuming elements should be strings
                            parsed_value = Csv(str)(env_value)
                        except Exception as parse_err:
                            # Log error if Csv parsing fails, keep default
                            log.warning(f"Could not parse list from env var '{env_var}' using decouple.Csv: {parse_err}. Keeping default value.")
                            continue # Skip setting attribute if parsing fails

                    else: # Default to string
                        parsed_value = env_value

                    # Only update if the value actually changed from the default/YAML loaded value
                    current_val = getattr(self, attr_name, None)
                    # Need careful comparison for lists/dicts
                    if isinstance(current_val, list) and isinstance(parsed_value, list):
                        if set(current_val) != set(parsed_value): # Compare sets for lists
                             setattr(self, attr_name, parsed_value)
                             updated_vars.append(f"{attr_name} (from {env_var})")
                    elif current_val != parsed_value:
                         setattr(self, attr_name, parsed_value)
                         updated_vars.append(f"{attr_name} (from {env_var})")

                except (ValueError, TypeError) as e:
                    log.warning(f"Could not apply env var '{env_var}' to '{attr_name}'. Invalid value '{env_value}' for type {target_type}: {e}")

        if updated_vars:
            log.info(f"Applied environment variable overrides for: {', '.join(updated_vars)}")
        else:
            log.debug("No environment variable overrides applied.")

    def _prepare_config_data(self) -> Dict[str, Any]:
        """Prepares the full configuration state for saving to YAML."""
        data_to_save = {}
        skip_attributes = {"decouple_instance", "dotenv_path"} # Attributes to skip saving

        for attr_name, attr_value in self.__dict__.items():
            if attr_name.startswith("_") or callable(attr_value) or attr_name in skip_attributes:
                continue

            if attr_name == "servers":
                serialized_servers: Dict[str, Dict[str, Any]] = {}
                for name, server_config in self.servers.items():
                    # Simplified server data for YAML (excluding volatile metrics unless needed)
                    server_data = {
                        'type': server_config.type.value, 'path': server_config.path,
                        'args': server_config.args, 'enabled': server_config.enabled,
                        'auto_start': server_config.auto_start, 'description': server_config.description,
                        'trusted': server_config.trusted, 'categories': server_config.categories,
                        'version': str(server_config.version) if server_config.version else None,
                        'rating': server_config.rating, 'retry_count': server_config.retry_count,
                        'timeout': server_config.timeout, 'retry_policy': server_config.retry_policy,
                        'registry_url': server_config.registry_url, 'capabilities': server_config.capabilities,
                        # Optionally include metrics if persistence is desired, otherwise skip
                        # 'metrics': { ... }
                    }
                    serialized_servers[name] = server_data
                data_to_save[attr_name] = serialized_servers
            elif attr_name == "cache_ttl_mapping":
                data_to_save[attr_name] = self.cache_ttl_mapping
            elif isinstance(attr_value, (str, int, float, bool, list, dict, type(None))):
                 data_to_save[attr_name] = attr_value
            elif isinstance(attr_value, Path):
                 data_to_save[attr_name] = str(attr_value)
            # Add other serializable types if needed
            # else: log.debug(f"Skipping attribute '{attr_name}' type {type(attr_value)} for YAML.")

        return data_to_save

    def load_from_yaml(self):
        """Loads configuration state from the YAML file, overwriting defaults."""
        if not CONFIG_FILE.exists():
            log.info(f"YAML config file {CONFIG_FILE} not found. Using defaults/env vars.")
            return

        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            log.info(f"Loading configuration from YAML file: {CONFIG_FILE}")
        except Exception as e:
            log.error(f"Error loading YAML config {CONFIG_FILE}: {e}", exc_info=True)
            return

        # Iterate through loaded data and update self attributes
        for key, value in config_data.items():
            if key == 'servers':
                self.servers = {} # Reset before loading
                if isinstance(value, dict):
                    for server_name, server_data in value.items():
                        if not isinstance(server_data, dict): continue
                        try:
                            srv_type = ServerType(server_data.get('type', 'stdio'))
                            version_str = server_data.get('version')
                            version = ServerVersion.from_string(version_str) if version_str else None
                            # Initialize metrics, don't load from YAML by default unless needed
                            metrics = ServerMetrics()
                            # Create ServerConfig, handle missing keys gracefully
                            default_config = ServerConfig(name=server_name, type=srv_type, path="")
                            server_kwargs = {k: v for k, v in server_data.items() if hasattr(default_config, k) and k not in ['type', 'version', 'metrics']}
                            server_kwargs['type'] = srv_type
                            server_kwargs['version'] = version
                            server_kwargs['metrics'] = metrics
                            server_kwargs['name'] = server_name
                            self.servers[server_name] = ServerConfig(**server_kwargs)
                        except Exception as server_load_err:
                            log.warning(f"Skipping server '{server_name}' from YAML: {server_load_err}", exc_info=True)
                else:
                    log.warning("'servers' key in YAML is not a dict.")
            elif key == 'cache_ttl_mapping':
                 if isinstance(value, dict):
                     valid_mapping = {}
                     for k, v in value.items():
                          if isinstance(k, str) and isinstance(v, int): valid_mapping[k] = v
                          else: log.warning(f"Invalid YAML cache_ttl_mapping: K='{k}', V='{v}'.")
                     self.cache_ttl_mapping = valid_mapping
                 else:
                     log.warning("'cache_ttl_mapping' in YAML is not a dict.")
                     self.cache_ttl_mapping = {}
            elif hasattr(self, key):
                # Update simple attributes if type matches or can be converted
                current_attr_value = getattr(self, key, None)
                current_type = type(current_attr_value) if current_attr_value is not None else None
                try:
                    if value is None and current_type is not type(None):
                        log.debug(f"YAML value for '{key}' is None, keeping default: {current_attr_value}")
                        # Keep the default value if YAML has None unless default is None
                        setattr(self, key, current_attr_value)
                    elif current_type is None or isinstance(value, current_type):
                        setattr(self, key, value)
                    elif current_type in [int, float, bool, list, str]: # Attempt basic type conversion
                         if current_type is bool:
                             converted_value = str(value).lower() in ('true', '1', 't', 'yes', 'y')
                         elif current_type is list and isinstance(value, list): # Ensure list conversion takes list
                             converted_value = value # Assume YAML list is correct
                         else:
                             converted_value = current_type(value) # Try direct conversion
                         setattr(self, key, converted_value)
                         log.debug(f"Converted YAML value for '{key}' from {type(value)} to {current_type}")
                    else:
                         log.warning(f"Type mismatch for '{key}' in YAML (Expected: {current_type}, Got: {type(value)}). Keeping default.")
                except (ValueError, TypeError) as conv_err:
                    log.warning(f"Could not convert YAML value for '{key}': {conv_err}. Keeping default.")
                except Exception as attr_set_err:
                     log.warning(f"Error setting attribute '{key}' from YAML: {attr_set_err}")
            else:
                 log.warning(f"Ignoring unknown config key '{key}' from YAML.")

    async def save_async(self):
        """Saves the full configuration state to the YAML file asynchronously."""
        config_data = self._prepare_config_data()
        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml_str = yaml.dump(config_data, allow_unicode=True, sort_keys=False, default_flow_style=False, indent=2)
                await f.write(yaml_str)
            log.debug(f"Saved full configuration async to {CONFIG_FILE}")
        except Exception as e:
            log.error(f"Error async saving YAML config {CONFIG_FILE}: {e}", exc_info=True)

    def save_sync(self):
        """Saves the full configuration state to the YAML file synchronously."""
        config_data = self._prepare_config_data()
        try:
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False, indent=2)
            log.debug(f"Saved full configuration sync to {CONFIG_FILE}")
        except Exception as e:
            log.error(f"Error sync saving YAML config {CONFIG_FILE}: {e}", exc_info=True)

    # Keep get_api_key and get_base_url
    def get_api_key(self, provider: str) -> Optional[str]:
        key_attr = f"{provider}_api_key"
        return getattr(self, key_attr, None)

    def get_base_url(self, provider: str) -> Optional[str]:
        url_attr = f"{provider}_base_url"
        return getattr(self, url_attr, None)


# --- History Class ---
class History:
    def __init__(self, max_entries=MAX_HISTORY_ENTRIES):
        self.entries = deque(maxlen=max_entries)
        self.max_entries = max_entries
        self.load_sync()

    def add(self, entry: ChatHistory):
        self.entries.append(entry)
        self.save_sync()

    async def add_async(self, entry: ChatHistory):
        self.entries.append(entry)
        await self.save()

    def _load_history_data(self, data_str: str):
        """Helper to parse history data"""
        history_data = json.loads(data_str) if data_str else []
        self.entries.clear()
        for entry_data in history_data:
            self.entries.append(ChatHistory(**entry_data))

    def _prepare_history_data(self) -> str:
        """Helper to format history data for saving"""
        history_data = []
        for entry in self.entries:
             # Convert ChatHistory dataclass to dict
             entry_dict = dataclasses.asdict(entry)
             history_data.append(entry_dict)
        return json.dumps(history_data, indent=2)

    def load_sync(self):
        if not HISTORY_FILE.exists(): return
        try:
            with open(HISTORY_FILE, 'r') as f: data_str = f.read()
            self._load_history_data(data_str)
        except (IOError, json.JSONDecodeError, Exception) as e:
             log.error(f"Error loading history {HISTORY_FILE}: {e}")

    def save_sync(self):
        try:
            data_str = self._prepare_history_data()
            with open(HISTORY_FILE, 'w') as f: f.write(data_str)
        except (IOError, TypeError, Exception) as e:
             log.error(f"Error saving history {HISTORY_FILE}: {e}")

    async def load(self):
        if not HISTORY_FILE.exists(): return
        try:
            async with aiofiles.open(HISTORY_FILE, 'r') as f: data_str = await f.read()
            self._load_history_data(data_str)
        except (IOError, json.JSONDecodeError, Exception) as e:
             log.error(f"Error async loading history {HISTORY_FILE}: {e}")

    async def save(self):
        try:
            data_str = self._prepare_history_data()
            async with aiofiles.open(HISTORY_FILE, 'w') as f: await f.write(data_str)
        except (IOError, TypeError, Exception) as e:
             log.error(f"Error async saving history {HISTORY_FILE}: {e}")

    def search(self, query: str, limit: int = 5) -> List[ChatHistory]:
        results = []
        query_lower = query.lower()
        for entry in reversed(self.entries):
            if (query_lower in entry.query.lower() or
                query_lower in entry.response.lower() or
                any(query_lower in tool.lower() for tool in entry.tools_used) or
                any(query_lower in server.lower() for server in entry.server_names)):
                results.append(entry)
                if len(results) >= limit: break
        return results

# --- ServerMonitor Class ---
class ServerMonitor:
    def __init__(self, server_manager: "ServerManager"):
        self.server_manager = server_manager
        self.monitoring = False
        self.monitor_task = None
        self.health_check_interval = 30

    async def start_monitoring(self):
        if self.monitoring: return
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        log.info("Server health monitoring started")

    async def stop_monitoring(self):
        if not self.monitoring: return
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                log.debug("Server monitor task successfully cancelled.")
            except Exception as e:
                # Log other potential errors during await if needed
                log.warning(f"Error awaiting cancelled monitor task: {e}")
        self.monitor_task = None
        log.info("Server health monitoring stopped")

    async def _monitor_loop(self):
        while self.monitoring:
            try:
                await self._check_all_servers()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError: 
                break
            except Exception as e: 
                log.error(f"Error in server monitor: {e}")
                await asyncio.sleep(5)

    async def _check_all_servers(self):
        for name, session in list(self.server_manager.active_sessions.items()):
            try: await self._check_server_health(name, session)
            except McpError as e: log.error(f"MCP error check health {name}: {e}")
            except httpx.RequestError as e: log.error(f"Net error check health {name}: {e}")
            except Exception as e: log.error(f"Unexpected error check health {name}: {e}")

    async def _check_server_health(self, server_name: str, session: ClientSession):
        if server_name not in self.server_manager.config.servers: return
        server_config = self.server_manager.config.servers[server_name]
        metrics = server_config.metrics
        metrics.uptime += self.health_check_interval / 60 # minutes
        start_time = time.time()
        try:
            await session.list_tools() # Simple health check
            response_time = time.time() - start_time
            metrics.update_response_time(response_time)
        except Exception as e:
            metrics.error_count += 1
            log.warning(f"Health check fail {server_name}: {e}")
        metrics.update_status()
        if metrics.status == ServerStatus.ERROR: await self._recover_server(server_name)

    async def _recover_server(self, server_name: str):
        if server_name not in self.server_manager.config.servers: return
        server_config = self.server_manager.config.servers[server_name]
        log.warning(f"Attempting recover server {server_name}")
        if server_config.type == ServerType.STDIO: await self.server_manager.restart_server(server_name)
        elif server_config.type == ServerType.SSE: await self.server_manager.reconnect_server(server_name)

# --- RobustStdioSession Class ---
class RobustStdioSession(ClientSession):
    def __init__(self, process: asyncio.subprocess.Process, server_name: str):
        self._process = process
        self._server_name = server_name
        self._stdin = process.stdin
        self._stderr_reader_task: Optional[asyncio.Task] = None
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._request_id_counter = 0
        self._lock = asyncio.Lock()
        self._is_active = True
        self._background_task_runner: Optional[asyncio.Task] = None
        log.debug(f"[{self._server_name}] Initializing RobustStdioSession")
        self._background_task_runner = asyncio.create_task(self._run_reader_processor_wrapper(), name=f"session-reader-{server_name}")

    async def initialize(self, capabilities: Optional[Dict[str, Any]] = None, response_timeout: float = 60.0) -> Any:
        log.info(f"[{self._server_name}] Sending initialize request...")
        client_capabilities = capabilities if capabilities is not None else {}
        params = {"processId": os.getpid(), "clientInfo": {"name": "mcp-client-multi", "version": "2.0.0"},
                  "rootUri": None, "capabilities": client_capabilities, "protocolVersion": "2025-03-25"}
        result = await self._send_request("initialize", params, response_timeout=response_timeout)
        log.info(f"[{self._server_name}] Initialize request successful.")
        return result

    async def send_initialized_notification(self):
        if not self._is_active: 
            log.warning(f"[{self._server_name}] Session inactive, skip initialized.")
            return
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        try:
            notification_str = json.dumps(notification) + '\n'
            notification_bytes = notification_str.encode('utf-8')
            log.info(f"[{self._server_name}] Sending initialized notification...")
            if self._stdin is None or self._stdin.is_closing(): 
                raise ConnectionAbortedError("Stdin closed")
            self._stdin.write(notification_bytes)
            await self._stdin.drain()
            log.debug(f"[{self._server_name}] Initialized notification sent.")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e: 
            log.error(f"[{self._server_name}] Conn error sending initialized: {e}")
            await self._close_internal_state(e)
        except Exception as e: 
            log.error(f"[{self._server_name}] Error sending initialized: {e}", exc_info=True)

    async def _run_reader_processor_wrapper(self):
        close_exception: Optional[BaseException] = None
        try:
            log.debug(f"[{self._server_name}] Entering reader/processor task wrapper.")
            await self._read_and_process_stdout_loop()
            log.info(f"[{self._server_name}] Reader/processor task finished normally.")
        except asyncio.CancelledError: 
            log.debug(f"[{self._server_name}] Reader/processor task wrapper cancelled.")
            close_exception = asyncio.CancelledError("Reader task cancelled")
        except Exception as e: 
            log.error(f"[{self._server_name}] Reader/processor task wrapper failed: {e}", exc_info=True)
            close_exception = e
        finally:
            log.debug(f"[{self._server_name}] Reader/processor task wrapper exiting.")
            if self._is_active:
                log_level = logging.DEBUG if isinstance(close_exception, asyncio.CancelledError) else logging.WARNING
                log.log(log_level, f"[{self._server_name}] Reader/processor finished. Forcing close.")
                final_exception = close_exception or ConnectionAbortedError("Reader task finished unexpectedly")
                await self._close_internal_state(final_exception)

    async def _read_and_process_stdout_loop(self):
        handshake_complete = False
        stream_limit = getattr(self._process.stdout, '_limit', 'Unknown')
        log.debug(f"[{self._server_name}] Starting combined reader/processor loop (Buffer limit: {stream_limit}).")
        try:
            while self._process.returncode is None:
                if not self._is_active: 
                    log.info(f"[{self._server_name}] Session inactive, exiting loop.")
                    break
                try:
                    line_bytes = await asyncio.wait_for(self._process.stdout.readline(), timeout=60.0)
                    if not line_bytes:
                        if self._process.stdout.at_eof(): 
                            log.warning(f"[{self._server_name}] Stdout EOF.")
                            break
                        else: 
                            log.debug(f"[{self._server_name}] readline() timeout.")
                            continue
                    line_str_raw = line_bytes.decode('utf-8', errors='replace')
                    if USE_VERBOSE_SESSION_LOGGING: log.debug(f"[{self._server_name}] READ/PROC RAW <<< {repr(line_str_raw)}")
                    line_str = line_str_raw.strip()
                    if not line_str: continue
                    try:
                        message = json.loads(line_str)
                        is_valid_rpc = (isinstance(message, dict) and message.get("jsonrpc") == "2.0" and ('id' in message or 'method' in message))
                        if not is_valid_rpc: 
                            log.debug(f"[{self._server_name}] Skipping non-MCP JSON: {line_str[:100]}...")
                            continue
                        if not handshake_complete: 
                            log.info(f"[{self._server_name}] First valid JSON-RPC detected.")
                            handshake_complete = True
                        msg_id = message.get("id")
                        if msg_id is not None:
                            str_msg_id = str(msg_id)
                            future = self._response_futures.pop(str_msg_id, None)
                            if future and not future.done():
                                if "result" in message: 
                                    log.debug(f"[{self._server_name}] READ/PROC: Resolving future ID {msg_id} with RESULT.")
                                    future.set_result(message["result"])
                                elif "error" in message:
                                    err_data = message["error"]
                                    err_msg = f"Server error ID {msg_id}: {err_data.get('message', 'Unk')} (Code: {err_data.get('code', 'N/A')})"
                                    if err_data.get('data'): err_msg += f" Data: {repr(err_data.get('data'))[:100]}..."
                                    log.warning(f"[{self._server_name}] READ/PROC: Resolving future ID {msg_id} with ERROR: {err_msg}")
                                    future.set_exception(RuntimeError(err_msg))
                                else: 
                                    log.error(f"[{self._server_name}] READ/PROC: Invalid response format ID {msg_id}.")
                                    future.set_exception(RuntimeError(f"Invalid response format ID {msg_id}"))
                            elif future: 
                                log.debug(f"[{self._server_name}] READ/PROC: Future for ID {msg_id} already done.")
                            else: log.warning(f"[{self._server_name}] READ/PROC: Received response for unknown/timed-out ID: {msg_id}.")
                        elif "method" in message:
                            method_name = message['method']
                            log.debug(f"[{self._server_name}] READ/PROC: Received server message: {method_name}")
                            # Handle notifications/requests here if needed
                        else: 
                            log.warning(f"[{self._server_name}] READ/PROC: Unknown message structure: {repr(message)}")
                    except json.JSONDecodeError: log.debug(f"[{self._server_name}] Skipping noisy line: {line_str[:100]}...")
                    except Exception as proc_err: 
                        log.error(f"[{self._server_name}] Error processing line '{line_str[:100]}...': {proc_err}", exc_info=True)
                except asyncio.TimeoutError: 
                    log.debug(f"[{self._server_name}] Outer timeout reading stdout.")
                    continue
                except (BrokenPipeError, ConnectionResetError): 
                    log.warning(f"[{self._server_name}] Stdout pipe broken.")
                    break
                except ValueError as e:
                     if "longer than limit" in str(e) or "too long" in str(e): 
                         log.error(f"[{self._server_name}] Buffer limit ({stream_limit}) exceeded!", exc_info=True)
                     else: 
                         log.error(f"[{self._server_name}] ValueError reading stdout: {e}", exc_info=True)
                     break
                except Exception as read_err: 
                    log.error(f"[{self._server_name}] Error reading stdout: {read_err}", exc_info=True)
                    break
            log.info(f"[{self._server_name}] Exiting combined reader/processor loop.")
        except asyncio.CancelledError: 
            log.info(f"[{self._server_name}] Reader/processor loop cancelled.")
            raise
        except Exception as loop_err: 
            log.error(f"[{self._server_name}] Unhandled error in reader/processor loop: {loop_err}", exc_info=True)
            raise

    async def _send_request(self, method: str, params: Dict[str, Any], response_timeout: float) -> Any:
        if not self._is_active or (self._process and self._process.returncode is not None): raise ConnectionAbortedError("Session inactive or process terminated")
        async with self._lock: 
            self._request_id_counter += 1
            request_id = str(self._request_id_counter)
        request = {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._response_futures[request_id] = future
        try:
            request_str = json.dumps(request) + '\n'
            request_bytes = request_str.encode('utf-8')
            log.debug(f"[{self._server_name}] SEND: ID {request_id} ({method}): {request_bytes.decode('utf-8', errors='replace')[:100]}...")
            if self._stdin is None or self._stdin.is_closing(): 
                raise ConnectionAbortedError("Stdin closed")
            if USE_VERBOSE_SESSION_LOGGING: 
                log.debug(f"[{self._server_name}] RAW >>> {repr(request_bytes)}")
            self._stdin.write(request_bytes)
            await self._stdin.drain()
            if USE_VERBOSE_SESSION_LOGGING: log.info(f"[{self._server_name}] Drain complete for ID {request_id}.")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            log.error(f"[{self._server_name}] SEND FAIL ID {request_id}: Pipe broken: {e}")
            self._response_futures.pop(request_id, None)
            if not future.done(): future.set_exception(e)
            raise ConnectionAbortedError(f"Conn lost sending ID {request_id}: {e}") from e
        except Exception as e:
            log.error(f"[{self._server_name}] SEND FAIL ID {request_id}: {e}", exc_info=True)
            self._response_futures.pop(request_id, None)
            if not future.done(): future.set_exception(e)
            raise RuntimeError(f"Failed to send ID {request_id}: {e}") from e
        try:
            log.debug(f"[{self._server_name}] WAIT: Waiting for future ID {request_id} ({method}) (timeout={response_timeout}s)")
            result = await asyncio.wait_for(future, timeout=response_timeout)
            log.debug(f"[{self._server_name}] WAIT: Future resolved for ID {request_id}. Result received.")
            return result
        except asyncio.TimeoutError as timeout_error:
            log.error(f"[{self._server_name}] WAIT: Timeout waiting for future ID {request_id} ({method})")
            self._response_futures.pop(request_id, None)
            raise RuntimeError(f"Timeout waiting response {method} ID {request_id}") from timeout_error
        except asyncio.CancelledError: 
            log.debug(f"[{self._server_name}] WAIT: Wait cancelled ID {request_id} ({method}).")
            self._response_futures.pop(request_id, None)
            raise
        except Exception as wait_err:
            if future.done() and future.exception(): 
                server_error = future.exception()
                log.warning(f"[{self._server_name}] WAIT: Future ID {request_id} failed server error: {server_error}")
                raise server_error from wait_err
            else: 
                log.error(f"[{self._server_name}] WAIT: Error waiting future ID {request_id}: {wait_err}", exc_info=True)
                self._response_futures.pop(request_id, None)
                raise RuntimeError(f"Error processing response {method} ID {request_id}: {wait_err}") from wait_err

    # MCP Method implementations (rely on _send_request)
    async def list_tools(self, response_timeout: float = 40.0) -> ListToolsResult:
        log.debug(f"[{self._server_name}] Calling list_tools")
        result = await self._send_request("tools/list", {}, response_timeout)
        return ListToolsResult(**result)
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], response_timeout: float = 250.0) -> CallToolResult:
        log.debug(f"[{self._server_name}] Calling call_tool: {tool_name}")
        params = {"name": tool_name, "arguments": arguments}
        result = await self._send_request("tools/call", params, response_timeout)
        return CallToolResult(**result)
    async def list_resources(self, response_timeout: float = 40.0) -> ListResourcesResult:
        log.debug(f"[{self._server_name}] Calling list_resources")
        result = await self._send_request("resources/list", {}, response_timeout)
        return ListResourcesResult(**result)
    async def read_resource(self, uri: AnyUrl, response_timeout: float = 30.0) -> ReadResourceResult:
        log.debug(f"[{self._server_name}] Calling read_resource: {uri}")
        params = {"uri": str(uri)}
        result = await self._send_request("resources/read", params, response_timeout)
        return ReadResourceResult(**result)
    async def list_prompts(self, response_timeout: float = 40.0) -> ListPromptsResult:
        log.debug(f"[{self._server_name}] Calling list_prompts")
        result = await self._send_request("prompts/list", {}, response_timeout)
        return ListPromptsResult(**result)
    async def get_prompt(self, prompt_name: str, variables: Dict[str, Any], response_timeout: float = 30.0) -> GetPromptResult:
        log.debug(f"[{self._server_name}] Calling get_prompt: {prompt_name}")
        params = {"name": prompt_name, "arguments": variables}
        result = await self._send_request("prompts/get", params, response_timeout)
        return GetPromptResult(**result)

    async def _close_internal_state(self, exception: Exception):
        if not self._is_active: return
        self._is_active = False
        log.debug(f"[{self._server_name}] Closing internal state due to: {exception}")
        await self._cancel_pending_futures(exception)

    async def _cancel_pending_futures(self, exception: Exception):
        log.debug(f"[{self._server_name}] Cancelling {len(self._response_futures)} pending futures with: {exception}")
        futures_to_cancel = list(self._response_futures.items())
        self._response_futures.clear()
        for _, future in futures_to_cancel:
            if future and not future.done(): await suppress(asyncio.InvalidStateError)(future.set_exception(exception))

    async def aclose(self):
        log.info(f"[{self._server_name}] Closing RobustStdioSession...")
        if not self._is_active: 
            log.debug(f"[{self._server_name}] Already closed.")
            return
        await self._close_internal_state(ConnectionAbortedError("Session closed by client"))
        if self._background_task_runner and not self._background_task_runner.done():
            log.debug(f"[{self._server_name}] Cancelling reader task...")
            self._background_task_runner.cancel()
            await suppress(asyncio.CancelledError)(self._background_task_runner)
        if self._stderr_reader_task and not self._stderr_reader_task.done():
            log.debug(f"[{self._server_name}] Cancelling stderr task...")
            self._stderr_reader_task.cancel()
            await suppress(asyncio.CancelledError)(self._stderr_reader_task)
        if self._process and self._process.returncode is None:
            log.info(f"[{self._server_name}] Terminating process PID {self._process.pid} during aclose...")
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=2.0)
                if self._process.returncode is None:
                    log.debug(f"[{self._server_name}] Killing process")
                    self._process.kill()
                    await asyncio.wait_for(self._process.wait(), timeout=1.0)
            except ProcessLookupError: pass
            except Exception as e: log.error(f"Terminate/kill error: {e}")
        log.info(f"[{self._server_name}] RobustStdioSession closed.")

class ServerManager:
    def __init__(self, config: Config, tool_cache=None, safe_printer=None):
        self.config = config
        self.exit_stack = AsyncExitStack() # Manages lifecycles of connected sessions/processes
        self.active_sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self.tool_cache = tool_cache
        self._safe_printer = safe_printer or get_safe_console().print # Default to safe print
        self.monitor = ServerMonitor(self)
        self.registry = ServerRegistry() if config.enable_registry else None
        self.registered_services: Dict[str, ServiceInfo] = {} # name -> ServiceInfo
        self._session_tasks: Dict[str, List[asyncio.Task]] = {} # Tracks background tasks per session (e.g., stderr reader)
        self.sanitized_to_original: Dict[str, str] = {} # tool name mapping
        self._port_scan_client: Optional[httpx.AsyncClient] = None
        self.discovered_servers_cache: List[Dict[str, Any]] = [] # Cache for API/CLI
        self._discovery_in_progress = asyncio.Lock()
        log.info("ServerManager initialized.")

    async def close(self):
        """Gracefully shut down the server manager and release resources."""
        log.info("Closing ServerManager...")
        # 1. Stop Monitor
        if hasattr(self, 'monitor') and self.monitor:
            await self.monitor.stop_monitoring()
        # 2. Close Registry (handles mDNS unregister and HTTP client)
        if hasattr(self, 'registry') and self.registry:
            await self.registry.close()
        # 3. Close the main ExitStack (this manages session/process cleanup)
        log.debug(f"Closing ServerManager AsyncExitStack (cleans up {len(self.active_sessions)} sessions)...")
        await self.exit_stack.aclose()
        log.debug("ServerManager AsyncExitStack closed.")
        # 4. Clear internal state (optional, as instance is being destroyed)
        self.active_sessions.clear()
        self.processes.clear()
        self.tools.clear()
        self.resources.clear()
        self.prompts.clear()
        self._session_tasks.clear()
        self.registered_services.clear()
        log.info("ServerManager closed.")
        
    @property
    def tools_by_server(self) -> Dict[str, List[MCPTool]]:
        """Group tools by server name for easier lookup."""
        result: Dict[str, List[MCPTool]] = {}
        for tool in self.tools.values():
            result.setdefault(tool.server_name, []).append(tool)
        return result

    # --- Discovery Methods ---

    async def _discover_local_servers(self):
        """Discover MCP servers in local filesystem paths."""
        discovered_local = []
        fs_paths = self.config.discovery_paths or []
        log.debug(f"Discovering local servers in paths: {fs_paths}")

        for base_path_str in fs_paths:
            base_path = Path(os.path.expanduser(base_path_str))
            if not base_path.exists() or not base_path.is_dir():
                log.debug(f"Skipping non-existent/invalid discovery path: {base_path}")
                continue

            log.info(f"Scanning for servers in {base_path}...")
            # Look for python and js files (potential STDIO servers)
            # Enhanced check: Look for 'mcp' and common server names
            potential_server_files = []
            try:
                for ext in ['.py', '.js', '.sh', '.bash', '.exe']: # Added shell scripts and exe
                    for item in base_path.rglob(f"*{ext}"): # Recursive glob
                        if item.is_file():
                            # Heuristic check - adjust as needed
                            fname_lower = item.name.lower()
                            if 'mcp' in fname_lower or 'server' in fname_lower or 'agent' in fname_lower:
                                potential_server_files.append(item)
            except PermissionError as e:
                log.warning(f"Permission error scanning {base_path}: {e}")
                continue
            except Exception as e:
                log.error(f"Error scanning filesystem path {base_path}: {e}", exc_info=True)
                continue

            log.debug(f"Found {len(potential_server_files)} potential server files in {base_path}.")

            for file_path in potential_server_files:
                try:
                    path_str = str(file_path.resolve())
                    name = file_path.stem # Use filename without extension as default name
                    server_type = "stdio" # Assume stdio for local files

                    # Avoid adding duplicates already in config by path
                    if any(s.path == path_str for s in self.config.servers.values()):
                        log.debug(f"Skipping discovery of '{name}' at '{path_str}' - already configured.")
                        continue

                    # Add to results if not already found via this method this run
                    if not any(d[1] == path_str for d in discovered_local):
                        discovered_local.append((name, path_str, server_type))
                        log.debug(f"Discovered potential local server: {name} ({path_str})")
                except Exception as file_proc_err:
                    log.warning(f"Error processing potential server file {file_path}: {file_proc_err}")

        # Store results on self for _process_discovery_results
        self._discovered_local = discovered_local
        log.info(f"Filesystem Discovery: Found {len(discovered_local)} potential new servers.")

    async def _discover_registry_servers(self):
        """Discover MCP servers from remote registries."""
        discovered_remote = []
        if not self.registry:
            log.info("Registry client not available, skipping remote discovery.")
            self._discovered_remote = discovered_remote
            return

        try:
            remote_servers = await self.registry.discover_remote_servers()
            configured_urls = {s.path for s in self.config.servers.values() if s.type == ServerType.SSE}

            for server_data in remote_servers:
                name = server_data.get("name", "")
                url = server_data.get("url", "")
                server_type_str = server_data.get("type", "sse").lower() # Assume sse if missing

                if not name or not url or server_type_str != "sse":
                    log.warning(f"Skipping invalid registry server entry: {server_data}")
                    continue

                if url in configured_urls:
                    log.debug(f"Skipping registry discovery of '{name}' at '{url}' - already configured.")
                    continue

                version_str = server_data.get("version")
                version = ServerVersion.from_string(version_str) if version_str else None
                categories = server_data.get("categories", [])
                rating = float(server_data.get("rating", 5.0))
                registry_url = server_data.get("registry_url") # Get source registry

                if not any(d[1] == url for d in discovered_remote):
                    discovered_remote.append((name, url, server_type_str, version, categories, rating, registry_url))
                    log.debug(f"Discovered registry server: {name} ({url})")

        except Exception as e:
            log.error(f"Error during registry discovery: {e}", exc_info=True)

        self._discovered_remote = discovered_remote
        log.info(f"Registry Discovery: Found {len(discovered_remote)} potential new servers.")

    async def _discover_mdns_servers(self):
        """Discover MCP servers on the local network using mDNS."""
        discovered_mdns = []
        if not self.registry or not self.config.enable_local_discovery:
            log.info("mDNS discovery disabled or registry unavailable.")
            self._discovered_mdns = discovered_mdns
            return

        if not self.registry.zeroconf or not self.registry.browser:
            log.info("Starting mDNS listener for discovery...")
            self.registry.start_local_discovery()
            # Give it a few seconds to find initial services
            await asyncio.sleep(3)
        else:
            log.info("mDNS listener already running. Checking current discoveries.")
            # For simplicity, we use the current cache populated by the listener.
            pass

        configured_urls = {s.path for s in self.config.servers.values()}

        # Process servers currently known to the listener
        for name, server_info in list(self.registry.discovered_servers.items()):
            path_or_url = server_info.get("url") # Use URL for SSE, path placeholder for others?
            server_type = server_info.get("type", "sse").lower()

            if not path_or_url:
                continue

            # Skip if already configured by path/url
            if path_or_url in configured_urls:
                log.debug(f"Skipping mDNS discovery of '{name}' at '{path_or_url}' - already configured.")
                continue

            version = server_info.get("version") # Already parsed ServerVersion object or None
            categories = server_info.get("categories", [])
            description = server_info.get("description", "")

            if not any(d[1] == path_or_url for d in discovered_mdns):
                discovered_mdns.append((name, path_or_url, server_type, version, categories, description))
                log.debug(f"Discovered mDNS server: {name} ({path_or_url})")

        self._discovered_mdns = discovered_mdns
        log.info(f"mDNS Discovery: Found {len(discovered_mdns)} potential new servers.")

    async def _probe_port(self, ip_address: str, port: int, probe_timeout: float, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        """
        Attempts to detect an MCP SSE server on a given IP:Port.
        Returns server details dict if MCP pattern detected, None otherwise.
        """
        tcp_check_timeout = 0.25 # Slightly longer TCP timeout
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(ip_address, port), timeout=tcp_check_timeout)
            writer.close()
            await writer.wait_closed()
            if USE_VERBOSE_SESSION_LOGGING:
                log.debug(f"TCP Port open: {ip_address}:{port}. Proceeding.")
            await asyncio.sleep(0.05)
        except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
            return None
        except Exception as e:
            log.warning(f"TCP pre-check error {ip_address}:{port}: {e}")
            return None

        base_url = f"http://{ip_address}:{port}"
        sse_url = f"{base_url}/sse"
        mcp_server_found = False
        server_details = None

        # --- Attempt GET /sse (Streaming Check) ---
        try:
            if USE_VERBOSE_SESSION_LOGGING:
                log.debug(f"Probing GET {sse_url} (timeout={probe_timeout}s)...")
            async with client.stream("GET", sse_url, timeout=probe_timeout) as response:
                if response.status_code == 200 and response.headers.get("content-type", "").lower().startswith("text/event-stream"):
                    log.info(f"MCP SSE detected via GET /sse at {sse_url}")
                    server_name = f"mcp-scan-{ip_address}-{port}-sse"
                    server_details = {"name": server_name, "path": sse_url, "type": ServerType.SSE, "args": [], "description": f"SSE server (GET /sse) on {ip_address}:{port}", "source": "portscan"}
                    mcp_server_found = True # Found it, no need for POST
                elif USE_VERBOSE_SESSION_LOGGING:
                    log.debug(f"GET {sse_url} failed: Status={response.status_code}, Type={response.headers.get('content-type')}")
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError, asyncio.TimeoutError) as http_err:
            if USE_VERBOSE_SESSION_LOGGING:
                log.debug(f"GET {sse_url} error: {type(http_err).__name__}")
        except Exception as e:
            log.warning(f"Unexpected error GET {sse_url}: {e}")

        # --- Attempt POST / initialize ---
        if not mcp_server_found:
            initialize_payload = {"jsonrpc": "2.0", "method": "initialize", "params": {"protocolVersion": "2025-03-25"}, "id": str(uuid.uuid4())}
            try:
                if USE_VERBOSE_SESSION_LOGGING:
                    log.debug(f"Probing POST {base_url} (timeout={probe_timeout}s)...")
                response = await client.post(base_url, json=initialize_payload, timeout=probe_timeout)
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        if isinstance(response_data, dict) and response_data.get("jsonrpc") == "2.0":
                            log.info(f"MCP detected via POST initialize at {base_url}")
                            server_name = f"mcp-scan-{ip_address}-{port}-post"
                            server_details = {"name": server_name, "path": base_url, "type": ServerType.SSE, "args": [], "description": f"SSE server (POST /) on {ip_address}:{port}", "source": "portscan"}
                        elif USE_VERBOSE_SESSION_LOGGING:
                            log.debug(f"Non-MCP JSON from POST {base_url}: {str(response_data)[:100]}...")
                    except json.JSONDecodeError:
                        if USE_VERBOSE_SESSION_LOGGING:
                            log.debug(f"Non-JSON response at POST {base_url}")
                elif USE_VERBOSE_SESSION_LOGGING:
                    log.debug(f"POST {base_url} failed: Status={response.status_code}")
            except (httpx.TimeoutException, httpx.RequestError, asyncio.TimeoutError) as http_err:
                if USE_VERBOSE_SESSION_LOGGING:
                    log.debug(f"POST {base_url} error: {type(http_err).__name__}")
            except Exception as e:
                log.warning(f"Unexpected error POST {base_url}: {e}")

        if server_details is None and USE_VERBOSE_SESSION_LOGGING:
            log.debug(f"TCP Port {ip_address}:{port} open, but no MCP pattern detected.")
        return server_details

    async def _discover_port_scan(self):
        """Scan configured local IP addresses and port ranges for MCP SSE servers."""
        if not self.config.enable_port_scanning:
            log.info("Port scanning discovery disabled by configuration.")
            self._discovered_port_scan = []
            return

        # --- Get Config ---
        start_port = self.config.port_scan_range_start
        end_port = self.config.port_scan_range_end
        concurrency = max(1, self.config.port_scan_concurrency) # Ensure at least 1
        probe_timeout = max(0.1, self.config.port_scan_timeout) # Ensure minimum timeout
        targets = self.config.port_scan_targets

        if start_port > end_port or start_port < 0 or end_port > 65535:
            log.error(f"Invalid port range: {start_port}-{end_port}. Skipping scan.")
            self._discovered_port_scan = []
            return
        if not targets:
            log.warning("No targets configured for port scanning. Skipping scan.")
            self._discovered_port_scan = []
            return

        log.info(f"Starting port scan: Ports [{start_port}-{end_port}], Targets={targets}, Concurrency={concurrency}, Timeout={probe_timeout}s")

        discovered_servers_to_add = []
        semaphore = asyncio.Semaphore(concurrency)
        # Use a shared client for efficiency
        client_timeout = httpx.Timeout(probe_timeout + 0.5, connect=probe_timeout) # Slightly larger overall timeout
        self._port_scan_client = httpx.AsyncClient(verify=False, timeout=client_timeout, limits=httpx.Limits(max_connections=concurrency + 10))

        try:
            total_ports_to_scan = (end_port - start_port + 1) * len(targets)
            log.info(f"Scanning {total_ports_to_scan} total IP:Port combinations.")

            async def bound_probe(ip: str, port: int):
                async with semaphore:
                    return await self._probe_port(ip, port, probe_timeout, self._port_scan_client) # type: ignore

            probe_coroutines = [bound_probe(ip, port) for ip in targets for port in range(start_port, end_port + 1)]
            results = await asyncio.gather(*probe_coroutines, return_exceptions=True)

        except Exception as gather_err:
            log.error(f"Error during port scan task gathering: {gather_err}", exc_info=True)
            results = [] # Ensure results is empty on error
        finally:
            # Close the shared client
            if self._port_scan_client:
                await self._port_scan_client.aclose()
                self._port_scan_client = None
            # Release semaphore if needed (though gather should handle this)
            # Log summary before processing
            mcp_found_count = sum(1 for r in results if isinstance(r, dict))
            error_count = sum(1 for r in results if isinstance(r, Exception))
            log.info(f"Port scan finished. Found {mcp_found_count} potential MCP endpoints. Encountered {error_count} errors during probing.")

        # Process results
        configured_urls = {s.path for s in self.config.servers.values()}
        for result in results:
            if isinstance(result, dict):
                server_path = result.get("path")
                if server_path and server_path not in configured_urls:
                    if not any(d[1] == server_path for d in discovered_servers_to_add):
                        discovered_servers_to_add.append((
                             result.get("name", "Unknown"), server_path, ServerType.SSE.value, # Type is always SSE from probe
                             None, [], result.get("description", "") # No version/categories from scan
                         ))
                        log.debug(f"Adding port scan result: {result.get('name')} ({server_path})")
                elif server_path and server_path in configured_urls:
                    log.debug(f"Skipping port scan discovery of '{server_path}' - already configured.")
            elif isinstance(result, Exception):
                if not isinstance(result, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout, asyncio.TimeoutError, ConnectionRefusedError, OSError)):
                    log.warning(f"Port probe task failed with unexpected exception: {result}")
                elif USE_VERBOSE_SESSION_LOGGING:
                    log.debug(f"Port probe task failed with expected exception: {type(result).__name__}")

        self._discovered_port_scan = discovered_servers_to_add
        log.info(f"Port Scan Discovery: Added {len(self._discovered_port_scan)} potential new servers to results.")

    async def discover_servers(self):
        """
        Orchestrates the discovery of MCP servers via all configured methods.
        Populates self.discovered_servers_cache.
        """
        async with self._discovery_in_progress:
            log.info("Starting server discovery process...")
            self.discovered_servers_cache = [] # Reset cache
            # Reset internal lists
            self._discovered_local = []
            self._discovered_remote = []
            self._discovered_mdns = []
            self._discovered_port_scan = []

            discovery_steps = []
            discovery_descriptions = []

            # Filesystem
            if self.config.auto_discover:
                discovery_steps.append(self._discover_local_servers)
                discovery_descriptions.append(f"{EMOJI_MAP['search']} Discovering local files...")
            else:
                log.debug("Skipping filesystem discovery (disabled).")

            # Registry
            if self.config.enable_registry and self.registry:
                discovery_steps.append(self._discover_registry_servers)
                discovery_descriptions.append(f"{EMOJI_MAP['search']} Discovering registry servers...")
            else:
                log.debug("Skipping registry discovery (disabled or unavailable).")

            # mDNS
            if self.config.enable_local_discovery and self.registry:
                discovery_steps.append(self._discover_mdns_servers)
                discovery_descriptions.append(f"{EMOJI_MAP['search']} Discovering local network (mDNS)...")
            else:
                log.debug("Skipping mDNS discovery (disabled or unavailable).")

            # Port Scan
            if self.config.enable_port_scanning:
                discovery_steps.append(self._discover_port_scan)
                discovery_descriptions.append(f"{EMOJI_MAP['port']} Scanning local ports...")
            else:
                log.debug("Skipping port scan discovery (disabled).")

            # Execute steps concurrently using the helper
            if discovery_steps:
                log.info(f"Running {len(discovery_steps)} discovery steps...")
                try:
                    # Use the helper method from MCPClient (assuming it's available via app.mcp_client)
                    # If running standalone, this needs adjustment.
                    if hasattr(app, "mcp_client") and hasattr(app.mcp_client, "run_multi_step_task"):
                        await app.mcp_client.run_multi_step_task(
                            steps=discovery_steps,
                            step_descriptions=discovery_descriptions,
                            title=f"{EMOJI_MAP['search']} Discovering MCP servers..."
                        )
                    else:
                        # Fallback: run sequentially if helper not available
                        log.warning("MCPClient helper not found, running discovery steps sequentially.")
                        for step, desc in zip(discovery_steps, discovery_descriptions, strict=True):
                            log.info(f"Running: {desc}")
                            await step()

                    log.info("Discovery steps execution complete.")
                except Exception as discover_err:
                    log.error(f"Error during multi-step discovery: {discover_err}", exc_info=True)
            else:
                log.info("No discovery methods enabled.")

            # --- Process All Results and Populate Cache ---
            log.info("Processing discovery results and populating cache...")
            configured_paths = {s.path for s in self.config.servers.values()}

            # Helper to add to cache, checking for duplicates within this run
            processed_paths_this_run = set()
            def add_to_cache(source, name, type_str, path_or_url, version=None, categories=None, description=None):
                if path_or_url in processed_paths_this_run:
                    return # Already added this path/url in this discovery run
                processed_paths_this_run.add(path_or_url)
                is_conf = path_or_url in configured_paths
                item = {
                    "name": name, "type": type_str, "path_or_url": path_or_url, "source": source,
                    "description": description, "is_configured": is_conf,
                    "version": str(version) if version else None, "categories": categories or [],
                }
                self.discovered_servers_cache.append(item)

            # Process Local
            for name, path, type_str in getattr(self, '_discovered_local', []):
                add_to_cache("filesystem", name, type_str, path, description=f"Discovered {type_str} server")
            # Process Remote
            for name, url, type_str, version, categories, rating, registry_url in getattr(self, '_discovered_remote', []):
                desc = f"From registry (Rating: {rating:.1f})" + (f" @ {registry_url}" if registry_url else "")
                add_to_cache("registry", name, type_str, url, version, categories, desc)
            # Process mDNS
            for name, url, type_str, version, categories, description in getattr(self, '_discovered_mdns', []):
                add_to_cache("mdns", name, type_str, url, version, categories, description or "Discovered on local network")
            # Process Port Scan
            for name, url, type_str, version, categories, description in getattr(self, '_discovered_port_scan', []):
                add_to_cache("portscan", name, type_str, url, version, categories, description or f"Discovered via port scan")

            log.info(f"Discovery complete. Cached {len(self.discovered_servers_cache)} potential servers.")

    async def get_discovery_results(self) -> List[Dict]:
        """Returns the cached discovery results."""
        # Return a copy to prevent external modification
        return copy.deepcopy(self.discovered_servers_cache)

    async def _process_discovery_results(self, interactive_mode: bool):
        """
        Processes cached discovery results: prompts user to add if interactive,
        otherwise logs discovered servers.
        """
        safe_console = get_safe_console()
        discovered_servers = self.discovered_servers_cache # Use the cached results
        newly_discovered = [s for s in discovered_servers if not s["is_configured"]]

        if not newly_discovered:
            log.info("No *new* unconfigured servers found during discovery processing.")
            if discovered_servers: # Still log if *any* were found, even if configured
                 log.info(f"Total discovered (including configured): {len(discovered_servers)}")
            return

        safe_console.print(f"\n[bold green]{EMOJI_MAP['success']} Discovered {len(newly_discovered)} new potential MCP servers:[/]")

        # --- Display Table ---
        table = Table(title="Newly Discovered Servers", box=box.ROUNDED)
        table.add_column("Index")
        table.add_column("Name")
        table.add_column("Source")
        table.add_column("Type")
        table.add_column("Path/URL")
        table.add_column("Description")
        for i, server in enumerate(newly_discovered, 1):
             table.add_row(
                 f"({i})", server['name'], server['source'], server['type'],
                 server['path_or_url'], server['description'] or ""
             )
        safe_console.print(table)

        # --- Add Logic ---
        added_count = 0
        servers_added_this_run = [] # Track names of successfully added servers

        if interactive_mode:
            if Confirm.ask("\nAdd selected discovered servers to configuration?", default=True, console=safe_console):
                servers_to_add_indices = set() # Initialize empty set

                # --- Auto-select if only one server ---
                if len(newly_discovered) == 1:
                    safe_console.print("[dim]Only one server found, selecting index 1 automatically.[/dim]")
                    servers_to_add_indices = {0} # Auto-select the first (and only) index
                # --- End Auto-select ---
                else: # More than one server, ask for selection
                    while True:
                        choice = Prompt.ask("Enter number(s) to add (e.g., 1, 3-5), 'all', or 'none'", default="none", console=safe_console)
                        if choice.lower() == 'none':
                            break # Exit loop if user chooses none
                        servers_to_add_indices = set() # Reset for each attempt inside the loop
                        try:
                            if choice.lower() == 'all':
                                servers_to_add_indices = set(range(len(newly_discovered)))
                            else:
                                parts = choice.replace(" ", "").split(',')
                                for part in parts:
                                    if '-' in part:
                                        range_parts = part.split('-')
                                        if len(range_parts) != 2: raise ValueError("Invalid range format")
                                        start, end = map(int, range_parts)
                                        if 1 <= start <= end <= len(newly_discovered):
                                            servers_to_add_indices.update(range(start - 1, end))
                                        else:
                                            raise ValueError("Range index out of bounds")
                                    else:
                                        idx = int(part)
                                        if 1 <= idx <= len(newly_discovered):
                                            servers_to_add_indices.add(idx - 1)
                                        else:
                                            raise ValueError("Index out of bounds")
                            break # Valid selection processed, exit the while True loop
                        except ValueError as e:
                            safe_console.print(f"[red]Invalid selection: {e}. Please try again.[/]")
                        # Loop continues if selection was invalid

                # Process selected servers if any were chosen
                if servers_to_add_indices:
                    for idx in sorted(servers_to_add_indices):
                        server_info = newly_discovered[idx]
                        success, msg = await self.add_discovered_server_config(server_info)
                        if success:
                            added_count += 1
                            # Store the *actual* name added (might be renamed)
                            added_config = next((s for s in self.config.servers.values() if s.path == server_info["path_or_url"]), None)
                            if added_config:
                                servers_added_this_run.append(added_config.name)
                            else:
                                log.warning(f"Could not find added server config for path {server_info['path_or_url']} after successful add.")
                            # --- REMOVED individual connection prompt ---
                            # if Confirm.ask(f"Connect to added server '{server_info['name']}' now?", default=False, console=safe_console):
                            #      await self.connect_to_server_by_name(server_info['name']) # Use actual name added
                        else:
                            safe_console.print(f"[red]Failed to add '{server_info['name']}': {msg}[/]")
            else:
                safe_console.print("[yellow]No servers added.[/]")

        else: # Non-interactive: Just log
             log.info(f"Non-interactive mode. Found {len(newly_discovered)} new servers. Add manually or via API.")

        # --- Save config and offer to connect *after* processing all selections ---
        if added_count > 0:
            await self.config.save_async() # Save YAML changes
            safe_console.print(f"\n[green]{EMOJI_MAP['success']} Added {added_count} server(s) to configuration.[/]")

            # --- Offer to connect to the group ---
            if servers_added_this_run and Confirm.ask(f"Connect to the {len(servers_added_this_run)} newly added server(s) now?", default=True, console=safe_console):
                log.info(f"Attempting to connect to newly added servers: {servers_added_this_run}")
                connect_tasks = [self.connect_to_server_by_name(name) for name in servers_added_this_run]
                results = await asyncio.gather(*connect_tasks, return_exceptions=True)
                success_conn = sum(1 for r in results if isinstance(r, bool) and r)
                fail_conn = len(results) - success_conn
                safe_console.print(f"[cyan]Connection attempt finished. Success: {success_conn}, Failed: {fail_conn}[/]")
            # --- End group connection ---

    async def add_discovered_server_config(self, server_info: Dict) -> Tuple[bool, str]:
        """Adds a server from discovery results to the config ONLY."""
        name = server_info.get("name")
        path_or_url = server_info.get("path_or_url")
        type_str = server_info.get("type")

        if not all([name, path_or_url, type_str]):
            return False, "Invalid server data."
        if any(s.path == path_or_url for s in self.config.servers.values()):
            return False, f"Server with path/url '{path_or_url}' already configured."
        if name in self.config.servers:
            # Handle name conflict - maybe auto-rename?
            original_name = name
            count = 1
            while f"{original_name}-{count}" in self.config.servers:
                count += 1
            name = f"{original_name}-{count}"
            log.warning(f"Server name conflict for '{original_name}', renaming to '{name}'.")

        try:
            server_type = ServerType(type_str.lower())
            version_obj = None
            if server_info.get("version"):
                version_obj = ServerVersion.from_string(server_info["version"]) # type: ignore

            new_config = ServerConfig(
                name=name, type=server_type, path=path_or_url,
                enabled=True, auto_start=False, # Defaults for discovered
                description=server_info.get("description", f"Discovered via {server_info.get('source', 'unknown')}"),
                categories=server_info.get("categories", []),
                version=version_obj
            )
            self.config.servers[name] = new_config
            log.info(f"Added server '{name}' to config from discovery source '{server_info.get('source')}'.")
            return True, f"Server '{name}' added to configuration."
        except ValueError:
            return False, f"Invalid server type '{type_str}'."
        except Exception as e:
            log.error(f"Failed to create ServerConfig for {name}: {e}", exc_info=True)
            return False, f"Internal error creating config: {e}"

    async def add_and_connect_discovered_server(self, discovered_server_info: Dict) -> Tuple[bool, str]:
        """Adds a server from discovery results and attempts to connect."""
        name = discovered_server_info.get("name", "Unknown Discovered Server")
        path_or_url = discovered_server_info.get("path_or_url")

        # Check if already configured by path/url FIRST
        existing_server = next((s for s in self.config.servers.values() if s.path == path_or_url), None)
        if existing_server:
            log.info(f"Server '{path_or_url}' already configured as '{existing_server.name}'. Attempting connection.")
            if existing_server.name in self.active_sessions:
                return True, f"Server already configured as '{existing_server.name}' and connected."
            # Try connecting existing config
            success = await self.connect_to_server_by_name(existing_server.name)
            return success, f"Connected to existing server '{existing_server.name}'." if success else f"Failed to connect to existing server '{existing_server.name}'."

        # Not configured, try adding it first
        add_success, add_message = await self.add_discovered_server_config(discovered_server_info)
        if not add_success:
            return False, f"Failed to add server '{name}' to config: {add_message}"

        # Configuration added successfully, now try to connect using the (potentially renamed) name
        # Find the actual name added to the config (handles auto-rename)
        newly_added_config = next((s for s in self.config.servers.values() if s.path == path_or_url), None)
        if not newly_added_config:
            # This should not happen if add_discovered_server_config succeeded
            return False, f"Internal error: Cannot find newly added server config for path {path_or_url}."

        final_added_name = newly_added_config.name
        await self.config.save_async() # Save config *after* adding, before connecting
        log.info(f"Attempting to connect to newly added server '{final_added_name}'...")
        connect_success = await self.connect_to_server_by_name(final_added_name)

        return connect_success, f"Server '{final_added_name}' added and connected." if connect_success else f"Server '{final_added_name}' added but failed to connect."

    # --- Connection Logic ---

    @asynccontextmanager
    async def _manage_process_lifetime(self, process: asyncio.subprocess.Process, server_name: str):
        """Async context manager to ensure a process is terminated."""
        try:
            yield process
        finally:
            log.debug(f"[{server_name}] Cleaning up process context...")
            await self.terminate_process(server_name, process) # Use helper

    async def _connect_stdio_server(
        self, server_config: ServerConfig, current_server_name: str, retry_count: int
    ) -> Tuple[Optional[ClientSession], Optional[Dict[str, Any]], Optional[asyncio.subprocess.Process]]:
        """Handles STDIO process start/reuse and initial handshake."""
        session: Optional[ClientSession] = None
        initialize_result_obj: Optional[Dict[str, Any]] = None
        process_this_attempt: Optional[asyncio.subprocess.Process] = None
        log_file_handle = None

        # === Process Start/Reuse Logic ===
        existing_process = self.processes.get(current_server_name)
        start_new_process = False
        process_to_use = None

        if existing_process:
            if existing_process.returncode is None:
                if retry_count > 0: # Only restart on retries
                    log.warning(f"[{current_server_name}] Restarting process on retry {retry_count}")
                    await self.terminate_process(current_server_name, existing_process)
                    start_new_process = True
                else: # First attempt, try to reuse existing RUNNING process
                    log.debug(f"[{current_server_name}] Reusing existing process PID {existing_process.pid}")
                    process_to_use = existing_process
            else: # Process existed but already terminated
                log.warning(f"[{current_server_name}] Previous process exited code {existing_process.returncode}. Cleaning up.")
                if current_server_name in self.processes:
                    del self.processes[current_server_name]
                start_new_process = True
        else: # No process found, start new one
            start_new_process = True

        # --- Start Process If Needed ---
        if start_new_process:
            executable = server_config.path
            current_args = server_config.args
            log_file_path = (CONFIG_DIR / f"{current_server_name}_stderr.log").resolve()
            log_file_path.parent.mkdir(parents=True, exist_ok=True)
            log.info(f"[{current_server_name}] Executing STDIO: '{executable}' {current_args}. Stderr -> {log_file_path}")

            process: Optional[asyncio.subprocess.Process] = None
            try:
                # Ensure log file handle is managed
                log_file_handle = await self.exit_stack.enter_async_context(aiofiles.open(log_file_path, "ab")) # type: ignore
                stderr_fileno = log_file_handle.fileno()

                # Check for shell execution hint (simple check, might need refinement)
                is_shell_cmd = isinstance(executable, str) and executable in ["bash", "sh", "zsh"] and len(current_args) == 2 and current_args[0] == '-c'

                if is_shell_cmd:
                    command_string = current_args[1]
                    log.debug(f"[{current_server_name}] Starting via shell: {executable} -c '{command_string[:50]}...'")
                    process = await asyncio.create_subprocess_shell(
                        command_string, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=stderr_fileno, limit=2**22, env=os.environ.copy(), executable=executable
                    )
                else:
                    final_cmd_list = [executable] + current_args
                    log.debug(f"[{current_server_name}] Starting via exec: {final_cmd_list}")
                    process = await asyncio.create_subprocess_exec(
                        *final_cmd_list, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=stderr_fileno, limit=2**22, env=os.environ.copy()
                    )

                process_this_attempt = process
                self.processes[current_server_name] = process # Track the newly started process
                log.info(f"[{current_server_name}] Started process PID {process.pid}")

                await asyncio.sleep(0.5) # Allow startup time
                if process.returncode is not None:
                    raise RuntimeError(f"Process failed immediately (code {process.returncode}). Check log '{log_file_path}'.")
                process_to_use = process

            except FileNotFoundError:
                log.error(f"[{current_server_name}] Executable not found: {executable}")
                raise
            except Exception as proc_start_err:
                log.error(f"[{current_server_name}] Error starting process: {proc_start_err}", exc_info=True)
                if process and process.returncode is None:
                    await self.terminate_process(current_server_name, process) # Cleanup failed process
                if current_server_name in self.processes:
                    del self.processes[current_server_name]
                raise

        # === Session Creation and Handshake ===
        if not process_to_use or process_to_use.returncode is not None:
            raise RuntimeError(f"Process for STDIO server {current_server_name} is invalid or exited.")

        try:
            log.info(f"[{current_server_name}] Initializing RobustStdioSession...")
            session = RobustStdioSession(process_to_use, current_server_name)
            # No separate stderr reader needed with RobustStdioSession changes

            log.info(f"[{current_server_name}] Attempting MCP handshake (initialize)...")
            init_timeout = server_config.timeout + 10.0 # Slightly longer timeout for handshake
            initialize_result_obj = await asyncio.wait_for(
                session.initialize(response_timeout=server_config.timeout),
                timeout=init_timeout
            )
            log.info(f"[{current_server_name}] Initialize successful.")

            await session.send_initialized_notification()
            log.info(f"[{current_server_name}] Initialized notification sent.")

            # Return session, result, and the process that was actually used/started
            return session, initialize_result_obj, process_this_attempt

        except Exception as setup_err:
            log.warning(f"[{current_server_name}] Failed STDIO setup/handshake: {setup_err}", exc_info=True)
            # Ensure session is closed if created
            if session and hasattr(session, 'aclose'):
                await suppress(Exception)(session.aclose())
            # Process termination handled by outer loop's exit stack if process_this_attempt was set
            raise setup_err

    async def _load_server_capabilities(
        self, server_name: str, session: ClientSession, initialize_result: Optional[Union[MCPInitializeResult, Dict[str, Any]]]
    ):
        """
        Loads tools, resources, and prompts from a connected server session AFTER potential rename.
        Uses the *final* server_name for registration. Uses anyio for structured concurrency.
        Correctly handles both dict (STDIO) and ServerCapabilities object (MCPInitializeResult) types.
        """
        log.info(f"[{server_name}] Loading capabilities using anyio...")
        capability_timeout = 30.0 # Timeout for individual capability list calls

        # Determine if capabilities are advertised
        has_tools = False
        has_resources = False
        has_prompts = False

        server_caps_obj = None
        if isinstance(initialize_result, dict): # From STDIO direct response
             caps_data = initialize_result.get("capabilities")
             if isinstance(caps_data, dict):
                 # If STDIO returns a dict, we might need to map its keys
                 # Assuming simple boolean flags for now based on previous code
                 has_tools = bool(caps_data.get("tools"))
                 has_resources = bool(caps_data.get("resources"))
                 has_prompts = bool(caps_data.get("prompts"))
        elif isinstance(initialize_result, MCPInitializeResult): # From standard ClientSession result object
             # Access the capabilities attribute directly from the MCPInitializeResult object
             server_caps_obj = initialize_result.capabilities # This is the ServerCapabilities object

        # Check the ServerCapabilities object if obtained from MCPInitializeResult
        if server_caps_obj and hasattr(server_caps_obj, 'tools') and hasattr(server_caps_obj, 'resources') and hasattr(server_caps_obj, 'prompts'):
             has_tools = bool(server_caps_obj.tools)
             has_resources = bool(server_caps_obj.resources)
             has_prompts = bool(server_caps_obj.prompts)
        elif server_caps_obj:
            # Log if capabilities object exists but lacks expected fields
            log.warning(f"[{server_name}] InitializeResult.capabilities object present but lacks expected fields (tools, resources, prompts). Type: {type(server_caps_obj)}")
        else: # Add an else to catch the case where initialize_result was neither dict nor MCPInitializeResult
            log.warning(f"[{server_name}] Unexpected initialize_result type ({type(initialize_result)}), cannot determine capabilities.")
        
        log.debug(f"[{server_name}] Determined capabilities: tools={has_tools}, resources={has_resources}, prompts={has_prompts}")
        # Store determined capabilities in the config
        if server_name in self.config.servers:
            self.config.servers[server_name].capabilities = {
                "tools": has_tools,
                "resources": has_resources,
                "prompts": has_prompts,
            }

        # --- Define async functions for loading each capability type ---
        async def load_tools_task():
            if has_tools and hasattr(session, 'list_tools'):
                log.info(f"[{server_name}] Loading tools...")
                try:
                    res = await asyncio.wait_for(session.list_tools(), timeout=capability_timeout)
                    # Use helper to process results, handle potential None or incorrect type
                    self._process_list_result(server_name, getattr(res, 'tools', None), self.tools, MCPTool, "tool")
                except asyncio.TimeoutError:
                    log.error(f"[{server_name}] Timeout loading tools.")
                except McpError as e: # Catch specific MCP errors
                    log.error(f"[{server_name}] MCP Error loading tools: {e}")
                except Exception as e:
                    log.error(f"[{server_name}] Unexpected error loading tools: {e}", exc_info=True)
                    raise # Re-raise other errors to potentially cancel the group
            else:
                log.info(f"[{server_name}] Skipping tools (not supported or method unavailable).")

        async def load_resources_task():
            if has_resources and hasattr(session, 'list_resources'):
                log.info(f"[{server_name}] Loading resources...")
                try:
                    res = await asyncio.wait_for(session.list_resources(), timeout=capability_timeout)
                    self._process_list_result(server_name, getattr(res, 'resources', None), self.resources, MCPResource, "resource")
                except asyncio.TimeoutError:
                    log.error(f"[{server_name}] Timeout loading resources.")
                except McpError as e:
                     log.error(f"[{server_name}] MCP Error loading resources: {e}")
                except Exception as e:
                    log.error(f"[{server_name}] Unexpected error loading resources: {e}", exc_info=True)
                    raise # Re-raise other errors
            else:
                log.info(f"[{server_name}] Skipping resources (not supported or method unavailable).")

        async def load_prompts_task():
            if has_prompts and hasattr(session, 'list_prompts'):
                log.info(f"[{server_name}] Loading prompts...")
                try:
                    res = await asyncio.wait_for(session.list_prompts(), timeout=capability_timeout)
                    self._process_list_result(server_name, getattr(res, 'prompts', None), self.prompts, MCPPrompt, "prompt")
                except asyncio.TimeoutError:
                    log.error(f"[{server_name}] Timeout loading prompts.")
                except McpError as e:
                     log.error(f"[{server_name}] MCP Error loading prompts: {e}")
                except Exception as e:
                    log.error(f"[{server_name}] Unexpected error loading prompts: {e}", exc_info=True)
                    raise # Re-raise other errors
            else:
                log.info(f"[{server_name}] Skipping prompts (not supported or method unavailable).")
        # --- End definition of loading tasks ---

        # --- Use anyio.create_task_group for structured concurrency ---
        try:
            async with anyio.create_task_group() as tg:
                tg.start_soon(load_tools_task)
                tg.start_soon(load_resources_task)
                tg.start_soon(load_prompts_task)
            # No results are explicitly returned by the tasks, they modify self state directly
        except Exception as e:
            # Log errors that caused the task group to cancel (e.g., re-raised exceptions from tasks)
            # Or other TaskGroup errors. anyio might raise an ExceptionGroup.
            if isinstance(e, BaseExceptionGroup): # Handle potential ExceptionGroup from anyio
                 log.error(f"[{server_name}] Multiple errors during concurrent capability loading:")
                 for i, sub_exc in enumerate(e.exceptions):
                      log.error(f"  Error {i+1}: {type(sub_exc).__name__} - {sub_exc}", exc_info=False) # Log basic info
                      log.debug(f"  Traceback {i+1}:", exc_info=sub_exc) # Log full traceback at debug
            else:
                # Log single exception
                log.error(f"[{server_name}] Error during concurrent capability loading task group: {e}", exc_info=True)
        # --- End anyio usage ---

        log.info(f"[{server_name}] Capability loading attempt finished.")

    def _process_list_result(self, server_name: str, result_list: Optional[List[Any]], target_dict: Dict[str, Any], item_class: type, item_type_name: str):
        """Helper to populate tools/resources/prompts dictionaries from list results."""
        items_added = 0
        items_skipped = 0

        # Clear existing items for this server first
        for key in list(target_dict.keys()):
            if hasattr(target_dict[key], 'server_name') and target_dict[key].server_name == server_name:
                del target_dict[key]

        if result_list is None:
            log.warning(f"[{server_name}] Received None instead of a list for {item_type_name}s.")
            return

        for item in result_list:
            try:
                if not hasattr(item, 'name') or not isinstance(item.name, str) or not item.name:
                    log.warning(f"[{server_name}] Skipping {item_type_name} item lacking valid 'name': {item}")
                    items_skipped += 1
                    continue

                # Specific checks (example for Tool)
                correct_input_schema = None
                if item_class is MCPTool:
                    schema = getattr(item, 'inputSchema', getattr(item, 'input_schema', None))
                    if schema is None or not isinstance(schema, dict):
                        log.warning(f"[{server_name}] Skipping tool '{item.name}' lacking valid schema. Item data: {item}")
                        items_skipped += 1
                        continue
                    correct_input_schema = schema

                # Construct name and create instance
                item_name_full = f"{server_name}:{item.name}" if ":" not in item.name else item.name
                instance_data = {
                    "name": item_name_full, "description": getattr(item, 'description', '') or '',
                    "server_name": server_name,
                }
                # Add type-specific data
                if item_class is MCPTool:
                    instance_data["input_schema"] = correct_input_schema
                    instance_data["original_tool"] = item
                elif item_class is MCPResource:
                    instance_data["template"] = getattr(item, 'uri', '')
                    instance_data["original_resource"] = item
                elif item_class is MCPPrompt:
                    instance_data["template"] = f"Prompt: {item.name}"
                    instance_data["original_prompt"] = item # Placeholder/Adapt

                target_dict[item_name_full] = item_class(**instance_data)
                items_added += 1
            except Exception as proc_err:
                log.error(f"[{server_name}] Error processing {item_type_name} item '{getattr(item, 'name', 'UNKNOWN')}': {proc_err}", exc_info=True)
                items_skipped += 1
        log.info(f"[{server_name}] Processed {items_added} {item_type_name}s ({items_skipped} skipped).")

    async def connect_to_server(self, server_config: ServerConfig) -> Optional[ClientSession]:
        """
        Connects to a single MCP server (STDIO or SSE) with retry logic.
        Handles server renaming based on InitializeResult BEFORE loading capabilities.
        """
        initial_server_name = server_config.name
        # Use a mutable copy for potential renaming within the loop
        current_server_config = dataclasses.replace(server_config)
        current_server_name = initial_server_name

        retry_count = 0
        config_updated_by_rename = False
        max_retries = current_server_config.retry_policy.get("max_attempts", 3)
        backoff_factor = current_server_config.retry_policy.get("backoff_factor", 0.5)
        timeout_increment = current_server_config.retry_policy.get("timeout_increment", 5.0)

        with safe_stdout(): # Protect stdout during connection attempts
            while retry_count <= max_retries:
                start_time = time.time()
                session: Optional[ClientSession] = None
                initialize_result_obj: Optional[Union[MCPInitializeResult, Dict[str, Any]]] = None
                process_this_attempt: Optional[asyncio.subprocess.Process] = None
                process_to_use: Optional[asyncio.subprocess.Process] = None #
                connection_error: Optional[BaseException] = None
                span = None
                span_context_manager = None
                attempt_exit_stack = AsyncExitStack() # Stack for THIS attempt's resources

                try: # Trace span setup
                    if tracer:
                        span_context_manager = tracer.start_as_current_span(
                            f"connect_server.{current_server_name}",
                            attributes={ "server.name": initial_server_name, "server.type": current_server_config.type.value, "retry": retry_count }
                        )
                        if span_context_manager:
                            span = span_context_manager.__enter__()
                except Exception as e:
                    log.warning(f"Failed to start trace span: {e}")
                    span = span_context_manager = None

                try:
                    # --- Main Connection Attempt ---
                    self._safe_printer(f"[cyan]Connecting to {initial_server_name} (as '{current_server_name}', Attempt {retry_count+1}/{max_retries+1})...[/]")

                    # --- Connection & Handshake ---
                    if current_server_config.type == ServerType.STDIO:
                        # Helper handles process start/reuse and handshake
                        session, initialize_result_obj, process_this_attempt = await self._connect_stdio_server(
                            current_server_config, current_server_name, retry_count
                        )
                        # If a new process was started, manage its lifetime for this attempt
                        if process_this_attempt:
                            await attempt_exit_stack.enter_async_context(self._manage_process_lifetime(process_this_attempt, f"attempt-{current_server_name}-{retry_count}"))
                        # Manage session lifetime for this attempt
                        if session:
                            await attempt_exit_stack.enter_async_context(session)

                    elif current_server_config.type == ServerType.SSE:
                        connect_url = current_server_config.path
                        parsed_connect_url = urlparse(connect_url)
                        if not parsed_connect_url.scheme:
                            connect_url = f"http://{connect_url}"

                        connect_timeout = current_server_config.timeout + (retry_count * timeout_increment)
                        # Use a very long read timeout for persistent SSE streams
                        persistent_read_timeout_secs = timedelta(days=1).total_seconds() # Effectively infinite
                        handshake_timeout = connect_timeout + 15.0 # Allow extra time for handshake

                        log.info(f"[{current_server_name}] Creating persistent SSE context for {connect_url}...")
                        sse_ctx = sse_client(
                            url=connect_url, headers=None, timeout=connect_timeout,
                            sse_read_timeout=persistent_read_timeout_secs
                        )
                        # Manage SSE streams within the attempt stack first
                        read_stream, write_stream = await attempt_exit_stack.enter_async_context(sse_ctx)
                        log.debug(f"[{current_server_name}] SSE streams obtained.")

                        # Use standard ClientSession, also managed by attempt stack initially
                        session_read_timeout = timedelta(seconds=persistent_read_timeout_secs + 10.0)
                        session = ClientSession(read_stream=read_stream, write_stream=write_stream, read_timeout_seconds=session_read_timeout)
                        await attempt_exit_stack.enter_async_context(session)
                        log.info(f"[{current_server_name}] Persistent SSE ClientSession created.")

                        # Perform handshake
                        log.info(f"[{current_server_name}] Performing MCP handshake (initialize) via SSE...")
                        initialize_result_obj = await asyncio.wait_for(session.initialize(), timeout=handshake_timeout)
                        log.info(f"[{current_server_name}] Initialize successful (SSE). Server reported: {getattr(initialize_result_obj, 'serverInfo', 'N/A')}")
                        process_this_attempt = None # No process for SSE

                    else:
                        raise RuntimeError(f"Unknown server type: {current_server_config.type}")

                    if not session:
                        raise RuntimeError(f"Session object is None after connection attempt for {current_server_name}")

                    # --- Rename Logic ---
                    original_name_before_rename = current_server_name
                    actual_server_name_from_init: Optional[str] = None
                    if isinstance(initialize_result_obj, dict): # STDIO Result
                        server_info = initialize_result_obj.get("serverInfo", {})
                        if isinstance(server_info, dict):
                            actual_server_name_from_init = server_info.get("name")
                    elif isinstance(initialize_result_obj, MCPInitializeResult): # Standard Result
                        if initialize_result_obj.serverInfo:
                            actual_server_name_from_init = initialize_result_obj.serverInfo.name

                    if actual_server_name_from_init and actual_server_name_from_init != current_server_name:
                        if actual_server_name_from_init in self.config.servers and actual_server_name_from_init != initial_server_name:
                            log.warning(f"Server '{initial_server_name}' reported name '{actual_server_name_from_init}', conflicts with existing. Keeping '{current_server_name}'.")
                        else:
                            log.info(f"Server '{initial_server_name}' identified as '{actual_server_name_from_init}'. Renaming config.")
                            self._safe_printer(f"[yellow]Server '{initial_server_name}' identified as '{actual_server_name_from_init}', updating config.[/]")
                            try:
                                if initial_server_name in self.config.servers:
                                    # Remove old entry, update config object, add new entry
                                    original_config_entry = self.config.servers.pop(initial_server_name)
                                    original_config_entry.name = actual_server_name_from_init # Update the object itself
                                    self.config.servers[actual_server_name_from_init] = original_config_entry # Add with new key

                                    # Update process mapping if STDIO
                                    if server_config.type == ServerType.STDIO:
                                        if original_name_before_rename in self.processes:
                                            proc = self.processes.pop(original_name_before_rename)
                                            self.processes[actual_server_name_from_init] = proc

                                    # Update the current loop variables
                                    current_server_name = actual_server_name_from_init
                                    current_server_config = original_config_entry # Use the updated config object
                                    config_updated_by_rename = True
                                    if span:
                                        span.set_attribute("server.name", current_server_name)
                                    if isinstance(session, RobustStdioSession):
                                        session._server_name = current_server_name # Update session's internal name
                                    log.info(f"Renamed config entry '{initial_server_name}' -> '{current_server_name}'.")
                                else:
                                    log.warning(f"Cannot rename: '{initial_server_name}' no longer in config.")
                            except Exception as rename_err:
                                log.error(f"Failed rename '{initial_server_name}' to '{actual_server_name_from_init}': {rename_err}", exc_info=True)
                                raise RuntimeError(f"Failed during server rename: {rename_err}") from rename_err

                    # --- Load Capabilities (AFTER potential rename) ---
                    await self._load_server_capabilities(current_server_name, session, initialize_result_obj)

                    # --- Success Path ---
                    connection_time_ms = (time.time() - start_time) * 1000
                    server_config_in_dict = self.config.servers.get(current_server_name)
                    if server_config_in_dict:
                        metrics = server_config_in_dict.metrics
                        metrics.request_count += 1 # Count connection attempt
                        metrics.update_response_time(connection_time_ms / 1000.0)
                        metrics.update_status()

                    if latency_histogram:
                        try:
                            latency_histogram.record(connection_time_ms, {"server.name": current_server_name})
                        except Exception as histo_err:
                            log.warning(f"Failed to record latency histogram for {current_server_name}: {histo_err}")
                    if span:
                        span.set_status(trace.StatusCode.OK)

                    tools_loaded_count = len([t for t in self.tools.values() if t.server_name == current_server_name])
                    log.info(f"Connected & loaded capabilities for {current_server_name} ({tools_loaded_count} tools) in {connection_time_ms:.0f}ms")
                    self._safe_printer(f"[green]{EMOJI_MAP['success']} Connected & loaded: {current_server_name} ({tools_loaded_count} tools)[/]")

                    # *** Success: Adopt resources into main exit stack ***
                    # This moves session, and potentially sse_client context manager
                    await self.exit_stack.enter_async_context(attempt_exit_stack.pop_all())
                    # Add session to active list *after* adopting to main stack
                    self.active_sessions[current_server_name] = session

                    # Track the *persistent* process object under the final name
                    if process_to_use and server_config.type == ServerType.STDIO:
                        # Ensure it's the correct process, especially if reused
                        if current_server_name not in self.processes or self.processes[current_server_name] != process_to_use:
                            self.processes[current_server_name] = process_to_use

                    # Save config YAML if rename occurred
                    if config_updated_by_rename:
                        log.info(f"Saving configuration YAML after server rename to {current_server_name}...")
                        await self.config.save_async()

                    # Advertise via Zeroconf if it's a newly connected STDIO server
                    if current_server_config.type == ServerType.STDIO:
                        await self.register_local_server(current_server_config)

                    if span_context_manager:
                        span_context_manager.__exit__(None, None, None)
                    return session # Return the successful session

                # --- Outer Exception Handling for Connection Attempt ---
                except (McpError, RuntimeError, ConnectionAbortedError, httpx.RequestError, subprocess.SubprocessError, OSError, FileNotFoundError, asyncio.TimeoutError, BaseExceptionGroup) as e:
                    connection_error = e
                    error_type_name = type(e).__name__
                    log.warning(f"[{initial_server_name}] Connection attempt {retry_count+1} failed: {error_type_name} - {e}", exc_info=False) # Log less verbosely initially
                    log.debug(f"Traceback for {initial_server_name} connection error:", exc_info=True) # Full traceback at debug level
                # --- End Connection Attempt Try/Except ---
                finally:
                    # Ensure attempt-specific resources are cleaned up *before* retry/failure
                    if span_context_manager and hasattr(span_context_manager, '__exit__'):
                        try:
                            # Exit the context manager, passing exception info if any
                            exc_type, exc_val, exc_tb = sys.exc_info()
                            span_context_manager.__exit__(exc_type, exc_val, exc_tb)
                        except Exception as span_exit_err:
                            log.warning(f"Error closing OpenTelemetry span context: {span_exit_err}")

                # --- Shared Error Handling & Retry Logic ---
                retry_count += 1
                # Update metrics even on failure attempt
                config_for_metrics = self.config.servers.get(current_server_name) # Use potentially renamed server
                if config_for_metrics:
                    config_for_metrics.metrics.error_count += 1
                    config_for_metrics.metrics.update_status()
                else:
                    log.error(f"Could not find server config '{current_server_name}' for error metric update.")

                if span:
                    span.set_status(trace.StatusCode.ERROR, f"Attempt {retry_count-1} failed: {type(connection_error).__name__}")

                if retry_count <= max_retries:
                    delay = min(backoff_factor * (2 ** (retry_count - 1)) + random.random() * 0.1, 10.0)
                    error_msg_display = str(connection_error or "Unknown error")[:150] + "..."
                    self._safe_printer(f"[yellow]{EMOJI_MAP['warning']} Error connecting {initial_server_name} (as '{current_server_name}'): {error_msg_display}[/]")
                    self._safe_printer(f"[cyan]Retrying connection in {delay:.2f}s...[/]")
                    await asyncio.sleep(delay)
                    # Continue main while loop
                else: # Max retries exceeded
                    final_error_msg = str(connection_error or "Unknown connection error")
                    log.error(f"Failed to connect to {initial_server_name} (as '{current_server_name}') after {max_retries+1} attempts. Final error: {final_error_msg}")
                    self._safe_printer(f"[red]{EMOJI_MAP['error']} Failed to connect: {initial_server_name} (as '{current_server_name}') after {max_retries+1} attempts.[/]")
                    if span:
                        span.set_status(trace.StatusCode.ERROR, f"Max retries exceeded. Final: {type(connection_error).__name__}")
                    if config_updated_by_rename: # Save config if rename happened but connection ultimately failed
                        log.info(f"Saving configuration YAML after failed connection for renamed server {current_server_name}...")
                        await self.config.save_async()
                    return None # Connection failed

            # Should only reach here if loop completes unexpectedly (should break or return)
            log.error(f"Connection loop for {initial_server_name} exited unexpectedly.")
            return None

    async def connect_to_server_by_name(self, server_name: str) -> bool:
        """Connects to a server specified by its name."""
        if server_name not in self.config.servers:
            log.error(f"Cannot connect: Server '{server_name}' not found in configuration.")
            return False
        if server_name in self.active_sessions:
            log.info(f"Server '{server_name}' is already connected.")
            return True

        server_config = self.config.servers[server_name]
        if not server_config.enabled:
            log.warning(f"Cannot connect: Server '{server_name}' is disabled.")
            return False

        session = await self.connect_to_server(server_config)
        return session is not None

    async def _run_with_simple_progress(self, tasks, title):
        """
        Simpler version of _run_with_progress without Rich Live/Progress display.
        Used as a fallback when nested progress displays would occur, providing
        basic sequential console feedback with emojis.

        Args:
            tasks: A list of tuples with (task_func, task_description, task_args).
            title: The title to print before starting tasks.

        Returns:
            A list of results from the successfully completed tasks. Errors are logged.
            Failed tasks will result in None being appended to the results list.
        """
        safe_console = get_safe_console()
        results = []
        total_tasks = len(tasks)

        # Use an appropriate emoji for the overall process start
        start_emoji = EMOJI_MAP.get('processing', '⚙️')
        safe_console.print(f"[bold cyan]{start_emoji} {title}[/]")

        for i, (task_func, description, task_args) in enumerate(tasks):
            task_idx = i + 1
            # Print start message for the task using a gear or processing emoji
            processing_emoji = EMOJI_MAP.get('gear', '⚙️')
            # Ensure description doesn't have leading/trailing whitespace that might mess up formatting
            clean_description = description.strip()
            safe_console.print(f"  ({task_idx}/{total_tasks}) {processing_emoji} Running: {clean_description}...")

            try:
                # Run the actual async task function
                # Use asyncio.wait_for if a timeout per task is desired, otherwise run directly
                # Example without timeout:
                result = await task_func(*task_args) if task_args else await task_func()
                results.append(result)

                # Print success message using a success emoji
                success_emoji = EMOJI_MAP.get('success', '✅')
                safe_console.print(f"  ({task_idx}/{total_tasks}) {success_emoji} Finished: {clean_description}")

            except Exception as e:
                # Print failure message using an error emoji
                error_emoji = EMOJI_MAP.get('error', '❌')
                safe_console.print(f"  ({task_idx}/{total_tasks}) {error_emoji} Failed: {clean_description}")

                # Print the error details concisely
                # Get first line of error message for console brevity
                error_msg_str = str(e).split('\n')[0]
                safe_console.print(f"      [red]Error: {error_msg_str}[/]")

                # Log the full error with traceback for debugging
                log.error(f"Task {task_idx} ('{clean_description}') failed", exc_info=True)

                # Append None for failed tasks to indicate failure in the results list
                # Or potentially append the exception object itself if callers need it
                results.append(None)
                # Continue with the next task
                continue

        # Final completion message
        finish_emoji = EMOJI_MAP.get('party_popper', '🎉') # Or use 'success' emoji again
        safe_console.print(f"[cyan]{finish_emoji} Completed: {title}[/]")
        return results # Return results (may contain None for failed tasks)
    
    async def _run_with_progress(self, tasks, title, transient=True, use_health_scores=False):
        """Run tasks with a progress bar, ensuring only one live display exists at a time.

        Args:
            tasks: A list of tuples with (task_func, task_description, task_args)
            title: The title for the progress bar
            transient: Whether the progress bar should disappear after completion
            use_health_scores: If True, uses 'health' value from result dict as progress percent

        Returns:
            A list of results from the tasks
        """
        # Check if we already have an active progress display to prevent nesting
        # Use self._active_live_display consistently
        if hasattr(self, '_active_live_display') and self._active_live_display:
            log.warning("Attempted to create nested progress display, using simpler output")
            # Assuming _run_with_simple_progress exists and is implemented elsewhere
            return await self._run_with_simple_progress(tasks, title)

        # Set a flag that we have an active progress
        self._active_live_display = True # Use the correct flag name

        results = []

        try:
            # Set total based on whether we're using health scores or not
            task_total = 100 if use_health_scores else 1

            # *** Ensure Progress and Columns are used ***
            columns = [
                TextColumn("[progress.description]{task.description}"), # Uses TextColumn
                BarColumn(complete_style="green", finished_style="green"), # Uses BarColumn
            ]
            if use_health_scores:
                # Display health score percentage
                columns.append(TextColumn("[progress.percentage]{task.percentage:>3.0f}%")) # Uses TextColumn
            else:
                # Display task progress (e.g., 1/1)
                columns.append(TaskProgressColumn()) # Uses TaskProgressColumn
            # Always add a spinner for visual activity
            columns.append(SpinnerColumn("dots")) # Uses SpinnerColumn

            # Ensure Progress is used with the defined columns
            with Progress(
                *columns,
                console=get_safe_console(), # Use the safe console
                transient=transient, # Controls if the bar disappears on completion
                expand=False # Prevent the progress bar from taking full width unnecessarily
            ) as progress: # Uses Progress
                # Create all Rich Progress tasks upfront
                task_ids = []
                for _, description, _ in tasks:
                    # Add task to Rich Progress, initially not started
                    task_id = progress.add_task(description, total=task_total, start=False)
                    task_ids.append(task_id)

                # Run each provided async task sequentially
                for i, (task_func, task_desc, task_args) in enumerate(tasks):
                    current_task_id = task_ids[i]
                    progress.start_task(current_task_id) # Mark the task as started in Rich Progress
                    progress.update(current_task_id, description=task_desc) # Ensure description is set

                    try:
                        # Execute the actual async function passed in the tasks list
                        result = await task_func(*task_args) if task_args else await task_func()
                        results.append(result)

                        # Update the Rich Progress bar based on the result or mode
                        if use_health_scores and isinstance(result, dict) and 'health' in result:
                            # Update progress bar based on the 'health' value (0-100)
                            health_score = max(0, min(100, int(result['health']))) # Clamp value
                            progress.update(current_task_id, completed=health_score)
                            # Optionally append more info like tool count if available
                            if 'tools' in result:
                                progress.update(current_task_id, description=f"{task_desc} - {result['tools']} tools")
                        else:
                            # For non-health score tasks, mark as fully complete
                            progress.update(current_task_id, completed=task_total)

                        # Mark task as finished (stops spinner)
                        progress.stop_task(current_task_id)

                    except Exception as e:
                        # If a task function raises an error
                        progress.stop_task(current_task_id) # Stop the spinner
                        # Update description to show failure state
                        progress.update(current_task_id, description=f"{task_desc} [bold red]Failed: {str(e)}[/]", completed=0)
                        log.error(f"Task '{task_desc}' failed: {str(e)}", exc_info=True)
                        # Re-raise the exception to potentially halt the entire sequence if needed
                        # Or handle more gracefully depending on desired behavior
                        raise e

                # All tasks completed successfully if no exception was raised
                return results # Return the list of results from each task function

        finally:
            # CRITICAL: Always clear the flag when exiting this method
            self._active_live_display = None # Use the correct flag name

    async def connect_to_servers(self):
        """Connect to all enabled MCP servers concurrently."""
        if not self.config.servers:
            log.warning("No servers configured. Use 'servers add' or discovery.")
            return

        enabled_servers = {name: cfg for name, cfg in self.config.servers.items() if cfg.enabled}
        if not enabled_servers:
            log.info("No enabled servers to connect to.")
            return

        log.info(f"Connecting to {len(enabled_servers)} enabled servers...")

        connection_tasks = []
        for name, server_config in enabled_servers.items():
            if name in self.active_sessions:
                log.debug(f"Server '{name}' is already connected, skipping redundant connection attempt.")
                continue
            # Create a task for each connection attempt
            connection_tasks.append(
                (self.connect_to_server, # The function to call
                f"{EMOJI_MAP['server']} Connecting to {name}...", # Description for progress
                (server_config,) # Args tuple for the function
                )
            )

        # Use the progress helper to run connections concurrently
        if connection_tasks: # Only run progress if there are tasks to run
            try:
                if hasattr(app, "mcp_client") and hasattr(app.mcp_client, "_run_with_progress"):
                    results = await app.mcp_client._run_with_progress(connection_tasks, "Connecting Servers", transient=False, use_health_scores=False) # Don't use health score here
                    success_count = sum(1 for r in results if r is not None)
                    log.info(f"Finished connecting. Successfully connected to {success_count}/{len(connection_tasks)} servers.") # Log count based on tasks run
                else:
                    log.warning("MCPClient progress helper not found, connecting sequentially.")
                    for task_func, desc, args_tuple in connection_tasks:
                        log.info(desc)
                        await task_func(*args_tuple)

            except Exception as e:
                log.error(f"Error during concurrent server connection: {e}", exc_info=True)
        else:
            log.info("No new servers needed connection.")

        # Verify stdout pollution after connecting
        if os.environ.get("MCP_VERIFY_STDOUT", "1") == "1":
            with safe_stdout():
                log.info("Verifying no stdout pollution after connecting servers...")
                verify_no_stdout_pollution()

        # Start server monitoring *after* initial connection attempts
        await self.monitor.start_monitoring()
        # Print status after connection attempts
        # Ensure MCPClient instance exists before calling print_status
        if hasattr(app, "mcp_client"):
             await app.mcp_client.print_status()
        else:
             log.warning("Cannot print status, MCPClient instance not found on app.")

    async def disconnect_server(self, server_name: str):
        """Disconnects from a specific server and cleans up resources."""
        log.info(f"Disconnecting from server: {server_name}...")
        if server_name not in self.active_sessions:
            log.warning(f"Server '{server_name}' is not currently connected.")
            return

        # --- Cleanup MCP Entities ---
        self.tools = {k: v for k, v in self.tools.items() if v.server_name != server_name}
        self.resources = {k: v for k, v in self.resources.items() if v.server_name != server_name}
        self.prompts = {k: v for k, v in self.prompts.items() if v.server_name != server_name}
        log.debug(f"Removed tools/resources/prompts for {server_name}.")

        # --- Close Session ---
        # The session and its associated resources (like sse_client context)
        # are managed by the main exit_stack. We need to find and pop them.
        session_to_close = self.active_sessions.pop(server_name, None)
        if session_to_close:
            # Attempt to gracefully close the session itself
            try:
                await session_to_close.aclose()
                log.debug(f"Gracefully closed ClientSession for {server_name}.")
            except Exception as e:
                 log.warning(f"Error during explicit session close for {server_name}: {e}")
            # Note: Closing the session *might* trigger cleanup in the underlying
            # streams managed by the exit_stack, but explicitly popping from
            # the stack is safer if possible (requires identifying the context).
            # For simplicity now, we rely on the session's aclose and the main
            # exit_stack's cleanup on overall client shutdown. A more granular
            # exit_stack per session might be needed for immediate resource release.

        # --- Terminate STDIO Process ---
        process = self.processes.pop(server_name, None)
        if process:
            await self.terminate_process(server_name, process) # Use helper

        # --- Unregister Zeroconf ---
        if server_name in self.registered_services:
             await self.unregister_local_server(server_name)

        # --- Cancel related background tasks ---
        tasks = self._session_tasks.pop(server_name, [])
        if tasks:
            log.debug(f"Cancelling {len(tasks)} background tasks for {server_name}...")
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        log.info(f"Successfully disconnected from server: {server_name}")

    async def restart_server(self, server_name: str):
        """Restarts a server (disconnects and connects again)."""
        if server_name not in self.config.servers:
            log.error(f"Cannot restart: Server '{server_name}' not found.")
            return False

        log.info(f"Restarting server: {server_name}...")
        if server_name in self.active_sessions:
            await self.disconnect_server(server_name)
            await asyncio.sleep(0.5) # Brief pause before reconnect

        return await self.connect_to_server_by_name(server_name)

    async def reconnect_server(self, server_name: str):
        """Alias for restart_server, common for SSE."""
        return await self.restart_server(server_name)

    async def test_server_connection(self, server_config: ServerConfig) -> bool:
        """
        Attempts a connection and handshake without adding to active sessions.
        Cleans up resources afterwards. Returns True if handshake succeeds.
        """
        current_server_name = server_config.name
        max_retries = server_config.retry_policy.get("max_attempts", 1)
        backoff_factor = server_config.retry_policy.get("backoff_factor", 0.2)
        timeout_increment = server_config.retry_policy.get("timeout_increment", 2.0)
        test_exit_stack = AsyncExitStack() # Local stack for test resources

        log.debug(f"[Test:{current_server_name}] Starting connection test...")
        for retry_count in range(max_retries + 1):
            session: Optional[ClientSession] = None
            process_this_attempt: Optional[asyncio.subprocess.Process] = None
            log_file_handle = None
            last_error = None
            is_connected = False

            try:
                log.debug(f"[Test:{current_server_name}] Attempt {retry_count+1}/{max_retries+1}")
                current_timeout = server_config.timeout + (retry_count * timeout_increment)

                if server_config.type == ServerType.STDIO:
                    log_file_path = (CONFIG_DIR / f"{current_server_name}_test_stderr.log").resolve()
                    log_file_path.parent.mkdir(parents=True, exist_ok=True)
                    log_file_handle = await test_exit_stack.enter_async_context(aiofiles.open(log_file_path, "ab")) # type: ignore
                    stderr_fileno = log_file_handle.fileno()

                    process_this_attempt = await asyncio.create_subprocess_exec(
                        server_config.path, *server_config.args,
                        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                        stderr=stderr_fileno, limit=2**22, env=os.environ.copy()
                    )
                    await test_exit_stack.enter_async_context(self._manage_process_lifetime(process_this_attempt, f"test-{current_server_name}"))
                    await asyncio.sleep(0.5)
                    if process_this_attempt.returncode is not None:
                        raise RuntimeError(f"Test process failed immediately (code {process_this_attempt.returncode})")

                    session = RobustStdioSession(process_this_attempt, f"test-{current_server_name}")
                    await test_exit_stack.enter_async_context(session)
                    await asyncio.wait_for(session.initialize(response_timeout=current_timeout), timeout=current_timeout + 5.0)
                    await session.send_initialized_notification()

                elif server_config.type == ServerType.SSE:
                    connect_url = server_config.path
                    if not urlparse(connect_url).scheme:
                        connect_url = f"http://{connect_url}"
                    sse_ctx = sse_client(url=connect_url, timeout=current_timeout)
                    read_stream, write_stream = await test_exit_stack.enter_async_context(sse_ctx)
                    session = ClientSession(read_stream=read_stream, write_stream=write_stream, read_timeout_seconds=timedelta(seconds=current_timeout + 15.0))
                    await test_exit_stack.enter_async_context(session)
                    await asyncio.wait_for(session.initialize(), timeout=current_timeout + 10.0)
                else:
                    raise RuntimeError(f"Unknown server type: {server_config.type}")

                log.debug(f"[Test:{current_server_name}] Handshake successful.")
                is_connected = True
                return True # Success

            except (McpError, RuntimeError, ConnectionAbortedError, httpx.RequestError, subprocess.SubprocessError, OSError, FileNotFoundError, asyncio.TimeoutError, BaseExceptionGroup) as e:
                last_error = e
                log_level = logging.DEBUG if isinstance(e, (httpx.ConnectError, asyncio.TimeoutError, ConnectionRefusedError)) else logging.WARNING
                log.log(log_level, f"[Test:{current_server_name}] Attempt {retry_count+1} failed: {type(e).__name__} - {e}", exc_info=(log_level >= logging.WARNING))
            except Exception as e:
                 last_error = e
                 log.error(f"[Test:{current_server_name}] Unexpected error: {e}", exc_info=True)
                 # Close stack and return False immediately on unexpected error
                 await test_exit_stack.aclose()
                 return False
            finally:
                 # Clean up resources for *this specific attempt* if it didn't succeed immediately
                 if not is_connected:
                      await test_exit_stack.aclose()
                      test_exit_stack = AsyncExitStack() # Reset for next potential retry

            # --- Retry Logic ---
            if retry_count < max_retries:
                delay = min(backoff_factor * (2 ** retry_count) + random.random() * 0.05, 5.0)
                log.debug(f"[Test:{current_server_name}] Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                log.warning(f"[Test:{current_server_name}] Final connection test failed after {max_retries+1} attempts. Last error: {type(last_error).__name__}")
                return False # Failed after all retries

        # Should not be reached normally
        return False


    # --- Zeroconf Registration ---
    async def register_local_server(self, server_config: ServerConfig):
        """Register a locally started STDIO MCP server with Zeroconf."""
        if not self.config.enable_local_discovery or not self.registry or not self.registry.zeroconf:
            log.debug("Zeroconf registration skipped (disabled or unavailable).")
            return
        if server_config.name in self.registered_services:
            log.debug(f"Zeroconf service for '{server_config.name}' already registered.")
            return
        if server_config.type != ServerType.STDIO:
            log.debug(f"Skipping Zeroconf registration for non-STDIO server '{server_config.name}'.")
            return

        try:
            # Determine local IP
            local_ip = '127.0.0.1' # Default, safer for local registration
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                try:
                    s.connect(('8.8.8.8', 80))
                    local_ip = s.getsockname()[0]
                except Exception:
                    pass # Keep 127.0.0.1 on error

            # Determine port (simple heuristic, may need improvement)
            port = 8080 # Default
            for i, arg in enumerate(server_config.args):
                if arg in ['--port', '-p'] and i + 1 < len(server_config.args):
                    try:
                        port = int(server_config.args[i+1])
                        break
                    except (ValueError, IndexError):
                        pass

            # Prepare properties
            props = {
                b'name': server_config.name.encode('utf-8'),
                b'type': server_config.type.value.encode('utf-8'),
                b'description': server_config.description.encode('utf-8'),
                b'version': str(server_config.version or '0.0.0').encode('utf-8'),
                # 'host' prop might not be standard, rely on address in ServiceInfo
            }

            # Create ServiceInfo
            service_type = "_mcp._tcp.local."
            service_name = f"{server_config.name}.{service_type}"
            service_info = ServiceInfo(
                type_=service_type, name=service_name,
                addresses=[ipaddress.ip_address(local_ip).packed], port=port,
                properties=props, server=f"{socket.gethostname()}.local." # Advertise server hostname
            )

            log.info(f"Registering local server '{server_config.name}' with Zeroconf ({local_ip}:{port})")
            # Use async registration directly if available (check zeroconf version)
            if hasattr(self.registry.zeroconf, 'async_register_service'):
                await self.registry.zeroconf.async_register_service(service_info)
            else: # Fallback for older zeroconf or sync context needs
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.registry.zeroconf.register_service, service_info)

            self.registered_services[server_config.name] = service_info
            log.info(f"Successfully registered '{server_config.name}' with Zeroconf.")

        except NonUniqueNameException:
            log.warning(f"Zeroconf: Name '{server_config.name}' already registered. Might be stale.")
        except Exception as e:
            log.error(f"Error registering '{server_config.name}' with Zeroconf: {e}", exc_info=True)

    async def unregister_local_server(self, server_name: str):
        """Unregister a server from Zeroconf."""
        if not self.registry or not self.registry.zeroconf:
            return
        if server_name in self.registered_services:
            service_info = self.registered_services.pop(server_name)
            log.info(f"Unregistering '{server_name}' from Zeroconf...")
            try:
                if hasattr(self.registry.zeroconf, 'async_unregister_service'):
                    await self.registry.zeroconf.async_unregister_service(service_info)
                else: # Fallback for older zeroconf or sync context needs
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self.registry.zeroconf.unregister_service, service_info)
                log.info(f"Successfully unregistered '{server_name}' from Zeroconf.")
            except Exception as e:
                log.error(f"Failed unregister '{server_name}' from Zeroconf: {e}", exc_info=True)
        else:
            log.debug(f"No active Zeroconf registration found for '{server_name}'.")

class MCPClient:
    def __init__(self):
        self.config = Config()
        self.history = History(max_entries=self.config.history_size)
        self.conversation_graph = ConversationGraph() # Start fresh

        app.mcp_client = self # Global access

        self.tool_cache = ToolCache(
            cache_dir=CACHE_DIR,
            custom_ttl_mapping=self.config.cache_ttl_mapping
        )
        self.server_manager = ServerManager(self.config, tool_cache=self.tool_cache, safe_printer=self.safe_print)

        # --- Initialize Provider Clients to None ---
        self.anthropic: Optional[AsyncAnthropic] = None
        self.openai_client: Optional[AsyncOpenAI] = None
        self.gemini_client: Optional[AsyncOpenAI] = None
        self.grok_client: Optional[AsyncOpenAI] = None
        self.deepseek_client: Optional[AsyncOpenAI] = None
        self.mistral_client: Optional[AsyncOpenAI] = None
        self.groq_client: Optional[AsyncOpenAI] = None
        self.cerebras_client: Optional[AsyncOpenAI] = None

        self.current_model = self.config.default_model
        self.server_monitor = ServerMonitor(self.server_manager)
        self.discovered_local_servers = set()
        self.local_discovery_task = None
        self.use_auto_summarization = self.config.use_auto_summarization
        self.auto_summarize_threshold = self.config.auto_summarize_threshold

        self.conversation_graph_file = Path(self.config.conversation_graphs_dir) / "default_conversation.json"
        self.conversation_graph_file.parent.mkdir(parents=True, exist_ok=True)
        # Graph loading deferred to setup

        self.current_query_task: Optional[asyncio.Task] = None
        self.session_input_tokens: int = 0
        self.session_output_tokens: int = 0
        self.session_total_cost: float = 0.0
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.tokens_saved_by_cache = 0

        # Pre-compile emoji pattern
        self._emoji_chars = [re.escape(str(emoji)) for emoji in EMOJI_MAP.values()]
        self._emoji_space_pattern = re.compile(f"({'|'.join(self._emoji_chars)})" + r"(\S)")
        self._current_query_text: str = ""
        self._current_status_messages: List[str] = []
        self._active_live_display: Optional[Live] = None # Track the Live instance

        # Command handler dictionary (populated later)
        self.commands: Dict[str, Callable[[str], Coroutine[Any, Any, None]]] = {} # Type hint

        # Initialize commands after all methods are defined
        self._initialize_commands()

        # Readline setup
        if platform.system() != "Windows": # Readline might behave differently on Windows
            try:
                readline.set_completer(self.completer)
                # Use libedit bindings for macOS compatibility
                if 'libedit' in readline.__doc__:
                    readline.parse_and_bind("bind ^I rl_complete") # Binds Tab to complete
                else:
                    readline.parse_and_bind("tab: complete") # Standard binding

                # --- Robust History Handling ---
                histfile_path = CONFIG_DIR / ".cli_history"
                histfile = str(histfile_path.resolve()) # Use resolved absolute path

                try:
                    # Ensure parent directory exists
                    histfile_path.parent.mkdir(parents=True, exist_ok=True)
                    # Try reading history
                    if histfile_path.exists():
                        # Only read if the file exists AND has content
                        if histfile_path.stat().st_size > 0:
                            log.debug(f"Attempting to load readline history from non-empty file: {histfile}")
                            try:
                                readline.read_history_file(histfile)
                            except OSError as read_err:
                                # Log specifically if the read fails, but continue
                                log.warning(f"Could not load readline history from '{histfile}': {read_err}", exc_info=False)
                            except Exception as read_generic_err: # Catch other potential read errors
                                log.warning(f"Unexpected error loading readline history from '{histfile}': {read_generic_err}", exc_info=False)
                        else:
                            log.debug(f"Readline history file exists but is empty, skipping read: {histfile}")
                    else:
                        try:
                            histfile_path.touch() # Create if not exists
                            log.debug(f"Created empty readline history file: {histfile}")
                        except OSError as touch_err:
                            log.warning(f"Could not create readline history file '{histfile}': {touch_err}")
                    readline.set_history_length(1000)
                    # Register saving history at exit
                    atexit.register(readline.write_history_file, histfile)
                    log.debug(f"Readline history configured using: {histfile}")
                # --- Keep the outer Exception block for other setup errors ---
                except Exception as e:
                    # Catch specific potential errors if needed (e.g., PermissionError, OSError)
                    log.warning(f"Could not load/save readline history from '{histfile}': {e}", exc_info=False) # Less verbose log
            except ImportError:
                 log.warning("Readline library not available, CLI history and completion disabled.")
            except Exception as e:
                 log.warning(f"Error setting up readline: {e}")
        else:
            log.info("Readline setup skipped on Windows.")

    def _initialize_commands(self):
        """Populates the command dictionary."""
        # Map command strings to their corresponding async methods
        self.commands = {
            'help': self.cmd_help,
            'exit': self.cmd_exit, 'quit': self.cmd_exit, # Aliases
            'config': self.cmd_config,
            'servers': self.cmd_servers,
            'tools': self.cmd_tools,
            'tool': self.cmd_tool,
            'resources': self.cmd_resources,
            'prompts': self.cmd_prompts,
            'model': self.cmd_model,
            'cache': self.cmd_cache,
            'clear': self.cmd_clear,
            'reload': self.cmd_reload,
            'fork': self.cmd_fork,
            'branch': self.cmd_branch,
            'export': self.cmd_export,
            'import': self.cmd_import,
            'history': self.cmd_history,
            'optimize': self.cmd_optimize,
            'apply-prompt': self.cmd_apply_prompt,
            'discover': self.cmd_discover,
            'dashboard': self.cmd_dashboard,
            # Add monitor, registry if needed
        }
        log.debug(f"Initialized {len(self.commands)} CLI commands.")

    # --- Readline Completer ---
    def completer(self, text: str, state: int) -> Optional[str]:
        """Readline completer for commands and arguments."""
        line = readline.get_line_buffer()
        parts = line.lstrip().split()

        options: List[str] = []
        prefix_to_match = text

        try:
            # --- Command Completion ---
            if line.startswith('/') and (len(parts) == 0 or (len(parts) == 1 and not line.endswith(" "))):
                cmd_prefix = line[1:]
                prefix_to_match = cmd_prefix # Match against the command part
                options = sorted([f"/{cmd}" for cmd in self.commands if cmd.startswith(cmd_prefix)])

            # --- Argument Completion ---
            elif len(parts) > 0 and parts[0].startswith('/'):
                cmd = parts[0][1:]
                num_args_entered = len(parts) - 1
                # If the last character is not a space, we are completing the *last* part
                # Otherwise, we are starting a *new* part.
                is_completing_last_part = not line.endswith(" ")
                arg_index_to_complete = num_args_entered -1 if is_completing_last_part else num_args_entered
                prefix_to_match = parts[-1] if is_completing_last_part else ""

                # --- Specific Command Argument Completion ---
                if cmd == "model" and arg_index_to_complete == 0:
                    options = sorted([m for m in COST_PER_MILLION_TOKENS if m.startswith(prefix_to_match)])
                elif cmd == "servers":
                    if arg_index_to_complete == 0: # Completing the subcommand
                        server_subcmds = ["list", "add", "remove", "connect", "disconnect", "enable", "disable", "status"]
                        options = sorted([s for s in server_subcmds if s.startswith(prefix_to_match)])
                    elif arg_index_to_complete >= 1 and parts[1] in ["remove", "connect", "disconnect", "enable", "disable", "status"]: # Completing server names
                        options = sorted([name for name in self.config.servers if name.startswith(prefix_to_match)])
                        if "all".startswith(prefix_to_match) and parts[1] in ["connect", "disconnect"]: options.append("all") # Add 'all' option
                elif cmd == "config":
                    if arg_index_to_complete == 0: # Completing the subcommand
                        config_subcmds = ["show", "edit", "reset", "api-key", "base-url", "model", "max-tokens", "history-size", "temperature"]
                        config_subcmds.extend([a.replace("enable_", "").replace("use_", "").replace("_", "-") for a in SIMPLE_SETTINGS_ENV_MAP.values() if type(getattr(Config(), a)) is bool])
                        config_subcmds.extend(["port-scan", "discovery-path", "registry-urls", "cache-ttl"])
                        options = sorted(set(config_subcmds))
                        options = [s for s in options if s.startswith(prefix_to_match)]
                    elif arg_index_to_complete == 1 and parts[1] in ["api-key", "base-url"]: # Completing provider name
                         options = sorted([p.value for p in Provider if p.value.startswith(prefix_to_match)])
                    # Add more specific config completions (e.g., for port-scan sub-subcommands)
                elif cmd == "tool" and arg_index_to_complete == 0: # Completing tool name
                     options = sorted([name for name in self.server_manager.tools if name.startswith(prefix_to_match)])
                elif cmd == "prompt" and arg_index_to_complete == 0: # Completing prompt name
                    options = sorted([name for name in self.server_manager.prompts if name.startswith(prefix_to_match)])
                elif cmd == "branch" and arg_index_to_complete == 0: # Completing branch subcommand
                     options = sorted([s for s in ["list", "checkout", "rename", "delete"] if s.startswith(prefix_to_match)])
                elif cmd == "branch" and parts[1] == "checkout" and arg_index_to_complete == 1: # Completing node ID for checkout
                     options = sorted([node_id[:12] for node_id in self.conversation_graph.nodes if node_id.startswith(prefix_to_match)]) # Show prefix
                elif cmd == "cache" and arg_index_to_complete == 0: # Completing cache subcommand
                     options = sorted([s for s in ["list", "clear", "clean", "dependencies", "deps"] if s.startswith(prefix_to_match)])
                elif cmd == "cache" and parts[1] == "clear" and arg_index_to_complete == 1: # Completing tool name for clear
                     options = sorted([name for name in self.server_manager.tools if name.startswith(prefix_to_match)])
                     if "--all".startswith(prefix_to_match): options.append("--all")
                elif cmd == "import" and arg_index_to_complete == 0: # Basic file completion
                    options = [f for f in os.listdir('.') if f.startswith(prefix_to_match) and f.endswith('.json')]

            # --- Return matching option based on state ---
            if state < len(options):
                return options[state]
            else:
                return None

        except Exception as e:
            log.error(f"Error during completion: {e}", exc_info=True)
            return None # Fail gracefully

        return None

    # --- Configuration Commands ---
    async def cmd_config(self, args):
        """Handle configuration commands (CLI interface)."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "show"
        subargs = parts[1] if len(parts) > 1 else ""

        config_changed = False # Flag to track if we need to save

        # --- Show Command ---
        if subcmd == "show":
            safe_console.print("\n[bold cyan]Current Configuration:[/]")
            display_data = {}
            for key, value in self.config.__dict__.items():
                # Skip internal attributes and complex structures handled separately
                if key.startswith("_") or key in ["servers", "cache_ttl_mapping"]:
                    continue
                # Mask API keys
                if "api_key" in key and isinstance(value, str) and value:
                    display_data[key] = f"***{value[-4:]}"
                # Handle lists for display
                elif isinstance(value, list):
                     display_data[key] = ", ".join(map(str, value)) if value else "[Empty List]"
                else:
                    display_data[key] = value

            # Print simple settings from display_data
            config_table = Table(box=box.ROUNDED, show_header=False)
            config_table.add_column("Setting", style="dim")
            config_table.add_column("Value")
            for key, value in sorted(display_data.items()):
                 config_table.add_row(key, str(value))
            safe_console.print(Panel(config_table, title="Settings", border_style="blue"))

            # Print servers
            if self.config.servers:
                 server_table = Table(box=box.ROUNDED, title="Servers (from config.yaml)")
                 server_table.add_column("Name")
                 server_table.add_column("Type")
                 server_table.add_column("Path/URL")
                 server_table.add_column("Enabled")
                 for name, server in self.config.servers.items():
                      server_table.add_row(name, server.type.value, server.path, str(server.enabled))
                 safe_console.print(server_table)
            else:
                 safe_console.print(Panel("[dim]No servers defined in config.yaml[/]", title="Servers", border_style="dim green"))

            # Print TTL mapping
            if self.config.cache_ttl_mapping:
                 ttl_table = Table(box=box.ROUNDED, title="Cache TTLs (from config.yaml)")
                 ttl_table.add_column("Tool Category/Name")
                 ttl_table.add_column("TTL (seconds)")
                 for name, ttl in self.config.cache_ttl_mapping.items(): ttl_table.add_row(name, str(ttl))
                 safe_console.print(ttl_table)
            else:
                 safe_console.print(Panel("[dim]No custom cache TTLs defined in config.yaml[/]", title="Cache TTLs", border_style="dim yellow"))
            return

        # --- Edit Command (Opens YAML file) ---
        elif subcmd == "edit":
             editor = os.environ.get("EDITOR", "vim")
             safe_console.print(f"Opening {CONFIG_FILE} in editor ('{editor}')...")
             safe_console.print("[yellow]Note: Changes require app restart or setting via '/config' command to take full effect.[/]")
             try:
                 # Ensure file exists for editor
                 CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
                 CONFIG_FILE.touch()
                 process = await asyncio.create_subprocess_exec(editor, str(CONFIG_FILE))
                 await process.wait()
                 if process.returncode == 0:
                     safe_console.print(f"[green]Editor closed. Reloading configuration from YAML...[/]")
                     # Reload the entire config state from YAML after editing
                     self.config.load_from_yaml()
                     # Re-apply env overrides to ensure they still take precedence
                     self.config._apply_env_overrides()
                     # Re-initialize provider clients based on potentially changed keys/URLs in YAML
                     await self._reinitialize_provider_clients()
                     safe_console.print("[green]Config reloaded from YAML (Env vars still override).[/]")
                 else:
                     safe_console.print(f"[yellow]Editor closed with code {process.returncode}. No changes loaded.[/]")
             except Exception as e:
                 safe_console.print(f"[red]Failed to open editor: {e}[/]")
             return # Edit doesn't trigger explicit save

        # --- Reset Command ---
        elif subcmd == "reset":
            if Confirm.ask("[bold yellow]Reset ALL settings to defaults and save to config.yaml? This erases current config file content. Env overrides still apply on next start.[/]", console=safe_console):
                 try:
                     if hasattr(self, 'server_manager'): await self.server_manager.close()
                     self.config = Config() # Re-initializes (defaults -> empty yaml -> env overrides)
                     await self.config.save_async() # Save the effective state to YAML
                     self.server_manager = ServerManager(self.config, self.tool_cache, self.safe_print)
                     await self._reinitialize_provider_clients()
                     safe_console.print("[green]Configuration reset to defaults and saved to config.yaml.[/]")
                 except Exception as e: safe_console.print(f"[red]Error during reset: {e}[/]")
            else: safe_console.print("[yellow]Reset cancelled.[/]")
            return # Reset handles its own save   

        # --- Subcommands for Specific Settings ---
        elif subcmd == "api-key":
            key_parts = subargs.split(maxsplit=1)
            if len(key_parts) != 2:
                safe_console.print("[yellow]Usage: /config api-key <provider> <api_key>[/]")
                return
            provider_str, key_value = key_parts[0].lower(), key_parts[1]
            try:
                provider = Provider(provider_str)
                attr_name = PROVIDER_CONFIG_KEY_ATTR_MAP.get(provider.value)
                if attr_name:
                    setattr(self.config, attr_name, key_value)
                    config_changed = True
                    safe_console.print(f"[green]API key set for {provider.value} (will save to config.yaml).[/]")
                    await self._reinitialize_provider_clients(providers=[provider.value])
                else: safe_console.print(f"[red]Invalid provider for API key: {provider_str}[/]")
            except ValueError: safe_console.print(f"[red]Invalid provider: {provider_str}[/]")

        elif subcmd == "base-url":
            url_parts = subargs.split(maxsplit=1)
            if len(url_parts) != 2:
                safe_console.print("[yellow]Usage: /config base-url <provider> <url>[/]")
                return
            provider_str, url_value = url_parts[0].lower(), url_parts[1]
            try:
                provider = Provider(provider_str)
                attr_name = PROVIDER_CONFIG_URL_ATTR_MAP.get(provider.value)
                if attr_name:
                    setattr(self.config, attr_name, url_value)
                    config_changed = True
                    safe_console.print(f"[green]Base URL set for {provider.value} (will save to config.yaml).[/]")
                    await self._reinitialize_provider_clients(providers=[provider.value])
                else: safe_console.print(f"[red]Base URL config not supported/invalid provider: {provider_str}[/]")
            except ValueError: safe_console.print(f"[red]Invalid provider: {provider_str}[/]")

        elif subcmd == "model":
             if not subargs:
                 safe_console.print("[yellow]Usage: /config model <model_name>[/]")
                 return
             self.config.default_model = subargs
             self.current_model = subargs
             config_changed = True
             safe_console.print(f"[green]Default model set to: {subargs}[/]")
        elif subcmd == "max-tokens":
             try:
                 self.config.default_max_tokens = int(subargs)
                 config_changed = True
                 safe_console.print(f"[green]Default max tokens set to: {subargs}[/]")
             except (ValueError, TypeError):
                 safe_console.print("[yellow]Usage: /config max-tokens <number>[/]")
        elif subcmd == "history-size":
             try:
                 new_size = int(subargs)
                 if new_size <= 0: raise ValueError("Must be positive")
                 self.config.history_size = new_size
                 self.history = History(max_entries=new_size) # Recreate history
                 config_changed = True
                 safe_console.print(f"[green]History size set to: {new_size}[/]")
             except (ValueError, TypeError):
                 safe_console.print("[yellow]Usage: /config history-size <positive_number>[/]")
        elif subcmd == "temperature":
             try:
                 temp = float(subargs)
                 if 0.0 <= temp <= 2.0:
                     self.config.temperature = temp
                     config_changed = True
                     safe_console.print(f"[green]Temperature set to: {temp}[/]")
                 else:
                     safe_console.print("[red]Temperature must be between 0.0 and 2.0[/]")
             except (ValueError, TypeError): safe_console.print("[yellow]Usage: /config temperature <number_between_0_and_2>[/]")

        # Boolean Flags (using helper)
        elif subcmd in [s.replace("enable_", "").replace("use_", "").replace("_", "-") for s in SIMPLE_SETTINGS_ENV_MAP.keys() if type(getattr(Config(), SIMPLE_SETTINGS_ENV_MAP[s])) is bool]:
             # Find the attribute name corresponding to the command
             attr_to_set = None
             for env_key, attr_name in SIMPLE_SETTINGS_ENV_MAP.items():
                 command_name = attr_name.replace("enable_", "").replace("use_", "").replace("_", "-")
                 if command_name == subcmd:
                     if type(getattr(self.config, attr_name)) is bool:
                         attr_to_set = attr_name
                         break
             if attr_to_set:
                 if not subargs:
                     safe_console.print(f"Current {attr_to_set}: {getattr(self.config, attr_to_set)}")
                     return
                 config_changed = self._set_bool_config(attr_to_set, subargs)
             else:
                 safe_console.print(f"[red]Internal error finding boolean attribute for command '{subcmd}'[/]") # Should not happen

        # Delegated subcommands
        elif subcmd == "port-scan": config_changed = await self._handle_config_port_scan(subargs)
        elif subcmd == "discovery-path": config_changed = await self._handle_config_discovery_path(subargs)
        elif subcmd == "registry-urls": config_changed = await self._handle_config_registry_urls(subargs)
        elif subcmd == "cache-ttl": config_changed = await self._handle_config_cache_ttl(subargs)

        else:
            safe_console.print(f"[yellow]Unknown config command: {subcmd}[/]")
            # List available simple config subcommands dynamically
            simple_bool_cmds = [a.replace("enable_", "").replace("use_", "").replace("_", "-") for a in SIMPLE_SETTINGS_ENV_MAP.values() if type(getattr(Config(), a)) is bool]
            simple_value_cmds = ["model", "max-tokens", "history-size", "temperature"]
            provider_cmds = ["api-key", "base-url"]
            complex_cmds = ["port-scan", "discovery-path", "registry-urls", "cache-ttl"]
            meta_cmds = ["show", "edit", "reset"]
            all_cmds = sorted(meta_cmds + simple_value_cmds + simple_bool_cmds + provider_cmds + complex_cmds)
            safe_console.print(f"Available subcommands: {', '.join(all_cmds)}")


        # --- Save if Changed ---
        if config_changed:
            await self.config.save_async()
            safe_console.print("[italic green](Configuration saved to config.yaml)[/]")

    # --- Helper for boolean config setting ---
    def _set_bool_config(self, attr_name: str, value_str: str) -> bool:
        """Sets a boolean config attribute and prints status. Returns True if changed."""
        safe_console = get_safe_console()
        current_value = getattr(self.config, attr_name, None)
        new_value = None
        changed = False

        if value_str.lower() in ("true", "yes", "on", "1"):
            new_value = True
        elif value_str.lower() in ("false", "no", "off", "0"):
            new_value = False
        else:
            safe_console.print(f"[yellow]Usage: /config {attr_name.replace('_', '-')} [true|false][/]")
            return False # Not changed

        if new_value != current_value:
            setattr(self.config, attr_name, new_value)
            changed = True

        # Print status regardless of change, showing the final value
        status_text = "enabled" if new_value else "disabled"
        color = "green" if new_value else "yellow"
        safe_console.print(f"[{color}]{attr_name.replace('_', ' ').capitalize()} {status_text}.[/]")
        return changed

    # --- Helper for port scan subcommands ---
    async def _handle_config_port_scan(self, args) -> bool:
        """Handles /config port-scan ... subcommands. Returns True if config changed."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "show"
        value = parts[1] if len(parts) > 1 else ""
        config_changed = False

        if action == "enable":
            if not value:
                safe_console.print("[yellow]Usage: /config port-scan enable [true|false][/]")
                return False
            config_changed = self._set_bool_config("enable_port_scanning", value)
        elif action == "range":
            range_parts = value.split()
            if len(range_parts) == 2 and range_parts[0].isdigit() and range_parts[1].isdigit():
                start, end = int(range_parts[0]), int(range_parts[1])
                if 0 <= start <= end <= 65535:
                    if self.config.port_scan_range_start != start or self.config.port_scan_range_end != end:
                        self.config.port_scan_range_start = start
                        self.config.port_scan_range_end = end
                        config_changed = True
                    safe_console.print(f"[green]Port scan range set to {start}-{end}[/]")
                else: safe_console.print("[red]Invalid port range (0-65535, start <= end).[/]")
            else: safe_console.print("[yellow]Usage: /config port-scan range START END[/]")
        elif action == "targets":
            targets = [t.strip() for t in value.split(',') if t.strip()]
            if targets:
                if set(self.config.port_scan_targets) != set(targets):
                     self.config.port_scan_targets = targets
                     config_changed = True
                safe_console.print(f"[green]Port scan targets set to: {', '.join(targets)}[/]")
            else: safe_console.print("[yellow]Usage: /config port-scan targets ip1,ip2,...[/]")
        elif action == "concurrency":
            try:
                val = int(value)
                if self.config.port_scan_concurrency != val:
                    self.config.port_scan_concurrency = val
                    config_changed = True
                safe_console.print(f"[green]Port scan concurrency set to: {val}[/]")
            except (ValueError, TypeError):
                safe_console.print("[yellow]Usage: /config port-scan concurrency <number>[/]")
        elif action == "timeout":
            try:
                val = float(value)
                if self.config.port_scan_timeout != val:
                    self.config.port_scan_timeout = val
                    config_changed = True
                safe_console.print(f"[green]Port scan timeout set to: {val}s[/]")
            except (ValueError, TypeError):
                safe_console.print("[yellow]Usage: /config port-scan timeout <seconds>[/]")
        elif action == "show":
             safe_console.print("\n[bold]Port Scanning Settings:[/]")
             safe_console.print(f"  Enabled: {'Yes' if self.config.enable_port_scanning else 'No'}")
             safe_console.print(f"  Range: {self.config.port_scan_range_start} - {self.config.port_scan_range_end}")
             safe_console.print(f"  Targets: {', '.join(self.config.port_scan_targets)}")
             safe_console.print(f"  Concurrency: {self.config.port_scan_concurrency}")
             safe_console.print(f"  Timeout: {self.config.port_scan_timeout}s")
        else: safe_console.print("[yellow]Unknown port-scan command. Use: enable, range, targets, concurrency, timeout, show[/]")

        # Return whether changes were made
        return config_changed

    # --- Helper for discovery path subcommands ---
    async def _handle_config_discovery_path(self, args) -> bool:
        """Handles /config discovery-path ... subcommands. Returns True if config changed."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "list"
        path = parts[1] if len(parts) > 1 else ""
        config_changed = False

        current_paths = self.config.discovery_paths.copy() # Work on a copy

        if action == "add" and path:
            if path not in current_paths:
                current_paths.append(path)
                config_changed = True
                safe_console.print(f"[green]Added discovery path: {path}[/]")
            else: safe_console.print(f"[yellow]Path already exists: {path}[/]")
        elif action == "remove" and path:
            if path in current_paths:
                current_paths.remove(path)
                config_changed = True
                safe_console.print(f"[green]Removed discovery path: {path}[/]")
            else: safe_console.print(f"[yellow]Path not found: {path}[/]")
        elif action == "list":
            safe_console.print("\n[bold]Discovery Paths:[/]")
            if self.config.discovery_paths:
                for i, p in enumerate(self.config.discovery_paths, 1): safe_console.print(f" {i}. {p}")
            else: safe_console.print("  [dim]No paths configured.[/]")
        else: safe_console.print("[yellow]Usage: /config discovery-path [add|remove|list] [PATH][/]")

        if config_changed:
            self.config.discovery_paths = current_paths # Update the actual config list

        return config_changed

    # --- Helper for registry URLs subcommands ---
    async def _handle_config_registry_urls(self, args) -> bool:
        """Handles /config registry-urls ... subcommands. Returns True if config changed."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "list"
        url = parts[1] if len(parts) > 1 else ""
        config_changed = False

        current_urls = self.config.registry_urls.copy() # Work on a copy

        if action == "add" and url:
            if url not in current_urls:
                current_urls.append(url)
                config_changed = True
                safe_console.print(f"[green]Added registry URL: {url}[/]")
            else: safe_console.print(f"[yellow]URL already exists: {url}[/]")
        elif action == "remove" and url:
            if url in current_urls:
                current_urls.remove(url)
                config_changed = True
                safe_console.print(f"[green]Removed registry URL: {url}[/]")
            else: safe_console.print(f"[yellow]URL not found: {url}[/]")
        elif action == "list":
             safe_console.print("\n[bold]Registry URLs:[/]")
             if self.config.registry_urls:
                 for i, u in enumerate(self.config.registry_urls, 1): safe_console.print(f" {i}. {u}")
             else: safe_console.print("  [dim]No URLs configured.[/]")
        else: safe_console.print("[yellow]Usage: /config registry-urls [add|remove|list] [URL][/]")

        if config_changed:
            self.config.registry_urls = current_urls # Update the actual config list

        return config_changed

    # --- Helper for Cache TTL subcommands ---
    async def _handle_config_cache_ttl(self, args) -> bool:
        """Handles /config cache-ttl ... subcommands. Returns True if config changed."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "list"
        params = parts[1] if len(parts) > 1 else ""
        config_changed = False

        if action == "set" and params:
            kv_parts = params.split(maxsplit=1)
            if len(kv_parts) == 2:
                tool_key, ttl_str = kv_parts[0], kv_parts[1]
                try:
                    ttl = int(ttl_str)
                    if ttl < 0: ttl = -1 # Allow negative for never expire
                    if self.config.cache_ttl_mapping.get(tool_key) != ttl:
                        self.config.cache_ttl_mapping[tool_key] = ttl
                        if self.tool_cache: self.tool_cache.ttl_mapping[tool_key] = ttl
                        config_changed = True
                    safe_console.print(f"[green]Set TTL for '{tool_key}' to {ttl} seconds.[/]")
                except ValueError: safe_console.print("[red]Invalid TTL value, must be an integer.[/]")
            else: safe_console.print("[yellow]Usage: /config cache-ttl set <tool_category_or_name> <ttl_seconds>[/]")
        elif action == "remove" and params:
            tool_key = params
            if tool_key in self.config.cache_ttl_mapping:
                del self.config.cache_ttl_mapping[tool_key]
                if self.tool_cache and tool_key in self.tool_cache.ttl_mapping:
                     del self.tool_cache.ttl_mapping[tool_key]
                config_changed = True
                safe_console.print(f"[green]Removed custom TTL for '{tool_key}'.[/]")
            else: safe_console.print(f"[yellow]No custom TTL found for '{tool_key}'.[/]")
        elif action == "list":
             safe_console.print("\n[bold]Custom Cache TTLs (from config.yaml):[/]")
             if self.config.cache_ttl_mapping:
                 ttl_table = Table(box=box.ROUNDED)
                 ttl_table.add_column("Tool Category/Name")
                 ttl_table.add_column("TTL (seconds)")
                 for name, ttl in self.config.cache_ttl_mapping.items():
                     ttl_table.add_row(name, str(ttl))
                 safe_console.print(ttl_table)
             else: safe_console.print("  [dim]No custom TTLs defined.[/]")
        else: safe_console.print("[yellow]Usage: /config cache-ttl [set|remove|list] [PARAMS...][/]")

        # Return if config was changed (for outer save logic)
        return config_changed

    # --- Helper to reinitialize provider clients ---
    async def _reinitialize_provider_clients(self, providers: Optional[List[str]] = None):
        """Re-initializes specific or all provider SDK clients based on current config."""
        providers_to_init = providers or [p.value for p in Provider]
        log.info(f"Re-initializing provider clients for: {providers_to_init}")
        msgs = []

        # Define provider details map inside method or load from class/global scope
        provider_details_map = {
            Provider.OPENAI.value: {"key_attr": "openai_api_key", "url_attr": "openai_base_url", "default_url": "https://api.openai.com/v1", "client_attr": "openai_client"},
            Provider.GROK.value: {"key_attr": "grok_api_key", "url_attr": "grok_base_url", "default_url": "https://api.x.ai/v1", "client_attr": "grok_client"},
            Provider.DEEPSEEK.value: {"key_attr": "deepseek_api_key", "url_attr": "deepseek_base_url", "default_url": "https://api.deepseek.com/v1", "client_attr": "deepseek_client"},
            Provider.MISTRAL.value: {"key_attr": "mistral_api_key", "url_attr": "mistral_base_url", "default_url": "https://api.mistral.ai/v1", "client_attr": "mistral_client"},
            Provider.GROQ.value: {"key_attr": "groq_api_key", "url_attr": "groq_base_url", "default_url": "https://api.groq.com/openai/v1", "client_attr": "groq_client"},
            Provider.CEREBRAS.value: {"key_attr": "cerebras_api_key", "url_attr": "cerebras_base_url", "default_url": "https://api.cerebras.ai/v1", "client_attr": "cerebras_client"},
            Provider.GEMINI.value: {"key_attr": "gemini_api_key", "url_attr": "gemini_base_url", "default_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "client_attr": "gemini_client"},
            Provider.OPENROUTER.value: {"key_attr": "openrouter_api_key", "url_attr": "openrouter_base_url", "default_url": "https://openrouter.ai/api/v1", "client_attr": "openrouter_client"}
        }

        # Anthropic (Special Case)
        if Provider.ANTHROPIC.value in providers_to_init:
            anthropic_key = self.config.anthropic_api_key
            emoji = EMOJI_MAP.get(Provider.ANTHROPIC.value, "")
            if anthropic_key:
                try:
                    # Close existing client if it exists and has aclose
                    if self.anthropic and hasattr(self.anthropic, 'aclose'):
                         await self.anthropic.aclose()
                    self.anthropic = AsyncAnthropic(api_key=anthropic_key)
                    msgs.append(f"{emoji} Anthropic: [green]OK[/]")
                except Exception as e:
                    log.error(f"Error re-initializing Anthropic: {e}")
                    self.anthropic = None
                    msgs.append(f"{emoji} Anthropic: [red]Failed[/]")
            else:
                if self.anthropic and hasattr(self.anthropic, 'aclose'): await self.anthropic.aclose() # Close if key removed
                self.anthropic = None
                msgs.append(f"{emoji} Anthropic: [yellow]No Key[/]")

        # OpenAI Compatible Providers
        for provider_value, details in provider_details_map.items():
            if provider_value in providers_to_init:
                # Close existing client instance before creating a new one
                client_attr = details["client_attr"]
                existing_client = getattr(self, client_attr, None)
                if existing_client and hasattr(existing_client, 'aclose'):
                    try: await existing_client.aclose()
                    except Exception as close_err: log.warning(f"Error closing existing {provider_value} client: {close_err}")
                # Initialize new client
                _, status_msg = await self._initialize_openai_compatible_client(
                    provider_name=provider_value,
                    api_key_attr=details["key_attr"],
                    base_url_attr=details["url_attr"],
                    default_base_url=details["default_url"],
                    client_attr=client_attr,
                    emoji_key=provider_value
                )
                msgs.append(status_msg)

        self.safe_print(f"Provider clients re-initialized: {' | '.join(msgs)}")

    def generate_dashboard_renderable(self) -> Layout:
        """Generates the Rich renderable Layout for the live dashboard."""

        layout = Layout(name="root")

        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=1),
        )

        layout["main"].split_row(
            Layout(name="servers", ratio=2),
            Layout(name="sidebar", ratio=1),
        )

        layout["sidebar"].split(
             Layout(name="tools", ratio=1),
             # Allocate fixed size for stats, adjust as needed
             Layout(name="stats", size=10),
        )

        # --- Header ---
        header_text = Text(
            f"MCP Client Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            style="bold white on blue", justify="center"
            )
        layout["header"].update(Panel(header_text, border_style="blue")) # Simpler panel for header

        # --- Footer ---
        layout["footer"].update(Text("Press Ctrl+C to exit dashboard", style="dim", justify="center"))

        # --- Servers Panel ---
        server_table = Table(
            title=f"{EMOJI_MAP['server']} Servers",
            box=box.ROUNDED, border_style="blue", show_header=True, header_style="bold blue"
            )
        server_table.add_column("Name", style="cyan", no_wrap=True)
        server_table.add_column("Status", justify="center")
        server_table.add_column("Type", style="dim")
        server_table.add_column("Conn", justify="center") # Abbreviated
        server_table.add_column("Avg Resp (ms)", justify="right", style="yellow")
        server_table.add_column("Errors", justify="right", style="red")
        server_table.add_column("Reqs", justify="right", style="green") # Abbreviated

        # Sort servers by name for consistent display
        sorted_server_names = sorted(self.config.servers.keys())

        for name in sorted_server_names:
            server_config = self.config.servers[name]
            # Optionally skip disabled servers, or show them dimmed
            if not server_config.enabled:
                # Example: show disabled server dimmed
                # server_table.add_row(f"[dim]{name}[/]", "[dim]Disabled[/]", f"[dim]{server_config.type.value}[/]", "-", "-", "-", "-")
                continue # Skip disabled for now

            metrics = server_config.metrics
            is_connected = name in self.server_manager.active_sessions

            # Connection Status Emoji
            conn_emoji_key = "green_circle" if is_connected else "red_circle"
            conn_status_emoji = EMOJI_MAP.get(conn_emoji_key, "?")

            # Health Status Emoji and Style
            health_status_key = f"status_{metrics.status.value}"
            health_status_emoji = EMOJI_MAP.get(health_status_key, EMOJI_MAP["question_mark"])
            status_style = f"status.{metrics.status.value}" if metrics.status != ServerStatus.UNKNOWN else "dim"
            health_text = Text(f"{health_status_emoji} {metrics.status.value.capitalize()}", style=status_style)

            # Format Metrics
            avg_resp_ms = metrics.avg_response_time * 1000 if metrics.request_count > 0 else 0
            error_count_str = str(metrics.error_count) if metrics.error_count > 0 else "-"
            req_count_str = str(metrics.request_count) if metrics.request_count > 0 else "-"

            server_table.add_row(
                name,
                health_text,
                server_config.type.value,
                conn_status_emoji,
                f"{avg_resp_ms:.1f}" if avg_resp_ms > 0 else "-",
                error_count_str,
                req_count_str
            )
        layout["servers"].update(Panel(server_table, title="[bold blue]MCP Servers[/]", border_style="blue"))

        # --- Tools Panel (Top N) ---
        tool_table = Table(
            title=f"{EMOJI_MAP['tool']} Tool Usage",
            box=box.ROUNDED, border_style="magenta", show_header=True, header_style="bold magenta"
            )
        tool_table.add_column("Name", style="magenta", no_wrap=True)
        tool_table.add_column("Server", style="blue")
        tool_table.add_column("Calls", justify="right", style="green")
        tool_table.add_column("Avg Time (ms)", justify="right", style="yellow")

        # Sort tools by call count descending and take top N
        sorted_tools = sorted(self.server_manager.tools.values(), key=lambda t: t.call_count, reverse=True)[:15]

        for tool in sorted_tools:
             # Show short name (part after ':')
             tool_short_name = tool.name.split(':')[-1] if ':' in tool.name else tool.name
             avg_time_ms = tool.avg_execution_time # Assuming this is already in ms
             tool_table.add_row(
                 tool_short_name,
                 tool.server_name,
                 str(tool.call_count),
                 f"{avg_time_ms:.1f}" if tool.call_count > 0 else "-"
             )
        layout["tools"].update(Panel(tool_table, title="[bold magenta]Tool Usage (Top 15)[/]", border_style="magenta"))

        # --- General Stats Panel ---
        stats_lines = []
        stats_lines.append(Text.assemble((f"{EMOJI_MAP['model']} Model: ", "dim"), (self.current_model, "cyan")))
        stats_lines.append(Text.assemble((f"{EMOJI_MAP['server']} Connected: ", "dim"), f"{len(self.server_manager.active_sessions)} / {len(self.config.servers)}"))
        stats_lines.append(Text.assemble((f"{EMOJI_MAP['tool']} Tools: ", "dim"), str(len(self.server_manager.tools))))
        stats_lines.append(Text.assemble((f"{EMOJI_MAP['history']} History: ", "dim"), f"{len(self.history.entries)} entries"))

        # Cache Stats
        mem_cache_count = 0
        disk_cache_count = 0
        if self.tool_cache:
             mem_cache_count = len(self.tool_cache.memory_cache)
             if self.tool_cache.disk_cache:
                  try: disk_cache_count = sum(1 for _ in self.tool_cache.disk_cache.iterkeys())
                  except Exception: pass # Ignore error counting disk
        stats_lines.append(Text.assemble((f"{EMOJI_MAP['cache']} Tool Cache: ", "dim"), f"{mem_cache_count} (Mem) / {disk_cache_count} (Disk)"))

        # LLM Prompt Cache Stats (Overall)
        cache_hits = getattr(self, 'cache_hit_count', 0)
        cache_misses = getattr(self, 'cache_miss_count', 0)
        total_lookups = cache_hits + cache_misses
        if total_lookups > 0:
             hit_rate = (cache_hits / total_lookups) * 100
             stats_lines.append(Text.assemble(
                  (f"{EMOJI_MAP['package']} LLM Cache: ", "dim"),
                  (f"{hit_rate:.1f}% Hits", "green"),
                  (" (", "dim"), (f"{self.tokens_saved_by_cache:,}", "green bold"), (" Tokens Saved)", "dim")
             ))
        else:
             stats_lines.append(Text.assemble((f"{EMOJI_MAP['package']} LLM Cache: ", "dim"), ("No usage yet", "dim")))

        # Conversation Branch
        current_node = self.conversation_graph.current_node
        stats_lines.append(Text.assemble(
            (f"{EMOJI_MAP['trident_emblem']} Branch: ", "dim"),
            (current_node.name, "yellow"),
            (f" ({current_node.id[:8]})", "dim cyan")
            ))

        # Use Group for layout within the panel
        stats_group = Group(*stats_lines)
        layout["stats"].update(Panel(stats_group, title="[bold cyan]Client Info[/]", border_style="cyan"))

        return layout

    # --- Server Management Commands ---
    async def cmd_servers(self, args):
        """Handle server management commands (CLI)."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"
        subargs = parts[1] if len(parts) > 1 else ""

        # Helper to parse server names (allows comma-separated or space-separated)
        def parse_server_names(names_str: str) -> List[str]:
            if not names_str:
                return []
            # Split by comma first, then flatten and split by space if needed
            names = [name.strip() for part in names_str.split(',') for name in part.split() if name.strip()]
            return list(dict.fromkeys(names)) # Remove duplicates while preserving order

        server_names_to_action = parse_server_names(subargs)

        try:
            if subcmd == "list":
                await self.list_servers() # Use the dedicated listing method
            elif subcmd == "add":
                # Requires multiple args: NAME TYPE PATH [ARGS...]
                await self.add_server(subargs) # Delegate parsing to add_server
            elif subcmd == "remove":
                if not server_names_to_action:
                    safe_console.print("[yellow]Usage: /servers remove <name1> [name2...] [/]")
                    return
                removed_count = 0
                for name in server_names_to_action:
                    if await self.remove_server(name): # remove_server returns True on success
                         removed_count += 1
                if removed_count > 0:
                    safe_console.print(f"[green]Removed {removed_count} server(s).[/]")
                    await self.config.save_async() # Save config after successful removals
            elif subcmd == "connect":
                if not server_names_to_action:
                    safe_console.print("[yellow]Usage: /servers connect <name1> [name2...] | all [/]")
                    return
                names_to_connect = server_names_to_action
                if "all" in names_to_connect:
                    names_to_connect = [name for name, cfg in self.config.servers.items() if cfg.enabled and name not in self.server_manager.active_sessions]
                if not names_to_connect:
                    safe_console.print("[yellow]No servers specified or all enabled servers already connected.[/]")
                    return

                connect_tasks = []
                for name in names_to_connect:
                    if name in self.config.servers and name not in self.server_manager.active_sessions:
                         # Directly append the coroutine object
                         connect_tasks.append(self.connect_server(name))
                    elif name in self.server_manager.active_sessions:
                         safe_console.print(f"[dim]Server '{name}' already connected.[/]")
                    else:
                        safe_console.print(f"[red]Server '{name}' not found.[/]")

                if connect_tasks:
                    results = await asyncio.gather(*connect_tasks, return_exceptions=True)
                    success_count = sum(1 for r in results if isinstance(r, bool) and r)
                    fail_count = len(results) - success_count
                    safe_console.print(f"[green]Connection process finished. Success: {success_count}, Failed: {fail_count}[/]")
            elif subcmd == "disconnect":
                if not server_names_to_action:
                    safe_console.print("[yellow]Usage: /servers disconnect <name1> [name2...] | all [/]")
                    return
                names_to_disconnect = server_names_to_action
                if "all" in names_to_disconnect:
                    names_to_disconnect = list(self.server_manager.active_sessions.keys())
                if not names_to_disconnect:
                    safe_console.print("[yellow]No connected servers specified or found.[/]")
                    return

                disconnect_tasks = [self.disconnect_server(name) for name in names_to_disconnect if name in self.server_manager.active_sessions]
                if disconnect_tasks:
                     await asyncio.gather(*disconnect_tasks, return_exceptions=True)
                     safe_console.print(f"[green]Disconnected from {len(disconnect_tasks)} server(s).[/]")
            elif subcmd == "enable":
                if not server_names_to_action:
                    safe_console.print("[yellow]Usage: /servers enable <name1> [name2...] [/]")
                    return
                changed = await self.enable_server(server_names_to_action, True)
                if changed:
                    await self.config.save_async()
            elif subcmd == "disable":
                if not server_names_to_action:
                    safe_console.print("[yellow]Usage: /servers disable <name1> [name2...] [/]")
                    return
                changed = await self.enable_server(server_names_to_action, False)
                if changed:
                    await self.config.save_async()
            elif subcmd == "status":
                if not server_names_to_action:
                    safe_console.print("[yellow]Usage: /servers status <name1> [name2...] [/]")
                    return
                for name in server_names_to_action:
                    await self.server_status(name) # Show status for each
            else:
                safe_console.print("[yellow]Unknown servers command. Available: list, add, remove, connect, disconnect, enable, disable, status[/]")
        except Exception as e:
            log.error(f"Error processing servers command '/servers {args}': {e}", exc_info=True)
            safe_console.print(f"[red]Error processing command: {e}[/]")

    async def list_servers(self):
        """Lists all configured servers with status details."""
        safe_console = get_safe_console()
        if not self.config.servers:
            safe_console.print(f"{EMOJI_MAP['warning']} [yellow]No servers configured[/]")
            return

        server_table = Table(title=f"{EMOJI_MAP['server']} Configured Servers", box=box.ROUNDED, show_lines=True)
        server_table.add_column("Name", style="bold blue")
        server_table.add_column("Type", style="dim")
        server_table.add_column("Path/URL")
        server_table.add_column("Status", justify="center")
        server_table.add_column("Health", justify="center")
        server_table.add_column("Enabled", justify="center")
        server_table.add_column("Tools", justify="right")

        sorted_names = sorted(self.config.servers.keys())
        for name in sorted_names:
            server = self.config.servers[name]
            is_connected = name in self.server_manager.active_sessions
            metrics = server.metrics
            health_status = metrics.status
            health_emoji = EMOJI_MAP.get(f"status_{health_status.value}", EMOJI_MAP["question_mark"])
            health_style = f"status.{health_status.value}" if health_status != ServerStatus.UNKNOWN else "dim"

            # Calculate health score for display
            health_score = 0
            if is_connected and metrics.request_count > 0:
                 health_penalty = (metrics.error_rate * 100) + max(0, (metrics.avg_response_time - 1.0) * 10)
                 health_score = max(0, min(100, int(100 - health_penalty)))

            conn_status_text = Text.assemble(
                (EMOJI_MAP["green_circle"], "green") if is_connected else (EMOJI_MAP["red_circle"], "red"),
                f" {'Conn' if is_connected else 'Disc'}"
            )
            enabled_text = Text.assemble(
                (EMOJI_MAP["success"], "green") if server.enabled else (EMOJI_MAP["error"], "red"),
            )
            health_display = Text.assemble((health_emoji, health_style), f" {health_status.value.capitalize()} ({health_score}%)")

            tools_count = sum(1 for t in self.server_manager.tools.values() if t.server_name == name)

            server_table.add_row(
                name, server.type.value, server.path, conn_status_text,
                health_display, enabled_text, str(tools_count)
            )
        safe_console.print(server_table)

    async def add_server(self, args: str):
        """Adds a new server configuration based on user input."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=2) # Split into name, type, rest
        if len(parts) < 3:
            safe_console.print("[yellow]Usage: /servers add NAME <stdio|sse> PATH [ARGS...][/]")
            safe_console.print("Example (stdio): /servers add mypy stdio /usr/bin/python /path/to/mypy_server.py --port 8080")
            safe_console.print("Example (sse):   /servers add remote-tools sse https://my-mcp.example.com")
            return

        name, type_str, path_and_args = parts[0], parts[1], parts[2]

        # Validate name and type
        if name in self.config.servers:
            safe_console.print(f"[red]Server name '{name}' already exists.[/]")
            return
        try:
            server_type = ServerType(type_str.lower())
        except ValueError:
            safe_console.print(f"[red]Invalid server type: {type_str}. Use 'stdio' or 'sse'.[/]")
            return

        path = path_and_args
        extra_args = []
        if server_type == ServerType.STDIO:
             # If stdio, split path_and_args into path and the rest as args
             stdio_parts = path_and_args.split(maxsplit=1)
             path = stdio_parts[0]
             if len(stdio_parts) > 1:
                 # Basic arg splitting, consider using shlex for robustness if needed
                 extra_args = stdio_parts[1].split()

        # Create and add config
        new_config = ServerConfig(
            name=name, type=server_type, path=path, args=extra_args,
            enabled=True, auto_start=True, # Defaults for new adds
            description=f"Added via CLI ({datetime.now().strftime('%Y-%m-%d')})"
        )
        self.config.servers[name] = new_config
        await self.config.save_async() # Save YAML
        safe_console.print(f"[green]{EMOJI_MAP['success']} Server '{name}' added to configuration.[/]")

        # Offer to connect
        if Confirm.ask(f"Connect to server '{name}' now?", default=True, console=safe_console):
            await self.connect_server(name) # Call connect helper

    async def remove_server(self, name: str) -> bool:
        """Removes a server configuration. Returns True if removed, False otherwise."""
        safe_console = get_safe_console()
        if not name:
            safe_console.print("[yellow]Usage: /servers remove <name>[/]")
            return False
        if name not in self.config.servers:
            safe_console.print(f"[red]Server '{name}' not found.[/]")
            return False

        if name in self.server_manager.active_sessions:
             await self.server_manager.disconnect_server(name) # Disconnect first

        del self.config.servers[name]
        # Note: config save is handled by the caller (cmd_servers) after all removals
        safe_console.print(f"[green]Server '{name}' removed from configuration.[/]")
        return True

    async def connect_server(self, name: str) -> bool:
        """Connects to a specific server by name. Returns True on success."""
        safe_console = get_safe_console()
        if not name:
            safe_console.print("[yellow]Usage: /servers connect <name>[/]")
            return False
        if name not in self.config.servers:
            safe_console.print(f"[red]Server '{name}' not found.[/]")
            return False
        if name in self.server_manager.active_sessions:
            safe_console.print(f"[yellow]Server '{name}' is already connected.[/]")
            return True

        server_config = self.config.servers[name]
        if not server_config.enabled:
            safe_console.print(f"[yellow]Server '{name}' is disabled. Enable first with '/servers enable {name}'.[/]")
            return False

        # Use the ServerManager's method
        session = await self.server_manager.connect_to_server(server_config)
        # Status printing is handled within connect_to_server
        return session is not None

    async def disconnect_server(self, name: str) -> bool:
        """Disconnects from a specific server by name. Returns True on success."""
        safe_console = get_safe_console()
        if not name:
            safe_console.print("[yellow]Usage: /servers disconnect <name>[/]")
            return False
        if name not in self.server_manager.active_sessions:
            safe_console.print(f"[yellow]Server '{name}' is not connected.[/]")
            return False

        # Use the ServerManager's method
        await self.server_manager.disconnect_server(name)
        # Status is printed within disconnect_server
        return True

    # --- Local Discovery Monitoring Control ---
    async def start_local_discovery_monitoring(self):
        """Starts the background mDNS listener if enabled."""
        if self.config.enable_local_discovery and self.server_manager and self.server_manager.registry:
            if not self.server_manager.registry.zeroconf:
                log.info("Starting background local discovery listener...")
                self.server_manager.registry.start_local_discovery()
                # Note: We don't explicitly wait here, it runs in background.
            else:
                log.debug("Local discovery listener already running.")
        else:
            log.debug("Local discovery monitoring skipped (disabled or registry unavailable).")

    async def stop_local_discovery_monitoring(self):
        """Stops the background mDNS listener."""
        if self.server_manager and self.server_manager.registry:
            log.info("Stopping background local discovery listener...")
            self.server_manager.registry.stop_local_discovery()             

    async def enable_server(self, names: List[str], enable: bool) -> bool:
        """Enables or disables multiple servers. Returns True if any config changed."""
        safe_console = get_safe_console()
        changed = False
        action_str = "enabled" if enable else "disabled"
        color = "green" if enable else "yellow"

        servers_to_disconnect = []
        servers_to_connect = []

        for name in names:
            if name not in self.config.servers:
                safe_console.print(f"[red]Server '{name}' not found.[/]")
                continue
            server_config = self.config.servers[name]
            if server_config.enabled != enable:
                server_config.enabled = enable
                safe_console.print(f"[{color}]Server '{name}' {action_str}.[/]")
                changed = True
                if not enable and name in self.server_manager.active_sessions:
                     servers_to_disconnect.append(name)
                elif enable and name not in self.server_manager.active_sessions:
                     servers_to_connect.append(name) # Track servers to potentially connect
            else:
                safe_console.print(f"[dim]Server '{name}' already {action_str}.[/]")

        # Disconnect servers that were disabled
        if servers_to_disconnect:
            if Confirm.ask(f"Disconnect {len(servers_to_disconnect)} server(s) now?", default=True, console=safe_console):
                tasks = [self.disconnect_server(n) for n in servers_to_disconnect]
                await asyncio.gather(*tasks, return_exceptions=True)

        # Offer to connect servers that were enabled
        if servers_to_connect:
            if Confirm.ask(f"Connect {len(servers_to_connect)} newly enabled server(s) now?", default=True, console=safe_console):
                tasks = [self.connect_server(n) for n in servers_to_connect]
                await asyncio.gather(*tasks, return_exceptions=True)

        return changed # Caller should save config if True

    async def server_status(self, name: str):
        """Shows detailed status for a specific server."""
        safe_console = get_safe_console()
        if not name:
            safe_console.print("[yellow]Usage: /servers status <name>[/]")
            return

        details = self.get_server_details(name) # Use the helper method
        if not details:
            safe_console.print(f"[red]Server '{name}' not found.[/]")
            return

        # --- Basic Info Panel ---
        basic_info = Group(
            Text(f"[dim]Type:[/dim] {details['type']}"),
            Text(f"[dim]Path/URL:[/dim] {details['path']}"),
            Text(f"[dim]Args:[/dim] {' '.join(details['args']) if details['args'] else '[None]'}"),
            Text(f"[dim]Enabled:[/dim] {'Yes' if details['enabled'] else 'No'}"),
            Text(f"[dim]Auto-Start:[/dim] {'Yes' if details['auto_start'] else 'No'}"),
            Text(f"[dim]Description:[/dim] {details['description'] or '[None]'}"),
            Text(f"[dim]Trusted:[/dim] {'Yes' if details['trusted'] else 'No'}"),
            Text(f"[dim]Categories:[/dim] {', '.join(details['categories']) if details['categories'] else '[None]'}"),
            Text(f"[dim]Version:[/dim] {details['version'] or '[Unknown]'}"),
            Text(f"[dim]Rating:[/dim] {details['rating']:.1f}/5.0"),
            Text(f"[dim]Timeout:[/dim] {details['timeout']:.1f}s"),
            Text(f"[dim]Registry:[/dim] {details['registry_url'] or '[None]'}")
        )
        safe_console.print(Panel(basic_info, title=f"{EMOJI_MAP['server']} Server Info: [bold blue]{name}[/]", border_style="blue", padding=(1, 2)))

        # --- Connection & Health Panel ---
        conn_status = details["metrics"]["status"]
        conn_emoji = EMOJI_MAP.get(f"status_{conn_status}", EMOJI_MAP['question_mark'])
        conn_style = f"status.{conn_status}" if conn_status != "unknown" else "dim"
        connect_info = Group(
             Text.assemble(("Connected: ", "dim"), (EMOJI_MAP["green_circle"], "green") if details['is_connected'] else (EMOJI_MAP["red_circle"], "red")),
             Text.assemble(("Health:", "dim"), (conn_emoji, conn_style), f" {conn_status.capitalize()}"),
             Text(f"[dim]Avg Response:[/dim] {details['metrics']['avg_response_time_ms']:.1f} ms"),
             Text(f"[dim]Requests:[/dim] {details['metrics']['request_count']}"),
             Text(f"[dim]Errors:[/dim] {details['metrics']['error_count']} ({details['metrics']['error_rate']:.1%} rate)"),
             Text(f"[dim]Uptime:[/dim] {details['metrics']['uptime_minutes']:.1f} min"),
             Text(f"[dim]Last Check:[/dim] {datetime.fromisoformat(details['metrics']['last_checked']).strftime('%Y-%m-%d %H:%M:%S') if details['metrics']['last_checked'] else 'N/A'}"),
        )
        safe_console.print(Panel(connect_info, title=f"{EMOJI_MAP['processing']} Connection & Health", border_style="green", padding=(1, 2)))

        # --- Capabilities Panel ---
        caps = details['capabilities']
        cap_info = Group(
             Text(f"[dim]Tools:[/dim] {'Yes' if caps.get('tools') else 'No'}"),
             Text(f"[dim]Resources:[/dim] {'Yes' if caps.get('resources') else 'No'}"),
             Text(f"[dim]Prompts:[/dim] {'Yes' if caps.get('prompts') else 'No'}"),
        )
        safe_console.print(Panel(cap_info, title=f"{EMOJI_MAP['tool']} Capabilities", border_style="magenta", padding=(1, 2)))

        # --- Process Info Panel (for STDIO) ---
        proc_info = details.get('process_info')
        if proc_info:
            if "error" in proc_info:
                proc_content = Text(f"[yellow]Error: {proc_info['error']}[/]")
            else:
                proc_content = Group(
                     Text(f"[dim]PID:[/dim] {proc_info.get('pid', 'N/A')}"),
                     Text(f"[dim]Status:[/dim] {proc_info.get('status', 'N/A')}"),
                     Text(f"[dim]CPU:[/dim] {proc_info.get('cpu_percent', 0.0):.1f}%"),
                     Text(f"[dim]Memory (RSS):[/dim] {proc_info.get('memory_rss_mb', 0.0):.1f} MB"),
                     Text(f"[dim]Create Time:[/dim] {datetime.fromisoformat(proc_info.get('create_time')).strftime('%H:%M:%S') if proc_info.get('create_time') else 'N/A'}"),
                 )
            safe_console.print(Panel(proc_content, title=f"{EMOJI_MAP['gear']} Process Info", border_style="yellow", padding=(1, 2)))


    async def cmd_exit(self, args: str):
        """Exit the client."""
        self.safe_print(f"{EMOJI_MAP['cancel']} Exiting...")
        # Let the main loop's finally block handle client.close()
        sys.exit(0) # Use sys.exit to trigger cleanup via main_async finally

    async def cmd_help(self, args: str):
        """Display help for commands."""
        # Group commands for better organization
        categories = {
            "General": [
                ("help", "Show this help message"),
                ("exit, quit", "Exit the client"),
            ],
            "Configuration": [
                ("config", "Manage config (keys, models, discovery, etc.)"),
                ("model [NAME]", "Change the current AI model"),
                ("cache [...]", "Manage tool result cache"),
            ],
            "Servers & Tools": [
                ("servers [...]", "Manage MCP servers (list, add, connect, etc.)"),
                ("discover [...]", "Discover/connect to local network servers"),
                ("tools [SERVER]", "List available tools (optionally filter by server)"),
                ("tool NAME {JSON}", "Directly execute a tool"),
                ("resources [SERVER]", "List available resources"),
                ("prompts [SERVER]", "List available prompts"),
                ("reload", "Reload MCP servers and capabilities"),
            ],
            "Conversation": [
                ("clear", "Clear conversation messages / reset to root"),
                ("history [N]", "View last N conversation history entries"),
                ("fork [NAME]", "Create a new conversation branch"),
                ("branch [...]", "Manage branches (list, checkout ID)"),
                ("optimize [...]", "Summarize conversation to reduce tokens"),
                ("prompt NAME", "Apply a prompt template"),
                ("export [...]", "Export conversation branch to file"),
                ("import FILE", "Import conversation from file"),
            ],
            "Monitoring": [
                ("dashboard", "Show live monitoring dashboard"),
                # ("monitor [...]", "Control server health monitoring"), # If implemented
                # ("registry [...]", "Manage server registry connections"), # If implemented
            ]
        }

        self.safe_print("\n[bold]Available Commands:[/]")
        for category, commands in categories.items():
            table = Table(title=category, box=box.MINIMAL, show_header=False, padding=(0, 1, 0, 0))
            table.add_column("Command", style="bold cyan", no_wrap=True)
            table.add_column("Description")
            for cmd, desc in commands:
                table.add_row(f"/{cmd}", desc)
            self.safe_print(table)
        self.safe_print("\nUse '/command --help' or see documentation for detailed usage.")
        
    async def cmd_resources(self, args: str):
        """List available resources."""
        safe_console = get_safe_console()
        if not self.server_manager.resources:
            safe_console.print(f"{EMOJI_MAP['warning']} [yellow]No resources available from connected servers[/]")
            return

        server_filter = args.strip() if args else None

        resource_table = Table(title=f"{EMOJI_MAP['resource']} Available Resources", box=box.ROUNDED, show_lines=True)
        resource_table.add_column("Name", style="cyan")
        resource_table.add_column("Server", style="blue")
        resource_table.add_column("Description")
        resource_table.add_column("Template/URI", style="dim")

        filtered_resources = []
        for name, resource in sorted(self.server_manager.resources.items()):
            if server_filter is None or resource.server_name == server_filter:
                filtered_resources.append(resource)
                resource_table.add_row(
                    name, resource.server_name, resource.description or "[No description]", resource.template
                )

        if not filtered_resources:
            safe_console.print(f"[yellow]No resources found" + (f" for server '{server_filter}'" if server_filter else "") + ".[/yellow]")
        else:
            safe_console.print(resource_table)

    async def cmd_prompts(self, args: str):
        """List available prompts."""
        safe_console = get_safe_console()
        if not self.server_manager.prompts:
            safe_console.print(f"{EMOJI_MAP['warning']} [yellow]No prompts available from connected servers[/]")
            return

        server_filter = args.strip() if args else None

        prompt_table = Table(title=f"{EMOJI_MAP['prompt']} Available Prompts", box=box.ROUNDED, show_lines=True)
        prompt_table.add_column("Name", style="yellow")
        prompt_table.add_column("Server", style="blue")
        prompt_table.add_column("Description")

        filtered_prompts = []
        for name, prompt in sorted(self.server_manager.prompts.items()):
            if server_filter is None or prompt.server_name == server_filter:
                filtered_prompts.append(prompt)
                prompt_table.add_row(
                    name, prompt.server_name, prompt.description or "[No description]"
                )

        if not filtered_prompts:
            safe_console.print(f"[yellow]No prompts found" + (f" for server '{server_filter}'" if server_filter else "") + ".[/yellow]")
        else:
            safe_console.print(prompt_table)

            # Offer to show template content
            if not server_filter:
                 try:
                     prompt_name_prompt = await asyncio.to_thread(
                          Prompt.ask, "Enter prompt name to view content (or press Enter to skip)",
                          console=safe_console, default=""
                     )
                     prompt_name_to_show = prompt_name_prompt.strip()
                     if prompt_name_to_show in self.server_manager.prompts:
                         template_content = self.get_prompt_template(prompt_name_to_show)
                         if template_content:
                             safe_console.print(Panel(
                                 Text(template_content),
                                 title=f"{EMOJI_MAP['prompt']} Prompt Content: {prompt_name_to_show}",
                                 border_style="yellow"
                             ))
                         else:
                              safe_console.print(f"[yellow]Prompt '{prompt_name_to_show}' has no template content.[/yellow]")
                 except Exception as e:
                     log.warning(f"Error during prompt content prompt: {e}")

    async def cmd_tools(self, args: str):
        """List available tools, optionally filtering by server."""
        safe_console = get_safe_console()
        if not self.server_manager.tools:
            safe_console.print(f"{EMOJI_MAP['warning']} [yellow]No tools available from connected servers[/]")
            return

        server_filter = args.strip() if args else None

        tool_table = Table(title=f"{EMOJI_MAP['tool']} Available Tools", box=box.ROUNDED, show_lines=True)
        tool_table.add_column("Name", style="magenta", no_wrap=True)
        tool_table.add_column("Server", style="blue")
        tool_table.add_column("Description")
        # Optional: Add input schema summary if desired
        # tool_table.add_column("Inputs", style="dim")

        filtered_tools = []
        # Sort by name for consistent listing
        for name, tool in sorted(self.server_manager.tools.items()):
            if server_filter is None or tool.server_name == server_filter:
                filtered_tools.append(tool)
                # Example schema summary: list keys
                # input_keys = ", ".join(tool.input_schema.get('properties', {}).keys())
                # input_summary = f"Keys: {input_keys}" if input_keys else "[No defined inputs]"
                tool_table.add_row(
                    name,
                    tool.server_name,
                    tool.description or "[No description]",
                    # input_summary # Add if column exists
                )

        if not filtered_tools:
            safe_console.print(f"[yellow]No tools found" + (f" for server '{server_filter}'" if server_filter else "") + ".[/yellow]")
        else:
            safe_console.print(tool_table)

    async def cmd_discover(self, args: str):
        """Discover MCP servers on the network and filesystem."""
        safe_console = get_safe_console()

        # Optional: Parse args if you want flags (e.g., --no-prompt, --source=mdns)
        # For now, assume no arguments, just trigger discovery and processing.
        if args:
            safe_console.print("[yellow]Warning: /discover command currently takes no arguments.[/yellow]")

        if not self.server_manager:
            safe_console.print("[red]Error: ServerManager not initialized. Cannot run discovery.[/]")
            return

        safe_console.print(f"{EMOJI_MAP['search']} Starting server discovery...")
        try:
            # Call the server manager's discovery orchestrator
            await self.server_manager.discover_servers() # Step 1: Populates cache

            # Process the results (prompts user if interactive)
            # Assuming interactive mode is True for CLI command execution
            await self.server_manager._process_discovery_results(interactive_mode=True) # Step 2: Uses cache, interacts

        except Exception as e:
            log.error("Error during /discover command execution", exc_info=True)
            safe_console.print(f"[red]Error during discovery: {e}[/]")

    # --- Direct Tool Execution ---
    async def cmd_tool(self, args: str):
        """Directly execute a tool with parameters."""
        safe_console = get_safe_console()
        # Use shlex to parse tool name and JSON args robustly
        try:
            parts = shlex.split(args)
            if len(parts) < 1:
                safe_console.print("[yellow]Usage: /tool TOOL_NAME ['{JSON_PARAMS}'] [/]")
                safe_console.print("Example: /tool filesystem:readFile '{\"path\": \"/tmp/myfile.txt\"}'")
                return
            tool_name = parts[0]
            params_str = parts[1] if len(parts) > 1 else "{}" # Default to empty JSON object
            params = json.loads(params_str)
            if not isinstance(params, dict):
                raise TypeError("Parameters must be a JSON object (dictionary).")
        except json.JSONDecodeError:
            safe_console.print(f"[red]Invalid JSON parameters: {params_str}[/]")
            return
        except (ValueError, TypeError) as e:
            safe_console.print(f"[red]Error parsing command/parameters: {e}[/]")
            return
        except Exception as e:
            safe_console.print(f"[red]Error parsing tool command: {e}[/]")
            return

        # Check if tool exists
        if tool_name not in self.server_manager.tools:
            safe_console.print(f"[red]Tool not found: {tool_name}[/]")
            return

        tool = self.server_manager.tools[tool_name]
        server_name = tool.server_name

        with Status(f"{EMOJI_MAP['tool']} Executing {tool_name} via {server_name}...", spinner="dots", console=safe_console) as status:
            try:
                start_time = time.time()
                # Use the internal execute_tool method
                result: CallToolResult = await self.execute_tool(server_name, tool_name, params)
                latency = time.time() - start_time

                status_emoji = EMOJI_MAP['success'] if not result.isError else EMOJI_MAP['error']
                status.update(f"{status_emoji} Tool execution finished in {latency:.2f}s")

                # Safely format result content for display
                result_content = result.content
                display_content_str = ""
                syntax_lang = "text"
                if result_content is not None:
                    try:
                        # Try pretty-printing as JSON
                        display_content_str = json.dumps(result_content, indent=2, ensure_ascii=False)
                        syntax_lang = "json"
                    except TypeError:
                        # Fallback to string representation
                        display_content_str = str(result_content)
                        syntax_lang = "text" # Treat as plain text

                # Determine panel style based on error status
                panel_border = "red" if result.isError else "magenta"
                panel_title = f"{EMOJI_MAP['tool']} Tool Result: {tool_name}"
                if result.isError: panel_title += " (Error)"

                safe_console.print(Panel.fit(
                    Syntax(display_content_str, syntax_lang, theme="monokai", line_numbers=True, word_wrap=True),
                    title=panel_title,
                    subtitle=f"Executed in {latency:.3f}s",
                    border_style=panel_border
                ))
            except RuntimeError as e: # Catch errors raised by execute_tool (circuit breaker, MCP errors etc.)
                status.update(f"{EMOJI_MAP['failure']} Tool execution failed")
                safe_console.print(f"[red]Error executing tool '{tool_name}': {e}[/]")
            except Exception as e: # Catch unexpected errors
                status.update(f"{EMOJI_MAP['failure']} Unexpected error")
                log.error(f"Unexpected error during /tool execution for {tool_name}: {e}", exc_info=True)
                safe_console.print(f"[red]Unexpected error: {e}[/]")


    # --- Model Selection ---
    async def cmd_model(self, args: str):
        """Change the current AI model."""
        safe_console = get_safe_console()
        new_model = args.strip()
        if not new_model:
            safe_console.print(f"Current default model: [cyan]{self.current_model}[/]")
            # List available models by provider
            models_by_provider = {}
            for model_name, provider_value in MODEL_PROVIDER_MAP.items():
                models_by_provider.setdefault(provider_value.capitalize(), []).append(model_name)
            safe_console.print("\n[bold]Available Models (based on cost data):[/]")
            for provider, models in sorted(models_by_provider.items()):
                 safe_console.print(f" [blue]{provider}:[/] {', '.join(sorted(models))}")
            safe_console.print("\nUsage: /model MODEL_NAME")
            return

        # Optional: Validate if the model name is known or seems valid
        if new_model not in COST_PER_MILLION_TOKENS:
             if not Confirm.ask(f"Model '{new_model}' not found in cost list. Use anyway?", default=False, console=safe_console):
                 safe_console.print("[yellow]Model change cancelled.[/]")
                 return

        self.current_model = new_model
        self.config.default_model = new_model # Also update the config default
        await self.config.save_async() # Persist the default model change
        safe_console.print(f"[green]{EMOJI_MAP['model']} Default model changed to: {new_model}[/]")

    # --- Cache Management ---
    async def cmd_cache(self, args: str):
        """Manage the tool result cache."""
        safe_console = get_safe_console()
        if not self.tool_cache:
            safe_console.print("[yellow]Tool result caching is disabled (ToolCache not initialized).[/]")
            return

        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"
        subargs = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            entries_data = self.get_cache_entries() # Use helper
            if not entries_data:
                 safe_console.print(f"{EMOJI_MAP['package']} Tool cache is empty.")
                 return

            cache_table = Table(title=f"{EMOJI_MAP['package']} Cached Tool Results ({len(entries_data)} entries)", box=box.ROUNDED)
            cache_table.add_column("Tool Name", style="magenta")
            cache_table.add_column("Params Hash", style="dim")
            cache_table.add_column("Created At")
            cache_table.add_column("Expires At")

            for entry_data in entries_data: # Assumes get_cache_entries sorts if needed
                key_parts = entry_data['key'].split(':', 1)
                params_hash = key_parts[1][:12] if len(key_parts) > 1 else "N/A"
                expires_str = entry_data['expires_at'].strftime("%Y-%m-%d %H:%M:%S") if entry_data['expires_at'] else "[Never]"
                cache_table.add_row(
                    entry_data['tool_name'],
                    params_hash + "...",
                    entry_data['created_at'].strftime("%Y-%m-%d %H:%M:%S"),
                    expires_str
                )
            safe_console.print(cache_table)

        elif subcmd == "clear":
            tool_name_to_clear = subargs if subargs and subargs != "--all" else None
            target_desc = f"'{tool_name_to_clear}'" if tool_name_to_clear else "ALL entries"

            if Confirm.ask(f"Clear cache for {target_desc}?", console=safe_console):
                 with Status(f"{EMOJI_MAP['processing']} Clearing cache for {target_desc}...", console=safe_console) as status:
                     removed_count = self.clear_cache(tool_name=tool_name_to_clear) # Use helper
                     status.update(f"{EMOJI_MAP['success']} Cleared {removed_count} entries.")
                 safe_console.print(f"[green]Cleared {removed_count} cache entries for {target_desc}.[/]")
            else:
                 safe_console.print("[yellow]Cache clear cancelled.[/]")

        elif subcmd == "clean":
            with Status(f"{EMOJI_MAP['processing']} Cleaning expired cache entries...", console=safe_console) as status:
                removed_count = self.clean_cache() # Use helper
                status.update(f"{EMOJI_MAP['success']} Finished cleaning.")
            safe_console.print(f"[green]Cleaned {removed_count} expired cache entries.[/]")

        elif subcmd == "dependencies" or subcmd == "deps":
            deps = self.get_cache_dependencies() # Use helper
            if not deps:
                safe_console.print(f"{EMOJI_MAP['warning']} No tool dependencies registered.")
                return

            safe_console.print("\n[bold]Tool Dependency Graph:[/]")
            dep_table = Table(title="Dependencies", box=box.ROUNDED)
            dep_table.add_column("Tool", style="magenta")
            dep_table.add_column("Depends On", style="cyan")
            for tool, dependencies in sorted(deps.items()):
                dep_table.add_row(tool, ", ".join(dependencies))
            safe_console.print(dep_table)

        else:
            safe_console.print("[yellow]Unknown cache command. Available: list, clear [tool_name | --all], clean, dependencies[/]")


    # --- Conversation Management ---
    async def cmd_clear(self, args: str):
        """Clear the conversation messages of the current node or reset to root."""
        safe_console = get_safe_console()
        current_node = self.conversation_graph.current_node

        if Confirm.ask(f"Clear all messages from current branch '{current_node.name}' ({current_node.id[:8]})?", default=True, console=safe_console):
            if not current_node.messages:
                self.safe_print("[yellow]Current branch already has no messages.[/]")
            else:
                current_node.messages = []
                current_node.modified_at = datetime.now()
                await self.conversation_graph.save(str(self.conversation_graph_file))
                self.safe_print(f"[green]Cleared messages for branch '{current_node.name}'.[/]")

            # Optionally offer to switch back to root
            if current_node.id != "root":
                if Confirm.ask(f"Switch back to root branch?", default=False, console=safe_console):
                     self.conversation_graph.set_current_node("root")
                     self.safe_print("[green]Switched to root branch.[/]")

        else:
            self.safe_print("[yellow]Clear cancelled.[/]")

    async def cmd_reload(self, args: str):
        """Reloads MCP server connections and capabilities."""
        safe_console = get_safe_console()
        with Status(f"{EMOJI_MAP['processing']} Reloading MCP servers...", console=safe_console) as status:
            try:
                await self.reload_servers() # Use internal helper
                status.update(f"{EMOJI_MAP['success']} Server reload complete.")
                await self.print_status() # Show updated status
            except Exception as e:
                log.error("Error during server reload", exc_info=True)
                status.update(f"{EMOJI_MAP['error']} Server reload failed: {e}")
                safe_console.print(f"[red]Error during reload: {e}[/]")

    async def cmd_fork(self, args: str):
        """Create a new conversation fork/branch from the current node."""
        safe_console = get_safe_console()
        fork_name = args.strip() if args else None
        try:
            new_node = self.conversation_graph.create_fork(name=fork_name)
            self.conversation_graph.set_current_node(new_node.id)
            await self.conversation_graph.save(str(self.conversation_graph_file)) # Save after fork
            self.safe_print(f"{EMOJI_MAP['success']} Created and switched to new branch:")
            self.safe_print(f"  ID: [cyan]{new_node.id}[/]" )
            self.safe_print(f"  Name: [yellow]{new_node.name}[/]")
            parent_info = f"{new_node.parent.name} ({new_node.parent.id[:8]})" if new_node.parent else "[None]"
            self.safe_print(f"  Branched from: [magenta]{parent_info}[/]")
        except Exception as e:
            log.error("Error creating fork", exc_info=True)
            safe_console.print(f"[red]Error creating fork: {e}[/]")

    async def cmd_branch(self, args: str):
        """Manage conversation branches (list, checkout, rename, delete)."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"
        subargs = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            safe_console.print("\n[bold]Conversation Branches:[/]")
            branch_tree = Tree(f"{EMOJI_MAP['trident_emblem']} [bold cyan]Conversations[/]", guide_style="dim")

            def build_tree(node: ConversationNode, tree_node):
                label_parts = [Text(f"{node.name}", style="yellow")]
                label_parts.append(Text(f" (ID: {node.id[:8]}", style="dim cyan"))
                if node.model:
                     label_parts.append(Text(f", Model: {node.model}", style="dim blue"))
                label_parts.append(Text(")", style="dim cyan"))

                if node.id == self.conversation_graph.current_node.id:
                     # Use Text.assemble for complex styling
                     label = Text.assemble("[bold green]▶ [/]", *label_parts)
                else:
                     label = Text.assemble(*label_parts)

                current_branch = tree_node.add(label)
                # Sort children by creation time for consistent display
                sorted_children = sorted(node.children, key=lambda x: x.created_at)
                for child in sorted_children:
                    build_tree(child, current_branch)

            # Start building from root
            build_tree(self.conversation_graph.root, branch_tree)
            safe_console.print(branch_tree)
            safe_console.print("\nUse '/branch checkout ID_PREFIX' to switch.")

        elif subcmd == "checkout":
            if not subargs:
                safe_console.print("[yellow]Usage: /branch checkout NODE_ID_PREFIX[/]")
                return

            node_id_prefix = subargs
            matched_node = None
            matches = []
            for node_id, node in self.conversation_graph.nodes.items():
                if node_id.startswith(node_id_prefix):
                     matches.append(node)

            if len(matches) == 1:
                matched_node = matches[0]
            elif len(matches) > 1:
                safe_console.print(f"[red]Ambiguous node ID prefix '{node_id_prefix}'. Matches:[/]")
                for n in matches: safe_console.print(f"  - {n.name} ({n.id})")
                return

            if matched_node:
                if self.conversation_graph.set_current_node(matched_node.id):
                    self.safe_print(f"{EMOJI_MAP['success']} Switched to branch:")
                    self.safe_print(f"  ID: [cyan]{matched_node.id}[/]")
                    self.safe_print(f"  Name: [yellow]{matched_node.name}[/]")
                else: # Should not happen
                    safe_console.print(f"[red]Internal error switching to node {node_id_prefix}[/]")
            else:
                safe_console.print(f"[red]Node ID prefix '{node_id_prefix}' not found.[/]")

        elif subcmd == "rename":
            rename_parts = subargs.split(maxsplit=1)
            if len(rename_parts) != 2:
                safe_console.print("[yellow]Usage: /branch rename NODE_ID \"New Name\"[/]")
                return
            node_id, new_name_quoted = rename_parts[0], rename_parts[1]
            # Use shlex to handle quoted new name
            try: 
                new_name = shlex.split(new_name_quoted)[0]
            except Exception: 
                safe_console.print("[red]Invalid new name format. Use quotes if it contains spaces.[/]")
                return

            node = self.conversation_graph.get_node(node_id)
            if not node:
                safe_console.print(f"[red]Node ID '{node_id}' not found.[/]")
                return

            if node.id == "root":
                safe_console.print("[red]Cannot rename the root node.[/]")
                return

            old_name = node.name
            node.name = new_name
            node.modified_at = datetime.now()
            await self.conversation_graph.save(str(self.conversation_graph_file))
            safe_console.print(f"{EMOJI_MAP['success']} Renamed branch '{node_id[:8]}' from '{old_name}' to '{new_name}'.[/]")

        elif subcmd == "delete":
            node_id = subargs
            if not node_id:
                safe_console.print("[yellow]Usage: /branch delete NODE_ID[/]")
                return

            node_to_delete = self.conversation_graph.get_node(node_id)
            if not node_to_delete:
                safe_console.print(f"[red]Node ID '{node_id}' not found.[/]")
                return

            if node_to_delete.id == "root":
                safe_console.print("[red]Cannot delete the root node.[/]")
                return

            if node_to_delete.id == self.conversation_graph.current_node.id:
                safe_console.print("[red]Cannot delete the current branch. Checkout another branch first.[/]")
                return

            # Find parent and remove child reference
            parent = node_to_delete.parent
            if parent and node_to_delete in parent.children:
                if Confirm.ask(f"Delete branch '{node_to_delete.name}' ({node_id[:8]}) and all its descendants?", console=safe_console):
                    # Recursive delete helper
                    nodes_to_remove = set()
                    def find_descendants(n):
                        nodes_to_remove.add(n.id)
                        for child in n.children: find_descendants(child)
                    find_descendants(node_to_delete)

                    # Remove from graph nodes dict
                    for removed_id in nodes_to_remove:
                        if removed_id in self.conversation_graph.nodes:
                            del self.conversation_graph.nodes[removed_id]
                    # Remove from parent's children list
                    parent.children.remove(node_to_delete)
                    parent.modified_at = datetime.now()

                    await self.conversation_graph.save(str(self.conversation_graph_file))
                    safe_console.print(f"[green]Deleted branch '{node_to_delete.name}' and {len(nodes_to_remove)-1} descendant(s).[/]")
                else:
                    safe_console.print("[yellow]Deletion cancelled.[/]")
            else:
                safe_console.print(f"[red]Could not find parent or reference for node {node_id}. Deletion failed.[/]")

        else:
            safe_console.print("[yellow]Unknown branch command. Available: list, checkout, rename, delete[/]")



    async def load_claude_desktop_config(self):
        """
        Look for and load the Claude desktop config file (claude_desktop_config.json),
        transforming wsl.exe commands for direct execution within the Linux environment,
        and adapting Windows paths in arguments for other commands.
        """
        config_path = Path("claude_desktop_config.json")
        if not config_path.exists():
            log.debug("claude_desktop_config.json not found, skipping.")
            return # No file, nothing to do

        try:
            # Use safe_print for user-facing status messages
            self.safe_print(f"{EMOJI_MAP['config'] }  Found Claude desktop config file, processing...")

            # Read the file content asynchronously
            async with aiofiles.open(config_path, 'r') as f:
                content = await f.read()

            # Attempt to parse JSON
            try:
                desktop_config = json.loads(content)
                log.debug(f"Claude desktop config keys: {list(desktop_config.keys())}")
            except json.JSONDecodeError as json_error:
                # Print user-facing error and log details
                self.safe_print(f"[red]Invalid JSON in Claude desktop config: {json_error}[/]")
                try:
                    # Try to log the specific line from the content for easier debugging
                    problem_line = content.splitlines()[max(0, json_error.lineno - 1)]
                    log.error(f"JSON error in {config_path} at line {json_error.lineno}, col {json_error.colno}: '{problem_line}'", exc_info=True)
                except Exception:
                    log.error(f"Failed to parse JSON from {config_path}", exc_info=True) # Log generic parse error with traceback
                return # Stop processing if JSON is invalid

            # --- Find the mcpServers key ---
            mcp_servers_key = 'mcpServers'
            if mcp_servers_key not in desktop_config:
                found_alt = False
                # Check alternative keys just in case
                for alt_key in ['mcp_servers', 'servers', 'MCP_SERVERS']:
                    if alt_key in desktop_config:
                        log.info(f"Using alternative key '{alt_key}' for MCP servers")
                        mcp_servers_key = alt_key
                        found_alt = True
                        break
                if not found_alt:
                    self.safe_print(f"{EMOJI_MAP['warning']} No MCP servers key ('mcpServers' or alternatives) found in {config_path}")
                    return # Stop if no server list found

            mcp_servers = desktop_config.get(mcp_servers_key) # Use .get for safety
            if not mcp_servers or not isinstance(mcp_servers, dict):
                self.safe_print(f"{EMOJI_MAP['warning']} No valid MCP server entries found under key '{mcp_servers_key}' in {config_path}")
                return # Stop if server list is empty or not a dictionary

            # --- Process Servers ---
            imported_servers = []
            skipped_servers = []

            for server_name, server_data in mcp_servers.items():
                # Inner try block for processing each server individually
                try:
                    # Check if server already exists in the current configuration
                    if server_name in self.config.servers:
                        log.info(f"Server '{server_name}' already exists in local config, skipping import.")
                        skipped_servers.append((server_name, "already exists"))
                        continue

                    log.debug(f"Processing server '{server_name}' from desktop config: {server_data}")

                    # Ensure the 'command' field is present
                    if 'command' not in server_data:
                        log.warning(f"Skipping server '{server_name}': Missing 'command' field.")
                        skipped_servers.append((server_name, "missing command field"))
                        continue

                    original_command = server_data['command']
                    original_args = server_data.get('args', [])

                    # Variables to store the final executable and arguments for ServerConfig
                    final_executable = None
                    final_args = []
                    is_shell_command = False # Flag to indicate if we need `create_subprocess_shell` later

                    # --- Detect and transform WSL commands ---
                    if isinstance(original_command, str) and original_command.lower().endswith("wsl.exe"):
                        log.info(f"Detected WSL command for '{server_name}'. Extracting Linux command.")
                        # Search for a known shell ('bash', 'sh', 'zsh') followed by '-c'
                        shell_path = None
                        shell_arg_index = -1
                        possible_shells = ["bash", "sh", "zsh"]

                        for i, arg in enumerate(original_args):
                            if isinstance(arg, str):
                                # Check if argument is one of the known shells (can be full path or just name)
                                arg_base = os.path.basename(arg.lower())
                                if arg_base in possible_shells:
                                    shell_path = arg # Keep the original path/name provided
                                    shell_arg_index = i
                                    break

                        # Check if shell was found, followed by '-c', and the command string exists
                        if shell_path is not None and shell_arg_index + 2 < len(original_args) and original_args[shell_arg_index + 1] == '-c':
                            # The actual command string to execute inside the Linux shell
                            linux_command_str = original_args[shell_arg_index + 2]
                            log.debug(f"Extracted Linux command string for shell '{shell_path}': {linux_command_str}")

                            # Find the absolute path of the shell if possible, default to /bin/<shell_name>
                            try:
                                import shutil
                                found_path = shutil.which(shell_path)
                                final_executable = found_path if found_path else f"/bin/{os.path.basename(shell_path)}"
                            except Exception:
                                final_executable = f"/bin/{os.path.basename(shell_path)}" # Fallback

                            final_args = ["-c", linux_command_str]
                            is_shell_command = True # Mark that this needs shell execution later

                            log.info(f"Remapped '{server_name}' to run directly via shell: {final_executable} -c '...'")

                        else:
                            # If parsing fails (e.g., no 'bash -c' found)
                            log.warning(f"Could not parse expected 'shell -c command' structure in WSL args for '{server_name}': {original_args}. Skipping.")
                            skipped_servers.append((server_name, "WSL command parse failed"))
                            continue # Skip this server
                    # --- End WSL command transformation ---
                    else:
                        # --- Handle Direct Command + Adapt Paths in Args ---
                        # Assume it's a direct Linux command (like 'npx')
                        final_executable = original_command
                        # *** APPLY PATH ADAPTATION TO ARGUMENTS HERE ***
                        # adapt_path_for_platform expects (command, args) but only modifies args usually.
                        # We only need to adapt the args, the command itself ('npx') is fine.
                        _, adapted_args = adapt_path_for_platform(original_command, original_args)
                        final_args = adapted_args
                        # *** END PATH ADAPTATION ***
                        is_shell_command = False # Will use create_subprocess_exec later
                        log.info(f"Using command directly for '{server_name}' with adapted args: {final_executable} {' '.join(map(str, final_args))}")
                    # --- End Direct Command Handling ---


                    # Create the ServerConfig if we successfully determined the command
                    if final_executable is not None:
                        server_config = ServerConfig(
                            name=server_name,
                            type=ServerType.STDIO, # Claude desktop config implies STDIO
                            path=final_executable, # The direct executable or the shell
                            args=final_args, # Args for the executable, or ['-c', cmd_string] for shell
                            enabled=True, # Default to enabled
                            auto_start=True, # Default to auto-start
                            description=f"Imported from Claude desktop config ({'Direct Shell' if is_shell_command else 'Direct Exec'})",
                            trusted=True, # Assume trusted if coming from local desktop config
                            # Add other fields like categories if available in server_data and needed
                        )
                        # Add the prepared config to the main configuration object
                        self.config.servers[server_name] = server_config
                        imported_servers.append(server_name)
                        log.info(f"Prepared server '{server_name}' for import with direct execution.")

                # Catch errors processing a single server definition within the loop
                except Exception as server_proc_error:
                    log.error(f"Error processing server definition '{server_name}' from desktop config", exc_info=True)
                    skipped_servers.append((server_name, f"processing error: {server_proc_error}"))
                    continue # Skip this server and continue with the next

            # --- Save Config and Report Results ---
            if imported_servers:
                try:
                    # Save the updated configuration asynchronously
                    await self.config.save_async()
                    self.safe_print(f"{EMOJI_MAP['success']} Imported {len(imported_servers)} servers from Claude desktop config.")

                    # Report imported servers using a Rich Table
                    server_table = Table(title="Imported Servers (Direct Execution)")
                    server_table.add_column("Name")
                    server_table.add_column("Executable/Shell")
                    server_table.add_column("Arguments")
                    for name in imported_servers:
                        server = self.config.servers[name]
                        # Format arguments for display
                        args_display = ""
                        if len(server.args) == 2 and server.args[0] == '-c':
                             # Special display for shell commands
                             args_display = f"-c \"{server.args[1][:60]}{'...' if len(server.args[1]) > 60 else ''}\""
                        else:
                             args_display = " ".join(map(str, server.args))

                        server_table.add_row(name, server.path, args_display)
                    self.safe_print(server_table)

                except Exception as save_error:
                    # Handle errors during saving
                    log.error("Error saving config after importing servers", exc_info=True)
                    self.safe_print(f"[red]Error saving imported server config: {save_error}[/]")
            else:
                self.safe_print(f"{EMOJI_MAP['warning']}  No new servers were imported from Claude desktop config (they might already exist or failed processing).")

            # Report skipped servers, if any
            if skipped_servers:
                skipped_table = Table(title="Skipped Servers During Import")
                skipped_table.add_column("Name")
                skipped_table.add_column("Reason")
                for name, reason in skipped_servers:
                    skipped_table.add_row(name, reason)
                self.safe_print(skipped_table)

        # --- Outer Exception Handling ---
        except FileNotFoundError:
            # This is normal if the file doesn't exist, already handled by the initial check.
            log.debug(f"{config_path} not found.")
        except Exception as outer_config_error:
            # Catch any other unexpected error during the whole process (file read, json parse, server loop)
            self.safe_print(f"[bold red]An unexpected error occurred while processing {config_path}: {outer_config_error}[/]")
            # Print traceback directly to stderr for diagnostics, bypassing logging/safe_print
            print(f"\n--- Traceback for Claude Desktop Config Error ({type(outer_config_error).__name__}) ---", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("--- End Traceback ---", file=sys.stderr)

    async def cmd_export(self, args: str):
        """Export the current conversation or a specific branch to a file."""
        safe_console = get_safe_console()
        # Use shlex to handle potentially quoted file paths
        try:
            parsed_args = shlex.split(args)
        except ValueError as e:
            safe_console.print(f"[red]Error parsing arguments: {e}[/]")
            safe_console.print("[yellow]Usage: /export [--id CONVERSATION_ID] [--output /path/to/file.json][/]")
            return

        conversation_id = self.conversation_graph.current_node.id # Default to current
        output_path = None

        i = 0
        while i < len(parsed_args):
            arg = parsed_args[i]
            if arg in ["--id", "-i"] and i + 1 < len(parsed_args):
                conversation_id = parsed_args[i+1]
                i += 1
            elif arg in ["--output", "-o"] and i + 1 < len(parsed_args):
                output_path = parsed_args[i+1]
                i += 1
            else:
                safe_console.print(f"[red]Unknown argument: {arg}[/]")
                safe_console.print("[yellow]Usage: /export [--id CONVERSATION_ID] [--output /path/to/file.json][/]")
                return
            i += 1

        # Default filename if not provided
        if not output_path:
            output_path = f"conversation_{conversation_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            safe_console.print(f"[dim]No output path specified, using default: {output_path}[/dim]")

        # Call the export method with progress bar
        with Status(f"{EMOJI_MAP['scroll']} Exporting conversation {conversation_id[:8]}...", spinner="dots", console=safe_console) as status:
            # get_conversation_export_data is async now
            export_data = await self.get_conversation_export_data(conversation_id)
            if export_data is None:
                status.stop()
                safe_console.print(f"[red]Conversation ID '{conversation_id}' not found for export.[/]")
                return

            status.update(f"{EMOJI_MAP['processing']} Writing to {output_path}...")
            try:
                async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                    json_string = json.dumps(export_data, indent=2, ensure_ascii=False)
                    await f.write(json_string)
                status.update(f"{EMOJI_MAP['success']} Conversation exported successfully.")
                safe_console.print(f"[green]Conversation exported to: [bold]{output_path}[/][/]")
            except Exception as e:
                status.stop()
                log.error(f"Failed to write export file {output_path}", exc_info=True)
                safe_console.print(f"[red]Failed to export conversation: {e}[/]")

    async def cmd_import(self, args: str):
        """Import a conversation from a file."""
        safe_console = get_safe_console()
        # Use shlex to handle potentially quoted file paths
        try:
            parsed_args = shlex.split(args)
            if len(parsed_args) != 1: raise ValueError("Requires exactly one argument: filepath")
            file_path = parsed_args[0]
        except ValueError as e:
            safe_console.print(f"[red]Error parsing arguments: {e}[/]")
            safe_console.print("[yellow]Usage: /import \"/path/to/your conversation.json\"[/]")
            return

        if not Path(file_path).exists():
             safe_console.print(f"[red]Import file not found: {file_path}[/]")
             return

        with Status(f"{EMOJI_MAP['scroll']} Importing conversation from {file_path}...", spinner="dots", console=safe_console) as status:
            success = await self.import_conversation(file_path) # Use helper method
            if success:
                status.update(f"{EMOJI_MAP['success']} Conversation imported successfully")
                # Message printed by import_conversation
            else:
                status.update(f"{EMOJI_MAP['failure']} Import failed")
                # Error printed by import_conversation

    async def cmd_history(self, args: str):
        """View conversation history."""
        safe_console = get_safe_console()
        if not self.history.entries:
            safe_console.print(f"{EMOJI_MAP['warning']} [yellow]No conversation history found.[/]")
            return

        # Parse args for count limit or search query
        limit = 10 # Default number of entries to show
        search_query = None
        if args:
            if args.isdigit():
                limit = int(args)
            else:
                search_query = args # Treat non-digit args as search query

        entries_to_display: List[ChatHistory] = []
        title = ""

        if search_query:
            limit = 50 # Increase limit for search results
            entries_to_display = self.history.search(search_query, limit=limit)
            title = f"{EMOJI_MAP['search']} History Search Results for '[yellow]{search_query}[/]' (max {limit})"
            if not entries_to_display:
                safe_console.print(f"[yellow]No history entries found matching '{search_query}'.[/]")
                return
        else:
            entries_to_display = list(self.history.entries)[-limit:] # Get last N entries
            entries_to_display.reverse() # Show newest first
            total_entries = len(self.history.entries)
            num_shown = min(limit, total_entries)
            title = f"{EMOJI_MAP['history']} Recent Conversation History (last {num_shown} of {total_entries})"

        safe_console.print(f"\n[bold]{title}[/]")

        for i, entry in enumerate(entries_to_display, 1):
            entry_panel_title = Text.assemble(
                (f"{i}. ", "dim"),
                (entry.timestamp, "cyan"),
                (" | Model: ", "dim"), (entry.model, "magenta"),
                (" | Tools: ", "dim"), (f"{len(entry.tools_used)}", "blue") if entry.tools_used else ("None", "dim"),
                (" | Latency: ", "dim"), (f"{entry.latency_ms:.0f}ms", "yellow")
            )
            content_group = Group(
                Text.assemble(("[Q] ", "bold blue"), Text(entry.query, overflow="ellipsis")),
                Text.assemble(("[A] ", "bold green"), Text(entry.response, overflow="ellipsis"))
            )
            safe_console.print(Panel(content_group, title=entry_panel_title, border_style="dim", padding=(0,1)))

    async def cmd_optimize(self, args: str):
        """Optimize conversation context via summarization."""
        safe_console = get_safe_console()
        # (Keep argument parsing logic as is)
        try:
            parsed_args = shlex.split(args)
        except ValueError as e:
             safe_console.print(f"[red]Error parsing arguments: {e}[/]")
             safe_console.print("[yellow]Usage: /optimize [--model MODEL_NAME] [--tokens TARGET_TOKENS][/]")
             return

        custom_model = None
        target_length = self.config.max_summarized_tokens
        i = 0
        while i < len(parsed_args):
            arg = parsed_args[i]
            if arg in ["--model", "-m"] and i + 1 < len(parsed_args):
                custom_model = parsed_args[i+1]
                i += 1
            elif arg in ["--tokens", "-t"] and i + 1 < len(parsed_args):
                try:
                    target_length = int(parsed_args[i+1])
                    i += 1
                except ValueError:
                    safe_console.print(f"[red]Invalid token count: {parsed_args[i+1]}[/]")
                    return
            else:
                safe_console.print(f"[red]Unknown argument: {arg}[/]")
                return
            i += 1

        initial_tokens = await self.count_tokens()
        log.info(f"Optimization requested. Initial tokens: {initial_tokens}. Target: ~{target_length}")

        with Status(f"{EMOJI_MAP['processing']} Optimizing conversation (currently {initial_tokens:,} tokens)...", console=safe_console) as status:
            try:
                # --- CALL THE REFACTORED METHOD ---
                summary = await self.summarize_conversation(
                    target_tokens=target_length,
                    model=custom_model
                )
                # --- END CALL ---

                if summary is None:
                    status.stop()
                    safe_console.print(f"[red]{EMOJI_MAP['error']} Summarization failed (LLM call returned no content or errored).[/]")
                    return

                # Apply the summary
                summary_system_message = f"The preceding conversation up to this point has been summarized:\n\n---\n{summary}\n---"
                self.conversation_graph.current_node.messages = [
                    InternalMessage(role="system", content=summary_system_message)
                ]
                self.conversation_graph.current_node.modified_at = datetime.now()
                await self.conversation_graph.save(str(self.conversation_graph_file))

                final_tokens = await self.count_tokens()
                status.update(f"{EMOJI_MAP['success']} Conversation optimized: {initial_tokens:,} -> {final_tokens:,} tokens")
                safe_console.print(f"[green]Optimization complete. Tokens reduced from {initial_tokens:,} to {final_tokens:,}.[/]")

            except Exception as e:
                status.stop()
                log.error("Error during conversation optimization command", exc_info=True)
                safe_console.print(f"[red]{EMOJI_MAP['error']} Optimization failed: {e}[/]")

    async def cmd_apply_prompt(self, args: str): # Renamed from cmd_prompt for clarity
        """Apply a prompt template to the current conversation."""
        safe_console = get_safe_console()
        prompt_name = args.strip()
        if not prompt_name:
            safe_console.print("[yellow]Usage: /prompt PROMPT_NAME[/]")
            await self.cmd_prompts("") # List available prompts if none given
            return

        success = await self.apply_prompt_to_conversation(prompt_name) # Use helper
        if success:
            safe_console.print(f"[green]{EMOJI_MAP['success']} Applied prompt template '{prompt_name}' as system message.[/]")
        else:
            safe_console.print(f"[red]{EMOJI_MAP['error']} Prompt '{prompt_name}' not found or could not be applied.[/]")

    async def cmd_dashboard(self, args: str):
        """Show the live monitoring dashboard."""
        safe_console = get_safe_console()
        if hasattr(self, '_active_live_display') and self._active_live_display:
            safe_console.print("[yellow]Cannot start dashboard while another live display (e.g., query) is active.[/]")
            return

        try:
            self._active_live_display = True # Use simple flag to block other Live instances
            # Start monitoring if not already running
            if not self.server_monitor.monitoring:
                 await self.server_monitor.start_monitoring()

            # Use Live context for the dashboard
            with Live(self.generate_dashboard_renderable(),
                      refresh_per_second=1.0/max(0.1, self.config.dashboard_refresh_rate), # Ensure positive rate
                      screen=True, transient=False, console=safe_console) as live:
                while True:
                    await asyncio.sleep(max(0.1, self.config.dashboard_refresh_rate))
                    live.update(self.generate_dashboard_renderable())
        except KeyboardInterrupt:
            self.safe_print("\n[yellow]Dashboard stopped.[/]")
        except Exception as e:
            log.error(f"Dashboard error: {e}", exc_info=True)
            self.safe_print(f"\n[red]{EMOJI_MAP['error']} Dashboard error: {e}[/]")
        finally:
            self._active_live_display = None # Clear the flag

    # --- Utility methods and decorators ---

    @staticmethod
    def safe_print(message, **kwargs): # No self parameter
            """Print using the appropriate console based on active stdio servers.

            Applies automatic spacing after known emojis defined in EMOJI_MAP.

            Args:
                message: The message to print (can be string or other Rich renderable)
                **kwargs: Additional arguments to pass to print
            """
            safe_console = get_safe_console()
            processed_message = message
            # Apply spacing logic ONLY if the message is a string
            if isinstance(message, str) and message: # Also check if message is not empty
                try:
                    # Extract actual emoji characters, escaping any potential regex special chars
                    emoji_chars = [re.escape(str(emoji)) for emoji in EMOJI_MAP.values()]
                    if emoji_chars: # Only compile if there are emojis
                        # Create a pattern that matches any of these emojis followed by a non-whitespace char
                        # (?:...) is a non-capturing group for the alternation
                        # Capturing group 1: the emoji | Capturing group 2: the non-whitespace character
                        emoji_space_pattern = re.compile(f"({'|'.join(emoji_chars)})" + r"(\S)")
                        # Apply the substitution
                        processed_message = emoji_space_pattern.sub(r"\1 \2", message)
                except Exception as e:
                    # Log error if regex fails, but proceed with original message
                    log.warning(f"Failed to apply emoji spacing regex: {e}")
                    processed_message = message # Fallback to original message
            # Print the processed (or original) message
            safe_console.print(processed_message, **kwargs)
    
    @staticmethod
    def with_tool_error_handling(func):
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            tool_name = kwargs.get("tool_name", args[1] if len(args) > 1 else "unknown")
            try: 
                return await func(self, *args, **kwargs)
            except McpError as e:
                log.error(f"MCP error exec {tool_name}: {e}")
                raise RuntimeError(f"MCP error: {e}") from e
            except httpx.RequestError as e:
                log.error(f"Net error exec {tool_name}: {e}")
                raise RuntimeError(f"Network error: {e}") from e
            except Exception as e:
                log.error(f"Unexpected error exec {tool_name}: {e}")
                raise RuntimeError(f"Unexpected error: {e}") from e
        return wrapper

    @staticmethod
    def retry_with_circuit_breaker(func):
        async def wrapper(self, server_name, *args, **kwargs):
            server_config = self.config.servers.get(server_name)
            if not server_config:
                raise RuntimeError(f"Server {server_name} not found")
                
            if server_config.metrics.error_rate > 0.5:
                log.warning(f"Circuit breaker triggered for server {server_name} (error rate: {server_config.metrics.error_rate:.2f})")
                raise RuntimeError(f"Server {server_name} in circuit breaker state")
                
            last_error = None
            for attempt in range(server_config.retry_policy["max_attempts"]):
                try:
                    # For each attempt, slightly increase the timeout
                    request_timeout = server_config.timeout + (attempt * server_config.retry_policy["timeout_increment"])
                    return await func(self, server_name, *args, **kwargs, request_timeout=request_timeout)
                except (RuntimeError, httpx.RequestError) as e:
                    last_error = e
                    server_config.metrics.request_count += 1
                    
                    if attempt < server_config.retry_policy["max_attempts"] - 1:
                        delay = server_config.retry_policy["backoff_factor"] * (2 ** attempt) + random.random()
                        log.warning(f"Retrying tool execution for server {server_name} (attempt {attempt+1}/{server_config.retry_policy['max_attempts']})")
                        log.warning(f"Retry will happen after {delay:.2f}s delay. Error: {str(e)}")
                        await asyncio.sleep(delay)
                    else:
                        server_config.metrics.error_count += 1
                        server_config.metrics.update_status()
                        raise RuntimeError(f"All {server_config.retry_policy['max_attempts']} attempts failed for server {server_name}: {str(last_error)}") from last_error
            
            return None  # Should never reach here
        return wrapper

    @retry_with_circuit_breaker
    @with_tool_error_handling
    async def execute_tool(self, server_name, tool_name, tool_args, request_timeout=None):
        """Execute a tool with retry and circuit breaker logic.
        Timeouts are handled by the session's default read timeout.

        Args:
            server_name: Name of the server to execute the tool on
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool

        Returns:
            The tool execution result
        """
        session = self.server_manager.active_sessions.get(server_name)
        if not session:
            raise RuntimeError(f"Server {server_name} not connected")

        tool = self.server_manager.tools.get(tool_name)
        if not tool:
            raise RuntimeError(f"Tool {tool_name} not found")

        try:
            with safe_stdout():
                async with self.tool_execution_context(tool_name, tool_args, server_name):
                    # *** REMOVE response_timeout argument from the call ***
                    result = await session.call_tool(
                        tool.original_tool.name, # Use the name from the original Tool object
                        tool_args
                    )

                    # Dependency check (unchanged)
                    if self.tool_cache:
                        dependencies = self.tool_cache.dependency_graph.get(tool_name, set())
                        if dependencies:
                            log.debug(f"Tool {tool_name} has dependencies: {dependencies}")

                    return result
        finally:
            pass # Context managers handle exit
        
    @asynccontextmanager
    async def tool_execution_context(self, tool_name, tool_args, server_name):
        """Context manager for tool execution metrics and tracing.
        
        Args:
            tool_name: The name of the tool being executed
            tool_args: The arguments passed to the tool
            server_name: The name of the server handling the tool
        """
        start_time = time.time()
        tool = None
        
        # Find the tool in our registry
        if server_name in self.server_manager.tools_by_server:
            for t in self.server_manager.tools_by_server[server_name]:
                if t.name == tool_name:
                    tool = t
                    break
        
        try:
            yield
        finally:
            # Update metrics if we found the tool
            if tool and isinstance(tool, MCPTool):
                execution_time = (time.time() - start_time) * 1000  # Convert to ms
                tool.update_execution_time(execution_time)
                        
    async def print_simple_status(self):
        """Print a simplified status without using Progress widgets"""
        # Count connected servers, available tools/resources
        connected_servers = len(self.server_manager.active_sessions)
        total_servers = len(self.config.servers)
        total_tools = len(self.server_manager.tools)
        total_resources = len(self.server_manager.resources)
        total_prompts = len(self.server_manager.prompts)
        # Print basic info table
        status_table = Table(title="MCP Client Status")
        status_table.add_column("Item")
        status_table.add_column("Status", justify="right")
        status_table.add_row(f"{EMOJI_MAP['model']} Model", self.current_model)
        status_table.add_row(f"{EMOJI_MAP['server']} Servers", f"{connected_servers}/{total_servers} connected")
        status_table.add_row(f"{EMOJI_MAP['tool']} Tools", str(total_tools))
        status_table.add_row(f"{EMOJI_MAP['resource']} Resources", str(total_resources))
        status_table.add_row(f"{EMOJI_MAP['prompt']} Prompts", str(total_prompts))
        self.safe_print(status_table)
        # Show connected server info
        if connected_servers > 0:
            self.safe_print("\n[bold]Connected Servers:[/]")
            for name, server in self.config.servers.items():
                if name in self.server_manager.active_sessions:
                    # Get number of tools for this server
                    server_tools = sum(1 for t in self.server_manager.tools.values() if t.server_name == name)
                    self.safe_print(f"[green]✓[/] {name} ({server.type.value}) - {server_tools} tools")
        self.safe_print("[green]Ready to process queries![/green]")
             # Inside class MCPClient:

    def _stringify_content(self, content: Any) -> str:
        """Converts complex content (dict, list) to string, otherwise returns string."""
        if isinstance(content, str):
            return content
        elif content is None:
             return ""
        elif isinstance(content, (dict, list)):
            try:
                # Pretty print JSON for readability if it's simple structures
                return json.dumps(content, indent=2)
            except TypeError:
                # Fallback for non-serializable objects
                return str(content)
        else:
            return str(content)

    @staticmethod
    def ensure_safe_console(func):
        """Decorator to ensure methods use safe console consistently
        
        This decorator:
        1. Gets a safe console once at the beginning of the method
        2. Stores it temporarily on the instance to prevent multiple calls
        3. Restores the previous value after method completes
        
        Args:
            func: The method to decorate
            
        Returns:
            Wrapped method that uses safe console consistently
        """
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            # Get safe console once at the beginning
            safe_console = get_safe_console()
            # Store temporarily on the instance to prevent multiple calls
            old_console = getattr(self, '_current_safe_console', None)
            self._current_safe_console = safe_console
            try:
                return await func(self, *args, **kwargs)
            finally:
                # Restore previous value if it existed
                if old_console is not None:
                    self._current_safe_console = old_console
                else:
                    delattr(self, '_current_safe_console')

    @ensure_safe_console
    async def print_status(self):
        """Print current status of servers, tools, and capabilities with progress bars"""
        # Use the stored safe console instance to prevent multiple calls
        safe_console = self._current_safe_console

        # Helper function using the pre-compiled pattern (still needed for progress bar)
        def apply_emoji_spacing(text: str) -> str:
             if isinstance(text, str) and text and hasattr(self, '_emoji_space_pattern'):
                 try: # Add try-except for robustness
                     return self._emoji_space_pattern.sub(r"\1 \2", text)
                 except Exception as e:
                     log.warning(f"Failed to apply emoji spacing regex in helper: {e}")
             return text # Return original text if not string, empty, pattern missing, or error

        # Count connected servers, available tools/resources
        connected_servers = len(self.server_manager.active_sessions)
        total_servers = len(self.config.servers)
        total_tools = len(self.server_manager.tools)
        total_resources = len(self.server_manager.resources)
        total_prompts = len(self.server_manager.prompts)

        # Print basic info table
        status_table = Table(title="MCP Client Status", box=box.ROUNDED) # Use a box style
        status_table.add_column("Item", style="dim") # Apply style to column
        status_table.add_column("Status", justify="right")

        # --- Use Text.assemble for the first column ---
        status_table.add_row(
            Text.assemble(str(EMOJI_MAP['model']), " Model"), # Note the space before "Model"
            self.current_model
        )
        status_table.add_row(
            Text.assemble(str(EMOJI_MAP['server']), " Servers"), # Note the space before "Servers"
            f"{connected_servers}/{total_servers} connected"
        )
        status_table.add_row(
            Text.assemble(str(EMOJI_MAP['tool']), " Tools"), # Note the space before "Tools"
            str(total_tools)
        )
        status_table.add_row(
            Text.assemble(str(EMOJI_MAP['resource']), " Resources"), # Note the space before "Resources"
            str(total_resources)
        )
        status_table.add_row(
            Text.assemble(str(EMOJI_MAP['prompt']), " Prompts"), # Note the space before "Prompts"
            str(total_prompts)
        )
        # --- End Text.assemble usage ---

        safe_console.print(status_table) # Use the regular safe_print

        if hasattr(self, 'cache_hit_count') and (self.cache_hit_count + self.cache_miss_count) > 0:
            cache_table = Table(title="Prompt Cache Statistics", box=box.ROUNDED)
            cache_table.add_column("Metric", style="dim")
            cache_table.add_column("Value", justify="right")
            
            hit_rate = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) * 100
            
            cache_table.add_row(
                Text.assemble(str(EMOJI_MAP['package']), " Cache Hits"),
                str(self.cache_hit_count)
            )
            cache_table.add_row(
                Text.assemble(str(EMOJI_MAP['warning']), " Cache Misses"),
                str(self.cache_miss_count)
            )
            cache_table.add_row(
                Text.assemble(str(EMOJI_MAP['success']), " Hit Rate"),
                f"{hit_rate:.1f}%"
            )
            cache_table.add_row(
                Text.assemble(str(EMOJI_MAP['speech_balloon']), " Tokens Saved"),
                f"{self.tokens_saved_by_cache:,}"
            )
            
            safe_console.print(cache_table)
            
        # Only show server progress if we have servers
        if total_servers > 0:
            server_tasks = []
            for name, server in self.config.servers.items():
                if name in self.server_manager.active_sessions:
                    # Apply spacing to progress description (keep using helper here)
                    task_description = apply_emoji_spacing(
                        f"{EMOJI_MAP['server']} {name} ({server.type.value})"
                    )
                    server_tasks.append(
                        (self._display_server_status,
                        task_description,
                        (name, server))
                    )

            if server_tasks:
                await self._run_with_progress(
                    server_tasks,
                    "Server Status",
                    transient=False,
                    use_health_scores=True
                )

        # Use safe_print for this final message too, in case emojis are added later
        self.safe_print("[green]Ready to process queries![/green]")

    def _calculate_and_log_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculates the estimated cost for a given number of input/output tokens and model.
        Logs the breakdown and returns the calculated cost.

        Args:
            model_name: The name of the model used.
            input_tokens: Number of input tokens for the API call.
            output_tokens: Number of output tokens for the API call.

        Returns:
            The estimated cost for this specific API call turn, or 0.0 if cost info is unavailable.
        """
        cost_info = COST_PER_MILLION_TOKENS.get(model_name)
        turn_cost = 0.0

        if cost_info:
            input_cost = (input_tokens * cost_info.get("input", 0)) / 1_000_000
            output_cost = (output_tokens * cost_info.get("output", 0)) / 1_000_000
            turn_cost = input_cost + output_cost
            log.info(f"Cost Calc ({model_name}): Input={input_tokens} (${input_cost:.6f}), "
                     f"Output={output_tokens} (${output_cost:.6f}), Turn Total=${turn_cost:.6f}")
        else:
            log.warning(f"Cost info not found for model '{model_name}'. Cannot calculate turn cost.")

        return turn_cost

    async def count_tokens(self, messages: Optional[InternalMessageList] = None) -> int:
        """
        Estimates the number of tokens in the provided messages or current conversation context
        using tiktoken (cl100k_base encoding). This is an estimation, actual provider counts may vary.
        """
        if messages is None:
            if not hasattr(self, 'conversation_graph') or not self.conversation_graph:
                log.warning("Conversation graph not available for token counting.")
                return 0
            messages = self.conversation_graph.current_node.messages

        if not messages:
            return 0

        try:
            # Use cl100k_base encoding which is common for many models (like GPT and Claude)
            encoding = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            log.warning(f"Failed to get tiktoken encoding 'cl100k_base': {e}. Cannot estimate tokens.")
            return 0 # Or potentially raise an error or return -1

        token_count = 0
        for message in messages:
            role = message.get("role") # noqa: F841
            content = message.get("content")
            msg_token_count = 0

            # Add tokens for role and message structure overhead
            # This varies slightly by model/provider, 4 is a common estimate
            msg_token_count += 4

            if isinstance(content, str):
                msg_token_count += len(encoding.encode(content))
            elif isinstance(content, list):
                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        msg_token_count += len(encoding.encode(block.get("text", "")))
                    elif block_type == "tool_use":
                        # Estimate tokens for tool use representation (name, id, input keys/values)
                        # This is a rough approximation
                        name_tokens = len(encoding.encode(block.get("name", "")))
                        input_str = json.dumps(block.get("input", {})) # Stringify input
                        input_tokens_est = len(encoding.encode(input_str))
                        # Add overhead for structure, id, name, input markers
                        msg_token_count += name_tokens + input_tokens_est + 10 # Rough overhead
                    elif block_type == "tool_result":
                        # Estimate tokens for tool result representation
                        result_content = block.get("content")
                        content_str = self._stringify_content(result_content) # Helper to handle complex content
                        result_tokens_est = len(encoding.encode(content_str))
                         # Add overhead for structure, id, content markers
                        msg_token_count += result_tokens_est + 10 # Rough overhead
                    else:
                        # Fallback for unknown block types
                        try:
                            block_str = json.dumps(block)
                            msg_token_count += len(encoding.encode(block_str)) + 5
                        except Exception:
                             msg_token_count += len(encoding.encode(str(block))) + 5


            # Add message tokens to total
            token_count += msg_token_count

        return token_count

    def _estimate_string_tokens(self, text: str) -> int:
        """Estimate token count for a given string using tiktoken (cl100k_base)."""
        if not text:
            return 0
        try:
            # Use the same encoding as count_tokens for consistency
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception as e:
            log.warning(f"Could not estimate string tokens: {e}")
            # Fallback: approximate based on characters
            return len(text) // 4 # Very rough fallback



    async def get_conversation_export_data(self, conversation_id: str) -> Optional[Dict]:
        """Gets the data for exporting a specific conversation branch."""
        node = self.conversation_graph.get_node(conversation_id)
        if not node:
            log.warning(f"Export failed: Conversation ID '{conversation_id}' not found.")
            return None

        all_nodes_in_path = self.conversation_graph.get_path_to_root(node)
        messages_export: InternalMessageList = []
        for ancestor_node in all_nodes_in_path:
            messages_export.extend(ancestor_node.messages)

        export_data = {
            "id": node.id,
            "name": node.name,
            "messages": messages_export, # Should be list of dicts
            "model": node.model or self.config.default_model, # Include model
            "exported_at": datetime.now().isoformat(),
            "path_ids": [n.id for n in all_nodes_in_path] # Include path for context
        }
        return export_data

    async def import_conversation_from_data(self, data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
        """Imports conversation data as a new branch under the current node."""
        try:
            # Basic validation
            if not isinstance(data.get("messages"), list):
                return False, "Invalid import data: 'messages' field missing or not a list.", None

            # Validate message structure (optional but recommended)
            validated_messages: InternalMessageList = []
            for i, msg_data in enumerate(data["messages"]):
                 # Basic check for role and content presence
                 if not isinstance(msg_data, dict) or "role" not in msg_data or "content" not in msg_data:
                      log.warning(f"Skipping invalid message structure at index {i} during import: {msg_data}")
                      continue
                 # Perform deeper validation if needed using InternalMessage structure or Pydantic
                 validated_messages.append(cast('InternalMessage', msg_data)) # Cast after basic check

            if not validated_messages and data["messages"]: # If all messages were invalid
                 return False, "Import failed: No valid messages found in the import data.", None

            new_node_id = str(uuid.uuid4())
            new_node = ConversationNode(
                id=new_node_id,
                name=f"Imported: {data.get('name', f'Branch-{new_node_id[:4]}')}",
                messages=validated_messages, # Use validated messages
                model=data.get('model', self.config.default_model),
                parent=self.conversation_graph.current_node # Attach to current node
            )

            self.conversation_graph.add_node(new_node)
            self.conversation_graph.current_node.add_child(new_node)
            await self.conversation_graph.save(str(self.conversation_graph_file))
            log.info(f"Imported conversation as new node '{new_node.id}' under '{self.conversation_graph.current_node.id}'.")
            return True, f"Import successful. New node ID: {new_node.id}", new_node.id

        except Exception as e:
            log.error(f"Error importing conversation data: {e}", exc_info=True)
            return False, f"Internal error during import: {e}", None

    def get_cache_entries(self) -> List[Dict]:
        """Gets details of all tool cache entries (memory and disk)."""
        if not self.tool_cache:
            return []
        entries = []
        all_keys = set(self.tool_cache.memory_cache.keys())
        if self.tool_cache.disk_cache:
             try:
                 # Iterate keys safely
                 disk_keys = set(self.tool_cache.disk_cache.iterkeys())
                 all_keys.update(disk_keys)
             except Exception as e:
                 log.warning(f"Could not iterate disk cache keys: {e}")

        for key in all_keys:
            entry_obj: Optional[CacheEntry] = self.tool_cache.memory_cache.get(key)
            if not entry_obj and self.tool_cache.disk_cache:
                try: entry_obj = self.tool_cache.disk_cache.get(key)
                except Exception: entry_obj = None # Skip potentially corrupted

            if entry_obj and isinstance(entry_obj, CacheEntry): # Ensure it's the correct type
                 entry_data = {
                     "key": key,
                     "tool_name": entry_obj.tool_name,
                     "created_at": entry_obj.created_at,
                     "expires_at": entry_obj.expires_at,
                 }
                 entries.append(entry_data)
            elif entry_obj:
                 log.warning(f"Found unexpected object type in cache for key '{key}': {type(entry_obj)}")

        # Sort entries, e.g., by creation date descending
        entries.sort(key=lambda x: x["created_at"], reverse=True)
        return entries

    def clear_cache(self, tool_name: Optional[str] = None) -> int:
        """Clears tool cache entries, optionally filtered by tool name. Returns count removed."""
        if not self.tool_cache: return 0

        # Count keys before
        keys_before_mem = set(self.tool_cache.memory_cache.keys())
        keys_before_disk = set()
        if self.tool_cache.disk_cache:
             with suppress(Exception): keys_before_disk = set(self.tool_cache.disk_cache.iterkeys())
        keys_before = keys_before_mem.union(keys_before_disk)

        # Perform invalidation (synchronous)
        self.tool_cache.invalidate(tool_name=tool_name)

        # Count keys after
        keys_after_mem = set(self.tool_cache.memory_cache.keys())
        keys_after_disk = set()
        if self.tool_cache.disk_cache:
             with suppress(Exception): keys_after_disk = set(self.tool_cache.disk_cache.iterkeys())
        keys_after = keys_after_mem.union(keys_after_disk)

        return len(keys_before) - len(keys_after)

    def clean_cache(self) -> int:
        """Cleans expired tool cache entries. Returns count removed."""
        if not self.tool_cache: return 0
        # clean() method handles both memory and disk (using expire())
        # We need to calculate the count manually for more accuracy
        mem_keys_before = set(self.tool_cache.memory_cache.keys())
        disk_keys_before = set()
        if self.tool_cache.disk_cache:
            with suppress(Exception): disk_keys_before = set(self.tool_cache.disk_cache.iterkeys())

        self.tool_cache.clean() # This performs the actual cleaning

        mem_keys_after = set(self.tool_cache.memory_cache.keys())
        disk_keys_after = set()
        if self.tool_cache.disk_cache:
            with suppress(Exception): disk_keys_after = set(self.tool_cache.disk_cache.iterkeys())

        mem_removed = len(mem_keys_before) - len(mem_keys_after)
        disk_removed = len(disk_keys_before) - len(disk_keys_after)
        return mem_removed + disk_removed

    def get_cache_dependencies(self) -> Dict[str, List[str]]:
        """Gets the tool dependency graph."""
        if not self.tool_cache: return {}
        # Convert sets to lists for JSON serialization
        result_dict = {}
        for k, v_set in self.tool_cache.dependency_graph.items():
            result_dict[k] = sorted(v_set) # Sort for consistency
        return result_dict

    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """Gets the input schema for a specific tool."""
        tool = self.server_manager.tools.get(tool_name)
        # Return a copy to prevent modification? Optional.
        return copy.deepcopy(tool.input_schema) if tool else None

    def get_prompt_template(self, prompt_name: str) -> Optional[str]:
        """Gets the template content for a specific prompt."""
        prompt = self.server_manager.prompts.get(prompt_name)
        # Assuming prompt.template holds the content. Adjust if fetching is needed.
        return prompt.template if prompt else None

    def get_server_details(self, server_name: str) -> Optional[Dict]:
        """Gets detailed information about a server configuration and its current state."""
        if server_name not in self.config.servers:
            return None

        server_config = self.config.servers[server_name]
        is_connected = server_name in self.server_manager.active_sessions
        metrics = server_config.metrics

        # Build the details dictionary iteratively
        details = {}
        details["name"] = server_config.name
        details["type"] = server_config.type # Keep as enum for ServerDetail model
        details["path"] = server_config.path
        details["args"] = server_config.args
        details["enabled"] = server_config.enabled
        details["auto_start"] = server_config.auto_start
        details["description"] = server_config.description
        details["trusted"] = server_config.trusted
        details["categories"] = server_config.categories
        details["version"] = str(server_config.version) if server_config.version else None
        details["rating"] = server_config.rating
        details["retry_count"] = server_config.retry_count
        details["timeout"] = server_config.timeout
        details["registry_url"] = server_config.registry_url
        details["capabilities"] = server_config.capabilities
        details["is_connected"] = is_connected
        details["metrics"] = {
            "status": metrics.status.value,
            "avg_response_time_ms": metrics.avg_response_time * 1000,
            "error_count": metrics.error_count,
            "request_count": metrics.request_count,
            "error_rate": metrics.error_rate,
            "uptime_minutes": metrics.uptime,
            "last_checked": metrics.last_checked.isoformat()
        }
        details["process_info"] = None # Initialize

        # Add process info for connected STDIO servers
        if is_connected and server_config.type == ServerType.STDIO and server_name in self.server_manager.processes:
            process = self.server_manager.processes[server_name]
            if process and process.returncode is None:
                try:
                    p = psutil.Process(process.pid)
                    with p.oneshot(): # Efficiently get multiple stats
                        mem_info = p.memory_info()
                        details["process_info"] = {
                            "pid": process.pid,
                            "cpu_percent": p.cpu_percent(interval=0.1), # Interval needed
                            "memory_rss_mb": mem_info.rss / (1024 * 1024),
                            "memory_vms_mb": mem_info.vms / (1024 * 1024),
                            "status": p.status(),
                            "create_time": datetime.fromtimestamp(p.create_time()).isoformat()
                        }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    details["process_info"] = {"error": "Could not retrieve process stats (permissions or process gone)"}
                except Exception as e:
                    details["process_info"] = {"error": f"Error retrieving stats: {e}"}

        return details

    async def reload_servers(self):
        """Disconnects all MCP servers, reloads config (YAML part), and reconnects."""
        log.info("Reloading servers via API request...")
        # Close existing connections first
        if self.server_manager:
             await self.server_manager.close()

        # Re-load the YAML parts of the config
        self.config.load_from_yaml()
        # Re-apply env overrides to ensure they take precedence over newly loaded YAML
        self.config._apply_env_overrides()

        # Re-create server manager with the reloaded config
        # Tool cache can persist
        self.server_manager = ServerManager(self.config, tool_cache=self.tool_cache, safe_printer=self.safe_print)

        # Reconnect to enabled servers based on the reloaded config
        await self.server_manager.connect_to_servers()
        log.info("Server reload complete.")

    async def apply_prompt_to_conversation(self, prompt_name: str) -> bool:
        """Applies a prompt template as a system message to the current conversation."""
        prompt = self.server_manager.prompts.get(prompt_name)
        if not prompt:
            log.warning(f"Prompt '{prompt_name}' not found.")
            return False

        prompt_content = prompt.template # Assuming template holds the content
        if not prompt_content:
            # Add logic here to fetch prompt content if template is just an ID
            log.warning(f"Prompt '{prompt_name}' found but has empty template content.")
            return False

        # Ensure messages list exists
        if not hasattr(self.conversation_graph.current_node, 'messages') or \
           self.conversation_graph.current_node.messages is None:
             self.conversation_graph.current_node.messages = []

        # Prepend the system message
        system_message: InternalMessage = {"role": "system", "content": prompt_content}
        self.conversation_graph.current_node.messages.insert(0, system_message)
        log.info(f"Applied prompt '{prompt_name}' as system message.")

        # Save the updated graph
        await self.conversation_graph.save(str(self.conversation_graph_file))
        return True

    async def reset_configuration(self):
        """Resets the configuration YAML file to defaults."""
        log.warning("Resetting configuration YAML to defaults via API request.")
        # Disconnect all servers first
        if self.server_manager:
             await self.server_manager.close()

        # Create a new default config in memory (sets defaults)
        default_config = Config()
        # Save ONLY its servers and cache_ttl (which will be empty) to YAML
        await default_config.save_async() # This saves the YAML part

        # Reload the current client's config state from the newly saved default YAML
        self.config.load_from_yaml()
        # Re-apply env overrides after loading defaults
        self.config._apply_env_overrides()
        # Re-create server manager
        self.server_manager = ServerManager(self.config, tool_cache=self.tool_cache, safe_printer=self.safe_print)
        # Re-initialize provider clients based on current (likely env var) keys/urls
        await self._reinitialize_provider_clients()
        log.info("Configuration YAML reset to defaults. Env vars still apply.")

    def get_dashboard_data(self) -> Dict:
         """Gets the data structure for the dashboard."""
         # --- Servers Data ---
         servers_data = []
         sorted_server_names = sorted(self.config.servers.keys())
         for name in sorted_server_names:
             server_config = self.config.servers[name]
             if not server_config.enabled: continue
             metrics = server_config.metrics
             is_connected = name in self.server_manager.active_sessions
             health_score = 0
             if is_connected and metrics.request_count > 0:
                 health_penalty = (metrics.error_rate * 100) + max(0, (metrics.avg_response_time - 1.0) * 10)
                 health_score = max(0, min(100, int(100 - health_penalty)))

             server_item = {
                 "name": name, "type": server_config.type.value,
                 "status": metrics.status.value, "is_connected": is_connected,
                 "avg_response_ms": metrics.avg_response_time * 1000,
                 "error_count": metrics.error_count, "request_count": metrics.request_count,
                 "health_score": health_score,
             }
             servers_data.append(server_item)

         # --- Tools Data (Top N by calls) ---
         tools_data = []
         sorted_tools = sorted(self.server_manager.tools.values(), key=lambda t: t.call_count, reverse=True)[:15]
         for tool in sorted_tools:
             tool_item = {
                 "name": tool.name, "server_name": tool.server_name,
                 "call_count": tool.call_count, "avg_execution_time_ms": tool.avg_execution_time,
             }
             tools_data.append(tool_item)

         # --- Client Info Data ---
         # Calculate cache hit rate safely
         cache_hits = getattr(self, 'cache_hit_count', 0)
         cache_misses = getattr(self, 'cache_miss_count', 0)
         total_lookups = cache_hits + cache_misses
         cache_hit_rate = (cache_hits / total_lookups * 100) if total_lookups > 0 else 0.0

         client_info = {
             "current_model": self.current_model,
             "history_entries": len(self.history.entries),
             "cache_entries_memory": len(self.tool_cache.memory_cache) if self.tool_cache else 0,
             "current_branch_id": self.conversation_graph.current_node.id,
             "current_branch_name": self.conversation_graph.current_node.name,
             "cache_hit_count": cache_hits, # Use calculated value
             "cache_miss_count": cache_misses, # Use calculated value
             "tokens_saved_by_cache": getattr(self, 'tokens_saved_by_cache', 0),
             "cache_hit_rate": cache_hit_rate
            }

         # Combine into final structure
         dashboard_result = {
             "timestamp": datetime.now().isoformat(),
             "client_info": client_info,
             "servers": servers_data,
             "tools": tools_data,
         }
         return dashboard_result

    # --- Helper method for processing stream events ---
    def _process_stream_event(self, event: MessageStreamEvent, current_text: str) -> Tuple[str, Optional[str]]:
        """Process a message stream event and handle different event types.
        (Example implementation for Anthropic - Adapt or remove if not used elsewhere)
        """
        text_to_yield = None
        if event.type == "content_block_delta":
            delta_event: ContentBlockDeltaEvent = event
            delta = delta_event.delta
            if delta.type == "text_delta":
                current_text += delta.text
                text_to_yield = delta.text
        elif event.type == "content_block_start":
            if event.content_block.type == "text":
                current_text = "" # Reset
            elif event.content_block.type == "tool_use":
                original_name = self.server_manager.sanitized_to_original.get(event.content_block.name, event.content_block.name)
                text_to_yield = f"\n[{EMOJI_MAP['tool']}] Using tool: {original_name}..."
        return current_text, text_to_yield

    async def export_conversation(self, conversation_id: str, file_path: str) -> bool:
        """Export a conversation branch to a file with progress tracking"""
        export_data = await self.get_conversation_export_data(conversation_id)
        if export_data is None:
             self.safe_print(f"[red]Conversation ID '{conversation_id}' not found for export.[/]")
             return False

        # Write to file asynchronously
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                # Use dumps with ensure_ascii=False for better unicode handling
                json_string = json.dumps(export_data, indent=2, ensure_ascii=False)
                await f.write(json_string)
            log.info(f"Conversation branch '{conversation_id}' exported to {file_path}")
            return True
        except Exception as e:
            self.safe_print(f"[red]Failed to write export file {file_path}: {e}[/]")
            log.error(f"Failed to export conversation {conversation_id} to {file_path}", exc_info=True)
            return False

    async def import_conversation(self, file_path: str) -> bool:
        """Import a conversation from a file into a new branch."""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
            data = json.loads(content)
        except FileNotFoundError:
            self.safe_print(f"[red]Import file not found: {file_path}[/]")
            return False
        except json.JSONDecodeError as e:
            self.safe_print(f"[red]Invalid JSON in import file {file_path}: {e}[/]")
            return False
        except Exception as e:
            self.safe_print(f"[red]Error reading import file {file_path}: {e}[/]")
            return False

        success, message, new_node_id = await self.import_conversation_from_data(data)

        if success:
             self.safe_print(f"[green]Conversation imported from {file_path}. New branch ID: {new_node_id}[/]")
             # Automatically checkout the new branch?
             if new_node_id and self.conversation_graph.set_current_node(new_node_id):
                  self.safe_print(f"[cyan]Switched to imported branch.[/]")
             else:
                  self.safe_print(f"[yellow]Could not switch to imported branch {new_node_id}.[/]")
        else:
            self.safe_print(f"[red]Import failed: {message}[/]")

        return success

    async def _execute_llm_call_no_history(
        self,
        messages: InternalMessageList,
        model: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Executes a non-streaming LLM call for internal purposes (like summarization)
        without updating the main conversation graph or history.

        Args:
            messages: The list of messages in InternalMessage format to send.
            model: The specific model name to use.
            max_tokens: Maximum tokens for the response.
            temperature: Temperature for the generation.

        Returns:
            The generated text content as a string, or None if the call fails.
        """
        provider_name = self.get_provider_from_model(model)
        if not provider_name:
            log.error(f"Cannot execute internal call: Unknown provider for model '{model}'")
            return None

        provider_client = getattr(self, f"{provider_name}_client", None)
        if provider_name == Provider.ANTHROPIC.value and not provider_client:
            provider_client = self.anthropic # Special case for anthropic client attribute name
        if not provider_client:
            log.error(f"Cannot execute internal call: Client not initialized for provider '{provider_name}'")
            return None

        max_tokens_to_use = max_tokens or self.config.default_max_tokens
        temperature_to_use = temperature if temperature is not None else self.config.temperature

        log.info(f"Executing internal LLM call (no history): Model='{model}', Provider='{provider_name}'")

        # Format messages for the specific provider
        formatted_messages, system_prompt, _ = self._format_messages_for_provider(
            messages, provider_name, model
        )

        try:
            response_text: Optional[str] = None

            # --- Provider-Specific Non-Streaming Call ---
            if provider_name == Provider.ANTHROPIC.value:
                response = await cast('AsyncAnthropic', provider_client).messages.create(
                    model=model,
                    messages=formatted_messages, # type: ignore # Pydantic v2 compat
                    system=system_prompt, # type: ignore # Pydantic v2 compat
                    max_tokens=max_tokens_to_use,
                    temperature=temperature_to_use,
                )
                # Extract text content (Anthropic usually returns content blocks)
                if response.content and isinstance(response.content, list):
                    text_parts = [block.text for block in response.content if block.type == 'text']
                    response_text = "\n".join(text_parts).strip()
                elif isinstance(response.content, str): # Should not happen based on API, but handle
                    response_text = response.content

            elif provider_name in [
                Provider.OPENAI.value, Provider.GROK.value, Provider.DEEPSEEK.value,
                Provider.GROQ.value, Provider.MISTRAL.value, Provider.CEREBRAS.value, Provider.GEMINI.value
            ]:
                response = await cast('AsyncOpenAI', provider_client).chat.completions.create(
                    model=model,
                    messages=formatted_messages, # type: ignore # Pydantic v2 compat
                    max_tokens=max_tokens_to_use,
                    temperature=temperature_to_use,
                    stream=False, # Explicitly non-streaming
                    # tools=None, tool_choice=None # No tools needed for internal call
                )
                if response.choices and response.choices[0].message:
                    response_text = response.choices[0].message.content

            else:
                log.error(f"Internal call not implemented for provider: {provider_name}")
                return None

            # --- Log completion and return ---
            log.info(f"Internal LLM call completed. Response length: {len(response_text or '')} chars.")
            return response_text.strip() if response_text else None

        # --- Error Handling ---
        except (anthropic.APIConnectionError, openai.APIConnectionError) as e:
            log.error(f"Internal Call Connection Error ({provider_name}): {e}")
        except (anthropic.AuthenticationError, openai.AuthenticationError):
            log.error(f"Internal Call Authentication Error ({provider_name}). Check API Key.")
        except (anthropic.RateLimitError, openai.RateLimitError):
            log.warning(f"Internal Call Rate Limit Exceeded ({provider_name}).")
        except (anthropic.APIError, openai.APIError) as e:
            log.error(f"Internal Call API Error ({provider_name}): {e}", exc_info=True)
        except Exception as e:
            log.error(f"Unexpected error during internal LLM call ({provider_name}): {e}", exc_info=True)

        return None # Return None on any failure        

# Inside class MCPClient:

    async def summarize_conversation(self, target_tokens: Optional[int] = None, model: Optional[str] = None) -> Optional[str]:
        """
        Generates a summary of the current conversation branch *without* modifying history.

        Args:
            target_tokens: Approximate target token length for the summary.
            model: The model to use for summarization. Falls back to config default if None.

        Returns:
            The generated summary string, or None if summarization failed.
        """
        summarization_model = model or self.config.summarization_model
        target_length = target_tokens or self.config.max_summarized_tokens
        # Use the messages from the *current* node for context
        current_messages_to_summarize = self.conversation_graph.current_node.messages

        if not current_messages_to_summarize:
            log.info("Cannot summarize empty conversation.")
            # Return specific string or None based on desired behavior for empty history
            return "Current conversation branch is empty."

        log.info(f"Generating summary using {summarization_model} (target: ~{target_length} tokens) without history update...")

        # --- Create the Summarization Prompt ---
        # Combine a system-like instruction with the actual history
        summarization_prompt = (
            "You are an expert summarizer. Please summarize the following conversation history. "
            "Focus on preserving key facts, decisions, action items, important code snippets, "
            "numerical values, and the overall context needed to continue the conversation effectively. "
            "Be concise but comprehensive."
            f" Aim for a summary that is roughly {target_length} tokens long."
            # "CONVERSATION HISTORY:" # Removed this header, just send messages
        )

        # Prepare the message list for the internal call:
        # System prompt + Current History
        messages_for_summary_call: InternalMessageList = [
            InternalMessage(role="system", content=summarization_prompt)
        ]
        messages_for_summary_call.extend(current_messages_to_summarize)

        # --- Call the new internal helper ---
        try:
            # Execute the call without side effects
            summary_text = await self._execute_llm_call_no_history(
                messages=messages_for_summary_call,
                model=summarization_model,
                max_tokens=target_length + 500, # Allow slightly more tokens for the summary itself
                temperature=0.5 # Lower temperature might be better for summarization
            )

            if summary_text:
                log.info(f"Summarization successful. Summary length: {len(summary_text)} chars.")
                return summary_text.strip()
            else:
                log.warning("Summarization model returned no content.")
                return None # Explicitly return None on empty response

        except Exception as e:
            # Error is logged within the helper, just return None here
            log.error(f"Summarization failed: Error occurred during internal LLM call - {e}", exc_info=False) # Don't need full traceback here again
            return None

    def _format_messages_for_provider(
        self,
        messages: InternalMessageList,
        provider: str,
        model_name: str # Keep model_name for potential future provider-specific logic
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Formats internal message list into the specific structure required by the provider.

        Args:
            messages: The list of messages in the internal canonical format.
            provider: The string identifier of the target provider.
            model_name: The specific model being used (for potential future use).

        Returns:
            A tuple containing:
            - formatted_messages: List of message dictionaries for the provider's API.
            - system_prompt: The extracted system prompt string (used by Anthropic), or None.
        """
        formatted_messages: List[Dict[str, Any]] = []
        system_prompt: Optional[str] = None
        provider_enum_val = provider # Use string value

        log.debug(f"Formatting {len(messages)} messages for provider '{provider}'.")

        # --- Anthropic Formatting ---
        if provider_enum_val == Provider.ANTHROPIC.value:
            first_system_processed = False
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                # Handle System Prompt
                if role == "system":
                    if not first_system_processed:
                        system_prompt = self._extract_text_from_internal_content(content)
                        first_system_processed = True
                        log.debug(f"Extracted Anthropic system prompt: '{system_prompt[:100]}...'")
                    else:
                        log.warning("Multiple system messages found; only the first is used for Anthropic.")
                    continue # Skip adding system messages to the main list for Anthropic

                # Handle User and Assistant Roles
                api_role = role # Roles 'user' and 'assistant' map directly
                api_content: Any

                if isinstance(content, str):
                    api_content = content
                elif isinstance(content, list):
                    api_content_list: List[Dict[str, Any]] = []
                    for block in content:
                        block_type = block.get("type")
                        if block_type == "text":
                            api_content_list.append({"type": "text", "text": block.get("text", "")})
                        elif block_type == "tool_use":
                            original_tool_name = block.get("name", "unknown_tool")
                            # Look up the SANITIZED name used when defining tools for Anthropic
                            sanitized_name = None
                            for s_name, o_name in self.server_manager.sanitized_to_original.items():
                                 if o_name == original_tool_name:
                                     sanitized_name = s_name
                                     break
                            if not sanitized_name:
                                 log.warning(f"Could not find sanitized name for Anthropic tool_use: '{original_tool_name}'. Using original.")
                                 sanitized_name = original_tool_name # Fallback, might fail API validation
                            api_content_list.append({
                                "type": "tool_use",
                                "id": block.get("id", ""),
                                "name": sanitized_name, # Use SANITIZED name for Anthropic tool_use
                                "input": block.get("input", {})
                            })
                        elif block_type == "tool_result":
                             # Anthropic expects tool result content as string or list of blocks.
                             # We'll stringify complex results for consistency for now.
                             result_content = block.get("content")
                             stringified_content = self._stringify_content(result_content)

                             # Add is_error field if present in the internal block
                             result_block = {
                                 "type": "tool_result",
                                 "tool_use_id": block.get("tool_use_id", ""),
                                 "content": stringified_content # Send stringified content
                             }
                             if block.get("is_error") is True: # Check explicitly for True
                                 result_block["is_error"] = True
                             api_content_list.append(result_block)

                    api_content = api_content_list
                else:
                    # Fallback for unexpected content type
                    log.warning(f"Unexpected content type for Anthropic: {type(content)}. Converting to string.")
                    api_content = str(content)

                # Append the formatted message
                formatted_messages.append({"role": api_role, "content": api_content})

        # --- OpenAI-Compatible Formatting (OpenAI, Grok, DeepSeek, Groq, Gemini, Mistral, Cerebras) ---
        elif provider_enum_val in [
            Provider.OPENAI.value, Provider.GROK.value, Provider.DEEPSEEK.value,
            Provider.GROQ.value, Provider.GEMINI.value, Provider.MISTRAL.value, Provider.CEREBRAS.value
        ]:
            for msg in messages:
                role = msg["role"]
                content = msg["content"]

                # Handle System Prompt
                if role == "system":
                    # OpenAI uses a standard message object for system prompts
                    system_text = self._extract_text_from_internal_content(content)
                    formatted_messages.append({"role": "system", "content": system_text})
                    log.debug(f"Adding OpenAI system message: '{system_text[:100]}...'")

                # Handle User Messages (excluding internal Tool Results)
                elif role == "user":
                    # Check if this user message is *actually* a tool result wrapper in our internal format
                    is_internal_tool_result = False
                    tool_call_id_for_result = None
                    tool_result_content = None
                    tool_is_error = False # Track if the result was an error

                    if isinstance(content, list) and content:
                        first_block = content[0]
                        if isinstance(first_block, dict) and first_block.get("type") == "tool_result":
                            is_internal_tool_result = True
                            tool_call_id_for_result = first_block.get("tool_use_id")
                            tool_result_content = first_block.get("content")
                            tool_is_error = first_block.get("is_error", False) # Check for error flag

                    if is_internal_tool_result:
                        # Create a 'tool' role message for OpenAI
                        if tool_call_id_for_result:
                            # OpenAI expects string content for the tool role
                            stringified_result = self._stringify_content(tool_result_content)
                            # Include error information implicitly in the content string if needed,
                            # as OpenAI 'tool' role doesn't have a dedicated error flag.
                            # If the result was marked as an error, we might prepend "Error: "
                            final_tool_content = f"Error: {stringified_result}" if tool_is_error else stringified_result

                            formatted_messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call_id_for_result,
                                "content": final_tool_content
                            })
                            log.debug(f"Adding OpenAI tool message for ID {tool_call_id_for_result}")
                        else:
                            log.warning("Internal tool result block found but missing 'tool_use_id'. Skipping.")
                    else:
                        # Regular user message - extract text content only
                        user_text = self._extract_text_from_internal_content(content)
                        formatted_messages.append({"role": "user", "content": user_text})
                        log.debug(f"Adding OpenAI user message: '{user_text[:100]}...'")

                # Handle Assistant Messages
                elif role == "assistant":
                    assistant_text_content: Optional[str] = None
                    tool_calls_for_api: List[Dict[str, Any]] = []

                    if isinstance(content, str):
                        assistant_text_content = content
                    elif isinstance(content, list):
                        text_parts = []
                        for block in content:
                            block_type = block.get("type")
                            if block_type == "text":
                                text_parts.append(block.get("text", ""))
                            elif block_type == "tool_use":
                                original_tool_name = block.get("name", "unknown_tool")
                                tool_call_id = block.get("id", "")
                                tool_input = block.get("input", {})

                                # Look up SANITIZED name used when defining tools for OpenAI
                                sanitized_name = None
                                for s_name, o_name in self.server_manager.sanitized_to_original.items():
                                    if o_name == original_tool_name:
                                        sanitized_name = s_name
                                        break
                                if not sanitized_name:
                                     log.warning(f"Could not find sanitized name for OpenAI tool_call: '{original_tool_name}'. Using original.")
                                     sanitized_name = original_tool_name # Fallback

                                # OpenAI expects arguments as a JSON *string*
                                try:
                                    arguments_str = json.dumps(tool_input)
                                except TypeError as e:
                                    log.error(f"Could not JSON-stringify tool input for '{sanitized_name}' (ID: {tool_call_id}): {e}. Sending empty args.")
                                    arguments_str = "{}" # Send empty JSON string on error

                                tool_calls_for_api.append({
                                    "id": tool_call_id,
                                    "type": "function",
                                    "function": {
                                        "name": sanitized_name, # Use SANITIZED name
                                        "arguments": arguments_str,
                                    }
                                })
                        assistant_text_content = "\n".join(text_parts).strip() or None # Use None if only whitespace

                    # Construct the OpenAI assistant message
                    assistant_msg_payload: Dict[str, Any] = {"role": "assistant"}
                    if assistant_text_content:
                        assistant_msg_payload["content"] = assistant_text_content
                    else:
                         # OpenAI requires *some* content field for assistant if tool_calls is present,
                         # even if it's null or empty string when only tools are called.
                         # Setting to None is generally accepted by the SDK.
                         assistant_msg_payload["content"] = None


                    if tool_calls_for_api:
                        assistant_msg_payload["tool_calls"] = tool_calls_for_api

                    # Add the message only if it has text content OR tool calls
                    if assistant_msg_payload.get("content") is not None or assistant_msg_payload.get("tool_calls"):
                         formatted_messages.append(assistant_msg_payload)
                         log.debug(f"Adding OpenAI assistant message. Text: {bool(assistant_text_content)}, Tools: {len(tool_calls_for_api)}")
                    else:
                         log.debug("Skipping empty OpenAI assistant message.")


        # --- Unknown Provider ---
        else:
            log.error(f"Message formatting failed: Provider '{provider}' is not supported.")
            # Return the original messages and no system prompt as a fallback? Or raise error?
            # Let's return empty list and None to signal failure clearly.
            return [], None

        log.debug(f"Formatted {len(formatted_messages)} messages. System Prompt: {bool(system_prompt)}")
        return formatted_messages, system_prompt

    async def _handle_anthropic_stream(self, stream: AsyncMessageStream) -> AsyncGenerator[Tuple[str, Any], None]:
        """Process Anthropic stream and emit standardized events."""
        current_text_block = None
        current_tool_use_block = None
        current_tool_input_json_accumulator = ""
        input_tokens = 0
        output_tokens = 0
        stop_reason = "unknown"

        try:
            async for event in stream:
                event_type = event.type
                if event_type == "message_start":
                    input_tokens = event.message.usage.input_tokens
                elif event_type == "content_block_start":
                    block_type = event.content_block.type
                    if block_type == "text":
                        current_text_block = {"type": "text", "text": ""}
                    elif block_type == "tool_use":
                        tool_id = event.content_block.id
                        tool_name = event.content_block.name
                        current_tool_use_block = {"id": tool_id, "name": tool_name}
                        current_tool_input_json_accumulator = ""
                        yield ("tool_call_start", {"id": tool_id, "name": tool_name})
                elif event_type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        if current_text_block is not None:
                            yield ("text_chunk", delta.text)
                    elif delta.type == "input_json_delta":
                        if current_tool_use_block is not None:
                            current_tool_input_json_accumulator += delta.partial_json
                            # Yield incremental input chunks for potential UI display
                            yield ("tool_call_input_chunk", {"id": current_tool_use_block["id"], "json_chunk": delta.partial_json})
                elif event_type == "content_block_stop":
                    if current_text_block is not None:
                        current_text_block = None # Block finished
                    elif current_tool_use_block is not None:
                        parsed_input = {}
                        try:
                            parsed_input = json.loads(current_tool_input_json_accumulator)
                        except json.JSONDecodeError as e:
                            log.error(f"Anthropic JSON parse failed: {e}. Raw: '{current_tool_input_json_accumulator}'")
                            parsed_input = {"_tool_input_parse_error": f"Failed: {e}"}
                        yield ("tool_call_end", {"id": current_tool_use_block["id"], "parsed_input": parsed_input})
                        current_tool_use_block = None
                elif event_type == "message_delta":
                    if hasattr(event.delta, 'stop_reason') and event.delta.stop_reason:
                        stop_reason = event.delta.stop_reason
                    if hasattr(event, 'usage') and event.usage: # Track output tokens from delta
                        output_tokens = event.usage.output_tokens # Anthropic provides cumulative output tokens
                elif event_type == "message_stop":
                    # Get final reason and usage from the final message object
                    final_message = await stream.get_final_message()
                    stop_reason = final_message.stop_reason
                    output_tokens = final_message.usage.output_tokens
                    # Break after processing final message info
                    break

        except anthropic.APIError as e:
            log.error(f"Anthropic stream API error: {e}")
            yield ("error", f"Anthropic API Error: {e}")
            stop_reason = "error"
        except Exception as e:
            log.error(f"Unexpected error in Anthropic stream handler: {e}", exc_info=True)
            yield ("error", f"Unexpected stream processing error: {e}")
            stop_reason = "error"
        finally:
            yield ("final_usage", {"input_tokens": input_tokens, "output_tokens": output_tokens})
            yield ("stop_reason", stop_reason)

    async def _handle_openai_stream(self, stream: AsyncStream[ChatCompletionChunk]) -> AsyncGenerator[Tuple[str, Any], None]:
        """Process OpenAI/Grok/DeepSeek stream and emit standardized events."""
        current_tool_calls: Dict[int, Dict] = {} # {index: {'id':..., 'name':..., 'args':...}}
        current_tool_args_accumulator: Dict[int, str] = {} # {index: accumulated_args_str}
        collected_chunks_for_usage = [] # Collect chunks if usage not in stream
        input_tokens = 0 # Often not available in stream
        output_tokens = 0 # Often not available in stream
        stop_reason = "stop" # Default
        finish_reason = None

        try:
            async for chunk in stream:
                collected_chunks_for_usage.append(chunk) # Collect for potential post-processing
                choice = chunk.choices[0] if chunk.choices else None
                if not choice: continue

                delta = choice.delta
                finish_reason = choice.finish_reason # Store the latest finish_reason

                # 1. Text Chunks
                if delta and delta.content:
                    yield ("text_chunk", delta.content)

                # 2. Tool Calls
                if delta and delta.tool_calls:
                    for tool_call_chunk in delta.tool_calls:
                        idx = tool_call_chunk.index
                        # Start of a new tool call
                        if tool_call_chunk.id and tool_call_chunk.function and tool_call_chunk.function.name:
                            tool_id = tool_call_chunk.id
                            tool_name = tool_call_chunk.function.name
                            current_tool_calls[idx] = {"id": tool_id, "name": tool_name, "args": ""}
                            current_tool_args_accumulator[idx] = ""
                            yield ("tool_call_start", {"id": tool_id, "name": tool_name})

                        # Argument chunks for an existing tool call
                        if tool_call_chunk.function and tool_call_chunk.function.arguments:
                            args_chunk = tool_call_chunk.function.arguments
                            if idx in current_tool_args_accumulator:
                                current_tool_args_accumulator[idx] += args_chunk
                                # Yield incremental input chunk
                                if idx in current_tool_calls: # Ensure id exists
                                    yield ("tool_call_input_chunk", {"id": current_tool_calls[idx]["id"], "json_chunk": args_chunk})
                            else:
                                log.warning(f"Received args chunk for unknown tool index {idx}")


            # After stream ends, finalize tool calls
            for idx, accumulated_args in current_tool_args_accumulator.items():
                if idx in current_tool_calls:
                    tool_info = current_tool_calls[idx]
                    parsed_input = {}
                    try:
                        if accumulated_args: # Only parse if not empty
                            parsed_input = json.loads(accumulated_args)
                    except json.JSONDecodeError as e:
                        log.error(f"OpenAI JSON parse failed for tool {tool_info['name']} (ID: {tool_info['id']}): {e}. Raw: '{accumulated_args}'")
                        parsed_input = {"_tool_input_parse_error": f"Failed: {e}"}
                    yield ("tool_call_end", {"id": tool_info["id"], "parsed_input": parsed_input})

            # Determine final stop reason
            stop_reason = finish_reason if finish_reason else "stop"
            # If finish_reason indicates tool use, standardize it
            if stop_reason == "tool_calls":
                stop_reason = "tool_use"

            # Try to get usage (may not be present in stream)
            # Some OpenAI proxy might add usage in the last chunk or x-headers
            final_usage = getattr(chunk, 'usage', None) # Check last chunk
            if final_usage:
                input_tokens = final_usage.prompt_tokens
                output_tokens = final_usage.completion_tokens

        except openai.APIError as e:
            log.error(f"OpenAI/Grok/DeepSeek stream API error: {e}")
            yield ("error", f"API Error: {e}")
            stop_reason = "error"
        except Exception as e:
            log.error(f"Unexpected error in OpenAI stream handler: {e}", exc_info=True)
            yield ("error", f"Unexpected stream processing error: {e}")
            stop_reason = "error"
        finally:
            # Yield potentially estimated usage
            if input_tokens == 0 or output_tokens == 0:
                log.warning("OpenAI stream did not provide token counts. Usage will be estimated.")
            yield ("final_usage", {"input_tokens": input_tokens, "output_tokens": output_tokens})
            yield ("stop_reason", stop_reason)

    async def _initialize_openai_compatible_client(
        self,
        provider_name: str,
        api_key_attr: str,
        base_url_attr: str,
        default_base_url: Optional[str],
        client_attr: str,
        emoji_key: str
    ) -> Tuple[Optional[AsyncOpenAI], str]:
        """
        Initializes and validates an AsyncOpenAI client for a compatible provider.
        (Enhanced Error Handling)
        """
        status_emoji = EMOJI_MAP.get(emoji_key, EMOJI_MAP['provider'])
        provider_title = provider_name.capitalize()
        api_key = getattr(self.config, api_key_attr, None)

        if not api_key:
            return None, f"{status_emoji} {provider_title}: [yellow]No Key[/]"

        client_instance = None # Initialize to None
        try:
            base_url = getattr(self.config, base_url_attr, None) or default_base_url
            if not base_url and provider_name != Provider.OPENAI.value: # OpenAI default is handled by SDK
                 log.warning(f"No base URL found for {provider_title}.")
                 # Allow SDK default ONLY for openai, fail others? Or use known defaults?
                 # Sticking with explicit defaults where possible.
                 # If default_base_url was None here, it's an issue.
                 if default_base_url is None:
                      raise ValueError(f"Default base URL is missing for {provider_title}")


            log.debug(f"Initializing {provider_title} client. Key: ***{api_key[-4:]}, Base URL: {base_url}")
            client_instance = AsyncOpenAI(api_key=api_key, base_url=base_url)

            # Lightweight validation check (e.g., list models)
            log.debug(f"Validating {provider_title} client by listing models (URL: {client_instance.base_url})...")
            await client_instance.models.list() # Raises errors on failure
            log.info(f"{provider_title} client initialized and validated successfully.")

            setattr(self, client_attr, client_instance)
            return client_instance, f"{status_emoji} {provider_title}: [green]OK[/]"

        # --- More Specific Error Handling ---
        except OpenAIAuthenticationError:
            error_msg = f"{provider_title} initialization failed: Invalid API Key."
            log.error(error_msg)
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Auth Error[/]"
        except OpenAIAPIConnectionError as e: # Often wraps httpx errors
            error_msg = f"{provider_title} initialization failed: Connection Error ({e})"
            log.error(error_msg, exc_info=True) # Include details
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Conn Error[/]"
        except openai.BadRequestError as e: # e.g., Malformed URL during validation
            error_msg = f"{provider_title} initialization failed: Bad Request ({e})"
            log.error(error_msg, exc_info=True)
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Bad Request[/]"
        except openai.NotFoundError as e: # e.g., Base URL valid but path incorrect
            error_msg = f"{provider_title} initialization failed: Not Found ({e})"
            log.error(error_msg, exc_info=True)
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Not Found[/]"
        except httpx.RequestError as e: # Catch underlying network issues if not wrapped by OpenAI error
            error_msg = f"{provider_title} initialization failed: Network Error ({e})"
            log.error(error_msg, exc_info=True)
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Network Err[/]"
        except ValueError as e: # Catch our manual ValueError for missing default URL
             error_msg = f"{provider_title} initialization failed: Configuration Error ({e})"
             log.error(error_msg)
             self.safe_print(f"[bold red]{error_msg}[/]")
             setattr(self, client_attr, None)
             return None, f"{status_emoji} {provider_title}: [red]Config Error[/]"
        except Exception as e: # Catch-all for unexpected init errors
            error_msg = f"{provider_title} initialization failed: Unexpected Error ({type(e).__name__})"
            log.error(error_msg, exc_info=True)
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Failed[/]"

    async def setup(self, interactive_mode=False):
        """Set up the client, load configs, initialize providers, discover servers."""
        safe_console = get_safe_console()

        # Mappings for provider names, config attributes, and env variable names
        provider_keys = { # Map provider enum value to config attribute
            Provider.ANTHROPIC.value: "anthropic_api_key",
            Provider.OPENAI.value: "openai_api_key",
            Provider.GEMINI.value: "gemini_api_key",
            Provider.GROK.value: "grok_api_key",
            Provider.DEEPSEEK.value: "deepseek_api_key",
            Provider.MISTRAL.value: "mistral_api_key",
            Provider.GROQ.value: "groq_api_key",
            Provider.CEREBRAS.value: "cerebras_api_key",
            Provider.OPENROUTER.value: "openrouter_api_key" # Added OpenRouter
        }
        provider_env_vars = { # Map provider enum value to expected .env var name
            Provider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
            Provider.OPENAI.value: "OPENAI_API_KEY",
            Provider.GEMINI.value: "GOOGLE_API_KEY", # Special case for Gemini - OR GEMINI_API_KEY? Check decouple logic. Assuming GOOGLE_API_KEY based on common practice.
            Provider.GROK.value: "GROK_API_KEY",
            Provider.DEEPSEEK.value: "DEEPSEEK_API_KEY",
            Provider.MISTRAL.value: "MISTRAL_API_KEY",
            Provider.GROQ.value: "GROQ_API_KEY",
            Provider.CEREBRAS.value: "CEREBRAS_API_KEY",
            Provider.OPENROUTER.value: "OPENROUTER_API_KEY" # Added OpenRouter
        }

        # --- 1. Default Provider API Key Check & Prompt ---
        default_provider = None
        default_provider_key_attr = None
        default_provider_key_env_var = None
        key_missing = False

        # Get the path to the .env file found or defaulted by Config.__init__
        # This path is now reliably set.
        dotenv_path = self.config.dotenv_path

        # Determine default provider
        default_provider_name = os.getenv("DEFAULT_PROVIDER")
        if not default_provider_name and dotenv_path and Path(dotenv_path).exists():
            env_values = dotenv_values(dotenv_path)
            default_provider_name = env_values.get("DEFAULT_PROVIDER")

        if default_provider_name:
            try:
                default_provider = Provider(default_provider_name.lower())
                default_provider_key_attr = provider_keys.get(default_provider.value)
                default_provider_key_env_var = provider_env_vars.get(default_provider.value)

                if default_provider_key_attr and default_provider_key_env_var:
                    # Check current config object (populated by defaults, YAML, env vars)
                    current_key_value = getattr(self.config, default_provider_key_attr, None)
                    if not current_key_value:
                        key_missing = True
                        log.warning(f"API key for default provider '{default_provider.value}' ({default_provider_key_env_var}) is missing or empty in current config.")
                    else:
                        log.info(f"API key for default provider '{default_provider.value}' found in current config.")
                else:
                    log.warning(f"Default provider '{default_provider_name}' specified but key/env mapping missing.")
                    default_provider = None
            except ValueError:
                log.error(f"Invalid DEFAULT_PROVIDER specified: '{default_provider_name}'")
                default_provider = None

        # Prompt only if interactive, default provider known, and key missing
        if interactive_mode and default_provider and key_missing:
            self.safe_print(f"[yellow]API key for default provider '{default_provider.value}' ({default_provider_key_env_var}) is missing.[/]")
            self.safe_print(f"You can enter the API key now, or press Enter to skip.")

            try:
                api_key_input = Prompt.ask(
                    f"Enter {default_provider.value.capitalize()} API Key",
                    default="",
                    console=safe_console,
                    password=True
                )

                if api_key_input.strip():
                    entered_key = api_key_input.strip()

                    # Use the dotenv_path stored in the config object
                    target_dotenv_path = self.config.dotenv_path

                    if target_dotenv_path: # Should always be true now
                        try:
                            # Ensure parent directory exists
                            Path(target_dotenv_path).parent.mkdir(parents=True, exist_ok=True)
                            # Use dotenv.set_key with the correct path
                            success = set_key(target_dotenv_path, default_provider_key_env_var, entered_key, quote_mode='always')
                            if success:
                                self.safe_print(f"[green]API key for {default_provider.value} saved to {target_dotenv_path}[/]")
                                # Update the *current* config object immediately
                                setattr(self.config, default_provider_key_attr, entered_key)
                                log.info(f"Updated config.{default_provider_key_attr} in memory.")
                            else:
                                 # Log error, but still use key for session
                                 self.safe_print(f"[red]Error: Failed to save API key to {target_dotenv_path}. Key used for session only.[/]")
                                 setattr(self.config, default_provider_key_attr, entered_key)
                        except Exception as save_err:
                             # Log error, but still use key for session
                             self.safe_print(f"[red]Error saving to {target_dotenv_path}: {save_err}. Key used for session only.[/]")
                             setattr(self.config, default_provider_key_attr, entered_key)
                    # --- REMOVED REDUNDANT BLOCK ---
                    # The logic previously here was duplicating the block above.
                else:
                    # User pressed Enter, key remains missing
                    self.safe_print(f"[yellow]Skipped entering key for {default_provider.value}. {default_provider.value.capitalize()} features might be unavailable.[/]")

            except Exception as prompt_err:
                self.safe_print(f"[red]Error during API key prompt/save: {prompt_err}[/]")
                log.error("Error during API key prompt/save", exc_info=True)

        # Exit if not interactive and default provider key is missing
        elif not interactive_mode and default_provider and key_missing:
             self.safe_print(f"[bold red]ERROR: API key for default provider '{default_provider.value}' ({default_provider_key_env_var}) not found.[/]")
             self.safe_print("Please set the key in your '.env' file or as an environment variable.")
             sys.exit(1)

        # --- 2. Initialize Provider SDK Clients ---
        # This will now use the correct key value in self.config if it was updated above
        await self._reinitialize_provider_clients() # Call the helper to init all needed clients

        # --- 3. Load Conversation Graph ---
        # (No changes needed here)
        self.conversation_graph = ConversationGraph()
        if self.conversation_graph_file.exists():
            with Status(f"{EMOJI_MAP['history']} Loading conversation state...", console=safe_console) as status:
                try:
                    loaded_graph = await ConversationGraph.load(str(self.conversation_graph_file))
                    self.conversation_graph = loaded_graph
                    is_new_graph = (loaded_graph.root.id == "root" and not loaded_graph.root.messages and not loaded_graph.root.children and len(loaded_graph.nodes) == 1)
                    if is_new_graph and self.conversation_graph_file.read_text().strip():
                        self.safe_print("[yellow]Could not parse previous conversation state, starting fresh.[/yellow]")
                        status.update(f"{EMOJI_MAP['warning']} Previous state invalid, starting fresh")
                    else:
                        log.info(f"Loaded conversation graph from {self.conversation_graph_file}")
                        status.update(f"{EMOJI_MAP['success']} Conversation state loaded")
                except Exception as setup_load_err:
                    log.error("Unexpected error during conversation graph loading", exc_info=True)
                    self.safe_print(f"[red]Error loading state: {setup_load_err}[/red]")
                    status.update(f"{EMOJI_MAP['error']} Error loading state")
                    self.conversation_graph = ConversationGraph()
        else:
            log.info("No existing conversation graph found, using new graph.")
        if not self.conversation_graph.get_node(self.conversation_graph.current_node.id):
            log.warning("Current node ID invalid, reset root.")
            self.conversation_graph.set_current_node("root")

        # --- 4. Load Claude Desktop Config ---
        await self.load_claude_desktop_config()

        # --- 5. Clean Duplicate Server Configs ---
        # (No changes needed here)
        log.info("Cleaning duplicate server configurations...")
        cleaned_servers: Dict[str, ServerConfig] = {}
        canonical_map: Dict[Tuple, str] = {}
        duplicates_found = False
        servers_to_process = list(self.config.servers.items())
        for name, server_config in servers_to_process:
            identifier: Optional[Tuple] = None
            if server_config.type == ServerType.STDIO: identifier = (server_config.type, server_config.path, frozenset(server_config.args))
            elif server_config.type == ServerType.SSE: identifier = (server_config.type, server_config.path)
            else:
                log.warning(f"Server '{name}' unknown type '{server_config.type}'")
                identifier = (server_config.type, name)
            if identifier is not None:
                if identifier not in canonical_map:
                    canonical_map[identifier] = name
                    cleaned_servers[name] = server_config
                    log.debug(f"Keeping server: '{name}'")
                else:
                    duplicates_found = True
                    kept_name = canonical_map[identifier]
                    log.debug(f"Duplicate server detected. Removing '{name}', keep '{kept_name}'.")
        if duplicates_found:
            num_removed = len(self.config.servers) - len(cleaned_servers)
            self.safe_print(f"[yellow]Removed {num_removed} duplicate server entries.[/yellow]")
            self.config.servers = cleaned_servers
            await self.config.save_async() # Saves YAML
        else:
            log.info("No duplicate server configurations found.")


        # --- 6. Stdout Pollution Check ---
        if os.environ.get("MCP_VERIFY_STDOUT", "1") == "1":
            with safe_stdout():
                log.info("Verifying no stdout pollution before connect...")
                verify_no_stdout_pollution()

        # --- 7. Discover Servers ---
        if self.config.auto_discover:
            self.safe_print(f"{EMOJI_MAP['search']} Discovering MCP servers...")
            try:
                await self.server_manager.discover_servers() # Populates cache
                await self.server_manager._process_discovery_results(interactive_mode=interactive_mode) # Adds to config
            except Exception as discover_error:
                log.error("Error during discovery", exc_info=True)
                self.safe_print(f"[red]Discovery error: {discover_error}[/]")

        # --- 8. Start Continuous Local Discovery ---
        await self.start_local_discovery_monitoring()

        # --- 9. Connect to Enabled MCP Servers ---
        servers_to_connect = {name: cfg for name, cfg in self.config.servers.items() if cfg.enabled}
        if servers_to_connect:
            self.safe_print(f"[bold blue]Connecting to {len(servers_to_connect)} MCP servers...[/]")
            connection_results = {}
            for name, server_config in list(servers_to_connect.items()):
                self.safe_print(f"[cyan]Connecting to MCP server {name}...[/]")
                try:
                    session = await self.server_manager.connect_to_server(server_config)
                    # Name might have changed during connection due to identification
                    final_name = server_config.name
                    connection_results[name] = (session is not None)
                    if session:
                        self.safe_print(f"  {EMOJI_MAP['success']} Connected to {final_name}")
                    else:
                        log.warning(f"Failed connect MCP server: {name}")
                        self.safe_print(f"  {EMOJI_MAP['warning']} Failed connect {name}")
                except Exception as e:
                    log.error(f"Exception connecting MCP server {name}", exc_info=True)
                    self.safe_print(f"  {EMOJI_MAP['error']} Error connect {name}: {e}")
                    connection_results[name] = False

        # --- 10. Start Server Monitoring ---
        try:
            with Status(f"{EMOJI_MAP['server']} Starting server monitoring...", spinner="dots", console=safe_console) as status:
                await self.server_monitor.start_monitoring()
                status.update(f"{EMOJI_MAP['success']} Server monitoring started")
        except Exception as monitor_error:
            log.error("Failed start server monitor", exc_info=True)
            self.safe_print(f"[red]Error starting monitor: {monitor_error}[/red]")

        # --- 11. Display Final Status ---
        try:
            log.info("Displaying simple status at end of setup...")
            await self.print_simple_status()
            log.info("Simple status display complete.")
        except Exception as status_err:
            log.error(f"Error calling print_simple_status: {status_err}", exc_info=True)
            self.safe_print(f"[bold red]Error displaying final status: {status_err}[/bold red]")

    # --- Provider Determination Helper ---
    def get_provider_from_model(self, model_name: str) -> Optional[str]:
        """Determine the provider based on the model name using MODEL_PROVIDER_MAP."""
        if not model_name:
            log.warning("get_provider_from_model called with empty model name.")
            return None

        # 1. Direct Lookup (Case-insensitive check just in case)
        if model_name.lower() in map(str.lower, MODEL_PROVIDER_MAP.keys()):
            for k, v in MODEL_PROVIDER_MAP.items():
                if k.lower() == model_name.lower():
                    log.debug(f"Provider for '{model_name}' found via direct map: {v}")
                    return v

        # 2. Check Prefixes (e.g., "openai/gpt-4o", "anthropic:claude-3...")
        parts = model_name.split('/', 1)
        if len(parts) == 1:
            parts = model_name.split(':', 1)

        if len(parts) == 2:
            prefix = parts[0].lower()
            try:
                provider_enum = Provider(prefix)
                log.debug(f"Provider for '{model_name}' found via prefix: {provider_enum.value}")
                return provider_enum.value
            except ValueError:
                log.debug(f"Prefix '{prefix}' in '{model_name}' is not a known provider.")
                pass

        # 4. Fallback / No Match
        log.warning(f"Could not automatically determine provider for model: '{model_name}'. Ensure it's in MODEL_PROVIDER_MAP or has a known prefix.")
        return None

    def _extract_text_from_internal_content(self, content: Any) -> str:
        """Extracts and concatenates text from internal content format."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            return "\n".join(text_parts)
        elif content is None:
            return "" # Return empty string for None content
        else:
            # Fallback for unexpected content types
            log.warning(f"Unexpected content type in _extract_text: {type(content)}. Converting to string.")
            return str(content)

    # --- Main Formatting Function ---
    def _format_tools_for_provider(self, provider: str) -> Optional[List[Dict[str, Any]]]:
        """
        Formats the available MCP tools into the specific structure required by the target LLM provider's API.

        Handles sanitization of tool names and validation of input schemas according to provider requirements.

        Args:
            provider: The string identifier of the target provider (e.g., "openai", "anthropic").

        Returns:
            A list of dictionaries representing the tools in the provider's expected format,
            or None if no tools are available.
        """
        mcp_tools = list(self.server_manager.tools.values())
        if not mcp_tools:
            log.debug(f"No MCP tools found to format for provider '{provider}'.")
            return None # No tools to format

        formatted_tools: List[Dict[str, Any]] = []
        # Ensure the mapping is cleared before formatting a new set of tools for a call
        self.server_manager.sanitized_to_original.clear()
        log.debug(f"Cleared sanitized_to_original map. Formatting {len(mcp_tools)} tools for provider: {provider}")

        provider_enum_val = provider # Use the string value directly

        # --- Anthropic ---
        if provider_enum_val == Provider.ANTHROPIC.value:
            log.debug("Formatting tools for Anthropic.")
            for tool in sorted(mcp_tools, key=lambda t: t.name):
                original_name = tool.name
                # Anthropic names: ^[a-zA-Z0-9_-]{1,64}$
                sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name)[:64]
                if not sanitized_name:
                    log.warning(f"Skipping Anthropic tool '{original_name}' due to invalid sanitized name.")
                    continue

                self.server_manager.sanitized_to_original[sanitized_name] = original_name
                log.debug(f"Mapping Anthropic sanitized '{sanitized_name}' -> original '{original_name}'")

                # Basic schema validation (must be a dict)
                input_schema = tool.input_schema
                if not isinstance(input_schema, dict):
                    log.warning(f"Tool '{original_name}' for Anthropic has invalid schema type ({type(input_schema)}), expected dict. Sending empty schema.")
                    input_schema = {"type": "object", "properties": {}, "required": []} # Default empty

                formatted_tools.append({
                    "name": sanitized_name,
                    "description": tool.description or "No description provided.", # Ensure description exists
                    "input_schema": input_schema,
                })
            # Anthropic specific: Add cache_control to the last tool if any were added
            if formatted_tools:
                formatted_tools[-1]["cache_control"] = {"type": "ephemeral"}
                log.debug("Added ephemeral cache_control to the last Anthropic tool.")

        # --- OpenAI / Grok / DeepSeek / Groq / Gemini / Mistral / Cerebras (using OpenAI compatibility) ---
        elif provider_enum_val in [
            Provider.OPENAI.value,
            Provider.GROK.value,
            Provider.DEEPSEEK.value,
            Provider.GROQ.value,
            Provider.MISTRAL.value,
            Provider.CEREBRAS.value,
            Provider.GROK.value,
            Provider.GEMINI.value
        ]:
            log.debug(f"Formatting tools for OpenAI-compatible provider: {provider_enum_val}")
            for tool in sorted(mcp_tools, key=lambda t: t.name):
                original_name = tool.name
                # OpenAI names: ^[a-zA-Z0-9_-]{1,64}$
                sanitized_name = re.sub(r'[^a-zA-Z0-9_-]', '_', original_name)[:64]
                if not sanitized_name:
                    log.warning(f"Skipping OpenAI-like tool '{original_name}' due to invalid sanitized name.")
                    continue

                self.server_manager.sanitized_to_original[sanitized_name] = original_name
                log.debug(f"Mapping OpenAI-like sanitized '{sanitized_name}' -> original '{original_name}'")

                # Schema validation: Must be 'object' type for OpenAI functions
                input_schema = tool.input_schema
                validated_schema: Dict[str, Any]
                if isinstance(input_schema, dict) and input_schema.get("type") == "object":
                    # Ensure properties is a dict and required is a list
                    validated_schema = {
                        "type": "object",
                        "properties": input_schema.get("properties", {}),
                        "required": input_schema.get("required", [])
                    }
                    if not isinstance(validated_schema["properties"], dict):
                        log.warning(f"Tool '{original_name}' schema 'properties' is not a dict. Using empty dict.")
                        validated_schema["properties"] = {}
                    if not isinstance(validated_schema["required"], list):
                        log.warning(f"Tool '{original_name}' schema 'required' is not a list. Using empty list.")
                        validated_schema["required"] = []
                else:
                    log.warning(f"Tool '{original_name}' for {provider_enum_val} has invalid schema root type ({input_schema.get('type') if isinstance(input_schema, dict) else type(input_schema)}), expected 'object'. Using empty schema.")
                    validated_schema = {"type": "object", "properties": {}, "required": []} # Default empty

                formatted_tools.append({
                    "type": "function",
                    "function": {
                        "name": sanitized_name,
                        "description": tool.description or "No description provided.", # Ensure description exists
                        "parameters": validated_schema,
                    }
                })

        # --- Unknown Provider ---
        else:
            log.warning(f"Tool formatting not implemented or provider '{provider}' unknown. Returning no tools.")
            return None

        log.info(f"Formatted {len(formatted_tools)} tools for provider '{provider}'.")
        return formatted_tools if formatted_tools else None

    # --- Streaming Handlers (_handle_*_stream) ---
    async def _handle_anthropic_stream(self, stream: AsyncMessageStream) -> AsyncGenerator[Tuple[str, Any], None]:
        """Process Anthropic stream and emit standardized events. (Enhanced Error Handling)"""
        # (Keep variable initializations as before)
        current_text_block = None
        current_tool_use_block = None
        current_tool_input_json_accumulator = ""
        input_tokens = 0
        output_tokens = 0
        stop_reason = "unknown"

        try:
            async for event in stream:
                try: # Add inner try for processing each event
                    event_type = event.type
                    # --- Event Processing Logic (mostly unchanged) ---
                    if event_type == "message_start": input_tokens = event.message.usage.input_tokens
                    elif event_type == "content_block_start":
                        block_type = event.content_block.type
                        if block_type == "text": 
                            current_text_block = {"type": "text", "text": ""}
                        elif block_type == "tool_use": 
                            tool_id = event.content_block.id
                            tool_name = event.content_block.name
                            original_tool_name = self.server_manager.sanitized_to_original.get(tool_name, tool_name)
                            current_tool_use_block = {"id": tool_id, "name": original_tool_name}
                            current_tool_input_json_accumulator = ""
                            yield ("tool_call_start", {"id": tool_id, "name": original_tool_name})
                    elif event_type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            if current_text_block is not None: 
                                yield ("text_chunk", delta.text)
                        elif delta.type == "input_json_delta":
                            if current_tool_use_block is not None: 
                                current_tool_input_json_accumulator += delta.partial_json
                                yield ("tool_call_input_chunk", {"id": current_tool_use_block["id"], "json_chunk": delta.partial_json})
                    elif event_type == "content_block_stop":
                        if current_text_block is not None: current_text_block = None
                        elif current_tool_use_block is not None:
                            parsed_input = {}
                            try: 
                                parsed_input = json.loads(current_tool_input_json_accumulator) if current_tool_input_json_accumulator else {}
                            except json.JSONDecodeError as e: 
                                log.error(f"Anthropic JSON parse failed: {e}. Raw: '{current_tool_input_json_accumulator}'")
                                parsed_input = {"_tool_input_parse_error": f"Failed: {e}"}
                            yield ("tool_call_end", {"id": current_tool_use_block["id"], "parsed_input": parsed_input})
                            current_tool_use_block = None
                    elif event_type == "message_delta":
                        if hasattr(event.delta, 'stop_reason') and event.delta.stop_reason:
                            stop_reason = event.delta.stop_reason
                        if hasattr(event, 'usage') and event.usage:
                            output_tokens = event.usage.output_tokens
                    elif event_type == "message_stop": 
                        final_message = await stream.get_final_message()
                        stop_reason = final_message.stop_reason
                        output_tokens = final_message.usage.output_tokens
                        break # Exit loop on message_stop
                    elif event_type == "error": # Handle explicit stream error event from Anthropic
                        stream_error = getattr(event, 'error', {})
                        error_message = stream_error.get('message', 'Unknown stream error')
                        log.error(f"Anthropic stream reported error event: {error_message}")
                        yield ("error", f"Anthropic Stream Error: {error_message}")
                        stop_reason = "error"
                        break # Exit loop on explicit error
                except Exception as event_proc_err:
                     # Catch errors processing a specific event, log, yield error, and break
                     log.error(f"Error processing Anthropic stream event ({event.type}): {event_proc_err}", exc_info=True)
                     yield ("error", f"Error processing stream event: {event_proc_err}")
                     stop_reason = "error"
                     break # Exit loop on event processing error

        # --- Catch errors related to the stream iteration itself ---
        except anthropic.APIConnectionError as e: 
            log.error(f"Anthropic stream connection error: {e}")
            yield ("error", f"Anthropic Conn Error: {e}")
            stop_reason = "error"
        except anthropic.RateLimitError as e: 
            log.warning(f"Anthropic stream rate limit error: {e}")
            yield ("error", f"Anthropic Rate Limit: {e}")
            stop_reason = "rate_limit" # Use specific reason
        except anthropic.APIStatusError as e: 
            log.error(f"Anthropic stream API status error ({e.status_code}): {e}")
            yield ("error", f"Anthropic API Error ({e.status_code}): {e}")
            stop_reason = "error"
        except anthropic.APIError as e: 
            log.error(f"Anthropic stream generic API error: {e}")
            yield ("error", f"Anthropic API Error: {e}")
            stop_reason = "error"
        except Exception as e: 
            log.error(f"Unexpected error in Anthropic stream handler: {e}", exc_info=True)
            yield ("error", f"Unexpected stream error: {e}")
            stop_reason = "error"
        finally: 
            yield ("final_usage", {"input_tokens": input_tokens, "output_tokens": output_tokens})
            yield ("stop_reason", stop_reason)

    # _handle_openai_compatible_stream (No changes needed)
    async def _handle_openai_compatible_stream(self, stream: AsyncStream[ChatCompletionChunk], provider_name: str) -> AsyncGenerator[Tuple[str, Any], None]:
        # (Implementation from previous response is complete)
        current_tool_calls: Dict[int, Dict] = {}
        output_tokens = 0
        input_tokens = 0
        stop_reason = "stop"
        finish_reason = None
        try:
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice: 
                    continue
                delta = choice.delta
                finish_reason = choice.finish_reason
                if delta and delta.content: 
                    yield ("text_chunk", delta.content)
                if delta and delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if tc_chunk.id and tc_chunk.function and tc_chunk.function.name:
                            tool_id = tc_chunk.id
                            sanitized_name = tc_chunk.function.name
                            original_name = self.server_manager.sanitized_to_original.get(sanitized_name, sanitized_name)
                            current_tool_calls[idx] = {"id": tool_id, "name": original_name, "args_acc": ""}
                            yield ("tool_call_start", {"id": tool_id, "name": original_name})
                        if tc_chunk.function and tc_chunk.function.arguments:
                            args_chunk = tc_chunk.function.arguments
                            if idx in current_tool_calls: 
                                current_tool_calls[idx]["args_acc"] += args_chunk
                                yield ("tool_call_input_chunk", {"id": current_tool_calls[idx]["id"], "json_chunk": args_chunk})
                            else: 
                                log.warning(f"Args chunk unknown tool index {idx} from {provider_name}")
                if provider_name == Provider.GROQ.value and hasattr(chunk, 'x_groq') and chunk.x_groq and hasattr(chunk.x_groq, 'usage'):
                    usage = chunk.x_groq.usage
                    if usage: 
                        input_tokens = getattr(usage, 'prompt_tokens', input_tokens)
                        current_chunk_output = getattr(usage, 'completion_tokens', 0)
                        output_tokens = max(output_tokens, current_chunk_output) # Use max for cumulative
            for idx, tool_data in current_tool_calls.items():
                accumulated_args = tool_data["args_acc"]
                parsed_input = {}
                try:
                    if accumulated_args: parsed_input = json.loads(accumulated_args)
                except json.JSONDecodeError as e: 
                    log.error(f"{provider_name} JSON parse failed tool {tool_data['name']} (ID: {tool_data['id']}): {e}. Raw: '{accumulated_args}'")
                    parsed_input = {"_tool_input_parse_error": f"Failed: {e}"}
                yield ("tool_call_end", {"id": tool_data["id"], "parsed_input": parsed_input})
            stop_reason = finish_reason if finish_reason else "stop"
            if stop_reason == "tool_calls": stop_reason = "tool_use"
        except (OpenAIAPIError, OpenAIAPIConnectionError, OpenAIAuthenticationError) as e: 
            log.error(f"{provider_name} stream API error: {e}")
            yield ("error", f"{provider_name} API Error: {e}")
            stop_reason = "error"
        except Exception as e: 
            log.error(f"Unexpected error {provider_name} stream handler: {e}", exc_info=True)
            yield ("error", f"Unexpected stream processing error: {e}")
            stop_reason = "error"
        finally:
            if input_tokens == 0 or output_tokens == 0: 
                log.warning(f"{provider_name} stream no token counts. Estimate needed.")
            yield ("final_usage", {"input_tokens": input_tokens, "output_tokens": output_tokens})
            yield ("stop_reason", stop_reason)

    def _filter_faulty_client_tool_results(self, messages_in: InternalMessageList) -> InternalMessageList:
        """
        Filters the message history to remove pairs of (assistant tool_use request)
        and (user tool_result response) where the tool result indicates a known
        client-side JSON parsing failure before sending to the LLM.

        Args:
            messages_in: The current list of InternalMessage objects.

        Returns:
            A new list of InternalMessage objects with the faulty pairs removed.
        """
        messages_to_send: InternalMessageList = []
        # More specific signature if possible, otherwise keep the general one
        client_error_signature = "Client failed to parse JSON input"
        # Use the more specific error if available from the new code's error handling
        # client_error_signature = "Client JSON parse error for tool" # Example if error changed

        skipped_indices = set() # To track indices of messages to skip

        log.debug(f"Filtering history ({len(messages_in)} messages) for known client tool result parse errors...")

        # First pass: identify indices of faulty interactions to skip
        # Stores index of assistant message -> set of tool_use_ids from that message
        assistant_tool_uses_to_check: Dict[int, Set[str]] = {}

        for idx, msg in enumerate(messages_in):
            # Ensure msg and msg.content are not None before proceeding
            if not msg or not msg.content:
                continue

            if msg.role == "assistant":
                # Check if content is a list (it should be for tool_use)
                if isinstance(msg.content, list):
                    tool_use_ids: Set[str] = set()
                    for block in msg.content:
                        # Use isinstance for type checking with Pydantic models/objects
                        if isinstance(block, ToolUseContentBlock) and block.id:
                            tool_use_ids.add(block.id)

                    if tool_use_ids:
                        assistant_tool_uses_to_check[idx] = tool_use_ids

            elif msg.role == "user":
                # Check if this user message corresponds to a preceding assistant tool use
                prev_idx = idx - 1
                if prev_idx in assistant_tool_uses_to_check:
                    # Ensure content is a list (it should be for tool_result)
                    if isinstance(msg.content, list):
                        corresponding_ids = assistant_tool_uses_to_check[prev_idx]
                        found_faulty_result = False
                        for block in msg.content:
                            # Use isinstance for type checking
                            if isinstance(block, ToolResultContentBlock) and block.tool_use_id in corresponding_ids:
                                result_content = block.content # Access the content attribute
                                # Check if the content contains our specific client error signature.
                                # The content might be a string, or sometimes structured error dict.
                                # We need a robust check. Let's check if it's a string first.
                                error_found_in_content = False
                                if isinstance(result_content, str):
                                    if client_error_signature in result_content:
                                        error_found_in_content = True
                                # Optional: Add check if result_content is a dict containing the error
                                # elif isinstance(result_content, dict) and "error" in result_content:
                                #     if client_error_signature in str(result_content["error"]):
                                #          error_found_in_content = True

                                if error_found_in_content:
                                    found_faulty_result = True
                                    log.warning(f"Found faulty client tool result for tool_use_id {block.tool_use_id} "
                                                f"at history index {idx}. Marking preceding assistant request "
                                                f"(index {prev_idx}) and this user result for filtering.")
                                    break # Found one faulty result for this user message turn

                        if found_faulty_result:
                            # Mark both the assistant request and the user result for skipping
                            skipped_indices.add(prev_idx)
                            skipped_indices.add(idx)
                            # We found a faulty result for this user message, potentially remove
                            # the corresponding entry from assistant_tool_uses_to_check to avoid
                            # accidentally matching it again if structure allows (though unlikely here).
                            del assistant_tool_uses_to_check[prev_idx]
        # Second pass: build the filtered list
        for idx, msg in enumerate(messages_in):
            if idx not in skipped_indices:
                messages_to_send.append(msg)
            else:
                # More informative log about *what* is being skipped
                role = msg.role if msg else "UnknownRole"
                content_preview = repr(msg.content)[:50] + "..." if msg and msg.content else "NoContent"
                log.debug(f"Skipping message at index {idx} (Role: {role}, Content Preview: {content_preview}) due to client tool result parse error linkage.")
        if len(messages_in) != len(messages_to_send):
            log.info(f"Filtered {len(messages_in) - len(messages_to_send)} messages due to client tool result parse errors.")
        else:
            log.debug("No client tool result parse errors found requiring filtering.")
        # Now use 'messages_to_send' for the API call
        return messages_to_send
    
    async def auto_prune_context(self):
        """Auto-prune context based on token count if enabled."""
        if not self.use_auto_summarization:
            return
        try:
            token_count = await self.count_tokens()
            if token_count > self.config.auto_summarize_threshold:
                self.safe_print(f"[yellow]{EMOJI_MAP['warning']} Context size ({token_count:,} tokens) exceeds threshold "
                                f"({self.config.auto_summarize_threshold:,}). Auto-summarizing...[/]")
                # Use cmd_optimize which now calls the correct summarize_conversation
                # We pass the target token argument specifically
                await self.cmd_optimize(f"--tokens {self.config.max_summarized_tokens}")
        except Exception as e:
             log.error(f"Error during auto-pruning: {e}", exc_info=True)
             self.safe_print(f"[red]Error during automatic context pruning: {e}[/]")

    async def process_streaming_query(self, query: str, model: Optional[str] = None,
                                    max_tokens: Optional[int] = None) -> AsyncIterator[str]:
        """
        Process a query using the specified model and available tools with streaming.
        Handles multiple providers, tool use, status updates, and error handling.
        Yields text chunks for the LLM response and status messages prefixed with @@STATUS@@.
        """
        # --- 0. Initial Setup & Validation ---
        span: Optional[trace.Span] = None
        span_context_manager = None
        current_task = asyncio.current_task()
        stop_reason: Optional[str] = "processing" # Track the final reason the loop stops
        error_occurred = False # Flag to indicate if any error happened

        # Reset session stats for this new query
        self.session_input_tokens = 0
        self.session_output_tokens = 0
        self.session_total_cost = 0.0
        self.cache_hit_count = 0
        self.tokens_saved_by_cache = 0

        with safe_stdout():
            start_time = time.time()
            # Determine model and provider
            if not model: model = self.current_model
            if not max_tokens: max_tokens = self.config.default_max_tokens
            provider_name = self.get_provider_from_model(model)

            if not provider_name:
                error_msg = f"Could not determine provider for model '{model}'."
                log.error(error_msg)
                yield f"@@STATUS@@\n[bold red]Error: {error_msg}[/]"
                return

            # Get provider client
            provider_client = getattr(self, f"{provider_name}_client", None)
            if not provider_client and provider_name == Provider.ANTHROPIC.value:
                 provider_client = self.anthropic
            if not provider_client:
                error_msg = f"API key/client for provider '{provider_name}' not configured or initialized."
                log.error(error_msg)
                yield f"@@STATUS@@\n[bold red]Error: {error_msg}[/]"
                return

            log.info(f"Streaming query: Model='{model}', Provider='{provider_name}'")

            # Start OpenTelemetry Span
            if tracer:
                try:
                    span_context_manager = tracer.start_as_current_span(
                        "process_streaming_query",
                        attributes={
                            "llm.model_name": model, "llm.provider": provider_name,
                            "query_length": len(query), "streaming": True
                        }
                    )
                    if span_context_manager: span = span_context_manager.__enter__()
                except Exception as e:
                    log.warning(f"Failed to start trace span: {e}")
                    span = None
                    span_context_manager = None

            # Prepare initial conversation state
            await self.auto_prune_context()
            messages: InternalMessageList = self.conversation_graph.current_node.messages.copy()
            messages.append(InternalMessage(role="user", content=query))
            if span: span.set_attribute("conversation_length", len(messages))

            # Function-scope state
            final_response_text: str = ""
            servers_used: Set[str] = set()
            tools_used: List[str] = []
            tool_results_for_history: List[Dict] = []
            cache_hits_during_query: int = 0

        # --- 1. Main Interaction Loop (Handles Multi-Turn Tool Use) ---
        try:
            while True: # Loop handles potential multi-turn tool use
                if current_task.cancelled():
                    log.debug("Query cancelled before API turn")
                    raise asyncio.CancelledError("Query cancelled before API turn")

                # --- 1a. Reset State For This Turn ---
                accumulated_text_this_turn: str = ""
                tool_calls_in_progress: Dict[str, Dict] = {}
                completed_tool_calls: List[Dict] = []
                turn_input_tokens: int = 0
                turn_output_tokens: int = 0
                turn_cost: float = 0.0
                turn_stop_reason: Optional[str] = None
                turn_api_error: Optional[str] = None # Track API/Stream error for this turn

                # --- 1b. Format Inputs for Provider ---
                messages_to_send_this_turn = self._filter_faulty_client_tool_results(messages)
                formatted_messages, system_prompt, _ = self._format_messages_for_provider(
                    messages_to_send_this_turn, provider_name, model
                )
                formatted_tools = self._format_tools_for_provider(provider_name)
                log.debug(f"[{provider_name}] Turn Start: Msgs={len(formatted_messages)}, Tools={len(formatted_tools) if formatted_tools else 0}")

                # --- 1c. API Call and Stream Handling ---
                stream_handler: Optional[AsyncGenerator] = None
                api_stream: Optional[Any] = None
                stream_start_time = time.time()

                try:
                    # --- Try making the API call ---
                    log.debug(f"Initiating API call to {provider_name}...")
                    if provider_name == Provider.ANTHROPIC.value:
                        api_stream = await cast('AsyncAnthropic', provider_client).messages.stream(
                            model=model, messages=formatted_messages, system=system_prompt, # type: ignore
                            tools=formatted_tools, max_tokens=max_tokens, temperature=self.config.temperature
                        )
                        stream_handler = self._handle_anthropic_stream(api_stream)
                    elif provider_name in [Provider.OPENAI.value, Provider.GROK.value, Provider.DEEPSEEK.value, Provider.GROQ.value, Provider.MISTRAL.value, Provider.CEREBRAS.value, Provider.GEMINI.value]:
                        api_stream = await cast('AsyncOpenAI', provider_client).chat.completions.create(
                            model=model, messages=formatted_messages, tools=formatted_tools, # type: ignore
                            max_tokens=max_tokens, temperature=self.config.temperature, stream=True
                        )
                        stream_handler = self._handle_openai_compatible_stream(api_stream, provider_name)
                    else:
                        raise NotImplementedError(f"Streaming API call not implemented for provider: {provider_name}")

                    log.debug(f"API call successful for {provider_name}. Processing stream...")

                    # --- Process Standardized Stream Events ---
                    async for std_event_type, std_event_data in stream_handler:
                        if current_task.cancelled():
                            log.debug("Query cancelled during stream processing")
                            raise asyncio.CancelledError("Query cancelled during stream processing")

                        # Handle events...
                        if std_event_type == "error":
                            turn_api_error = str(std_event_data)
                            log.error(f"Stream error event from {provider_name} handler: {turn_api_error}")
                            yield f"@@STATUS@@\n[bold red]Stream Error ({provider_name}): {turn_api_error}[/]"
                            turn_stop_reason = "error"
                            break
                        elif std_event_type == "text_chunk":
                            if isinstance(std_event_data, str):
                                accumulated_text_this_turn += std_event_data
                                yield std_event_data
                        elif std_event_type == "tool_call_start":
                            tool_id = std_event_data.get('id', str(uuid.uuid4()))
                            original_tool_name = std_event_data.get('name', 'unknown_tool')
                            tool_calls_in_progress[tool_id] = {"name": original_tool_name, "args_acc": ""}
                            tool_short_name = original_tool_name.split(':')[-1] if ':' in original_tool_name else original_tool_name
                            yield f"@@STATUS@@\n{EMOJI_MAP['tool']} Preparing tool: [bold]{tool_short_name}[/] (ID: {tool_id[:8]})..."
                        elif std_event_type == "tool_call_input_chunk":
                            tool_id = std_event_data.get('id')
                            json_chunk = std_event_data.get('json_chunk')
                            if tool_id and json_chunk and tool_id in tool_calls_in_progress:
                                tool_calls_in_progress[tool_id]["args_acc"] += json_chunk
                            elif tool_id: log.warning(f"Input chunk for unknown tool call ID: {tool_id}")
                        elif std_event_type == "tool_call_end":
                            tool_id = std_event_data.get('id')
                            parsed_input = std_event_data.get('parsed_input', {})
                            if tool_id and tool_id in tool_calls_in_progress:
                                tool_info = tool_calls_in_progress.pop(tool_id)
                                completed_tool_calls.append({
                                    "id": tool_id, "name": tool_info["name"], "input": parsed_input
                                })
                                log.debug(f"Completed parsing tool call: ID={tool_id}, Name={tool_info['name']}")
                            elif tool_id: log.warning(f"End event for unknown tool call ID: {tool_id}")
                        elif std_event_type == "final_usage":
                            turn_input_tokens = std_event_data.get("input_tokens", 0)
                            turn_output_tokens = std_event_data.get("output_tokens", 0)
                            turn_cost = self._calculate_and_log_cost(model, turn_input_tokens, turn_output_tokens)
                            self.session_input_tokens += turn_input_tokens
                            self.session_output_tokens += turn_output_tokens
                            self.session_total_cost += turn_cost
                            yield (f"@@STATUS@@\n{EMOJI_MAP['token']} Turn Tokens: In={turn_input_tokens:,}, Out={turn_output_tokens:,} | "
                                   f"{EMOJI_MAP['cost']} Turn Cost: ${turn_cost:.4f}")
                            if span:
                                turn_idx = sum(1 for m in messages if m.get("role") == "assistant")
                                span.set_attribute(f"turn_{turn_idx}.input_tokens", turn_input_tokens)
                                span.set_attribute(f"turn_{turn_idx}.output_tokens", turn_output_tokens)
                                span.set_attribute(f"turn_{turn_idx}.cost", turn_cost)
                        elif std_event_type == "stop_reason":
                            turn_stop_reason = std_event_data
                            log.debug(f"Received stop reason: {turn_stop_reason}")
                        # else: log.warning(f"Unhandled standardized event type: {std_event_type}")

                # --- Catch API Call / Connection Errors ---
                except (anthropic.APIConnectionError, openai.APIConnectionError) as e:
                    turn_api_error = f"Connection Error: {e}"
                    log.error(f"{provider_name} {turn_api_error}", exc_info=True)
                except (anthropic.AuthenticationError, openai.AuthenticationError) as e:
                    turn_api_error = f"Authentication Error (Check API Key): {e}"
                    log.error(f"{provider_name} {turn_api_error}")
                except (anthropic.PermissionDeniedError, openai.PermissionDeniedError) as e:
                    turn_api_error = f"Permission Denied: {e}"
                    log.error(f"{provider_name} {turn_api_error}")
                except (anthropic.NotFoundError, openai.NotFoundError) as e:
                    turn_api_error = f"API Endpoint or Model Not Found: {e}"
                    log.error(f"{provider_name} {turn_api_error}")
                except (anthropic.RateLimitError, openai.RateLimitError) as e:
                    turn_api_error = f"Rate Limit Exceeded: {e}"
                    log.warning(f"{provider_name} {turn_api_error}")
                except (anthropic.BadRequestError, openai.BadRequestError) as e:
                    turn_api_error = f"Invalid Request / Bad Request: {e}"
                    log.error(f"{provider_name} {turn_api_error}", exc_info=True)
                except (anthropic.APIStatusError, openai.APIStatusError) as e:
                    turn_api_error = f"API Status Error ({e.status_code}): {e}"
                    log.error(f"{provider_name} {turn_api_error}", exc_info=True)
                except (anthropic.APIError, openai.APIError) as e: # Catch other provider base errors
                    turn_api_error = f"API Error: {e}"
                    log.error(f"{provider_name} {turn_api_error}", exc_info=True)
                except httpx.RequestError as e: # Catch general network errors
                    turn_api_error = f"Network Error: {e}"
                    log.error(f"{provider_name} Network Error: {turn_api_error}", exc_info=True)
                except asyncio.CancelledError: # Catch cancellation during API call/stream setup
                    log.debug(f"API call/stream setup cancelled for {provider_name}")
                    raise # Propagate cancellation immediately
                except NotImplementedError as e: # Catch our own error
                    turn_api_error = str(e)
                    log.error(turn_api_error)
                except Exception as api_err: # Catch unexpected errors
                    turn_api_error = f"Unexpected API/Stream Error: {api_err}"
                    log.error(f"Unexpected error during API call/stream for {provider_name}: {api_err}", exc_info=True)
                # --- End API/Stream Try Block ---
                finally:
                    if api_stream and hasattr(api_stream, 'aclose'):
                        await suppress(Exception)(api_stream.aclose())
                    log.debug(f"Stream processing finished for turn. Duration: {time.time() - stream_start_time:.2f}s. API Error: {turn_api_error}. Stop Reason: {turn_stop_reason}")

                # If an API or stream error occurred, report it and break the main loop
                if turn_api_error:
                    error_occurred = True
                    yield f"@@STATUS@@\n[bold red]Error ({provider_name}): {turn_api_error}[/]"
                    stop_reason = "error" # Set overall stop reason for the query
                    break # Exit the main while loop

                # --- 1e. Post-Stream Processing & Assistant Message Update ---
                assistant_content_blocks: List[Union[TextContentBlock, ToolUseContentBlock]] = []
                if accumulated_text_this_turn:
                    assistant_content_blocks.append(TextContentBlock(type="text", text=accumulated_text_this_turn))
                    final_response_text += accumulated_text_this_turn
                for tc in completed_tool_calls:
                    assistant_content_blocks.append(ToolUseContentBlock(type="tool_use", id=tc["id"], name=tc["name"], input=tc["input"]))
                if assistant_content_blocks:
                    messages.append(InternalMessage(role="assistant", content=assistant_content_blocks))

                # --- 1f. Handle Stop Reason ---
                if current_task.cancelled(): raise asyncio.CancelledError("Cancelled before stop reason handling")
                stop_reason = turn_stop_reason # Update the overall stop reason

                if stop_reason == "tool_use":
                    if not completed_tool_calls:
                        log.warning(f"{provider_name} stop reason 'tool_use' but no calls parsed.")
                        yield f"@@STATUS@@\n[yellow]Warning: Model requested tools, but none were identified.[/]"
                        break # Exit loop

                    tool_results_for_api: List[InternalMessage] = []
                    yield f"@@STATUS@@\n{EMOJI_MAP['tool']} Processing {len(completed_tool_calls)} tool call(s)..."

                    # --- Tool Execution Loop ---
                    try:
                        for tool_call in completed_tool_calls:
                            if current_task.cancelled(): raise asyncio.CancelledError("Cancelled during tool processing")

                            tool_use_id = tool_call["id"]
                            original_tool_name = tool_call["name"]
                            tool_args = tool_call["input"]
                            tool_short_name = original_tool_name.split(':')[-1] if ':' in original_tool_name else original_tool_name
                            tool_start_time = time.time()
                            tool_result_content: Union[str, List[Dict], Dict] = "Error: Tool execution failed unexpectedly."
                            log_content_for_history: Any = tool_result_content
                            is_error_flag = True
                            cache_used_flag = False

                            if isinstance(tool_args, dict) and "_tool_input_parse_error" in tool_args:
                                error_text = f"Client JSON parse error for '{original_tool_name}'."
                                tool_result_content = f"Error: {error_text}"
                                log_content_for_history = {"error": error_text, "raw_json": tool_args.get("_raw_json")}
                                yield f"@@STATUS@@\n{EMOJI_MAP['failure']} Input Error [bold]{tool_short_name}[/]: Client parse failed."
                                log.error(f"{error_text} Raw: {tool_args.get('_raw_json', 'N/A')}")
                            else:
                                mcp_tool_obj = self.server_manager.tools.get(original_tool_name)
                                if not mcp_tool_obj:
                                    error_text = f"Tool '{original_tool_name}' not found by client."
                                    tool_result_content = f"Error: {error_text}"
                                    log_content_for_history = {"error": error_text}
                                    yield f"@@STATUS@@\n{EMOJI_MAP['failure']} Tool Error: Tool '[bold]{original_tool_name}[/]' not found."
                                else:
                                    server_name = mcp_tool_obj.server_name
                                    servers_used.add(server_name)
                                    tools_used.append(original_tool_name)
                                    cached_result = None
                                    if self.tool_cache and self.config.enable_caching:
                                        try: cached_result = self.tool_cache.get(original_tool_name, tool_args)
                                        except TypeError: cached_result = None
                                        if cached_result is not None and not (isinstance(cached_result, str) and cached_result.startswith("Error:")):
                                            tool_result_content = cached_result
                                            log_content_for_history = cached_result
                                            is_error_flag = False
                                            cache_used_flag = True
                                            cache_hits_during_query += 1
                                            content_str_tokens = self._stringify_content(cached_result)
                                            cached_tokens = self._estimate_string_tokens(content_str_tokens)
                                            yield f"@@STATUS@@\n{EMOJI_MAP['cached']} Using cache [bold]{tool_short_name}[/] ({cached_tokens:,} tokens)"
                                            log.info(f"Using cached result for {original_tool_name}")
                                        elif cached_result is not None:
                                            log.info(f"Ignoring cached error for {original_tool_name}")
                                            cached_result = None

                                    if not cache_used_flag:
                                        yield f"@@STATUS@@\n{EMOJI_MAP['server']} Executing [bold]{tool_short_name}[/] via {server_name}..."
                                        log.info(f"Executing tool '{original_tool_name}' via server '{server_name}'...")
                                        try:
                                            if current_task.cancelled(): raise asyncio.CancelledError("Cancelled before tool execution")
                                            with safe_stdout():
                                                mcp_result: CallToolResult = await self.execute_tool(server_name, original_tool_name, tool_args)
                                            tool_latency = time.time() - tool_start_time
                                            if mcp_result.isError:
                                                error_detail = str(mcp_result.content) if mcp_result.content else "Unknown server error"
                                                tool_result_content = f"Error: Tool execution failed: {error_detail}"
                                                log_content_for_history = {"error": error_detail, "raw_content": mcp_result.content}
                                                yield f"@@STATUS@@\n{EMOJI_MAP['failure']} Error [bold]{tool_short_name}[/] ({tool_latency:.1f}s): {error_detail[:100]}..."
                                                log.warning(f"Tool '{original_tool_name}' failed on '{server_name}': {error_detail}")
                                            else:
                                                tool_result_content = mcp_result.content if mcp_result.content is not None else ""
                                                log_content_for_history = mcp_result.content
                                                is_error_flag = False
                                                content_str_tokens = self._stringify_content(tool_result_content)
                                                result_tokens = self._estimate_string_tokens(content_str_tokens)
                                                yield f"@@STATUS@@\n{EMOJI_MAP['success']} Result [bold]{tool_short_name}[/] ({result_tokens:,} tokens, {tool_latency:.1f}s)"
                                                log.info(f"Tool '{original_tool_name}' OK ({result_tokens:,} tokens, {tool_latency:.1f}s)")
                                                if self.tool_cache and self.config.enable_caching and not is_error_flag:
                                                    try: self.tool_cache.set(original_tool_name, tool_args, tool_result_content)
                                                    except TypeError: log.warning(f"Failed cache {original_tool_name}: unhashable args")
                                        except asyncio.CancelledError:
                                            log.debug(f"Tool execution cancelled: {original_tool_name}")
                                            tool_result_content = "Error: Tool execution cancelled by user."
                                            log_content_for_history = {"error": "Tool execution cancelled"}
                                            is_error_flag = True
                                            yield f"@@STATUS@@\n[yellow]Tool [bold]{tool_short_name}[/] aborted.[/]"
                                            raise
                                        except Exception as exec_err:
                                            tool_latency = time.time() - tool_start_time
                                            log.error(f"Client error during tool execution {original_tool_name}: {exec_err}", exc_info=True)
                                            error_text = f"Client error: {str(exec_err)}"
                                            tool_result_content = f"Error: {error_text}"
                                            log_content_for_history = {"error": error_text}
                                            yield f"@@STATUS@@\n{EMOJI_MAP['failure']} Client Error [bold]{tool_short_name}[/] ({tool_latency:.2f}s): {str(exec_err)}"

                            # Append result to message list for *next* turn
                            tool_results_for_api.append(InternalMessage(
                                role="user",
                                content=[ToolResultContentBlock(type="tool_result", tool_use_id=tool_use_id, content=tool_result_content, is_error=is_error_flag)]
                            ))
                            tool_results_for_history.append({
                                "tool_name": original_tool_name, "tool_use_id": tool_use_id, "content": log_content_for_history,
                                "is_error": is_error_flag, "cache_used": cache_used_flag
                            })
                            await asyncio.sleep(0.01) # Tiny sleep between tool results

                    # --- End Tool Execution Loop ---
                    except asyncio.CancelledError: raise # Propagate cancellation
                    except Exception as tool_loop_err: # Catch unexpected errors in the loop itself
                        log.error(f"Unexpected error during tool execution loop: {tool_loop_err}", exc_info=True)
                        yield f"@@STATUS@@\n[bold red]Error during tool processing: {tool_loop_err}[/]"
                        error_occurred = True
                        stop_reason = "error"
                        break # Mark error and break outer loop

                    # If tool loop finished without error/cancellation
                    messages.extend(tool_results_for_api)
                    log.info(f"Added {len(tool_results_for_api)} tool results. Continuing interaction loop.")
                    await asyncio.sleep(0.01)
                    continue # Continue main loop for next API call

                elif stop_reason == "error": # Break if stream handler or API call yielded error
                    log.error(f"Exiting interaction loop due to error during turn for {provider_name}.")
                    yield f"@@STATUS@@\n[bold red]Exiting due to turn error.[/]"
                    error_occurred = True
                    break
                else: # Normal finish (end_turn, max_tokens, etc.)
                    log.info(f"LLM interaction finished normally. Stop reason: {stop_reason}")
                    break # Exit the main loop

        # --- Handle Outer Loop Exceptions ---
        except asyncio.CancelledError:
            log.info("Query processing was cancelled.")
            yield f"@@STATUS@@\n[yellow]Request cancelled by user.[/]"
            if span: span.set_status(trace.StatusCode.ERROR, description="Query cancelled by user")
            stop_reason = "cancelled" # Set specific stop reason
            error_occurred = True # Treat cancellation as an error condition
            raise # Re-raise CancellationError
        except Exception as e:
            error_msg = f"Unexpected error during query processing loop: {str(e)}"
            log.error(error_msg, exc_info=True)
            if span:
                span.set_status(trace.StatusCode.ERROR, description=error_msg)
                if hasattr(span, 'record_exception'): span.record_exception(e) # Check if method exists
            yield f"@@STATUS@@\n[bold red]Error: {error_msg}[/]"
            stop_reason = "error"
            error_occurred = True

        # --- Final Updates ---
        finally:
            # Finalize OpenTelemetry Span
            if span:
                final_status_code = trace.StatusCode.OK
                final_desc = f"Query finished: {stop_reason or 'completed'}"
                if error_occurred or stop_reason == "error":
                    final_status_code = trace.StatusCode.ERROR
                    final_desc = f"Query failed or cancelled: {stop_reason or 'unknown error'}"
                elif stop_reason == "cancelled": # Handle explicit cancellation case
                    final_status_code = trace.StatusCode.ERROR
                    final_desc = "Query cancelled by user"

                span.set_status(final_status_code, description=final_desc)
                # Record final session totals on the main span
                span.set_attribute("total_input_tokens", self.session_input_tokens)
                span.set_attribute("total_output_tokens", self.session_output_tokens)
                span.set_attribute("total_estimated_cost", self.session_total_cost)
                span.set_attribute("cache_hits", cache_hits_during_query)
                span.add_event("query_processing_ended", {"final_stop_reason": stop_reason})

            if span_context_manager and hasattr(span_context_manager, '__exit__'):
                with suppress(Exception): span_context_manager.__exit__(*sys.exc_info())

            # Update Graph and History *only if not cancelled*
            if not (current_task and current_task.cancelled()):
                try:
                    self.conversation_graph.current_node.messages = messages # Save final state
                    self.conversation_graph.current_node.model = model # Record model used
                    await self.conversation_graph.save(str(self.conversation_graph_file))

                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    tokens_used_hist = self.session_input_tokens + self.session_output_tokens
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(self, 'history') and hasattr(self.history, 'add_async'):
                        await self.history.add_async(ChatHistory(
                            query=query, response=final_response_text, model=model, timestamp=timestamp,
                            server_names=list(servers_used), tools_used=tools_used,
                            conversation_id=self.conversation_graph.current_node.id, latency_ms=latency_ms,
                            tokens_used=tokens_used_hist, streamed=True, cached=(cache_hits_during_query > 0)
                        ))
                    else: 
                        log.warning("History object/method not found, cannot save history.")
                except Exception as final_update_err:
                    log.error(f"Error during final graph/history update: {final_update_err}", exc_info=True)

        log.info(f"Streaming query finished. Final Stop Reason: {stop_reason}. Total Latency: {(time.time() - start_time)*1000:.0f}ms")

    # --- Interactive Loop ---
    async def interactive_loop(self):
        """Runs the main interactive command loop."""
        # *** GET the console EXPLICITLY inside the loop ***
        interactive_console = get_safe_console() # Get it once here

        self.safe_print("\n[bold green]MCP Client Interactive Mode[/]")
        self.safe_print(f"Default Model: [cyan]{self.current_model}[/]. Type '/help' for commands.")
        self.safe_print("[italic dim]Press Ctrl+C once to abort request, twice quickly to exit[/italic dim]")

        # Define fixed layout dimensions
        RESPONSE_PANEL_HEIGHT = 35
        STATUS_PANEL_HEIGHT = 10 # Reduced status height

        while True:
            self._active_live_display = None
            self.current_query_task = None
            try:
                # Use Rich Prompt for better experience
                user_input = await asyncio.to_thread(
                    Prompt.ask, "\n[bold blue]>>> [/]", console=interactive_console, default=""
                )
                user_input = user_input.strip()

                if not user_input:
                    continue

                # --- Command Handling ---
                if user_input.startswith('/'):
                    # Use shlex for robust command parsing
                    try:
                        cmd_parts = shlex.split(user_input[1:])
                        if not cmd_parts: continue # Handle empty command "/"
                        cmd = cmd_parts[0].lower()
                        # Re-join remaining parts as args string, preserving quotes if needed
                        args = shlex.join(cmd_parts[1:]) if len(cmd_parts) > 1 else ""
                    except ValueError as e:
                         self.safe_print(f"[red]Error parsing command: {e}[/]")
                         continue

                    handler = self.commands.get(cmd)
                    if handler:
                        log.debug(f"Executing command: /{cmd} with args: '{args}'")
                        await handler(args) # Call the async command handler
                    else:
                        self.safe_print(f"[yellow]Unknown command: [bold]/{cmd}[/]. Type /help for list.[/yellow]")

                # --- Query Handling ---
                else:
                    # Reset session stats before processing query
                    self.session_input_tokens = 0
                    self.session_output_tokens = 0
                    self.session_total_cost = 0.0
                    self.cache_hit_count = 0
                    self.tokens_saved_by_cache = 0

                    query = user_input
                    status_lines = deque(maxlen=STATUS_PANEL_HEIGHT)
                    abort_message = Text("Press Ctrl+C once to abort...", style="dim yellow")
                    current_response_text = ""
                    query_error: Optional[Exception] = None
                    query_cancelled = False

                    # --- Setup Live Display ---
                    response_panel = Panel("", title="Response", height=RESPONSE_PANEL_HEIGHT, border_style="dim blue")
                    status_panel = Panel("", title="Status", height=STATUS_PANEL_HEIGHT + 2, border_style="dim blue")
                    abort_panel = Panel(abort_message, height=3, border_style="none")
                    main_group = Group(response_panel, status_panel, abort_panel)
                    live_panel = Panel(main_group, title=f"Querying {self.current_model}...", border_style="dim green")

                    self._active_live_display = Live(live_panel, console=interactive_console, refresh_per_second=12, transient=True, vertical_overflow="crop")

                    @contextmanager
                    def suppress_logs_during_live():
                        """Temporarily suppress logging below WARNING during Live display."""
                        logger = logging.getLogger("mcpclient_multi") # Get specific logger
                        original_level = logger.level
                        # Suppress INFO and DEBUG during live updates
                        if original_level < logging.WARNING:
                             logger.setLevel(logging.WARNING)
                        try: yield
                        finally: logger.setLevel(original_level) # Restore original level

                    with suppress_logs_during_live():
                        self._active_live_display.start()
                        try:
                            # --- Start Query Task ---
                            log.debug(f"Starting query task for: '{query[:50]}...'")
                            query_task = asyncio.create_task(
                                self._consume_streaming_query(query, status_lines),
                                name=f"query-{query[:20].replace(' ','_')}"
                            )
                            self.current_query_task = query_task

                            # --- Live Update Loop ---
                            last_update_time = time.monotonic()
                            while not query_task.done():
                                if query_task.cancelled():
                                    break # Check for early exit 
                                # Rate limit updates slightly
                                if time.monotonic() - last_update_time < (1.0 / 15.0): # Max 15fps
                                    await asyncio.sleep(0.01)
                                    continue
                                last_update_time = time.monotonic()

                                # Fetch latest data (updated by _consume_streaming_query)
                                current_response_text = getattr(self, "_current_query_text", "")
                                current_status_list = list(status_lines) # Get snapshot

                                # Build renderables
                                response_renderable = Markdown(current_response_text, code_theme="monokai") if current_response_text else Text("Waiting for response...", style="dim")
                                display_status = current_status_list[-STATUS_PANEL_HEIGHT:] # Get last N status lines
                                if len(display_status) < STATUS_PANEL_HEIGHT:
                                    padding = [Text("")] * (STATUS_PANEL_HEIGHT - len(display_status))
                                    display_status = padding + display_status
                                status_renderable = Group(*display_status)

                                # Update panels
                                response_panel.renderable = response_renderable
                                response_panel.border_style = "blue" if current_response_text else "dim blue"
                                status_panel.renderable = status_renderable
                                status_panel.border_style = "blue" if current_status_list else "dim blue"
                                abort_panel.renderable = abort_message # Always show while running

                                live_panel.title = f"Querying {self.current_model}..."
                                live_panel.border_style = "green"

                                self._active_live_display.update(live_panel)

                                # Efficiently wait for task completion or timeout
                                try: 
                                    await asyncio.wait_for(asyncio.shield(query_task), timeout=0.1)
                                except asyncio.TimeoutError: 
                                    pass # Expected timeout for refresh
                                except asyncio.CancelledError: 
                                    log.debug("Live update loop detected query cancellation.")
                                break # Exit update loop

                            # --- Query Task Finished ---
                            # Await potentially cancelled task to retrieve exception/result
                            try: 
                                await query_task
                            except asyncio.CancelledError: 
                                query_cancelled = True
                            except Exception as e: 
                                query_error = e

                        finally:
                            self._active_live_display.stop() # Stop live display first
                            self._active_live_display = None
                            self.current_query_task = None # Clear task reference

                    # --- Print Final Result Panel ---
                    final_response_text = getattr(self, "_current_query_text", "")
                    final_status_lines = list(status_lines)

                    final_response_renderable: Union[Markdown, Text]
                    final_title = f"Result ({self.current_model})"
                    final_border = "green"

                    if query_cancelled:
                        final_response_renderable = Markdown(final_response_text, code_theme="monokai") if final_response_text else Text("Query aborted by user.", style="yellow")
                        final_title += " - Cancelled"
                        final_border = "yellow"
                        final_status_lines.append(Text("[bold yellow]Request Aborted.[/]", style="yellow"))
                    elif query_error:
                        final_response_renderable = Text(f"Error: {query_error}", style="red")
                        final_title += " - Error"
                        final_border = "red"
                        final_status_lines.append(Text(f"[bold red]Error: {query_error}[/]", style="red"))
                    else:
                        final_response_renderable = Markdown(final_response_text, code_theme="monokai") if final_response_text else Text("[dim]No text content received.[/dim]", style="dim")

                    # Prepare final status display
                    display_status_final = final_status_lines[-STATUS_PANEL_HEIGHT:]
                    if len(display_status_final) < STATUS_PANEL_HEIGHT:
                        padding = [Text("")] * (STATUS_PANEL_HEIGHT - len(display_status_final))
                        display_status_final = padding + display_status_final
                    final_status_renderable = Group(*display_status_final)

                    # Build final panels
                    final_response_panel = Panel(final_response_renderable, title="Response", height=RESPONSE_PANEL_HEIGHT, border_style=final_border)
                    final_status_panel = Panel(final_status_renderable, title="Status", height=STATUS_PANEL_HEIGHT + 2, border_style=final_border)

                    # Assemble and print final output panel
                    final_output_panel = Panel(Group(final_response_panel, final_status_panel), title=final_title, border_style=final_border)
                    interactive_console.print(final_output_panel)

                    # Print final stats panel if not cancelled/errored significantly early
                    if not query_error: # Only show stats on normal completion or cancellation
                        self._print_final_query_stats(interactive_console)

            # --- Outer Loop Error Handling ---
            except KeyboardInterrupt:
                # This is caught when Prompt.ask is interrupted
                self.safe_print("\n[yellow]Input interrupted. Type /exit or Ctrl+C again to quit.[/yellow]")
                continue
            except EOFError: # Handle Ctrl+D
                 self.safe_print("\n[yellow]EOF received, exiting...[/]")
                 break # Exit the loop cleanly
            except Exception as loop_err:
                self.safe_print(f"\n[bold red]Unexpected Error in interactive loop:[/] {loop_err}")
                log.error("Unexpected error in interactive loop", exc_info=True)
                await asyncio.sleep(1) # Prevent rapid error loops

            finally:
                # Ensure Live display is stopped if loop breaks unexpectedly
                if self._active_live_display and self._active_live_display.is_started:
                    self._active_live_display.stop()
                self._active_live_display = None
                # Ensure current task is cleared if loop breaks
                if self.current_query_task:
                     self.current_query_task = None
                if hasattr(self, "_current_query_text"):
                     delattr(self, "_current_query_text")
    
    async def _consume_streaming_query(self, query: str, status_lines: deque):
        """Helper task to consume the stream and update shared state."""
        try:
            # Reset state before consuming
            self._current_query_text = ""
            self._current_status_messages = []

            # Iterate through the stream generator
            async for chunk in self.process_streaming_query(query):
                if chunk.startswith("@@STATUS@@"):
                    status_message = chunk[len("@@STATUS@@"):].strip()
                    self._current_status_messages.append(status_message)
                    status_lines.append(Text.from_markup(status_message)) # Keep deque updated
                else:
                    self._current_query_text += chunk # Append text chunks

                # Minimal sleep to yield control, Live update handles refresh
                await asyncio.sleep(0.001)

        except asyncio.CancelledError:
            log.debug("Query consumer task cancelled.")
            # Append cancellation status
            status_lines.append(Text.from_markup("[yellow]Query Aborted.[/yellow]"))
            self._current_status_messages.append("[yellow]Query Aborted.[/yellow]")
            # Re-raise so the main loop knows it was cancelled
            raise
        except Exception as e:
            log.error(f"Error consuming query stream: {e}", exc_info=True)
            # Append error status
            status_lines.append(Text.from_markup(f"[bold red]Error: {e}[/bold red]"))
            self._current_status_messages.append(f"[bold red]Error: {e}[/bold red]")
            # Re-raise the exception so the main loop knows about the error
            raise
        finally:
             log.debug("Query consumer task finished.")

    async def close(self):
        """Gracefully shut down the client and release resources."""
        log.info("Closing MCPClient...")
        # Stop monitoring first
        if hasattr(self, 'server_monitor'):
            await self.server_monitor.stop_monitoring()

        # Stop local discovery listener
        await self.stop_local_discovery_monitoring() # ADDED CALL

        # Close server manager (handles sessions/processes)
        if hasattr(self, 'server_manager'):
            await self.server_manager.close()

        # Close provider clients
        async def _safe_aclose(client):
            if client and hasattr(client, 'aclose'):
                try: await client.aclose()
                except Exception as e: log.warning(f"Error closing client {type(client).__name__}: {e}")

        await _safe_aclose(self.anthropic)
        await _safe_aclose(self.openai_client)
        await _safe_aclose(self.gemini_client)
        await _safe_aclose(self.grok_client)
        await _safe_aclose(self.deepseek_client)
        await _safe_aclose(self.mistral_client)
        await _safe_aclose(self.groq_client)
        await _safe_aclose(self.cerebras_client)
        # Add openrouter if needed

        # Close tool cache
        if hasattr(self, 'tool_cache') and hasattr(self.tool_cache, 'close'):
            self.tool_cache.close()

        # Cancel any lingering query task (should be handled elsewhere, but belt-and-suspenders)
        if self.current_query_task and not self.current_query_task.done():
            self.current_query_task.cancel()
            try:
                await self.current_query_task
            except asyncio.CancelledError:
                log.debug("Lingering query task successfully cancelled during close.")
            except Exception as e:
                log.warning(f"Error awaiting cancelled query task during close: {e}")

        log.info("MCPClient cleanup finished.")

    def _print_final_query_stats(self, target_console: Console):
        """Helper to format and print the final query statistics panel."""
        # Calculate hit rate for *this query*
        hit_rate = 0.0
        total_lookups = self.cache_hit_count + self.cache_miss_count
        if total_lookups > 0:
            hit_rate = (self.cache_hit_count / total_lookups) * 100

        # Calculate cost savings for *this query*
        cost_saved = 0.0
        if self.tokens_saved_by_cache > 0:
            model_cost_info = COST_PER_MILLION_TOKENS.get(self.current_model, {})
            input_cost_per_token = model_cost_info.get("input", 0) / 1_000_000
            cost_saved = self.tokens_saved_by_cache * input_cost_per_token * 0.9 # 90% saving

        # Assemble token stats text using the reset session variables
        token_stats_text = Text.assemble(
            "Tokens: ",
            ("Input: ", "dim cyan"), (f"{self.session_input_tokens:,}", "cyan"), " | ",
            ("Output: ", "dim magenta"), (f"{self.session_output_tokens:,}", "magenta"), " | ",
            ("Total: ", "dim white"), (f"{self.session_input_tokens + self.session_output_tokens:,}", "white"),
            " | ",
            ("Cost: ", "dim yellow"), (f"${self.session_total_cost:.4f}", "yellow")
        )

        cache_stats_text = Text()
        if total_lookups > 0 or self.tokens_saved_by_cache > 0:
            cache_stats_text = Text.assemble(
                "\nCache: ",
                ("Hits: ", "dim green"), (f"{self.cache_hit_count}", "green"), " | ",
                ("Misses: ", "dim yellow"), (f"{self.cache_miss_count}", "yellow"), " | ",
                ("Hit Rate: ", "dim green"), (f"{hit_rate:.1f}%", "green"), " | ",
                ("Tokens Saved: ", "dim green"), (f"{self.tokens_saved_by_cache:,}", "green bold"), " | ",
                ("Cost Saved: ", "dim green"), (f"${cost_saved:.4f}", "green bold")
            )

        # Combine and create panel
        stats_group = Group(token_stats_text, cache_stats_text)
        final_stats_panel = Panel(
            stats_group,
            title="Final Stats (This Query)",
            border_style="green",
            padding=(0, 1) # Less vertical padding
        )
        target_console.print(final_stats_panel)
            
    @app.command()
    def config(
        show: Annotated[bool, typer.Option("--show", "-s", help="Show current configuration")] = False,
        edit: Annotated[bool, typer.Option("--edit", "-e", help="Edit configuration YAML file in editor")] = False,
        reset: Annotated[bool, typer.Option("--reset", "-r", help="Reset configuration YAML to defaults (use with caution!)")] = False,
    ):
        """Manage client configuration (view, edit YAML, reset to defaults)."""
        # Run the config management function
        asyncio.run(config_async(show, edit, reset))

async def main_async(query, model, server, dashboard, interactive, webui_flag, webui_host, webui_port, serve_ui_file, cleanup_servers):
    """Main async entry point - Handles CLI, Interactive, Dashboard, and Web UI modes."""
    client = None # Initialize client to None
    safe_console = get_safe_console()
    max_shutdown_timeout = 10

    # --- Shared Setup ---
    try:
        log.info("Initializing MCPClient...")
        client = MCPClient() # Instantiation inside the try block
        # Pass interactive_mode=True if any mode *might* need it later (safer default)
        await client.setup(interactive_mode=(interactive or webui_flag or dashboard))

        if cleanup_servers:
            log.info("Cleanup flag detected. Testing and removing unreachable servers...")
            await client.cleanup_non_working_servers()
            log.info("Server cleanup process complete.")

        # --- Mode Selection and Execution ---
        if webui_flag:
            # --- Start Web UI Server ---
            log.info(f"Starting Web UI server on {webui_host}:{webui_port}")

            @asynccontextmanager
            async def lifespan(app: FastAPI):
                # --- Startup ---
                log.info("FastAPI lifespan startup: Setting up MCPClient...")
                # The client is already initialized and set up by the outer scope
                app.state.mcp_client = client # Make client accessible
                log.info("MCPClient setup complete for Web UI.")
                yield # Server runs here
                # --- Shutdown ---
                log.info("FastAPI lifespan shutdown: Closing MCPClient...")
                if app.state.mcp_client:
                    await app.state.mcp_client.close()
                log.info("MCPClient closed.")

            # Define FastAPI app within this scope
            app = FastAPI(title="Ultimate MCP Client API", lifespan=lifespan)
            global web_app # Allow modification of global var
            web_app = app # Assign to global var for uvicorn

            # Make client accessible to endpoints via dependency injection
            async def get_mcp_client(request: Request) -> MCPClient:
                if not hasattr(request.app.state, 'mcp_client') or request.app.state.mcp_client is None:
                    # This should ideally not happen due to lifespan
                    log.error("MCPClient not found in app state during request!")
                    raise HTTPException(status_code=500, detail="MCP Client not initialized")
                return request.app.state.mcp_client

            # --- CORS Middleware ---
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"], # Allow all for development, restrict in production
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            # --- API Endpoints ---
            log.info("Registering API endpoints...")

            @app.get("/api/status")
            async def get_status(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Returns basic status information about the client."""
                # Calculate disk cache entries safely
                disk_cache_entries = 0
                if mcp_client.tool_cache and mcp_client.tool_cache.disk_cache:
                    try:
                        # Use len(disk_cache) for potentially faster count if supported and safe
                        # For safety/compatibility, iterkeys is okay for moderate sizes
                        disk_cache_entries = sum(1 for _ in mcp_client.tool_cache.disk_cache.iterkeys())
                    except Exception as e:
                        log.warning(f"Could not count disk cache entries: {e}")

                status_data = {
                    "currentModel": mcp_client.current_model,
                    "connectedServersCount": len(mcp_client.server_manager.active_sessions),
                    "totalServers": len(mcp_client.config.servers),
                    "totalTools": len(mcp_client.server_manager.tools),
                    "totalResources": len(mcp_client.server_manager.resources),
                    "totalPrompts": len(mcp_client.server_manager.prompts),
                    "historyEntries": len(mcp_client.history.entries),
                    "cacheEntriesMemory": len(mcp_client.tool_cache.memory_cache) if mcp_client.tool_cache else 0,
                    "cacheEntriesDisk": disk_cache_entries,
                    "currentNodeId": mcp_client.conversation_graph.current_node.id,
                    "currentNodeName": mcp_client.conversation_graph.current_node.name,
                }
                return status_data
            
            @app.get("/api/config", response_model=ConfigGetResponse)
            async def get_config(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Returns the current non-sensitive client configuration."""
                try:
                    # Create dict from config state, excluding specific sensitive fields
                    config_dict = {}
                    skip_keys = {
                        "anthropic_api_key", "openai_api_key", "gemini_api_key",
                        "grok_api_key", "deepseek_api_key", "mistral_api_key",
                        "groq_api_key", "cerebras_api_key", "openrouter_api_key",
                        "servers", # Servers have their own endpoint
                        "decouple_instance", "dotenv_path" # Internal attributes
                    }
                    for key, value in mcp_client.config.__dict__.items():
                        if not key.startswith("_") and key not in skip_keys:
                            config_dict[key] = value

                    # Validate and return using the response model
                    # Pydantic handles alias mapping during serialization
                    return ConfigGetResponse(**config_dict)
                except Exception as e:
                    log.error(f"Error preparing GET /api/config response: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail="Internal server error retrieving configuration.") from e

            @app.put("/api/config")
            async def update_config(
                update_request: ConfigUpdateRequest,
                mcp_client: MCPClient = Depends(get_mcp_client)
            ):
                """
                Updates the **running** configuration based on the request.
                Note: Simple settings (keys, URLs, flags, etc.) are NOT persisted
                    to .env by this endpoint. Only 'cache_ttl_mapping' is saved
                    to the YAML configuration file.
                """
                updated_fields = False
                providers_to_reinit = set()
                fields_updated = []
                config_yaml_needs_save = False # Track if YAML needs saving

                log.debug(f"Received config update request via API: {update_request.model_dump(exclude_unset=True)}")

                for key_alias, value in update_request.model_dump(exclude_unset=True, by_alias=True).items():
                    # Find the corresponding attribute name in the Config class
                    # This handles the alias mapping back from camelCase to snake_case
                    attr_name = key_alias # Default if no specific mapping needed
                    # Check if it's a provider key or URL based on alias
                    found_provider_attr = False
                    for provider in Provider:
                        # Check API Keys
                        key_config_attr = PROVIDER_CONFIG_KEY_ATTR_MAP.get(provider.value)
                        if key_config_attr and key_alias == ConfigUpdateRequest.model_fields[key_config_attr].alias:
                            attr_name = key_config_attr
                            providers_to_reinit.add(provider.value)
                            found_provider_attr = True
                            break
                        # Check Base URLs
                        url_config_attr = PROVIDER_CONFIG_URL_ATTR_MAP.get(provider.value)
                        if url_config_attr and key_alias == ConfigUpdateRequest.model_fields[url_config_attr].alias:
                            attr_name = url_config_attr
                            providers_to_reinit.add(provider.value)
                            found_provider_attr = True
                            break

                    # Find matching attribute for general settings based on alias
                    if not found_provider_attr:
                        for config_attr, field_info in ConfigUpdateRequest.model_fields.items():
                            if field_info.alias == key_alias:
                                attr_name = config_attr
                                break

                    # Update the attribute on the config object if it exists
                    if hasattr(mcp_client.config, attr_name):
                        current_value = getattr(mcp_client.config, attr_name)
                        if current_value != value:
                            setattr(mcp_client.config, attr_name, value)
                            log.info(f"Config updated via API: {attr_name} = {value}")
                            updated_fields = True
                            fields_updated.append(attr_name)

                            # Handle side effects for specific attributes
                            if attr_name == 'default_model':
                                mcp_client.current_model = value
                            elif attr_name == 'history_size':
                                if isinstance(value, int) and value > 0:
                                    mcp_client.history = History(max_entries=value) # Recreate history
                                else:
                                    log.warning(f"Invalid history_size '{value}' ignored.")
                            elif attr_name == 'cache_ttl_mapping':
                                if isinstance(value, dict):
                                    # Update the cache instance as well
                                    if mcp_client.tool_cache:
                                        mcp_client.tool_cache.ttl_mapping = value.copy()
                                    config_yaml_needs_save = True # Mark YAML for saving
                                else:
                                    log.warning(f"Invalid type for cache_ttl_mapping '{type(value)}' ignored.")
                            # Add other side-effect handlers here if needed

                        else:
                            log.debug(f"API Config: Value for '{attr_name}' unchanged.")
                    else:
                        log.warning(f"Attribute '{attr_name}' (from alias '{key_alias}') not found in Config class, skipping update.")

                # Re-initialize provider clients if keys/URLs changed
                if providers_to_reinit:
                    log.info(f"Providers needing re-initialization due to config change: {providers_to_reinit}")
                    await mcp_client._reinitialize_provider_clients(list(providers_to_reinit))

                # Save the YAML part of the config *only* if relevant fields were updated
                if config_yaml_needs_save:
                    await mcp_client.config.save_async()
                    log.info("Saved updated cache_ttl_mapping to config.yaml.")

                if updated_fields:
                    return {"message": f"Configuration updated successfully for fields: {', '.join(fields_updated)} (Simple settings are session-only)"}
                else:
                    return {"message": "No configuration changes applied."}

            @app.get("/api/models")
            async def list_models_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Lists available models, grouped by provider, based on cost data and initialized clients."""
                models_by_provider = {}
                initialized_providers = set()

                # Check which providers have initialized clients
                if mcp_client.anthropic: initialized_providers.add(Provider.ANTHROPIC.value)
                if mcp_client.openai_client: initialized_providers.add(Provider.OPENAI.value)
                if mcp_client.gemini_client: initialized_providers.add(Provider.GEMINI.value)
                if mcp_client.grok_client: initialized_providers.add(Provider.GROK.value)
                if mcp_client.deepseek_client: initialized_providers.add(Provider.DEEPSEEK.value)
                if mcp_client.mistral_client: initialized_providers.add(Provider.MISTRAL.value)
                if mcp_client.groq_client: initialized_providers.add(Provider.GROQ.value)
                if mcp_client.cerebras_client: initialized_providers.add(Provider.CEREBRAS.value)
                # Add OpenRouter check if needed
                # if mcp_client.openrouter_client: initialized_providers.add(Provider.OPENROUTER.value)

                # Group models based on the static map and check if provider is initialized
                for model_name, provider_value in MODEL_PROVIDER_MAP.items():
                    if provider_value not in models_by_provider:
                        models_by_provider[provider_value] = []
                    cost_info = COST_PER_MILLION_TOKENS.get(model_name, {})
                    models_by_provider[provider_value].append({
                        "name": model_name,
                        "cost_input_per_million": cost_info.get("input"),
                        "cost_output_per_million": cost_info.get("output"),
                        "is_active": provider_value in initialized_providers # Indicate if provider client is ready
                    })

                # Sort models within each provider list
                for provider_list in models_by_provider.values():
                    provider_list.sort(key=lambda x: x["name"])

                # Return sorted providers
                return dict(sorted(models_by_provider.items()))
            
            @app.get("/api/servers")
            async def list_servers_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Lists all configured MCP servers with their status."""
                server_list = []
                for name, server in mcp_client.config.servers.items():
                    metrics = server.metrics
                    is_connected = name in mcp_client.server_manager.active_sessions
                    health_score = 0
                    if is_connected and metrics.request_count > 0:
                        # Adjusted health score logic - ensure positive result
                        health_penalty = (metrics.error_rate * 100) + max(0, (metrics.avg_response_time - 1.0) * 10)
                        health_score = max(0, min(100, int(100 - health_penalty)))

                    # Count tools for this server
                    tools_count = 0
                    for tool in mcp_client.server_manager.tools.values():
                        if tool.server_name == name:
                            tools_count += 1

                    server_data = {
                        "name": server.name,
                        "type": server.type.value,
                        "path": server.path,
                        "args": server.args,
                        "enabled": server.enabled,
                        "isConnected": is_connected,
                        "status": metrics.status.value,
                        "statusText": metrics.status.value.capitalize(),
                        "health": health_score,
                        "toolsCount": tools_count
                    }
                    server_list.append(server_data)

                # Sort alphabetically by name
                server_list.sort(key=lambda s: s['name'])
                return server_list

            @app.post("/api/servers", status_code=201)
            async def add_server_api(req: ServerAddRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Adds a new server configuration."""
                if req.name in mcp_client.config.servers:
                    raise HTTPException(status_code=409, detail=f"Server name '{req.name}' already exists")

                args_list = req.argsString.split() if req.argsString else []
                new_server_config = ServerConfig(
                    name=req.name, type=req.type, path=req.path, args=args_list,
                    enabled=True, auto_start=False,
                    description=f"Added via Web UI ({req.type.value})"
                )
                mcp_client.config.servers[req.name] = new_server_config
                await mcp_client.config.save_async() # Save YAML
                log.info(f"Added server '{req.name}' via API.")

                # Get full details of the newly added server to return
                server_details = mcp_client.get_server_details(req.name)
                if server_details is None:
                    # Should not happen, but handle defensively
                    raise HTTPException(status_code=500, detail="Failed to retrieve details for newly added server.")

                return {"message": f"Server '{req.name}' added.", "server": ServerDetail(**server_details).model_dump()}

            @app.delete("/api/servers/{server_name}")
            async def remove_server_api(server_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Removes a server configuration."""
                import urllib.parse
                decoded_server_name = urllib.parse.unquote(server_name)

                if decoded_server_name not in mcp_client.config.servers:
                    raise HTTPException(status_code=404, detail=f"Server '{decoded_server_name}' not found")

                if decoded_server_name in mcp_client.server_manager.active_sessions:
                    await mcp_client.server_manager.disconnect_server(decoded_server_name)

                del mcp_client.config.servers[decoded_server_name]
                await mcp_client.config.save_async() # Save YAML
                log.info(f"Removed server '{decoded_server_name}' via API.")
                return {"message": f"Server '{decoded_server_name}' removed"}

            @app.post("/api/servers/{server_name}/connect")
            async def connect_server_api(server_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Connects to a specific configured server."""
                import urllib.parse
                decoded_server_name = urllib.parse.unquote(server_name)

                if decoded_server_name not in mcp_client.config.servers:
                    raise HTTPException(status_code=404, detail=f"Server '{decoded_server_name}' not found")
                if decoded_server_name in mcp_client.server_manager.active_sessions:
                    return {"message": f"Server '{decoded_server_name}' already connected"}

                try:
                    server_config = mcp_client.config.servers[decoded_server_name]
                    session = await mcp_client.server_manager.connect_to_server(server_config)
                    if session:
                        final_name = server_config.name # Name might change during connect
                        log.info(f"Connected to server '{final_name}' via API.")
                        return {"message": f"Successfully connected to server '{final_name}'"}
                    else:
                        raise HTTPException(status_code=500, detail=f"Failed to connect to server '{decoded_server_name}' (check server logs)")
                except Exception as e:
                    log.error(f"API Error connecting to {decoded_server_name}: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Error connecting to server '{decoded_server_name}': {str(e)}") from e

            @app.post("/api/servers/{server_name}/disconnect")
            async def disconnect_server_api(server_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Disconnects from a specific connected server."""
                import urllib.parse
                decoded_server_name = urllib.parse.unquote(server_name)

                if decoded_server_name not in mcp_client.config.servers:
                    raise HTTPException(status_code=404, detail=f"Server '{decoded_server_name}' not found")
                if decoded_server_name not in mcp_client.server_manager.active_sessions:
                    return {"message": f"Server '{decoded_server_name}' is not connected"}

                await mcp_client.server_manager.disconnect_server(decoded_server_name)
                log.info(f"Disconnected from server '{decoded_server_name}' via API.")
                return {"message": f"Disconnected from server '{decoded_server_name}'"}

            @app.put("/api/servers/{server_name}/enable")
            async def enable_server_api(server_name: str, enabled: bool = True, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Enables or disables a server configuration."""
                import urllib.parse
                decoded_server_name = urllib.parse.unquote(server_name)

                if decoded_server_name not in mcp_client.config.servers:
                    raise HTTPException(status_code=404, detail=f"Server '{decoded_server_name}' not found")

                server_config = mcp_client.config.servers[decoded_server_name]
                if server_config.enabled == enabled:
                    action = "enabled" if enabled else "disabled"
                    return {"message": f"Server '{decoded_server_name}' is already {action}"}

                server_config.enabled = enabled
                await mcp_client.config.save_async() # Save YAML
                action_str = "enabled" if enabled else "disabled"
                log.info(f"Server '{decoded_server_name}' {action_str} via API.")

                if not enabled and decoded_server_name in mcp_client.server_manager.active_sessions:
                    await mcp_client.server_manager.disconnect_server(decoded_server_name)
                    log.info(f"Automatically disconnected disabled server '{decoded_server_name}'.")

                return {"message": f"Server '{decoded_server_name}' {action_str}"}

            @app.get("/api/servers/{server_name}/details", response_model=ServerDetail)
            async def get_server_details_api(server_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Gets detailed information about a specific configured server."""
                import urllib.parse
                decoded_server_name = urllib.parse.unquote(server_name)
                details = mcp_client.get_server_details(decoded_server_name)
                if details is None:
                    raise HTTPException(status_code=404, detail=f"Server '{decoded_server_name}' not found")
                try:
                    details_model = ServerDetail(**details)
                    return details_model
                except ValidationError as e:
                    log.error(f"Data validation error for server '{decoded_server_name}' details: {e}")
                    raise HTTPException(status_code=500, detail="Internal error retrieving server details.") from e

            @app.get("/api/tools")
            async def list_tools_api(
                server_name: Optional[str] = None, # Add optional query parameter
                mcp_client: MCPClient = Depends(get_mcp_client)
            ):
                """Lists available tools, optionally filtered by server_name."""
                tools_list = []
                # Sort tools by name before potential filtering
                sorted_tools = sorted(mcp_client.server_manager.tools.values(), key=lambda t: t.name)
                for tool in sorted_tools:
                    # Apply filter if provided
                    if server_name is None or tool.server_name == server_name:
                        tool_data = {
                            "name": tool.name, "description": tool.description,
                            "server_name": tool.server_name, "input_schema": tool.input_schema,
                            "call_count": tool.call_count, "avg_execution_time": tool.avg_execution_time,
                            "last_used": tool.last_used.isoformat() if isinstance(tool.last_used, datetime) else None,
                        }
                        tools_list.append(tool_data)
                return tools_list # Return the filtered list
            
            @app.get("/api/tools/{tool_name:path}/schema")
            async def get_tool_schema_api(tool_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Gets the input schema for a specific tool."""
                import urllib.parse
                decoded_tool_name = urllib.parse.unquote(tool_name)
                schema = mcp_client.get_tool_schema(decoded_tool_name)
                if schema is None:
                    raise HTTPException(status_code=404, detail=f"Tool '{decoded_tool_name}' not found")
                return schema

            @app.get("/api/resources")
            async def list_resources_api(
                server_name: Optional[str] = None, # Add optional query parameter
                mcp_client: MCPClient = Depends(get_mcp_client)
            ):
                """Lists available resources, optionally filtered by server_name."""
                resources_list = []
                sorted_resources = sorted(mcp_client.server_manager.resources.values(), key=lambda r: r.name)
                for resource in sorted_resources:
                    if server_name is None or resource.server_name == server_name:
                        resource_data = {
                            "name": resource.name, "description": resource.description,
                            "server_name": resource.server_name, "template": resource.template,
                            "call_count": resource.call_count,
                            "last_used": resource.last_used.isoformat() if isinstance(resource.last_used, datetime) else None,
                        }
                        resources_list.append(resource_data)
                return resources_list

            @app.get("/api/prompts")
            async def list_prompts_api(
                server_name: Optional[str] = None, # Add optional query parameter
                mcp_client: MCPClient = Depends(get_mcp_client)
            ):
                """Lists available prompts, optionally filtered by server_name."""
                prompts_list = []
                sorted_prompts = sorted(mcp_client.server_manager.prompts.values(), key=lambda p: p.name)
                for prompt in sorted_prompts:
                    if server_name is None or prompt.server_name == server_name:
                        prompt_data = {
                            "name": prompt.name, "description": prompt.description,
                            "server_name": prompt.server_name, "template": prompt.template,
                            "call_count": prompt.call_count,
                            "last_used": prompt.last_used.isoformat() if isinstance(prompt.last_used, datetime) else None,
                        }
                        prompts_list.append(prompt_data)
                return prompts_list
            
            @app.get("/api/prompts/{prompt_name:path}/template")
            async def get_prompt_template_api(prompt_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Gets the full template content for a specific prompt."""
                import urllib.parse
                decoded_prompt_name = urllib.parse.unquote(prompt_name)
                template = mcp_client.get_prompt_template(decoded_prompt_name)
                if template is None:
                    raise HTTPException(status_code=404, detail=f"Prompt '{decoded_prompt_name}' not found or has no template.")
                return {"template": template}

            @app.post("/api/tool/execute")
            async def execute_tool_api(req: ToolExecuteRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Executes a specified tool with the given parameters."""
                if req.tool_name not in mcp_client.server_manager.tools:
                    raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found")
                tool = mcp_client.server_manager.tools[req.tool_name]
                tool_short_name = req.tool_name.split(':')[-1]
                mcp_client.safe_print(f"{EMOJI_MAP['server']} API executing [bold]{tool_short_name}[/] via {tool.server_name}...")
                try:
                    start_time = time.time()
                    result : CallToolResult = await mcp_client.execute_tool(tool.server_name, req.tool_name, req.params)
                    latency = (time.time() - start_time) * 1000

                    # Prepare response content safely for JSON
                    content_to_return = None
                    if result.content is not None:
                        try:
                            # Attempt to serialize complex content, fallback to string
                            _ = json.dumps(result.content) # Test serialization
                            content_to_return = result.content
                        except TypeError:
                            content_to_return = str(result.content)
                            log.warning(f"Tool result content for {req.tool_name} not JSON serializable, sending as string.")

                    if result.isError:
                        error_text = str(content_to_return)[:150] + "..." if content_to_return else "Unknown Error"
                        mcp_client.safe_print(f"{EMOJI_MAP['failure']} API Tool Error [bold]{tool_short_name}[/] ({latency:.0f}ms): {error_text}")
                    else:
                        content_str = mcp_client._stringify_content(content_to_return)
                        result_tokens = mcp_client._estimate_string_tokens(content_str)
                        mcp_client.safe_print(f"{EMOJI_MAP['success']} API Tool Result [bold]{tool_short_name}[/] ({result_tokens:,} tokens, {latency:.0f}ms)")

                    return {"isError": result.isError, "content": content_to_return, "latency_ms": latency}

                except asyncio.CancelledError as e:
                    mcp_client.safe_print(f"[yellow]API Tool execution [bold]{tool_short_name}[/] cancelled.[/]")
                    raise HTTPException(status_code=499, detail="Tool execution cancelled by client") from e
                except Exception as e:
                    mcp_client.safe_print(f"{EMOJI_MAP['failure']} API Tool Execution Failed [bold]{tool_short_name}[/]: {str(e)}")
                    log.error(f"Error executing tool '{req.tool_name}' via API: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}") from e

            @app.get("/api/conversation")
            async def get_conversation_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Gets the current state of the conversation graph."""
                node = mcp_client.conversation_graph.current_node
                # Build node list iteratively
                nodes_data = []
                for n in mcp_client.conversation_graph.nodes.values():
                    nodes_data.append(n.to_dict())

                # Ensure messages are serializable
                try:
                    _ = json.dumps(node.messages)
                    _ = json.dumps(nodes_data)
                except TypeError as e:
                    log.error(f"Conversation data not serializable: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail="Internal error: Conversation state not serializable") from e

                return {
                    "currentNodeId": node.id,
                    "currentNodeName": node.name,
                    "messages": node.messages,
                    "model": node.model or mcp_client.config.default_model,
                    "nodes": nodes_data,
                }

            @app.post("/api/conversation/fork")
            async def fork_conversation_api(req: ForkRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Creates a new branch (fork) from the current conversation node."""
                try:
                    new_node = mcp_client.conversation_graph.create_fork(name=req.name)
                    mcp_client.conversation_graph.set_current_node(new_node.id)
                    await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                    log.info(f"Forked conversation via API. New node: {new_node.id} ({new_node.name})")
                    return {"message": "Fork created", "newNodeId": new_node.id, "newNodeName": new_node.name}
                except Exception as e:
                    log.error(f"Error forking conversation via API: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Error forking: {str(e)}") from e

            @app.post("/api/conversation/checkout")
            async def checkout_branch_api(req: CheckoutRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Switches the current conversation context to a specified node/branch."""
                node_id = req.node_id
                node = mcp_client.conversation_graph.get_node(node_id)
                if not node:
                    # Attempt partial match
                    matched_node = None
                    for n_id, n in mcp_client.conversation_graph.nodes.items():
                        if n_id.startswith(node_id):
                            if matched_node: # Ambiguous prefix
                                raise HTTPException(status_code=400, detail=f"Ambiguous node ID prefix '{node_id}'")
                            matched_node = n
                    node = matched_node

                if node and mcp_client.conversation_graph.set_current_node(node.id):
                    log.info(f"API checked out branch: {node.name} ({node.id})")
                    # Return messages of the newly checked-out node
                    return {"message": f"Switched to branch {node.name}", "currentNodeId": node.id, "messages": node.messages}
                else:
                    raise HTTPException(status_code=404, detail=f"Node ID '{node_id}' not found or switch failed")

            @app.post("/api/conversation/clear")
            async def clear_conversation_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Clears the messages of the current conversation node and switches to root."""
                current_node_id = mcp_client.conversation_graph.current_node.id
                mcp_client.conversation_graph.current_node.messages = []
                # Option: Reset current node to root after clearing, or stay on cleared node?
                # Let's stay on the current node, just clear its messages.
                # mcp_client.conversation_graph.set_current_node("root")
                await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                log.info(f"Cleared messages for node {current_node_id} via API.")
                cleared_node = mcp_client.conversation_graph.get_node(current_node_id) # Get potentially updated node
                return {"message": f"Messages cleared for node {cleared_node.name}", "currentNodeId": cleared_node.id, "messages": cleared_node.messages}


            @app.get("/api/conversation/graph", response_model=List[GraphNodeData])
            async def get_conversation_graph_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Returns a flat list of nodes structured for graph visualization."""
                graph_nodes = []
                for node_id, node_obj in mcp_client.conversation_graph.nodes.items():
                    node_data = {
                        "id": node_obj.id,
                        "name": node_obj.name,
                        "parentId": node_obj.parent.id if node_obj.parent else None,
                        "model": node_obj.model,
                        "createdAt": node_obj.created_at.isoformat(),
                        "modifiedAt": node_obj.modified_at.isoformat(),
                        "messageCount": len(node_obj.messages)
                    }
                    # Validate and potentially add using the Pydantic model
                    try:
                        graph_nodes.append(GraphNodeData(**node_data))
                    except ValidationError as e:
                        log.warning(f"Skipping invalid graph node data for node {node_id}: {e}")

                return graph_nodes # Return the list of validated Pydantic models

            @app.put("/api/conversation/nodes/{node_id}/rename")
            async def rename_conversation_node_api(
                node_id: str,
                req: NodeRenameRequest,
                mcp_client: MCPClient = Depends(get_mcp_client)
            ):
                """Renames a specific conversation node/branch."""
                node = mcp_client.conversation_graph.get_node(node_id)
                if not node:
                    raise HTTPException(status_code=404, detail=f"Node ID '{node_id}' not found.")
                if node.id == "root": # Prevent renaming root
                    raise HTTPException(status_code=400, detail="Cannot rename the root node.")

                old_name = node.name
                new_name = req.new_name.strip()
                if not new_name:
                    raise HTTPException(status_code=400, detail="New name cannot be empty.")

                if old_name == new_name:
                    return {"message": "Node name unchanged.", "node_id": node_id, "new_name": new_name}

                node.name = new_name
                node.modified_at = datetime.now() # Update modification time
                await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                log.info(f"Renamed conversation node '{node_id}' from '{old_name}' to '{new_name}' via API.")
                return {"message": f"Node '{node_id}' renamed successfully.", "node_id": node_id, "new_name": new_name}
            
            @app.get("/api/usage")
            async def get_token_usage(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Gets the token usage and cost statistics for the current session."""
                hit_rate = 0.0
                cache_hits = getattr(mcp_client, 'cache_hit_count', 0)
                cache_misses = getattr(mcp_client, 'cache_miss_count', 0)
                total_cache_lookups = cache_hits + cache_misses
                if total_cache_lookups > 0:
                    hit_rate = (cache_hits / total_cache_lookups) * 100

                cost_saved = 0.0
                tokens_saved = getattr(mcp_client, 'tokens_saved_by_cache', 0)
                if tokens_saved > 0:
                    model_cost_info = COST_PER_MILLION_TOKENS.get(mcp_client.current_model, {})
                    input_cost_per_token = model_cost_info.get("input", 0) / 1_000_000
                    cost_saved = tokens_saved * input_cost_per_token * 0.9 # 90% saving estimate

                usage_data = {
                    "input_tokens": getattr(mcp_client, 'session_input_tokens', 0),
                    "output_tokens": getattr(mcp_client, 'session_output_tokens', 0),
                    "total_tokens": getattr(mcp_client, 'session_input_tokens', 0) + getattr(mcp_client, 'session_output_tokens', 0),
                    "total_cost": getattr(mcp_client, 'session_total_cost', 0.0),
                    "cache_metrics": {
                        "hit_count": cache_hits,
                        "miss_count": cache_misses,
                        "hit_rate_percent": hit_rate,
                        "tokens_saved": tokens_saved,
                        "estimated_cost_saved": cost_saved
                    }
                }
                return usage_data

            @app.post("/api/conversation/optimize")
            async def optimize_conversation_api(req: OptimizeRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Optimizes the current conversation branch via summarization."""
                summarization_model = req.model or mcp_client.config.summarization_model
                target_length = req.target_tokens or mcp_client.config.max_summarized_tokens
                initial_tokens = await mcp_client.count_tokens()
                original_messages = mcp_client.conversation_graph.current_node.messages.copy()
                log.info(f"Starting conversation optimization. Initial tokens: {initial_tokens}. Target: ~{target_length}")

                try:
                    # Use a separate method to handle the summarization logic
                    summary = await mcp_client.summarize_conversation(
                        target_tokens=target_length,
                        model=summarization_model
                    )
                    if summary is None: # Check if summarization failed internally
                        raise RuntimeError("Summarization failed or returned no content.")

                    # Replace messages with summary
                    summary_system_message = f"The preceding conversation up to this point has been summarized:\n\n---\n{summary}\n---"
                    mcp_client.conversation_graph.current_node.messages = [
                        InternalMessage(role="system", content=summary_system_message)
                    ]
                    final_tokens = await mcp_client.count_tokens()
                    await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))

                    log.info(f"Optimization complete. Tokens: {initial_tokens} -> {final_tokens}")
                    return {"message": "Optimization complete", "initialTokens": initial_tokens, "finalTokens": final_tokens}

                except Exception as e:
                    log.error(f"Error optimizing conversation via API: {e}", exc_info=True)
                    mcp_client.conversation_graph.current_node.messages = original_messages # Restore on failure
                    raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}") from e

            @app.post("/api/discover/trigger", status_code=202)
            async def trigger_discovery_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Triggers a background task to discover MCP servers."""
                log.info("API triggered server discovery...")
                if mcp_client.server_manager._discovery_in_progress.locked():
                    return {"message": "Discovery process already running."}
                else:
                    asyncio.create_task(mcp_client.server_manager.discover_servers())
                    return {"message": "Server discovery process initiated."}

            @app.get("/api/discover/results", response_model=List[DiscoveredServer])
            async def get_discovery_results_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Returns the results from the last server discovery scan."""
                results = await mcp_client.server_manager.get_discovery_results()
                # Convert list of dicts to list of Pydantic models
                validated_results = []
                for item in results:
                    try:
                        validated_results.append(DiscoveredServer(**item))
                    except ValidationError as e:
                        log.warning(f"Skipping invalid discovered server data: {item}. Error: {e}")
                return validated_results

            @app.post("/api/discover/connect")
            async def connect_discovered_server_api(server_info: DiscoveredServer, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Adds a discovered server to the configuration and attempts to connect."""
                if not server_info.name or not server_info.path_or_url or not server_info.type:
                    raise HTTPException(status_code=400, detail="Incomplete server information provided.")

                info_dict = server_info.model_dump()
                success, message = await mcp_client.server_manager.add_and_connect_discovered_server(info_dict)

                if success:
                    return {"message": message}
                else:
                    status_code = 409 if "already configured" in message else 500
                    raise HTTPException(status_code=status_code, detail=message)

            @app.get("/api/conversation/{conversation_id}/export")
            async def export_conversation_api(conversation_id: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Exports a specific conversation branch as JSON data."""
                data = await mcp_client.get_conversation_export_data(conversation_id)
                if data is None:
                    raise HTTPException(status_code=404, detail=f"Conversation ID '{conversation_id}' not found")
                return data

            @app.post("/api/conversation/import")
            async def import_conversation_api(file: UploadFile = File(...), mcp_client: MCPClient = Depends(get_mcp_client)):
                """Imports a conversation from an uploaded JSON file."""
                if not file.filename or not file.filename.endswith(".json"):
                    raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JSON file.")
                try:
                    content_bytes = await file.read()
                    content_str = content_bytes.decode('utf-8')
                    data = json.loads(content_str)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid JSON content in uploaded file: {e}") from e
                except Exception as e:
                    log.error(f"Error reading uploaded file {file.filename}: {e}")
                    raise HTTPException(status_code=500, detail="Error reading uploaded file.") from e
                finally:
                    await file.close()

                success, message, new_node_id = await mcp_client.import_conversation_from_data(data)
                if success:
                    log.info(f"Conversation imported via API. New node: {new_node_id}")
                    return {"message": message, "newNodeId": new_node_id}
                else:
                    raise HTTPException(status_code=500, detail=f"Import failed: {message}")

            @app.get("/api/history/search", response_model=List[ChatHistoryResponse])
            async def search_history_api(
                q: str, # Query parameter 'q'
                limit: int = 10, # Optional limit parameter
                mcp_client: MCPClient = Depends(get_mcp_client)
            ):
                """Searches the conversation history."""
                if not q:
                    raise HTTPException(status_code=400, detail="Search query 'q' cannot be empty.")
                if limit <= 0:
                    limit = 10 # Default limit if invalid

                results = mcp_client.history.search(q, limit=limit)
                # Convert ChatHistory dataclass instances to dicts for Pydantic validation
                response_data = []
                for entry in results:
                    entry_dict = dataclasses.asdict(entry)
                    response_data.append(entry_dict)

                # Validate the list of dicts against the response model
                # This implicitly handles converting dataclasses to the response model format
                try:
                    validated_response = [ChatHistoryResponse(**item) for item in response_data]
                    return validated_response
                except ValidationError as e:
                    log.error(f"Error validating history search results: {e}")
                    raise HTTPException(status_code=500, detail="Internal error processing history search results.") from e

            @app.delete("/api/history")
            async def clear_history_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Clears the entire conversation history."""
                history_count = len(mcp_client.history.entries)
                if history_count == 0:
                    return {"message": "History is already empty."}

                mcp_client.history.entries.clear()
                # Save the now empty history
                await mcp_client.history.save()
                log.info(f"Cleared {history_count} history entries via API.")
                return {"message": f"Cleared {history_count} history entries successfully."}
            
            @app.get("/api/cache/statistics")
            async def get_cache_statistics_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Returns statistics about the tool cache and LLM prompt cache."""
                tool_cache_stats = {"memory_entries": 0, "disk_entries": 0}
                if mcp_client.tool_cache:
                    tool_cache_stats["memory_entries"] = len(mcp_client.tool_cache.memory_cache)
                    if mcp_client.tool_cache.disk_cache:
                        try:
                            tool_cache_stats["disk_entries"] = sum(1 for _ in mcp_client.tool_cache.disk_cache.iterkeys())
                        except Exception: pass # Ignore errors counting disk cache

                # Get LLM prompt cache stats (session-level)
                prompt_cache_stats = await get_token_usage(mcp_client) # Reuses the logic from /api/usage
                prompt_cache_stats = prompt_cache_stats.get("cache_metrics", {}) # Extract cache part

                return { "tool_cache": tool_cache_stats, "prompt_cache": prompt_cache_stats }

            @app.post("/api/cache/reset_stats")
            async def reset_cache_stats_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Resets the session-level prompt cache statistics."""
                # Explicitly check and reset attributes
                if hasattr(mcp_client, 'cache_hit_count'): mcp_client.cache_hit_count = 0
                if hasattr(mcp_client, 'cache_miss_count'): mcp_client.cache_miss_count = 0
                if hasattr(mcp_client, 'tokens_saved_by_cache'): mcp_client.tokens_saved_by_cache = 0
                log.info("Reset LLM prompt cache statistics via API.")
                return {"message": "LLM prompt cache statistics reset successfully"}

            @app.get("/api/cache/entries", response_model=List[CacheEntryDetail])
            async def get_cache_entries_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Lists entries currently in the tool cache."""
                if not mcp_client.tool_cache:
                    return [] # Return empty list if caching disabled
                entries_data = mcp_client.get_cache_entries()
                # Validate and convert using Pydantic model
                validated_entries = []
                for entry_dict in entries_data:
                    try:
                        validated_entries.append(CacheEntryDetail(**entry_dict))
                    except ValidationError as e:
                        log.warning(f"Skipping invalid cache entry data: {entry_dict}. Error: {e}")
                return validated_entries

            @app.delete("/api/cache/entries")
            async def clear_cache_all_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Clears all entries from the tool cache."""
                if not mcp_client.tool_cache:
                    raise HTTPException(status_code=404, detail="Tool caching is disabled")
                count = mcp_client.clear_cache()
                log.info(f"Cleared {count} tool cache entries via API.")
                return {"message": f"Cleared {count} tool cache entries."}

            @app.delete("/api/cache/entries/{tool_name:path}")
            async def clear_cache_tool_api(tool_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Clears tool cache entries for a specific tool name."""
                if not mcp_client.tool_cache:
                    raise HTTPException(status_code=404, detail="Tool caching is disabled")
                import urllib.parse
                decoded_tool_name = urllib.parse.unquote(tool_name)
                count = mcp_client.clear_cache(tool_name=decoded_tool_name)
                log.info(f"Cleared {count} tool cache entries for '{decoded_tool_name}' via API.")
                return {"message": f"Cleared {count} cache entries for tool '{decoded_tool_name}'."}

            @app.post("/api/cache/clean")
            async def clean_cache_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Removes expired entries from the tool cache."""
                if not mcp_client.tool_cache:
                    raise HTTPException(status_code=404, detail="Tool caching is disabled")
                count = mcp_client.clean_cache()
                log.info(f"Cleaned {count} expired tool cache entries via API.")
                return {"message": f"Cleaned {count} expired tool cache entries."}

            @app.get("/api/cache/dependencies", response_model=CacheDependencyInfo)
            async def get_cache_dependencies_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Gets the registered dependencies between tools for cache invalidation."""
                if not mcp_client.tool_cache:
                    # Return empty dependencies if caching is off
                    return CacheDependencyInfo(dependencies={})
                deps = mcp_client.get_cache_dependencies()
                return CacheDependencyInfo(dependencies=deps)

            @app.post("/api/runtime/reload")
            async def reload_servers_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Disconnects and reconnects to all enabled MCP servers."""
                try:
                    await mcp_client.reload_servers() # Assumes this method exists and works
                    return {"message": "Servers reloaded successfully."}
                except Exception as e:
                    log.error(f"Error reloading servers via API: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Server reload failed: {e}") from e

            @app.post("/api/conversation/apply_prompt")
            async def apply_prompt_api(req: ApplyPromptRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Applies a predefined prompt template to the current conversation."""
                success = await mcp_client.apply_prompt_to_conversation(req.prompt_name)
                if success:
                    log.info(f"Applied prompt '{req.prompt_name}' via API.")
                    # Return the updated message list for the UI
                    updated_messages = mcp_client.conversation_graph.current_node.messages
                    return {"message": f"Prompt '{req.prompt_name}' applied.", "messages": updated_messages}
                else:
                    raise HTTPException(status_code=404, detail=f"Prompt '{req.prompt_name}' not found.")

            @app.post("/api/config/reset")
            async def reset_config_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Resets the configuration (specifically the YAML part) to defaults."""
                try:
                    await mcp_client.reset_configuration()
                    log.info("Configuration reset to defaults via API.")
                    # Return the new default config state (non-sensitive parts)
                    new_config_state = await get_config(mcp_client) # Reuse the GET endpoint logic
                    return {"message": "Configuration reset to defaults (YAML file updated).", "config": new_config_state}
                except Exception as e:
                    log.error(f"Error resetting configuration via API: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Configuration reset failed: {e}") from e

            @app.post("/api/query/abort", status_code=200)
            async def abort_query_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Attempts to abort the currently running LLM query task."""
                log.info("Received API request to abort query.")
                task_to_cancel = mcp_client.current_query_task
                aborted = False
                message = "No active query found to abort."
                if task_to_cancel and not task_to_cancel.done():
                    try:
                        task_to_cancel.cancel()
                        log.info(f"Cancellation signal sent to task {task_to_cancel.get_name()}.")
                        # Give cancellation a moment to potentially complete
                        await asyncio.sleep(0.1)
                        if task_to_cancel.cancelled():
                            message = "Abort signal sent and task cancelled."
                            aborted = True
                        else:
                            message = "Abort signal sent. Task cancellation pending."
                            # Even if not confirmed cancelled yet, signal was sent
                            aborted = True # Consider it "aborted" from API perspective
                    except Exception as e:
                        log.error(f"Error trying to cancel task via API: {e}")
                        raise HTTPException(status_code=500, detail=f"Error sending abort signal: {str(e)}") from e
                else:
                    log.info("No active query task found to abort.")

                return {"message": message, "aborted": aborted}

            @app.get("/api/dashboard", response_model=DashboardData)
            async def get_dashboard_data_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Gets data suitable for populating a dashboard view."""
                data = mcp_client.get_dashboard_data()
                try:
                    # Validate against Pydantic model
                    dashboard_model = DashboardData(**data)
                    return dashboard_model
                except ValidationError as e:
                    log.error(f"Data validation error for dashboard data: {e}")
                    raise HTTPException(status_code=500, detail="Internal error generating dashboard data.") from e

            # --- WebSocket Chat Endpoint ---
            @app.websocket("/ws/chat")
            async def websocket_chat(websocket: WebSocket):
                """Handles real-time chat interactions via WebSocket."""
                try:
                    mcp_client: MCPClient = websocket.app.state.mcp_client
                except AttributeError:
                    log.error("app.state.mcp_client not available during WebSocket connection!")
                    await websocket.close(code=1011)
                    return

                await websocket.accept()
                connection_id = str(uuid.uuid4())[:8]
                log.info(f"WebSocket connection accepted (ID: {connection_id}).")
                active_query_task: Optional[asyncio.Task] = None # Track task specific to this WS connection

                async def send_ws_message(msg_type: str, payload: Any):
                    try:
                        await websocket.send_json(WebSocketMessage(type=msg_type, payload=payload).model_dump())
                    except Exception as send_err:
                        log.warning(f"WS-{connection_id}: Failed send (Type: {msg_type}): {send_err}")

                async def send_command_response(success: bool, message: str, data: Optional[Dict] = None):
                    payload = {"success": success, "message": message}
                    payload.update(data or {})
                    await send_ws_message("command_response", payload)

                async def send_error_response(message: str, cmd: Optional[str] = None):
                    log_msg = f"WS-{connection_id} Error: {message}" + (f" (Cmd: /{cmd})" if cmd else "")
                    log.warning(log_msg)
                    await send_ws_message("error", {"message": message})
                    if cmd is not None: await send_command_response(False, message)

                try:
                    while True:
                        raw_data = await websocket.receive_text()
                        try:
                            data = json.loads(raw_data)
                            message = WebSocketMessage(**data)
                            log.debug(f"WS-{connection_id} Received: Type={message.type}, Payload={str(message.payload)[:100]}...")

                            # --- Handle Query ---
                            if message.type == "query":
                                query_text = str(message.payload or "").strip()
                                if not query_text: continue
                                if active_query_task and not active_query_task.done():
                                    await send_error_response("Previous query still running.")
                                    continue

                                mcp_client.session_input_tokens = 0
                                mcp_client.session_output_tokens = 0
                                mcp_client.session_total_cost = 0.0
                                mcp_client.cache_hit_count = 0
                                mcp_client.tokens_saved_by_cache = 0

                                # Create and track the task
                                query_task = asyncio.create_task(mcp_client.process_streaming_query(query_text))
                                active_query_task = query_task
                                # Link to client instance ONLY if no other task is running (for global abort)
                                if not mcp_client.current_query_task or mcp_client.current_query_task.done():
                                    mcp_client.current_query_task = query_task
                                else:
                                    log.warning("Another query task is already running globally, global abort might affect wrong task.")

                                try:
                                    async for chunk in query_task:
                                        if chunk.startswith("@@STATUS@@"):
                                            status_payload = chunk[len("@@STATUS@@"):].strip()
                                            # Optionally log status to console: mcp_client.safe_print(status_payload)
                                            await send_ws_message("status", status_payload)
                                        else:
                                            await send_ws_message("text_chunk", chunk)
                                    await send_ws_message("query_complete", None)
                                    usage_data = await get_token_usage(mcp_client)
                                    await send_ws_message("token_usage", usage_data)

                                except asyncio.CancelledError:
                                    log.info(f"WS-{connection_id}: Query cancelled.")
                                    await send_ws_message("status", "[yellow]Request Aborted by User.[/]")
                                except Exception as e:
                                    error_msg = f"Error processing query: {str(e)}"
                                    log.error(f"WS-{connection_id}: Error processing query: {e}", exc_info=True)
                                    await send_ws_message("error", {"message": error_msg})
                                finally:
                                    active_query_task = None # Clear task for this WS connection
                                    # Clear global task reference ONLY if it was THIS task
                                    if mcp_client.current_query_task == query_task:
                                        mcp_client.current_query_task = None

                            # --- Handle Command ---
                            elif message.type == "command":
                                command_str = str(message.payload).strip()
                                if command_str.startswith('/'):
                                    parts = command_str[1:].split(maxsplit=1)
                                    cmd = parts[0].lower()
                                    args = parts[1] if len(parts) > 1 else ""
                                    log.info(f"WS-{connection_id} processing command: /{cmd} {args}")
                                    try:
                                        if cmd == "clear":
                                            mcp_client.conversation_graph.current_node.messages = []
                                            mcp_client.conversation_graph.set_current_node("root")
                                            await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                                            await send_command_response(True, "Conversation cleared.", {"messages": []}) # Send cleared messages
                                        elif cmd == "model":
                                            if args:
                                                mcp_client.current_model = args
                                                mcp_client.config.default_model = args
                                                asyncio.create_task(mcp_client.config.save_async()) # Persist default model change
                                                await send_command_response(True, f"Model set to: {args}", {"currentModel": args})
                                            else: await send_command_response(True, f"Current model: {mcp_client.current_model}", {"currentModel": mcp_client.current_model})
                                        elif cmd == "fork":
                                            new_node = mcp_client.conversation_graph.create_fork(name=args if args else None)
                                            mcp_client.conversation_graph.set_current_node(new_node.id)
                                            await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                                            await send_command_response(True, f"Created branch: {new_node.name}", {"newNodeId": new_node.id, "newNodeName": new_node.name})
                                        elif cmd == "checkout":
                                            if not args: 
                                                await send_error_response("Usage: /checkout NODE_ID_or_Prefix", cmd) 
                                                continue
                                            node_id=args
                                            node=mcp_client.conversation_graph.get_node(node_id)
                                            if not node: 
                                                matched_node=None
                                                [matched_node := n for n_id, n in mcp_client.conversation_graph.nodes.items() if n_id.startswith(node_id) if not matched_node]
                                                node=matched_node
                                            if node and mcp_client.conversation_graph.set_current_node(node.id):
                                                await send_command_response(True, f"Switched to branch: {node.name}", {"currentNodeId": node.id, "messages": node.messages}) # Send messages of new node
                                            else: 
                                                await send_error_response(f"Node ID '{node_id}' not found.", cmd)
                                        elif cmd == "apply_prompt": # Renamed from 'prompt' for clarity
                                            if not args: 
                                                await send_error_response("Usage: /apply_prompt <prompt_name>", cmd)
                                                continue
                                            success = await mcp_client.apply_prompt_to_conversation(args)
                                            if success:
                                                await send_command_response(True, f"Applied prompt: {args}", {"messages": mcp_client.conversation_graph.current_node.messages})
                                            else: 
                                                await send_error_response(f"Prompt not found: {args}", cmd)
                                        # --- Add Abort Command Handler ---
                                        elif cmd == "abort":
                                            log.info(f"WS-{connection_id} received abort command.")
                                            if active_query_task and not active_query_task.done():
                                                active_query_task.cancel()
                                                await send_command_response(True, "Abort signal sent to running query.")
                                                # Status update will be sent when CancelledError is caught
                                            else:
                                                await send_command_response(False, "No active query running for this connection.")
                                        else: await send_command_response(False, f"Command '/{cmd}' not supported via WebSocket.")
                                    except Exception as cmd_err: 
                                        await send_error_response(f"Error executing '/{cmd}': {cmd_err}", cmd)
                                        log.error(f"WS-{connection_id} Cmd Error /{cmd}: {cmd_err}", exc_info=True)
                                else: 
                                    await send_error_response("Invalid command format.", None)

                            # Ignore other message types for now
                            # elif message.type == "other_type": ...

                        except (json.JSONDecodeError, ValidationError) as e: 
                            log.warning(f"WS-{connection_id} invalid message: {raw_data[:100]}... Error: {e}")
                            await send_ws_message("error", {"message": "Invalid message format."})
                        except WebSocketDisconnect: raise
                        except Exception as e: 
                            log.error(f"WS-{connection_id} error processing message: {e}", exc_info=True)
                            await suppress(Exception)(send_ws_message("error", {"message": f"Internal error: {str(e)}"}))

                except WebSocketDisconnect: 
                    log.info(f"WebSocket connection closed (ID: {connection_id}).")
                except Exception as e: 
                    log.error(f"WS-{connection_id} unexpected handler error: {e}", exc_info=True)
                    await suppress(Exception)(websocket.close(code=1011))
                finally: # Ensure task is cancelled if WS connection closes unexpectedly
                    if active_query_task and not active_query_task.done():
                        log.warning(f"WS-{connection_id} closing, cancelling active query task.")
                        active_query_task.cancel()
                    # Clear global reference if this was the task
                    if mcp_client.current_query_task == active_query_task:
                        mcp_client.current_query_task = None

            # --- Static File Serving ---
            if serve_ui_file:
                ui_file = Path(__file__).parent / "mcp_client_ui.html" # Look relative to script
                if ui_file.exists():
                    log.info(f"Serving static UI file from {ui_file.resolve()}")
                    @app.get("/", response_class=FileResponse, include_in_schema=False)
                    async def serve_html(): 
                        return FileResponse(str(ui_file.resolve()))
                else: 
                    log.warning(f"UI file {ui_file} not found. Cannot serve.")
            
            log.info("Starting Uvicorn server...")
            config = uvicorn.Config(app, host=webui_host, port=webui_port, log_level="info")
            server_instance = uvicorn.Server(config)
            await server_instance.serve()
            log.info("Web UI server shut down.")
            # NOTE: Cleanup for webui is handled by lifespan, so return here
        elif dashboard:
            # --- Run Dashboard ---
            if not client.server_monitor.monitoring:
                 await client.server_monitor.start_monitoring()
            await client.cmd_dashboard("")
            # No explicit return needed, finally block handles cleanup
        elif query:
            # --- Run Single Query ---
            # ... (Single query logic as before) ...
            query_text = query
            model_to_use = model or client.current_model
            provider_name = client.get_provider_from_model(model_to_use)
            provider_title = provider_name.capitalize() if provider_name else "Unknown Provider"

            safe_console.print(f"[cyan]Processing single query with {provider_title} ({model_to_use})...[/]")
            final_result_text = ""
            query_error: Optional[Exception] = None
            query_cancelled = False
            status_updates_internal: List[str] = []
            client.session_input_tokens = 0; client.session_output_tokens = 0; client.session_total_cost = 0.0; client.cache_hit_count = 0; client.tokens_saved_by_cache = 0 # Reset stats

            with Status(f"{EMOJI_MAP['processing']} Processing query with {model_to_use}...", console=safe_console, spinner="dots") as status:
                try:
                    query_task = asyncio.create_task(client.process_streaming_query(query_text, model=model_to_use), name=f"single_query-{query_text[:20].replace(' ','_')}")
                    client.current_query_task = query_task
                    async for chunk in query_task:
                        if chunk.startswith("@@STATUS@@"):
                            status_updates_internal.append(chunk[len("@@STATUS@@"):].strip())
                        else: final_result_text += chunk
                    await query_task # Ensure completion/catch errors
                except asyncio.CancelledError: query_cancelled = True; log.info("Single query processing cancelled.")
                except Exception as e: query_error = e; log.error(f"Error processing single query: {e}", exc_info=True)
                finally: status.stop(); client.current_query_task = None

            safe_console.print() # Blank line
            if query_cancelled:
                if final_result_text: safe_console.print(Panel.fit(Markdown(final_result_text), title=f"Partial Result ({model_to_use}) - Cancelled", border_style="yellow"))
                else: safe_console.print("[yellow]Query cancelled, no result generated.[/yellow]")
            elif query_error:
                safe_console.print(f"[bold red]Error processing query:[/] {str(query_error)}")
                if status_updates_internal:
                    safe_console.print("\n[dim]Status updates during failed query:[/dim]")
                    for line in status_updates_internal: safe_console.print(f"[dim]- {line}[/dim]")
            elif not final_result_text: safe_console.print(Panel.fit("[dim]Model returned no text content.[/dim]", title=f"Result ({model_to_use}) - Empty", border_style="yellow"))
            else: safe_console.print(Panel.fit(Markdown(final_result_text), title=f"Result ({model_to_use})", border_style="green"))

            if not query_error: client._print_final_query_stats(safe_console)
            # No explicit return needed, finally block handles cleanup

        elif interactive: # Check interactive explicitly last
             # --- Run Interactive Mode ---
             await client.interactive_loop()
             # No explicit return needed, finally block handles cleanup

        # --- Specific Server Connection (if requested via --server) ---
        # This should run REGARDLESS of the mode, as it's a startup connection request
        if server: # Check if the --server argument was provided
            log.info(f"Attempting specific connection to server '{server}' due to --server flag.")
            if server in client.config.servers:
                if server not in client.server_manager.active_sessions:
                    safe_console.print(f"[cyan]Attempting to connect to specific server: {server}...[/]")
                    connected = await client.connect_server(server) # Use existing method
                    if not connected:
                        safe_console.print(f"[yellow]Warning: Failed to connect to specified server '{server}'.[/]")
                else:
                    log.info(f"Specified server '{server}' is already connected.")
            else:
                safe_console.print(f"[red]Error: Specified server '{server}' not found in configuration.[/]")
    except KeyboardInterrupt:
        safe_console.print("\n[yellow]Interrupted, shutting down...[/]")
    except Exception as main_async_error:
        safe_console.print(f"[bold red]An unexpected error occurred in the main process: {main_async_error}[/]")
        print(f"\n--- Traceback for Main Process Error ({type(main_async_error).__name__}) ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("--- End Traceback ---", file=sys.stderr)
        # Exit only for non-interactive, non-webui modes on error
        if not interactive and not webui_flag:
             if client and hasattr(client, 'close'):
                 try: await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout / 2)
                 except Exception: pass
             sys.exit(1) # Exit with error code
    finally:
        # Cleanup logic (Handles closing client *unless* webui is running, as lifespan handles it)
        if not webui_flag and client and hasattr(client, 'close'):
            log.info("Performing final cleanup...")
            try:
                await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout)
            except asyncio.TimeoutError:
                safe_console.print("[red]Shutdown timed out. Some processes may still be running.[/]")
                # ... force kill logic ...
                if hasattr(client, 'server_manager') and hasattr(client.server_manager, 'processes'):
                     for name, process in client.server_manager.processes.items():
                         if process and process.returncode is None:
                             try:
                                 safe_console.print(f"[yellow]Force killing process: {name}[/]")
                                 process.kill()
                             except Exception: pass
            except Exception as close_error:
                log.error(f"Error during final cleanup: {close_error}", exc_info=True)
        log.info("Application shutdown complete.")

async def servers_async(search, list_all, json_output):
    """Server management async function"""
    client = MCPClient()
    safe_console = get_safe_console()
    
    try:
        if search:
            # Discover servers
            with Status("[cyan]Searching for servers...[/]", console=safe_console):
                await client.server_manager.discover_servers()
        
        if list_all or not search:
            # List servers
            if json_output:
                # Output as JSON
                server_data = {}
                for name, server in client.config.servers.items():
                    server_data[name] = {
                        "type": server.type.value,
                        "path": server.path,
                        "enabled": server.enabled,
                        "auto_start": server.auto_start,
                        "description": server.description,
                        "categories": server.categories
                    }
                safe_console.print(json.dumps(server_data, indent=2))
            else:
                # Normal output
                await client.list_servers()
    
    finally:
        await client.close()

async def config_async(show, edit, reset):
    """Config management async function - delegates to client method."""
    client = None
    safe_console = get_safe_console()
    try:
        # Instantiate client to access config and methods
        # No need to run full setup just for config command
        client = MCPClient()

        # Determine argument string for client.cmd_config
        args = ""
        if reset: args = "reset"
        elif edit: args = "edit"
        elif show: args = "show"
        else: args = "show" # Default action if no flags are given

        # Await the internal command handler in the client
        await client.cmd_config(args)

    except Exception as e:
        safe_console.print(f"[bold red]Error running config command:[/] {str(e)}")
        log.error("Error in config_async", exc_info=True)
    finally:
        # Minimal cleanup for config command - client wasn't fully set up
        # We don't need to call client.close() here as setup wasn't run
        if client:
             # We might need to close specific resources if cmd_config opened them,
             # but in this design, it mostly reads/writes config files.
             pass

async def _export_conv_async(conversation_id: Optional[str], output_path: Optional[str]):
    """Async implementation for exporting."""
    client = None
    safe_console = get_safe_console() # Get safe console here
    try:
        client = MCPClient()
        # Minimal setup needed? Just loading graph might be enough.
        # Load graph if not loaded by constructor
        if not client.conversation_graph.nodes:
             client.conversation_graph = await ConversationGraph.load(str(client.conversation_graph_file))

        target_id = conversation_id or client.conversation_graph.current_node.id
        final_output_path = output_path # Need default logic here if None

        if not final_output_path:
            final_output_path = f"conversation_{target_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            safe_console.print(f"[dim]Output path not specified, using default: {final_output_path}[/dim]")

        success = await client.export_conversation(target_id, final_output_path)
        if not success:
            safe_console.print(f"[red]Export failed for conversation '{target_id}'.[/]")
            # Raising typer.Exit here might be cleaner than returning bool
            raise typer.Exit(code=1)
        # Success message printed by export_conversation internally

    except Exception as e:
        safe_console.print(f"[bold red]Error during export: {e}[/bold red]")
        log.error("Export error", exc_info=True)
        raise typer.Exit(code=1) from e
    finally:
        if client and hasattr(client, 'close'): # Check if close method exists
             await client.close() # Close client if setup was performed

async def _import_conv_async(file_path_str: str):
     """Async implementation for importing."""
     client = None
     safe_console = get_safe_console() # Get safe console here
     try:
         client = MCPClient()
         # Load graph if not loaded by constructor
         if not client.conversation_graph.nodes:
             client.conversation_graph = await ConversationGraph.load(str(client.conversation_graph_file))

         success = await client.import_conversation(file_path_str)
         if not success:
             # Message printed by import_conversation
             raise typer.Exit(code=1)
         # Success message printed by import_conversation

     except Exception as e:
         safe_console.print(f"[bold red]Error during import: {e}[/bold red]")
         log.error("Import error", exc_info=True)
         raise typer.Exit(code=1) from e
     finally:
        if client and hasattr(client, 'close'): # Check if close method exists
             await client.close() # Close client if setup was performed


# =============================================================================
# Typer Commands 
# =============================================================================

@app.command()
def run(
    query: Annotated[Optional[str], typer.Option("--query", "-q", help="Execute a single query and exit.")] = None,
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Specify the AI model to use for the query.")] = None,
    server: Annotated[Optional[str], typer.Option("--server", help="Connect to a specific server (by name) on startup.")] = None,
    dashboard: Annotated[bool, typer.Option("--dashboard", "-d", help="Show the live monitoring dashboard instead of running a query.")] = False,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Run in interactive CLI mode.")] = False,
    verbose_logging: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose session logging.")] = False,
    webui_flag: Annotated[bool, typer.Option("--webui", help="Launch the Web UI instead of CLI.")] = False,
    webui_host: Annotated[str, typer.Option("--host", help="Host for the Web UI server.")] = "127.0.0.1",
    webui_port: Annotated[int, typer.Option("--port", help="Port for the Web UI server.")] = 8000,
    serve_ui_file: Annotated[bool, typer.Option("--serve-ui", help="Serve the default HTML UI file from the current directory.")] = True,
    cleanup_servers: Annotated[bool, typer.Option("--cleanup-servers", help="Test and remove unreachable servers on startup.")] = False,
):
    """
    Run the MCP client (Interactive, Single Query, Web UI, or Dashboard).

    If no mode flag (--interactive, --query, --dashboard, --webui) is provided, defaults to interactive mode.
    """
    global USE_VERBOSE_SESSION_LOGGING
    if verbose_logging:
        USE_VERBOSE_SESSION_LOGGING = True
        log.setLevel(logging.DEBUG)
        stderr_console.print("[dim]Verbose logging enabled.[/dim]")

    modes_selected = sum([dashboard, interactive, webui_flag, query is not None])
    actual_interactive = interactive

    if modes_selected > 1:
        stderr_console.print("[bold red]Error: Please specify only one mode: --interactive, --query, --dashboard, or --webui.[/bold red]")
        raise typer.Exit(code=1)
    elif modes_selected == 0:
        stderr_console.print("[dim]No mode specified, defaulting to interactive mode.[/dim]")
        actual_interactive = True

    # Always call main_async, it will handle the mode internally
    asyncio.run(main_async(
        query=query, 
        model=model, 
        server=server, 
        dashboard=dashboard,
        interactive=actual_interactive,
        webui_flag=webui_flag,
        webui_host=webui_host, 
        webui_port=webui_port,
        serve_ui_file=serve_ui_file, 
        cleanup_servers=cleanup_servers,
    ))

@app.command()
def config(
    show: Annotated[bool, typer.Option("--show", "-s", help="Show current configuration.")] = False,
    edit: Annotated[bool, typer.Option("--edit", "-e", help="Edit configuration YAML file in editor.")] = False,
    reset: Annotated[bool, typer.Option("--reset", "-r", help="Reset configuration YAML to defaults (use with caution!).")] = False,
):
    """Manage client configuration (view, edit YAML, reset to defaults)."""
    # Determine default action
    actual_show = show
    if not show and not edit and not reset:
        actual_show = True # Default to showing if no other action specified

    # Execute the async config logic
    asyncio.run(config_async(
        show=actual_show,
        edit=edit,
        reset=reset
    ))

@app.command()
def servers(
    search: Annotated[bool, typer.Option("--search", "-s", help="Search for discoverable servers.")] = False,
    list_all: Annotated[bool, typer.Option("--list", "-l", help="List configured servers.")] = False,
    json_output: Annotated[bool, typer.Option("--json", help="Output server list in JSON format.")] = False,
):
    """List or discover MCP servers."""
    # Determine default action if no flags given
    actual_list_all = list_all
    if not search and not list_all and not json_output:
        actual_list_all = True # Default to listing if no other action

    # Execute the async server logic
    asyncio.run(servers_async(
        search=search,
        list_all=actual_list_all,
        json_output=json_output
    ))

@app.command("export")
def export_conv_command(
    conversation_id: Annotated[Optional[str], typer.Option("--id", "-i", help="ID of the conversation branch to export (default: current).")] = None,
    output_path: Annotated[Optional[str], typer.Option("--output", "-o", help="Path to save the exported JSON file.")] = None,
):
    """Export a conversation branch to a JSON file."""
    asyncio.run(_export_conv_async(conversation_id, output_path))

@app.command("import-conv")
def import_conv_command(
    file_path: Annotated[Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True, help="Path to the JSON conversation file to import.")],
):
    """Import a conversation from a JSON file as a new branch."""
    asyncio.run(_import_conv_async(str(file_path)))


# =============================================================================
# Main execution block
# =============================================================================

if __name__ == "__main__":
    if platform.system() == "Windows":
        colorama.init(convert=True) # Initialize colorama for Windows

    # Directly call the Typer app instance. Typer handles argument parsing
    # and calling the appropriate command function.
    app()
