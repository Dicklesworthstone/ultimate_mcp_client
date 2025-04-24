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
import inspect
import io
import ipaddress
import json
import logging
import os
import platform
import random
import re
import readline
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
from email.mime import base
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, AsyncIterator, Callable, Dict, List, NotRequired, Optional, Set, Tuple, Type, TypedDict, Union, cast
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
    MessageParam,
    MessageStreamEvent,
    ToolParam,
)
from decouple import Config as DecoupleConfig
from decouple import Csv, RepositoryEnv, UndefinedValueError
from dotenv import dotenv_values, find_dotenv, set_key
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
from openai import APIConnectionError as OpenAIAPIConnectionError, AsyncStream
from openai import APIError as OpenAIAPIError
from openai import AsyncOpenAI  # For OpenAI, Grok, DeepSeek, Mistral, Groq, Cerebras, Gemini
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
from rich.emoji import Emoji
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
    COMPLETION = "completion"; CHAT = "chat"; SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"; GENERATION = "generation"; ANALYSIS = "analysis"
    CLASSIFICATION = "classification"; TRANSLATION = "translation"; QA = "qa"
    DATABASE = "database"; QUERY = "query"; BROWSER = "browser"; DOWNLOAD = "download"
    UPLOAD = "upload"; DOCUMENT_PROCESSING = "document_processing"; DOCUMENT = "document"

class LogLevel(str, Enum): # Keep as is
    DEBUG = "DEBUG"; INFO = "INFO"; WARNING = "WARNING"; ERROR = "ERROR"; CRITICAL = "CRITICAL"

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
app = typer.Typer(help="🔌 Ultimate MCP Client - Multi-Provider Edition")

# --- StdioProtectionWrapper ---
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
def adapt_path_for_platform(command: str, args: List[str]) -> Tuple[str, List[str]]:
    # (Keep existing implementation)
    log.debug(f"adapt_path_for_platform: Initial input - command='{command}', args={args}")
    def convert_windows_path_to_linux(path_str: str) -> str:
        log.debug(f"convert_windows_path_to_linux: Checking path string: {repr(path_str)}")
        if isinstance(path_str, str) and len(path_str) > 2 and path_str[1] == ':' and path_str[2] in ['\\', '/'] and path_str[0].isalpha():
            try:
                drive_letter = path_str[0].lower()
                rest_of_path = path_str[3:].replace("\\", "/")
                linux_path = f"/mnt/{drive_letter}/{rest_of_path}"
                log.debug(f"Converted Windows path '{path_str}' to Linux path '{linux_path}'")
                return linux_path
            except Exception as e:
                log.error(f"Error during path conversion for '{path_str}': {e}", exc_info=True)
                return path_str
        log.debug(f"convert_windows_path_to_linux: Path '{path_str}' did not match Windows pattern or wasn't converted.")
        return path_str

    adapted_command = command
    if isinstance(command, str) and ':' in command and ('\\' in command or '/' in command):
         log.debug(f"Attempting conversion for command part: '{command}'")
         adapted_command = convert_windows_path_to_linux(command)
    else:
         log.debug(f"Command part '{command}' likely not a path, skipping conversion.")

    adapted_args = []
    for i, arg in enumerate(args):
        if isinstance(arg, str):
            log.debug(f"adapt_path_for_platform: Processing arg {i}: {repr(arg)}")
            converted_arg = convert_windows_path_to_linux(arg)
            adapted_args.append(converted_arg)
        else:
            log.debug(f"adapt_path_for_platform: Skipping non-string arg {i}: {repr(arg)}")
            adapted_args.append(arg)

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
    # (Keep existing implementation, checking app.mcp_client.server_manager)
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
    # (Keep existing implementation)
    original_stdout = sys.stdout
    test_buffer = io.StringIO()
    sys.stdout = test_buffer
    try:
        test_buffer.write("TEST_STDOUT_POLLUTION_VERIFICATION")
        captured = test_buffer.getvalue()
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
    HEALTHY = "healthy"; DEGRADED = "degraded"; ERROR = "error"; UNKNOWN = "unknown"

# --- ServerVersion Class ---
@dataclass
class ServerVersion:
    major: int; minor: int; patch: int
    @classmethod
    def from_string(cls, version_str: str) -> "ServerVersion":
        parts = version_str.split(".") + ["0"] * 3
        return cls(major=int(parts[0]), minor=int(parts[1]), patch=int(parts[2]))
    def __str__(self) -> str: return f"{self.major}.{self.minor}.{self.patch}"
    def is_compatible_with(self, other: "ServerVersion") -> bool: return self.major == other.major

# --- ServerMetrics Class ---
@dataclass
class ServerMetrics:
    uptime: float = 0.0; request_count: int = 0; error_count: int = 0
    avg_response_time: float = 0.0; last_checked: datetime = field(default_factory=datetime.now)
    status: ServerStatus = ServerStatus.UNKNOWN; response_times: List[float] = field(default_factory=list)
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
    name: str; type: ServerType; path: str
    args: List[str] = field(default_factory=list); enabled: bool = True; auto_start: bool = True
    description: str = ""; trusted: bool = False; categories: List[str] = field(default_factory=list)
    version: Optional[ServerVersion] = None; rating: float = 5.0; retry_count: int = 3; timeout: float = 250.0
    metrics: ServerMetrics = field(default_factory=ServerMetrics); registry_url: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {"max_attempts": 3, "backoff_factor": 0.5, "timeout_increment": 5})
    capabilities: Dict[str, bool] = field(default_factory=lambda: {"tools": True, "resources": True, "prompts": True})

# --- MCPTool, MCPResource, MCPPrompt Classes ---
@dataclass
class MCPTool:
    name: str; description: str; server_name: str
    input_schema: Dict[str, Any]; original_tool: Tool
    call_count: int = 0; avg_execution_time: float = 0.0
    execution_times: deque = field(default_factory=lambda: deque(maxlen=100))
    last_used: datetime = field(default_factory=datetime.now)
    def update_execution_time(self, time_ms: float) -> None:
        self.execution_times.append(time_ms)
        if self.execution_times: self.avg_execution_time = sum(self.execution_times) / len(self.execution_times)
        self.call_count += 1; self.last_used = datetime.now()

@dataclass
class MCPResource:
    name: str; description: str; server_name: str; template: str
    original_resource: Resource; call_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)

@dataclass
class MCPPrompt:
    name: str; description: str; server_name: str; template: str
    original_prompt: McpPromptType; call_count: int = 0
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
    query: str; response: str; model: str; timestamp: str
    server_names: List[str]; tools_used: List[str] = field(default_factory=list)
    conversation_id: Optional[str] = None; latency_ms: float = 0.0
    tokens_used: int = 0; cached: bool = False; streamed: bool = False

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
                    if target_type == bool:
                        parsed_value = env_value.lower() in ('true', '1', 't', 'yes', 'y')
                    elif target_type == int:
                        parsed_value = int(env_value)
                    elif target_type == float:
                        parsed_value = float(env_value)
                    elif target_type == list:
                        parsed_value = [item.strip() for item in env_value.split(',') if item.strip()]
                    else: # Default to string
                        parsed_value = env_value

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
                            # (Keep the existing robust server loading logic)
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
                     log.warning("'cache_ttl_mapping' in YAML is not a dict."); self.cache_ttl_mapping = {}
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
                         if current_type == bool:
                             converted_value = str(value).lower() in ('true', '1', 't', 'yes', 'y')
                         elif current_type == list and isinstance(value, list): # Ensure list conversion takes list
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
        self.entries.append(entry); self.save_sync()
    async def add_async(self, entry: ChatHistory):
        self.entries.append(entry); await self.save()

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
        if self.monitor_task: self.monitor_task.cancel(); await suppress(asyncio.CancelledError)(self.monitor_task)
        self.monitor_task = None; log.info("Server health monitoring stopped")

    async def _monitor_loop(self):
        while self.monitoring:
            try:
                await self._check_all_servers()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError: break
            except Exception as e: log.error(f"Error in server monitor: {e}"); await asyncio.sleep(5)

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
            response_time = time.time() - start_time; metrics.update_response_time(response_time)
        except Exception as e:
            metrics.error_count += 1; log.warning(f"Health check fail {server_name}: {e}")
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
        self._process = process; self._server_name = server_name; self._stdin = process.stdin
        self._stderr_reader_task: Optional[asyncio.Task] = None
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._request_id_counter = 0; self._lock = asyncio.Lock(); self._is_active = True
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
        if not self._is_active: log.warning(f"[{self._server_name}] Session inactive, skip initialized."); return
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        try:
            notification_str = json.dumps(notification) + '\n'; notification_bytes = notification_str.encode('utf-8')
            log.info(f"[{self._server_name}] Sending initialized notification...")
            if self._stdin is None or self._stdin.is_closing(): raise ConnectionAbortedError("Stdin closed")
            self._stdin.write(notification_bytes); await self._stdin.drain()
            log.debug(f"[{self._server_name}] Initialized notification sent.")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e: log.error(f"[{self._server_name}] Conn error sending initialized: {e}"); await self._close_internal_state(e)
        except Exception as e: log.error(f"[{self._server_name}] Error sending initialized: {e}", exc_info=True)

    async def _run_reader_processor_wrapper(self):
        close_exception: Optional[BaseException] = None
        try:
            log.debug(f"[{self._server_name}] Entering reader/processor task wrapper.")
            await self._read_and_process_stdout_loop()
            log.info(f"[{self._server_name}] Reader/processor task finished normally.")
        except asyncio.CancelledError: log.debug(f"[{self._server_name}] Reader/processor task wrapper cancelled."); close_exception = asyncio.CancelledError("Reader task cancelled")
        except Exception as e: log.error(f"[{self._server_name}] Reader/processor task wrapper failed: {e}", exc_info=True); close_exception = e
        finally:
            log.debug(f"[{self._server_name}] Reader/processor task wrapper exiting.")
            if self._is_active:
                log_level = logging.DEBUG if isinstance(close_exception, asyncio.CancelledError) else logging.WARNING
                log.log(log_level, f"[{self._server_name}] Reader/processor finished. Forcing close.")
                final_exception = close_exception or ConnectionAbortedError("Reader task finished unexpectedly")
                await self._close_internal_state(final_exception)

    async def _read_and_process_stdout_loop(self):
        handshake_complete = False; stream_limit = getattr(self._process.stdout, '_limit', 'Unknown')
        log.debug(f"[{self._server_name}] Starting combined reader/processor loop (Buffer limit: {stream_limit}).")
        try:
            while self._process.returncode is None:
                if not self._is_active: log.info(f"[{self._server_name}] Session inactive, exiting loop."); break
                try:
                    line_bytes = await asyncio.wait_for(self._process.stdout.readline(), timeout=60.0)
                    if not line_bytes:
                        if self._process.stdout.at_eof(): log.warning(f"[{self._server_name}] Stdout EOF."); break
                        else: log.debug(f"[{self._server_name}] readline() timeout."); continue
                    line_str_raw = line_bytes.decode('utf-8', errors='replace')
                    if USE_VERBOSE_SESSION_LOGGING: log.debug(f"[{self._server_name}] READ/PROC RAW <<< {repr(line_str_raw)}")
                    line_str = line_str_raw.strip()
                    if not line_str: continue
                    try:
                        message = json.loads(line_str)
                        is_valid_rpc = (isinstance(message, dict) and message.get("jsonrpc") == "2.0" and ('id' in message or 'method' in message))
                        if not is_valid_rpc: log.debug(f"[{self._server_name}] Skipping non-MCP JSON: {line_str[:100]}..."); continue
                        if not handshake_complete: log.info(f"[{self._server_name}] First valid JSON-RPC detected."); handshake_complete = True
                        msg_id = message.get("id")
                        if msg_id is not None:
                            str_msg_id = str(msg_id); future = self._response_futures.pop(str_msg_id, None)
                            if future and not future.done():
                                if "result" in message: log.debug(f"[{self._server_name}] READ/PROC: Resolving future ID {msg_id} with RESULT."); future.set_result(message["result"])
                                elif "error" in message:
                                    err_data = message["error"]
                                    err_msg = f"Server error ID {msg_id}: {err_data.get('message', 'Unk')} (Code: {err_data.get('code', 'N/A')})"
                                    if err_data.get('data'): err_msg += f" Data: {repr(err_data.get('data'))[:100]}..."
                                    log.warning(f"[{self._server_name}] READ/PROC: Resolving future ID {msg_id} with ERROR: {err_msg}")
                                    future.set_exception(RuntimeError(err_msg))
                                else: log.error(f"[{self._server_name}] READ/PROC: Invalid response format ID {msg_id}."); future.set_exception(RuntimeError(f"Invalid response format ID {msg_id}"))
                            elif future: log.debug(f"[{self._server_name}] READ/PROC: Future for ID {msg_id} already done.")
                            else: log.warning(f"[{self._server_name}] READ/PROC: Received response for unknown/timed-out ID: {msg_id}.")
                        elif "method" in message:
                             method_name = message['method']; log.debug(f"[{self._server_name}] READ/PROC: Received server message: {method_name}")
                             # Handle notifications/requests here if needed
                        else: log.warning(f"[{self._server_name}] READ/PROC: Unknown message structure: {repr(message)}")
                    except json.JSONDecodeError: log.debug(f"[{self._server_name}] Skipping noisy line: {line_str[:100]}...")
                    except Exception as proc_err: log.error(f"[{self._server_name}] Error processing line '{line_str[:100]}...': {proc_err}", exc_info=True)
                except asyncio.TimeoutError: log.debug(f"[{self._server_name}] Outer timeout reading stdout."); continue
                except (BrokenPipeError, ConnectionResetError): log.warning(f"[{self._server_name}] Stdout pipe broken."); break
                except ValueError as e:
                     if "longer than limit" in str(e) or "too long" in str(e): log.error(f"[{self._server_name}] Buffer limit ({stream_limit}) exceeded!", exc_info=True)
                     else: log.error(f"[{self._server_name}] ValueError reading stdout: {e}", exc_info=True)
                     break
                except Exception as read_err: log.error(f"[{self._server_name}] Error reading stdout: {read_err}", exc_info=True); break
            log.info(f"[{self._server_name}] Exiting combined reader/processor loop.")
        except asyncio.CancelledError: log.info(f"[{self._server_name}] Reader/processor loop cancelled."); raise
        except Exception as loop_err: log.error(f"[{self._server_name}] Unhandled error in reader/processor loop: {loop_err}", exc_info=True); raise

    async def _send_request(self, method: str, params: Dict[str, Any], response_timeout: float) -> Any:
        if not self._is_active or (self._process and self._process.returncode is not None): raise ConnectionAbortedError("Session inactive or process terminated")
        async with self._lock: self._request_id_counter += 1; request_id = str(self._request_id_counter)
        request = {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}
        loop = asyncio.get_running_loop(); future = loop.create_future()
        self._response_futures[request_id] = future
        try:
            request_str = json.dumps(request) + '\n'; request_bytes = request_str.encode('utf-8')
            log.debug(f"[{self._server_name}] SEND: ID {request_id} ({method}): {request_bytes.decode('utf-8', errors='replace')[:100]}...")
            if self._stdin is None or self._stdin.is_closing(): raise ConnectionAbortedError("Stdin closed")
            if USE_VERBOSE_SESSION_LOGGING: log.debug(f"[{self._server_name}] RAW >>> {repr(request_bytes)}")
            self._stdin.write(request_bytes); await self._stdin.drain()
            if USE_VERBOSE_SESSION_LOGGING: log.info(f"[{self._server_name}] Drain complete for ID {request_id}.")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            log.error(f"[{self._server_name}] SEND FAIL ID {request_id}: Pipe broken: {e}"); self._response_futures.pop(request_id, None)
            if not future.done(): future.set_exception(e)
            raise ConnectionAbortedError(f"Conn lost sending ID {request_id}: {e}") from e
        except Exception as e:
            log.error(f"[{self._server_name}] SEND FAIL ID {request_id}: {e}", exc_info=True); self._response_futures.pop(request_id, None)
            if not future.done(): future.set_exception(e)
            raise RuntimeError(f"Failed to send ID {request_id}: {e}") from e
        try:
            log.debug(f"[{self._server_name}] WAIT: Waiting for future ID {request_id} ({method}) (timeout={response_timeout}s)")
            result = await asyncio.wait_for(future, timeout=response_timeout)
            log.debug(f"[{self._server_name}] WAIT: Future resolved for ID {request_id}. Result received.")
            return result
        except asyncio.TimeoutError as timeout_error:
            log.error(f"[{self._server_name}] WAIT: Timeout waiting for future ID {request_id} ({method})"); self._response_futures.pop(request_id, None)
            raise RuntimeError(f"Timeout waiting response {method} ID {request_id}") from timeout_error
        except asyncio.CancelledError: log.debug(f"[{self._server_name}] WAIT: Wait cancelled ID {request_id} ({method})."); self._response_futures.pop(request_id, None); raise
        except Exception as wait_err:
             if future.done() and future.exception(): server_error = future.exception(); log.warning(f"[{self._server_name}] WAIT: Future ID {request_id} failed server error: {server_error}"); raise server_error from wait_err
             else: log.error(f"[{self._server_name}] WAIT: Error waiting future ID {request_id}: {wait_err}", exc_info=True); self._response_futures.pop(request_id, None); raise RuntimeError(f"Error processing response {method} ID {request_id}: {wait_err}") from wait_err

    # MCP Method implementations (rely on _send_request)
    async def list_tools(self, response_timeout: float = 40.0) -> ListToolsResult:
        log.debug(f"[{self._server_name}] Calling list_tools"); result = await self._send_request("tools/list", {}, response_timeout); return ListToolsResult(**result)
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], response_timeout: float = 250.0) -> CallToolResult:
        log.debug(f"[{self._server_name}] Calling call_tool: {tool_name}"); params = {"name": tool_name, "arguments": arguments}; result = await self._send_request("tools/call", params, response_timeout); return CallToolResult(**result)
    async def list_resources(self, response_timeout: float = 40.0) -> ListResourcesResult:
        log.debug(f"[{self._server_name}] Calling list_resources"); result = await self._send_request("resources/list", {}, response_timeout); return ListResourcesResult(**result)
    async def read_resource(self, uri: AnyUrl, response_timeout: float = 30.0) -> ReadResourceResult:
        log.debug(f"[{self._server_name}] Calling read_resource: {uri}"); params = {"uri": str(uri)}; result = await self._send_request("resources/read", params, response_timeout); return ReadResourceResult(**result)
    async def list_prompts(self, response_timeout: float = 40.0) -> ListPromptsResult:
        log.debug(f"[{self._server_name}] Calling list_prompts"); result = await self._send_request("prompts/list", {}, response_timeout); return ListPromptsResult(**result)
    async def get_prompt(self, prompt_name: str, variables: Dict[str, Any], response_timeout: float = 30.0) -> GetPromptResult:
        log.debug(f"[{self._server_name}] Calling get_prompt: {prompt_name}"); params = {"name": prompt_name, "arguments": variables}; result = await self._send_request("prompts/get", params, response_timeout); return GetPromptResult(**result)

    async def _close_internal_state(self, exception: Exception):
        if not self._is_active: return; self._is_active = False
        log.debug(f"[{self._server_name}] Closing internal state due to: {exception}")
        await self._cancel_pending_futures(exception)

    async def _cancel_pending_futures(self, exception: Exception):
        log.debug(f"[{self._server_name}] Cancelling {len(self._response_futures)} pending futures with: {exception}")
        futures_to_cancel = list(self._response_futures.items()); self._response_futures.clear()
        for _, future in futures_to_cancel:
            if future and not future.done(): await suppress(asyncio.InvalidStateError)(future.set_exception(exception))

    async def aclose(self):
        log.info(f"[{self._server_name}] Closing RobustStdioSession...")
        if not self._is_active: log.debug(f"[{self._server_name}] Already closed."); return
        await self._close_internal_state(ConnectionAbortedError("Session closed by client"))
        if self._background_task_runner and not self._background_task_runner.done():
            log.debug(f"[{self._server_name}] Cancelling reader task..."); self._background_task_runner.cancel()
            await suppress(asyncio.CancelledError)(self._background_task_runner)
        if self._stderr_reader_task and not self._stderr_reader_task.done():
             log.debug(f"[{self._server_name}] Cancelling stderr task..."); self._stderr_reader_task.cancel()
             await suppress(asyncio.CancelledError)(self._stderr_reader_task)
        if self._process and self._process.returncode is None:
            log.info(f"[{self._server_name}] Terminating process PID {self._process.pid} during aclose...")
            try:
                self._process.terminate(); await asyncio.wait_for(self._process.wait(), timeout=2.0)
                if self._process.returncode is None:
                    log.debug(f"[{self._server_name}] Killing process"); self._process.kill(); await asyncio.wait_for(self._process.wait(), timeout=1.0)
            except ProcessLookupError: pass
            except Exception as e: log.error(f"Terminate/kill error: {e}")
        log.info(f"[{self._server_name}] RobustStdioSession closed.")


# --- ServerManager Class ---
class ServerManager:
    def __init__(self, config: Config, tool_cache=None, safe_printer=None):
        self.config = config
        self.exit_stack = AsyncExitStack()
        self.active_sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.processes: Dict[str, asyncio.subprocess.Process] = {} # Changed type hint
        self.tool_cache = tool_cache
        self._safe_printer = safe_printer or print # Use passed printer
        self.monitor = ServerMonitor(self)
        self.registry = ServerRegistry() if config.enable_registry else None
        self.registered_services: Dict[str, ServiceInfo] = {}
        self._session_tasks: Dict[str, List[asyncio.Task]] = {}
        self.sanitized_to_original = {}
        self._port_scan_client: Optional[httpx.AsyncClient] = None
        self.discovered_servers_cache: List[Dict] = []
        self._discovery_in_progress = asyncio.Lock()

    @property
    def tools_by_server(self) -> Dict[str, List[MCPTool]]:
        result = {}
        for tool in self.tools.values(): result.setdefault(tool.server_name, []).append(tool)
        return result

    async def run_multi_step_task(self, steps: List[Callable], step_descriptions: List[str], title: str = "Processing...", show_spinner: bool = True) -> bool:
        if len(steps) != len(step_descriptions): log.error("Steps/descriptions length mismatch"); return False
        safe_console = get_safe_console()
        if hasattr(app, "mcp_client") and hasattr(app.mcp_client, "_run_with_progress"):
            tasks = [(steps[i], step_descriptions[i], None) for i in range(len(steps))]
            try: await app.mcp_client._run_with_progress(tasks, title, transient=True); return True
            except Exception as e: log.error(f"Error multi-step task: {e}"); return False

        progress_columns = []
        if show_spinner: progress_columns.append(SpinnerColumn())
        progress_columns.extend([TextColumn("[progress.description]{task.description}"), BarColumn(),
                                TextColumn("[cyan]{task.completed}/{task.total}"), TaskProgressColumn()])
        with Progress(*progress_columns, console=safe_console) as progress:
            task = progress.add_task(title, total=len(steps))
            for i, (step, description) in enumerate(zip(steps, step_descriptions, strict=False)):
                try:
                    progress.update(task, description=description)
                    await step()
                    progress.update(task, advance=1)
                except Exception as e:
                    log.error(f"Error in step {i+1}: {e}")
                    progress.update(task, description=f"{EMOJI_MAP['error']} {description} failed: {e}")
                    return False
            progress.update(task, description=f"{EMOJI_MAP['success']} Complete")
            return True

    def _process_list_result(self, server_name: str, result_list: Optional[List[Any]], target_dict: Dict[str, Any], item_class: Type, item_type_name: str):
        items_added = 0; items_skipped = 0
        for key in list(target_dict.keys()): # Clear existing for this server
            if hasattr(target_dict[key], 'server_name') and target_dict[key].server_name == server_name: del target_dict[key]
        if result_list is None: log.warning(f"[{server_name}] Received None for {item_type_name}s."); return
        if not isinstance(result_list, list): log.warning(f"[{server_name}] Expected list for {item_type_name}s, got {type(result_list).__name__}."); return

        for item in result_list:
            try:
                if not hasattr(item, 'name') or not isinstance(item.name, str) or not item.name:
                     log.warning(f"[{server_name}] Skipping {item_type_name} item lack valid 'name': {item}"); items_skipped += 1; continue
                if item_class is MCPTool:
                    schema = getattr(item, 'inputSchema', getattr(item, 'input_schema', None))
                    if schema is None or not isinstance(schema, dict):
                        log.warning(f"[{server_name}] Skipping tool '{item.name}' lack valid schema. Item: {item}"); items_skipped += 1; continue
                    correct_input_schema = schema
                item_name_full = f"{server_name}:{item.name}" if ":" not in item.name else item.name
                instance_data = {"name": item_name_full, "description": getattr(item, 'description', '') or '', "server_name": server_name}
                if item_class is MCPTool: instance_data["input_schema"] = correct_input_schema; instance_data["original_tool"] = item
                elif item_class is MCPResource: instance_data["template"] = getattr(item, 'uri', ''); instance_data["original_resource"] = item
                elif item_class is MCPPrompt: instance_data["template"] = f"Prompt: {item.name}"; instance_data["original_prompt"] = item
                target_dict[item_name_full] = item_class(**instance_data)
                items_added += 1
            except Exception as proc_err: log.error(f"[{server_name}] Error process {item_type_name} '{getattr(item, 'name', 'UNK')}': {proc_err}", exc_info=True); items_skipped += 1
        log.info(f"[{server_name}] Processed {items_added} {item_type_name}s ({items_skipped} skipped).")

    async def close(self):
        try:
            cleanup_timeout = 5
            if self._port_scan_client: await suppress(asyncio.TimeoutError, Exception)(asyncio.wait_for(self._port_scan_client.aclose(), timeout=cleanup_timeout / 2))
            for name in list(self.registered_services.keys()): await self.unregister_local_server(name)
            log.debug(f"Closing {len(self.active_sessions)} sessions via exit stack..."); await self.exit_stack.aclose()
            log.debug("Exit stack closed.")
            log.debug(f"Terminating {len(self.processes)} processes..."); process_terminations = []
            for name, process in list(self.processes.items()): process_terminations.append(self.terminate_process(name, process)); del self.processes[name]
            if process_terminations: await asyncio.gather(*process_terminations, return_exceptions=True)
            tasks_to_cancel = []; [tasks_to_cancel.extend(tl) for tl in self._session_tasks.values()]; self._session_tasks.clear()
            if tasks_to_cancel:
                 log.debug(f"Cancelling {len(tasks_to_cancel)} tasks..."); [t.cancel() for t in tasks_to_cancel if t and not t.done()]
                 await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
            if self.registry: await self.registry.close()
        except Exception as e: log.error(f"Error ServerManager cleanup: {e}", exc_info=True)

    async def terminate_process(self, server_name: str, process: Optional[asyncio.subprocess.Process]):
         if process is None or process.returncode is not None: log.debug(f"Process {server_name} done/None."); return
         log.info(f"Terminating process {server_name} (PID {process.pid})")
         try:
             process.terminate(); await asyncio.wait_for(process.wait(), timeout=2.0); log.info(f"Process {server_name} terminated.")
         except asyncio.TimeoutError:
             log.warning(f"Process {server_name} kill required.")
             if process.returncode is None:
                 try: process.kill(); await process.wait(); log.info(f"Process {server_name} killed.")
                 except ProcessLookupError: log.info(f"Process {server_name} already gone.")
                 except Exception as kill_err: log.error(f"Error kill {server_name}: {kill_err}")
         except ProcessLookupError: log.info(f"Process {server_name} already gone.")
         except Exception as e: log.error(f"Error terminating {server_name}: {e}")


# --- MCPClient Class  ---
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

        # Command handlers (keep list)
        self.commands = {
            'exit': self.cmd_exit, 'quit': self.cmd_exit, 'help': self.cmd_help, 'config': self.cmd_config,
            'servers': self.cmd_servers, 'tools': self.cmd_tools, 'resources': self.cmd_resources,
            'prompts': self.cmd_prompts, 'history': self.cmd_history, 'model': self.cmd_model,
            'clear': self.cmd_clear, 'reload': self.cmd_reload, 'cache': self.cmd_cache,
            'fork': self.cmd_fork, 'branch': self.cmd_branch, 'dashboard': self.cmd_dashboard,
            'optimize': self.cmd_optimize, 'tool': self.cmd_tool, 'prompt': self.cmd_prompt,
            'export': self.cmd_export, 'import': self.cmd_import, 'discover': self.cmd_discover,
        }

        # Readline setup
        readline.set_completer(self.completer)
        readline.parse_and_bind("tab: complete")

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
                 server_table.add_column("Name"); server_table.add_column("Type"); server_table.add_column("Path/URL"); server_table.add_column("Enabled")
                 for name, server in self.config.servers.items():
                      server_table.add_row(name, server.type.value, server.path, str(server.enabled))
                 safe_console.print(server_table)
            else:
                 safe_console.print(Panel("[dim]No servers defined in config.yaml[/]", title="Servers", border_style="dim green"))

            # Print TTL mapping
            if self.config.cache_ttl_mapping:
                 ttl_table = Table(box=box.ROUNDED, title="Cache TTLs (from config.yaml)")
                 ttl_table.add_column("Tool Category/Name"); ttl_table.add_column("TTL (seconds)")
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
                safe_console.print("[yellow]Usage: /config api-key <provider> <api_key>[/]"); return
            provider_str, key_value = key_parts[0].lower(), key_parts[1]
            try:
                provider = Provider(provider_str)
                attr_name = PROVIDER_CONFIG_KEY_ATTR_MAP.get(provider.value)
                if attr_name:
                    setattr(self.config, attr_name, key_value); config_changed = True
                    safe_console.print(f"[green]API key set for {provider.value} (will save to config.yaml).[/]")
                    await self._reinitialize_provider_clients(providers=[provider.value])
                else: safe_console.print(f"[red]Invalid provider for API key: {provider_str}[/]")
            except ValueError: safe_console.print(f"[red]Invalid provider: {provider_str}[/]")

        elif subcmd == "base-url":
            url_parts = subargs.split(maxsplit=1)
            if len(url_parts) != 2:
                safe_console.print("[yellow]Usage: /config base-url <provider> <url>[/]"); return
            provider_str, url_value = url_parts[0].lower(), url_parts[1]
            try:
                provider = Provider(provider_str)
                attr_name = PROVIDER_CONFIG_URL_ATTR_MAP.get(provider.value)
                if attr_name:
                    setattr(self.config, attr_name, url_value); config_changed = True
                    safe_console.print(f"[green]Base URL set for {provider.value} (will save to config.yaml).[/]")
                    await self._reinitialize_provider_clients(providers=[provider.value])
                else: safe_console.print(f"[red]Base URL config not supported/invalid provider: {provider_str}[/]")
            except ValueError: safe_console.print(f"[red]Invalid provider: {provider_str}[/]")

        elif subcmd == "model":
             if not subargs: safe_console.print("[yellow]Usage: /config model <model_name>[/]"); return
             self.config.default_model = subargs; self.current_model = subargs
             config_changed = True; safe_console.print(f"[green]Default model set to: {subargs}[/]")
        elif subcmd == "max-tokens":
             try: self.config.default_max_tokens = int(subargs); config_changed = True; safe_console.print(f"[green]Default max tokens set to: {subargs}[/]")
             except (ValueError, TypeError): safe_console.print("[yellow]Usage: /config max-tokens <number>[/]")
        elif subcmd == "history-size":
             try:
                 new_size = int(subargs)
                 if new_size <= 0: raise ValueError("Must be positive")
                 self.config.history_size = new_size; self.history = History(max_entries=new_size) # Recreate history
                 config_changed = True; safe_console.print(f"[green]History size set to: {new_size}[/]")
             except (ValueError, TypeError): safe_console.print("[yellow]Usage: /config history-size <positive_number>[/]")
        elif subcmd == "temperature":
             try:
                 temp = float(subargs)
                 if 0.0 <= temp <= 2.0: self.config.temperature = temp; config_changed = True; safe_console.print(f"[green]Temperature set to: {temp}[/]")
                 else: safe_console.print("[red]Temperature must be between 0.0 and 2.0[/]")
             except (ValueError, TypeError): safe_console.print("[yellow]Usage: /config temperature <number_between_0_and_2>[/]")

        # Boolean Flags (using helper)
        elif subcmd in [s.replace("enable_", "").replace("use_", "").replace("_", "-") for s in SIMPLE_SETTINGS_ENV_MAP.keys() if type(getattr(Config(), SIMPLE_SETTINGS_ENV_MAP[s])) == bool]:
             # Find the attribute name corresponding to the command
             attr_to_set = None
             for env_key, attr_name in SIMPLE_SETTINGS_ENV_MAP.items():
                 command_name = attr_name.replace("enable_", "").replace("use_", "").replace("_", "-")
                 if command_name == subcmd:
                     if type(getattr(self.config, attr_name)) == bool:
                         attr_to_set = attr_name
                         break
             if attr_to_set:
                 if not subargs: safe_console.print(f"Current {attr_to_set}: {getattr(self.config, attr_to_set)}"); return
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
            simple_bool_cmds = [a.replace("enable_", "").replace("use_", "").replace("_", "-") for a in SIMPLE_SETTINGS_ENV_MAP.values() if type(getattr(Config(), a)) == bool]
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
            if not value: safe_console.print("[yellow]Usage: /config port-scan enable [true|false][/]"); return False
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
                if self.config.port_scan_concurrency != val: self.config.port_scan_concurrency = val; config_changed = True
                safe_console.print(f"[green]Port scan concurrency set to: {val}[/]")
            except (ValueError, TypeError): safe_console.print("[yellow]Usage: /config port-scan concurrency <number>[/]")
        elif action == "timeout":
            try:
                val = float(value)
                if self.config.port_scan_timeout != val: self.config.port_scan_timeout = val; config_changed = True
                safe_console.print(f"[green]Port scan timeout set to: {val}s[/]")
            except (ValueError, TypeError): safe_console.print("[yellow]Usage: /config port-scan timeout <seconds>[/]")
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
                current_paths.append(path); config_changed = True; safe_console.print(f"[green]Added discovery path: {path}[/]")
            else: safe_console.print(f"[yellow]Path already exists: {path}[/]")
        elif action == "remove" and path:
            if path in current_paths:
                current_paths.remove(path); config_changed = True; safe_console.print(f"[green]Removed discovery path: {path}[/]")
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
                current_urls.append(url); config_changed = True; safe_console.print(f"[green]Added registry URL: {url}[/]")
            else: safe_console.print(f"[yellow]URL already exists: {url}[/]")
        elif action == "remove" and url:
            if url in current_urls:
                current_urls.remove(url); config_changed = True; safe_console.print(f"[green]Removed registry URL: {url}[/]")
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
                 ttl_table = Table(box=box.ROUNDED); ttl_table.add_column("Tool Category/Name"); ttl_table.add_column("TTL (seconds)")
                 for name, ttl in self.config.cache_ttl_mapping.items(): ttl_table.add_row(name, str(ttl))
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
            Provider.OPENAI.value: {"key_attr": "openai_api_key", "url_attr": "openai_base_url", "default_url": None, "client_attr": "openai_client"},
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
            try: return await func(self, *args, **kwargs)
            except McpError as e: log.error(f"MCP error exec {tool_name}: {e}"); raise RuntimeError(f"MCP error: {e}") from e
            except httpx.RequestError as e: log.error(f"Net error exec {tool_name}: {e}"); raise RuntimeError(f"Network error: {e}") from e
            except Exception as e: log.error(f"Unexpected error exec {tool_name}: {e}"); raise RuntimeError(f"Unexpected error: {e}") from e
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

    def completer(self, text, state):
        """Tab completion for commands"""
        options = [cmd for cmd in self.commands.keys() if cmd.startswith(text)]
        if state < len(options):
            return options[state]
        return None
        
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
            role = message.get("role")
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
                 validated_messages.append(cast(InternalMessage, msg_data)) # Cast after basic check

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
            result_dict[k] = sorted(list(v_set)) # Sort for consistency
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

    async def summarize_conversation(self, target_tokens: Optional[int] = None, model: Optional[str] = None) -> Optional[str]:
        """
        Generates a summary of the current conversation branch.

        Args:
            target_tokens: Approximate target token length for the summary.
            model: The model to use for summarization.

        Returns:
            The generated summary string, or None if summarization failed.
        """
        summarization_model = model or self.config.summarization_model
        target_length = target_tokens or self.config.max_summarized_tokens
        current_messages = self.conversation_graph.current_node.messages

        if not current_messages:
            log.info("Cannot summarize empty conversation.")
            return "Conversation is empty."

        log.info(f"Generating summary using {summarization_model} (target: ~{target_length} tokens)...")

        # Create the prompt for the summarization model
        # Include context about the goal of the summary
        prompt_text = (
            "You are an expert summarizer. Please summarize the following conversation history. "
            "Focus on preserving key facts, decisions, action items, important code snippets, "
            "numerical values, and the overall context needed to continue the conversation effectively. "
            "Be concise but comprehensive."
            f" Aim for a summary that is roughly {target_length} tokens long.\n\n"
            "CONVERSATION HISTORY:\n---\n"
        )

        # Append a string representation of the history
        # This simplistic approach might lose some nuance compared to sending structured messages
        history_text_parts = []
        for msg in current_messages:
            role = msg.get("role", "unknown")
            content_str = self._extract_text_from_internal_content(msg.get("content"))
            history_text_parts.append(f"{role.upper()}: {content_str}")
        prompt_text += "\n\n".join(history_text_parts)
        prompt_text += "\n---\nSUMMARY:"

        try:
            # Use process_query (non-streaming version needed) or adapt streaming
            # --- Adaptation: Simulate non-streaming call ---
            summary_result = ""
            # Use a temporary message list if process_streaming_query modifies history
            temp_summary_query_message = InternalMessage(role="user", content=prompt_text)

            # Call process_streaming_query but consume the generator to get the full result
            # Note: This assumes process_streaming_query takes a single query string
            # If it needs a full message list, adapt accordingly.
            # For this specific task, we might want a dedicated non-streaming call.
            # Let's assume process_streaming_query works for this simulation:
            async for chunk in self.process_streaming_query(prompt_text, model=summarization_model):
                if not chunk.startswith("@@STATUS@@"):
                    summary_result += chunk
            # --- End Adaptation ---

            if not summary_result:
                log.warning("Summarization model returned an empty response.")
                return None

            log.info(f"Summarization successful. Summary length: {len(summary_result)} chars.")
            return summary_result.strip()

        except Exception as e:
            log.error(f"Error during summarization: {e}", exc_info=True)
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

        Args:
            provider_name: The canonical name of the provider (e.g., "openai").
            api_key_attr: The attribute name in self.config for the API key.
            base_url_attr: The attribute name in self.config for the base URL.
            default_base_url: The default base URL if not found in config.
            client_attr: The attribute name on self to store the client instance.
            emoji_key: The key for the provider's emoji in EMOJI_MAP.

        Returns:
            A tuple containing the initialized client (or None if failed) and a status message string.
        """
        status_emoji = EMOJI_MAP.get(emoji_key, EMOJI_MAP['provider'])
        provider_title = provider_name.capitalize()
        api_key = getattr(self.config, api_key_attr, None)

        if not api_key:
            return None, f"{status_emoji} {provider_title}: [yellow]No Key[/]"

        try:
            # Determine base URL: Use config value if present, otherwise use default
            base_url = getattr(self.config, base_url_attr, None) or default_base_url
            if not base_url:
                 # Should not happen if default_base_url is provided for relevant providers
                 log.warning(f"No base URL configured or defaulted for {provider_title}. Using OpenAI default.")
                 base_url = None # Let AsyncOpenAI use its default

            log.debug(f"Initializing {provider_title} client. Key: ***{api_key[-4:]}, Base URL: {base_url}")
            # Initialize client
            client_instance = AsyncOpenAI(api_key=api_key, base_url=base_url)

            # Lightweight validation check
            await client_instance.models.list()

            # Store client on self using the provided attribute name
            setattr(self, client_attr, client_instance)
            log.info(f"{provider_title} client initialized successfully.")
            return client_instance, f"{status_emoji} {provider_title}: [green]OK[/]"

        except OpenAIAuthenticationError:
            log.error(f"{provider_title} initialization failed: Invalid API Key.")
            self.safe_print(f"[bold red]{provider_title} Error: Invalid API Key.[/]")
            setattr(self, client_attr, None) # Ensure client is None on error
            return None, f"{status_emoji} {provider_title}: [red]Auth Error[/]"
        except OpenAIAPIConnectionError as e:
            log.error(f"{provider_title} initialization failed: Connection Error - {e}")
            self.safe_print(f"[bold red]{provider_title} Error: Connection Failed ({e})[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Connection Error[/]"
        except Exception as e:
            log.error(f"Failed {provider_title} init: {e}", exc_info=True)
            self.safe_print(f"[bold red]{provider_title} Error: Initialization failed ({type(e).__name__})[/]")
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
            # Add other providers if needed
        }
        provider_env_vars = { # Map provider enum value to expected .env var name
            Provider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
            Provider.OPENAI.value: "OPENAI_API_KEY",
            Provider.GEMINI.value: "GOOGLE_API_KEY", # Special case for Gemini
            Provider.GROK.value: "GROK_API_KEY",
            Provider.DEEPSEEK.value: "DEEPSEEK_API_KEY",
            Provider.MISTRAL.value: "MISTRAL_API_KEY",
            Provider.GROQ.value: "GROQ_API_KEY",
            Provider.CEREBRAS.value: "CEREBRAS_API_KEY",
            # Add other providers if needed
        }

        # --- 1. Default Provider API Key Check & Prompt ---
        default_provider = None
        default_provider_key_attr = None
        default_provider_key_env_var = None
        key_missing = False
        key_updated_in_session = False # Flag to track if we updated the key

        # Get the path to the .env file stored during Config init
        dotenv_path = self.config.dotenv_path

        # Use dotenv to read the DEFAULT_PROVIDER value directly
        # Check os.environ first, then the .env file if found
        default_provider_name = os.getenv("DEFAULT_PROVIDER")
        if not default_provider_name and dotenv_path and Path(dotenv_path).exists():
            env_values = dotenv_values(dotenv_path)
            default_provider_name = env_values.get("DEFAULT_PROVIDER")

        if default_provider_name:
            try:
                # Validate provider name (case-insensitive)
                default_provider = Provider(default_provider_name.lower())
                default_provider_key_attr = provider_keys.get(default_provider.value)
                default_provider_key_env_var = provider_env_vars.get(default_provider.value)

                if default_provider_key_attr and default_provider_key_env_var:
                    # Check if key exists AND is non-empty in the *current* config object
                    # (which was loaded by decouple initially)
                    current_key_value = getattr(self.config, default_provider_key_attr, None)
                    if not current_key_value:
                        key_missing = True
                        log.warning(f"API key for default provider '{default_provider.value}' ({default_provider_key_env_var}) is missing or empty.")
                    else:
                        log.info(f"API key for default provider '{default_provider.value}' found.")
                else:
                    log.warning(f"Default provider '{default_provider_name}' is specified but mapping for its key/env var is missing.")
                    default_provider = None # Invalidate if mapping missing
            except ValueError:
                log.error(f"Invalid DEFAULT_PROVIDER specified: '{default_provider_name}'")
                default_provider = None # Reset if invalid

        # Prompt only if interactive, default provider is known, and key is missing
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
                    key_updated_in_session = True

                    # Persist to .env file if path exists
                    if dotenv_path and Path(dotenv_path).exists():
                        # Use dotenv.set_key to write back non-destructively
                        success = set_key(dotenv_path, default_provider_key_env_var, entered_key, quote_mode='always')
                        if success:
                            self.safe_print(f"[green]API key for {default_provider.value} saved to {dotenv_path}[/]")
                            # Update the *current* config object immediately
                            setattr(self.config, default_provider_key_attr, entered_key)
                            log.info(f"Updated config.{default_provider_key_attr} in memory.")
                        else:
                             self.safe_print(f"[red]Error: Failed to save API key to {dotenv_path}. Key will only be used for this session.[/]")
                             # Still update config for current session even if save failed
                             setattr(self.config, default_provider_key_attr, entered_key)
                    elif dotenv_path:
                        # If find_dotenv found a place but it doesn't exist (e.g., empty project)
                        # Offer to create it? For now, just warn.
                        try:
                            # Attempt to create and write
                            Path(dotenv_path).touch()
                            success = set_key(dotenv_path, default_provider_key_env_var, entered_key, quote_mode='always')
                            if success:
                                self.safe_print(f"[green]Created '{dotenv_path}' and saved API key for {default_provider.value}.[/]")
                                setattr(self.config, default_provider_key_attr, entered_key)
                                log.info(f"Created .env and updated config.{default_provider_key_attr} in memory.")
                            else:
                                raise OSError("set_key failed after creating file.")
                        except Exception as create_write_err:
                            self.safe_print(f"[red]Error: Could not create or write to '{dotenv_path}'. Key will only be used for this session. Error: {create_write_err}[/]")
                            setattr(self.config, default_provider_key_attr, entered_key)
                    else:
                        # .env file not found initially anywhere find_dotenv looked
                        self.safe_print("[yellow]Warning: '.env' file not found. API key will only be used for this session.[/]")
                        # Update config for current session
                        setattr(self.config, default_provider_key_attr, entered_key)
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
        # This section remains the same. It will use the potentially updated
        # self.config.<provider>_api_key value if the prompt was successful.
        with Status(f"{EMOJI_MAP['provider']} Initializing AI Providers...", console=safe_console, spinner="dots") as status:
            provider_status_msgs = []

            # Anthropic
            anthropic_key = self.config.anthropic_api_key # Re-read from potentially updated config
            anthropic_emoji = EMOJI_MAP.get(Provider.ANTHROPIC.value, EMOJI_MAP['provider'])
            # (Keep existing Anthropic init logic)
            if anthropic_key:
                try:
                    self.anthropic = AsyncAnthropic(api_key=anthropic_key)
                    provider_status_msgs.append(f"{anthropic_emoji} Anthropic: [green]OK[/]")
                except anthropic.AuthenticationError:
                    log.error("Anthropic: Invalid API Key."); self.anthropic = None; provider_status_msgs.append(f"{anthropic_emoji} Anthropic: [red]Auth Error[/]")
                except anthropic.APIConnectionError as e:
                    log.error(f"Anthropic: Conn Error - {e}"); self.anthropic = None; provider_status_msgs.append(f"{anthropic_emoji} Anthropic: [red]Conn Error[/]")
                except Exception as e:
                    log.error(f"Anthropic Init Error: {e}", exc_info=True); self.anthropic = None; provider_status_msgs.append(f"{anthropic_emoji} Anthropic: [red]Failed[/]")
            else:
                provider_status_msgs.append(f"{anthropic_emoji} Anthropic: [yellow]No Key[/]")

            # OpenAI-Compatible Providers (using helper)
            # (Keep existing loop calling _initialize_openai_compatible_client)
            openai_compatible_providers = [
                {"name": Provider.OPENAI.value, "key_attr": "openai_api_key", "url_attr": "openai_base_url", "default_url": None, "client_attr": "openai_client"},
                {"name": Provider.GROK.value, "key_attr": "grok_api_key", "url_attr": "grok_base_url", "default_url": "https://api.x.ai/v1", "client_attr": "grok_client"},
                {"name": Provider.DEEPSEEK.value, "key_attr": "deepseek_api_key", "url_attr": "deepseek_base_url", "default_url": "https://api.deepseek.com", "client_attr": "deepseek_client"},
                {"name": Provider.MISTRAL.value, "key_attr": "mistral_api_key", "url_attr": "mistral_base_url", "default_url": "https://api.mistral.ai/v1", "client_attr": "mistral_client"},
                {"name": Provider.GROQ.value, "key_attr": "groq_api_key", "url_attr": "groq_base_url", "default_url": "https://api.groq.com/openai/v1", "client_attr": "groq_client"},
                {"name": Provider.CEREBRAS.value, "key_attr": "cerebras_api_key", "url_attr": "cerebras_base_url", "default_url": "https://api.cerebras.ai/v1", "client_attr": "cerebras_client"},
                {"name": Provider.GEMINI.value, "key_attr": "gemini_api_key", "url_attr": "gemini_base_url", "default_url": "https://generativelanguage.googleapis.com/v1beta", "client_attr": "gemini_client"},
            ]
            for provider_info in openai_compatible_providers:
                # The helper reads the key from self.config, which might have been updated by the prompt
                client_instance, status_msg = await self._initialize_openai_compatible_client(
                    provider_name=provider_info["name"],
                    api_key_attr=provider_info["key_attr"],
                    base_url_attr=provider_info["url_attr"],
                    default_base_url=provider_info["default_url"],
                    client_attr=provider_info["client_attr"],
                    emoji_key=provider_info["name"] # Use provider name as emoji key
                )
                provider_status_msgs.append(status_msg)


            status.update(f"{EMOJI_MAP['success']} Providers checked: {' | '.join(provider_status_msgs)}")

        # --- 3. Load Conversation Graph (Keep existing) ---
        self.conversation_graph = ConversationGraph() # Start fresh
        if self.conversation_graph_file.exists():
            with Status(f"{EMOJI_MAP['history']} Loading conversation state...", console=safe_console) as status:
                try:
                    loaded_graph = await ConversationGraph.load(str(self.conversation_graph_file))
                    self.conversation_graph = loaded_graph
                    is_new_graph = (loaded_graph.root.id == "root" and not loaded_graph.root.messages and not loaded_graph.root.children and len(loaded_graph.nodes) == 1)
                    if is_new_graph and self.conversation_graph_file.read_text().strip():
                        self.safe_print("[yellow]Could not parse previous conversation state, starting fresh.[/yellow]")
                        status.update(f"{EMOJI_MAP['warning']} Previous state invalid, starting fresh")
                    else: log.info(f"Loaded conversation graph from {self.conversation_graph_file}"); status.update(f"{EMOJI_MAP['success']} Conversation state loaded")
                except Exception as setup_load_err: log.error("Unexpected error during conversation graph loading", exc_info=True); self.safe_print(f"[red]Error loading state: {setup_load_err}[/red]"); status.update(f"{EMOJI_MAP['error']} Error loading state"); self.conversation_graph = ConversationGraph()
        else: log.info("No existing conversation graph found, using new graph.")
        if not self.conversation_graph.get_node(self.conversation_graph.current_node.id): log.warning("Current node ID invalid, reset root."); self.conversation_graph.set_current_node("root")

        # --- 4. Load Claude Desktop Config (Keep existing) ---
        await self.load_claude_desktop_config()

        # --- 5. Clean Duplicate Server Configs (Keep existing) ---
        log.info("Cleaning duplicate server configurations...")
        cleaned_servers: Dict[str, ServerConfig] = {}; canonical_map: Dict[Tuple, str] = {}; duplicates_found = False
        servers_to_process = list(self.config.servers.items())
        for name, server_config in servers_to_process:
            identifier: Optional[Tuple] = None
            if server_config.type == ServerType.STDIO: identifier = (server_config.type, server_config.path, frozenset(server_config.args))
            elif server_config.type == ServerType.SSE: identifier = (server_config.type, server_config.path)
            else: log.warning(f"Server '{name}' unknown type '{server_config.type}'"); identifier = (server_config.type, name)
            if identifier is not None:
                if identifier not in canonical_map: canonical_map[identifier] = name; cleaned_servers[name] = server_config; log.debug(f"Keeping server: '{name}'")
                else: duplicates_found = True; kept_name = canonical_map[identifier]; log.debug(f"Duplicate server detected. Removing '{name}', keep '{kept_name}'.")
        if duplicates_found:
            num_removed = len(self.config.servers) - len(cleaned_servers)
            self.safe_print(f"[yellow]Removed {num_removed} duplicate server entries.[/yellow]"); self.config.servers = cleaned_servers; await self.config.save_async() # Saves YAML
        else: log.info("No duplicate server configurations found.")

        # --- 6. Stdout Pollution Check (Keep existing) ---
        if os.environ.get("MCP_VERIFY_STDOUT", "1") == "1":
            with safe_stdout(): log.info("Verifying no stdout pollution before connect..."); verify_no_stdout_pollution()

        # --- 7. Discover Servers (Keep existing) ---
        if self.config.auto_discover:
            self.safe_print(f"{EMOJI_MAP['search']} Discovering MCP servers...")
            try:
                await self.server_manager.discover_servers() # Populates cache
                await self.server_manager._process_discovery_results(interactive_mode=interactive_mode) # Adds to config
            except Exception as discover_error: log.error("Error during discovery", exc_info=True); self.safe_print(f"[red]Discovery error: {discover_error}[/]")

        # --- 8. Start Continuous Local Discovery (Keep existing) ---
        if self.config.enable_local_discovery and self.server_manager.registry:
            await self.start_local_discovery_monitoring()

        # --- 9. Connect to Enabled MCP Servers (Keep existing) ---
        servers_to_connect = {name: cfg for name, cfg in self.config.servers.items() if cfg.enabled}
        if servers_to_connect:
            self.safe_print(f"[bold blue]Connecting to {len(servers_to_connect)} MCP servers...[/]")
            connection_results = {}
            for name, server_config in list(servers_to_connect.items()):
                self.safe_print(f"[cyan]Connecting to MCP server {name}...[/]")
                try:
                    session = await self.server_manager.connect_to_server(server_config)
                    final_name = server_config.name # Name might change during connect
                    connection_results[name] = (session is not None)
                    if session: self.safe_print(f"  {EMOJI_MAP['success']} Connected to {final_name}")
                    else: log.warning(f"Failed connect MCP server: {name}"); self.safe_print(f"  {EMOJI_MAP['warning']} Failed connect {name}")
                except Exception as e: log.error(f"Exception connecting MCP server {name}", exc_info=True); self.safe_print(f"  {EMOJI_MAP['error']} Error connect {name}: {e}"); connection_results[name] = False

        # --- 10. Start Server Monitoring (Keep existing) ---
        try:
            with Status(f"{EMOJI_MAP['server']} Starting server monitoring...", spinner="dots", console=safe_console) as status:
                await self.server_monitor.start_monitoring()
                status.update(f"{EMOJI_MAP['success']} Server monitoring started")
        except Exception as monitor_error: log.error("Failed start server monitor", exc_info=True); self.safe_print(f"[red]Error starting monitor: {monitor_error}[/red]")

        # --- 11. Display Final Status (Keep existing) ---
        await self.print_status()

    # --- Provider Determination Helper (Updated) ---
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
    # _handle_anthropic_stream (No changes needed)
    async def _handle_anthropic_stream(self, stream: AsyncMessageStream) -> AsyncGenerator[Tuple[str, Any], None]:
        # (Implementation from previous response is complete)
        current_text_block = None; current_tool_use_block = None; current_tool_input_json_accumulator = ""
        input_tokens = 0; output_tokens = 0; stop_reason = "unknown"
        try:
            async for event in stream:
                event_type = event.type
                if event_type == "message_start": input_tokens = event.message.usage.input_tokens
                elif event_type == "content_block_start":
                    block_type = event.content_block.type
                    if block_type == "text": current_text_block = {"type": "text", "text": ""}
                    elif block_type == "tool_use": tool_id = event.content_block.id; tool_name = event.content_block.name; original_tool_name = self.server_manager.sanitized_to_original.get(tool_name, tool_name); current_tool_use_block = {"id": tool_id, "name": original_tool_name}; current_tool_input_json_accumulator = ""; yield ("tool_call_start", {"id": tool_id, "name": original_tool_name})
                elif event_type == "content_block_delta":
                    delta = event.delta
                    if delta.type == "text_delta":
                        if current_text_block is not None: yield ("text_chunk", delta.text)
                    elif delta.type == "input_json_delta":
                        if current_tool_use_block is not None: current_tool_input_json_accumulator += delta.partial_json; yield ("tool_call_input_chunk", {"id": current_tool_use_block["id"], "json_chunk": delta.partial_json})
                elif event_type == "content_block_stop":
                    if current_text_block is not None: current_text_block = None
                    elif current_tool_use_block is not None:
                        parsed_input = {}
                        try: parsed_input = json.loads(current_tool_input_json_accumulator) if current_tool_input_json_accumulator else {}
                        except json.JSONDecodeError as e: log.error(f"Anthropic JSON parse failed: {e}. Raw: '{current_tool_input_json_accumulator}'"); parsed_input = {"_tool_input_parse_error": f"Failed: {e}"}
                        yield ("tool_call_end", {"id": current_tool_use_block["id"], "parsed_input": parsed_input}); current_tool_use_block = None
                elif event_type == "message_delta":
                    if hasattr(event.delta, 'stop_reason') and event.delta.stop_reason: stop_reason = event.delta.stop_reason
                    if hasattr(event, 'usage') and event.usage: output_tokens = event.usage.output_tokens
                elif event_type == "message_stop": final_message = await stream.get_final_message(); stop_reason = final_message.stop_reason; output_tokens = final_message.usage.output_tokens; break
        except anthropic.APIError as e: log.error(f"Anthropic stream API error: {e}"); yield ("error", f"Anthropic API Error: {e}"); stop_reason = "error"
        except Exception as e: log.error(f"Unexpected Anthropic stream handler error: {e}", exc_info=True); yield ("error", f"Stream error: {e}"); stop_reason = "error"
        finally: yield ("final_usage", {"input_tokens": input_tokens, "output_tokens": output_tokens}); yield ("stop_reason", stop_reason)

    # _handle_openai_compatible_stream (No changes needed)
    async def _handle_openai_compatible_stream(self, stream: AsyncStream[ChatCompletionChunk], provider_name: str) -> AsyncGenerator[Tuple[str, Any], None]:
        # (Implementation from previous response is complete)
        current_tool_calls: Dict[int, Dict] = {}; output_tokens = 0; input_tokens = 0; stop_reason = "stop"; finish_reason = None
        try:
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if not choice: continue
                delta = choice.delta; finish_reason = choice.finish_reason
                if delta and delta.content: yield ("text_chunk", delta.content)
                if delta and delta.tool_calls:
                    for tc_chunk in delta.tool_calls:
                        idx = tc_chunk.index
                        if tc_chunk.id and tc_chunk.function and tc_chunk.function.name:
                            tool_id = tc_chunk.id; sanitized_name = tc_chunk.function.name; original_name = self.server_manager.sanitized_to_original.get(sanitized_name, sanitized_name)
                            current_tool_calls[idx] = {"id": tool_id, "name": original_name, "args_acc": ""}; yield ("tool_call_start", {"id": tool_id, "name": original_name})
                        if tc_chunk.function and tc_chunk.function.arguments:
                            args_chunk = tc_chunk.function.arguments
                            if idx in current_tool_calls: current_tool_calls[idx]["args_acc"] += args_chunk; yield ("tool_call_input_chunk", {"id": current_tool_calls[idx]["id"], "json_chunk": args_chunk})
                            else: log.warning(f"Args chunk unknown tool index {idx} from {provider_name}")
                if provider_name == Provider.GROQ.value and hasattr(chunk, 'x_groq') and chunk.x_groq and hasattr(chunk.x_groq, 'usage'):
                    usage = chunk.x_groq.usage
                    if usage: input_tokens = getattr(usage, 'prompt_tokens', input_tokens); current_chunk_output = getattr(usage, 'completion_tokens', 0); output_tokens = max(output_tokens, current_chunk_output) # Use max for cumulative
            for idx, tool_data in current_tool_calls.items():
                accumulated_args = tool_data["args_acc"]; parsed_input = {}
                try:
                    if accumulated_args: parsed_input = json.loads(accumulated_args)
                except json.JSONDecodeError as e: log.error(f"{provider_name} JSON parse failed tool {tool_data['name']} (ID: {tool_data['id']}): {e}. Raw: '{accumulated_args}'"); parsed_input = {"_tool_input_parse_error": f"Failed: {e}"}
                yield ("tool_call_end", {"id": tool_data["id"], "parsed_input": parsed_input})
            stop_reason = finish_reason if finish_reason else "stop"
            if stop_reason == "tool_calls": stop_reason = "tool_use"
        except (OpenAIAPIError, OpenAIAPIConnectionError, OpenAIAuthenticationError) as e: log.error(f"{provider_name} stream API error: {e}"); yield ("error", f"{provider_name} API Error: {e}"); stop_reason = "error"
        except Exception as e: log.error(f"Unexpected error {provider_name} stream handler: {e}", exc_info=True); yield ("error", f"Unexpected stream processing error: {e}"); stop_reason = "error"
        finally:
            if input_tokens == 0 or output_tokens == 0: log.warning(f"{provider_name} stream no token counts. Estimate needed.")
            yield ("final_usage", {"input_tokens": input_tokens, "output_tokens": output_tokens}); yield ("stop_reason", stop_reason)

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
                turn_input_tokens: int = 0; turn_output_tokens: int = 0; turn_cost: float = 0.0
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
                        api_stream = await cast(AsyncAnthropic, provider_client).messages.stream(
                            model=model, messages=formatted_messages, system=system_prompt, # type: ignore
                            tools=formatted_tools, max_tokens=max_tokens, temperature=self.config.temperature
                        )
                        stream_handler = self._handle_anthropic_stream(api_stream)
                    elif provider_name in [Provider.OPENAI.value, Provider.GROK.value, Provider.DEEPSEEK.value, Provider.GROQ.value, Provider.MISTRAL.value, Provider.CEREBRAS.value, Provider.GEMINI.value]:
                        api_stream = await cast(AsyncOpenAI, provider_client).chat.completions.create(
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
                    turn_api_error = f"Connection Error: {e}"; log.error(f"{provider_name} {turn_api_error}", exc_info=True)
                except (anthropic.AuthenticationError, openai.AuthenticationError) as e:
                    turn_api_error = f"Authentication Error (Check API Key): {e}"; log.error(f"{provider_name} {turn_api_error}")
                except (anthropic.PermissionDeniedError, openai.PermissionDeniedError) as e:
                    turn_api_error = f"Permission Denied: {e}"; log.error(f"{provider_name} {turn_api_error}")
                except (anthropic.NotFoundError, openai.NotFoundError) as e:
                    turn_api_error = f"API Endpoint or Model Not Found: {e}"; log.error(f"{provider_name} {turn_api_error}")
                except (anthropic.RateLimitError, openai.RateLimitError) as e:
                    turn_api_error = f"Rate Limit Exceeded: {e}"; log.warning(f"{provider_name} {turn_api_error}")
                except (anthropic.BadRequestError, openai.BadRequestError) as e:
                    turn_api_error = f"Invalid Request / Bad Request: {e}"; log.error(f"{provider_name} {turn_api_error}", exc_info=True)
                except (anthropic.APIStatusError, openai.APIStatusError) as e:
                    turn_api_error = f"API Status Error ({e.status_code}): {e}"; log.error(f"{provider_name} {turn_api_error}", exc_info=True)
                except (anthropic.APIError, openai.APIError) as e: # Catch other provider base errors
                    turn_api_error = f"API Error: {e}"; log.error(f"{provider_name} {turn_api_error}", exc_info=True)
                except httpx.RequestError as e: # Catch general network errors
                     turn_api_error = f"Network Error: {e}"; log.error(f"{provider_name} Network Error: {turn_api_error}", exc_info=True)
                except asyncio.CancelledError: # Catch cancellation during API call/stream setup
                    log.debug(f"API call/stream setup cancelled for {provider_name}")
                    raise # Propagate cancellation immediately
                except NotImplementedError as e: # Catch our own error
                     turn_api_error = str(e); log.error(turn_api_error)
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
                            is_error_flag = True; cache_used_flag = False

                            if isinstance(tool_args, dict) and "_tool_input_parse_error" in tool_args:
                                error_text = f"Client JSON parse error for '{original_tool_name}'."; tool_result_content = f"Error: {error_text}"
                                log_content_for_history = {"error": error_text, "raw_json": tool_args.get("_raw_json")}
                                yield f"@@STATUS@@\n{EMOJI_MAP['failure']} Input Error [bold]{tool_short_name}[/]: Client parse failed."
                                log.error(f"{error_text} Raw: {tool_args.get('_raw_json', 'N/A')}")
                            else:
                                mcp_tool_obj = self.server_manager.tools.get(original_tool_name)
                                if not mcp_tool_obj:
                                    error_text = f"Tool '{original_tool_name}' not found by client."; tool_result_content = f"Error: {error_text}"
                                    log_content_for_history = {"error": error_text}
                                    yield f"@@STATUS@@\n{EMOJI_MAP['failure']} Tool Error: Tool '[bold]{original_tool_name}[/]' not found."
                                else:
                                    server_name = mcp_tool_obj.server_name; servers_used.add(server_name); tools_used.append(original_tool_name)
                                    cached_result = None
                                    if self.tool_cache and self.config.enable_caching:
                                        try: cached_result = self.tool_cache.get(original_tool_name, tool_args)
                                        except TypeError: cached_result = None
                                        if cached_result is not None and not (isinstance(cached_result, str) and cached_result.startswith("Error:")):
                                            tool_result_content = cached_result; log_content_for_history = cached_result
                                            is_error_flag = False; cache_used_flag = True; cache_hits_during_query += 1
                                            content_str_tokens = self._stringify_content(cached_result)
                                            cached_tokens = self._estimate_string_tokens(content_str_tokens)
                                            yield f"@@STATUS@@\n{EMOJI_MAP['cached']} Using cache [bold]{tool_short_name}[/] ({cached_tokens:,} tokens)"
                                            log.info(f"Using cached result for {original_tool_name}")
                                        elif cached_result is not None: log.info(f"Ignoring cached error for {original_tool_name}"); cached_result = None

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
                                                log_content_for_history = mcp_result.content; is_error_flag = False
                                                content_str_tokens = self._stringify_content(tool_result_content)
                                                result_tokens = self._estimate_string_tokens(content_str_tokens)
                                                yield f"@@STATUS@@\n{EMOJI_MAP['success']} Result [bold]{tool_short_name}[/] ({result_tokens:,} tokens, {tool_latency:.1f}s)"
                                                log.info(f"Tool '{original_tool_name}' OK ({result_tokens:,} tokens, {tool_latency:.1f}s)")
                                                if self.tool_cache and self.config.enable_caching and not is_error_flag:
                                                    try: self.tool_cache.set(original_tool_name, tool_args, tool_result_content)
                                                    except TypeError: log.warning(f"Failed cache {original_tool_name}: unhashable args")
                                        except asyncio.CancelledError:
                                            log.debug(f"Tool execution cancelled: {original_tool_name}")
                                            tool_result_content = "Error: Tool execution cancelled by user."; log_content_for_history = {"error": "Tool execution cancelled"}
                                            is_error_flag = True; yield f"@@STATUS@@\n[yellow]Tool [bold]{tool_short_name}[/] aborted.[/]"; raise
                                        except Exception as exec_err:
                                            tool_latency = time.time() - tool_start_time
                                            log.error(f"Client error during tool execution {original_tool_name}: {exec_err}", exc_info=True)
                                            error_text = f"Client error: {str(exec_err)}"; tool_result_content = f"Error: {error_text}"
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
                        error_occurred = True; stop_reason = "error"; break # Mark error and break outer loop

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

                    end_time = time.time(); latency_ms = (end_time - start_time) * 1000
                    tokens_used_hist = self.session_input_tokens + self.session_output_tokens
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(self, 'history') and hasattr(self.history, 'add_async'):
                        await self.history.add_async(ChatHistory(
                            query=query, response=final_response_text, model=model, timestamp=timestamp,
                            server_names=list(servers_used), tools_used=tools_used,
                            conversation_id=self.conversation_graph.current_node.id, latency_ms=latency_ms,
                            tokens_used=tokens_used_hist, streamed=True, cached=(cache_hits_during_query > 0)
                        ))
                    else: log.warning("History object/method not found, cannot save history.")
                except Exception as final_update_err:
                    log.error(f"Error during final graph/history update: {final_update_err}", exc_info=True)

        log.info(f"Streaming query finished. Final Stop Reason: {stop_reason}. Total Latency: {(time.time() - start_time)*1000:.0f}ms")

    async def interactive_loop(self):
        """Run interactive command loop with smoother live streaming output and abort capability."""
        interactive_console = get_safe_console()

        self.safe_print("\n[bold green]MCP Client Interactive Mode[/]")
        self.safe_print("Type your query to Claude, or a command (type 'help' for available commands)")
        self.safe_print("[italic]Press Ctrl+C once to abort request, twice quickly to force exit[/italic]")

        # --- Define constants for stability ---
        RESPONSE_HEIGHT = 35 # Fixed height for the response panel
        STATUS_HEIGHT = 13 # Number of status lines to show
        TOTAL_PANEL_HEIGHT = RESPONSE_HEIGHT + STATUS_HEIGHT + 5 # Response + Status + Borders + Title + Abort message space
        REFRESH_RATE = 10.0 # Increased refresh rate (10 times per second)

        @contextmanager
        def suppress_all_logs():
            """Temporarily suppress ALL logging output."""
            root_logger = logging.getLogger()
            original_level = root_logger.level
            try:
                root_logger.setLevel(logging.CRITICAL + 1) # Suppress everything below CRITICAL
                yield
            finally:
                root_logger.setLevel(original_level)

        while True:
            live_display: Optional[Live] = None
            self.current_query_task = None
            try:
                user_input = Prompt.ask("\n[bold blue]>>[/]", console=interactive_console)

                # Check if it's a commandbase
                if user_input.startswith('/'):
                    cmd_parts = user_input[1:].split(maxsplit=1)
                    cmd = cmd_parts[0].lower()
                    args = cmd_parts[1] if len(cmd_parts) > 1 else ""

                    if cmd in self.commands:
                        # Ensure Live display is stopped before running a command
                        if live_display and live_display.is_started:
                            live_display.stop()
                            live_display = None
                        await self.commands[cmd](args)
                    else:
                        interactive_console.print(f"[yellow]Unknown command: {cmd}[/]")
                        interactive_console.print("Type '/help' for available commands")

                # Empty input
                elif not user_input.strip():
                    continue

                # Process as a query to Claude
                else:
                    # ================================================================
                    # <<< FIX: RESET SESSION STATS BEFORE PROCESSING QUERY >>>
                    # ================================================================
                    self.session_input_tokens = 0
                    self.session_output_tokens = 0
                    self.session_total_cost = 0.0
                    self.cache_hit_count = 0 # Reset per-query cache stats too
                    self.cache_miss_count = 0
                    self.tokens_saved_by_cache = 0
                    # ================================================================

                    status_lines = deque(maxlen=STATUS_HEIGHT) # Store recent status lines
                    abort_message = Text("Press Ctrl+C once to abort...", style="dim yellow")
                    first_response_received = False

                    # --- Pre-create empty placeholders ---
                    empty_response_text = "\n" * (RESPONSE_HEIGHT - 2) # Approx lines inside panel
                    empty_status_lines = [Text("", style="dim") for _ in range(STATUS_HEIGHT)]

                    # --- Create the initial panel structure with fixed heights ---
                    response_content = Panel(
                        Text(f"Waiting for Claude's response...\n{empty_response_text}", style="dim"),
                        title="Response",
                        height=RESPONSE_HEIGHT, # Fixed height
                        border_style="dim blue"
                    )

                    status_content = Panel(
                        Group(*empty_status_lines),
                        title="Status",
                        height=STATUS_HEIGHT + 2, # Fixed height + borders/title
                        border_style="dim blue"
                    )

                    initial_panel = Panel(
                        Group(
                            response_content,
                            status_content,
                            abort_message # Reserve space for this
                        ),
                        title="Claude",
                        border_style="dim green",
                        height=TOTAL_PANEL_HEIGHT # Fixed height
                    )

                    # Initialize Live display with updated settings
                    live_display = Live(
                        initial_panel,
                        console=interactive_console,
                        refresh_per_second=REFRESH_RATE, # Use the new rate
                        transient=True, # Clears the display on exit
                        vertical_overflow="crop" # Crop content outside panels
                    )

                    # Suppress logs *only* during the live update part
                    with suppress_all_logs():
                        try:
                            live_display.start()

                            log.debug("Creating query task...")
                            query_task = asyncio.create_task(
                                self._iterate_streaming_query(user_input, status_lines),
                                name=f"query-{user_input[:20]}"
                            )
                            self.current_query_task = query_task
                            log.debug(f"Query task {self.current_query_task.get_name()} started.")

                            # --- Live Update Loop ---
                            while not self.current_query_task.done():
                                claude_text_content = getattr(self, "_current_query_text", "")
                                if not first_response_received and (claude_text_content or status_lines):
                                    first_response_received = True

                                # --- Prepare Response Renderable ---
                                if claude_text_content:
                                    response_renderable = Markdown(claude_text_content)
                                else:
                                    response_renderable = Text(f"Waiting for Claude's response...\n{empty_response_text}", style="dim")

                                # --- Prepare Status Renderable ---
                                current_status_list = list(status_lines)
                                display_status_lines = current_status_list[-STATUS_HEIGHT:]
                                if len(display_status_lines) < STATUS_HEIGHT:
                                    padding = [Text("", style="dim") for _ in range(STATUS_HEIGHT - len(display_status_lines))]
                                    display_status_lines = padding + display_status_lines
                                status_renderable = Group(*display_status_lines)

                                # --- Rebuild Panels ---
                                response_panel = Panel(
                                    response_renderable, title="Response", height=RESPONSE_HEIGHT,
                                    border_style="blue" if first_response_received else "dim blue"
                                )
                                status_panel = Panel(
                                    status_renderable, title="Status", height=STATUS_HEIGHT + 2,
                                    border_style="blue" if status_lines else "dim blue"
                                )

                                # --- Check Abort ---
                                abort_needed = self.current_query_task and not self.current_query_task.done()

                                # --- Assemble Panel ---
                                updated_panel = Panel(
                                    Group(response_panel, status_panel, abort_message if abort_needed else Text("")),
                                    title="Claude", border_style="green" if first_response_received else "dim green",
                                    height=TOTAL_PANEL_HEIGHT
                                )

                                # --- Update Live ---
                                live_display.update(updated_panel)

                                # --- Wait ---
                                try:
                                    await asyncio.wait_for(asyncio.shield(self.current_query_task), timeout=1.0 / REFRESH_RATE)
                                except asyncio.TimeoutError:
                                    pass
                                except asyncio.CancelledError:
                                    log.debug("Query task cancelled while display loop was waiting.")
                                    break

                            # --- Await task completion ---
                            await self.current_query_task

                            # --- Prepare Final Display (Normal Completion) ---
                            claude_text_content = getattr(self, "_current_query_text", "")
                            final_response_renderable = Markdown(claude_text_content) if claude_text_content else Text("No response received.", style="dim")

                            final_status_list = list(status_lines)[-STATUS_HEIGHT:]
                            if len(final_status_list) < STATUS_HEIGHT:
                                padding = [Text("", style="dim") for _ in range(STATUS_HEIGHT - len(final_status_list))]
                                final_status_list = padding + final_status_list
                            final_status_renderable = Group(*final_status_list)

                            final_response_panel = Panel(final_response_renderable, title="Response", height=RESPONSE_HEIGHT, border_style="blue")
                            final_status_panel = Panel(final_status_renderable, title="Status", height=STATUS_HEIGHT + 2, border_style="blue")
                            final_panel = Panel(Group(final_response_panel, final_status_panel), title="Claude", border_style="green", height=TOTAL_PANEL_HEIGHT)


                        except asyncio.CancelledError:
                            # --- Prepare Final Display (Cancellation) ---
                            log.debug("Query task caught CancelledError in live block.")
                            claude_text_content = getattr(self, "_current_query_text", "")
                            response_renderable = Markdown(claude_text_content) if claude_text_content else Text("Response aborted.", style="dim")
                            status_lines.append(Text("[bold yellow]Request Aborted.[/]", style="yellow"))

                            aborted_status_list = list(status_lines)[-STATUS_HEIGHT:]
                            if len(aborted_status_list) < STATUS_HEIGHT:
                                padding = [Text("", style="dim") for _ in range(STATUS_HEIGHT - len(aborted_status_list))]
                                aborted_status_list = padding + aborted_status_list
                            aborted_status_renderable = Group(*aborted_status_list)

                            aborted_response_panel = Panel(response_renderable, title="Response", height=RESPONSE_HEIGHT, border_style="yellow")
                            aborted_status_panel = Panel(aborted_status_renderable, title="Status", height=STATUS_HEIGHT + 2, border_style="yellow")
                            final_panel = Panel(Group(aborted_response_panel, aborted_status_panel), title="Claude - Aborted", border_style="yellow", height=TOTAL_PANEL_HEIGHT)


                        except Exception as e:
                             # --- Prepare Final Display (Error) ---
                            log.error(f"Error during query/live update: {e}", exc_info=True)
                            claude_text_content = getattr(self, "_current_query_text", "")
                            response_renderable = Markdown(claude_text_content) if claude_text_content else Text("Error occurred.", style="dim")
                            status_lines.append(Text(f"[bold red]Error: {e}[/]", style="red"))

                            error_status_list = list(status_lines)[-STATUS_HEIGHT:]
                            if len(error_status_list) < STATUS_HEIGHT:
                                padding = [Text("", style="dim") for _ in range(STATUS_HEIGHT - len(error_status_list))]
                                error_status_list = padding + error_status_list
                            error_status_renderable = Group(*error_status_list)

                            error_response_panel = Panel(response_renderable, title="Response", height=RESPONSE_HEIGHT, border_style="red")
                            error_status_panel = Panel(error_status_renderable, title="Status", height=STATUS_HEIGHT + 2, border_style="red")
                            final_panel = Panel(Group(error_response_panel, error_status_panel), title="Claude - ERROR", border_style="red", height=TOTAL_PANEL_HEIGHT)

                        finally:
                            # --- Cleanup Live Display ---
                            log.debug(f"Query task cleanup. Task ref: {self.current_query_task.get_name() if self.current_query_task else 'None'}")
                            self.current_query_task = None
                            if hasattr(self, "_current_query_text"): delattr(self, "_current_query_text")
                            if live_display and live_display.is_started: live_display.stop()
                            live_display = None

                            # Print the final state panel
                            if 'final_panel' in locals():
                                interactive_console.print(final_panel)

                            # --- Print Final Stats (Now reflects *this query's* stats) ---
                            # Calculate hit rate for *this query*
                            hit_rate = 0
                            if hasattr(self, 'cache_hit_count') and (self.cache_hit_count + self.cache_miss_count) > 0:
                                hit_rate = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) * 100

                            # Calculate cost savings for *this query*
                            cost_saved = 0
                            if hasattr(self, 'tokens_saved_by_cache') and self.tokens_saved_by_cache > 0:
                                model_cost_info = COST_PER_MILLION_TOKENS.get(self.current_model, {})
                                input_cost_per_token = model_cost_info.get("input", 0) / 1_000_000
                                cost_saved = self.tokens_saved_by_cache * input_cost_per_token * 0.9 # 90% saving

                            # Assemble token stats text using the reset session variables
                            token_stats = [
                                "Tokens: ",
                                ("Input: ", "dim cyan"), (f"{self.session_input_tokens:,}", "cyan"), " | ",
                                ("Output: ", "dim magenta"), (f"{self.session_output_tokens:,}", "magenta"), " | ",
                                ("Total: ", "dim white"), (f"{self.session_input_tokens + self.session_output_tokens:,}", "white"),
                                " | ",
                                ("Cost: ", "dim yellow"), (f"${self.session_total_cost:.4f}", "yellow")
                            ]

                            # Add cache stats if applicable for *this query*
                            if hasattr(self, 'cache_hit_count') and (self.cache_hit_count + self.cache_miss_count) > 0:
                                token_stats.extend([
                                    "\n",
                                    ("Cache: ", "dim green"),
                                    (f"Hits: {self.cache_hit_count}", "green"), " | ",
                                    (f"Misses: {self.cache_miss_count}", "yellow"), " | ",
                                    (f"Hit Rate: {hit_rate:.1f}%", "green"), " | ",
                                    (f"Tokens Saved: {self.tokens_saved_by_cache:,}", "green bold"), " | ",
                                    (f"Cost Saved: ${cost_saved:.4f}", "green bold")
                                ])

                            # Create and print the final stats panel
                            final_stats_panel = Panel(
                                Text.assemble(*token_stats),
                                title="Final Stats (This Query)", # Title reflects it's per-query now
                                border_style="green"
                            )
                            interactive_console.print(final_stats_panel)

            # --- Outer Loop Exception Handling ---
            except KeyboardInterrupt:
                if live_display and live_display.is_started:
                    live_display.stop()
                self.safe_print("\n[yellow]Input interrupted.[/]")
                continue # Go to the next loop iteration

            except Exception as e:
                if live_display and live_display.is_started:
                    live_display.stop()
                self.safe_print(f"[bold red]Unexpected Error:[/] {str(e)}")
                log.error(f"Unexpected error in interactive loop: {e}", exc_info=True)
                # Continue the loop after an unexpected error
                continue

            finally:
                # Final cleanup for this loop iteration
                if live_display and live_display.is_started:
                    live_display.stop()
                if self.current_query_task:
                    # Ensure task is cancelled if loop exits unexpectedly
                    if not self.current_query_task.done():
                        self.current_query_task.cancel()
                    self.current_query_task = None
                if hasattr(self, "_current_query_text"):
                    delattr(self, "_current_query_text")

    # --- Close Method ---
    async def close(self):
        """Clean up resources including provider clients."""
        log.info("Closing MCPClient Multi...")
        # Stop local discovery monitoring
        if self.local_discovery_task: await self.stop_local_discovery_monitoring()
        # Save conversation graph
        try: await self.conversation_graph.save(str(self.conversation_graph_file)); log.info(f"Saved graph {self.conversation_graph_file}")
        except Exception as e: log.error(f"Failed save graph: {e}")
        # Stop server monitor
        if hasattr(self, 'server_monitor'): await self.server_monitor.stop_monitoring()
        # Close MCP server connections
        if hasattr(self, 'server_manager'): await self.server_manager.close()

        # --- Close Provider Clients ---
        # Close OpenAI SDK based clients (if initialized and have aclose)
        clients_to_close = [self.openai_client, self.grok_client, self.deepseek_client]
        for client_instance in clients_to_close:
            if client_instance and hasattr(client_instance, 'aclose'):
                try: await client_instance.aclose()
                except Exception as e: log.warning(f"Error closing SDK client: {e}")

        # Anthropic usually don't need explicit close
        log.info("MCPClient Multi closed.")


    async def _iterate_streaming_query(self, query: str, status_lines: deque):
        """Helper to run streaming query and store text for Live display."""
        self._current_query_text = ""  # Initialize response text storage
        self._current_status_messages = []  # Initialize status messages storage
        
        try:
            # Directly iterate the stream generator provided by process_streaming_query
            async for chunk in self.process_streaming_query(query):
                if chunk.startswith("@@STATUS@@"):
                    status_message = chunk[len("@@STATUS@@"):].strip()
                    # Add to our status message list
                    self._current_status_messages.append(status_message)
                    # Also add to the deque for compatibility
                    status_lines.append(Text.from_markup(status_message))
                else:
                    # It's a regular text chunk from Claude
                    if asyncio.current_task().cancelled(): 
                        raise asyncio.CancelledError()
                    self._current_query_text += chunk
                # Yield control briefly to allow display updates
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            log.debug("Streaming query iteration cancelled internally.")
            raise  # Re-raise CancelledError
        except Exception as e:
            log.error(f"Error in _iterate_streaming_query: {e}", exc_info=True)
            # Store error in status to be displayed
            self._current_status_messages.append(f"[bold red]Query Error: {e}[/]")
            status_lines.append(Text(f"[bold red]Query Error: {e}[/]", style="red"))


    # --- Command Handlers ---

@app.command()
def export(
    conversation_id: Annotated[str, typer.Option("--id", "-i", help="Conversation ID to export")] = None,
    output: Annotated[str, typer.Option("--output", "-o", help="Output file path")] = None,
):
    """Export a conversation to a file"""
    asyncio.run(export_async(conversation_id, output))

async def export_async(conversation_id: str = None, output: str = None):
    """Async implementation of the export command"""
    client = MCPClient()
    safe_console = get_safe_console()
    try:
        # Get current conversation if not specified
        if not conversation_id:
            conversation_id = client.conversation_graph.current_node.id
            
        # Default filename if not provided
        if not output:
            output = f"conversation_{conversation_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        success = await client.export_conversation(conversation_id, output)
        if success:
            safe_console.print(f"[green]Conversation exported to: {output}[/]")
        else:
            safe_console.print(f"[red]Failed to export conversation.[/]")
    finally:
        await client.close()

@app.command()
def import_conv(
    file_path: Annotated[str, typer.Argument(help="Path to the exported conversation file")],
):
    """Import a conversation from a file"""
    asyncio.run(import_async(file_path))

async def import_async(file_path: str):
    """Async implementation of the import command"""
    client = MCPClient()
    safe_console = get_safe_console()
    try:
        success = await client.import_conversation(file_path)
        if success:
            safe_console.print(f"[green]Conversation imported successfully from: {file_path}[/]")
        else:
            safe_console.print(f"[red]Failed to import conversation.[/]")
    finally:
        await client.close()

# Define Typer CLI commands
@app.command()
def run(
    query: Annotated[str, typer.Option("--query", "-q", help="Single query to process")] = None,
    model: Annotated[str, typer.Option("--model", "-m", help="Model to use for query")] = None,
    server: Annotated[List[str], typer.Option("--server", "-s", help="Connect to specific server(s)")] = None,
    dashboard: Annotated[bool, typer.Option("--dashboard", "-d", help="Show dashboard")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable verbose logging")] = False,
    interactive: Annotated[bool, typer.Option("--interactive", "-i", help="Run in interactive mode")] = False,
    # --- Added Web UI Flag ---
    webui: Annotated[bool, typer.Option("--webui", "-w", help="Run the experimental Web UI instead of CLI")] = False,
    webui_host: Annotated[str, typer.Option("--host", "-h", help="Host for Web UI")] = "127.0.0.1",
    webui_port: Annotated[int, typer.Option("--port", "-p", help="Port for Web UI")] = 8017,
    serve_ui_file: Annotated[bool, typer.Option("--serve-ui", help="Serve the default mcp_client_ui.html file")] = True,
    cleanup_servers: Annotated[bool, typer.Option("--cleanup-servers", help="Test and remove unreachable servers from config")] = False, # <-- ADDED FLAG
):
    """Run the MCP client in various modes (CLI, Interactive, Dashboard, or Web UI)."""
    # Configure logging based on verbosity
    if verbose:
        logging.getLogger("mcpclient").setLevel(logging.DEBUG)
        global USE_VERBOSE_SESSION_LOGGING # Allow modification
        USE_VERBOSE_SESSION_LOGGING = True
        log.info("Verbose logging enabled.")

    # Run the main async function
    # Pass new webui flags
    asyncio.run(main_async(query, model, server, dashboard, interactive, verbose, webui, webui_host, webui_port, serve_ui_file, cleanup_servers))

@app.command()
def servers(
    search: Annotated[bool, typer.Option("--search", "-s", help="Search for servers to add")] = False,
    list_all: Annotated[bool, typer.Option("--list", "-l", help="List all configured servers")] = False,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
):
    """Manage MCP servers"""
    # Run the server management function
    asyncio.run(servers_async(search, list_all, json_output))

@app.command()
def config(
    show: Annotated[bool, typer.Option("--show", "-s", help="Show current configuration")] = False,
    edit: Annotated[bool, typer.Option("--edit", "-e", help="Edit configuration YAML file in editor")] = False,
    reset: Annotated[bool, typer.Option("--reset", "-r", help="Reset configuration YAML to defaults (use with caution!)")] = False,
):
    """Manage client configuration (view, edit YAML, reset to defaults)."""
    # Run the config management function
    asyncio.run(config_async(show, edit, reset))

async def main_async(query, model, server, dashboard, interactive, verbose_logging, webui_flag, webui_host, webui_port, serve_ui_file, cleanup_servers):
    """Main async entry point - Handles CLI, Interactive, Dashboard, and Web UI modes."""
    client = None # Initialize client to None
    safe_console = get_safe_console()
    max_shutdown_timeout = 10

    # --- Shared Setup ---
    try:
        log.info("Initializing MCPClient...")
        client = MCPClient() # Instantiation inside the try block
        await client.setup(interactive_mode=interactive or webui_flag) # Pass interactive if either mode uses it

        if cleanup_servers:
            log.info("Cleanup flag detected. Testing and removing unreachable servers...")
            await client.cleanup_non_working_servers()
            log.info("Server cleanup process complete.")

        # --- Mode Selection ---
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
                    raise HTTPException(status_code=500, detail="Internal server error retrieving configuration.")

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
                     raise HTTPException(status_code=500, detail="Internal error retrieving server details.")

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

                except asyncio.CancelledError:
                    mcp_client.safe_print(f"[yellow]API Tool execution [bold]{tool_short_name}[/] cancelled.[/]")
                    raise HTTPException(status_code=499, detail="Tool execution cancelled by client")
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
                    raise HTTPException(status_code=500, detail="Internal error: Conversation state not serializable")

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
                     raise HTTPException(status_code=500, detail="Internal error processing history search results.")

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
                     raise HTTPException(status_code=500, detail="Internal error generating dashboard data.")

            # --- WebSocket Chat Endpoint ---
            @app.websocket("/ws/chat")
            async def websocket_chat(websocket: WebSocket):
                """Handles real-time chat interactions via WebSocket."""
                try:
                    mcp_client: MCPClient = websocket.app.state.mcp_client
                except AttributeError:
                    log.error("app.state.mcp_client not available during WebSocket connection!")
                    await websocket.close(code=1011); return

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
                    payload = {"success": success, "message": message}; payload.update(data or {})
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

                                mcp_client.session_input_tokens = 0; mcp_client.session_output_tokens = 0
                                mcp_client.session_total_cost = 0.0; mcp_client.cache_hit_count = 0
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
                                    cmd = parts[0].lower(); args = parts[1] if len(parts) > 1 else ""
                                    log.info(f"WS-{connection_id} processing command: /{cmd} {args}")
                                    try:
                                        if cmd == "clear":
                                             mcp_client.conversation_graph.current_node.messages = []
                                             mcp_client.conversation_graph.set_current_node("root")
                                             await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                                             await send_command_response(True, "Conversation cleared.", {"messages": []}) # Send cleared messages
                                        elif cmd == "model":
                                             if args:
                                                 mcp_client.current_model = args; mcp_client.config.default_model = args
                                                 asyncio.create_task(mcp_client.config.save_async()) # Persist default model change
                                                 await send_command_response(True, f"Model set to: {args}", {"currentModel": args})
                                             else: await send_command_response(True, f"Current model: {mcp_client.current_model}", {"currentModel": mcp_client.current_model})
                                        elif cmd == "fork":
                                             new_node = mcp_client.conversation_graph.create_fork(name=args if args else None)
                                             mcp_client.conversation_graph.set_current_node(new_node.id)
                                             await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                                             await send_command_response(True, f"Created branch: {new_node.name}", {"newNodeId": new_node.id, "newNodeName": new_node.name})
                                        elif cmd == "checkout":
                                             if not args: await send_error_response("Usage: /checkout NODE_ID_or_Prefix", cmd); continue
                                             node_id=args; node=mcp_client.conversation_graph.get_node(node_id)
                                             if not node: matched_node=None; [matched_node := n for n_id, n in mcp_client.conversation_graph.nodes.items() if n_id.startswith(node_id) if not matched_node]; node=matched_node
                                             if node and mcp_client.conversation_graph.set_current_node(node.id):
                                                 await send_command_response(True, f"Switched to branch: {node.name}", {"currentNodeId": node.id, "messages": node.messages}) # Send messages of new node
                                             else: await send_error_response(f"Node ID '{node_id}' not found.", cmd)
                                        elif cmd == "apply_prompt": # Renamed from 'prompt' for clarity
                                             if not args: await send_error_response("Usage: /apply_prompt <prompt_name>", cmd); continue
                                             success = await mcp_client.apply_prompt_to_conversation(args)
                                             if success:
                                                 await send_command_response(True, f"Applied prompt: {args}", {"messages": mcp_client.conversation_graph.current_node.messages})
                                             else: await send_error_response(f"Prompt not found: {args}", cmd)
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
                                    except Exception as cmd_err: await send_error_response(f"Error executing '/{cmd}': {cmd_err}", cmd); log.error(f"WS-{connection_id} Cmd Error /{cmd}: {cmd_err}", exc_info=True)
                                else: await send_error_response("Invalid command format.", None)

                            # Ignore other message types for now
                            # elif message.type == "other_type": ...

                        except (json.JSONDecodeError, ValidationError) as e: log.warning(f"WS-{connection_id} invalid message: {raw_data[:100]}... Error: {e}"); await send_ws_message("error", {"message": "Invalid message format."})
                        except WebSocketDisconnect: raise
                        except Exception as e: log.error(f"WS-{connection_id} error processing message: {e}", exc_info=True); await suppress(Exception)(send_ws_message("error", {"message": f"Internal error: {str(e)}"}))

                except WebSocketDisconnect: log.info(f"WebSocket connection closed (ID: {connection_id}).")
                except Exception as e: log.error(f"WS-{connection_id} unexpected handler error: {e}", exc_info=True); await suppress(Exception)(websocket.close(code=1011))
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
                    async def serve_html(): return FileResponse(str(ui_file.resolve()))
                else: log.warning(f"UI file {ui_file} not found. Cannot serve.")

            log.info("Starting Uvicorn server...")
            # --- Run Uvicorn Programmatically ---
            config = uvicorn.Config(app, host=webui_host, port=webui_port, log_level="info")
            server_instance = uvicorn.Server(config)
            await server_instance.serve()
            # Server runs here until interrupted

            log.info("Web UI server shut down.")
            # Cleanup is handled by the lifespan manager

        # --- Original CLI/Dashboard/Interactive Logic ---
        # (Connect to specific server block remains the same)
        # ... (Connect to specific server logic - Keep this) ...

        elif dashboard:
            # (Dashboard logic remains the same)
            # ...
            if not client.server_monitor.monitoring:
                 await client.server_monitor.start_monitoring()
            await client.cmd_dashboard("")
            await client.close() # Ensure cleanup after dashboard closes
            return

        elif query:
            # (Single query logic remains the same)
            # ...
             try:
                 result = await client.process_query(query, model=model)
                 safe_console.print()
                 safe_console.print(Panel.fit(
                     Markdown(result),
                     title=f"Claude ({client.current_model})",
                     border_style="green"
                 ))
             except Exception as query_error:
                 safe_console.print(f"[bold red]Error processing query:[/] {str(query_error)}")
                 if verbose_logging:
                     import traceback
                     safe_console.print(traceback.format_exc())

        elif interactive or not query:
            # (Interactive loop logic remains the same)
            # ...
             if not client.config.api_key and interactive:
                  # ... (API key prompt logic) ...
                 pass # Keep the logic
             await client.interactive_loop()

    except KeyboardInterrupt:
        safe_console.print("\n[yellow]Interrupted, shutting down...[/]")
        # Cleanup is handled in finally block
    except Exception as main_async_error:
        # (Main error handling remains mostly the same)
        safe_console.print(f"[bold red]An unexpected error occurred in the main process: {main_async_error}[/]")
        print(f"\n--- Traceback for Main Process Error ({type(main_async_error).__name__}) ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("--- End Traceback ---", file=sys.stderr)
        if not interactive and not webui_flag: # Exit only for non-interactive CLI modes
             if client and hasattr(client, 'close'):
                 try: await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout / 2)
                 except Exception: pass
             sys.exit(1)
    finally:
        # Cleanup: Ensure client.close is called if client was initialized
        # Note: If FastAPI is running, its lifespan manager handles client.close()
        if not webui_flag and client and hasattr(client, 'close'):
            log.info("Performing final cleanup...")
            try:
                await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout)
            except asyncio.TimeoutError:
                safe_console.print("[red]Shutdown timed out. Some processes may still be running.[/]")
                # (Force kill logic remains the same)
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

# --- main function ---
async def main():
    is_webui_mode = "--webui" in sys.argv
    if not is_webui_mode:
        try: app()
        except McpError as e: get_safe_console().print(f"[bold red]MCP Error:[/] {str(e)}"); sys.exit(1)
        except httpx.RequestError as e: get_safe_console().print(f"[bold red]Network Error:[/] {str(e)}"); sys.exit(1)
        except anthropic.APIError as e: get_safe_console().print(f"[bold red]Anthropic API Error:[/] {str(e)}"); sys.exit(1)
        except openai.APIError as e: get_safe_console().print(f"[bold red]OpenAI Compatible API Error:[/] {str(e)}"); sys.exit(1)
        except (OSError, yaml.YAMLError, json.JSONDecodeError) as e: get_safe_console().print(f"[bold red]File/Config Error:[/] {str(e)}"); sys.exit(1)
        except Exception as e:
            get_safe_console().print(f"[bold red]Unexpected Error:[/] {str(e)}")
            if os.environ.get("MCP_CLIENT_DEBUG"): get_safe_console().print_exception(show_locals=True)
            sys.exit(1)

# --- if __name__ == "__main__": ---
if __name__ == "__main__":
    if platform.system() == "Windows": colorama.init(convert=True)
    # Call main which handles the Typer app logic for non-webui modes
    asyncio.run(main()) # Run the main async function

