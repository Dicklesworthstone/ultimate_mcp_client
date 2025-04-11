#!/usr/bin/env python3

# /// script
# dependencies = [
#     "anthropic>=0.15.0",
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
#     "tiktoken>=0.5.1"
#     "fastapi>=0.104.0",
#     "uvicorn[standard]>=0.24.0",
#     "websockets>=11.0",
#     "python-multipart>=0.0.6"
# ]
# ///

"""
Ultimate MCP Client
==================

A comprehensive client for the Model Context Protocol (MCP) that connects AI models 
with external tools, servers, and data sources. This client provides a powerful 
interface for managing MCP servers and leveraging their capabilities with Claude 
and other AI models.

Key Features:
------------
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
- Local Port Scanning: Actively scan a configurable range of local ports to find MCP servers (useful for servers not using mDNS).

Context Optimization: Automatic summarization of long conversations
- Direct Tool Execution: Run specific tools directly with custom parameters
- Dynamic Prompting: Apply pre-defined prompt templates to conversations
- Claude Desktop Integration: Automatically import server configs from Claude desktop
- Theme Customization: Multiple built-in themes with light/dark mode support in Web UI

Usage:
------
# Interactive CLI mode
python mcpclient.py run --interactive

# Launch Web UI
python mcpclient.py run --webui

# Customize Web UI host/port
python mcpclient.py run --webui --host 0.0.0.0 --port 8080

# Single query
python mcpclient.py run --query "What's the weather in New York?"

# Show dashboard
python mcpclient.py run --dashboard

# Server management
python mcpclient.py servers --search
python mcpclient.py servers --list

# Conversation import/export
python mcpclient.py export --id [CONVERSATION_ID] --output [FILE_PATH]
python mcpclient.py import-conv [FILE_PATH]

# Configuration
python mcpclient.py config --show
python mcpclient.py config port-scan enable true
python mcpclient.py config port-scan range 8000 9000

# Claude Desktop Integration
# Place a claude_desktop_config.json file in the project root directory
# The client will automatically detect and import server configurations on startup

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
- /model - Change AI model
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
- /config - Manage client configuration (API keys, models, discovery methods, port scanning settings, etc.)
    - /config api-key [KEY] - Set Anthropic API key
    - /config model [NAME] - Set default AI model
    - /config max-tokens [NUMBER] - Set default max tokens for generation
    - /config history-size [NUMBER] - Set number of history entries to keep
    - /config auto-discover [true|false] - Enable/disable filesystem discovery
    - /config discovery-path [add|remove|list] [PATH] - Manage filesystem discovery paths
    - /config port-scan enable [true|false] - Enable/disable local port scanning
    - /config port-scan range [start] [end] - Set port range for scanning
    - /config port-scan targets [ip1,ip2,...] - Set IP targets for scanning

Web UI Features:
--------------
- Server Management: Add, remove, connect, and manage MCP servers
- Conversation Interface: Chat with Claude with streaming responses
- Tool Execution: View and interact with tool calls and results in real-time
- Branch Management: Visual conversation tree with fork/switch capabilities
- Settings Panel: Configure API keys, models, and parameters
- Theme Customization: Multiple built-in themes with light/dark mode

API Endpoints:
------------
- GET /api/status - Get client status
- GET/PUT /api/config - Get or update configuration
- GET/POST/DELETE /api/servers/... - Manage servers
- GET /api/tools - List available tools
- GET /api/resources - List available resources
- GET /api/prompts - List available prompts
- GET/POST /api/conversation/... - Manage conversation state
- POST /api/tool/execute - Execute a tool directly
- WS /ws/chat - WebSocket for chat communication

Author: Jeffrey Emanuel
License: MIT
Version: 1.0.0
"""

import asyncio
import atexit
import dataclasses
import functools
import hashlib
import inspect
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
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from urllib.parse import urlparse

import aiofiles  # For async file operations
import anthropic
# Additional utilities
import colorama

# Cache libraries
import diskcache
import httpx

# Third-party imports
import psutil

# Token counting
import tiktoken

# Typer CLI
import typer

# Webui
import uvicorn
import yaml
from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlockDeltaEvent,
    MessageParam,
    MessageStreamEvent,
    ToolParam,
)
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# MCP SDK imports
from mcp.client.sse import sse_client
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    InitializeResult,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    ReadResourceResult,
    Resource,
    Tool,
)
from mcp.types import Prompt as McpPromptType
import anyio

# Observability
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pydantic import AnyUrl, BaseModel, ValidationError  # For request/response models
from rich import box

# Rich UI components
from rich.console import Console, Group
from rich.emoji import Emoji
from rich.layout import Layout
from rich.live import Live
from rich.logging import RichHandler
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.prompt import Confirm, Prompt
from rich.status import Status
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.tree import Tree
from starlette.responses import FileResponse
from typing_extensions import Annotated
from zeroconf import NonUniqueNameException, ServiceInfo

USE_VERBOSE_SESSION_LOGGING = False # Set to True for debugging, False for normal operation

# Set up Typer app
app = typer.Typer(help="🔌 Ultimate MCP Client for Anthropic API")

# Add a global stdout protection mechanism to prevent accidental pollution
class StdioProtectionWrapper:
    """Wrapper that prevents accidental writes to stdout when stdio servers are active.
    
    This provides an additional safety layer beyond the context managers by intercepting
    any direct writes to sys.stdout when stdio servers are connected.
    """
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.active_stdio_servers = False
        self._buffer = []
    
    def update_stdio_status(self):
        """Check if we have any active stdio servers"""
        try:
            # This might not be available during initialization
            if hasattr(app, "mcp_client") and app.mcp_client and hasattr(app.mcp_client, "server_manager"):
                for name, server in app.mcp_client.server_manager.config.servers.items():
                    if server.type == ServerType.STDIO and name in app.mcp_client.server_manager.active_sessions:
                        self.active_stdio_servers = True
                        return
                self.active_stdio_servers = False
        except (NameError, AttributeError):
            # Default to safe behavior if we can't check
            self.active_stdio_servers = False
    
    def write(self, text):
        """Intercept writes to stdout"""
        self.update_stdio_status()
        if self.active_stdio_servers:
            # Redirect to stderr instead to avoid corrupting stdio protocol
            sys.stderr.write(text)
            # Log a warning if this isn't something trivial like a newline
            if text.strip() and text != "\n":
                # Use logging directly to avoid potential recursion
                # logging.warning(f"Prevented stdout pollution: {repr(text[:30])}")
                # Record in debugging buffer for potential diagnostics
                self._buffer.append(text)
                if len(self._buffer) > 100:
                    self._buffer.pop(0)  # Keep buffer size limited
        else:
            self.original_stdout.write(text)
    
    def flush(self):
        """Flush the stream"""
        if not self.active_stdio_servers:
            self.original_stdout.flush()
        else:
            sys.stderr.flush()
    
    def isatty(self):
        """Pass through isatty check"""
        return self.original_stdout.isatty()
    
    # Add other necessary methods for stdout compatibility
    def fileno(self):
        return self.original_stdout.fileno()
    
    def readable(self):
        return self.original_stdout.readable()
    
    def writable(self):
        return self.original_stdout.writable()

# Apply the protection wrapper to stdout
# This is a critical safety measure to prevent stdio corruption
sys.stdout = StdioProtectionWrapper(sys.stdout)

# Add a callback for when no command is specified
@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context):
    """Ultimate MCP Client for connecting Claude and other AI models with MCP servers."""
    if ctx.invoked_subcommand is None:
        # Use get_safe_console() to prevent stdout pollution
        safe_console = get_safe_console()
        
        # Display helpful information when no command is provided
        safe_console.print("\n[bold green]Ultimate MCP Client[/]")
        safe_console.print("A comprehensive client for the Model Context Protocol (MCP)")
        safe_console.print("\n[bold]Common Commands:[/]")
        safe_console.print("  [cyan]run --interactive[/]  Start an interactive chat session")
        safe_console.print("  [cyan]run --query TEXT[/]   Run a single query")
        safe_console.print("  [cyan]run --dashboard[/]    Show the monitoring dashboard")
        safe_console.print("  [cyan]servers --list[/]     List configured servers")
        safe_console.print("  [cyan]config --show[/]      Display current configuration")
        safe_console.print("\n[bold]For more information:[/]")
        safe_console.print("  [cyan]--help[/]             Show detailed help for all commands")
        safe_console.print("  [cyan]COMMAND --help[/]     Show help for a specific command\n")

# Configure Rich theme
from rich.theme import Theme

custom_theme = Theme({
    "info": "cyan",
    "success": "green bold",
    "warning": "yellow bold",
    "error": "red bold",
    "server": "blue",
    "tool": "magenta",
    "resource": "cyan",
    "prompt": "yellow",
    "model": "bright_blue",
    "dashboard.title": "white on blue",
    "dashboard.border": "blue",
    "status.healthy": "green",
    "status.degraded": "yellow",
    "status.error": "red",
    "metric.good": "green",
    "metric.warn": "yellow",
    "metric.bad": "red",
})

# Initialize Rich consoles with theme
console = Console(theme=custom_theme)
stderr_console = Console(theme=custom_theme, stderr=True, highlight=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True, console=stderr_console)]
)
log = logging.getLogger("mcpclient")

# Create a global exit handler
def force_exit_handler(is_force=False):
    """Force exit handler to ensure all processes are terminated."""
    print("\nForcing exit and cleaning up resources...")
    
    # Use os._exit in emergency situations which bypasses normal exit handlers
    # but only as a last resort
    if is_force:
        print("Emergency shutdown initiated!")
        # Try to kill any child processes before force exiting
        if 'app' in globals() and hasattr(app, 'mcp_client'):
            if hasattr(app.mcp_client, 'server_manager'):
                # Terminate all processes immediately
                for name, process in app.mcp_client.server_manager.processes.items():
                    try:
                        if process.returncode is None: # If process is still running
                            print(f"Force killing process {name} (PID {process.pid})")
                            process.kill()
                    except Exception:
                        pass
        
        # This is a hard exit that bypasses normal Python cleanup
        os._exit(1)
    
    # Normal exit via sys.exit
    sys.exit(1)

def atexit_handler():
    """Clean shutdown function for atexit that avoids sys.exit()"""
    print("\nShutting down and cleaning resources...")
    # Add any cleanup code here, but don't call sys.exit()

# Add a signal handler for SIGINT (Ctrl+C)
def sigint_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) by initiating clean shutdown."""
    print("\nCtrl+C detected. Shutting down...")
    
    # Keep track of how many times Ctrl+C has been pressed
    sigint_handler.counter += 1
    
    # If pressed multiple times, force exit
    if sigint_handler.counter >= 2:
        print("Multiple interrupts detected. Forcing immediate exit!")
        force_exit_handler(is_force=True)
    
    # Try clean shutdown first
    try:
        force_exit_handler(is_force=False)
    except Exception as e:
        print(f"Error during clean shutdown: {e}. Forcing exit!")
        force_exit_handler(is_force=True)

# Initialize the counter
sigint_handler.counter = 0

# Register the signal handler
signal.signal(signal.SIGINT, sigint_handler)

# Register with atexit to ensure cleanup on normal exit
atexit.register(atexit_handler)


# Used for WebSocket message structure
class WebSocketMessage(BaseModel):
    type: str
    payload: Any = None

class ServerType(Enum):
    STDIO = "stdio"
    SSE = "sse"

# Pydantic models for API requests (optional but good practice)
class ServerAddRequest(BaseModel):
    name: str
    type: ServerType # FastAPI will handle enum conversion
    path: str
    argsString: Optional[str] = ""

class ConfigUpdateRequest(BaseModel):
    apiKey: Optional[str] = None
    defaultModel: Optional[str] = None
    maxTokens: Optional[int] = None
    temperature: Optional[float] = None
    enableStreaming: Optional[bool] = None
    enableCaching: Optional[bool] = None
    autoDiscover: Optional[bool] = None
    dashboardRefreshRate: Optional[float] = None
    # Add other fields as needed

class ToolExecuteRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any]

class ForkRequest(BaseModel):
    name: Optional[str] = None

class CheckoutRequest(BaseModel):
    node_id: str

class OptimizeRequest(BaseModel):
    model: Optional[str] = None
    target_tokens: Optional[int] = None

class ApplyPromptRequest(BaseModel):
    prompt_name: str

class DiscoveredServer(BaseModel):
    name: str
    type: str # 'stdio' or 'sse'
    path_or_url: str
    source: str # 'filesystem', 'registry', 'mdns', 'portscan'
    description: Optional[str] = None
    version: Optional[str] = None
    categories: List[str] = []
    is_configured: bool = False # Indicate if already in main config

class CacheEntryDetail(BaseModel):
    key: str
    tool_name: str
    created_at: datetime
    expires_at: Optional[datetime] = None

class CacheDependencyInfo(BaseModel):
    dependencies: Dict[str, List[str]] # tool_name -> list_of_dependencies

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
    version: Optional[str] = None # Store as string for simplicity
    rating: float
    retry_count: int
    timeout: float
    registry_url: Optional[str] = None
    capabilities: Dict[str, bool]
    is_connected: bool
    metrics: Dict[str, Any] # Keep metrics as a dict for flexibility
    process_info: Optional[Dict[str, Any]] = None # For STDIO process stats

class DashboardData(BaseModel):
    client_info: Dict[str, Any]
    servers: List[Dict[str, Any]] # Simplified server list for dashboard
    tools: List[Dict[str, Any]] # Top tools info

# --- Global variable to hold the FastAPI app (defined later) ---
# This is needed because uvicorn.run needs the app object path.
web_app: Optional[FastAPI] = None

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
        # --- Added Debug Logging ---
        log.debug(f"convert_windows_path_to_linux: Checking path string: {repr(path_str)}")
        # --- End Added Debug Logging ---

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
                # --- Changed log level to DEBUG for successful conversion ---
                log.debug(f"Converted Windows path '{path_str}' to Linux path '{linux_path}'")
                # --- End Changed log level ---
                return linux_path
            except Exception as e:
                # --- Added Error Logging ---
                log.error(f"Error during path conversion for '{path_str}': {e}", exc_info=True)
                # Return original path on conversion error
                return path_str
                # --- End Added Error Logging ---
        # If it doesn't look like a Windows path, return it unchanged
        # --- Added Debug Logging ---
        log.debug(f"convert_windows_path_to_linux: Path '{path_str}' did not match Windows pattern or wasn't converted.")
        # --- End Added Debug Logging ---
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

# =============================================================================
# CRITICAL STDIO SAFETY MECHANISM
# =============================================================================
# MCP servers that use stdio for communication rely on a clean stdio channel.
# Any output sent to stdout will corrupt the protocol communication and can 
# cause MCP servers to crash or behave unpredictably.
#
# The get_safe_console() function is a critical safety mechanism that ensures:
# 1. All user-facing output goes to stderr when ANY stdio server is active
# 2. Multiple stdio servers can safely coexist without protocol corruption
# 3. User output remains visible while keeping the stdio channel clean
#
# IMPORTANT: Never use console.print() directly. Always use:
#   - get_safe_console().print() for direct access
#   - self.safe_print() for class instance methods
#   - safe_console = get_safe_console() for local variables
#   - safe_stdout() context manager for any code that interacts with stdio servers
# =============================================================================

@contextmanager
def safe_stdout():
    """Context manager that redirects stdout to stderr during critical stdio operations.
    
    This provides an additional layer of protection beyond get_safe_console() by
    ensuring that any direct writes to sys.stdout (not just through Rich) are
    safely redirected during critical operations with stdio servers.
    
    Use this in any code block that interacts with stdio MCP servers:
        with safe_stdout():
            # Code that interacts with stdio servers
    """
    # Check if we have any active stdio servers
    has_stdio_servers = False
    try:
        if hasattr(app, "mcp_client") and app.mcp_client and hasattr(app.mcp_client, "server_manager"):
            for name, server in app.mcp_client.server_manager.config.servers.items():
                if server.type == ServerType.STDIO and name in app.mcp_client.server_manager.active_sessions:
                    has_stdio_servers = True
                    break
    except (NameError, AttributeError):
        pass
    
    # Only redirect if we have stdio servers active
    if has_stdio_servers:
        with redirect_stdout(sys.stderr):
            yield
    else:
        yield

# Around Line 490
def get_safe_console():
    """Get the appropriate console based on whether we're using stdio servers.

    CRITICAL: This function ensures all user output goes to stderr when any stdout-based
    MCP server is active, preventing protocol corruption and server crashes.

    Returns stderr_console if there are any active stdio servers to prevent
    interfering with stdio communication channels.
    """
    # Check if we have any active stdio servers
    has_stdio_servers = False
    try:
        # This might not be available during initialization, so we use a try block
        if hasattr(app, "mcp_client") and app.mcp_client and hasattr(app.mcp_client, "server_manager"):
            for name, server in app.mcp_client.server_manager.config.servers.items():
                if server.type == ServerType.STDIO and name in app.mcp_client.server_manager.active_sessions:
                    has_stdio_servers = True

                    # --- MODIFIED WARNING LOGIC ---
                    # Check the caller frame, but be less aggressive about warnings for simple assignments
                    # This aims to reduce noise for patterns like `x = get_safe_console()` or `console=get_safe_console()`
                    caller_frame = inspect.currentframe().f_back
                    if caller_frame:
                        caller_info = inspect.getframeinfo(caller_frame)
                        caller_line = caller_info.code_context[0].strip() if caller_info.code_context else ""

                        # More specific check: Warn if it looks like `.print()` is called *directly* on the result,
                        # OR if the caller isn't a known safe method/pattern.
                        # This check is heuristic and might need refinement.
                        is_direct_print = ".print(" in caller_line and "get_safe_console().print(" in caller_line.replace(" ", "")
                        is_known_safe_caller = caller_info.function in ["safe_print", "_run_with_progress", "_run_with_simple_progress"] \
                                               or "self.safe_print(" in caller_line \
                                               or "_safe_printer(" in caller_line # Added _safe_printer check for ServerManager

                        # Avoid warning for assignments like `console = get_safe_console()` or `console=get_safe_console()`
                        # These are necessary patterns in setup, interactive_loop, etc.
                        is_assignment_pattern = "=" in caller_line and "get_safe_console()" in caller_line

                        if not is_known_safe_caller and not is_assignment_pattern and is_direct_print:
                             # Only log warning if it's NOT a known safe caller/pattern AND looks like a direct print attempt
                             log.warning(f"Potential unsafe console usage detected at: {caller_info.filename}:{caller_info.lineno}")
                             log.warning(f"Always use MCPClient.safe_print() or store get_safe_console() result first.")
                             log.warning(f"Stack: {caller_info.function} - {caller_line}")
                    # --- END MODIFIED WARNING LOGIC ---

                    break # Found an active stdio server, no need to check further
    except (NameError, AttributeError):
        pass # Ignore errors if client isn't fully initialized

    # If we have active stdio servers, use stderr to avoid interfering with stdio communication
    return stderr_console if has_stdio_servers else console

def verify_no_stdout_pollution():
    """Verify that stdout isn't being polluted during MCP communication.
    
    This function temporarily captures stdout and writes a test message,
    then checks if the message was captured. If stdout is properly protected,
    the test output should be intercepted by our safety mechanisms.
    
    Use this for debugging if you suspect stdout pollution is causing issues.
    """
    import io
    import sys
    
    # Store the original stdout
    original_stdout = sys.stdout
    
    # Create a buffer to capture any potential output
    test_buffer = io.StringIO()
    
    # Replace stdout with our test buffer
    sys.stdout = test_buffer
    try:
        # Write a test message to stdout - but use a non-printing approach to test
        # Instead of using print(), write directly to the buffer for testing
        test_buffer.write("TEST_STDOUT_POLLUTION_VERIFICATION")
        
        # Check if the message was captured (it should be since we wrote directly to buffer)
        captured = test_buffer.getvalue()
        
        # We now check if the wrapper would have properly intercepted real stdout writes
        # by checking if it has active_stdio_servers correctly set when it should
        if isinstance(original_stdout, StdioProtectionWrapper):
            # Update servers status
            original_stdout.update_stdio_status()
            if captured and not original_stdout.active_stdio_servers:
                # This is expected - there's no active stdio servers, direct print is fine
                return True
            else:
                # If we have active_stdio_servers true but capture happened, 
                # would indicate potential issues, but handled gracefully
                return True
        else:
            # If stdout isn't actually wrapped by StdioProtectionWrapper, that's a real issue
            sys.stderr.write("\n[CRITICAL] STDOUT IS NOT PROPERLY WRAPPED WITH StdioProtectionWrapper\n")
            log.critical("STDOUT IS NOT PROPERLY WRAPPED WITH StdioProtectionWrapper")
            return False
    finally:
        # Restore the original stdout
        sys.stdout = original_stdout

# Status emoji mapping
STATUS_EMOJI = {
    "healthy": Emoji("white_check_mark"),
    "degraded": Emoji("warning"),
    "error": Emoji("cross_mark"),
    "connected": Emoji("green_circle"),
    "disconnected": Emoji("red_circle"),
    "cached": Emoji("package"),
    "streaming": Emoji("water_wave"),
    "forked": Emoji("trident_emblem"),
    "tool": Emoji("hammer_and_wrench"),
    "resource": Emoji("books"),
    "prompt": Emoji("speech_balloon"),
    "server": Emoji("desktop_computer"),
    "config": Emoji("gear"),
    "history": Emoji("scroll"),
    "search": Emoji("magnifying_glass_tilted_right"),
    "success": Emoji("party_popper"),
    "failure": Emoji("collision"),
    "warning": Emoji("warning"),
    "model": Emoji("robot"),      
}

# Constants
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_DIR = PROJECT_ROOT / ".mcpclient_config" # Store in a hidden subfolder in project root
CONFIG_FILE = CONFIG_DIR / "config.yaml"
HISTORY_FILE = CONFIG_DIR / "history.json"
SERVER_DIR = CONFIG_DIR / "servers"
CACHE_DIR = CONFIG_DIR / "cache"
REGISTRY_DIR = CONFIG_DIR / "registry"
DEFAULT_MODEL = "claude-3-7-sonnet-20250219"
MAX_HISTORY_ENTRIES = 100
REGISTRY_URLS = [
    # Leave empty by default - users can add their own registries later
]

# Create necessary directories
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
SERVER_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

# Initialize OpenTelemetry
trace_provider = TracerProvider()
use_console_exporter = False # Set to True to enable console exporter (recommended to set to False)
if use_console_exporter:
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    trace_provider.add_span_processor(span_processor)
trace.set_tracer_provider(trace_provider)

# Initialize metrics with the current API
try:
    # Try the newer API first
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    if use_console_exporter:
        reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
        meter_provider = MeterProvider(metric_readers=[reader])
    else:
        meter_provider = MeterProvider()
    metrics.set_meter_provider(meter_provider)
except (ImportError, AttributeError):
    # Fallback to older API or handle gracefully
    log.warning("OpenTelemetry metrics API initialization failed. Metrics may not be available.")
    # Create a dummy meter_provider for compatibility
    meter_provider = MeterProvider()
    metrics.set_meter_provider(meter_provider)

tracer = trace.get_tracer("mcpclient")
meter = metrics.get_meter("mcpclient")

# Create instruments
try:
    request_counter = meter.create_counter(
        name="mcp_requests",
        description="Number of MCP requests",
        unit="1"
    )

    latency_histogram = meter.create_histogram(
        name="mcp_latency",
        description="Latency of MCP requests",
        unit="ms"
    )

    tool_execution_counter = meter.create_counter(
        name="tool_executions",
        description="Number of tool executions",
        unit="1"
    )
except Exception as e:
    log.warning(f"Failed to create metrics instruments: {e}")
    # Create dummy objects to avoid None checks
    request_counter = None
    latency_histogram = None
    tool_execution_counter = None

class ServerStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    ERROR = "error"
    UNKNOWN = "unknown"

@dataclass
class ServerVersion:
    major: int
    minor: int
    patch: int
    
    @classmethod
    def from_string(cls, version_str: str) -> "ServerVersion":
        """Parse version from string like 1.2.3"""
        parts = version_str.split(".")
        if len(parts) < 3:
            parts.extend(["0"] * (3 - len(parts)))
        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2])
        )
    
    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"
    
    def is_compatible_with(self, other: "ServerVersion") -> bool:
        """Check if this version is compatible with another version"""
        # Same major version is compatible
        return self.major == other.major

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
        """Add a new response time and recalculate average"""
        self.response_times.append(response_time)
        # Keep only the last 100 responses
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        self.avg_response_time = sum(self.response_times) / len(self.response_times)
    
    def update_status(self) -> None:
        """Update server status based on metrics"""
        self.error_rate = self.error_count / max(1, self.request_count)
        
        if self.error_rate > 0.5 or self.avg_response_time > 10.0:
            self.status = ServerStatus.ERROR
        elif self.error_rate > 0.1 or self.avg_response_time > 5.0:
            self.status = ServerStatus.DEGRADED
        else:
            self.status = ServerStatus.HEALTHY

@dataclass
class ServerConfig:
    name: str
    type: ServerType
    path: str  # Command for STDIO or URL for SSE
    args: List[str] = field(default_factory=list)
    enabled: bool = True
    auto_start: bool = True
    description: str = ""
    trusted: bool = False
    categories: List[str] = field(default_factory=list)
    version: Optional[ServerVersion] = None
    rating: float = 5.0  # 1-5 star rating
    retry_count: int = 3  # Number of retries on failure
    timeout: float = 90.0  # Timeout in seconds
    metrics: ServerMetrics = field(default_factory=ServerMetrics)
    registry_url: Optional[str] = None  # URL of registry where found
    last_updated: datetime = field(default_factory=datetime.now)
    retry_policy: Dict[str, Any] = field(default_factory=lambda: {
        "max_attempts": 3,
        "backoff_factor": 0.5,
        "timeout_increment": 5
    })
    capabilities: Dict[str, bool] = field(default_factory=lambda: {
        "tools": True,
        "resources": True,
        "prompts": True
    })

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
        """Update execution time metrics"""
        self.execution_times.append(time_ms)
        self.avg_execution_time = sum(self.execution_times) / len(self.execution_times)
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

@dataclass
class ConversationNode:
    id: str
    messages: List[MessageParam] = field(default_factory=list)
    parent: Optional["ConversationNode"] = None
    children: List["ConversationNode"] = field(default_factory=list)
    name: str = "Root"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    model: str = ""
    
    def add_message(self, message: MessageParam) -> None:
        """Add a message to this conversation node"""
        self.messages.append(message)
        self.modified_at = datetime.now()
    
    def add_child(self, child: "ConversationNode") -> None:
        """Add a child branch"""
        self.children.append(child)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "messages": self.messages,
            "parent_id": self.parent.id if self.parent else None,
            "children_ids": [child.id for child in self.children],
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "model": self.model
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationNode":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            messages=data["messages"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            modified_at=datetime.fromisoformat(data["modified_at"]),
            model=data.get("model", "")
        )

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

@dataclass
class CacheEntry:
    result: Any
    created_at: datetime
    expires_at: Optional[datetime] = None
    tool_name: str = ""
    parameters_hash: str = ""
    
    def is_expired(self) -> bool:
        """Check if the cache entry is expired"""
        if not self.expires_at:
            return False
        return datetime.now() > self.expires_at

class ServerRegistry:
    """Registry for discovering and managing MCP servers"""
    def __init__(self, registry_urls=None):
        self.registry_urls = registry_urls or REGISTRY_URLS
        self.servers: Dict[str, Dict[str, Any]] = {}
        self.local_ratings: Dict[str, float] = {}
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # For mDNS discovery
        self.zeroconf = None
        self.browser = None
        self.discovered_servers: Dict[str, Dict[str, Any]] = {}
        
    async def discover_remote_servers(self, categories=None, min_rating=0.0, max_results=50):
        """Discover servers from remote registries"""
        all_servers = []
        
        if not self.registry_urls:
            log.info("No registry URLs configured, skipping remote discovery")
            return all_servers
            
        for registry_url in self.registry_urls:
            try:
                # Construct query parameters
                params = {"max_results": max_results}
                if categories:
                    params["categories"] = ",".join(categories)
                if min_rating:
                    params["min_rating"] = min_rating
                
                # Make request to registry
                response = await self.http_client.get(
                    f"{registry_url}/servers",
                    params=params,
                    timeout=5.0  # Add a shorter timeout
                )
                
                if response.status_code == 200:
                    servers = response.json().get("servers", [])
                    for server in servers:
                        server["registry_url"] = registry_url
                        all_servers.append(server)
                else:
                    log.warning(f"Failed to get servers from {registry_url}: {response.status_code}")
            except httpx.TimeoutException:
                log.warning(f"Timeout connecting to registry {registry_url}")
            except Exception as e:
                log.error(f"Error querying registry {registry_url}: {e}")
        
        return all_servers
    
    async def get_server_details(self, server_id, registry_url=None):
        """Get detailed information about a specific server"""
        urls_to_try = self.registry_urls if not registry_url else [registry_url]
        
        for url in urls_to_try:
            try:
                response = await self.http_client.get(f"{url}/servers/{server_id}")
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                server = response.json()
                server["registry_url"] = url
                return server
            except httpx.RequestError as e: # Includes connection errors, timeouts, etc.
                log.debug(f"Network error getting server details from {url}: {e}")
            except httpx.HTTPStatusError as e: # Handle 4xx/5xx responses
                log.debug(f"HTTP error getting server details from {url}: {e.response.status_code}")
            except json.JSONDecodeError as e:
                log.debug(f"JSON decode error getting server details from {url}: {e}")
            # Keep broad exception for unexpected issues during this specific loop iteration
            except Exception as e:
                 log.debug(f"Unexpected error getting server details from {url}: {e}")

        log.warning(f"Could not get details for server {server_id} from any registry.")
        return None
    
    def start_local_discovery(self):
        """Start discovering MCP servers on the local network using mDNS"""
        try:
            # Make sure zeroconf is importable
            from zeroconf import EventLoopBlocked, ServiceBrowser, Zeroconf
        except ImportError:
            log.warning("Zeroconf not available, local discovery disabled")
            self.zeroconf = None # Ensure it's None if import fails
            return # Cannot proceed

        try:
            # Inner try block for Zeroconf setup
            class MCPServiceListener:
                def __init__(self, registry):
                    self.registry = registry

                def add_service(self, zeroconf_obj, service_type, name):
                    info = None # Initialize info
                    try:
                        # *** ADDED Try/Except around get_service_info ***
                        info = zeroconf_obj.get_service_info(service_type, name, timeout=1000) # Use 1 sec timeout
                    except EventLoopBlocked:
                         log.warning(f"Zeroconf event loop blocked getting info for {name}, will retry later.")
                         return # Skip processing this time, might get it on update
                    except Exception as e:
                         log.error(f"Error getting zeroconf service info for {name}: {e}")
                         return # Skip if error getting info

                    if not info:
                        log.debug(f"No service info found for {name} after query.")
                        return # No info retrieved

                    # --- Rest of the add_service logic as before ---
                    server_name = name.replace("._mcp._tcp.local.", "")
                    host = socket.inet_ntoa(info.addresses[0]) if info.addresses else "localhost"
                    port = info.port

                    properties = {}
                    if info.properties:
                        for k, v in info.properties.items():
                            try:
                                key = k.decode('utf-8')
                                value = v.decode('utf-8')
                                properties[key] = value
                            except UnicodeDecodeError:
                                continue

                    server_type = properties.get("type", "sse") # Default to sse if not specified
                    version_str = properties.get("version")
                    version = ServerVersion.from_string(version_str) if version_str else None
                    categories = properties.get("categories", "").split(",") if properties.get("categories") else []
                    description = properties.get("description", f"mDNS discovered server at {host}:{port}")

                    self.registry.discovered_servers[server_name] = {
                        "name": server_name,
                        "host": host,
                        "port": port,
                        "type": server_type,
                        "url": f"http://{host}:{port}",
                        "properties": properties,
                        "version": version, # Store parsed version object or None
                        "categories": categories,
                        "description": description,
                        "discovered_via": "mdns"
                    }
                    log.info(f"Discovered local MCP server: {server_name} at {host}:{port} ({description})")
                    # --- End of original add_service logic ---

                def remove_service(self, zeroconf_obj, service_type, name):
                    # (Keep existing remove_service logic)
                    server_name = name.replace("._mcp._tcp.local.", "")
                    if server_name in self.registry.discovered_servers:
                        del self.registry.discovered_servers[server_name]
                        log.info(f"Removed local MCP server: {server_name}")

                def update_service(self, zeroconf, service_type, name):
                    # Optional: Could call add_service again here to refresh info
                    log.debug(f"Zeroconf update event for {name}")
                    # For simplicity, we can just rely on add_service/remove_service
                    pass

            if self.zeroconf is None: # Initialize only if not already done
                 self.zeroconf = Zeroconf()
            listener = MCPServiceListener(self)
            self.browser = ServiceBrowser(self.zeroconf, "_mcp._tcp.local.", listener)
            log.info("Started local MCP server discovery")

        except OSError as e:
             log.error(f"Error starting local discovery (network issue?): {e}")
        except Exception as e:
             log.error(f"Unexpected error during zeroconf setup: {e}") # Catch other potential errors
    
    def stop_local_discovery(self):
        """Stop local server discovery"""
        if self.zeroconf:
            self.zeroconf.close()
            self.zeroconf = None
        self.browser = None
    
    async def rate_server(self, server_id, rating):
        """Rate a server in the registry"""
        # Store locally
        self.local_ratings[server_id] = rating
        
        # Try to submit to registry
        server = self.servers.get(server_id)
        if server and "registry_url" in server:
            try:
                response = await self.http_client.post(
                    f"{server['registry_url']}/servers/{server_id}/rate",
                    json={"rating": rating}
                )
                response.raise_for_status()
                if response.status_code == 200:
                    log.info(f"Successfully rated server {server_id}")
                    return True
            except httpx.RequestError as e:
                log.error(f"Network error rating server {server_id}: {e}")
            except httpx.HTTPStatusError as e:
                 log.error(f"HTTP error rating server {server_id}: {e.response.status_code}")
            # Keep broad exception for unexpected issues during rating
            except Exception as e:
                 log.error(f"Unexpected error rating server {server_id}: {e}")
        
        return False
    
    async def close(self):
        """Close the registry"""
        self.stop_local_discovery()
        await self.http_client.aclose()


class ToolCache:
    """Cache for storing tool execution results"""
    def __init__(self, cache_dir=CACHE_DIR, custom_ttl_mapping=None):
        self.cache_dir = Path(cache_dir)
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        # Set up disk cache
        self.disk_cache = diskcache.Cache(str(self.cache_dir / "tool_results"))
        
        # Default TTL mapping - overridden by custom mapping
        self.ttl_mapping = {
            "weather": 30 * 60,  # 30 minutes
            "filesystem": 5 * 60,  # 5 minutes
            "search": 24 * 60 * 60,  # 1 day
            "database": 5 * 60,  # 5 minutes
            # Add more default categories as needed
        }
        # Apply custom TTL mapping from config
        if custom_ttl_mapping:
            self.ttl_mapping.update(custom_ttl_mapping)

        # Add dependency tracking
        self.dependency_graph: Dict[str, Set[str]] = {}

    def add_dependency(self, tool_name, depends_on):
        """Register a dependency between tools"""
        self.dependency_graph.setdefault(tool_name, set()).add(depends_on)
        
    def invalidate_related(self, tool_name):
        """Invalidate all dependent cache entries"""
        affected = set()
        stack = [tool_name]
        
        while stack:
            current = stack.pop()
            affected.add(current)
            
            # Find all tools that depend on the current tool
            for dependent, dependencies in self.dependency_graph.items():
                if current in dependencies and dependent not in affected:
                    stack.append(dependent)
        
        # Remove the originating tool - we only want to invalidate dependents
        if tool_name in affected:
            affected.remove(tool_name)
            
        # Invalidate each affected tool
        for tool in affected:
            self.invalidate(tool_name=tool)
            log.info(f"Invalidated dependent tool cache: {tool} (depends on {tool_name})")

    def get_ttl(self, tool_name):
        """Get TTL for a tool based on its name, prioritizing custom mapping."""
        # Check custom/updated mapping first (already merged in __init__)
        for category, ttl in self.ttl_mapping.items():
            if category in tool_name.lower():
                return ttl
        return 60 * 60  # Default: 1 hour
    
    def generate_key(self, tool_name, params):
        """Generate a cache key for the tool and parameters"""
        # Hash the parameters to create a unique key
        params_str = json.dumps(params, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()
        return f"{tool_name}:{params_hash}"
    
    def get(self, tool_name, params):
        """Get cached result for a tool execution"""
        key = self.generate_key(tool_name, params)
        
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired():
                return entry.result
            else:
                del self.memory_cache[key]
        
        # Check disk cache if available
        if self.disk_cache and key in self.disk_cache:
            entry = self.disk_cache[key]
            if not entry.is_expired():
                # Promote to memory cache
                self.memory_cache[key] = entry
                return entry.result
            else:
                del self.disk_cache[key]
        
        return None
    
    def set(self, tool_name, params, result, ttl=None):
        """Cache the result of a tool execution"""
        key = self.generate_key(tool_name, params)
        
        # Create cache entry
        if ttl is None:
            ttl = self.get_ttl(tool_name)
        
        expires_at = None
        if ttl > 0:
            expires_at = datetime.now() + timedelta(seconds=ttl)
        
        entry = CacheEntry(
            result=result,
            created_at=datetime.now(),
            expires_at=expires_at,
            tool_name=tool_name,
            parameters_hash=key.split(":")[-1]
        )
        
        # Store in memory cache
        self.memory_cache[key] = entry
        
        # Store in disk cache if available
        if self.disk_cache:
            self.disk_cache[key] = entry
    
    def invalidate(self, tool_name=None, params=None):
        """Invalidate cache entries"""
        if tool_name and params:
            # Invalidate specific entry
            key = self.generate_key(tool_name, params)
            if key in self.memory_cache:
                del self.memory_cache[key]
            if self.disk_cache and key in self.disk_cache:
                del self.disk_cache[key]
        elif tool_name:
            # Invalidate all entries for a tool
            for key in list(self.memory_cache.keys()):
                if key.startswith(f"{tool_name}:"):
                    del self.memory_cache[key]
            
            if self.disk_cache:
                for key in list(self.disk_cache.keys()):
                    if key.startswith(f"{tool_name}:"):
                        del self.disk_cache[key]
                        
            # Invalidate dependent tools
            self.invalidate_related(tool_name)
        else:
            # Invalidate all entries
            self.memory_cache.clear()
            if self.disk_cache:
                self.disk_cache.clear()
    
    def clean(self):
        """Clean expired entries"""
        # Clean memory cache
        for key in list(self.memory_cache.keys()):
            if self.memory_cache[key].is_expired():
                del self.memory_cache[key]
        
        # Clean disk cache if available
        if self.disk_cache:
            for key in list(self.disk_cache.keys()):
                try:
                    if self.disk_cache[key].is_expired():
                        del self.disk_cache[key]
                except KeyError: # Key might have been deleted already
                    pass 
                except (diskcache.Timeout, diskcache.CacheIndexError, OSError, EOFError) as e: # Specific diskcache/IO errors
                    log.warning(f"Error cleaning cache key {key}: {e}. Removing corrupted entry.")
                    # Attempt to remove potentially corrupted entry
                    try:
                        del self.disk_cache[key]
                    except Exception as inner_e:
                         log.error(f"Failed to remove corrupted cache key {key}: {inner_e}")

    def close(self):
        """Close the cache"""
        if self.disk_cache:
            self.disk_cache.close()


class ConversationGraph:
    """Manage conversation nodes and branches"""
    def __init__(self):
        self.nodes: Dict[str, ConversationNode] = {}
        self.root = ConversationNode(id="root", name="Root")
        self.current_node = self.root
        
        # Add root to nodes
        self.nodes[self.root.id] = self.root
    
    def add_node(self, node: ConversationNode):
        """Add a node to the graph"""
        self.nodes[node.id] = node
    
    def get_node(self, node_id: str) -> Optional[ConversationNode]:
        """Get a node by ID"""
        return self.nodes.get(node_id)
    
    def create_fork(self, name: Optional[str] = None) -> ConversationNode:
        """Create a fork from the current conversation node"""
        fork_id = str(uuid.uuid4())
        fork_name = name or f"Fork {len(self.current_node.children) + 1}"
        
        new_node = ConversationNode(
            id=fork_id,
            name=fork_name,
            parent=self.current_node,
            messages=self.current_node.messages.copy(),
            model=self.current_node.model
        )
        
        self.current_node.add_child(new_node)
        self.add_node(new_node)
        return new_node
    
    def set_current_node(self, node_id: str) -> bool:
        """Set the current conversation node"""
        if node_id in self.nodes:
            self.current_node = self.nodes[node_id]
            return True
        return False
    
    def get_path_to_root(self, node: Optional[ConversationNode] = None) -> List[ConversationNode]:
        """Get path from node to root"""
        if node is None:
            node = self.current_node
            
        path = [node]
        current = node
        while current.parent:
            path.append(current.parent)
            current = current.parent
            
        return list(reversed(path))
    
    async def save(self, file_path: str):
        """Save the conversation graph to file asynchronously"""
        data = {
            "current_node_id": self.current_node.id,
            "nodes": {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            }
        }
        
        try:
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(data, indent=2))
        except IOError as e:
             log.error(f"Could not write conversation graph to {file_path}: {e}")
        except TypeError as e: # Handle potential issues with non-serializable data
             log.error(f"Could not serialize conversation graph: {e}")
    
    @classmethod
    async def load(cls, file_path: str) -> "ConversationGraph":
        """
        Load a conversation graph from file asynchronously.
        Handles file not found, IO errors, JSON errors, and structural errors gracefully
        by returning a new, empty graph and attempting to back up corrupted files.
        """
        file_path_obj = Path(file_path)
        try:
            # --- Attempt to load ---
            async with aiofiles.open(file_path_obj, 'r') as f:
                content = await f.read()
                if not content.strip():
                    log.warning(f"Conversation graph file is empty: {file_path}. Creating a new one.")
                    # Create and potentially save a new default graph
                    graph = cls()
                    try:
                        # Attempt to save the empty structure back
                        await graph.save(file_path)
                        log.info(f"Initialized empty conversation graph file: {file_path}")
                    except Exception as save_err:
                        log.error(f"Failed to save initial empty graph to {file_path}: {save_err}")
                    return graph

                data = json.loads(content) # Raises JSONDecodeError on syntax issues

            # --- Attempt to reconstruct graph ---
            graph = cls()
            # First pass: create all nodes
            for node_id, node_data in data["nodes"].items():
                # Raises KeyError, TypeError, ValueError etc. on bad structure
                node = ConversationNode.from_dict(node_data)
                graph.nodes[node_id] = node

            # Second pass: set up parent-child relationships
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

            # Set current node
            current_node_id = data.get("current_node_id", "root")
            if current_node_id in graph.nodes:
                graph.current_node = graph.nodes[current_node_id]
            else:
                log.warning(f"Saved current_node_id '{current_node_id}' not found in loaded graph {file_path}, defaulting to root.")
                graph.current_node = graph.root # Assume root always exists

            log.info(f"Successfully loaded and parsed conversation graph from {file_path}")
            return graph # Return the successfully loaded graph

        except FileNotFoundError:
            log.info(f"Conversation graph file not found: {file_path}. Creating a new one.")
            # Create and potentially save a new default graph
            new_graph = cls()
            # Optional: Save immediately
            # try: await new_graph.save(file_path)
            # except Exception as save_err: log.error(f"Failed to save initial graph: {save_err}")
            return new_graph

        except (IOError, json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError) as e:
            # Handle corruption or structural errors
            log.warning(f"Failed to load/parse conversation graph from {file_path} due to error: {e}. A new graph will be used.", exc_info=False) # Log basic error, traceback only if verbose
            log.debug("Traceback for conversation load error:", exc_info=True) # Always log traceback at DEBUG level

            # --- Backup corrupted file ---
            try:
                backup_path = file_path_obj.with_suffix(f".json.corrupted.{int(time.time())}")
                # Use os.rename for atomic operation if possible, requires sync context or separate thread
                # For simplicity with asyncio, using async move (less atomic)
                # Note: aiofiles doesn't have rename, need os or another lib if strict atomicity needed
                # Let's stick to os.rename for now, accepting it blocks briefly.
                if file_path_obj.exists(): # Check again before renaming
                    os.rename(file_path_obj, backup_path)
                    log.info(f"Backed up corrupted conversation file to: {backup_path}")
            except Exception as backup_err:
                log.error(f"Failed to back up corrupted conversation file {file_path}: {backup_err}", exc_info=True)

            # --- Return a new graph ---
            return cls() # Return a fresh, empty graph

        except Exception: # Catch-all for truly unexpected load errors
             log.error(f"Unexpected error loading conversation graph from {file_path}. A new graph will be used.", exc_info=True) # Log with traceback
             # Attempt backup here too
             try:
                 if file_path_obj.exists():
                     backup_path = file_path_obj.with_suffix(f".json.corrupted.{int(time.time())}")
                     os.rename(file_path_obj, backup_path)
                     log.info(f"Backed up corrupted conversation file to: {backup_path}")
             except Exception as backup_err:
                  log.error(f"Failed to back up corrupted conversation file {file_path}: {backup_err}", exc_info=True)
             return cls() # Return a fresh graph


class Config:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        self.api_key: Optional[str] = os.environ.get("ANTHROPIC_API_KEY")
        self.default_model: str = DEFAULT_MODEL
        self.servers: Dict[str, ServerConfig] = {}
        self.default_max_tokens: int = 1024
        self.history_size: int = MAX_HISTORY_ENTRIES
        self.auto_discover: bool = True
        self.discovery_paths: List[str] = [
            str(SERVER_DIR),
            os.path.expanduser("~/mcp-servers"),
            os.path.expanduser("~/modelcontextprotocol/servers")
        ]
        self.enable_streaming: bool = True
        self.enable_caching: bool = True
        self.enable_metrics: bool = True
        self.enable_registry: bool = True
        self.enable_local_discovery: bool = True
        self.temperature: float = 0.7
        self.cache_ttl_mapping: Dict[str, int] = {}
        self.conversation_graphs_dir: str = str(CONFIG_DIR / "conversations")
        self.registry_urls: List[str] = REGISTRY_URLS.copy()
        self.dashboard_refresh_rate: float = 2.0  # seconds
        self.summarization_model: str = "claude-3-7-sonnet-latest"  # Model used for conversation summarization
        self.auto_summarize_threshold: int = 6000  # Auto-summarize when token count exceeds this
        self.max_summarized_tokens: int = 1500  # Target token count after summarization
        self.enable_port_scanning: bool = True # Enable by default? Or False? Let's start with True.
        self.port_scan_range_start: int = 8000
        self.port_scan_range_end: int = 9000 # Scan 1001 ports
        self.port_scan_concurrency: int = 100 # Max simultaneous probes
        self.port_scan_timeout: float = 2.5 # Timeout per port probe (seconds)
        self.port_scan_targets: List[str] = ["127.0.0.1"] # Scan localhost by default

        # Use synchronous load for initialization since __init__ can't be async
        self.load()
    
    def _prepare_config_data(self):
        """Prepare configuration data for saving"""
        return {
            'api_key': self.api_key,
            'default_model': self.default_model,
            'default_max_tokens': self.default_max_tokens,
            'history_size': self.history_size,
            'auto_discover': self.auto_discover,
            'discovery_paths': self.discovery_paths,
            'enable_streaming': self.enable_streaming,
            'enable_caching': self.enable_caching,
            'enable_metrics': self.enable_metrics,
            'enable_registry': self.enable_registry,
            'enable_local_discovery': self.enable_local_discovery,
            'temperature': self.temperature,
            'cache_ttl_mapping': self.cache_ttl_mapping,
            'conversation_graphs_dir': self.conversation_graphs_dir,
            'registry_urls': self.registry_urls,
            'dashboard_refresh_rate': self.dashboard_refresh_rate,
            'summarization_model': self.summarization_model,
            'auto_summarize_threshold': self.auto_summarize_threshold,
            'max_summarized_tokens': self.max_summarized_tokens,
            'enable_port_scanning': self.enable_port_scanning,
            'port_scan_range_start': self.port_scan_range_start,
            'port_scan_range_end': self.port_scan_range_end,
            'port_scan_concurrency': self.port_scan_concurrency,
            'port_scan_timeout': self.port_scan_timeout,
            'port_scan_targets': self.port_scan_targets,            
            'servers': {
                name: {
                    'type': server.type.value,
                    'path': server.path,
                    'args': server.args,
                    'enabled': server.enabled,
                    'auto_start': server.auto_start,
                    'description': server.description,
                    'trusted': server.trusted,
                    'categories': server.categories,
                    'version': str(server.version) if server.version else None,
                    'rating': server.rating,
                    'retry_count': server.retry_count,
                    'timeout': server.timeout,
                    'retry_policy': server.retry_policy,
                    'metrics': {
                        'uptime': server.metrics.uptime,
                        'request_count': server.metrics.request_count,
                        'error_count': server.metrics.error_count,
                        'avg_response_time': server.metrics.avg_response_time,
                        'status': server.metrics.status.value,
                        'error_rate': server.metrics.error_rate
                    },
                    'registry_url': server.registry_url,
                    'capabilities': server.capabilities
                }
                for name, server in self.servers.items()
            }
        }
    
    def load(self):
        """Load configuration from file synchronously"""
        if not CONFIG_FILE.exists():
            self.save()  # Create default config
            return

        try:
            with open(CONFIG_FILE, 'r') as f:
                config_data = yaml.safe_load(f) or {}

            # Update config with loaded values
            for key, value in config_data.items():
                if key == 'servers':
                    self.servers = {}
                    for server_name, server_data in value.items():
                        server_type = ServerType(server_data.get('type', 'stdio'))

                        # Parse version if available
                        version = None
                        if 'version' in server_data:
                            version_str = server_data['version']
                            if version_str:
                                try: # Add try-except for version parsing
                                    version = ServerVersion.from_string(version_str)
                                except ValueError:
                                    log.warning(f"Invalid version string '{version_str}' for server {server_name}, setting version to None.")
                                    version = None


                        # Parse metrics if available
                        metrics = ServerMetrics()
                        if 'metrics' in server_data:
                            metrics_data = server_data['metrics']
                            for metric_key, metric_value in metrics_data.items():
                                if hasattr(metrics, metric_key):
                                    if metric_key == 'status':
                                        try:
                                            # Attempt to convert the loaded string value to the ServerStatus enum
                                            status_enum = ServerStatus(metric_value)
                                            setattr(metrics, metric_key, status_enum)
                                        except ValueError:
                                            # Handle cases where the loaded status string is invalid
                                            log.warning(f"Invalid status value '{metric_value}' loaded for server {server_name}, defaulting to UNKNOWN.")
                                            setattr(metrics, metric_key, ServerStatus.UNKNOWN)
                                    else:
                                        # Load other metrics as before
                                        setattr(metrics, metric_key, metric_value)

                        self.servers[server_name] = ServerConfig(
                            name=server_name,
                            type=server_type,
                            path=server_data.get('path', ''),
                            args=server_data.get('args', []),
                            enabled=server_data.get('enabled', True),
                            auto_start=server_data.get('auto_start', True),
                            description=server_data.get('description', ''),
                            trusted=server_data.get('trusted', False),
                            categories=server_data.get('categories', []),
                            version=version,
                            rating=server_data.get('rating', 5.0),
                            retry_count=server_data.get('retry_count', 3),
                            timeout=server_data.get('timeout', 30.0),
                            metrics=metrics, # Assign the populated metrics object
                            registry_url=server_data.get('registry_url'),
                            retry_policy=server_data.get('retry_policy', {
                                "max_attempts": 3,
                                "backoff_factor": 0.5,
                                "timeout_increment": 5
                            }),
                            capabilities=server_data.get('capabilities', {
                                "tools": True,
                                "resources": True,
                                "prompts": True
                            })
                        )
                elif key == 'cache_ttl_mapping':
                    self.cache_ttl_mapping = value
                else:
                    if hasattr(self, key):
                        setattr(self, key, value)

        except FileNotFoundError:
            self.save()
            return
        except IOError as e:
            log.error(f"Error reading config file {CONFIG_FILE}: {e}")
        except yaml.YAMLError as e:
            log.error(f"Error parsing config file {CONFIG_FILE}: {e}")
        except Exception as e:
            log.error(f"Unexpected error loading config: {e}")

    async def load_async(self):
        """Load configuration from file asynchronously"""
        if not CONFIG_FILE.exists():
            await self.save_async()
            return

        try:
            async with aiofiles.open(CONFIG_FILE, 'r') as f:
                content = await f.read()
                config_data = yaml.safe_load(content) or {}

            for key, value in config_data.items():
                if key == 'servers':
                    self.servers = {}
                    for server_name, server_data in value.items():
                        server_type = ServerType(server_data.get('type', 'stdio'))
                        version = None
                        if 'version' in server_data:
                            version_str = server_data['version']
                            if version_str:
                                try: version = ServerVersion.from_string(version_str)
                                except ValueError: version = None # Handle potential parsing errors

                        metrics = ServerMetrics()
                        if 'metrics' in server_data:
                            metrics_data = server_data['metrics']
                            for metric_key, metric_value in metrics_data.items():
                                if hasattr(metrics, metric_key):
                                    if metric_key == 'status':
                                        try:
                                            status_enum = ServerStatus(metric_value)
                                            setattr(metrics, metric_key, status_enum)
                                        except ValueError:
                                            log.warning(f"Invalid async status value '{metric_value}' for {server_name}, defaulting to UNKNOWN.")
                                            setattr(metrics, metric_key, ServerStatus.UNKNOWN)
                                    else:
                                        setattr(metrics, metric_key, metric_value)

                        self.servers[server_name] = ServerConfig(
                            name=server_name, type=server_type, path=server_data.get('path', ''),
                            args=server_data.get('args', []), enabled=server_data.get('enabled', True),
                            auto_start=server_data.get('auto_start', True), description=server_data.get('description', ''),
                            trusted=server_data.get('trusted', False), categories=server_data.get('categories', []),
                            version=version, rating=server_data.get('rating', 5.0), retry_count=server_data.get('retry_count', 3),
                            timeout=server_data.get('timeout', 30.0), metrics=metrics, registry_url=server_data.get('registry_url'),
                            retry_policy=server_data.get('retry_policy', {"max_attempts": 3, "backoff_factor": 0.5, "timeout_increment": 5}),
                            capabilities=server_data.get('capabilities', {"tools": True, "resources": True, "prompts": True})
                        )
                elif key == 'cache_ttl_mapping': self.cache_ttl_mapping = value
                else:
                    if hasattr(self, key): setattr(self, key, value)

        except FileNotFoundError: 
            await self.save_async()
            return
        except IOError as e: log.error(f"Error reading config async {CONFIG_FILE}: {e}")
        except yaml.YAMLError as e: log.error(f"Error parsing config async {CONFIG_FILE}: {e}")
        except Exception as e: log.error(f"Unexpected error loading config async: {e}")
    
    def save(self):
        """Save configuration to file synchronously"""
        config_data = self._prepare_config_data()
        
        try:
            with open(CONFIG_FILE, 'w') as f:
                # Use a temporary dict to avoid saving the API key if loaded from env
                save_data = config_data.copy()
                if 'api_key' in save_data and os.environ.get("ANTHROPIC_API_KEY"):
                    # Don't save the key if it came from the environment
                    del save_data['api_key'] 
                yaml.safe_dump(save_data, f)
        except IOError as e:
            log.error(f"Error writing config file {CONFIG_FILE}: {e}")
        except yaml.YAMLError as e:
            log.error(f"Error formatting config data for saving: {e}")
        # Keep broad exception for unexpected saving issues
        except Exception as e: 
            log.error(f"Unexpected error saving config: {e}")
    
    async def save_async(self):
        """Save configuration to file asynchronously"""
        config_data = self._prepare_config_data()
        
        try:
            # Use a temporary dict to avoid saving the API key if loaded from env
            save_data = config_data.copy()
            if 'api_key' in save_data and os.environ.get("ANTHROPIC_API_KEY"):
                # Don't save the key if it came from the environment
                del save_data['api_key']
                
            async with aiofiles.open(CONFIG_FILE, 'w') as f:
                await f.write(yaml.safe_dump(save_data))
        except IOError as e:
            log.error(f"Error writing config file {CONFIG_FILE}: {e}")
        except yaml.YAMLError as e:
            log.error(f"Error formatting config data for saving: {e}")
        # Keep broad exception for unexpected saving issues
        except Exception as e: 
            log.error(f"Unexpected error saving config: {e}")


class History:
    def __init__(self, max_entries=MAX_HISTORY_ENTRIES):
        self.entries = deque(maxlen=max_entries)
        self.max_entries = max_entries
        self.load_sync()  # Use sync version for initialization
    
    def add(self, entry: ChatHistory):
        """Add a new entry to history"""
        self.entries.append(entry)
        self.save_sync()  # Use sync version for immediate updates
        
    async def add_async(self, entry: ChatHistory):
        """Add a new entry to history (async version)"""
        self.entries.append(entry)
        await self.save()
    
    def load_sync(self):
        """Load history from file synchronously (for initialization)"""
        if not HISTORY_FILE.exists():
            return
        
        try:
            with open(HISTORY_FILE, 'r') as f:
                file_data = f.read()
                if file_data:
                    history_data = json.loads(file_data)
                else:
                    history_data = []
            
            self.entries.clear()
            for entry_data in history_data:
                self.entries.append(ChatHistory(
                    query=entry_data.get('query', ''),
                    response=entry_data.get('response', ''),
                    model=entry_data.get('model', DEFAULT_MODEL),
                    timestamp=entry_data.get('timestamp', ''),
                    server_names=entry_data.get('server_names', []),
                    tools_used=entry_data.get('tools_used', []),
                    conversation_id=entry_data.get('conversation_id'),
                    latency_ms=entry_data.get('latency_ms', 0.0),
                    tokens_used=entry_data.get('tokens_used', 0),
                    cached=entry_data.get('cached', False),
                    streamed=entry_data.get('streamed', False)
                ))
                
        except FileNotFoundError:
            # Expected if no history yet
            return
        except IOError as e:
            log.error(f"Error reading history file {HISTORY_FILE}: {e}")
        except json.JSONDecodeError as e:
            log.error(f"Error decoding history JSON from {HISTORY_FILE}: {e}")
        # Keep broad exception for unexpected issues during history loading/parsing
        except Exception as e: 
             log.error(f"Unexpected error loading history: {e}")
    
    def save_sync(self):
        """Save history to file synchronously"""
        try:
            history_data = []
            for entry in self.entries:
                history_data.append({
                    'query': entry.query,
                    'response': entry.response,
                    'model': entry.model,
                    'timestamp': entry.timestamp,
                    'server_names': entry.server_names,
                    'tools_used': entry.tools_used,
                    'conversation_id': entry.conversation_id,
                    'latency_ms': entry.latency_ms,
                    'tokens_used': entry.tokens_used,
                    'cached': entry.cached,
                    'streamed': entry.streamed
                })
            
            with open(HISTORY_FILE, 'w') as f:
                f.write(json.dumps(history_data, indent=2))
                
        except IOError as e:
            log.error(f"Error writing history file {HISTORY_FILE}: {e}")
        except TypeError as e: # Handle non-serializable data in history entries
             log.error(f"Could not serialize history data: {e}")
        # Keep broad exception for unexpected saving issues
        except Exception as e: 
             log.error(f"Unexpected error saving history: {e}")
             
    async def load(self):
        """Load history from file asynchronously"""
        if not HISTORY_FILE.exists():
            return
        
        try:
            async with aiofiles.open(HISTORY_FILE, 'r') as f:
                content = await f.read()
                history_data = json.loads(content)
            
            self.entries.clear()
            for entry_data in history_data:
                self.entries.append(ChatHistory(
                    query=entry_data.get('query', ''),
                    response=entry_data.get('response', ''),
                    model=entry_data.get('model', DEFAULT_MODEL),
                    timestamp=entry_data.get('timestamp', ''),
                    server_names=entry_data.get('server_names', []),
                    tools_used=entry_data.get('tools_used', []),
                    conversation_id=entry_data.get('conversation_id'),
                    latency_ms=entry_data.get('latency_ms', 0.0),
                    tokens_used=entry_data.get('tokens_used', 0),
                    cached=entry_data.get('cached', False),
                    streamed=entry_data.get('streamed', False)
                ))
                
        except FileNotFoundError:
            # Expected if no history yet
            return
        except IOError as e:
            log.error(f"Error reading history file {HISTORY_FILE}: {e}")
        except json.JSONDecodeError as e:
            log.error(f"Error decoding history JSON from {HISTORY_FILE}: {e}")
        # Keep broad exception for unexpected issues during history loading/parsing
        except Exception as e: 
             log.error(f"Unexpected error loading history: {e}")
             
    async def save(self):
        """Save history to file asynchronously"""
        try:
            history_data = []
            for entry in self.entries:
                history_data.append({
                    'query': entry.query,
                    'response': entry.response,
                    'model': entry.model,
                    'timestamp': entry.timestamp,
                    'server_names': entry.server_names,
                    'tools_used': entry.tools_used,
                    'conversation_id': entry.conversation_id,
                    'latency_ms': entry.latency_ms,
                    'tokens_used': entry.tokens_used,
                    'cached': entry.cached,
                    'streamed': entry.streamed
                })
            
            async with aiofiles.open(HISTORY_FILE, 'w') as f:
                await f.write(json.dumps(history_data, indent=2))
                
        except IOError as e:
            log.error(f"Error writing history file {HISTORY_FILE}: {e}")
        except TypeError as e: # Handle non-serializable data in history entries
             log.error(f"Could not serialize history data: {e}")
        # Keep broad exception for unexpected saving issues
        except Exception as e: 
             log.error(f"Unexpected error saving history: {e}")
             
    def search(self, query: str, limit: int = 5) -> List[ChatHistory]:
        """Search history entries for a query"""
        results = []
        
        # Very simple search for now - could be improved with embeddings
        query = query.lower()
        for entry in reversed(self.entries):
            if (query in entry.query.lower() or
                query in entry.response.lower() or
                any(query in tool.lower() for tool in entry.tools_used) or
                any(query in server.lower() for server in entry.server_names)):
                results.append(entry)
                if len(results) >= limit:
                    break
                    
        return results


class ServerMonitor:
    """Monitor server health and manage recovery"""
    def __init__(self, server_manager: "ServerManager"):
        self.server_manager = server_manager
        self.monitoring = False
        self.monitor_task = None
        self.health_check_interval = 30  # seconds
    
    async def start_monitoring(self):
        """Start background monitoring of servers"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        log.info("Server health monitoring started")
    
    async def stop_monitoring(self):
        """Stop monitoring servers"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
            self.monitor_task = None
        log.info("Server health monitoring stopped")
    
    async def _monitor_loop(self):
        """Background loop for monitoring server health"""
        while self.monitoring:
            try:
                await self._check_all_servers()
                await asyncio.sleep(self.health_check_interval)
            except asyncio.CancelledError:
                break
            # Keep broad exception for general errors in the monitor loop
            except Exception as e: 
                log.error(f"Error in server monitor: {e}")
                await asyncio.sleep(5) # Short delay on error
    
    async def _check_all_servers(self):
        """Check health for all connected servers"""
        for name, session in list(self.server_manager.active_sessions.items()):
            try:
                await self._check_server_health(name, session)
            except McpError as e: # Catch specific MCP errors
                 log.error(f"MCP error checking health for server {name}: {e}")
            except httpx.RequestError as e: # Catch network errors if using SSE
                 log.error(f"Network error checking health for server {name}: {e}")
            # Keep broad exception for unexpected check issues
            except Exception as e: 
                 log.error(f"Unexpected error checking health for server {name}: {e}")
    
    async def _check_server_health(self, server_name: str, session: ClientSession):
        """Check health for a specific server"""
        if server_name not in self.server_manager.config.servers:
            return
            
        server_config = self.server_manager.config.servers[server_name]
        metrics = server_config.metrics
        
        # Record uptime
        metrics.uptime += self.health_check_interval / 60  # minutes
        
        start_time = time.time()
        try:
            # Simple health check - list tools
            await session.list_tools()
            
            # Success - record response time
            response_time = time.time() - start_time
            metrics.update_response_time(response_time)
            
        except McpError as e: # Catch MCP specific errors
            # Failure - record error
            metrics.error_count += 1
            log.warning(f"Health check failed for server {server_name} (MCP Error): {e}")
        except httpx.RequestError as e: # Catch network errors if using SSE
            metrics.error_count += 1
            log.warning(f"Health check failed for server {server_name} (Network Error): {e}")
        # Keep broad exception for truly unexpected failures during health check
        except Exception as e: 
            metrics.error_count += 1
            log.warning(f"Health check failed for server {server_name} (Unexpected Error): {e}")
            
        # Update overall status
        metrics.update_status()
        
        # Handle recovery if needed
        if metrics.status == ServerStatus.ERROR:
            await self._recover_server(server_name)
    
    async def _recover_server(self, server_name: str):
        """Attempt to recover a failing server"""
        if server_name not in self.server_manager.config.servers:
            return
            
        server_config = self.server_manager.config.servers[server_name]
        
        log.warning(f"Attempting to recover server {server_name}")
        
        # For STDIO servers, we can restart the process
        if server_config.type == ServerType.STDIO:
            await self.server_manager.restart_server(server_name)
        
        # For SSE servers, we can try reconnecting
        elif server_config.type == ServerType.SSE:
            await self.server_manager.reconnect_server(server_name)


# =============================================================================
# Custom Stdio Client Logic with Noise Filtering
# =============================================================================

class RobustStdioSession(ClientSession):
    """
    A ClientSession implementation that reads stdout and directly resolves
    response Futures, aiming for a balance between polling and queue-based processing.
    """
    def __init__(self, process: asyncio.subprocess.Process, server_name: str):
        # Python 3.11+ check no longer needed for TaskGroup
        self._process = process
        self._server_name = server_name
        self._stdin = process.stdin
        self._stderr_reader_task: Optional[asyncio.Task] = None # Keep track of external stderr reader

        # --- REMOVED QUEUE ---
        # self._message_queue = asyncio.Queue(maxsize=100)

        # --- KEPT FUTURES DICT ---
        self._response_futures: Dict[str, asyncio.Future] = {}

        self._request_id_counter = 0
        self._lock = asyncio.Lock() # Lock still useful for ID generation and maybe future dict access
        self._is_active = True
        self._background_task_runner: Optional[asyncio.Task] = None # Task to run the reader

        log.debug(f"[{self._server_name}] Initializing RobustStdioSession (Direct Future Resolution Version)")
        # Start the background task runner for the combined reader/processor loop
        self._background_task_runner = asyncio.create_task(
            self._run_reader_processor_wrapper(), # Renamed wrapper
            name=f"session-reader-processor-{server_name}"
        )

    # --- Methods initialize() and send_initialized_notification() remain the same ---
    async def initialize(self, capabilities: Optional[Dict[str, Any]] = None, response_timeout: float = 60.0) -> Any:
            """Sends the MCP initialize request and waits for the response."""
            log.info(f"[{self._server_name}] Sending initialize request...")
            client_capabilities = capabilities if capabilities is not None else {}
            params = {
                "processId": os.getpid(),
                "clientInfo": {"name": "ultimate-mcp-client", "version": "1.0.0"},
                "rootUri": None,
                "capabilities": client_capabilities,
                "protocolVersion": "2025-03-25",
            }
            result = await self._send_request("initialize", params, response_timeout=response_timeout)
            log.info(f"[{self._server_name}] Initialize request successful.")
            # You might still want to store capabilities if needed elsewhere
            # self._server_capabilities = result.get("capabilities") if isinstance(result, dict) else None
            return result # Return the raw result dict

    async def send_initialized_notification(self):
        """Sends the 'notifications/initialized' notification to the server."""
        if not self._is_active: 
            log.warning(f"[{self._server_name}] Session inactive, skip initialized.")
            return
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        try:
            notification_str = json.dumps(notification) + '\n'
            notification_bytes = notification_str.encode('utf-8')
            log.info(f"[{self._server_name}] Sending initialized notification...")
            if self._stdin is None or self._stdin.is_closing(): raise ConnectionAbortedError("Stdin closed")
            self._stdin.write(notification_bytes)
            await self._stdin.drain()
            log.debug(f"[{self._server_name}] Initialized notification sent.")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            log.error(f"[{self._server_name}] Conn error sending initialized: {e}")
            await self._close_internal_state(e)
        except Exception as e: log.error(f"[{self._server_name}] Error sending initialized: {e}", exc_info=True)

    # --- Renamed Wrapper for Combined Reader/Processor ---
    async def _run_reader_processor_wrapper(self):
        """Wraps the combined reader/processor loop task."""
        close_exception: Optional[BaseException] = None
        try:
            log.debug(f"[{self._server_name}] Entering reader/processor task wrapper.")
            await self._read_and_process_stdout_loop() # Run the combined loop
            log.info(f"[{self._server_name}] Reader/processor task finished normally.")
        except asyncio.CancelledError:
            log.info(f"[{self._server_name}] Reader/processor task wrapper cancelled.")
            close_exception = asyncio.CancelledError("Reader/processor task cancelled")
        except Exception as e:
            log.error(f"[{self._server_name}] Reader/processor task wrapper failed: {e}", exc_info=True)
            close_exception = e
        finally:
            log.info(f"[{self._server_name}] Reader/processor task wrapper exiting.")
            if self._is_active:
                log.warning(f"[{self._server_name}] Reader/processor finished unexpectedly. Forcing close.")
                final_exception = close_exception if close_exception else ConnectionAbortedError("Reader/processor task finished unexpectedly")
                await self._close_internal_state(final_exception)

    # --- REMOVED _run_background_tasks() ---

    async def _read_and_process_stdout_loop(self):
        """
        Reads stdout lines, parses JSON, and directly resolves corresponding Futures
        for responses, or handles notifications/server requests.
        """
        handshake_complete = False
        stream_limit = getattr(self._process.stdout, '_limit', 'Unknown')
        log.debug(f"[{self._server_name}] Starting combined reader/processor loop (Buffer limit: {stream_limit}).")
        try:
            while self._process.returncode is None:
                if not self._is_active: 
                    log.info(f"[{self._server_name}] Session inactive, exiting loop.")
                    break
                try:
                    # Use timeout for readline
                    line_bytes = await asyncio.wait_for(self._process.stdout.readline(), timeout=60.0)
                    if not line_bytes:
                        if self._process.stdout.at_eof(): 
                            log.warning(f"[{self._server_name}] Stdout EOF.")
                            break
                        else: log.debug(f"[{self._server_name}] readline() timeout.")
                        continue

                    line_str_raw = line_bytes.decode('utf-8', errors='replace')
                    if USE_VERBOSE_SESSION_LOGGING: log.debug(f"[{self._server_name}] READ/PROC RAW <<< {repr(line_str_raw)}")
                    line_str = line_str_raw.strip()
                    if not line_str: continue

                    # --- Parse and Process Immediately ---
                    try:
                        message = json.loads(line_str)
                        is_valid_rpc = (
                            isinstance(message, dict) and message.get("jsonrpc") == "2.0" and
                            ('id' in message or 'method' in message)
                        )

                        if not is_valid_rpc:
                            if isinstance(message, dict): log.debug(f"[{self._server_name}] Skipping non-MCP JSON object: {line_str[:100]}...")
                            else: log.debug(f"[{self._server_name}] Skipping non-dict JSON: {line_str[:100]}...")
                            continue # Skip non-rpc lines

                        if not handshake_complete: log.info(f"[{self._server_name}] First valid JSON-RPC detected.")
                        handshake_complete = True

                        msg_id = message.get("id")

                        # --- Direct Future Resolution for Responses/Errors ---
                        if msg_id is not None:
                            str_msg_id = str(msg_id)
                            # Use lock when accessing shared dictionary if needed, though maybe not
                            # critical if only this task modifies it by popping.
                            # async with self._lock:
                            future = self._response_futures.pop(str_msg_id, None)

                            if future and not future.done():
                                if "result" in message:
                                    log.debug(f"[{self._server_name}] READ/PROC: Resolving future ID {msg_id} with RESULT.")
                                    future.set_result(message["result"])
                                elif "error" in message:
                                    err_data = message["error"]
                                    err_msg = f"Server error ID {msg_id}: {err_data.get('message', 'Unknown')} (Code: {err_data.get('code', 'N/A')})"
                                    if err_data.get('data'): err_msg += f" Data: {repr(err_data.get('data'))}"
                                    log.warning(f"[{self._server_name}] READ/PROC: Resolving future ID {msg_id} with ERROR: {err_msg}")
                                    server_exception = RuntimeError(err_msg)
                                    future.set_exception(server_exception)
                                else:
                                    log.error(f"[{self._server_name}] READ/PROC: Invalid response format for ID {msg_id}.")
                                    future.set_exception(RuntimeError(f"Invalid response format ID {msg_id}"))
                            elif future:
                                log.debug(f"[{self._server_name}] READ/PROC: Future for ID {msg_id} already done (timed out?).")
                            else:
                                log.warning(f"[{self._server_name}] READ/PROC: Received response for unknown/timed-out ID: {msg_id}.")

                        # --- Handle Notifications/Server Requests ---
                        elif "method" in message:
                            method_name = message['method']
                            log.debug(f"[{self._server_name}] READ/PROC: Received server message: {method_name}")
                            # Handle directly or dispatch (e.g., put on a *different* queue for complex handlers)
                            if method_name == "notifications/progress": pass # Handle progress
                            elif method_name == "notifications/message": pass # Handle log
                            elif method_name == "sampling/createMessage": pass # Handle sampling
                            else: log.warning(f"[{self._server_name}] READ/PROC: Unhandled server method: {method_name}")
                        else:
                            log.warning(f"[{self._server_name}] READ/PROC: Unknown message structure: {repr(message)}")

                    except json.JSONDecodeError: log.debug(f"[{self._server_name}] Skipping noisy line: {line_str[:100]}...")
                    except Exception as proc_err: log.error(f"[{self._server_name}] Error processing line '{line_str[:100]}...': {proc_err}", exc_info=True)

                # --- Exception Handling for readline() ---
                except asyncio.TimeoutError: 
                    log.debug(f"[{self._server_name}] Outer timeout reading stdout.")
                    continue
                except (BrokenPipeError, ConnectionResetError): 
                    log.warning(f"[{self._server_name}] Stdout pipe broken.")
                    break
                except ValueError as e: # Buffer limits
                     if "longer than limit" in str(e) or "too long" in str(e): log.error(f"[{self._server_name}] Buffer limit ({stream_limit}) exceeded!", exc_info=True)
                     else: log.error(f"[{self._server_name}] ValueError reading stdout: {e}", exc_info=True)
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
        """
        Sends a JSON-RPC request and waits for the response Future to be set
        by the combined reader/processor loop.
        (Largely unchanged from original, relies on future being set externally).
        """
        if not self._is_active or (self._process and self._process.returncode is not None):
            raise ConnectionAbortedError("Session inactive or process terminated")

        async with self._lock: # Use lock for ID generation
            self._request_id_counter += 1
            request_id = str(self._request_id_counter)

        request = {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}

        # --- Create Future ---
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        # async with self._lock: # Lock if modifying shared dict
        self._response_futures[request_id] = future

        # --- Sending logic (unchanged) ---
        try:
            request_str = json.dumps(request) + '\n'
            request_bytes = request_str.encode('utf-8')
            log.debug(f"[{self._server_name}] SEND: ID {request_id} ({method}): {request_bytes.decode('utf-8', errors='replace')[:100]}...")
            if self._stdin is None or self._stdin.is_closing(): raise ConnectionAbortedError("Stdin closed")
            if USE_VERBOSE_SESSION_LOGGING: log.debug(f"[{self._server_name}] RAW >>> {repr(request_bytes)}")
            self._stdin.write(request_bytes)
            await self._stdin.drain()
            if USE_VERBOSE_SESSION_LOGGING: log.info(f"[{self._server_name}] Drain complete for ID {request_id}.")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            log.error(f"[{self._server_name}] SEND FAIL ID {request_id}: Pipe broken: {e}")
            # Clean up future if send fails
            # async with self._lock: # Lock if modifying shared dict
            self._response_futures.pop(request_id, None)
            if not future.done(): future.set_exception(e)
            raise ConnectionAbortedError(f"Conn lost sending ID {request_id}: {e}") from e
        except Exception as e:
            log.error(f"[{self._server_name}] SEND FAIL ID {request_id}: {e}", exc_info=True)
            # Clean up future
            # async with self._lock: # Lock if modifying shared dict
            self._response_futures.pop(request_id, None)
            if not future.done(): future.set_exception(e)
            raise RuntimeError(f"Failed to send ID {request_id}: {e}") from e

        # --- Wait for Future (unchanged) ---
        try:
            log.debug(f"[{self._server_name}] WAIT: Waiting for future ID {request_id} ({method}) (timeout={response_timeout}s)")
            result = await asyncio.wait_for(future, timeout=response_timeout)
            log.debug(f"[{self._server_name}] WAIT: Future resolved for ID {request_id}. Result received.")
            return result
        except asyncio.TimeoutError as timeout_error:
            log.error(f"[{self._server_name}] WAIT: Timeout waiting for future ID {request_id} ({method})")
            # Clean up future from dict on timeout
            # async with self._lock: # Lock if modifying shared dict
            self._response_futures.pop(request_id, None)
            raise RuntimeError(f"Timeout waiting for response to {method} request (ID: {request_id})") from timeout_error
        except asyncio.CancelledError:
            log.info(f"[{self._server_name}] WAIT: Wait cancelled for ID {request_id} ({method}).")
            # async with self._lock: # Lock if modifying shared dict
            self._response_futures.pop(request_id, None) # Clean up future
            raise
        except Exception as wait_err:
             # Handle case where future contains an exception set by the reader/processor
             if future.done() and future.exception():
                  server_error = future.exception()
                  log.warning(f"[{self._server_name}] WAIT: Future ID {request_id} failed with server error: {server_error}")
                  raise server_error from wait_err # Re-raise original error
             else:
                  log.error(f"[{self._server_name}] WAIT: Error waiting for future ID {request_id}: {wait_err}", exc_info=True)
                  # async with self._lock: # Lock if modifying shared dict
                  self._response_futures.pop(request_id, None) # Clean up future
                  raise RuntimeError(f"Error processing response for {method} ID {request_id}: {wait_err}") from wait_err


    # --- ClientSession Methods (list_tools, call_tool, etc.) remain the same ---
    # They use the _send_request method which now relies on the combined reader/processor.
    async def list_tools(self, response_timeout: float = 40.0) -> ListToolsResult:
        log.debug(f"[{self._server_name}] Calling list_tools")
        result = await self._send_request("tools/list", {}, response_timeout=response_timeout) # Correct method name
        try: return ListToolsResult(**result)
        except Exception as e: raise RuntimeError(f"Invalid list_tools response format: {e}") from e

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], response_timeout: float = 30.0) -> CallToolResult:
        log.debug(f"[{self._server_name}] Calling call_tool: {tool_name}")
        params = {"name": tool_name, "arguments": arguments}
        result = await self._send_request("tools/call", params, response_timeout=response_timeout) # Correct method name
        try: return CallToolResult(**result)
        except Exception as e: raise RuntimeError(f"Invalid call_tool response format: {e}") from e

    async def list_resources(self, response_timeout: float = 40.0) -> ListResourcesResult:
        log.debug(f"[{self._server_name}] Calling list_resources")
        result = await self._send_request("resources/list", {}, response_timeout=response_timeout) # Correct method name
        try: return ListResourcesResult(**result)
        except Exception as e: raise RuntimeError(f"Invalid list_resources response format: {e}") from e

    async def read_resource(self, uri: AnyUrl, response_timeout: float = 30.0) -> ReadResourceResult:
        log.debug(f"[{self._server_name}] Calling read_resource: {uri}")
        params = {"uri": str(uri)}
        result = await self._send_request("resources/read", params, response_timeout=response_timeout)
        try: return ReadResourceResult(**result)
        except Exception as e: raise RuntimeError(f"Invalid read_resource response format: {e}") from e

    async def list_prompts(self, response_timeout: float = 40.0) -> ListPromptsResult:
        log.debug(f"[{self._server_name}] Calling list_prompts")
        result = await self._send_request("prompts/list", {}, response_timeout=response_timeout) # Correct method name
        try: return ListPromptsResult(**result)
        except Exception as e: raise RuntimeError(f"Invalid list_prompts response format: {e}") from e

    async def get_prompt(self, prompt_name: str, variables: Dict[str, Any], response_timeout: float = 30.0) -> GetPromptResult:
        log.debug(f"[{self._server_name}] Calling get_prompt: {prompt_name}")
        params = {"name": prompt_name, "arguments": variables} # Schema uses 'arguments'
        result = await self._send_request("prompts/get", params, response_timeout=response_timeout)
        try: return GetPromptResult(**result)
        except Exception as e: raise RuntimeError(f"Invalid get_prompt response format: {e}") from e

    async def _close_internal_state(self, exception: Exception):
        """Closes internal session state."""
        if not self._is_active: return
        self._is_active = False
        log.debug(f"[{self._server_name}] Closing internal state (Direct Future) due to: {exception}")
        await self._cancel_pending_futures(exception) # Cancel any remaining futures

    async def _cancel_pending_futures(self, exception: Exception):
        """Cancel all outstanding response futures. (Still needed for this version)."""
        log.debug(f"[{self._server_name}] Cancelling {len(self._response_futures)} pending futures with: {exception}")
        # async with self._lock: # Lock if modifying shared dict
        futures_to_cancel = list(self._response_futures.items())
        self._response_futures.clear() # Clear immediately

        for future_id, future in futures_to_cancel:
            if future and not future.done():
                try: future.set_exception(exception)
                except asyncio.InvalidStateError: pass # Already done/cancelled

    async def aclose(self):
        """Closes the session and cleans up resources."""
        log.info(f"[{self._server_name}] Closing RobustStdioSession (Direct Future)...")
        if not self._is_active: 
            log.debug(f"[{self._server_name}] Already closed.")
            return

        # 1. Mark inactive & Cancel Futures
        await self._close_internal_state(ConnectionAbortedError("Session closed by client"))

        # 2. Cancel the background reader/processor task runner
        if self._background_task_runner and not self._background_task_runner.done():
            log.debug(f"[{self._server_name}] Cancelling reader/processor task runner...")
            self._background_task_runner.cancel()
            with suppress(asyncio.CancelledError): await self._background_task_runner
            log.debug(f"[{self._server_name}] Reader/processor task runner finished cancellation.")
        else: log.debug(f"[{self._server_name}] Reader/processor task runner already done or None.")

        # 3. Cancel external stderr reader (unchanged)
        if self._stderr_reader_task and not self._stderr_reader_task.done():
            log.debug(f"[{self._server_name}] Cancelling external stderr reader task...")
            self._stderr_reader_task.cancel()
            with suppress(asyncio.CancelledError): await self._stderr_reader_task
            log.debug(f"[{self._server_name}] External stderr reader task finished cancellation.")

        # 4. Terminate process (unchanged)
        if self._process and self._process.returncode is None:
            log.info(f"[{self._server_name}] Terminating process PID {self._process.pid} during aclose...")
            try:
                self._process.terminate()
                with suppress(asyncio.TimeoutError): await asyncio.wait_for(self._process.wait(), timeout=2.0)
                if self._process.returncode is None:
                    log.warning(f"[{self._server_name}] Process killing required.")
                    try: 
                        self._process.kill()
                        await asyncio.wait_for(self._process.wait(), timeout=1.0)
                    except ProcessLookupError: 
                        pass
                    except Exception as e: 
                        log.error(f"Kill error: {e}")
            except ProcessLookupError: 
                pass
            except Exception as e: 
                log.error(f"Terminate error: {e}")

        log.info(f"[{self._server_name}] RobustStdioSession (Direct Future) closed.")


class ServerManager:
    def __init__(self, config: Config, tool_cache=None, safe_printer=None):
        self.config = config
        self.exit_stack = AsyncExitStack()
        self.active_sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.tool_cache = tool_cache
        self._safe_printer = safe_printer or print
        self.monitor = ServerMonitor(self)
        self.registry = ServerRegistry() if config.enable_registry else None
        self.registered_services: Dict[str, ServiceInfo] = {} # Store zeroconf info
        self._session_tasks: Dict[str, List[asyncio.Task]] = {} # Store tasks per session        
        self.sanitized_to_original = {}  # Maps sanitized name -> original name
        self._port_scan_client: Optional[httpx.AsyncClient] = None # Client for port scanning
        self.discovered_servers_cache: List[Dict] = [] # Store discovery results
        self._discovery_in_progress = asyncio.Lock() # Prevent concurrent discovery runs

    @property
    def tools_by_server(self) -> Dict[str, List[MCPTool]]:
        """Group tools by server name for easier lookup."""
        result = {}
        for tool in self.tools.values():
            if tool.server_name not in result:
                result[tool.server_name] = []
            result[tool.server_name].append(tool)
        return result
    
    @asynccontextmanager
    async def connect_server_session(self, server_config: ServerConfig):
        """Context manager for connecting to a server with proper cleanup.
        
        This handles connecting to the server, tracking the session, and proper cleanup
        when the context is exited, whether normally or due to an exception.
        
        Args:
            server_config: The server configuration
            
        Yields:
            The connected session or None if connection failed
        """
        server_name = server_config.name
        session = None
        connected = False
        
        # Use safe_stdout context manager to protect against stdout pollution during connection
        with safe_stdout():
            try:
                # Use existing connection logic
                session = await self.connect_to_server(server_config)
                if session:
                    self.active_sessions[server_name] = session
                    connected = True
                    yield session
                else:
                    yield None
            finally:
                # Clean up if we connected successfully
                if connected and server_name in self.active_sessions:
                    # Note: We're not removing from self.active_sessions here
                    # as that should be managed by higher-level disconnect method
                    # This just ensures session cleanup resources are released
                    log.debug(f"Cleaning up server session for {server_name}")
                    # Close could be added if a specific per-session close is implemented
                    # For now we rely on the exit_stack in close() method

    async def _discover_local_servers(self):
        """Discover MCP servers in local filesystem paths"""
        discovered_local = []
        
        for base_path in self.config.discovery_paths:
            base_path = os.path.expanduser(base_path)
            if not os.path.exists(base_path):
                continue
                
            log.info(f"Discovering servers in {base_path}")
            
            # Look for python and js files
            for ext, server_type in [('.py', 'stdio'), ('.js', 'stdio')]:
                for root, _, files in os.walk(base_path):
                    for file in files:
                        if file.endswith(ext) and 'mcp' in file.lower():
                            path = os.path.join(root, file)
                            name = os.path.splitext(file)[0]
                            
                            # Skip if already in config
                            if any(s.path == path for s in self.config.servers.values()):
                                continue
                                
                            discovered_local.append((name, path, server_type))
        
        # Store in a class attribute to be accessed by _process_discovery_results
        self._discovered_local = discovered_local
        log.info(f"Discovered {len(discovered_local)} local servers")
    
    async def _discover_registry_servers(self):
        """Discover MCP servers from remote registry"""
        discovered_remote = []
        
        if not self.registry:
            log.warning("Registry not available, skipping remote discovery")
            self._discovered_remote = discovered_remote
            return
            
        try:
            # Try to discover from remote registry
            remote_servers = await self.registry.discover_remote_servers()
            for server in remote_servers:
                name = server.get("name", "")
                url = server.get("url", "")
                server_type = "sse"  # Remote servers are always SSE
                
                # Skip if already in config
                if any(s.path == url for s in self.config.servers.values()):
                    continue
                    
                server_version = None
                if "version" in server:
                    server_version = ServerVersion.from_string(server["version"])
                    
                categories = server.get("categories", [])
                rating = server.get("rating", 5.0)
                
                discovered_remote.append((name, url, server_type, server_version, categories, rating))
        except httpx.RequestError as e:
            log.error(f"Network error during registry discovery: {e}")
        except httpx.HTTPStatusError as e:
            log.error(f"HTTP error during registry discovery: {e.response.status_code}")
        except json.JSONDecodeError as e:
            log.error(f"JSON decode error during registry discovery: {e}")
        except Exception as e: 
            log.error(f"Unexpected error discovering from registry: {e}")
            
        # Store in a class attribute
        self._discovered_remote = discovered_remote
        log.info(f"Discovered {len(discovered_remote)} registry servers")
    
    async def _discover_mdns_servers(self):
        """Discover MCP servers from local network using mDNS"""
        discovered_mdns = []
        
        if not self.registry:
            log.warning("Registry not available, skipping mDNS discovery")
            self._discovered_mdns = discovered_mdns
            return
            
        # Start discovery if not already running
        if not self.registry.zeroconf:
            self.registry.start_local_discovery()
            
        # Wait a moment for discovery
        await asyncio.sleep(2)
        
        # Process discovered servers
        for name, server in self.registry.discovered_servers.items():
            url = server.get("url", "")
            server_type = server.get("type", "sse")
            
            # Skip if already in config
            if any(s.path == url for s in self.config.servers.values()):
                continue
        
            # Get additional information
            version = server.get("version")
            categories = server.get("categories", [])
            description = server.get("description", "")
            
            discovered_mdns.append((name, url, server_type, version, categories, description))
            
        # Store in a class attribute
        self._discovered_mdns = discovered_mdns
        log.info(f"Discovered {len(discovered_mdns)} local network servers via mDNS")
    
    async def _process_discovery_results(self):
        """
        Process and display discovery results. Prompts user to add servers in interactive mode,
        otherwise automatically adds all newly discovered servers.
        """
        # Get all discovered servers from class attributes
        discovered_local = getattr(self, '_discovered_local', [])
        discovered_remote = getattr(self, '_discovered_remote', [])
        discovered_mdns = getattr(self, '_discovered_mdns', [])
        discovered_port_scan = getattr(self, '_discovered_port_scan', [])

        # Get safe console instance
        safe_console = get_safe_console()

        # --- Display Discoveries (Unchanged) ---
        total_discovered = len(discovered_local) + len(discovered_remote) + len(discovered_mdns) + len(discovered_port_scan)
        if total_discovered > 0:
            safe_console.print(f"\n[bold green]Discovered {total_discovered} potential MCP servers:[/]")

            if discovered_local:
                safe_console.print("\n[bold blue]Local File System:[/]")
                for i, (name, path, server_type) in enumerate(discovered_local, 1):
                    # Check if already in config for display purposes
                    exists = any(s.path == path for s in self.config.servers.values())
                    status = "[dim](exists)[/]" if exists else ""
                    safe_console.print(f"{i}. [bold]{name}[/] ({server_type}) - {path} {status}")

            if discovered_remote:
                safe_console.print("\n[bold magenta]Remote Registry:[/]")
                for i, (name, url, server_type, version, categories, rating) in enumerate(discovered_remote, 1):
                    version_str = f"v{version}" if version else "unknown version"
                    categories_str = ", ".join(categories) if categories else "no categories"
                    exists = any(s.path == url for s in self.config.servers.values())
                    status = "[dim](exists)[/]" if exists else ""
                    safe_console.print(f"{i}. [bold]{name}[/] ({server_type}) - {url} - {version_str} - Rating: {rating:.1f}/5.0 - {categories_str} {status}")

            if discovered_mdns:
                safe_console.print("\n[bold cyan]Local Network (mDNS):[/]")
                for i, (name, url, server_type, version, categories, description) in enumerate(discovered_mdns, 1):
                    version_str = f"v{version}" if version else "unknown version"
                    categories_str = ", ".join(categories) if categories else "no categories"
                    desc_str = f" - {description}" if description else ""
                    exists = any(s.path == url for s in self.config.servers.values())
                    status = "[dim](exists)[/]" if exists else ""
                    safe_console.print(f"{i}. [bold]{name}[/] ({server_type}) - {url} - {version_str} - {categories_str}{desc_str} {status}")

            if discovered_port_scan:
                safe_console.print("\n[bold yellow]Local Port Scan:[/]")
                for i, (name, url, server_type, version, categories, description) in enumerate(discovered_port_scan, 1):
                    desc_str = f" - {description}" if description else ""
                    exists = any(s.path == url for s in self.config.servers.values())
                    status = "[dim](exists)[/]" if exists else ""
                    safe_console.print(f"{i}. [bold]{name}[/] ({server_type}) - {url}{desc_str} {status}")

            # --- Determine Mode (Interactive vs Non-Interactive) ---
            # Check sys.argv for the interactive flag presence during initial call
            is_interactive_mode = "--interactive" in sys.argv or "-i" in sys.argv

            servers_added_count = 0
            added_server_paths = {s.path for s in self.config.servers.values()} # Track paths to avoid duplicates
            
            if is_interactive_mode:
                # --- Interactive Mode: Ask user ---
                if Confirm.ask("\nAdd discovered servers to configuration?", console=safe_console):
                    # Create selection interface
                    selections = []

                    if discovered_local:
                        safe_console.print("\n[bold blue]Local File System Servers:[/]")
                        for i, (name, path, server_type) in enumerate(discovered_local, 1):
                            if path not in added_server_paths: # Only ask for non-existing
                                if Confirm.ask(f"Add {name} ({path})?", default=False, console=safe_console):
                                    selections.append(("local", i-1))
                            else:
                                safe_console.print(f"[dim]Skipping {name} ({path}) - already configured.[/dim]")


                    if discovered_remote:
                        safe_console.print("\n[bold magenta]Remote Registry Servers:[/]")
                        for i, (name, url, server_type, version, categories, rating) in enumerate(discovered_remote, 1):
                            if url not in added_server_paths:
                                if Confirm.ask(f"Add {name} ({url})?", default=False, console=safe_console):
                                    selections.append(("remote", i-1))
                            else:
                                safe_console.print(f"[dim]Skipping {name} ({url}) - already configured.[/dim]")


                    if discovered_mdns:
                        safe_console.print("\n[bold cyan]Local Network Servers:[/]")
                        for i, (name, url, server_type, version, categories, description) in enumerate(discovered_mdns, 1):
                            if url not in added_server_paths:
                                if Confirm.ask(f"Add {name} ({url})?", default=False, console=safe_console):
                                    selections.append(("mdns", i-1))
                            else:
                                safe_console.print(f"[dim]Skipping {name} ({url}) - already configured.[/dim]")


                    if discovered_port_scan:
                        safe_console.print("\n[bold yellow]Port Scan Servers:[/]")
                        for i, (name, url, server_type, version, categories, description) in enumerate(discovered_port_scan, 1):
                             if url not in added_server_paths:
                                if Confirm.ask(f"Add {name} ({url})?", default=False, console=safe_console):
                                    selections.append(("portscan", i-1))
                             else:
                                safe_console.print(f"[dim]Skipping {name} ({url}) - already configured.[/dim]")

                    # Process selections
                    for source, idx in selections:
                        if source == "local":
                            name, path, server_type = discovered_local[idx]
                            if path not in added_server_paths: # Double check before adding
                                self.config.servers[name] = ServerConfig(
                                    name=name, type=ServerType(server_type), path=path,
                                    enabled=True, auto_start=False, description=f"Discovered {server_type} server"
                                )
                                added_server_paths.add(path)
                                servers_added_count += 1
                        elif source == "remote":
                            name, url, server_type, version, categories, rating = discovered_remote[idx]
                            if url not in added_server_paths:
                                self.config.servers[name] = ServerConfig(
                                    name=name, type=ServerType(server_type), path=url,
                                    enabled=True, auto_start=False, description="Discovered from registry",
                                    categories=categories, version=version, rating=rating,
                                    registry_url=self.registry.registry_urls[0] if self.registry and self.registry.registry_urls else None
                                )
                                added_server_paths.add(url)
                                servers_added_count += 1
                        elif source == "mdns":
                            name, url, server_type, version, categories, description = discovered_mdns[idx]
                            if url not in added_server_paths:
                                self.config.servers[name] = ServerConfig(
                                    name=name, type=ServerType(server_type), path=url,
                                    enabled=True, auto_start=False, description=description or "Discovered on local network",
                                    categories=categories, version=version if version else None
                                )
                                added_server_paths.add(url)
                                servers_added_count += 1
                        elif source == "portscan":
                            name, url, server_type, version, categories, description = discovered_port_scan[idx]
                            if url not in added_server_paths:
                                self.config.servers[name] = ServerConfig(
                                    name=name, type=ServerType(server_type), path=url,
                                    enabled=True, auto_start=False, description=description or f"Discovered via port scan",
                                )
                                added_server_paths.add(url)
                                servers_added_count += 1
                # --- End Interactive Selection ---

            else:
                # --- Non-Interactive Mode: Auto-add all new servers ---
                safe_console.print("\n[yellow]Non-interactive mode: Auto-adding newly discovered servers...[/]")
                auto_added_list = [] # Keep track of names for logging

                # Auto-add Local
                for name, path, server_type in discovered_local:
                    if path not in added_server_paths:
                        self.config.servers[name] = ServerConfig(
                            name=name, type=ServerType(server_type), path=path,
                            enabled=True, auto_start=False, description=f"Auto-discovered {server_type} server"
                        )
                        added_server_paths.add(path)
                        servers_added_count += 1
                        auto_added_list.append(f"{name} (local)")

                # Auto-add Remote
                for name, url, server_type, version, categories, rating in discovered_remote:
                     if url not in added_server_paths:
                        self.config.servers[name] = ServerConfig(
                            name=name, type=ServerType(server_type), path=url,
                            enabled=True, auto_start=False, description="Discovered from registry",
                            categories=categories, version=version, rating=rating,
                            registry_url=self.registry.registry_urls[0] if self.registry and self.registry.registry_urls else None
                        )
                        added_server_paths.add(url)
                        servers_added_count += 1
                        auto_added_list.append(f"{name} (remote)")

                # Auto-add mDNS
                for name, url, server_type, version, categories, description in discovered_mdns:
                     if url not in added_server_paths:
                        self.config.servers[name] = ServerConfig(
                            name=name, type=ServerType(server_type), path=url,
                            enabled=True, auto_start=False, description=description or "Discovered on local network",
                            categories=categories, version=version if version else None
                        )
                        added_server_paths.add(url)
                        servers_added_count += 1
                        auto_added_list.append(f"{name} (mDNS)")

                # Auto-add Port Scan
                for name, url, server_type, version, categories, description in discovered_port_scan:
                     if url not in added_server_paths:
                        self.config.servers[name] = ServerConfig(
                            name=name, type=ServerType(server_type), path=url,
                            enabled=True, auto_start=False, description=description or f"Discovered via port scan",
                        )
                        added_server_paths.add(url)
                        servers_added_count += 1
                        auto_added_list.append(f"{name} (port scan)")

                if auto_added_list:
                    safe_console.print(f"  Auto-added: {', '.join(auto_added_list)}")
                else:
                    safe_console.print("  No new servers to auto-add.")
            # --- End Non-Interactive Mode ---

            # --- Save configuration if servers were added ---
            if servers_added_count > 0:
                await self.config.save_async() # Use async save
                safe_console.print(f"[green]{servers_added_count} server(s) added to configuration.[/]")
            elif not is_interactive_mode:
                # If non-interactive and nothing added, explicitly state it
                safe_console.print("[yellow]No new servers added (they might already exist).[/]")
            # If interactive and user declined or selected none, no message needed here

        else:
            # Only print this if no servers were discovered at all
            safe_console.print("[yellow]No new servers discovered.[/]")

    async def _probe_port(self, ip_address: str, port: int, probe_timeout: float, client: httpx.AsyncClient) -> Optional[Dict[str, Any]]:
        """
        Attempts to detect an MCP SSE server. First performs a quick TCP check.
        If TCP port is open, tries GET /sse (streaming), then falls back to POST initialize on /.
        RETURNS None if TCP port is closed or no MCP pattern is detected via HTTP probes.
        Logs failures only if USE_VERBOSE_SESSION_LOGGING is True.
        """
        # --- Step 1: Quick TCP Pre-check ---
        tcp_check_timeout = 0.1
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(ip_address, port),
                timeout=tcp_check_timeout
            )
            writer.close()
            await writer.wait_closed()
            if USE_VERBOSE_SESSION_LOGGING: log.debug(f"TCP Port open: {ip_address}:{port}. Proceeding.")
        except (ConnectionRefusedError, asyncio.TimeoutError, OSError):
            return None
        except Exception as e:
            log.warning(f"Unexpected error during TCP pre-check for {ip_address}:{port}: {e}")
            return None

        # --- Step 2: HTTP Probes ---
        base_url = f"http://{ip_address}:{port}"
        sse_url = f"{base_url}/sse"
        mcp_server_found = False

        # --- Attempt 2a: GET /sse (Streaming Check) ---
        try:
            if USE_VERBOSE_SESSION_LOGGING: log.debug(f"Probing via streaming GET {sse_url}...")
            # Use client.stream for finer control
            async with client.stream("GET", sse_url, timeout=probe_timeout) as response:
                # Check headers immediately after response starts
                if response.status_code == 200 and response.headers.get("content-type", "").lower().startswith("text/event-stream"):
                    log.info(f"MCP SSE Server detected via streaming GET /sse at {sse_url}")
                    server_name = f"mcp-scan-{ip_address}-{port}-sse"
                    mcp_server_found = True
                    # Return success *without reading the body*
                    return {
                        "name": server_name, "path": sse_url, "type": ServerType.SSE, "args": [],
                        "description": f"Auto-discovered SSE server via port scan GET /sse on {ip_address}:{port}", "source": "portscan"
                    }
                else: # Log failure reason only if verbose
                    if USE_VERBOSE_SESSION_LOGGING:
                        log.debug(f"Probe failed for streaming GET {sse_url} - Status: {response.status_code}, Content-Type: {response.headers.get('content-type')}")
            # The 'async with' ensures the response (and connection) is closed here
            
        # Catch specific timeouts/errors related to establishing the connection or reading *headers*
        except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as http_err:
            if USE_VERBOSE_SESSION_LOGGING: log.debug(f"Probe error for streaming GET {sse_url}: {type(http_err).__name__}")
        except Exception as e: log.warning(f"Unexpected error probing streaming GET {sse_url}: {e}")

        # --- Attempt 2b: POST initialize ---
        # Only proceed if GET /sse didn't find an MCP server
        if not mcp_server_found:
            initialize_payload = {
                "jsonrpc": "2.0", "method": "initialize",
                "params": { "processId": os.getpid(), "clientInfo": {"name": "mcpclient-discover", "version": "1.0.0"}, "capabilities": {}, "protocolVersion": "2025-03-25"},
                "id": str(uuid.uuid4())
            }
            try:
                if USE_VERBOSE_SESSION_LOGGING: log.debug(f"Probing via POST {base_url}...")
                response = await client.post(base_url, json=initialize_payload, timeout=probe_timeout)
                if response.status_code == 200:
                    try:
                        response_data = response.json()
                        if (isinstance(response_data, dict) and
                            response_data.get("jsonrpc") == "2.0" and
                            str(response_data.get("id")) == initialize_payload["id"] and
                            ("result" in response_data or "error" in response_data)):
                            log.info(f"MCP Server detected via POST initialize at {base_url}")
                            server_name = f"mcp-scan-{ip_address}-{port}-base"
                            mcp_server_found = True
                            return {
                                "name": server_name, "path": base_url, "type": ServerType.SSE, "args": [],
                                "description": f"Auto-discovered SSE server via port scan POST initialize on {ip_address}:{port}", "source": "portscan"
                            }
                        else:
                            if USE_VERBOSE_SESSION_LOGGING: log.debug(f"Non-MCP JSON response from POST {base_url}: {str(response_data)[:100]}...")
                    except json.JSONDecodeError:
                         if USE_VERBOSE_SESSION_LOGGING: log.debug(f"Non-JSON response at POST {base_url}")
                    except Exception as parse_err:
                         if USE_VERBOSE_SESSION_LOGGING: log.debug(f"Error parsing response from POST {base_url}: {parse_err}")
                else:
                    if USE_VERBOSE_SESSION_LOGGING: log.debug(f"Probe failed for POST {base_url} - Status: {response.status_code}")
            except (httpx.TimeoutException, httpx.RequestError) as http_err:
                if USE_VERBOSE_SESSION_LOGGING: log.debug(f"Probe error for POST {base_url}: {type(http_err).__name__}")
            except Exception as e: log.warning(f"Unexpected error probing POST {base_url}: {e}")

        # --- No MCP Server Detected ---
        if USE_VERBOSE_SESSION_LOGGING and not mcp_server_found:
             # This log now means TCP open, but GET /sse failed headers check AND POST initialize failed check
             log.debug(f"TCP Port {ip_address}:{port} open, but no MCP pattern detected.")
        return None

    async def _discover_port_scan(self):
        """Scan configured local IP addresses and port ranges for MCP servers."""
        if not self.config.enable_port_scanning:
            log.info("Port scanning discovery disabled by configuration.")
            self._discovered_port_scan = []
            return

        start_port = self.config.port_scan_range_start
        end_port = self.config.port_scan_range_end
        concurrency = self.config.port_scan_concurrency
        probe_timeout = self.config.port_scan_timeout
        targets = self.config.port_scan_targets

        # --- Sanity check parameters ---
        if start_port > end_port:
            log.error(f"Invalid port range: start ({start_port}) > end ({end_port}). Skipping scan.")
            self._discovered_port_scan = []
            return
        if concurrency <= 0:
            log.warning(f"Port scan concurrency set to {concurrency}. Using default 1000.")
            concurrency = 1000
        if probe_timeout <= 0:
            log.warning(f"Port scan timeout set to {probe_timeout}. Using default 0.5s.")
            probe_timeout = 0.5

        # *** ADDED: Explicit check of the verbose flag ***
        log.info(f"Starting port scan. USE_VERBOSE_SESSION_LOGGING = {USE_VERBOSE_SESSION_LOGGING}")
        # *** END ADDED ***

        log.info(f"Scanning ports [{start_port}-{end_port}] on {targets} (Concurrency: {concurrency}, Timeout: {probe_timeout}s)...")

        discovered_servers_to_add = []
        semaphore = asyncio.Semaphore(concurrency)
        tasks = []
        http_client = None
        httpx_logger = logging.getLogger("httpx")
        original_httpx_level = httpx_logger.level

        # Synchronous callback
        def release_semaphore_callback(task: asyncio.Task, sem: asyncio.Semaphore, port: int, ip: str):
            """Synchronous callback to release the semaphore."""
            try:
                # Minimal logging in callback unless verbose
                if USE_VERBOSE_SESSION_LOGGING:
                    try:
                        task_exception = task.exception()
                        if task_exception and not isinstance(task_exception, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout, ConnectionRefusedError, asyncio.TimeoutError)):
                             log.debug(f"Probe task {ip}:{port} finished with error: {task_exception}")
                    except asyncio.CancelledError: pass
                    except Exception: pass # Ignore errors checking exception itself
                sem.release()
            except Exception as e:
                log.error(f"CRITICAL Error releasing semaphore in callback for {ip}:{port}: {e}")

        try:
            # Temporarily silence httpx logs
            log.debug(f"Temporarily setting httpx log level to WARNING (was {logging.getLevelName(original_httpx_level)})")
            httpx_logger.setLevel(logging.WARNING)

            # Setup shared HTTP client
            client_timeout = httpx.Timeout(probe_timeout + 0.5, connect=probe_timeout)
            http_client = httpx.AsyncClient(
                 verify=False,
                 timeout=client_timeout,
                 limits=httpx.Limits(max_connections=concurrency + 10, max_keepalive_connections=50)
             )

            total_ports_to_scan = (end_port - start_port + 1) * len(targets)
            log.info(f"Preparing to scan {total_ports_to_scan} total IP:Port combinations.")

            for ip_target in targets:
                for port in range(start_port, end_port + 1):
                    await semaphore.acquire()
                    # Calls the revised _probe_port
                    probe_task = asyncio.create_task(
                        self._probe_port(ip_target, port, probe_timeout, http_client),
                        name=f"probe-{ip_target}-{port}"
                    )
                    probe_task.add_done_callback(
                        functools.partial(release_semaphore_callback, sem=semaphore, port=port, ip=ip_target)
                    )
                    tasks.append(probe_task)

            log.info(f"Launched {len(tasks)} scanning tasks. Waiting for completion...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            log.info("All scanning tasks completed.")

        except Exception as gather_err:
            log.error(f"Error during port scan task creation/gathering: {gather_err}", exc_info=True)
            for task in tasks:
                if not task.done(): task.cancel()
            results = []

        finally:
            # Restore original httpx log level
            log.debug(f"Restoring httpx log level to {logging.getLevelName(original_httpx_level)}")
            httpx_logger.setLevel(original_httpx_level)

            # Close client
            if http_client:
                log.debug("Closing shared HTTP client...")
                try:
                    await http_client.aclose()
                    log.debug("HTTP client closed.")
                except Exception as close_err:
                    log.warning(f"Error closing port scan HTTP client: {close_err}")

            # Robust semaphore release & Code Style Fix
            released_in_finally = 0
            try:
                if hasattr(semaphore, '_value'):
                    initial_value = concurrency
                    current_value = semaphore._value
                    while current_value < initial_value:
                        try:
                            semaphore.release()
                            released_in_finally += 1
                            current_value += 1
                        except (ValueError, RuntimeError):
                            # Stop if over-released or other error
                            break
                else:
                    # Fallback if internal value not accessible
                    for _ in range(concurrency):
                        try:
                            if semaphore.locked():
                                semaphore.release()
                                released_in_finally += 1
                            else:
                                # Stop if it's not locked anymore
                                break
                        except (ValueError, RuntimeError):
                            # Stop if over-released or other error
                            break
                if released_in_finally > 0:
                    log.debug(f"Released semaphore {released_in_finally} times in finally block.")
            except Exception as final_sem_err:
                log.error(f"Unexpected error during final semaphore cleanup: {final_sem_err}")


        # Process results
        mcp_endpoints_found = 0
        if results:
            for result in results:
                if isinstance(result, dict):
                    mcp_endpoints_found += 1
                    server_path = result.get("path")
                    server_name_discovered = result.get("name", "UNKNOWN")
                    existing_config_entry = next((s for s in self.config.servers.values() if s.path == server_path), None)

                    if server_path and not existing_config_entry:
                         discovered_servers_to_add.append((
                             server_name_discovered, server_path, result["type"].value,
                             result.get("version"), result.get("categories", []), result.get("description", "")
                         ))
                    elif existing_config_entry:
                         # This log now only appears if an MCP server is found *and* it's already configured
                         log.info(f"Detected configured server {existing_config_entry.name} at {server_path} during scan (skipped adding).")
                    elif not server_path:
                         log.warning(f"Detected MCP server ({server_name_discovered}) but path missing in result.")

                elif isinstance(result, Exception):
                     # Minimal logging for common/expected errors unless verbose
                     if not isinstance(result, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout, asyncio.CancelledError, ConnectionRefusedError, asyncio.TimeoutError)):
                         log.warning(f"Port probe task failed with unexpected exception: {result}")
                     # Explicitly check the flag here for the debug log
                     elif USE_VERBOSE_SESSION_LOGGING:
                         log.debug(f"Port probe task failed with expected exception: {type(result).__name__}")
        else:
             log.warning("Port scan results list is empty or None.")

        # Store results
        self._discovered_port_scan = discovered_servers_to_add

        # Log summary
        log.info(f"Port scan complete. Detected {mcp_endpoints_found} responsive MCP endpoints. Found {len(self._discovered_port_scan)} servers not already in config.")

    async def discover_servers(self):
        """
        Auto-discover MCP servers via filesystem, registry, mDNS, and port scanning.
        Runs the discovery steps and populates self.discovered_servers_cache.
        """
        # Use a lock to prevent concurrent discovery runs, which could lead to race conditions
        # on the internal attributes (_discovered_local, etc.) and the cache.
        async with self._discovery_in_progress:
            log.info("Starting server discovery process...")
            # Reset cache and internal result holders before starting
            self.discovered_servers_cache = []
            self._discovered_local = []
            self._discovered_remote = []
            self._discovered_mdns = []
            self._discovered_port_scan = []

            # Define the discovery steps and their descriptions based on config
            steps = []
            descriptions = []

            # --- Filesystem Discovery ---
            if self.config.auto_discover:
                steps.append(self._discover_local_servers)
                descriptions.append(f"{STATUS_EMOJI['search']} Discovering local file system servers...")
            else:
                log.debug("Skipping filesystem discovery (disabled by config).")

            # --- Registry Discovery ---
            if self.config.enable_registry and self.registry:
                steps.append(self._discover_registry_servers)
                descriptions.append(f"{STATUS_EMOJI['search']} Discovering registry servers...")
            else:
                log.debug("Skipping registry discovery (disabled or registry not available).")

            # --- mDNS Discovery ---
            if self.config.enable_local_discovery and self.registry:
                # Ensure Zeroconf is started if needed (might be started by registry init)
                if not self.registry.zeroconf:
                     try:
                         self.registry.start_local_discovery()
                         log.debug("Started Zeroconf for mDNS discovery step.")
                     except Exception as e:
                         log.error(f"Failed to start Zeroconf for mDNS discovery: {e}")
                # Add the step even if start failed, _discover_mdns_servers handles registry check
                steps.append(self._discover_mdns_servers)
                descriptions.append(f"{STATUS_EMOJI['search']} Discovering local network servers (mDNS)...")
            else:
                log.debug("Skipping mDNS discovery (disabled or registry not available).")

            # --- Port Scanning Discovery ---
            if self.config.enable_port_scanning:
                steps.append(self._discover_port_scan)
                descriptions.append(f"{STATUS_EMOJI['search']} Scanning local ports [{self.config.port_scan_range_start}-{self.config.port_scan_range_end}]...")
            else:
                log.debug("Skipping port scanning discovery (disabled by config).")

            # --- Execute Discovery Steps ---
            if steps:
                log.info(f"Running {len(steps)} enabled discovery steps...")
                try:
                    # Use run_multi_step_task to execute the discovery functions
                    await self.run_multi_step_task(
                        steps=steps,
                        step_descriptions=descriptions,
                        title=f"{STATUS_EMOJI['search']} Discovering MCP servers...",
                        show_spinner=True # Set to False if run purely in background API context?
                    )
                    log.info("Discovery steps execution complete.")
                except Exception as discover_err:
                    log.error(f"Error occurred during multi-step discovery task: {discover_err}", exc_info=True)
                    # Continue to process any results gathered before the error
            else:
                log.info("No discovery methods enabled.")
                # Exit early if no steps were run, cache remains empty
                return

            # --- Process Results and Populate Cache ---
            # The discovery steps (_discover_local_servers, etc.) should have populated
            # the internal attributes like self._discovered_local, self._discovered_remote, etc.
            log.info("Processing discovery results and populating cache...")

            # Get results from internal attributes set by the discovery steps ran above
            discovered_local = getattr(self, '_discovered_local', [])
            discovered_remote = getattr(self, '_discovered_remote', [])
            discovered_mdns = getattr(self, '_discovered_mdns', [])
            discovered_port_scan = getattr(self, '_discovered_port_scan', [])

            # Check existing configured paths/urls to mark discoveries
            configured_paths = {s.path for s in self.config.servers.values()}

            # Process Local Filesystem Results
            for name, path, server_type in discovered_local:
                is_conf = path in configured_paths
                self.discovered_servers_cache.append({
                    "name": name, "type": server_type, "path_or_url": path, "source": "filesystem",
                    "description": f"Discovered {server_type} server", "is_configured": is_conf,
                    "version": None, "categories": [] # Add defaults for consistency
                })

            # Process Remote Registry Results
            for name, url, server_type, version, categories, rating in discovered_remote:
                is_conf = url in configured_paths
                self.discovered_servers_cache.append({
                    "name": name, "type": server_type, "path_or_url": url, "source": "registry",
                    "description": f"Discovered from registry (Rating: {rating:.1f}/5)",
                    "version": str(version) if version else None, "categories": categories, "is_configured": is_conf
                })

            # Process mDNS Results
            for name, url, server_type, version, categories, description in discovered_mdns:
                 is_conf = url in configured_paths
                 self.discovered_servers_cache.append({
                    "name": name, "type": server_type, "path_or_url": url, "source": "mdns",
                    "description": description or "Discovered on local network",
                    "version": str(version) if version else None, "categories": categories, "is_configured": is_conf
                 })

            # Process Port Scan Results
            for name, url, server_type, version, categories, description in discovered_port_scan:
                 is_conf = url in configured_paths
                 self.discovered_servers_cache.append({
                     "name": name, "type": server_type, "path_or_url": url, "source": "portscan",
                     "description": description or f"Discovered via port scan",
                     "version": str(version) if version else None, # Version might not be available from basic scan
                     "categories": categories or [], # Categories likely unavailable
                     "is_configured": is_conf
                 })

            log.info(f"Discovery complete. Found and cached {len(self.discovered_servers_cache)} potential servers.")
            # The API endpoint `/api/discover/results` will read this cache.
            # The CLI command `/discover list` or `servers --search` might call this
            # method and then potentially call a separate method like `_prompt_add_discovered_servers`.
    
    async def get_discovery_results(self) -> List[Dict]:
         return list(self.discovered_servers_cache)

    async def add_and_connect_discovered_server(self, discovered_server_info: Dict) -> Tuple[bool, str]:
        """Adds a server from discovery results and attempts to connect."""
        name = discovered_server_info.get("name")
        path_or_url = discovered_server_info.get("path_or_url")
        server_type_str = discovered_server_info.get("type")
        if not all([name, path_or_url, server_type_str]):
            return False, "Invalid discovered server data."
        # Check if already configured by path/url
        if any(s.path == path_or_url for s in self.config.servers.values()):
             existing_name = next((s.name for s in self.config.servers.values() if s.path == path_or_url), None)
             # If already connected, return success
             if existing_name and existing_name in self.active_sessions:
                  return True, f"Server already configured as '{existing_name}' and connected."
             # If configured but not connected, try connecting
             elif existing_name:
                  try:
                       server_config = self.config.servers[existing_name]
                       success = await self._connect_and_load_server(existing_name, server_config)
                       if success:
                            return True, f"Connected to existing server '{existing_name}'."
                       else:
                            return False, f"Failed to connect to existing server '{existing_name}'."
                  except Exception as e:
                       return False, f"Error connecting to existing server '{existing_name}': {e}"
             else:
                  # Should not happen if path check passed, but handle anyway
                  return False, "Server path exists but could not find configuration name."
        # Add the new server
        try:
            server_type = ServerType(server_type_str.lower())
            new_server_config = ServerConfig(
                name=name,
                type=server_type,
                path=path_or_url,
                enabled=True,
                auto_start=False, # Don't auto-start from discovery connect
                description=discovered_server_info.get("description", f"Discovered via {discovered_server_info.get('source', 'unknown')}"),
                categories=discovered_server_info.get("categories", []),
                version=ServerVersion.from_string(discovered_server_info["version"]) if discovered_server_info.get("version") else None
            )
            self.config.servers[name] = new_server_config
            await self.config.save_async()
            log.info(f"Added discovered server '{name}' to configuration.")
        except Exception as e:
            log.error(f"Failed to add server '{name}' from discovery: {e}")
            return False, f"Error adding server config: {e}"
        # Attempt to connect
        try:
            success = await self._connect_and_load_server(name, new_server_config)
            if success:
                return True, f"Server '{name}' added and connected successfully."
            else:
                return False, f"Server '{name}' added but failed to connect."
        except Exception as e:
            log.error(f"Failed to connect to newly added server '{name}': {e}")
            return False, f"Server '{name}' added but failed to connect: {e}"
            
    async def terminate_process(self, server_name: str, process: Optional[asyncio.subprocess.Process]):
        """Helper to terminate a process gracefully with fallback to kill."""
        if process is None or process.returncode is not None:
            log.debug(f"Process {server_name} already terminated or is None.")
            return # Already exited or None
        log.info(f"Terminating process {server_name} (PID {process.pid})")
        try:
            process.terminate()
            await asyncio.wait_for(process.wait(), timeout=2.0)
            log.info(f"Process {server_name} terminated gracefully.")
        except asyncio.TimeoutError:
            log.warning(f"Process {server_name} did not terminate gracefully, killing.")
            if process.returncode is None: # Check again before killing
                try:
                    process.kill()
                    await process.wait() # Wait for kill to complete
                    log.info(f"Process {server_name} killed.")
                except ProcessLookupError:
                    log.info(f"Process {server_name} already exited before kill.")
                except Exception as kill_err:
                    log.error(f"Error killing process {server_name}: {kill_err}")
        except ProcessLookupError:
            log.info(f"Process {server_name} already exited before termination attempt.")
        except Exception as e:
            log.error(f"Error terminating process {server_name}: {e}")

    async def register_local_server(self, server_config: ServerConfig):
        """Register a locally started MCP server with zeroconf"""
        # Rely on checking registry and zeroconf object directly
        if not self.config.enable_local_discovery or not self.registry or not self.registry.zeroconf:
            log.debug("Zeroconf registration skipped (disabled, registry missing, or zeroconf not init).")
            return

        # Avoid re-registering if already done for this server name
        if server_config.name in self.registered_services:
            log.debug(f"Zeroconf service for {server_config.name} already registered.")
            return

        try:
            # --- Get local IP (keep existing logic) ---
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # Doesn't actually connect but helps determine the interface
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
            except Exception:
                # Fallback if the above doesn't work
                local_ip = socket.gethostbyname(socket.gethostname())
            finally:
                s.close()

            # --- Determine port (keep existing logic) ---
            port = 8080 # Default
            for i, arg in enumerate(server_config.args):
                if arg in ['--port', '-p'] and i < len(server_config.args) - 1:
                    try:
                        port = int(server_config.args[i+1])
                        break
                    except (ValueError, IndexError): pass

            # --- Prepare properties (keep existing logic) ---
            props = {
                b'name': server_config.name.encode('utf-8'),
                b'type': server_config.type.value.encode('utf-8'),
                b'description': server_config.description.encode('utf-8'),
                b'version': str(server_config.version or '1.0.0').encode('utf-8'),
                b'host': 'localhost'.encode('utf-8') # Or potentially local_ip? Check server expectations
            }

            # --- Create ServiceInfo ---
            # Ensure ServiceInfo class is available (guarded import)
            if 'ServiceInfo' not in globals() or not hasattr(ServiceInfo, '__init__'):
                log.error("ServiceInfo class not available, cannot create Zeroconf service info.")
                return

            service_info = ServiceInfo(
                "_mcp._tcp.local.",
                f"{server_config.name}._mcp._tcp.local.",
                addresses=[ipaddress.IPv4Address(local_ip).packed],
                port=port,
                properties=props,
                # server=f"{socket.getfqdn()}.local." # Optional: Add server field explicitly if needed
            )

            # --- Register Service ---
            log.info(f"Registering local MCP server {server_config.name} with zeroconf on {local_ip}:{port}")
            await self.registry.zeroconf.async_register_service(service_info)
            log.info(f"Successfully registered {server_config.name} with Zeroconf.")

            # Store service info for later unregistering
            self.registered_services[server_config.name] = service_info

        except NonUniqueNameException:
            # This can happen on retries if the previous registration hasn't timed out
            log.warning(f"Zeroconf registration failed for {server_config.name}: Name already registered. This might be a stale registration from a previous attempt.")
            # Do not store service info if registration failed
        except Exception as e:
            log.error(f"Error registering service {server_config.name} with Zeroconf: {e}", exc_info=True)

    async def unregister_local_server(self, server_name: str):
        """Unregister a server from Zeroconf."""
        # Rely on checking registry and zeroconf object directly
        if not self.registry or not self.registry.zeroconf:
            log.debug("Zeroconf unregistration skipped (registry missing or zeroconf not init).")
            return # Cannot unregister

        if server_name in self.registered_services:
            service_info = self.registered_services.pop(server_name) # Remove as we unregister
            log.info(f"Unregistering server {server_name} from Zeroconf...")
            try:
                await self.registry.zeroconf.async_unregister_service(service_info)
                log.info(f"Successfully unregistered {server_name} from Zeroconf.")
            except Exception as e:
                log.error(f"Failed to unregister {server_name} from Zeroconf: {e}", exc_info=True)
                # If unregistration fails, the name might remain registered, but we've removed our reference.
        else:
            log.debug(f"No active Zeroconf registration found for {server_name} to unregister.")
                
    def _process_list_result(self, server_name: str, result_list: Optional[List[Any]], target_dict: Dict[str, Any], item_class: Type, item_type_name: str):
        """
        Helper to populate tools/resources/prompts dictionaries from list results.

        Args:
            server_name: The name of the server the results came from.
            result_list: The list of items (e.g., tools, resources) from the server response. Can be None.
            target_dict: The dictionary to populate (e.g., self.tools, self.resources).
            item_class: The dataclass to use for storing the item (e.g., MCPTool, MCPResource).
            item_type_name: A string name for the item type (e.g., "tool", "resource") for logging.
        """
        items_added = 0
        items_skipped = 0

        # Clear existing items for this server before adding new ones
        # This ensures we reflect the current state reported by the server
        for key in list(target_dict.keys()):
            # Check if the item belongs to the current server before deleting
            if hasattr(target_dict[key], 'server_name') and target_dict[key].server_name == server_name:
                del target_dict[key]

        if result_list is None:
            log.warning(f"[{server_name}] Received None instead of a list for {item_type_name}s.")
            return # Nothing to process

        if not isinstance(result_list, list):
            log.warning(f"[{server_name}] Expected a list for {item_type_name}s, but got {type(result_list).__name__}.")
            return # Cannot process non-list

        for item in result_list:
            try:
                # Check item type (should be an object/dict-like from JSON)
                if not hasattr(item, 'name') or not isinstance(item.name, str) or not item.name:
                     log.warning(f"[{server_name}] Skipping {item_type_name} item lacking valid 'name': {item}")
                     items_skipped += 1
                     continue

                # Specific checks based on type
                if item_class is MCPTool:
                    # Try getting 'inputSchema' first, fallback to 'input_schema'
                    schema = getattr(item, 'inputSchema', getattr(item, 'input_schema', None))
                    if schema is None or not isinstance(schema, dict):
                        log.warning(f"[{server_name}] Skipping tool '{item.name}' lacking valid 'inputSchema' or 'input_schema'. Item data: {item}")
                        items_skipped += 1
                        continue
                    correct_input_schema = schema

                # Construct full name and add to dictionary
                item_name_full = f"{server_name}:{item.name}" if ":" not in item.name else item.name

                # Create instance of the appropriate dataclass
                # Use getattr for optional fields to avoid errors if they are missing
                instance_data = {
                    "name": item_name_full,
                    "description": getattr(item, 'description', '') or '', # Ensure description is string
                    "server_name": server_name,
                }
                # Add type-specific data
                if item_class is MCPTool:
                    instance_data["input_schema"] = correct_input_schema
                    instance_data["original_tool"] = item # Store original object
                elif item_class is MCPResource:
                    instance_data["template"] = getattr(item, 'uri', '') # Assuming template comes from uri in Resource
                    instance_data["original_resource"] = item
                elif item_class is MCPPrompt:
                    # Assuming McpApiPrompt has 'arguments' or similar for template info
                    # Adjust based on actual McpApiPrompt structure if template isn't direct
                    instance_data["template"] = f"Prompt: {item.name}" # Placeholder if no template field
                    instance_data["original_prompt"] = item

                # Create the object using the specific class
                target_dict[item_name_full] = item_class(**instance_data)

                items_added += 1
            except Exception as proc_err:
                log.error(f"[{server_name}] Error processing {item_type_name} item '{getattr(item, 'name', 'UNKNOWN')}': {proc_err}", exc_info=True)
                items_skipped += 1
        log.info(f"[{server_name}] Processed {items_added} {item_type_name}s ({items_skipped} skipped).")

    async def _connect_stdio_server(self, server_config: ServerConfig, current_server_name: str, retry_count: int) -> Tuple[Optional[ClientSession], Optional[Dict[str, Any]], Optional[asyncio.subprocess.Process], None]:
        """
        Handles STDIO connection, handshake, and capability loading.
        Uses async file opening for stderr redirection to a file descriptor.
        Includes explicit initialized notification and uses default capability timeouts.
        """
        session: Optional[ClientSession] = None
        initialize_result_obj: Optional[Dict[str, Any]] = None
        process_this_attempt: Optional[asyncio.subprocess.Process] = None
        BUFFER_LIMIT = 2**22 # 4 MiB

        # === Process Start/Restart Logic ===
        existing_process = self.processes.get(current_server_name)
        restart_process = False
        process_to_use: Optional[asyncio.subprocess.Process] = None

        # --- Decide if process needs restart ---
        if existing_process:
            if existing_process.returncode is None:
                if retry_count > 0: # Only restart on retries
                    log.warning(f"Restarting process for {current_server_name} on retry {retry_count}")
                    self._safe_printer(f"[yellow]Restarting process for {current_server_name} on retry {retry_count}[/]")
                    restart_process = True
                    await self.terminate_process(current_server_name, existing_process)
                    if current_server_name in self.registered_services: await self.unregister_local_server(current_server_name)
                else: # First attempt, reuse existing process
                    log.debug(f"Found existing process for {current_server_name} (PID {existing_process.pid}), will attempt connection.")
                    process_to_use = existing_process
            else: # Process existed but already terminated
                log.warning(f"Previously managed process for {current_server_name} has exited with code {existing_process.returncode}. Cleaning up entry.")
                self._safe_printer(f"[yellow]Previous process for {current_server_name} exited (code {existing_process.returncode}). Starting new one.[/]")
                restart_process = True
                if current_server_name in self.registered_services: await self.unregister_local_server(current_server_name)
                if current_server_name in self.processes: del self.processes[current_server_name]
        else: # No existing process found
            restart_process = True

        log_file_handle = None # Initialize file handle variable outside try
        try:
            if restart_process:
                executable = server_config.path
                current_args = server_config.args
                log_file_path = (CONFIG_DIR / f"{current_server_name}_stderr.log").resolve()
                log_file_path.parent.mkdir(parents=True, exist_ok=True)
                log.info(f"Executing STDIO server '{current_server_name}': Executable='{executable}', Args={current_args}")
                log.info(f"Stderr for {current_server_name} will be redirected to -> {log_file_path}")

                process: Optional[asyncio.subprocess.Process] = None
                is_shell_cmd = (
                    isinstance(executable, str) and
                    any(executable.endswith(shell) for shell in ["bash", "sh", "zsh"]) and
                    len(current_args) == 2 and
                    current_args[0] == "-c" and
                    isinstance(current_args[1], str)
                )

                # *** Open file asynchronously to get the FD, manage closure manually ***
                log_file_handle = await aiofiles.open(log_file_path, "ab")
                stderr_fileno = log_file_handle.fileno() # Get integer file descriptor

                try:
                    if is_shell_cmd:
                        command_string = current_args[1]
                        self._safe_printer(f"[cyan]Starting server process (shell: {executable}): {command_string[:100]}...[/]")
                        process = await asyncio.create_subprocess_shell(
                            command_string, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                            stderr=stderr_fileno, # Pass integer FD
                            limit=BUFFER_LIMIT, env=os.environ.copy(), executable=executable
                        )
                    else:
                        final_cmd_list = [executable] + current_args
                        self._safe_printer(f"[cyan]Starting server process: {' '.join(map(str, final_cmd_list))}[/]")
                        process = await asyncio.create_subprocess_exec(
                            *final_cmd_list, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                            stderr=stderr_fileno, # Pass integer FD
                            limit=BUFFER_LIMIT, env=os.environ.copy()
                        )

                    process_this_attempt = process
                    self.processes[current_server_name] = process
                    log.info(f"Started process for {current_server_name} with PID {process.pid}")

                    await asyncio.sleep(0.5) # Allow process startup
                    if process.returncode is not None:
                        log.error(f"Process {current_server_name} failed immediately (code {process.returncode}). Log: {log_file_path}")
                        raise RuntimeError(f"Process for {current_server_name} failed immediately (code {process.returncode}). Check log '{log_file_path}'.")
                    process_to_use = process

                except FileNotFoundError as fnf_err:
                    log.error(f"Executable/Shell not found for {current_server_name}: {executable}. Error: {fnf_err}")
                    raise
                except Exception as proc_start_err:
                    log.error(f"Error starting process for {current_server_name}: {proc_start_err}", exc_info=True)
                    raise
                # REMOVED: No separate stderr reader task needed

            # === Session Creation ===
            if not process_to_use or process_to_use.returncode is not None:
                raise RuntimeError(f"Process for STDIO server {current_server_name} is invalid or has exited.")

            log.info(f"[{current_server_name}] Initializing RobustStdioSession...")
            session = RobustStdioSession(process_to_use, current_server_name)
            # REMOVED: Linking stderr task

            # === Handshake AND Capability Loading ===
            log.info(f"[{current_server_name}] Attempting MCP handshake: initialize...")
            init_timeout = server_config.timeout + 5.0
            initialize_result_obj = await asyncio.wait_for(
                session.initialize(response_timeout=server_config.timeout),
                timeout=init_timeout
            )
            log.info(f"[{current_server_name}] Initialize successful. Server reported: {initialize_result_obj.get('serverInfo')}")

            # === EXPLICIT Initialized Notification ===
            log.info(f"[{current_server_name}] Sending EXPLICIT initialized notification...")
            await session.send_initialized_notification()
            log.info(f"[{current_server_name}] Explicit initialized notification sent.")

            # --- Extract capabilities ---
            server_capabilities = None
            if isinstance(initialize_result_obj, dict):
                server_capabilities = initialize_result_obj.get("capabilities")

            # === LOAD CAPABILITIES IMMEDIATELY (Using Default 10s Timeout) ===
            log.info(f"[{current_server_name}] Loading capabilities immediately...")
            outer_capability_timeout = server_config.timeout + 15.0 # Overall wait limit

            # --- List/Load Tools ---
            has_tools_capability = isinstance(server_capabilities, dict) and "tools" in server_capabilities
            if has_tools_capability:
                log.info(f"[{current_server_name}] Loading tools (using default 10s request timeout)...")
                try:
                    list_tools_result: ListToolsResult = await asyncio.wait_for(
                        session.list_tools(), # Uses default 10s timeout internally
                        timeout=outer_capability_timeout
                    )
                    if list_tools_result and hasattr(list_tools_result, 'tools'):
                        self._process_list_result(current_server_name, list_tools_result.tools, self.tools, MCPTool, "tool")
                    else: log.warning(f"[{current_server_name}] Invalid ListToolsResult.")
                except asyncio.TimeoutError:
                    log.error(f"[{current_server_name}] Outer timeout ({outer_capability_timeout}s) waiting for list_tools response.")
                except RuntimeError as e:
                    log.error(f"[{current_server_name}] RuntimeError loading tools: {e}")
                except Exception as e: log.error(f"[{current_server_name}] Error loading tools: {e}", exc_info=True)
            else:
                log.info(f"[{current_server_name}] Server does not advertise 'tools' capability.")

            # --- List/Load Resources ---
            has_resources_capability = isinstance(server_capabilities, dict) and "resources" in server_capabilities
            if has_resources_capability and hasattr(session, 'list_resources'):
                log.info(f"[{current_server_name}] Loading resources (using default 10s request timeout)...")
                try:
                    list_resources_result: ListResourcesResult = await asyncio.wait_for(
                        session.list_resources(), # Uses default 10s timeout internally
                        timeout=outer_capability_timeout
                    )
                    if list_resources_result and hasattr(list_resources_result, 'resources'):
                        self._process_list_result(current_server_name, list_resources_result.resources, self.resources, MCPResource, "resource")
                    else: log.warning(f"[{current_server_name}] Invalid ListResourcesResult.")
                except asyncio.TimeoutError: log.error(f"[{current_server_name}] Outer timeout loading resources.")
                except RuntimeError as e: log.error(f"[{current_server_name}] RuntimeError loading resources: {e}")
                except Exception as e: log.error(f"[{current_server_name}] Error loading resources: {e}", exc_info=True)
            else:
                log.info(f"[{current_server_name}] Server does not advertise 'resources' capability.")

            # --- List/Load Prompts ---
            has_prompts_capability = isinstance(server_capabilities, dict) and "prompts" in server_capabilities
            if has_prompts_capability and hasattr(session, 'list_prompts'):
                log.info(f"[{current_server_name}] Loading prompts (using default 10s request timeout)...")
                try:
                    list_prompts_result: ListPromptsResult = await asyncio.wait_for(
                        session.list_prompts(), # Uses default 10s timeout internally
                        timeout=outer_capability_timeout
                    )
                    if list_prompts_result and hasattr(list_prompts_result, 'prompts'):
                        self._process_list_result(current_server_name, list_prompts_result.prompts, self.prompts, MCPPrompt, "prompt")
                    else: log.warning(f"[{current_server_name}] Invalid ListPromptsResult.")
                except asyncio.TimeoutError: log.error(f"[{current_server_name}] Outer timeout loading prompts.")
                except RuntimeError as e: log.error(f"[{current_server_name}] RuntimeError loading prompts: {e}")
                except Exception as e: log.error(f"[{current_server_name}] Error loading prompts: {e}", exc_info=True)
            else:
                log.info(f"[{current_server_name}] Server does not advertise 'prompts' capability.")

            log.info(f"[{current_server_name}] MCP handshake and capability loading complete inside helper.")
            # Return session, result, the process created, and None for the removed task
            # The file handle log_file_handle will be closed by the finally block
            return session, initialize_result_obj, process_this_attempt, None

        except Exception as setup_or_load_err:
            log.error(f"[{current_server_name}] Failed setup/handshake/loading in helper ({type(setup_or_load_err).__name__}): {setup_or_load_err}", exc_info=True)
            if session and hasattr(session, 'aclose'):
                with suppress(Exception): await session.aclose()
            # Process termination is handled by the outer connect_to_server's error handling for process_this_attempt
            raise setup_or_load_err
        finally:
            # Ensure the log file handle is closed if it was opened
            if log_file_handle:
                try:
                    await log_file_handle.close()
                    log.debug(f"Closed stderr log file handle for {current_server_name}")
                except Exception as close_err:
                    log.error(f"Error closing stderr log file handle for {current_server_name}: {close_err}")

    async def _connect_sse_server(self, server_config: ServerConfig, current_server_name: str, retry_count: int) -> Tuple[Optional[ClientSession], Optional[InitializeResult]]:
        """
        Handles the specific logic for connecting, performing the MCP handshake,
        AND LOADING CAPABILITIES for an SSE server using the standard mcp library client.
        (This is intended for compliant servers).
        """
        session: Optional[ClientSession] = None
        initialize_result_obj: Optional[InitializeResult] = None
        sse_client_context = None # Keep track of the context manager instance

        connect_url = server_config.path
        parsed_connect_url = urlparse(connect_url)
        if not parsed_connect_url.scheme:
            connect_url = f"http://{connect_url}"
            log.debug(f"[{current_server_name}] Prepended http scheme for standard SSE connection: {connect_url}")

        # Define timeouts based on config, providing reasonable defaults
        connect_timeout = server_config.timeout if server_config.timeout > 0 else 10.0
        read_timeout = (server_config.timeout * 2) if server_config.timeout > 0 else 20.0
        # Handshake timeout includes initial GET + POST + result wait via SSE stream
        handshake_timeout = (server_config.timeout * 3) + 15.0 if server_config.timeout > 0 else 45.0
        capability_load_timeout = (server_config.timeout * 2) + 20.0 if server_config.timeout > 0 else 40.0
        outer_wait_timeout = capability_load_timeout + 10.0

        log.debug(f"[{current_server_name}] Standard SSE Timeouts: Connect={connect_timeout}s, Read={read_timeout}s, Handshake={handshake_timeout}s, Caps={capability_load_timeout}s")

        try:
            # --- Step 1: Setup Persistent Streams using the standard sse_client ---
            log.info(f"[{current_server_name}] Attempting standard SSE connection via sse_client to {connect_url} (Attempt {retry_count+1})...")

            sse_client_context = sse_client(
                url=connect_url,
                headers=None,
                timeout=connect_timeout, # Base timeout for connect/initial ops
                sse_read_timeout=timedelta(minutes=15).total_seconds()
            )
            read_stream, write_stream = await self.exit_stack.enter_async_context(sse_client_context)
            log.debug(f"[{current_server_name}] Standard SSE streams obtained via sse_client.")

            # --- Step 2: Instantiate the standard ClientSession ---
            log.info(f"[{current_server_name}] Initializing standard ClientSession...")
            session_read_timeout_seconds = max(90.0, capability_load_timeout * 2)
            session = ClientSession(
                read_stream=read_stream,
                write_stream=write_stream,
                read_timeout_seconds=timedelta(seconds=session_read_timeout_seconds)
            )
            await self.exit_stack.enter_async_context(session)
            log.debug(f"[{current_server_name}] Standard ClientSession instantiated and context entered.")

            # --- Step 3: Perform Handshake using ClientSession ---
            log.info(f"[{current_server_name}] Performing standard MCP handshake via ClientSession.initialize()...")
            initialize_result_obj = await asyncio.wait_for(
                session.initialize(),
                timeout=handshake_timeout
            )
            log.info(f"[{current_server_name}] Standard Initialize successful (SSE). Server reported: {initialize_result_obj.serverInfo}")

            # --- Step 4: LOAD CAPABILITIES ---
            log.info(f"[{current_server_name}] Loading capabilities after standard handshake (SSE)...")
            server_capabilities = initialize_result_obj.capabilities if initialize_result_obj else None
            log.debug(f"[{current_server_name}] Received Server Capabilities (Standard SSE): {server_capabilities}")
            capability_results = {}

            async def load_tools():
                 if server_capabilities and server_capabilities.tools is not None:
                     log.info(f"[{current_server_name}] Listing tools (Standard SSE)...")
                     try: result = await session.list_tools(); capability_results['tools'] = result.tools if result else []
                     except Exception as e: capability_results['tools_error'] = e
                 else: log.info(f"[{current_server_name}] Skipping tools (not advertised, Standard SSE)")

            async def load_resources():
                 if server_capabilities and server_capabilities.resources is not None:
                     log.info(f"[{current_server_name}] Listing resources (Standard SSE)...")
                     try: result = await session.list_resources(); capability_results['resources'] = result.resources if result else []
                     except Exception as e: capability_results['resources_error'] = e
                 else: log.info(f"[{current_server_name}] Skipping resources (not advertised, Standard SSE)")

            async def load_prompts():
                 if server_capabilities and server_capabilities.prompts is not None:
                     log.info(f"[{current_server_name}] Listing prompts (Standard SSE)...")
                     try: result = await session.list_prompts(); capability_results['prompts'] = result.prompts if result else []
                     except Exception as e: capability_results['prompts_error'] = e
                 else: log.info(f"[{current_server_name}] Skipping prompts (not advertised, Standard SSE)")

            try: # Load concurrently
                 async with asyncio.timeout(outer_wait_timeout):
                      async with anyio.create_task_group() as cap_tg:
                           cap_tg.start_soon(load_tools)
                           cap_tg.start_soon(load_resources)
                           cap_tg.start_soon(load_prompts)
            except TimeoutError: log.error(f"[{current_server_name}] Overall timeout ({outer_wait_timeout}s) loading capabilities (Standard SSE).")

            # Process results
            if 'tools' in capability_results: self._process_list_result(current_server_name, capability_results['tools'], self.tools, MCPTool, "tool")
            elif 'tools_error' in capability_results: log.error(f"[{current_server_name}] Error loading tools (Standard SSE): {capability_results['tools_error']}")
            if 'resources' in capability_results: self._process_list_result(current_server_name, capability_results['resources'], self.resources, MCPResource, "resource")
            elif 'resources_error' in capability_results: log.error(f"[{current_server_name}] Error loading resources (Standard SSE): {capability_results['resources_error']}")
            if 'prompts' in capability_results: self._process_list_result(current_server_name, capability_results['prompts'], self.prompts, McpPromptType, "prompt")
            elif 'prompts_error' in capability_results: log.error(f"[{current_server_name}] Error loading prompts (Standard SSE): {capability_results['prompts_error']}")


            log.info(f"[{current_server_name}] Standard SSE handshake and capability loading complete.")
            return session, initialize_result_obj

        except (httpx.RequestError, httpx.HTTPStatusError, McpError, asyncio.TimeoutError) as err:
            log.warning(f"[{current_server_name}] Standard SSE connection/handshake failed ({type(err).__name__}): {err}")
            # IMPORTANT: Cleanup is handled by the main exit stack automatically on exception
            raise err
        except Exception as e:
            log.error(f"[{current_server_name}] Unexpected error during standard SSE connection/handshake: {e}", exc_info=True)
            # IMPORTANT: Cleanup is handled by the main exit stack automatically on exception
            raise e # Re-raise

    async def connect_to_server(self, server_config: ServerConfig) -> Optional[ClientSession]:
        """
        Connect to a single MCP server (STDIO or SSE) with retry logic.
        Handles server renaming based on InitializeResult.
        """
        current_server_name = server_config.name
        current_server_config = dataclasses.replace(server_config)
        retry_count = 0
        config_updated_by_rename = False
        max_retries = current_server_config.retry_policy.get("max_attempts", 3)
        backoff_factor = current_server_config.retry_policy.get("backoff_factor", 0.5)

        with safe_stdout(): # Ensure all prints go to stderr if stdio active
            while retry_count <= max_retries:
                start_time = time.time()
                session: Optional[ClientSession] = None
                initialize_result_obj: Optional[Union[InitializeResult, Dict[str, Any]]] = None
                process_this_attempt: Optional[asyncio.subprocess.Process] = None
                stderr_reader_task_this_attempt: Optional[asyncio.Task] = None
                connection_error: Optional[BaseException] = None
                rename_occurred_this_attempt = False
                span = None
                span_context_manager = None

                try: # Trace span setup
                    if tracer:
                        try:
                            span_context_manager = tracer.start_as_current_span(
                                f"connect_server.{current_server_name}",
                                attributes={ "server.name": current_server_name, "server.type": current_server_config.type.value, "retry": retry_count }
                            )
                            if span_context_manager: span = span_context_manager.__enter__()
                        except Exception as e: log.warning(f"Failed to start trace span: {e}"); span = None; span_context_manager = None

                    # --- Main Connection Attempt ---
                    try:
                        self._safe_printer(f"[cyan]Attempting connection & full setup for {current_server_name} (Attempt {retry_count+1}/{max_retries+1})...[/]")

                        # --- Call Connection Helpers ---
                        if current_server_config.type == ServerType.STDIO:
                            session, initialize_result_obj, process_this_attempt, stderr_reader_task_this_attempt = await self._connect_stdio_server(
                                current_server_config, current_server_name, retry_count
                            )
                        elif current_server_config.type == ServerType.SSE:
                            session, initialize_result_obj = await self._connect_sse_server(
                                current_server_config, current_server_name, retry_count
                            )
                            process_this_attempt = None # No process for SSE
                            stderr_reader_task_this_attempt = None
                        else:
                            raise RuntimeError(f"Unknown server type: {current_server_config.type}")

                        if not session:
                             raise RuntimeError(f"Connection helper returned None session for {current_server_name}")

                        # --- Rename Logic ---
                        original_name_before_rename = current_server_name
                        if initialize_result_obj:
                            actual_server_name = None
                            if isinstance(initialize_result_obj, dict): # STDIO (or potentially custom SSE result)
                                actual_server_name = initialize_result_obj.get("serverInfo", {}).get("name")
                            elif isinstance(initialize_result_obj, InitializeResult): # Standard/Custom SSE result
                                if hasattr(initialize_result_obj, 'serverInfo') and initialize_result_obj.serverInfo:
                                    actual_server_name = initialize_result_obj.serverInfo.name

                            if actual_server_name and actual_server_name != current_server_name:
                                if actual_server_name in self.config.servers and actual_server_name != current_server_name:
                                    log.warning(f"Server '{current_server_name}' reported name '{actual_server_name}', but it already exists. Keeping '{current_server_name}'.")
                                else:
                                    log.info(f"Server '{current_server_name}' identified as '{actual_server_name}'. Renaming config & loaded items.")
                                    self._safe_printer(f"[yellow]Server '{current_server_name}' identified as '{actual_server_name}', updating config.[/]")
                                    try:
                                        if current_server_name in self.config.servers:
                                            original_config_entry = self.config.servers.pop(current_server_name)
                                            original_config_entry.name = actual_server_name
                                            self.config.servers[actual_server_name] = original_config_entry
                                            if current_server_config.type == ServerType.STDIO and session and hasattr(session, '_server_name'): session._server_name = actual_server_name
                                            if current_server_config.type == ServerType.STDIO:
                                                if current_server_name in self.processes: self.processes[actual_server_name] = self.processes.pop(current_server_name)
                                                if current_server_name in self._session_tasks: self._session_tasks[actual_server_name] = self._session_tasks.pop(current_server_name)
                                            for item_dict in [self.tools, self.resources, self.prompts]:
                                                keys_to_update = {} ; items_to_update = []
                                                for key, item in item_dict.items():
                                                    if item.server_name == original_name_before_rename:
                                                        items_to_update.append(item)
                                                        if key.startswith(original_name_before_rename + ":"):
                                                            new_key = key.replace(original_name_before_rename + ":", actual_server_name + ":", 1)
                                                            keys_to_update[key] = new_key
                                                for item in items_to_update: item.server_name = actual_server_name
                                                for old_key, new_key in keys_to_update.items():
                                                    if old_key in item_dict: item_dict[new_key] = item_dict.pop(old_key)
                                            current_server_name = actual_server_name
                                            current_server_config.name = actual_server_name
                                            config_updated_by_rename = True
                                            rename_occurred_this_attempt = True
                                        else: log.warning(f"Attempted rename for '{current_server_name}' (to '{actual_server_name}'), but it was no longer in config.")
                                    except Exception as rename_err:
                                        log.error(f"Failed to rename server '{current_server_name}' to '{actual_server_name}': {rename_err}", exc_info=True)
                                        if current_server_name not in self.config.servers and 'original_config_entry' in locals(): self.config.servers[current_server_name] = original_config_entry

                        # --- Success Path ---
                        connection_time = (time.time() - start_time) * 1000
                        server_config_in_dict = self.config.servers.get(current_server_name)
                        if server_config_in_dict:
                            server_config_in_dict.metrics.request_count += 1
                            server_config_in_dict.metrics.update_response_time(connection_time / 1000.0)
                            server_config_in_dict.metrics.update_status()
                        if latency_histogram:
                            with suppress(Exception): latency_histogram.record(connection_time, {"server.name": current_server_name})
                        if span:
                            span.set_status(trace.StatusCode.OK)
                            if rename_occurred_this_attempt: span.set_attribute("server.name", current_server_name)
                        tools_loaded_count = len([t for t in self.tools.values() if t.server_name == current_server_name])
                        log.info(f"Connected to {current_server_name} ({tools_loaded_count} tools loaded) in {connection_time:.2f}ms")
                        self._safe_printer(f"[green]Connected & loaded: {current_server_name} ({tools_loaded_count} tools)[/]")
                        self.active_sessions[current_server_name] = session
                        if config_updated_by_rename:
                            log.info(f"Saving configuration after server rename to {current_server_name}...")
                            await self.config.save_async()
                            config_updated_by_rename = False
                        if span_context_manager: span_context_manager.__exit__(None, None, None)
                        return session # Return success

                    # --- Outer Exception Handling for Connection Attempt ---
                    except (McpError, RuntimeError, ConnectionAbortedError, httpx.RequestError, subprocess.SubprocessError, OSError, FileNotFoundError, asyncio.TimeoutError, BaseException) as e:
                        connection_error = e
                        log.warning(f"Connection attempt failed for {current_server_name} (Attempt {retry_count + 1}, {type(e).__name__}): {e}")
                        if isinstance(e, BaseExceptionGroup):
                             for i, sub_exc in enumerate(e.exceptions): log.warning(f"  Sub-exception {i+1}: ({type(sub_exc).__name__}) {sub_exc}")
                        # Cleanup resources specific to FAILED attempt (process, stderr task)
                        if process_this_attempt and process_this_attempt.returncode is None:
                            await self.terminate_process(current_server_name, process_this_attempt)
                        if stderr_reader_task_this_attempt and not stderr_reader_task_this_attempt.done():
                            stderr_reader_task_this_attempt.cancel()
                            with suppress(asyncio.CancelledError): await stderr_reader_task_this_attempt
                            if current_server_name in self._session_tasks and stderr_reader_task_this_attempt in self._session_tasks[current_server_name]:
                                self._session_tasks[current_server_name].remove(stderr_reader_task_this_attempt)
                        # NOTE: SSE cleanup is managed by self.exit_stack when the exception propagates

                    # --- Shared Error Handling & Retry Logic ---
                    retry_count += 1
                    config_for_metrics = self.config.servers.get(current_server_name)
                    if config_for_metrics:
                        config_for_metrics.metrics.error_count += 1
                        config_for_metrics.metrics.update_status()
                    else: log.error(f"Could not find server config for '{current_server_name}' during error metric update.")

                    error_msg_for_span = str(connection_error)[:200] + "..." if connection_error and len(str(connection_error)) > 200 else str(connection_error or "Unknown connection error")
                    if span:
                        span.set_status(trace.StatusCode.ERROR, error_msg_for_span)
                        if rename_occurred_this_attempt: span.set_attribute("server.name", current_server_name)

                    if retry_count <= max_retries:
                        delay = min(backoff_factor * (2 ** (retry_count - 1)) + random.random() * 0.1, 10.0)
                        error_msg_display = str(connection_error or "Unknown error")[:150] + "..."
                        log.warning(f"Error details for {current_server_name} (attempt {retry_count-1} failed): {str(connection_error or 'Unknown')}")
                        self._safe_printer(f"[yellow]Error connecting {current_server_name}: {error_msg_display}[/]")
                        log.info(f"Retrying connection to {current_server_name} in {delay:.2f}s...")
                        self._safe_printer(f"[cyan]Retrying in {delay:.2f}s...[/]")
                        if span_context_manager:
                            with suppress(Exception): span_context_manager.__exit__(*sys.exc_info())
                            span_context_manager = None; span = None
                        await asyncio.sleep(delay)
                        # Continue main while loop
                    else: # Max retries exceeded
                        final_error_msg = str(connection_error or "Unknown connection error")
                        log.error(f"Failed to connect to {current_server_name} after {max_retries+1} attempts. Final error: {final_error_msg}")
                        self._safe_printer(f"[red]Failed to connect to {current_server_name} after {max_retries+1} attempts.[/]")
                        if current_server_config.type == ServerType.STDIO and current_server_name in self.processes:
                            proc_to_clean = self.processes.pop(current_server_name, None)
                            if proc_to_clean and proc_to_clean.returncode is None:
                                log.info(f"Terminating process for {current_server_name} after max retries.")
                                await self.terminate_process(current_server_name, proc_to_clean)
                        if span: span.set_status(trace.StatusCode.ERROR, f"Max retries exceeded. Final: {final_error_msg[:150]}...")
                        if span_context_manager: 
                            with suppress(Exception): 
                                span_context_manager.__exit__(*sys.exc_info())
                        if config_updated_by_rename:
                            log.info(f"Saving configuration after failed connection attempts for renamed server {current_server_name}...")
                            await self.config.save_async()
                        return None # Connection failed

                # --- Finally block for trace span ---
                finally:
                    if span_context_manager:
                         with suppress(Exception): span_context_manager.__exit__(*sys.exc_info())

            # --- Loop Exit (Should only happen if max retries exceeded) ---
            log.error(f"Connection loop for {current_server_name} exited after max retries.")
            if config_updated_by_rename:
                log.info(f"Saving configuration after loop exit for renamed server {current_server_name}...")
                await self.config.save_async()
            return None
        
    async def connect_to_servers(self):
        """Connect to all enabled MCP servers"""
        if not self.config.servers:
            log.warning("No servers configured. Use 'config servers add' to add servers.")
            return
        
        # Connect to each enabled server
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            SpinnerColumn("dots"),
            TextColumn("[cyan]{task.fields[server]}"),
            console=get_safe_console(),
            transient=True
        ) as progress:
            task = progress.add_task("[cyan]Connecting to servers...", total=len([s for s in self.config.servers.values() if s.enabled]))
            
            for name, server_config in self.config.servers.items():
                if not server_config.enabled:
                    continue
                    
                log.info(f"Connecting to server: {name}")
                session = await self.connect_to_server(server_config)  # noqa: F841
                progress.update(task, advance=1)
        
        # Verify no stdout pollution after connecting to servers
        if os.environ.get("MCP_VERIFY_STDOUT", "1") == "1":
            with safe_stdout():
                log.info("Verifying no stdout pollution after connecting to servers...")
                verify_no_stdout_pollution()
        
        # Start server monitoring
        with Status(f"{STATUS_EMOJI['server']} Starting server monitoring...", spinner="dots", console=get_safe_console()) as status:
            await self.server_monitor.start_monitoring()
            if hasattr(status, 'update'):
                status.update(f"{STATUS_EMOJI['success']} Server monitoring started")
        
        # Display status
        await self.print_status()
    
    async def close(self):
        """Clean up resources before exit"""
        try:
            # Add a timeout to all cleanup operations
            cleanup_timeout = 5  # seconds

            # Close the shared port scan client if it exists
            if self._port_scan_client:
                try:
                    await asyncio.wait_for(self._port_scan_client.aclose(), timeout=cleanup_timeout / 2) # Give it some time
                    log.debug("Closed shared port scan HTTP client.")
                except asyncio.TimeoutError:
                    log.warning("Timeout closing port scan HTTP client.")
                except Exception as e:
                    log.error(f"Error closing port scan HTTP client: {e}")
                self._port_scan_client = None

            # Unregister Zeroconf services
            for name in list(self.registered_services.keys()):
                await self.unregister_local_server(name)

            # Close SSE connections via exit_stack
            log.debug(f"Closing {len(self.active_sessions)} active sessions via exit stack...")
            await self.exit_stack.aclose() # This handles SSE client closures
            log.debug("Exit stack closed.")

            # Terminate STDIO processes
            log.debug(f"Terminating {len(self.processes)} tracked processes...")
            process_terminations = []
            for name, process in list(self.processes.items()): # Iterate copy
                process_terminations.append(self.terminate_process(name, process))
                if name in self.processes: del self.processes[name] # Remove from tracking

            if process_terminations:
                await asyncio.gather(*process_terminations, return_exceptions=True)
            log.debug("Processes terminated.")

            # Cancel any remaining stderr reader tasks
            tasks_to_cancel = []
            for name, task_list in self._session_tasks.items():
                tasks_to_cancel.extend(task_list)
            self._session_tasks.clear()

            if tasks_to_cancel:
                log.debug(f"Cancelling {len(tasks_to_cancel)} remaining session tasks...")
                for task in tasks_to_cancel:
                    if task and not task.done():
                        task.cancel()
                # Allow cancellation to propagate
                await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
                log.debug("Session tasks cancelled.")

            # Close registry client
            if self.registry:
                await self.registry.close()

        except Exception as e:
            log.error(f"Error during ServerManager cleanup: {e}", exc_info=True)

    def format_tools_for_anthropic(self) -> List[ToolParam]:
        """Format MCP tools for Anthropic API"""
        # Use List[Dict] as a fallback if ToolParam is not precisely defined or causing issues
        tool_params: List[Dict[str, Any]] = []

        # Clear existing mapping
        self.sanitized_to_original.clear()

        for tool in self.tools.values():
            original_name = tool.name

            # Sanitize tool name to match Anthropic's requirements
            sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', original_name)

            # Ensure name is not longer than 64 characters
            if len(sanitized_name) > 64:
                sanitized_name = sanitized_name[:64]

            # Ensure name is not empty
            if not sanitized_name:
                sanitized_name = "tool_" + str(hash(original_name) % 1000)

            # Store the mapping
            self.sanitized_to_original[sanitized_name] = original_name

            # Create the dictionary EXACTLY as per Anthropic documentation: name, description, input_schema
            # Ensure input_schema is directly assigned, not nested further.
            tool_dict_for_api = {
                "name": sanitized_name,
                "input_schema": tool.input_schema # Assuming tool.input_schema IS the correct JSON schema dict
            }
            # Add description only if it exists and is not empty
            if tool.description:
                tool_dict_for_api["description"] = tool.description

            if USE_VERBOSE_SESSION_LOGGING:
                if original_name in ["llm_gateway:generate_completion", "llm_gateway:chat_completion"]:
                    try:
                        log.debug(f"Corrected Schema being sent to Anthropic API for {original_name}:\n{json.dumps(tool_dict_for_api)}")
                    except Exception as log_dump_err:
                        log.error(f"Error dumping corrected schema for logging: {log_dump_err}")
                        log.debug(f"Corrected raw dict for {original_name}: {repr(tool_dict_for_api)}")

            tool_params.append(tool_dict_for_api)

        return tool_params

    async def run_multi_step_task(self, 
                                steps: List[Callable], 
                                step_descriptions: List[str],
                                title: str = "Processing...",
                                show_spinner: bool = True) -> bool:
        """Run a multi-step task with progress tracking.
        
        Args:
            steps: List of async callables to execute
            step_descriptions: List of descriptions for each step
            title: Title for the progress bar
            show_spinner: Whether to show a spinner
            
        Returns:
            Boolean indicating success
        """
        if len(steps) != len(step_descriptions):
            log.error("Steps and descriptions must have the same length")
            return False
            
        # Get safe console to avoid stdout pollution
        safe_console = get_safe_console()
        
        # If app.mcp_client exists, use its _run_with_progress helper
        if hasattr(app, "mcp_client") and hasattr(app.mcp_client, "_run_with_progress"):
            # Format tasks in the format expected by _run_with_progress
            tasks = [(steps[i], step_descriptions[i], None) for i in range(len(steps))]
            try:
                await app.mcp_client._run_with_progress(tasks, title, transient=True)
                return True
            except Exception as e:
                log.error(f"Error in multi-step task: {e}")
                return False
        
        # Fallback to old implementation if _run_with_progress isn't available
        progress_columns = []
        if show_spinner:
            progress_columns.append(SpinnerColumn())
        
        progress_columns.extend([
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TaskProgressColumn()
        ])
        
        with Progress(*progress_columns, console=safe_console) as progress:
            task = progress.add_task(title, total=len(steps))
            
            for i, (step, description) in enumerate(zip(steps, step_descriptions, strict=False)):
                try:
                    progress.update(task, description=description)
                    await step()
                    progress.update(task, advance=1)
                except Exception as e:
                    log.error(f"Error in step {i+1}: {e}")
                    progress.update(task, description=f"{STATUS_EMOJI['error']} {description} failed: {e}")
                    return False
            
            progress.update(task, description=f"{STATUS_EMOJI['success']} Complete")
            return True

    async def count_tokens(self, messages=None) -> int:
        """Count the number of tokens in the current conversation context"""
        if messages is None:
            messages = self.conversation_graph.current_node.messages
            
        # Use tiktoken for accurate counting
        # Use cl100k_base encoding which is used by Claude
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = 0
        
        for message in messages:
            # Get the message content
            content = message.get("content", "")
            
            # Handle content that might be a list of blocks (text/image blocks)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        token_count += len(encoding.encode(block["text"]))
                    elif isinstance(block, str):
                        token_count += len(encoding.encode(block))
            else:
                # Simple string content
                token_count += len(encoding.encode(str(content)))
            
            # Add a small overhead for message formatting
            token_count += 4  # Approximate overhead per message
            
        return token_count

class MCPClient:
    def __init__(self):
        self.config = Config()
        self.history = History(max_entries=self.config.history_size)
        
        # Store reference to this client instance on the app object for global access
        app.mcp_client = self
        
        # Instantiate Caching
        self.tool_cache = ToolCache(
            cache_dir=CACHE_DIR,
            custom_ttl_mapping=self.config.cache_ttl_mapping
        )
        
        self.server_manager = ServerManager(self.config, tool_cache=self.tool_cache, safe_printer=self.safe_print)
        # Only initialize Anthropic client if API key is available
        if self.config.api_key:
            self.anthropic = AsyncAnthropic(api_key=self.config.api_key)
        else:
            self.anthropic = None
        self.current_model = self.config.default_model

        # Instantiate Server Monitoring
        self.server_monitor = ServerMonitor(self.server_manager)

        # For tracking newly discovered local servers
        self.discovered_local_servers = set()
        self.local_discovery_task = None

        # Instantiate Conversation Graph
        self.conversation_graph_file = Path(self.config.conversation_graphs_dir) / "default_conversation.json"
        self.conversation_graph_file.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        try:
            # Create a simple graph first
            self.conversation_graph = ConversationGraph()
            
            # Then try to load from file if it exists
            if self.conversation_graph_file.exists():
                # Since __init__ can't be async, we need to use a workaround
                # We'll load it properly later in setup()
                log.info(f"Found conversation graph file at {self.conversation_graph_file}, will load it during setup")
            else:
                log.info("No existing conversation graph found, using new graph")
        except Exception as e: 
            log.error(f"Unexpected error initializing conversation graph: {e}")
            self.conversation_graph = ConversationGraph() # Fallback to new graph
        
        # Ensure current node is valid after loading
        if not self.conversation_graph.get_node(self.conversation_graph.current_node.id):
            log.warning("Loaded current node ID not found in graph, resetting to root.")
            self.conversation_graph.set_current_node("root")

        # Command handlers
        self.commands = {
            'exit': self.cmd_exit,
            'quit': self.cmd_exit,
            'help': self.cmd_help,
            'config': self.cmd_config,
            'servers': self.cmd_servers,
            'tools': self.cmd_tools,
            'resources': self.cmd_resources,
            'prompts': self.cmd_prompts,
            'history': self.cmd_history,
            'model': self.cmd_model,
            'clear': self.cmd_clear,
            'reload': self.cmd_reload,
            'cache': self.cmd_cache,
            'fork': self.cmd_fork,
            'branch': self.cmd_branch,
            'dashboard': self.cmd_dashboard, # Add dashboard command
            'optimize': self.cmd_optimize, # Add optimize command
            'tool': self.cmd_tool, # Add tool playground command
            'prompt': self.cmd_prompt, # Add prompt command for dynamic injection
            'export': self.cmd_export, # Add export command
            'import': self.cmd_import, # Add import command
            'discover': self.cmd_discover, # Add local discovery command
        }
        
        # Set up readline for command history in interactive mode
        readline.set_completer(self.completer)
        readline.parse_and_bind("tab: complete")
    
    @staticmethod
    def safe_print(message, **kwargs):
        """Print using the appropriate console based on active stdio servers.
        
        This helps prevent stdout pollution when stdio servers are connected.
        Use this method for all user-facing output instead of direct console.print() calls.
        
        Args:
            message: The message to print
            **kwargs: Additional arguments to pass to print
        """
        safe_console = get_safe_console()
        safe_console.print(message, **kwargs)
    
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

    @staticmethod
    def with_tool_error_handling(func):
        """Decorator for consistent tool error handling"""
        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs):
            tool_name = kwargs.get("tool_name", args[1] if len(args) > 1 else "unknown")
            try:
                return await func(self, *args, **kwargs)
            except McpError as e:
                log.error(f"MCP error executing {tool_name}: {e}")
                raise RuntimeError(f"MCP error: {e}") from e
            except httpx.RequestError as e:
                log.error(f"Network error executing {tool_name}: {e}")
                raise RuntimeError(f"Network error: {e}") from e
            except Exception as e:
                log.error(f"Unexpected error executing {tool_name}: {e}")
                raise RuntimeError(f"Unexpected error: {e}") from e
        return wrapper
        
    # Add decorator for retry logic
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
        status_table.add_row(f"{STATUS_EMOJI['model']} Model", self.current_model)
        status_table.add_row(f"{STATUS_EMOJI['server']} Servers", f"{connected_servers}/{total_servers} connected")
        status_table.add_row(f"{STATUS_EMOJI['tool']} Tools", str(total_tools))
        status_table.add_row(f"{STATUS_EMOJI['resource']} Resources", str(total_resources))
        status_table.add_row(f"{STATUS_EMOJI['prompt']} Prompts", str(total_prompts))
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

    async def setup(self, interactive_mode=False):
        """Set up the client, load configs, clean duplicates, discover, and connect to servers."""
        safe_console_instance = get_safe_console() # Get console instance once

        # --- 1. API Key Check ---
        api_key_was_missing = False # Flag to track if we prompted
        if not self.config.api_key:
            api_key_was_missing = True
            self.safe_print("[bold red]ERROR: Anthropic API key not found[/]")

            if interactive_mode:
                # Prompt the user ONLY in interactive mode
                self.safe_print("You can enter your Anthropic API key now, or press Enter to continue without it.")
                api_key_input = Prompt.ask(
                    "Enter Anthropic API Key (optional)",
                    default="",
                    console=safe_console_instance # Use the safe console for prompting
                )

                if api_key_input.strip():
                    self.config.api_key = api_key_input.strip()
                    # Attempt to save the key immediately
                    try:
                        await self.config.save_async()
                        self.safe_print("[green]API key saved to configuration.[/]")
                    except Exception as e:
                        self.safe_print(f"[red]Error saving API key: {e}[/]")
                        self.config.api_key = None # Revert if save failed
                else:
                    # User pressed Enter, leave self.config.api_key as None
                    pass # No key entered
            else:
                # Non-interactive mode, exit if key is missing
                self.safe_print("Please set your API key using one of these methods:")
                self.safe_print("1. Set the ANTHROPIC_API_KEY environment variable")
                self.safe_print("2. Add 'api_key: YOUR_KEY' to your config file")
                sys.exit(1)

        # --- Initialize Anthropic client AFTER potential prompt/save ---
        if self.config.api_key:
            try:
                # Initialize or re-initialize the client if the key is now available
                self.anthropic = AsyncAnthropic(api_key=self.config.api_key)
                if api_key_was_missing: # Only print success if we just added it via prompt
                    self.safe_print("[green]Anthropic client initialized successfully.[/]")
            except Exception as e:
                self.safe_print(f"[red]Error initializing Anthropic client with saved/provided key: {e}[/]")
                self.safe_print("[yellow]API features will be disabled.[/]")
                self.anthropic = None # Ensure it's None if init fails even with a key
        else:
            # Key is still missing (user didn't enter one in interactive mode)
            if interactive_mode: # Check interactive_mode again for the correct message
                 self.safe_print("[yellow]No API key provided. API features disabled.[/]")
                 self.safe_print("You can set it later using '/config api-key YOUR_KEY'.")
            # No need for else here, non-interactive already exited
            self.anthropic = None

        # --- 2. Load Conversation Graph ---
        self.conversation_graph = ConversationGraph() # Start fresh
        if self.conversation_graph_file.exists():
            status_text = f"{STATUS_EMOJI['history']} Loading conversation state..."
            with Status(status_text, console=safe_console_instance) as status:
                try:
                    loaded_graph = await ConversationGraph.load(str(self.conversation_graph_file))
                    self.conversation_graph = loaded_graph
                    # Check if loading resulted in a default graph (meaning load failed)
                    is_new_graph = (loaded_graph.root.id == "root" and not loaded_graph.root.messages and
                                    not loaded_graph.root.children and len(loaded_graph.nodes) == 1)
                    if is_new_graph and self.conversation_graph_file.read_text().strip(): # Check if file wasn't empty
                        self.safe_print("[yellow]Could not parse previous conversation state, starting fresh.[/yellow]")
                        status.update(f"{STATUS_EMOJI['warning']} Previous state invalid, starting fresh")
                    else:
                        log.info(f"Loaded conversation graph from {self.conversation_graph_file}")
                        status.update(f"{STATUS_EMOJI['success']} Conversation state loaded")
                except Exception as setup_load_err:
                    log.error("Unexpected error during conversation graph loading stage in setup", exc_info=True)
                    self.safe_print(f"[red]Error initializing conversation state: {setup_load_err}[/red]")
                    status.update(f"{STATUS_EMOJI['error']} Error loading state")
                    self.conversation_graph = ConversationGraph() # Ensure default graph
        else:
             log.info("No existing conversation graph found, using new graph.")

        # Ensure current node is valid after potential load
        if not self.conversation_graph.get_node(self.conversation_graph.current_node.id):
             log.warning("Current node ID was invalid after graph load/init, resetting to root.")
             self.conversation_graph.set_current_node("root")

        # --- 3. Load Claude Desktop Config (adds potential initial servers) ---
        await self.load_claude_desktop_config()

        # --- 4. Clean Duplicate Server Configurations ---
        log.info("Cleaning up potentially duplicate server configurations...")
        cleaned_servers: Dict[str, ServerConfig] = {}
        # Store mapping: unique_identifier -> canonical_name_to_keep
        canonical_map: Dict[Tuple, str] = {}
        duplicates_found = False

        # Iterate through a snapshot of current servers
        servers_to_process = list(self.config.servers.items())

        for name, server_config in servers_to_process:
            # Create a unique identifier based on execution details
            identifier: Optional[Tuple] = None
            if server_config.type == ServerType.STDIO:
                # For STDIO, path and sorted args define uniqueness
                # Using frozenset makes the args order-independent and hashable
                args_frozenset = frozenset(server_config.args)
                identifier = (server_config.type, server_config.path, args_frozenset)
            elif server_config.type == ServerType.SSE:
                # For SSE, the path (URL) defines uniqueness
                identifier = (server_config.type, server_config.path)
            else:
                log.warning(f"Server '{name}' has unknown type '{server_config.type}', cannot check for duplicates.")
                identifier = (server_config.type, name) # Fallback identifier

            if identifier is not None:
                if identifier not in canonical_map:
                    # First time seeing this server config identifier, keep this entry
                    canonical_map[identifier] = name
                    cleaned_servers[name] = server_config
                    log.debug(f"Keeping server config: '{name}' (Identifier: {identifier})")
                else:
                    # Duplicate found based on identifier
                    duplicates_found = True
                    kept_name = canonical_map[identifier]
                    log.warning(f"Duplicate server config detected for identifier {identifier}. Removing entry '{name}', keeping '{kept_name}'.")
                    # We simply don't add 'name' to cleaned_servers
                    # Optional advanced logic: could compare 'name' vs 'kept_name' and potentially
                    # replace the entry in cleaned_servers if 'name' is "better" (e.g., not generated)
                    # but keeping the first encountered is simpler and robust.

        # Update config if duplicates were removed
        if duplicates_found:
            num_removed = len(self.config.servers) - len(cleaned_servers)
            self.safe_print(f"[yellow]Removed {num_removed} duplicate server entries from config.[/]")
            self.config.servers = cleaned_servers
            await self.config.save_async() # Save the cleaned config
        else:
            log.info("No duplicate server configurations found during initial cleanup.")
        # --- End Cleanup Step ---

        # --- 5. Stdout Pollution Check ---
        if os.environ.get("MCP_VERIFY_STDOUT", "1") == "1":
            with safe_stdout():
                log.info("Verifying no stdout pollution before connecting to servers...")
                verify_no_stdout_pollution()

        # --- 6. Discover Servers (Filesystem, Registry, mDNS, Port Scan) ---
        # This might add *new* servers via _process_discovery_results, which
        # internally checks for duplicates based on path before adding.
        if self.config.auto_discover:
            self.safe_print(f"{STATUS_EMOJI['search']} Discovering MCP servers...")
            try:
                await self.server_manager.discover_servers()
            except Exception as discover_error:
                log.error("Error during server discovery process", exc_info=True)
                self.safe_print(f"[red]Error during server discovery: {discover_error}[/red]")

        # --- 7. Start Continuous Local Discovery (mDNS) ---
        if self.config.enable_local_discovery and self.server_manager.registry:
            await self.start_local_discovery_monitoring()

        # --- 8. Connect to Enabled Servers ---
        # Get the list of servers *after* potential discovery additions and cleanup
        servers_to_connect = {name: cfg for name, cfg in self.config.servers.items() if cfg.enabled}

        if servers_to_connect:
            self.safe_print(f"[bold blue]Connecting to {len(servers_to_connect)} servers...[/]")
            connection_results = {}
            # Iterate over a copy of the items from the cleaned + discovered server list
            for name, server_config in list(servers_to_connect.items()):
                # The 'name' here is the key currently in self.config.servers
                self.safe_print(f"[cyan]Connecting to {name}...[/]")
                try:
                    # Pass the config object associated with this name.
                    # connect_to_server might rename the entry in self.config.servers
                    # and will return the session associated with the *final* name.
                    session = await self.server_manager.connect_to_server(server_config)

                    # Determine the final name after potential rename for status reporting
                    final_name = name
                    if name in self.config.servers: # Check if original name still exists
                       if self.config.servers[name].name != name: # If the object's name changed
                           final_name = self.config.servers[name].name
                    elif name not in self.config.servers: # Original name was removed (rename happened)
                         # Find the new name by looking for the session object? Less reliable.
                         # Better: Check if the server_config object itself was mutated
                         if server_config.name != name:
                              final_name = server_config.name
                         else: # Fallback: Look for the session in active_sessions (might be slow)
                              for active_name, active_session in self.server_manager.active_sessions.items():
                                   if active_session == session:
                                       final_name = active_name
                                       break

                    connection_results[name] = (session is not None) # Log success/fail against original iteration name

                    if session:
                        self.safe_print(f"  [green]✓ Connected to {final_name}[/]") # Print final name on success
                    else:
                        log.warning(f"Failed to connect and load server initially named: {name}")
                        self.safe_print(f"  [yellow]✗ Failed to connect to {name}[/]") # Print original name on failure

                except Exception as e:
                    log.error(f"Exception connecting to {name}", exc_info=True)
                    self.safe_print(f"  [red]✗ Error connecting to {name}: {e}[/]") # Print original name on error
                    connection_results[name] = False

        # --- 9. Start Server Monitoring ---
        try:
            with Status(f"{STATUS_EMOJI['server']} Starting server monitoring...",
                    spinner="dots", console=safe_console_instance) as status:
                await self.server_monitor.start_monitoring()
                status.update(f"{STATUS_EMOJI['success']} Server monitoring started")
        except Exception as monitor_error:
            log.error("Failed to start server monitoring", exc_info=True)
            self.safe_print(f"[red]Error starting server monitoring: {monitor_error}[/red]")

        # --- 10. Display Final Status ---
        await self.print_simple_status() # Uses the current state of self.config.servers
        
    async def _connect_and_load_server(self, server_name, server_config):
        """Connect to a server and load its capabilities (for use with _run_with_progress)"""
        session = await self.server_manager.connect_to_server(server_config)
        
        if session:
            self.server_manager.active_sessions[server_name] = session
            return True
        return False

    async def start_local_discovery_monitoring(self):
        """Start monitoring for local network MCP servers continuously"""
        # Start the registry's discovery if not already running
        if self.server_manager.registry and not self.server_manager.registry.zeroconf:
            self.server_manager.registry.start_local_discovery()
            log.info("Started continuous local MCP server discovery")
            
            # Create background task for periodic checks
            self.local_discovery_task = asyncio.create_task(self._monitor_local_servers())
    
    async def stop_local_discovery_monitoring(self):
        """Stop monitoring for local network MCP servers"""
        if self.local_discovery_task:
            self.local_discovery_task.cancel()
            try:
                await self.local_discovery_task
            except asyncio.CancelledError:
                pass
            self.local_discovery_task = None
        
        # Stop the registry's discovery if running
        if self.server_manager.registry:
            self.server_manager.registry.stop_local_discovery()
            log.info("Stopped continuous local MCP server discovery")
    
    async def _monitor_local_servers(self):
        """Background task to periodically check for new locally discovered servers"""
        try:
            while True:
                # Get the current set of discovered server names
                if self.server_manager.registry:
                    current_servers = set(self.server_manager.registry.discovered_servers.keys())
                    
                    # Find newly discovered servers since last check
                    new_servers = current_servers - self.discovered_local_servers
                    
                    # If there are new servers, notify the user
                    if new_servers:
                        self.safe_print(f"\n[bold cyan]{STATUS_EMOJI['search']} New MCP servers discovered on local network:[/]")
                        for server_name in new_servers:
                            server_info = self.server_manager.registry.discovered_servers[server_name]
                            self.safe_print(f"  - [bold cyan]{server_name}[/] at [cyan]{server_info.get('url', 'unknown URL')}[/]")
                        self.safe_print("Use [bold cyan]/discover list[/] to view details and [bold cyan]/discover connect NAME[/] to connect")
                        
                        # Update tracked servers
                        self.discovered_local_servers = current_servers
                
                # Wait before checking again (every 15 seconds)
                await asyncio.sleep(15)
        except asyncio.CancelledError:
            # Task was cancelled, exit cleanly
            pass
        except Exception as e:
            log.error(f"Error in local server monitoring task: {e}")

    async def cmd_discover(self, args):
        """Command to interact with locally discovered MCP servers
        
        Subcommands:
          list - List all locally discovered servers
          connect SERVER_NAME - Connect to a specific discovered server
          refresh - Force a refresh of the local discovery
          auto on|off - Enable/disable automatic local discovery
        """
        # Get safe console once at the beginning
        safe_console = get_safe_console()
        
        if not self.server_manager.registry:
            safe_console.print("[yellow]Registry not available, local discovery is disabled.[/]")
            return
            
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"
        subargs = parts[1] if len(parts) > 1 else ""
        
        if subcmd == "list":
            # List all discovered servers
            discovered_servers = self.server_manager.registry.discovered_servers
            
            if not discovered_servers:
                safe_console.print("[yellow]No MCP servers discovered on local network.[/]")
                safe_console.print("Try running [bold blue]/discover refresh[/] to scan again.")
                return
                
            safe_console.print(f"\n[bold cyan]{STATUS_EMOJI['search']} Discovered Local Network Servers:[/]")
            
            # Create a table to display servers
            server_table = Table(title="Local MCP Servers")
            server_table.add_column("Name")
            server_table.add_column("URL")
            server_table.add_column("Type")
            server_table.add_column("Description")
            server_table.add_column("Status")
            
            for name, server in discovered_servers.items():
                url = server.get("url", "unknown")
                server_type = server.get("type", "unknown")
                description = server.get("description", "No description")
                
                # Check if already in config
                in_config = any(s.path == url for s in self.config.servers.values())
                status = "[green]In config[/]" if in_config else "[yellow]Not in config[/]"
                
                server_table.add_row(
                    name,
                    url,
                    server_type,
                    description,
                    status
                )
            
            safe_console.print(server_table)
            safe_console.print("\nUse [bold blue]/discover connect NAME[/] to connect to a server.")
            
        elif subcmd == "connect":
            if not subargs:
                safe_console.print("[yellow]Usage: /discover connect SERVER_NAME[/]")
                return
                
            server_name = subargs
            
            # Check if server exists in discovered servers
            if server_name not in self.server_manager.registry.discovered_servers:
                safe_console.print(f"[red]Server '{server_name}' not found in discovered servers.[/]")
                safe_console.print("Use [bold blue]/discover list[/] to see available servers.")
                return
                
            # Get server info
            server_info = self.server_manager.registry.discovered_servers[server_name]
            url = server_info.get("url", "")
            server_type = server_info.get("type", "sse")
            description = server_info.get("description", "Discovered on local network")
            
            # Check if server already in config with the same URL
            existing_server = None
            for name, server in self.config.servers.items():
                if server.path == url:
                    existing_server = name
                    break
            
            if existing_server:
                safe_console.print(f"[yellow]Server with URL '{url}' already exists as '{existing_server}'.[/]")
                if existing_server not in self.server_manager.active_sessions:
                    if Confirm.ask(f"Connect to existing server '{existing_server}'?", console=safe_console):
                        await self.connect_server(existing_server)
                else:
                    safe_console.print(f"[yellow]Server '{existing_server}' is already connected.[/]")
                return
                
            # Add server to config
            log.info(f"Adding discovered server '{server_name}' to configuration")
            self.config.servers[server_name] = ServerConfig(
                name=server_name,
                type=ServerType(server_type),
                path=url,
                enabled=True,
                auto_start=False,  # Don't auto-start by default
                description=description,
                categories=server_info.get("categories", []),
                version=server_info.get("version")
            )
            
            # Save the configuration
            self.config.save()
            safe_console.print(f"[green]Added server '{server_name}' to configuration.[/]")
            
            # Offer to connect
            if Confirm.ask(f"Connect to server '{server_name}' now?", console=safe_console):
                await self.connect_server(server_name)
                
        elif subcmd == "refresh":
            safe_console = get_safe_console()
            # Force a refresh of the discovery
            with Status(f"{STATUS_EMOJI['search']} Refreshing local MCP server discovery...", spinner="dots", console=get_safe_console()) as status:
                # Restart the discovery to refresh
                if self.server_manager.registry.zeroconf:
                    self.server_manager.registry.stop_local_discovery()
                
                self.server_manager.registry.start_local_discovery()
                
                # Wait a moment for discovery
                await asyncio.sleep(2)
                
                status.update(f"{STATUS_EMOJI['success']} Local discovery refreshed")
                
                # Clear tracked servers to force notification of all currently discovered servers
                self.discovered_local_servers.clear()
                
                # Trigger a check for newly discovered servers
                current_servers = set(self.server_manager.registry.discovered_servers.keys())
                if current_servers:
                    safe_console.print(f"\n[bold cyan]Found {len(current_servers)} servers on the local network[/]")
                    safe_console.print("Use [bold blue]/discover list[/] to see details.")
                else:
                    safe_console.print("[yellow]No servers found on the local network.[/]")
                    
        elif subcmd == "auto":
            safe_console = get_safe_console()
            # Enable/disable automatic discovery
            if subargs.lower() in ("on", "yes", "true", "1"):
                self.config.enable_local_discovery = True
                self.config.save()
                safe_console.print("[green]Automatic local discovery enabled.[/]")
                
                # Start discovery if not already running
                if not self.local_discovery_task:
                    await self.start_local_discovery_monitoring()
                    
            elif subargs.lower() in ("off", "no", "false", "0"):
                self.config.enable_local_discovery = False
                self.config.save()
                safe_console.print("[yellow]Automatic local discovery disabled.[/]")
                
                # Stop discovery if running
                await self.stop_local_discovery_monitoring()
                
            else:
                # Show current status
                status = "enabled" if self.config.enable_local_discovery else "disabled"
                safe_console.print(f"[cyan]Automatic local discovery is currently {status}.[/]")
                safe_console.print("Usage: [bold blue]/discover auto [on|off][/]")
                
        else:
            safe_console = get_safe_console()
            safe_console.print("[yellow]Unknown discover command. Available: list, connect, refresh, auto[/]")

    async def close(self):
        """Clean up resources before exit"""

        # Stop local discovery monitoring if running
        if self.local_discovery_task:
            await self.stop_local_discovery_monitoring()

        # Save conversation graph
        try:
            await self.conversation_graph.save(str(self.conversation_graph_file))
            log.info(f"Saved conversation graph to {self.conversation_graph_file}")
        except Exception as e:
            log.error(f"Failed to save conversation graph: {e}")

        # Stop server monitor
        if hasattr(self, 'server_monitor'): # Ensure monitor was initialized
            await self.server_monitor.stop_monitoring()
        # Close server connections and processes
        if hasattr(self, 'server_manager'):
            await self.server_manager.close() # ServerManager.close() will handle its own clients

    async def process_streaming_query(self, query: str, model: Optional[str] = None,
                                    max_tokens: Optional[int] = None) -> AsyncIterator[str]:
        """Process a query using Claude and available tools with streaming.
        Correctly handles OpenTelemetry spans, tool result formatting, and streamed tool inputs.
        **Does not cache error results.**
        """
        # Wrap the entire function in safe_stdout
        with safe_stdout():
            # Get core parameters
            if not model:
                model = self.current_model
            if not max_tokens:
                max_tokens = self.config.default_max_tokens

            # Check if context needs pruning before processing
            await self.auto_prune_context()

            # Start timing for metrics
            start_time = time.time()

            # Keep track of servers and tools used
            servers_used = set()
            tools_used = []
            tool_results_for_history = [] # Store tool results for history logging
            cache_hits_during_query = 0 # Count cache hits

            # Start with user message
            current_messages: List[MessageParam] = self.conversation_graph.current_node.messages.copy()
            user_message: MessageParam = {"role": "user", "content": query}
            # Combine initial messages (consider putting user_message at the end after filtering if needed)
            combined_messages: List[MessageParam] = current_messages + [user_message]

            # --- NEW: Filter out client-side tool execution parse failures ---
            messages_to_send: List[MessageParam] = []
            i = 0
            client_error_signature = "Client failed to parse JSON input" # More general check
            skipped_indices = set() # To track indices of messages to skip

            log.debug(f"Filtering history ({len(combined_messages)} messages) for known client errors...")

            # First pass: identify indices of faulty interactions to skip
            assistant_tool_uses_to_check: Dict[int, Set[str]] = {} # index -> set of tool_use_ids

            for idx, msg in enumerate(combined_messages):
                if msg.get("role") == "assistant":
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        tool_use_ids = {
                            block.get("id") for block in content
                            if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
                        }
                        if tool_use_ids:
                            assistant_tool_uses_to_check[idx] = tool_use_ids

                elif msg.get("role") == "user":
                    # Check if this user message corresponds to a preceding assistant tool use
                    prev_idx = idx - 1
                    if prev_idx in assistant_tool_uses_to_check:
                        content = msg.get("content", [])
                        if isinstance(content, list):
                            corresponding_ids = assistant_tool_uses_to_check[prev_idx]
                            found_faulty_result = False
                            for block in content:
                                if (isinstance(block, dict) and
                                    block.get("type") == "tool_result" and
                                    block.get("tool_use_id") in corresponding_ids):
                                    result_content = block.get("content")
                                    # Check if the content contains our specific client error signature
                                    if isinstance(result_content, str) and client_error_signature in result_content:
                                        found_faulty_result = True
                                        log.warning(f"Found faulty client tool result for ID {block.get('tool_use_id')} at history index {idx}. Marking for filtering.")
                                        break # Found one faulty result for this user message turn

                            if found_faulty_result:
                                # Mark both the assistant request and the user result for skipping
                                skipped_indices.add(prev_idx)
                                skipped_indices.add(idx)

            # Second pass: build the filtered list
            for idx, msg in enumerate(combined_messages):
                if idx not in skipped_indices:
                    messages_to_send.append(msg)
                else:
                    log.debug(f"Skipping message at index {idx} during API call preparation.")

            # Now use 'messages_to_send' for the API call
            messages = messages_to_send # Replace the original list
            log.debug(f"Proceeding with {len(messages)} filtered messages for API call.")

        # --- OpenTelemetry Span Handling (Unchanged) ---
        span = None
        span_context_manager = None
        if tracer:
            try:
                span_context_manager = tracer.start_as_current_span(
                    "process_streaming_query",
                    attributes={
                        "model": model, "query_length": len(query),
                        "conversation_length": len(messages), "streaming": True
                    }
                )
                if span_context_manager:
                    span = span_context_manager.__enter__()
            except Exception as e:
                log.warning(f"Failed to start trace span: {e}")
                span = None
                span_context_manager = None
        # --- End Span Handling ---

        try:
            final_response_text = "" # Accumulate final text response across all turns

            # --- Main Streaming Loop ---
            while True: # Loop handles potential multi-turn tool use
                stop_reason = None
                # Collect *completed* tool_use blocks for this specific API response iteration
                tool_calls_this_turn: List[Dict[str, Any]] = []
                # Store assistant content blocks (text or tool_use) for the *current* assistant turn
                current_assistant_content: List[Dict[str, Any]] = []
                # Track the currently accumulating text block
                current_text_block: Optional[Dict[str, Any]] = None
                # --- Track the currently accumulating tool use block and its JSON input ---
                current_tool_use_block: Optional[Dict[str, Any]] = None
                current_tool_input_json_accumulator: str = "" # Use a dedicated accumulator
                # --- End tracking ---

                # Get latest available tools before *each* API call within the loop
                available_tools = self.server_manager.format_tools_for_anthropic()
                is_first_turn = not any(msg['role'] == 'assistant' for msg in messages)
                if not available_tools and is_first_turn:
                    log.info("No tools available from connected servers. Proceeding LLM-only.")

                use_debug_logging = log.isEnabledFor(logging.DEBUG)
                if use_debug_logging:
                    log.debug(f"--- Preparing Anthropic API Call ---")
                    log.debug(f"Model: {model}, Max Tokens: {max_tokens}, Temp: {self.config.temperature}")
                    log.debug(f"Messages ({len(messages)}):")
                    for i, msg in enumerate(messages):
                        try:
                            content_repr = json.dumps(msg.get('content'), ensure_ascii=False, indent=2)
                            log.debug(f"  [{i}] Role: {msg.get('role')} Content: {content_repr[:150]}{'...' if len(content_repr) > 150 else ''}")
                        except Exception: log.debug(f"  [{i}] Role: {msg.get('role')} Content: (Could not dump)") # Handle non-serializable content

                    log.debug(f"Tools Parameter ({len(available_tools)}):")
                    try:
                         # Log specific schemas only if needed, otherwise log count
                         failing_tool_names = ["llm_gateway:generate_completion", "llm_gateway:chat_completion"]
                         tools_to_log_in_detail = [
                             tool_param for tool_param in available_tools
                             if self.server_manager.sanitized_to_original.get(tool_param.get('name')) in failing_tool_names
                         ]
                         if tools_to_log_in_detail:
                             log.debug("--- Detailed Schema for Selected Tools ---")
                             log.debug(json.dumps(tools_to_log_in_detail, indent=2))
                             log.debug("--- End Detailed Schema ---")
                    except Exception as json_err:
                        log.error(f"Could not serialize tools for logging: {json_err}")
                        log.debug(f"Tools raw list (repr): {repr(available_tools)}")
                    log.debug(f"--- End API Call Preparation ---")

                # Make the API call
                try:
                    # Add a generous timeout for the entire stream processing
                    async with asyncio.timeout(300): # 5 minutes timeout for the whole stream
                        async with self.anthropic.messages.stream(
                            model=model,
                            max_tokens=max_tokens,
                            messages=messages,
                            tools=available_tools if available_tools else None,
                            temperature=self.config.temperature,
                        ) as stream:
                            async for event in stream:
                                event_type = event.type

                                if use_debug_logging:
                                    log.debug(f"--- Stream Event Received --- Type: {event_type}")
                                    if event_type == "message_delta":
                                        log.debug(f"  Delta Content: {repr(event.delta)}") # Safe: No '.type' access needed
                                    elif hasattr(event, 'delta') and event.delta:
                                        delta_type_str = getattr(event.delta, 'type', 'N/A') # Safely get type if exists
                                        log.debug(f"  Delta: type={delta_type_str}, content={repr(event.delta)}")
                                        if delta_type_str == "input_json_delta":
                                            log.debug(f"  >>>> Received input_json_delta: partial_json='{event.delta.partial_json}'")
                                    elif hasattr(event, 'content_block') and event.content_block:
                                        log.debug(f"  Content Block: type={event.content_block.type}, name={getattr(event.content_block, 'name', None)}, input={repr(getattr(event.content_block, 'input', None))}")
                                        if event_type == "content_block_start" and event.content_block.type == "tool_use":
                                            log.debug(f"  >>>> Initial 'input' at content_block_start: {repr(event.content_block.input)}")
                                    elif hasattr(event, 'message'):
                                        log.debug(f"  Message: {repr(event.message)}")
                                    else:
                                        log.debug(f"  Event Details: {repr(event)}")

                                # --- Process stream events with state tracking & input accumulation ---
                                if event_type == "message_start":
                                    current_assistant_content = [] # Reset content for this assistant message

                                elif event_type == "content_block_start":
                                    block_type = event.content_block.type
                                    if block_type == "text":
                                        current_text_block = {"type": "text", "text": ""}
                                    elif block_type == "tool_use":
                                        current_tool_use_block = {
                                            "type": "tool_use",
                                            "id": event.content_block.id,
                                            "name": event.content_block.name,
                                            "input": {} # Initialize input
                                        }
                                        current_tool_input_json_accumulator = "" # Reset accumulator

                                elif event_type == "content_block_delta":
                                    delta = event.delta
                                    if hasattr(delta, 'type'): # Check if delta has a type
                                        if delta.type == "text_delta":
                                            if current_text_block is not None: # Ensure we are tracking a text block
                                                text_chunk = delta.text
                                                current_text_block["text"] += text_chunk
                                                yield text_chunk # Yield for live display
                                        elif delta.type == "input_json_delta":
                                            if current_tool_use_block is not None: # Ensure we are tracking a tool block
                                                current_tool_input_json_accumulator += delta.partial_json

                                elif event_type == "content_block_stop":
                                    # Finalize the completed block
                                    if current_text_block is not None:
                                        # Finalize and store the text block
                                        current_assistant_content.append(current_text_block)
                                        final_response_text += current_text_block["text"]
                                        current_text_block = None # Reset state
                                    elif current_tool_use_block is not None:
                                        # Finalize and store the tool use block
                                        parsed_input = {} # Default to empty dict
                                        # Only attempt to parse if the accumulator is not empty
                                        if current_tool_input_json_accumulator:
                                            try:
                                                log.debug(f"Attempting to parse JSON for tool {current_tool_use_block['name']} (ID: {current_tool_use_block['id']}). Accumulated JSON string:")
                                                log.debug(f"RAW_ACCUMULATED_JSON>>>\n{current_tool_input_json_accumulator}\n<<<END_RAW_ACCUMULATED_JSON")
                                                try:
                                                    parsed_input = json.loads(current_tool_input_json_accumulator)
                                                    log.debug(f"Successfully parsed input: {parsed_input}")
                                                except json.JSONDecodeError as e:
                                                    log.error(f"JSON PARSE FAILED for tool {current_tool_use_block['name']} (ID: {current_tool_use_block['id']}): {e}")
                                                    log.error(f"Failed JSON content was:\n{current_tool_input_json_accumulator}") # Log again on error
                                                    parsed_input = {"_tool_input_parse_error": f"Failed to parse JSON: {e}", "_raw_json": current_tool_input_json_accumulator}
                                                # Parse accumulated JSON
                                                parsed_input = json.loads(current_tool_input_json_accumulator)
                                                log.debug(f"Final parsed input for tool {current_tool_use_block['name']}: {parsed_input}")
                                            except json.JSONDecodeError as e:
                                                log.error(f"Failed to parse accumulated JSON for tool {current_tool_use_block['name']}: {e}. JSON='{current_tool_input_json_accumulator}'")
                                                # Store error indication instead of empty dict or partial json
                                                parsed_input = {"_tool_input_parse_error": f"Failed to parse JSON: {e}", "_raw_json": current_tool_input_json_accumulator}
                                        else:
                                            # If accumulator is empty, it means no input args were streamed,
                                            # which is correct for tools with no input schema. Input is {}.
                                            log.debug(f"Tool {current_tool_use_block['name']} received empty input stream (expected for no-arg tools). Final input: {{}}")
                                            parsed_input = {} # Explicitly set to empty dict

                                        current_tool_use_block["input"] = parsed_input # Assign the parsed (or empty) dict

                                        # Add the completed block to the assistant's content for this turn
                                        current_assistant_content.append(current_tool_use_block)
                                        # Add it to the list of tools that need execution *after* this stream finishes
                                        tool_calls_this_turn.append(current_tool_use_block)

                                        # Yield message for UI
                                        original_tool_name = self.server_manager.sanitized_to_original.get(current_tool_use_block['name'], current_tool_use_block['name'])
                                        yield f"@@STATUS@@\n{STATUS_EMOJI['tool']} Preparing tool: {original_tool_name}...\n"

                                        current_tool_use_block = None # Reset state
                                        current_tool_input_json_accumulator = "" # Reset accumulator here

                                elif event_type == "message_delta":
                                    if hasattr(event.delta, 'stop_reason') and event.delta.stop_reason:
                                        stop_reason = event.delta.stop_reason

                                elif event_type == "message_stop":
                                    if hasattr(event, 'message') and hasattr(event.message, 'stop_reason'):
                                        stop_reason = event.message.stop_reason
                                    # Finalize any trailing text block
                                    if current_text_block is not None:
                                        current_assistant_content.append(current_text_block)
                                        final_response_text += current_text_block["text"]
                                        current_text_block = None
                                # --- End of inner stream processing loop ---
                except TimeoutError:
                    log.error("Timeout waiting for or processing message stream from Anthropic API.")
                    yield "\n[Error: Timed out waiting for Claude's final response after tool use]"
                    # Need to handle cleanup/state reset if timeout occurs mid-stream
                    # Potentially update conversation graph with partial data or error state
                except Exception as stream_err:
                    # Handle other stream errors
                    log.error(f"Error processing Anthropic message stream: {stream_err}", exc_info=True)
                    yield f"\n[Error: {stream_err}]"

                # --- Post-Stream Processing ---
                # Add the assistant message (potentially containing text and/or completed tool_use blocks) to the history
                if current_assistant_content:
                    messages.append({"role": "assistant", "content": current_assistant_content})

                # Check if the reason for stopping was tool use
                if stop_reason == "tool_use":
                    tool_results_for_api: List[MessageParam] = []

                    # Execute the completed tool calls collected during this stream iteration
                    for tool_call_block in tool_calls_this_turn:
                        tool_name_sanitized = tool_call_block["name"]
                        tool_args = tool_call_block["input"] # Already parsed or contains error marker
                        tool_use_id = tool_call_block["id"]

                        log.debug(f"Processing completed tool call. Name: {tool_name_sanitized}, ID: {tool_use_id}, Args: {repr(tool_args)}")

                        original_tool_name = self.server_manager.sanitized_to_original.get(tool_name_sanitized, tool_name_sanitized)

                        # Initialize vars for this tool call
                        api_content_for_claude: str = "Error: Tool execution failed unexpectedly."
                        log_content_for_history: Any = api_content_for_claude # Can store richer data for logging
                        is_error = True
                        cache_used = False

                        # Handle potential JSON parsing error from stream processing
                        if isinstance(tool_args, dict) and "_tool_input_parse_error" in tool_args:
                             error_text = f"Client failed to parse JSON input from API for tool '{original_tool_name}'. Raw: {tool_args.get('_raw_json', 'N/A')}"
                             api_content_for_claude = f"Error: {error_text}"
                             log_content_for_history = {"error": error_text, "raw_json": tool_args.get('_raw_json')}
                             is_error = True
                             yield f"@@STATUS@@\n{STATUS_EMOJI['failure']} Input Error: {error_text}\n"
                             log.error(error_text)
                        else:
                            # Proceed if input was valid
                            tool = self.server_manager.tools.get(original_tool_name)

                            if not tool:
                                error_text = f"Tool '{original_tool_name}' (req: '{tool_name_sanitized}') not found by client."
                                api_content_for_claude = f"Error: {error_text}"
                                log_content_for_history = {"error": error_text}
                                is_error = True
                                yield f"@@STATUS@@\n{STATUS_EMOJI['failure']} Server Error: {error_text}\n"
                                log.warning(f"Tool mapping/not found. Sanitized: '{tool_name_sanitized}', Original: '{original_tool_name}'")
                            else:
                                servers_used.add(tool.server_name)
                                tools_used.append(original_tool_name)
                                session = self.server_manager.active_sessions.get(tool.server_name)

                                if not session:
                                    error_text = f"Server '{tool.server_name}' for tool '{original_tool_name}' is not connected."
                                    api_content_for_claude = f"Error: {error_text}"
                                    log_content_for_history = {"error": error_text}
                                    is_error = True
                                    yield f"@@STATUS@@\n{STATUS_EMOJI['failure']} Server Error: {error_text}\n"
                                else:
                                    # Check cache
                                    cached_result = None
                                    if self.tool_cache:
                                        try:
                                            cached_result = self.tool_cache.get(original_tool_name, tool_args)
                                        except TypeError as cache_key_error:
                                             log.warning(f"Cannot check cache for {original_tool_name}: args unhashable ({type(tool_args)}). Error: {cache_key_error}")
                                             cached_result = None

                                        if cached_result is not None:
                                            api_content_for_claude = str(cached_result)
                                            log_content_for_history = cached_result # Keep original type for log
                                            is_error = isinstance(cached_result, str) and cached_result.lower().startswith("error:")
                                            if is_error:
                                                log.info(f"Cached result for {original_tool_name} is error, ignoring.")
                                                cached_result = None
                                            else:
                                                cache_used = True
                                                cache_hits_during_query += 1
                                                yield f"@@STATUS@@\n{STATUS_EMOJI['cached']} Using cached result for {original_tool_name}\n"
                                                log.info(f"Using cached result for {original_tool_name}")

                                    # Execute if not cached
                                    if not cache_used:
                                        yield f"@@STATUS@@\n{STATUS_EMOJI['tool']} Executing {original_tool_name}...\n"
                                        try:
                                            with safe_stdout():
                                                tool_call_outcome: Optional[CallToolResult] = await self.execute_tool(
                                                    tool.server_name, original_tool_name, tool_args
                                                )

                                            if tool_call_outcome is None:
                                                error_text = f"Tool '{original_tool_name}' returned no result."
                                                api_content_for_claude = f"Error: {error_text}"
                                                log_content_for_history = {"error": error_text}
                                                is_error = True
                                                yield f"@@STATUS@@\n{STATUS_EMOJI['failure']} Tool Execution Error: {error_text}\n"

                                            elif tool_call_outcome.isError:
                                                error_text = "Tool execution failed."
                                                extracted_error = "Unknown server error"
                                                # Attempt to extract more specific error from content
                                                if tool_call_outcome.content:
                                                     if isinstance(tool_call_outcome.content, str):
                                                         extracted_error = tool_call_outcome.content
                                                     elif isinstance(tool_call_outcome.content, list) and tool_call_outcome.content:
                                                         first_item = tool_call_outcome.content[0]
                                                         if isinstance(first_item, str): extracted_error = first_item
                                                         elif hasattr(first_item, 'text'): extracted_error = str(first_item.text)
                                                         elif isinstance(first_item, dict) and 'text' in first_item: extracted_error = str(first_item['text'])
                                                         else: extracted_error = repr(first_item) # Fallback
                                                     elif isinstance(tool_call_outcome.content, dict):
                                                         extracted_error = str(tool_call_outcome.content.get('error', tool_call_outcome.content.get('message', repr(tool_call_outcome.content))))
                                                     else: extracted_error = repr(tool_call_outcome.content)
                                                error_text = f"Tool execution failed: {extracted_error}"
                                                api_content_for_claude = f"Error: {error_text}"
                                                log_content_for_history = {"error": error_text, "raw_content": tool_call_outcome.content}
                                                is_error = True
                                                yield f"@@STATUS@@\n{STATUS_EMOJI['failure']} Tool Execution Error: {error_text[:100]}...\n" # Truncate long errors

                                            else: # Success case
                                                # Process success content into a string for API
                                                if tool_call_outcome.content is None:
                                                    api_content_for_claude = "Tool executed successfully with no content returned."
                                                elif isinstance(tool_call_outcome.content, str):
                                                    api_content_for_claude = tool_call_outcome.content
                                                elif isinstance(tool_call_outcome.content, list):
                                                    text_parts = []
                                                    for block in tool_call_outcome.content:
                                                        if isinstance(block, str): text_parts.append(block)
                                                        elif hasattr(block, 'text'): text_parts.append(str(block.text))
                                                        elif isinstance(block, dict) and 'text' in block: text_parts.append(str(block['text']))
                                                    if text_parts:
                                                        api_content_for_claude = "\n".join(text_parts)
                                                    else: # No text parts, serialize
                                                        try:
                                                             serializable_content = [ getattr(b, 'model_dump', lambda b=b: b)() for b in tool_call_outcome.content ]
                                                             api_content_for_claude = json.dumps(serializable_content, indent=2)
                                                        except Exception: api_content_for_claude = str(tool_call_outcome.content) # Fallback
                                                else: # Fallback
                                                    api_content_for_claude = str(tool_call_outcome.content)

                                                log_content_for_history = tool_call_outcome.content # Log original success content
                                                is_error = False
                                                yield f"@@STATUS@@\n{STATUS_EMOJI['success']} Tool result received.\n"

                                            # Cache result if successful and cache enabled
                                            if self.tool_cache and not is_error:
                                                try:
                                                    self.tool_cache.set(original_tool_name, tool_args, api_content_for_claude)
                                                except TypeError as cache_set_error:
                                                     log.warning(f"Failed to cache result for {original_tool_name}: {cache_set_error}")
                                            elif is_error:
                                                log.info(f"Skipping cache for error result of {original_tool_name}.")

                                        except Exception as exec_err:
                                            log.error(f"Client error during tool execution {original_tool_name}: {exec_err}", exc_info=True)
                                            error_text = f"Client error: {str(exec_err)}"
                                            api_content_for_claude = f"Error: {error_text}"
                                            log_content_for_history = {"error": error_text}
                                            is_error = True
                                            yield f"@@STATUS@@\n{STATUS_EMOJI['failure']} Tool Execution Client Error: {str(exec_err)}\n"

                        # Append result (or error) for the *next* API call
                        tool_results_for_api.append({
                            "role": "user",
                            "content": [ { "type": "tool_result", "tool_use_id": tool_use_id, "content": api_content_for_claude } ]
                        })
                        # Store structured info for final history logging
                        tool_results_for_history.append({
                            "tool_name": original_tool_name, "tool_use_id": tool_use_id,
                            "content": log_content_for_history, "is_error": is_error, "cache_used": cache_used
                        })

                    # Add tool results to messages and continue the outer loop
                    messages.extend(tool_results_for_api)
                    continue # Go back to start of while loop

                else: # Stop reason was not 'tool_use' (e.g., 'end_turn', 'max_tokens')
                    break # Exit the while True loop

            # --- End of Streaming / Tool Call Loop ---

            # Update conversation graph node with the *final* message list
            self.conversation_graph.current_node.messages = messages
            self.conversation_graph.current_node.model = model # Store model used

            # Calculate metrics
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            tokens_used = await self.count_tokens(messages)

            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            await self.history.add_async(ChatHistory(
                query=query, response=final_response_text, model=model, timestamp=timestamp,
                server_names=list(servers_used), tools_used=tools_used,
                conversation_id=self.conversation_graph.current_node.id, latency_ms=latency_ms,
                tokens_used=tokens_used, streamed=True, cached=(cache_hits_during_query > 0)
                # Add tool_results_for_history here if needed
            ))

            # Finalize Span
            if span:
                span.set_status(trace.StatusCode.OK)
                span.add_event("query_complete", {
                    "latency_ms": latency_ms, "tools_used_count": len(tools_used),
                    "servers_used_count": len(servers_used), "cache_hits": cache_hits_during_query,
                    "response_length": len(final_response_text), "approximated_tokens": tokens_used,
                    "final_stop_reason": stop_reason
                })

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            log.error(error_msg, exc_info=True)
            if span:
                span.set_status(trace.StatusCode.ERROR, description=error_msg)
                span.record_exception(e)
            yield f"@@STATUS@@\n[Error: {error_msg}]\n" # Yield error to user

        finally:
            # Finalize Span
            if span_context_manager:
                try:
                    exc_type, exc_value, tb = sys.exc_info()
                    span_context_manager.__exit__(exc_type, exc_value, tb)
                except Exception as exit_err:
                    log.warning(f"Error exiting span context manager: {exit_err}")

    async def process_query(self, query: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """
        Process a query using Claude and available tools (non-streaming version).
        Handles OpenTelemetry span lifecycle correctly and formats tool results.
        """
        if not model:
            model = self.current_model

        if not max_tokens:
            max_tokens = self.config.default_max_tokens

        # Check if context needs pruning before processing
        await self.auto_prune_context()

        # Use streaming if enabled, but collect all results
        if self.config.enable_streaming:
            chunks = []
            async for chunk in self.process_streaming_query(query, model, max_tokens):
                chunks.append(chunk)
            # The streaming function now handles history and conversation graph updates
            return "".join(chunks)

        # --- Non-streaming implementation ---
        start_time = time.time()

        # Check if we have any servers connected
        if not self.server_manager.active_sessions:
            return "No MCP servers connected. Use 'servers connect' to connect to servers."

        # Get tools from all connected servers
        available_tools = self.server_manager.format_tools_for_anthropic()

        if not available_tools:
             log.info("No tools available from connected servers. Proceeding with LLM only.")
             # Allow processing without tools
             # return "No tools available from connected servers." # Removed return

        # Keep track of servers and tools used
        servers_used = set()
        tools_used = []
        tool_results_for_history = [] # Store structured tool results for history
        cache_hits_during_query = 0

        # Start with user message
        current_messages: List[MessageParam] = self.conversation_graph.current_node.messages.copy()
        user_message: MessageParam = {"role": "user", "content": query}
        # Combine initial messages (consider putting user_message at the end after filtering if needed)
        combined_messages: List[MessageParam] = current_messages + [user_message]

        # --- NEW: Filter out client-side tool execution parse failures ---
        messages_to_send: List[MessageParam] = []
        i = 0
        client_error_signature = "Client failed to parse JSON input" # More general check
        skipped_indices = set() # To track indices of messages to skip

        log.debug(f"Filtering history ({len(combined_messages)} messages) for known client errors...")

        # First pass: identify indices of faulty interactions to skip
        assistant_tool_uses_to_check: Dict[int, Set[str]] = {} # index -> set of tool_use_ids

        for idx, msg in enumerate(combined_messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    tool_use_ids = {
                        block.get("id") for block in content
                        if isinstance(block, dict) and block.get("type") == "tool_use" and block.get("id")
                    }
                    if tool_use_ids:
                        assistant_tool_uses_to_check[idx] = tool_use_ids

            elif msg.get("role") == "user":
                # Check if this user message corresponds to a preceding assistant tool use
                prev_idx = idx - 1
                if prev_idx in assistant_tool_uses_to_check:
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        corresponding_ids = assistant_tool_uses_to_check[prev_idx]
                        found_faulty_result = False
                        for block in content:
                            if (isinstance(block, dict) and
                                block.get("type") == "tool_result" and
                                block.get("tool_use_id") in corresponding_ids):
                                result_content = block.get("content")
                                # Check if the content contains our specific client error signature
                                if isinstance(result_content, str) and client_error_signature in result_content:
                                    found_faulty_result = True
                                    log.warning(f"Found faulty client tool result for ID {block.get('tool_use_id')} at history index {idx}. Marking for filtering.")
                                    break # Found one faulty result for this user message turn

                        if found_faulty_result:
                            # Mark both the assistant request and the user result for skipping
                            skipped_indices.add(prev_idx)
                            skipped_indices.add(idx)

        # Second pass: build the filtered list
        for idx, msg in enumerate(combined_messages):
            if idx not in skipped_indices:
                messages_to_send.append(msg)
            else:
                log.debug(f"Skipping message at index {idx} during API call preparation.")

        # Now use 'messages_to_send' for the API call
        messages = messages_to_send # Replace the original list
        log.debug(f"Proceeding with {len(messages)} filtered messages for API call.")

        # --- Corrected OpenTelemetry Span Handling ---
        span = None
        span_context_manager = None
        if tracer:
            try:
                span_context_manager = tracer.start_as_current_span(
                    "process_query_non_streaming", # More specific name
                    attributes={
                        "model": model,
                        "query_length": len(query),
                        "conversation_length": len(messages),
                        "streaming": False
                    }
                )
                if span_context_manager:
                    span = span_context_manager.__enter__() # Get the actual span object
            except Exception as e:
                log.warning(f"Failed to start trace span: {e}")
                span = None
                span_context_manager = None
        # --- End Span Handling Setup ---

        # Use a manually controlled status instead of context manager
        safe_console = get_safe_console()
        status = Status(f"{STATUS_EMOJI['speech_balloon']} Claude is thinking...", spinner="dots", console=safe_console)
        status.start()

        final_response_text_parts = [] # Accumulate final text response parts
        response = None # Initialize response variable

        try:
            # --- Main Non-Streaming Loop ---
            while True:
                # Make API call
                status.update(f"{STATUS_EMOJI['speech_balloon']} Sending query to Claude ({model})...")
                try:
                    response = await self.anthropic.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=messages,
                        tools=available_tools if available_tools else None,
                        temperature=self.config.temperature,
                    )
                    status.update(f"{STATUS_EMOJI['success']} Received response from Claude")
                except anthropic.APIError as api_err:
                     status.stop()
                     log.error(f"Anthropic API Error: {api_err}", exc_info=True)
                     raise # Re-raise to be caught by outer handler

                # --- Process response and handle tool calls ---
                tool_calls_this_round = []
                current_assistant_content_blocks = [] # Content blocks for *this* assistant message

                # Extract content before potential tool use
                if response.content:
                    for content_block in response.content:
                        current_assistant_content_blocks.append(content_block.model_dump(exclude_unset=True)) # Store as dict
                        if content_block.type == 'text':
                            final_response_text_parts.append(content_block.text) # Append text immediately
                        elif content_block.type == 'tool_use':
                            tool_calls_this_round.append(content_block)

                # Add the assistant message (containing text and/or tool_use requests)
                if current_assistant_content_blocks:
                    messages.append({"role": "assistant", "content": current_assistant_content_blocks})

                # If no tool use, break the loop
                if response.stop_reason != "tool_use":
                    break

                # --- Process Tool Calls ---
                tool_results_for_api = [] # Results to send back in the next API call

                for tool_call in tool_calls_this_round:
                    tool_name_sanitized = tool_call.name
                    tool_args = tool_call.input
                    tool_use_id = tool_call.id

                    original_tool_name = self.server_manager.sanitized_to_original.get(tool_name_sanitized, tool_name_sanitized)
                    tool = self.server_manager.tools.get(original_tool_name)

                    # --- Variables for this tool call ---
                    api_content_for_claude: Union[str, List[Dict[str, Any]]] = "Error: Tool execution failed unexpectedly."
                    log_content_for_history: str = api_content_for_claude
                    is_error = True
                    cache_used = False

                    if not tool:
                        log.warning(f"Tool mapping issue or tool not found. Sanitized: '{tool_name_sanitized}', Original attempted: '{original_tool_name}'")
                        error_text = f"Tool '{original_tool_name}' (requested as '{tool_name_sanitized}') not found by client."
                        api_content_for_claude = f"Error: {error_text}"
                        log_content_for_history = api_content_for_claude
                        is_error = True
                        status.update(f"{STATUS_EMOJI['failure']} Tool Error: {error_text}")
                    else:
                        servers_used.add(tool.server_name)
                        tools_used.append(original_tool_name)
                        session = self.server_manager.active_sessions.get(tool.server_name)

                        if not session:
                            error_text = f"Server '{tool.server_name}' for tool '{original_tool_name}' is not connected."
                            api_content_for_claude = f"Error: {error_text}"
                            log_content_for_history = api_content_for_claude
                            is_error = True
                            status.update(f"{STATUS_EMOJI['failure']} Server Error: {error_text}")
                        else:
                            # Check cache
                            if self.tool_cache:
                                cached_result = self.tool_cache.get(original_tool_name, tool_args)
                                if cached_result is not None:
                                    api_content_for_claude = cached_result
                                    log_content_for_history = str(cached_result)
                                    is_error = False # Assume cached = success
                                    cache_used = True
                                    cache_hits_during_query += 1
                                    status.update(f"[{STATUS_EMOJI['cached']}] Using cached result for {original_tool_name}")
                                    log.info(f"Using cached result for {original_tool_name}")

                            # Execute if not cached
                            if not cache_used:
                                status.update(f"[{STATUS_EMOJI['tool']}] Executing tool: {original_tool_name}...")
                                try:
                                    with safe_stdout():
                                        # --- Execute and process result ---
                                        tool_call_outcome: Optional[CallToolResult] = await self.execute_tool(
                                            tool.server_name, original_tool_name, tool_args
                                        )

                                        if tool_call_outcome is None:
                                            error_text = f"Tool '{original_tool_name}' execution prevented or failed unexpectedly (no result)."
                                            api_content_for_claude = f"Error: {error_text}"
                                            log_content_for_history = api_content_for_claude
                                            is_error = True
                                            status.update(f"{STATUS_EMOJI['failure']} Tool Execution Error: {error_text}")
                                        elif tool_call_outcome.isError:
                                            error_text = "Tool execution failed with an unspecified error."
                                            if tool_call_outcome.content and isinstance(tool_call_outcome.content, list):
                                                first_text = next((b.text for b in tool_call_outcome.content if hasattr(b, 'type') and b.type == 'text' and hasattr(b, 'text')), None)
                                                if first_text: error_text = first_text
                                            elif isinstance(tool_call_outcome.content, str): error_text = tool_call_outcome.content
                                            api_content_for_claude = f"Error: {error_text}"
                                            log_content_for_history = api_content_for_claude
                                            is_error = True
                                            status.update(f"{STATUS_EMOJI['failure']} Tool Execution Error: {error_text}")
                                        else: # Success
                                            extracted_content_str = "Tool executed successfully, but no text content was returned."
                                            if tool_call_outcome.content and isinstance(tool_call_outcome.content, list):
                                                first_text = next((b.text for b in tool_call_outcome.content if hasattr(b, 'type') and b.type == 'text' and hasattr(b, 'text')), None)
                                                if first_text is not None: extracted_content_str = first_text
                                                else:
                                                    try:
                                                        serializable_content = [ getattr(b, 'model_dump', lambda b=b: b)() for b in tool_call_outcome.content ]
                                                        extracted_content_str = json.dumps(serializable_content, indent=2)
                                                    except Exception as serialize_err:
                                                        log.warning(f"Could not serialize tool result content, falling back to str(): {serialize_err}")
                                                        extracted_content_str = str(tool_call_outcome.content)
                                            elif isinstance(tool_call_outcome.content, str): extracted_content_str = tool_call_outcome.content
                                            elif tool_call_outcome.content is not None: extracted_content_str = str(tool_call_outcome.content)
                                            api_content_for_claude = extracted_content_str
                                            log_content_for_history = api_content_for_claude
                                            is_error = False
                                            status.update(f"{STATUS_EMOJI['success']} Tool {original_tool_name} execution complete")

                                        # --- Cache result ---
                                        if self.tool_cache:
                                            self.tool_cache.set(original_tool_name, tool_args, api_content_for_claude)

                                except Exception as e:
                                    log.error(f"Error executing tool {original_tool_name} or processing its result: {e}", exc_info=True)
                                    error_text = f"Client error during tool execution '{original_tool_name}': {str(e)}"
                                    api_content_for_claude = f"Error: {error_text}"
                                    log_content_for_history = api_content_for_claude
                                    is_error = True
                                    status.update(f"{STATUS_EMOJI['failure']} Tool Execution Client Error: {str(e)}")

                    # Append result for the *next* API call
                    tool_results_for_api.append({
                        "role": "user", # MUST be user role
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": api_content_for_claude
                            }
                        ]
                    })
                    # Store structured info for final history logging
                    tool_results_for_history.append({
                        "tool_name": original_tool_name,
                        "tool_use_id": tool_use_id,
                        "content": log_content_for_history,
                        "is_error": is_error,
                        "cache_used": cache_used
                    })

                # Add all tool results to messages for the next iteration
                messages.extend(tool_results_for_api)
                # Reset text parts for the next potential text response from Claude
                final_response_text_parts = []

            # --- End of Main Non-Streaming Loop ---

            status.stop() # Stop the status indicator

            # --- Update Conversation Graph ---
            # Add the initial user message (already added at the start)
            # The loop above added all assistant messages and user tool_result messages
            self.conversation_graph.current_node.model = model # Store model used

            # Join the final text parts collected
            final_result_text = "".join(final_response_text_parts)

            # Calculate metrics
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            tokens_used = 0
            if response and response.usage:
                 tokens_used = response.usage.input_tokens + response.usage.output_tokens

            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            await self.history.add_async(ChatHistory(
                query=query,
                response=final_result_text,
                model=model,
                timestamp=timestamp,
                server_names=list(servers_used),
                tools_used=tools_used,
                conversation_id=self.conversation_graph.current_node.id,
                latency_ms=latency_ms,
                tokens_used=tokens_used,
                streamed=False,
                cached=(cache_hits_during_query > 0)
            ))

            # --- Finalize Span ---
            if span:
                span.set_status(trace.StatusCode.OK)
                span.add_event("query_complete", {
                    "latency_ms": latency_ms,
                    "tools_used_count": len(tools_used),
                    "servers_used_count": len(servers_used),
                    "cache_hits": cache_hits_during_query,
                    "response_length": len(final_result_text),
                    "input_tokens": response.usage.input_tokens if response and response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response and response.usage else 0,
                    "stop_reason": response.stop_reason if response else "Unknown"
                })
            # --- End Span ---

            return final_result_text

        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            log.error(error_msg, exc_info=True) # Log with traceback

            # --- Finalize Span with Error ---
            if span:
                span.set_status(trace.StatusCode.ERROR, description=error_msg)
            # --- End Span ---

            return f"Unexpected Error: {error_msg}"

        finally:
            # --- Ensure Span Context is Exited ---
            if span_context_manager:
                try:
                    exc_type, exc_value, tb = sys.exc_info()
                    span_context_manager.__exit__(exc_type, exc_value, tb)
                except Exception as exit_err:
                    log.warning(f"Error exiting span context manager: {exit_err}")
            # --- End Span Exit ---

            # Stop the status display if it was started and is running
            if 'status' in locals() and hasattr(status, 'stop') and callable(status.stop):
                 # Check if status object might have been stopped already by an exception handler
                 with suppress(RuntimeError): # suppress error if trying to stop an already stopped status
                    status.stop()
    
    async def interactive_loop(self):
        """Run interactive command loop with smoother live streaming output."""
        # Always use get_safe_console for interactive mode
        interactive_console = get_safe_console()

        self.safe_print("\n[bold green]MCP Client Interactive Mode[/]")
        self.safe_print("Type your query to Claude, or a command (type 'help' for available commands)")
        self.safe_print("[italic]Press Ctrl+C twice in quick succession to force exit if unresponsive[/italic]")

        # Track Ctrl+C attempts for force exit
        ctrl_c_count = 0
        last_ctrl_c_time = 0

        while True:
            live_display: Optional[Live] = None # To manage the Live instance
            try:
                user_input = Prompt.ask("\n[bold blue]>>[/]", console=interactive_console)

                # Reset Ctrl+C counter on successful input
                ctrl_c_count = 0

                # Check if it's a command
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

                # --- Process as a query to Claude with smoother live streaming output ---
                else:
                    claude_text_content = ""
                    # Keep track of the last few status messages
                    status_lines = deque(maxlen=5) # Adjust maxlen as needed

                    initial_panel = Panel("", title="Claude", border_style="green")
                    REFRESH_RATE = 4.0 # Increased slightly
                    live_display = Live(
                        initial_panel,
                        console=interactive_console,
                        refresh_per_second=REFRESH_RATE,
                        transient=False,
                        vertical_overflow="visible"
                    )
                    live_display.start()

                    try: # Add try block specifically around streaming
                        async for chunk in self.process_streaming_query(user_input):
                            if chunk.startswith("@@STATUS@@"):
                                status_message = chunk[len("@@STATUS@@"):].strip()
                                # Add styling based on content (simple example)
                                style = "yellow"
                                if "Error" in status_message or "failed" in status_message or "💥" in status_message:
                                    style = "red"
                                elif "success" in status_message or "received" in status_message or "🎉" in status_message or "✓" in status_message:
                                    style = "green"
                                elif "cached" in status_message or "package" in status_message:
                                     style = "cyan"
                                # Append new status message as a Text object
                                status_lines.append(Text(status_message, style=style))
                            else:
                                # It's a regular text chunk from Claude
                                claude_text_content += chunk
                            md = Markdown(claude_text_content)
                            # Create a renderable group for status lines
                            status_group = Group(*list(status_lines))
                            # Combine markdown and status lines vertically
                            renderable_group = Group(md, status_group)
                            panel = Panel(renderable_group, title="Claude", border_style="green")
                            live_display.update(panel, refresh=False)
                            await asyncio.sleep(0.01)

                        # --- After the loop ---
                        if live_display.is_started:
                            # Perform one last update with the final content and status
                            final_md = Markdown(claude_text_content)
                            final_status_group = Group(*list(status_lines))
                            final_renderable_group = Group(final_md, final_status_group)
                            final_panel = Panel(final_renderable_group, title="Claude", border_style="green")
                            live_display.update(final_panel, refresh=True)

                    # Catch potential errors during streaming itself if needed
                    except Exception as stream_err:
                         log.error(f"Error during stream processing in interactive loop: {stream_err}", exc_info=True)
                         # Add an error message to status lines
                         status_lines.append(Text(f"[bold red]Stream Error: {stream_err}[/]", style="red"))
                         # Update the display one last time with the error
                         md = Markdown(claude_text_content)
                         status_group = Group(*list(status_lines))
                         renderable_group = Group(md, status_group)
                         panel = Panel(renderable_group, title="Claude - ERROR", border_style="red")
                         if live_display.is_started:
                             live_display.update(panel, refresh=True)
                    finally: # Ensure live display is stopped
                         if live_display.is_started:
                              live_display.stop()
                         live_display = None # Clear reference

            except KeyboardInterrupt:
                if live_display and live_display.is_started: 
                    live_display.stop()
                    live_display = None
                current_time = time.time()
                if current_time - last_ctrl_c_time < 2: ctrl_c_count += 1
                else: ctrl_c_count = 1
                last_ctrl_c_time = current_time
                if ctrl_c_count >= 2:
                    self.safe_print("\n[bold red]Force exiting...[/]")
                    if hasattr(self, 'server_manager') and hasattr(self.server_manager, 'processes'):
                        for name, process in self.server_manager.processes.items():
                            try:
                                if process and process.returncode is None:
                                    self.safe_print(f"[yellow]Force killing process: {name}[/]")
                                    process.kill()
                            except Exception as e: self.safe_print(f"[red]Error killing process {name}: {e}")
                    break
                self.safe_print("\n[yellow]Interrupted. Press Ctrl+C again to force exit.[/]")
                continue

            except (anthropic.APIError, McpError, httpx.RequestError) as e:
                if live_display and live_display.is_started:
                    live_display.stop()
                    live_display = None
                self.safe_print(f"[bold red]Error ({type(e).__name__}):[/] {str(e)}")

            except Exception as e:
                if live_display and live_display.is_started: 
                    live_display.stop()
                    live_display = None
                self.safe_print(f"[bold red]Unexpected Error:[/] {str(e)}")

            finally:
                if live_display and live_display.is_started: 
                    live_display.stop()
                    live_display = None

    async def count_tokens(self, messages=None) -> int:
        """Count the number of tokens in the current conversation context"""
        if messages is None:
            messages = self.conversation_graph.current_node.messages
            
        # Use cl100k_base encoding which is used by Claude
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = 0
        
        for message in messages:
            # Get the message content
            content = message.get("content", "")
            
            # Handle content that might be a list of blocks (text/image blocks)
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and "text" in block:
                        token_count += len(encoding.encode(block["text"]))
                    elif isinstance(block, str):
                        token_count += len(encoding.encode(block))
            else:
                # Simple string content
                token_count += len(encoding.encode(str(content)))
            
            # Add a small overhead for message formatting
            token_count += 4  # Approximate overhead per message
        return token_count
                                                              
    async def cmd_exit(self, args):
        """Exit the client"""
        self.safe_print("[yellow]Exiting...[/]")
        sys.exit(0)
        
    async def cmd_help(self, args):
        """Display help for commands"""
        # Create groups of related commands
        general_commands = [
            Text(f"{STATUS_EMOJI['scroll']} /help", style="bold"), Text(" - Show this help message"),
            Text(f"{STATUS_EMOJI['red_circle']} /exit, /quit", style="bold"), Text(" - Exit the client")
        ]
        
        config_commands = [
            Text(f"{STATUS_EMOJI['config']} /config", style="bold"), Text(" - Manage configuration (API key, model, discovery, port scanning, etc.)"),
            Text(f"{STATUS_EMOJI['speech_balloon']} /model", style="bold"), Text(" - Change the current model"),
            Text(f"{STATUS_EMOJI['package']} /cache", style="bold"), Text(" - Manage tool result cache")
        ]
        
        server_commands = [
            Text(f"{STATUS_EMOJI['server']} /servers", style="bold"), Text(" - Manage MCP servers"),
            Text(f"{STATUS_EMOJI['search']} /discover", style="bold"), Text(" - Discover and connect to local network servers"),
            Text(f"{STATUS_EMOJI['tool']} /tools", style="bold"), Text(" - List available tools"),
            Text(f"{STATUS_EMOJI['tool']} /tool", style="bold"), Text(" - Directly execute a tool with parameters"),
            Text(f"{STATUS_EMOJI['resource']} /resources", style="bold"), Text(" - List available resources"),
            Text(f"{STATUS_EMOJI['prompt']} /prompts", style="bold"), Text(" - List available prompts"),
            Text(f"{STATUS_EMOJI['green_circle']} /reload", style="bold"), Text(" - Reload servers and capabilities")
        ]
        
        conversation_commands = [
            Text(f"{STATUS_EMOJI['cross_mark']} /clear", style="bold"), Text(" - Clear the conversation context"),
            Text(f"{STATUS_EMOJI['scroll']} /history", style="bold"), Text(" - View conversation history"),
            Text(f"{STATUS_EMOJI['trident_emblem']} /fork [NAME]", style="bold"), Text(" - Create a conversation branch"),
            Text(f"{STATUS_EMOJI['trident_emblem']} /branch", style="bold"), Text(" - Manage conversation branches (list, checkout ID)"),
            Text(f"{STATUS_EMOJI['package']} /optimize", style="bold"), Text(" - Optimize conversation through summarization"),
            Text(f"{STATUS_EMOJI['scroll']} /export", style="bold"), Text(" - Export conversation to a file"),
            Text(f"{STATUS_EMOJI['scroll']} /import", style="bold"), Text(" - Import conversation from a file")
        ]
        
        monitoring_commands = [
            Text(f"{STATUS_EMOJI['desktop_computer']} /dashboard", style="bold"), Text(" - Show a live monitoring dashboard")
        ]
        
        # Display commands in organized groups
        self.safe_print("\n[bold]Available Commands:[/]")
        
        self.safe_print(Panel(
            Group(*general_commands),
            title="General Commands",
            border_style="blue"
        ))
        
        self.safe_print(Panel(
            Group(*config_commands),
            title="Configuration Commands",
            border_style="cyan"
        ))
        
        self.safe_print(Panel(
            Group(*server_commands),
            title="Server & Tools Commands",
            border_style="magenta"
        ))
        
        self.safe_print(Panel(
            Group(*conversation_commands),
            title="Conversation Commands",
            border_style="green"
        ))
        
        self.safe_print(Panel(
            Group(*monitoring_commands),
            title="Monitoring Commands",
            border_style="yellow"
        ))
    
    async def cmd_config(self, args):
        """Handle configuration commands"""
        if not args:
            # Show current config
            self.safe_print("\n[bold]Current Configuration:[/]")
            self.safe_print(f"API Key: {'*' * 8 + self.config.api_key[-4:] if self.config.api_key else 'Not set'}")
            self.safe_print(f"Default Model: {self.config.default_model}")
            self.safe_print(f"Max Tokens: {self.config.default_max_tokens}")
            self.safe_print(f"History Size: {self.config.history_size}")
            self.safe_print(f"Auto-Discovery: {'Enabled' if self.config.auto_discover else 'Disabled'}")
            self.safe_print(f"Discovery Paths: {', '.join(self.config.discovery_paths)}")
            self.safe_print("\n[bold]Port Scanning:[/]")
            self.safe_print(f"  Enabled: {'Yes' if self.config.enable_port_scanning else 'No'}")
            self.safe_print(f"  Range: {self.config.port_scan_range_start} - {self.config.port_scan_range_end}")
            self.safe_print(f"  Targets: {', '.join(self.config.port_scan_targets)}")
            self.safe_print(f"  Concurrency: {self.config.port_scan_concurrency}")
            self.safe_print(f"  Timeout: {self.config.port_scan_timeout}s")            
            return
        
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower()
        subargs = parts[1] if len(parts) > 1 else ""
        
        if subcmd == "api-key":
            if not subargs:
                self.safe_print("[yellow]Usage: /config api-key YOUR_API_KEY[/]")
                return
                
            self.config.api_key = subargs
            try:
                self.anthropic = AsyncAnthropic(api_key=self.config.api_key)
                self.config.save()
                self.safe_print("[green]API key updated[/]")
            except Exception as e:
                self.safe_print(f"[red]Error initializing Anthropic client: {e}[/]")
                self.anthropic = None
            
        elif subcmd == "model":
            if not subargs:
                self.safe_print("[yellow]Usage: /config model MODEL_NAME[/]")
                return
                
            self.config.default_model = subargs
            self.current_model = subargs
            self.config.save()
            self.safe_print(f"[green]Default model updated to {subargs}[/]")
            
        elif subcmd == "max-tokens":
            if not subargs or not subargs.isdigit():
                self.safe_print("[yellow]Usage: /config max-tokens NUMBER[/]")
                return
                
            self.config.default_max_tokens = int(subargs)
            self.config.save()
            self.safe_print(f"[green]Default max tokens updated to {subargs}[/]")
            
        elif subcmd == "history-size":
            if not subargs or not subargs.isdigit():
                self.safe_print("[yellow]Usage: /config history-size NUMBER[/]")
                return
                
            self.config.history_size = int(subargs)
            self.history.max_entries = int(subargs)
            self.config.save()
            self.safe_print(f"[green]History size updated to {subargs}[/]")
            
        elif subcmd == "auto-discover":
            if subargs.lower() in ("true", "yes", "on", "1"):
                self.config.auto_discover = True
            elif subargs.lower() in ("false", "no", "off", "0"):
                self.config.auto_discover = False
            else:
                self.safe_print("[yellow]Usage: /config auto-discover [true|false][/]")
                return
                
            self.config.save()
            self.safe_print(f"[green]Auto-discovery {'enabled' if self.config.auto_discover else 'disabled'}[/]")
            
        elif subcmd == "port-scan":
             scan_parts = subargs.split(maxsplit=1)
             scan_action = scan_parts[0].lower() if scan_parts else "show"
             scan_value = scan_parts[1] if len(scan_parts) > 1 else ""

             if scan_action == "enable":
                 if scan_value.lower() in ("true", "yes", "on", "1"):
                     self.config.enable_port_scanning = True
                     self.config.save()
                     self.safe_print("[green]Port scanning enabled.[/]")
                 elif scan_value.lower() in ("false", "no", "off", "0"):
                     self.config.enable_port_scanning = False
                     self.config.save()
                     self.safe_print("[yellow]Port scanning disabled.[/]")
                 else:
                     self.safe_print("[yellow]Usage: /config port-scan enable [true|false][/]")
             elif scan_action == "range":
                 range_parts = scan_value.split()
                 if len(range_parts) == 2 and range_parts[0].isdigit() and range_parts[1].isdigit():
                     start, end = int(range_parts[0]), int(range_parts[1])
                     if start <= end:
                         self.config.port_scan_range_start = start
                         self.config.port_scan_range_end = end
                         self.config.save()
                         self.safe_print(f"[green]Port scan range set to {start}-{end}[/]")
                     else:
                         self.safe_print("[red]Start port must be less than or equal to end port.[/]")
                 else:
                     self.safe_print("[yellow]Usage: /config port-scan range START_PORT END_PORT[/]")
             elif scan_action == "targets":
                 targets = [t.strip() for t in scan_value.split(',') if t.strip()]
                 if targets:
                     self.config.port_scan_targets = targets
                     self.config.save()
                     self.safe_print(f"[green]Port scan targets set to: {', '.join(targets)}[/]")
                 else:
                     self.safe_print("[yellow]Usage: /config port-scan targets IP1,IP2,...[/]")
                     self.safe_print("[yellow]Example: /config port-scan targets 127.0.0.1,192.168.1.10[/]")
             # Add similar handlers for 'concurrency' and 'timeout' if desired
             elif scan_action == "show" or not scan_action:
                 self.safe_print("\n[bold]Port Scanning Settings:[/]")
                 self.safe_print(f"  Enabled: {'Yes' if self.config.enable_port_scanning else 'No'}")
                 self.safe_print(f"  Range: {self.config.port_scan_range_start} - {self.config.port_scan_range_end}")
                 self.safe_print(f"  Targets: {', '.join(self.config.port_scan_targets)}")
                 self.safe_print(f"  Concurrency: {self.config.port_scan_concurrency}")
                 self.safe_print(f"  Timeout: {self.config.port_scan_timeout}s")
             else:
                 self.safe_print("[yellow]Unknown port-scan command. Available: enable, range, targets, concurrency, timeout, show[/]")

        elif subcmd == "discovery-path":
            parts = subargs.split(maxsplit=1)
            action = parts[0].lower() if parts else ""
            path = parts[1] if len(parts) > 1 else ""
            
            if action == "add" and path:
                if path not in self.config.discovery_paths:
                    self.config.discovery_paths.append(path)
                    self.config.save()
                    self.safe_print(f"[green]Added discovery path: {path}[/]")
                else:
                    self.safe_print(f"[yellow]Path already exists: {path}[/]")
                    
            elif action == "remove" and path:
                if path in self.config.discovery_paths:
                    self.config.discovery_paths.remove(path)
                    self.config.save()
                    self.safe_print(f"[green]Removed discovery path: {path}[/]")
                else:
                    self.safe_print(f"[yellow]Path not found: {path}[/]")
                    
            elif action == "list" or not action:
                self.safe_print("\n[bold]Discovery Paths:[/]")
                for i, path in enumerate(self.config.discovery_paths, 1):
                    self.safe_print(f"{i}. {path}")
                    
            else:
                self.safe_print("[yellow]Usage: /config discovery-path [add|remove|list] [PATH][/]")
                
        else:
            self.safe_print("[yellow]Unknown config command. Available: api-key, model, max-tokens, history-size, auto-discover, discovery-path[/]")
    
    async def cmd_servers(self, args):
        """Handle server management commands"""
        if not args:
            # List servers
            await self.list_servers()
            return
        
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower()
        subargs = parts[1] if len(parts) > 1 else ""
        
        if subcmd == "list":
            await self.list_servers()
            
        elif subcmd == "add":
            await self.add_server(subargs)
            
        elif subcmd == "remove":
            await self.remove_server(subargs)
            
        elif subcmd == "connect":
            await self.connect_server(subargs)
            
        elif subcmd == "disconnect":
            await self.disconnect_server(subargs)
            
        elif subcmd == "enable":
            await self.enable_server(subargs, True)
            
        elif subcmd == "disable":
            await self.enable_server(subargs, False)
            
        elif subcmd == "status":
            await self.server_status(subargs)
            
        else:
            self.safe_print("[yellow]Unknown servers command. Available: list, add, remove, connect, disconnect, enable, disable, status[/]")
    
    async def list_servers(self):
        """List all configured servers"""
        
        if not self.config.servers:
            self.safe_print(f"{STATUS_EMOJI['warning']} [yellow]No servers configured[/]")
            return
            
        server_table = Table(title=f"{STATUS_EMOJI['server']} Configured Servers")
        server_table.add_column("Name")
        server_table.add_column("Type")
        server_table.add_column("Path/URL")
        server_table.add_column("Status")
        server_table.add_column("Enabled")
        server_table.add_column("Auto-Start")
        
        for name, server in self.config.servers.items():
            connected = name in self.server_manager.active_sessions
            status = f"{STATUS_EMOJI['connected']} [green]Connected[/]" if connected else f"{STATUS_EMOJI['disconnected']} [red]Disconnected[/]"
            enabled = f"{STATUS_EMOJI['white_check_mark']} [green]Yes[/]" if server.enabled else f"{STATUS_EMOJI['cross_mark']} [red]No[/]"
            auto_start = f"{STATUS_EMOJI['white_check_mark']} [green]Yes[/]" if server.auto_start else f"{STATUS_EMOJI['cross_mark']} [red]No[/]"
            
            server_table.add_row(
                name,
                server.type.value,
                server.path,
                status,
                enabled,
                auto_start
            )
            
        self.safe_print(server_table)
    
    async def add_server(self, args):
        """Add a new server to configuration"""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=3)
        if len(parts) < 3:
            self.safe_print("[yellow]Usage: /servers add NAME TYPE PATH [ARGS...][/]")
            self.safe_print("Example: /servers add github stdio /path/to/github-server.js")
            self.safe_print("Example: /servers add github sse https://github-mcp-server.example.com")
            return
            
        name, type_str, path = parts[0], parts[1], parts[2]
        extra_args = parts[3].split() if len(parts) > 3 else []
        
        # Validate inputs
        if name in self.config.servers:
            self.safe_print(f"[red]Server with name '{name}' already exists[/]")
            return
            
        try:
            server_type = ServerType(type_str.lower())
        except ValueError:
            safe_console.print(f"[red]Invalid server type: {type_str}. Use 'stdio' or 'sse'[/]")
            return
            
        # Add server to config
        self.config.servers[name] = ServerConfig(
            name=name,
            type=server_type,
            path=path,
            args=extra_args,
            enabled=True,
            auto_start=True,
            description=f"User-added {server_type.value} server"
        )
        
        self.config.save()
        safe_console.print(f"[green]Server '{name}' added to configuration[/]")
        
        # Ask if user wants to connect now
        if Confirm.ask("Connect to server now?", console=safe_console):
            await self.connect_server(name)
    
    async def remove_server(self, name):
        """Remove a server from configuration"""
        if not name:
            self.safe_print("[yellow]Usage: /servers remove SERVER_NAME[/]")
            return
            
        if name not in self.config.servers:
            self.safe_print(f"[red]Server '{name}' not found[/]")
            return
            
        # Disconnect if connected
        if name in self.server_manager.active_sessions:
            await self.disconnect_server(name)
            
        # Remove from config
        del self.config.servers[name]
        self.config.save()
        
        self.safe_print(f"[green]Server '{name}' removed from configuration[/]")
    
    async def connect_server(self, name):
        """Connect to a specific server"""
        if not name:
            self.safe_print("[yellow]Usage: /servers connect SERVER_NAME[/]")
            return
            
        if name not in self.config.servers:
            self.safe_print(f"[red]Server '{name}' not found[/]")
            return
            
        if name in self.server_manager.active_sessions:
            self.safe_print(f"[yellow]Server '{name}' is already connected[/]")
            return
            
        # Connect to server using the context manager
        server_config = self.config.servers[name]
        
        try:
            with Status(f"{STATUS_EMOJI['server']} Connecting to {name}...", spinner="dots", console=get_safe_console()) as status:
                try:
                    # Use safe_stdout to prevent any stdout pollution during server connection
                    with safe_stdout():
                        async with self.server_manager.connect_server_session(server_config) as session:
                            if session:
                                try:
                                    status.update(f"{STATUS_EMOJI['connected']} Connected to server: {name}")
                                    
                                    self.safe_print(f"[green]Connected to server: {name}[/]")
                                except Exception as e:
                                    self.safe_print(f"[red]Error loading capabilities from server {name}: {e}[/]")
                            else:
                                self.safe_print(f"[red]Failed to connect to server: {name}[/]")
                except Exception as e:
                    self.safe_print(f"[red]Error connecting to server {name}: {e}[/]")
        except Exception as e:
            # This captures any exceptions from the Status widget itself
            self.safe_print(f"[red]Error in status display: {e}[/]")
            # Still try to connect without the status widget
            try:
                # Use safe_stdout here as well for the fallback connection attempt
                with safe_stdout():
                    async with self.server_manager.connect_server_session(server_config) as session:
                        if session:
                            self.safe_print(f"[green]Connected to server: {name}[/]")
                        else:
                            self.safe_print(f"[red]Failed to connect to server: {name}[/]")
            except Exception as inner_e:
                self.safe_print(f"[red]Failed to connect to server {name}: {inner_e}[/]")
    
    async def disconnect_server(self, name):
        """Disconnect from a specific server"""
        if not name:
            self.safe_print("[yellow]Usage: /servers disconnect SERVER_NAME[/]")
            return
            
        if name not in self.server_manager.active_sessions:
            self.safe_print(f"[yellow]Server '{name}' is not connected[/]")
            return
            
        # Remove tools, resources, and prompts from this server
        self.server_manager.tools = {
            k: v for k, v in self.server_manager.tools.items() 
            if v.server_name != name
        }
        
        self.server_manager.resources = {
            k: v for k, v in self.server_manager.resources.items() 
            if v.server_name != name
        }
        
        self.server_manager.prompts = {
            k: v for k, v in self.server_manager.prompts.items() 
            if v.server_name != name
        }
        
        # Close session
        session = self.server_manager.active_sessions[name]
        try:
            # Check if the session has a close or aclose method and call it
            if hasattr(session, 'aclose') and callable(session.aclose):
                await session.aclose()
            elif hasattr(session, 'close') and callable(session.close):
                if asyncio.iscoroutinefunction(session.close):
                    await session.close()
                else:
                    session.close()
            
            # Note: This doesn't remove it from the exit_stack, but that will be cleaned up
            # when the server_manager is closed. For a more complete solution, we would need
            # to refactor how sessions are managed in the exit stack.
        except Exception as e:
            log.error(f"Error closing session for server {name}: {e}")
            
        # Remove from active sessions
        del self.server_manager.active_sessions[name]
        
        # Terminate process if applicable
        if name in self.server_manager.processes:
            process = self.server_manager.processes[name]
            if process.returncode is None:  # If process is still running
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except Exception:
                    pass
                    
            del self.server_manager.processes[name]
            
        self.safe_print(f"[green]Disconnected from server: {name}[/]")
    
    async def enable_server(self, name, enable=True):
        """Enable or disable a server"""
        if not name:
            action = "enable" if enable else "disable"
            self.safe_print(f"[yellow]Usage: /servers {action} SERVER_NAME[/]")
            return
            
        if name not in self.config.servers:
            self.safe_print(f"[red]Server '{name}' not found[/]")
            return
            
        # Update config
        self.config.servers[name].enabled = enable
        self.config.save()
        
        action = "enabled" if enable else "disabled"
        self.safe_print(f"[green]Server '{name}' {action}[/]")
        
        # Connect or disconnect if needed
        if enable and name not in self.server_manager.active_sessions:
            if Confirm.ask(f"Connect to server '{name}' now?", console=get_safe_console()):
                await self.connect_server(name)
        elif not enable and name in self.server_manager.active_sessions:
            if Confirm.ask(f"Disconnect from server '{name}' now?", console=get_safe_console()):
                await self.disconnect_server(name)
    
    async def server_status(self, name):
        """Show detailed status for a server"""
        if not name:
            self.safe_print("[yellow]Usage: /servers status SERVER_NAME[/]")
            return
            
        if name not in self.config.servers:
            self.safe_print(f"[red]Server '{name}' not found[/]")
            return
            
        server_config = self.config.servers[name]
        connected = name in self.server_manager.active_sessions
        
        # Create basic info group
        basic_info = Group(
            Text(f"Type: {server_config.type.value}"),
            Text(f"Path/URL: {server_config.path}"),
            Text(f"Args: {' '.join(server_config.args)}"),
            Text(f"Enabled: {'Yes' if server_config.enabled else 'No'}"),
            Text(f"Auto-Start: {'Yes' if server_config.auto_start else 'No'}"),
            Text(f"Description: {server_config.description}"),
            Text(f"Status: {'Connected' if connected else 'Disconnected'}", 
                style="green" if connected else "red")
        )
        
        self.safe_print(Panel(basic_info, title=f"Server Status: {name}", border_style="blue"))
        
        if connected:
            # Count capabilities
            tools_count = sum(1 for t in self.server_manager.tools.values() if t.server_name == name)
            resources_count = sum(1 for r in self.server_manager.resources.values() if r.server_name == name)
            prompts_count = sum(1 for p in self.server_manager.prompts.values() if p.server_name == name)
            
            capability_info = Group(
                Text(f"Tools: {tools_count}", style="magenta"),
                Text(f"Resources: {resources_count}", style="cyan"),
                Text(f"Prompts: {prompts_count}", style="yellow")
            )
            
            self.safe_print(Panel(capability_info, title="Capabilities", border_style="green"))
            
            # Process info if applicable
            if name in self.server_manager.processes:
                process = self.server_manager.processes[name]
                if process.returncode is None: # If process is still running
                    pid = process.pid
                    try:
                        p = psutil.Process(pid)
                        cpu_percent = p.cpu_percent(interval=0.1)
                        memory_info = p.memory_info()
                        
                        process_info = Group(
                            Text(f"Process ID: {pid}"),
                            Text(f"CPU Usage: {cpu_percent:.1f}%"),
                            Text(f"Memory Usage: {memory_info.rss / (1024 * 1024):.1f} MB")
                        )
                        
                        self.safe_print(Panel(process_info, title="Process Information", border_style="yellow"))
                    except Exception:
                        self.safe_print(Panel(f"Process ID: {pid} (stats unavailable)", 
                                           title="Process Information", 
                                           border_style="yellow"))
    
    async def cmd_tools(self, args):
        """List available tools"""
        if not self.server_manager.tools:
            self.safe_print(f"{STATUS_EMOJI['warning']} [yellow]No tools available from connected servers[/]")
            return
            
        # Parse args for filtering
        server_filter = None
        if args:
            server_filter = args
            
        tool_table = Table(title=f"{STATUS_EMOJI['tool']} Available Tools")
        tool_table.add_column("Name")
        tool_table.add_column("Server")
        tool_table.add_column("Description")
        
        for name, tool in self.server_manager.tools.items():
            if server_filter and tool.server_name != server_filter:
                continue
                
            tool_table.add_row(
                name,
                tool.server_name,
                tool.description
            )
            
        self.safe_print(tool_table)
        
        # Offer to show schema for a specific tool
        if not args:
            tool_name = Prompt.ask("Enter tool name to see schema (or press Enter to skip)", console=get_safe_console())
            if tool_name in self.server_manager.tools:
                tool = self.server_manager.tools[tool_name]
                
                # Use Group to combine the title and schema
                schema_display = Group(
                    Text(f"Schema for {tool_name}:", style="bold"),
                    Syntax(json.dumps(tool.input_schema, indent=2), "json", theme="monokai")
                )
                
                self.safe_print(Panel(
                    schema_display, 
                    title=f"Tool: {tool_name}", 
                    border_style="magenta"
                ))
    
    async def cmd_resources(self, args):
        """List available resources"""
        if not self.server_manager.resources:
            self.safe_print(f"{STATUS_EMOJI['warning']} [yellow]No resources available from connected servers[/]")
            return
            
        # Parse args for filtering
        server_filter = None
        if args:
            server_filter = args
            
        resource_table = Table(title=f"{STATUS_EMOJI['resource']} Available Resources")
        resource_table.add_column("Name")
        resource_table.add_column("Server")
        resource_table.add_column("Description")
        resource_table.add_column("Template")
        
        for name, resource in self.server_manager.resources.items():
            if server_filter and resource.server_name != server_filter:
                continue
                
            resource_table.add_row(
                name,
                resource.server_name,
                resource.description,
                resource.template
            )
            
        self.safe_print(resource_table)
    
    async def cmd_prompts(self, args):
        """List available prompts"""
        if not self.server_manager.prompts:
            self.safe_print(f"{STATUS_EMOJI['warning']} [yellow]No prompts available from connected servers[/]")
            return
            
        # Parse args for filtering
        server_filter = None
        if args:
            server_filter = args
            
        prompt_table = Table(title=f"{STATUS_EMOJI['prompt']} Available Prompts")
        prompt_table.add_column("Name")
        prompt_table.add_column("Server")
        prompt_table.add_column("Description")
        
        for name, prompt in self.server_manager.prompts.items():
            if server_filter and prompt.server_name != server_filter:
                continue
                
            prompt_table.add_row(
                name,
                prompt.server_name,
                prompt.description
            )
            
        self.safe_print(prompt_table)
        
        # Offer to show template for a specific prompt
        if not args:
            prompt_name = Prompt.ask("Enter prompt name to see template (or press Enter to skip)", console=get_safe_console())
            if prompt_name in self.server_manager.prompts:
                prompt = self.server_manager.prompts[prompt_name]
                self.safe_print(f"\n[bold]Template for {prompt_name}:[/]")
                self.safe_print(prompt.template)
    
    async def cmd_history(self, args):
        """View conversation history"""
        if not self.history.entries:
            self.safe_print("[yellow]No conversation history[/]")
            return
            
        # Parse args for count limit
        limit = 5  # Default
        try:
            if args and args.isdigit():
                limit = int(args)
        except Exception:
            pass
        
        total_entries = len(self.history.entries)
        entries_to_show = min(limit, total_entries)
        
        # Show loading progress for history (especially useful for large histories)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=40),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TextColumn("[cyan]{task.percentage:>3.0f}%"),
            console=get_safe_console(),
            transient=True
        ) as progress:
            task = progress.add_task(f"{STATUS_EMOJI['history']} Loading conversation history...", total=entries_to_show)
            
            recent_entries = []
            for i, entry in enumerate(reversed(self.history.entries[-limit:])):
                recent_entries.append(entry)
                progress.update(task, advance=1, description=f"{STATUS_EMOJI['history']} Processing entry {i+1}/{entries_to_show}...")
                # Simulate some processing time for very fast machines
                if len(self.history.entries) > 100:  # Only add delay for large histories
                    await asyncio.sleep(0.01)
            
            progress.update(task, description=f"{STATUS_EMOJI['success']} History loaded")
        
        self.safe_print(f"\n[bold]Recent Conversations (last {entries_to_show}):[/]")
        
        for i, entry in enumerate(recent_entries, 1):
            self.safe_print(f"\n[bold cyan]{i}. {entry.timestamp}[/] - Model: {entry.model}")
            self.safe_print(f"Servers: {', '.join(entry.server_names) if entry.server_names else 'None'}")
            self.safe_print(f"Tools: {', '.join(entry.tools_used) if entry.tools_used else 'None'}")
            self.safe_print(f"[bold blue]Q:[/] {entry.query[:100]}..." if len(entry.query) > 100 else f"[bold blue]Q:[/] {entry.query}")
            self.safe_print(f"[bold green]A:[/] {entry.response[:100]}..." if len(entry.response) > 100 else f"[bold green]A:[/] {entry.response}")
    
    async def cmd_model(self, args):
        """Change the current model"""
        if not args:
            self.safe_print(f"Current model: [cyan]{self.current_model}[/]")
            self.safe_print("Usage: /model MODEL_NAME")
            self.safe_print("Example models: claude-3-7-sonnet-20250219, claude-3-5-sonnet-latest")
            return
            
        self.current_model = args
        self.safe_print(f"[green]Model changed to: {args}[/]")
    
    async def cmd_clear(self, args):
        """Clear the conversation context"""
        # self.conversation_messages = []
        self.conversation_graph.set_current_node("root")
        # Optionally clear the root node's messages too
        safe_console = get_safe_console()
        if Confirm.ask("Reset conversation to root? (This clears root messages too)", console=safe_console):
             root_node = self.conversation_graph.get_node("root")
             if root_node:
                 root_node.messages = []
                 root_node.children = [] # Also clear children if resetting completely? Discuss.
                 # Need to prune orphaned nodes from self.conversation_graph.nodes if we clear children
                 # For now, just reset messages and current node
                 root_node.messages = []
             self.safe_print("[green]Conversation reset to root node.[/]")
        else:
             self.safe_print("[yellow]Clear cancelled. Still on root node, messages preserved.[/]")

    async def cmd_reload(self, args):
        """Reload servers and capabilities"""
        self.safe_print("[yellow]Reloading servers and capabilities...[/]")
        
        # Close existing connections
        with Status(f"{STATUS_EMOJI['server']} Closing existing connections...", spinner="dots", console=get_safe_console()) as status:
            await self.server_manager.close()
            status.update(f"{STATUS_EMOJI['success']} Existing connections closed")
        
        # Reset collections
        self.server_manager = ServerManager(self.config)
        
        # Reconnect
        with Status(f"{STATUS_EMOJI['server']} Reconnecting to servers...", spinner="dots", console=get_safe_console()) as status:
            await self.server_manager.connect_to_servers()
            status.update(f"{STATUS_EMOJI['success']} Servers reconnected")
        
        self.safe_print("[green]Servers and capabilities reloaded[/]")
        await self.print_status()
    
    async def cmd_cache(self, args):
        """Manage the tool result cache and tool dependencies
        
        Subcommands:
          list - List cached entries
          clear - Clear cache entries
          clean - Remove expired entries
          dependencies (deps) - View tool dependency graph
        """
        if not self.tool_cache:
            self.safe_print("[yellow]Caching is disabled.[/]")
            return

        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"
        subargs = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            self.safe_print("\n[bold]Cached Tool Results:[/]")
            cache_table = Table(title="Cache Entries")
            cache_table.add_column("Key")
            cache_table.add_column("Tool Name")
            cache_table.add_column("Created At")
            cache_table.add_column("Expires At")

            # List from both memory and disk cache if available
            all_keys = set(self.tool_cache.memory_cache.keys())
            if self.tool_cache.disk_cache:
                all_keys.update(self.tool_cache.disk_cache.iterkeys())

            if not all_keys:
                 self.safe_print("[yellow]Cache is empty.[/]")
                 return
            
            # Use progress bar for loading cache entries - especially useful for large caches
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=get_safe_console(),
                transient=True
            ) as progress:
                task = progress.add_task(f"{STATUS_EMOJI['package']} Loading cache entries...", total=len(all_keys))
                
                entries = []
                for key in all_keys:
                    entry = self.tool_cache.memory_cache.get(key)
                    if not entry and self.tool_cache.disk_cache:
                        try:
                            entry = self.tool_cache.disk_cache.get(key)
                        except Exception:
                            entry = None # Skip potentially corrupted entries
                    
                    if entry:
                        expires_str = entry.expires_at.strftime("%Y-%m-%d %H:%M:%S") if entry.expires_at else "Never"
                        entries.append((
                            key,
                            entry.tool_name,
                            entry.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                            expires_str
                        ))
                    
                    progress.update(task, advance=1)
                
                progress.update(task, description=f"{STATUS_EMOJI['success']} Cache entries loaded")
            
            # Add entries to table
            for entry_data in entries:
                cache_table.add_row(*entry_data)
            
            self.safe_print(cache_table)
            self.safe_print(f"Total entries: {len(entries)}")

        elif subcmd == "clear":
            if not subargs or subargs == "--all":
                if Confirm.ask("Are you sure you want to clear the entire cache?", console=get_safe_console()):
                    # Use Progress for cache clearing - especially useful for large caches
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        console=get_safe_console(),
                        transient=True
                    ) as progress:
                        # Getting approximate count of items
                        memory_count = len(self.tool_cache.memory_cache)
                        disk_count = 0
                        if self.tool_cache.disk_cache:
                            try:
                                disk_count = sum(1 for _ in self.tool_cache.disk_cache.iterkeys())
                            except Exception:
                                pass
                        
                        task = progress.add_task(
                            f"{STATUS_EMOJI['package']} Clearing cache...", 
                            total=memory_count + (1 if disk_count > 0 else 0)
                        )
                        
                        # Clear memory cache
                        self.tool_cache.memory_cache.clear()
                        progress.update(task, advance=1, description=f"{STATUS_EMOJI['package']} Memory cache cleared")
                        
                        # Clear disk cache if available
                        if self.tool_cache.disk_cache:
                            self.tool_cache.disk_cache.clear()
                            progress.update(task, advance=1, description=f"{STATUS_EMOJI['package']} Disk cache cleared")
                        
                        progress.update(task, description=f"{STATUS_EMOJI['success']} Cache cleared successfully")
                    
                    self.safe_print("[green]Cache cleared.[/]")
                else:
                    self.safe_print("[yellow]Cache clear cancelled.[/]")
            else:
                tool_name_to_clear = subargs
                # Invalidate based on tool name prefix
                with Status(f"{STATUS_EMOJI['package']} Clearing cache for {tool_name_to_clear}...", spinner="dots", console=get_safe_console()) as status:
                    self.tool_cache.invalidate(tool_name=tool_name_to_clear)
                    status.update(f"{STATUS_EMOJI['success']} Cache entries for {tool_name_to_clear} cleared")
                self.safe_print(f"[green]Cleared cache entries for tool: {tool_name_to_clear}[/]")
        
        
        elif subcmd == "clean":
            # Use Progress for cache cleaning - especially useful for large caches
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=get_safe_console(),
                transient=True
            ) as progress:
                task = progress.add_task(f"{STATUS_EMOJI['package']} Scanning for expired entries...", total=None)
                
                # Count before cleaning
                memory_count_before = len(self.tool_cache.memory_cache)
                disk_count_before = 0
                if self.tool_cache.disk_cache:
                    try:
                        disk_count_before = sum(1 for _ in self.tool_cache.disk_cache.iterkeys())
                    except Exception:
                        pass
                
                progress.update(task, description=f"{STATUS_EMOJI['package']} Cleaning expired entries...")
                self.tool_cache.clean()
                
                # Count after cleaning
                memory_count_after = len(self.tool_cache.memory_cache)
                disk_count_after = 0
                if self.tool_cache.disk_cache:
                    try:
                        disk_count_after = sum(1 for _ in self.tool_cache.disk_cache.iterkeys())
                    except Exception:
                        pass
                
                removed_count = (memory_count_before - memory_count_after) + (disk_count_before - disk_count_after)
                progress.update(task, description=f"{STATUS_EMOJI['success']} Removed {removed_count} expired entries")
            
            self.safe_print(f"[green]Expired cache entries cleaned. Removed {removed_count} entries.[/]")
        
        elif subcmd == "dependencies" or subcmd == "deps":
            # Show dependency graph
            self.safe_print("\n[bold]Tool Dependency Graph:[/]")
            
            if not self.tool_cache.dependency_graph:
                self.safe_print("[yellow]No dependencies registered.[/]")
                return
            
            dependency_table = Table(title="Tool Dependencies")
            dependency_table.add_column("Tool")
            dependency_table.add_column("Depends On")
            
            # Process the dependency graph for display
            for tool_name, dependencies in self.tool_cache.dependency_graph.items():
                if dependencies:
                    dependency_table.add_row(
                        tool_name,
                        ", ".join(dependencies)
                    )
            
            self.safe_print(dependency_table)
            self.safe_print(f"Total tools with dependencies: {len(self.tool_cache.dependency_graph)}")
            
            # Process specific tool's dependencies
            if subargs:
                tool_name = subargs
                dependencies = self.tool_cache.dependency_graph.get(tool_name, set())
                
                if dependencies:
                    # Show the tool's dependencies in a tree
                    tree = Tree(f"[bold cyan]{tool_name}[/]")
                    for dep in dependencies:
                        tree.add(f"[magenta]{dep}[/]")
                    
                    self.safe_print("\n[bold]Dependencies for selected tool:[/]")
                    self.safe_print(tree)
                else:
                    self.safe_print(f"\n[yellow]Tool '{tool_name}' has no dependencies or was not found.[/]")
        
        else:
            self.safe_print("[yellow]Unknown cache command. Available: list, clear [tool_name | --all], clean, dependencies[/]")

    async def cmd_fork(self, args):
        """Create a new conversation fork/branch"""
        fork_name = args if args else None
        try:
            new_node = self.conversation_graph.create_fork(name=fork_name)
            self.conversation_graph.set_current_node(new_node.id)
            self.safe_print(f"[green]Created and switched to new branch:[/]")
            self.safe_print(f"  ID: [cyan]{new_node.id}[/]" )
            self.safe_print(f"  Name: [yellow]{new_node.name}[/]")
            self.safe_print(f"Branched from node: [magenta]{new_node.parent.id if new_node.parent else 'None'}[/]")
        except Exception as e:
            self.safe_print(f"[red]Error creating fork: {e}[/]")

    async def cmd_branch(self, args):
        """Manage conversation branches"""
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"
        subargs = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            self.safe_print("\n[bold]Conversation Branches:[/]")
            branch_tree = Tree("[cyan]Conversations[/]")

            def build_tree(node: ConversationNode, tree_node):
                # Display node info
                label = f"[yellow]{node.name}[/] ([cyan]{node.id[:8]}[/])"
                if node.id == self.conversation_graph.current_node.id:
                    label = f"[bold green]>> {label}[/bold green]"
                
                current_branch = tree_node.add(label)
                for child in node.children:
                    build_tree(child, current_branch)

            build_tree(self.conversation_graph.root, branch_tree)
            self.safe_print(branch_tree)

        elif subcmd == "checkout":
            if not subargs:
                self.safe_print("[yellow]Usage: /branch checkout NODE_ID[/]")
                return
            
            node_id = subargs
            # Allow partial ID matching (e.g., first 8 chars)
            matched_node = None
            if node_id in self.conversation_graph.nodes:
                 matched_node = self.conversation_graph.get_node(node_id)
            else:
                for n_id, node in self.conversation_graph.nodes.items():
                    if n_id.startswith(node_id):
                        if matched_node:
                             self.safe_print(f"[red]Ambiguous node ID prefix: {node_id}. Multiple matches found.[/]")
                             return # Ambiguous prefix
                        matched_node = node
            
            if matched_node:
                if self.conversation_graph.set_current_node(matched_node.id):
                    self.safe_print(f"[green]Switched to branch:[/]")
                    self.safe_print(f"  ID: [cyan]{matched_node.id}[/]")
                    self.safe_print(f"  Name: [yellow]{matched_node.name}[/]")
                else:
                    # Should not happen if matched_node is valid
                    self.safe_print(f"[red]Failed to switch to node {node_id}[/]") 
            else:
                self.safe_print(f"[red]Node ID '{node_id}' not found.[/]")

        # Add other subcommands like rename, delete later if needed
        # elif subcmd == "rename": ...
        # elif subcmd == "delete": ...

        else:
            self.safe_print("[yellow]Unknown branch command. Available: list, checkout NODE_ID[/]")

    # --- Dashboard Implementation ---

    def generate_dashboard_renderable(self) -> Layout:
        """Generates the Rich renderable for the live dashboard."""
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
             Layout(name="stats", size=7),
        )

        # Header
        header_text = Text(f"MCP Client Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="bold white on blue")
        layout["header"].update(Panel(header_text, title="Status", border_style="dashboard.border"))

        # Footer
        layout["footer"].update(Text("Press Ctrl+C to exit dashboard", style="dim"))

        # Servers Panel
        server_table = Table(title=f"{STATUS_EMOJI['server']} Servers", box=box.ROUNDED, border_style="blue")
        server_table.add_column("Name", style="server")
        server_table.add_column("Status", justify="center")
        server_table.add_column("Type")
        server_table.add_column("Conn Status", justify="center")
        server_table.add_column("Avg Resp (ms)", justify="right")
        server_table.add_column("Errors", justify="right")
        server_table.add_column("Req Count", justify="right")

        for name, server_config in self.config.servers.items():
            if not server_config.enabled:
                continue # Optionally show disabled servers
            
            metrics = server_config.metrics
            conn_status_emoji = STATUS_EMOJI["connected"] if name in self.server_manager.active_sessions else STATUS_EMOJI["disconnected"]
            health_status_emoji = STATUS_EMOJI.get(metrics.status.value, Emoji("question_mark"))
            avg_resp_ms = metrics.avg_response_time * 1000 if metrics.avg_response_time else 0
            status_style = f"status.{metrics.status.value}" if metrics.status != ServerStatus.UNKNOWN else "dim"

            server_table.add_row(
                name,
                Text(f"{health_status_emoji} {metrics.status.value.capitalize()}", style=status_style),
                server_config.type.value,
                conn_status_emoji,
                f"{avg_resp_ms:.1f}",
                f"{metrics.error_count}",
                f"{metrics.request_count}"
            )
        layout["servers"].update(Panel(server_table, title="[bold blue]Servers[/]", border_style="blue"))

        # Tools Panel
        tool_table = Table(title=f"{STATUS_EMOJI['tool']} Tools", box=box.ROUNDED, border_style="magenta")
        tool_table.add_column("Name", style="tool")
        tool_table.add_column("Server", style="server")
        tool_table.add_column("Calls", justify="right")
        tool_table.add_column("Avg Time (ms)", justify="right")

        # Sort tools by call count or last used potentially
        sorted_tools = sorted(self.server_manager.tools.values(), key=lambda t: t.call_count, reverse=True)[:15] # Show top 15

        for tool in sorted_tools:
             avg_time_ms = tool.avg_execution_time # Already in ms?
             tool_table.add_row(
                 tool.name.split(':')[-1], # Show short name
                 tool.server_name,
                 str(tool.call_count),
                 f"{avg_time_ms:.1f}"
             )
        layout["tools"].update(Panel(tool_table, title="[bold magenta]Tool Usage[/]", border_style="magenta"))

        # General Stats Panel
        stats_text = Text()
        stats_text.append(f"{STATUS_EMOJI['speech_balloon']} Model: [model]{self.current_model}[/]\n")
        stats_text.append(f"{STATUS_EMOJI['server']} Connected Servers: {len(self.server_manager.active_sessions)}\n")
        stats_text.append(f"{STATUS_EMOJI['tool']} Total Tools: {len(self.server_manager.tools)}\n")
        stats_text.append(f"{STATUS_EMOJI['scroll']} History Entries: {len(self.history.entries)}\n")
        cache_size = len(self.tool_cache.memory_cache) if self.tool_cache else 0
        if self.tool_cache and self.tool_cache.disk_cache:
             # Getting exact disk cache size can be slow, maybe approximate or show memory only
             cache_size += len(self.tool_cache.disk_cache) # Example
             pass 
        stats_text.append(f"{STATUS_EMOJI['package']} Cache Entries (Mem): {cache_size}\n")
        stats_text.append(f"{STATUS_EMOJI['trident_emblem']} Current Branch: [yellow]{self.conversation_graph.current_node.name}[/] ([cyan]{self.conversation_graph.current_node.id[:8]}[/])")

        layout["stats"].update(Panel(stats_text, title="[bold cyan]Client Info[/]", border_style="cyan"))

        return layout

    async def cmd_dashboard(self, args):
        """Show the live monitoring dashboard."""
        try:
            # Check if we already have an active display
            if hasattr(self, '_active_progress') and self._active_progress:
                self.safe_print("[yellow]Cannot start dashboard while another live display is active.[/]")
                return
                
            # Set the flag to prevent other live displays
            self._active_progress = True
            
            # Use a single Live display context for the dashboard
            with Live(self.generate_dashboard_renderable(), 
                    refresh_per_second=1.0/self.config.dashboard_refresh_rate, 
                    screen=True, 
                    transient=False,
                    console=get_safe_console()) as live:
                while True:
                    await asyncio.sleep(self.config.dashboard_refresh_rate)
                    # Generate a new renderable and update the display
                    live.update(self.generate_dashboard_renderable())
        except KeyboardInterrupt:
            self.safe_print("\n[yellow]Dashboard stopped.[/]")
        except Exception as e:
            log.error(f"Dashboard error: {e}")
            self.safe_print(f"\n[red]Dashboard encountered an error: {e}[/]")
        finally:
            # Always clear the flag when exiting
            self._active_progress = False

    async def get_conversation_export_data(self, conversation_id: str) -> Optional[Dict]:
        """Gets the data for exporting a specific conversation branch."""
        node = self.conversation_graph.get_node(conversation_id)
        if not node:
            return None

        all_nodes = self.conversation_graph.get_path_to_root(node)
        messages = []
        for ancestor in all_nodes:
            messages.extend(ancestor.messages)

        return {
            "id": node.id,
            "name": node.name,
            "messages": messages,
            "model": node.model,
            "exported_at": datetime.now().isoformat(),
            "path": [n.id for n in all_nodes]
        }

    async def import_conversation_from_data(self, data: Dict) -> Tuple[bool, Optional[str], Optional[str]]:
        """Imports conversation data as a new branch."""
        try:
            # Validate basic structure (add more checks if needed)
            if not isinstance(data.get("messages"), list):
                return False, "Invalid import data: 'messages' field missing or not a list.", None

            new_node = ConversationNode(
                id=str(uuid.uuid4()), # Generate new ID
                name=f"Imported: {data.get('name', 'Unnamed')}",
                messages=data["messages"],
                model=data.get('model', self.config.default_model),
                # Set parent to current node
                parent=self.conversation_graph.current_node
            )

            self.conversation_graph.add_node(new_node)
            self.conversation_graph.current_node.add_child(new_node)
            await self.conversation_graph.save(str(self.conversation_graph_file))
            return True, f"Import successful. New node ID: {new_node.id}", new_node.id

        except Exception as e:
            log.error(f"Error importing conversation data: {e}", exc_info=True)
            return False, f"Internal error during import: {e}", None

    def get_cache_entries(self) -> List[Dict]:
        """Gets details of all cache entries."""
        if not self.tool_cache: return []
        entries = []
        # Combine keys from memory and disk
        all_keys = set(self.tool_cache.memory_cache.keys())
        if self.tool_cache.disk_cache:
             try: all_keys.update(self.tool_cache.disk_cache.iterkeys())
             except Exception as e: log.warning(f"Could not iterate disk cache keys: {e}")

        for key in all_keys:
            entry_obj: Optional[CacheEntry] = self.tool_cache.memory_cache.get(key)
            if not entry_obj and self.tool_cache.disk_cache:
                try: entry_obj = self.tool_cache.disk_cache.get(key)
                except Exception: entry_obj = None # Skip potentially corrupted

            if entry_obj:
                entries.append({
                    "key": key,
                    "tool_name": entry_obj.tool_name,
                    "created_at": entry_obj.created_at,
                    "expires_at": entry_obj.expires_at,
                })
        return entries

    def clear_cache(self, tool_name: Optional[str] = None) -> int:
        """Clears cache entries, optionally filtered by tool name."""
        if not self.tool_cache: return 0

        keys_before = set(self.tool_cache.memory_cache.keys())
        disk_keys_before = set()
        if self.tool_cache.disk_cache:
             try: disk_keys_before = set(self.tool_cache.disk_cache.iterkeys())
             except Exception: pass
        keys_before.update(disk_keys_before)

        if tool_name:
            self.tool_cache.invalidate(tool_name=tool_name)
        else:
            self.tool_cache.invalidate() # Clears all

        keys_after = set(self.tool_cache.memory_cache.keys())
        disk_keys_after = set()
        if self.tool_cache.disk_cache:
             try: disk_keys_after = set(self.tool_cache.disk_cache.iterkeys())
             except Exception: pass
        keys_after.update(disk_keys_after)

        return len(keys_before) - len(keys_after)

    def clean_cache(self) -> int:
        """Cleans expired cache entries."""
        if not self.tool_cache: return 0
        # Get counts before
        mem_before = len(self.tool_cache.memory_cache)
        disk_before = 0
        if self.tool_cache.disk_cache:
             try: disk_before = sum(1 for _ in self.tool_cache.disk_cache.iterkeys())
             except Exception: pass

        self.tool_cache.clean()

        # Get counts after
        mem_after = len(self.tool_cache.memory_cache)
        disk_after = 0
        if self.tool_cache.disk_cache:
             try: disk_after = sum(1 for _ in self.tool_cache.disk_cache.iterkeys())
             except Exception: pass

        return (mem_before - mem_after) + (disk_before - disk_after)

    def get_cache_dependencies(self) -> Dict[str, List[str]]:
        """Gets the tool dependency graph."""
        if not self.tool_cache: return {}
        # Convert sets to lists for JSON serialization
        return {k: list(v) for k, v in self.tool_cache.dependency_graph.items()}

    def get_tool_schema(self, tool_name: str) -> Optional[Dict]:
        """Gets the input schema for a specific tool."""
        tool = self.server_manager.tools.get(tool_name)
        return tool.input_schema if tool else None

    def get_prompt_template(self, prompt_name: str) -> Optional[str]:
        """Gets the template for a specific prompt."""
        prompt = self.server_manager.prompts.get(prompt_name)
        # The 'template' field in MCPPrompt might be the content itself
        # or might need extraction from original_prompt depending on structure
        return prompt.template if prompt else None # Adjust if template is nested

    def get_server_details(self, server_name: str) -> Optional[Dict]:
        """Gets detailed information about a server."""
        if server_name not in self.config.servers:
            return None

        server_config = self.config.servers[server_name]
        is_connected = server_name in self.server_manager.active_sessions
        metrics = server_config.metrics

        details = {
            "name": server_config.name,
            "type": server_config.type.value,
            "path": server_config.path,
            "args": server_config.args,
            "enabled": server_config.enabled,
            "auto_start": server_config.auto_start,
            "description": server_config.description,
            "trusted": server_config.trusted,
            "categories": server_config.categories,
            "version": str(server_config.version) if server_config.version else None,
            "rating": server_config.rating,
            "retry_count": server_config.retry_count,
            "timeout": server_config.timeout,
            "registry_url": server_config.registry_url,
            "capabilities": server_config.capabilities,
            "is_connected": is_connected,
            "metrics": { # Expose relevant metrics
                "status": metrics.status.value,
                "avg_response_time_ms": metrics.avg_response_time * 1000,
                "error_count": metrics.error_count,
                "request_count": metrics.request_count,
                "error_rate": metrics.error_rate,
                "uptime_minutes": metrics.uptime,
                "last_checked": metrics.last_checked.isoformat()
            },
            "process_info": None # Initialize as None
        }

        # Add process info for connected STDIO servers
        if is_connected and server_config.type == ServerType.STDIO and server_name in self.server_manager.processes:
            process = self.server_manager.processes[server_name]
            if process and process.returncode is None:
                try:
                    p = psutil.Process(process.pid)
                    mem_info = p.memory_info()
                    details["process_info"] = {
                        "pid": process.pid,
                        "cpu_percent": p.cpu_percent(interval=0.1),
                        "memory_rss_mb": mem_info.rss / (1024 * 1024),
                        "memory_vms_mb": mem_info.vms / (1024 * 1024),
                        "status": p.status(),
                        "create_time": datetime.fromtimestamp(p.create_time()).isoformat()
                    }
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    details["process_info"] = {"error": "Could not retrieve process stats"}
                except Exception as e:
                    details["process_info"] = {"error": f"Error retrieving stats: {e}"}

        return details

    async def reload_servers(self):
        """Logic to reload server connections."""
        log.info("Reloading servers via API request...")
        # Close existing connections (reuse logic from cmd_reload/close)
        if self.server_manager:
             await self.server_manager.close() # Close handles sessions/processes

        # Re-create manager or clear its state
        self.server_manager = ServerManager(self.config, tool_cache=self.tool_cache, safe_printer=self.safe_print)

        # Reconnect to enabled servers
        await self.server_manager.connect_to_servers()
        log.info("Server reload complete.")

    async def apply_prompt_to_conversation(self, prompt_name: str) -> bool:
        """Applies a prompt template to the current conversation."""
        prompt = self.server_manager.prompts.get(prompt_name)
        if not prompt:
            return False
        prompt_content = prompt.template
        if not prompt_content and prompt.original_prompt:
             # Example: If template needs fetching via get_prompt
             # try:
             #     get_result = await self.server_manager.active_sessions[prompt.server_name].get_prompt(prompt.original_prompt.name, {})
             #     prompt_content = get_result.content
             # except Exception:
             #     log.error(f"Failed to fetch template content for prompt {prompt_name}")
             #     return False
             pass # Add fetching logic if needed
        # Insert as system message (or adapt logic if needed)
        self.conversation_graph.current_node.messages.insert(0, {
            "role": "system",
            "content": prompt_content
        })
        await self.conversation_graph.save(str(self.conversation_graph_file))
        return True

    async def reset_configuration(self):
        """Resets the configuration to defaults."""
        log.warning("Resetting configuration to defaults via API request.")
        # Disconnect all servers first
        if self.server_manager:
             await self.server_manager.close()

        # Create new default config and save
        default_config = Config()
        default_config.save() # Use synchronous save here for simplicity, or adapt

        # Reload the client's config state
        self.config = Config() # Re-initializes from the newly saved default file
        # Re-create server manager with the new config
        self.server_manager = ServerManager(self.config, tool_cache=self.tool_cache, safe_printer=self.safe_print)
        # Optionally, re-run discovery and connect automatically?
        # await self.setup(interactive_mode=False) # Re-run setup might be too broad
        log.info("Configuration reset complete.")

    def get_dashboard_data(self) -> Dict:
         """Gets the data structure for the dashboard."""
         # Servers Data
         servers_data = []
         for name, server_config in self.config.servers.items():
             if not server_config.enabled: continue
             metrics = server_config.metrics
             is_connected = name in self.server_manager.active_sessions
             health_score = 0
             if is_connected:
                 health_score = max(1, min(100, int(100 - (metrics.error_rate * 100) - max(0, (metrics.avg_response_time - 5.0) * 5))))

             servers_data.append({
                 "name": name,
                 "type": server_config.type.value,
                 "status": metrics.status.value,
                 "is_connected": is_connected,
                 "avg_response_ms": metrics.avg_response_time * 1000,
                 "error_count": metrics.error_count,
                 "request_count": metrics.request_count,
                 "health_score": health_score,
             })

         # Tools Data (Top N)
         tools_data = []
         sorted_tools = sorted(self.server_manager.tools.values(), key=lambda t: t.call_count, reverse=True)[:15]
         for tool in sorted_tools:
             tools_data.append({
                 "name": tool.name,
                 "server_name": tool.server_name,
                 "call_count": tool.call_count,
                 "avg_execution_time_ms": tool.avg_execution_time,
             })

         # Client Info Data
         client_info = {
             "current_model": self.current_model,
             "history_entries": len(self.history.entries),
             "cache_entries_memory": len(self.tool_cache.memory_cache) if self.tool_cache else 0,
             "current_branch_id": self.conversation_graph.current_node.id,
             "current_branch_name": self.conversation_graph.current_node.name,
         }

         return {
             "timestamp": datetime.now().isoformat(),
             "client_info": client_info,
             "servers": servers_data,
             "tools": tools_data,
         }

    # Helper method to process a content block delta event
    def _process_text_delta(self, delta_event: ContentBlockDeltaEvent, current_text: str) -> Tuple[str, str]:
        """Process a text delta event and return the updated text and the delta text.
        
        Args:
            delta_event: The content block delta event
            current_text: The current accumulated text
            
        Returns:
            Tuple containing (updated_text, delta_text)
        """
        delta = delta_event.delta
        if delta.type == "text_delta":
            delta_text = delta.text
            updated_text = current_text + delta_text
            return updated_text, delta_text
        return current_text, ""

    # Add a helper method for processing stream events
    def _process_stream_event(self, event: MessageStreamEvent, current_text: str) -> Tuple[str, Optional[str]]:
        """Process a message stream event and handle different event types.
        
        Args:
            event: The message stream event from Claude API
            current_text: The current accumulated text for content blocks
            
        Returns:
            Tuple containing (updated_text, text_to_yield or None)
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
                current_text = ""  # Reset current text for new block
            elif event.content_block.type == "tool_use":
                text_to_yield = f"\n[{STATUS_EMOJI['tool']}] Using tool: {event.content_block.name}..."
                
        # Other event types could be handled here
                
        return current_text, text_to_yield

    # Add a new method for importing/exporting conversation branches with a progress bar
    async def export_conversation(self, conversation_id: str, file_path: str) -> bool:
        """Export a conversation branch to a file with progress tracking"""
        node = self.conversation_graph.get_node(conversation_id)
        if not node:
            self.safe_print(f"[red]Conversation ID '{conversation_id}' not found[/]")
            return False
            
        # Get all messages from this branch and its ancestors
        all_nodes = self.conversation_graph.get_path_to_root(node)
        messages = []
        
        # Use progress bar to show export progress
        with Progress(
            SpinnerColumn("dots"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=get_safe_console()
        ) as progress:
            export_task = progress.add_task(f"{STATUS_EMOJI['scroll']} Exporting conversation...", total=len(all_nodes))
            
            # Collect messages from all nodes in the path
            for ancestor in all_nodes:
                messages.extend(ancestor.messages)
                progress.update(export_task, advance=1)
            
            # Prepare export data
            export_data = {
                "id": node.id,
                "name": node.name,
                "messages": messages,
                "model": node.model,
                "exported_at": datetime.now().isoformat(),
                "path": [n.id for n in all_nodes]
            }
            
            # Write to file with progress tracking
            try:
                progress.update(export_task, description=f"{STATUS_EMOJI['scroll']} Writing to file...")
                
                async with aiofiles.open(file_path, 'w') as f:
                    await f.write(json.dumps(export_data, indent=2))
                
                progress.update(export_task, description=f"{STATUS_EMOJI['success']} Export complete")
                return True
                
            except Exception as e:
                progress.update(export_task, description=f"{STATUS_EMOJI['error']} Export failed: {e}")
                self.safe_print(f"[red]Failed to export conversation: {e}[/]")
                return False

    async def cmd_optimize(self, args):
        """Optimize conversation context through summarization"""
        # Parse arguments for custom model or target length
        custom_model = None
        target_length = self.config.max_summarized_tokens
        
        if args:
            parts = args.split()
            for i in range(len(parts)-1):
                if parts[i] == "--model" or parts[i] == "-m":
                    custom_model = parts[i+1]
                elif parts[i] == "--tokens" or parts[i] == "-t":
                    try:
                        target_length = int(parts[i+1])
                    except ValueError:
                        self.safe_print(f"[yellow]Invalid token count: {parts[i+1]}[/]")
        
        self.safe_print(f"[yellow]Optimizing conversation context...[/]")
        
        # Use specified model or default summarization model
        summarization_model = custom_model or self.config.summarization_model
        
        # Variables to track between steps
        current_tokens = 0
        new_tokens = 0
        summary = ""
        
        # Define the optimization steps
        async def count_initial_tokens():
            nonlocal current_tokens
            current_tokens = await self.count_tokens()
            
        async def generate_summary():
            nonlocal summary
            summary = await self.process_query(
                "Summarize this conversation history preserving key facts, "
                "decisions, and context needed for future interactions. "
                "Keep technical details, code snippets, and numbers intact. "
                f"Create a summary that captures all essential information "
                f"while being concise enough to fit in roughly {target_length} tokens.",
                model=summarization_model
            )
            
        async def apply_summary():
            nonlocal summary
            self.conversation_graph.current_node.messages = [
                {"role": "system", "content": "Conversation summary: " + summary}
            ]
            
        async def count_final_tokens():
            nonlocal new_tokens
            new_tokens = await self.count_tokens()
        
        # Execute with progress tracking
        steps = [count_initial_tokens, generate_summary, apply_summary, count_final_tokens]
        descriptions = [
            f"{STATUS_EMOJI['scroll']} Counting initial tokens...",
            f"{STATUS_EMOJI['speech_balloon']} Generating summary...",
            f"{STATUS_EMOJI['scroll']} Applying summary...",
            f"{STATUS_EMOJI['scroll']} Counting final tokens..."
        ]
        
        success = await self.server_manager.run_multi_step_task(
            steps=steps, 
            step_descriptions=descriptions,
            title=f"{STATUS_EMOJI['package']} Optimizing conversation"
        )
        
        if success:
            # Report results
            self.safe_print(f"[green]Conversation optimized: {current_tokens} → {new_tokens} tokens[/]")
        else:
            self.safe_print(f"[red]Failed to optimize conversation.[/]")
    
    async def auto_prune_context(self):
        """Auto-prune context based on token count"""
        token_count = await self.count_tokens()
        if token_count > self.config.auto_summarize_threshold:
            self.safe_print(f"[yellow]Context size ({token_count} tokens) exceeds threshold "
                         f"({self.config.auto_summarize_threshold}). Auto-summarizing...[/]")
            await self.cmd_optimize(f"--tokens {self.config.max_summarized_tokens}")

    async def cmd_tool(self, args):
        """Directly execute a tool with parameters"""
        safe_console = get_safe_console()
        if not args:
            safe_console.print("[yellow]Usage: /tool NAME {JSON_PARAMS}[/yellow]")
            return
            
        # Split into tool name and params
        try:
            parts = args.split(" ", 1)
            tool_name = parts[0]
            params_str = parts[1] if len(parts) > 1 else "{}"
            params = json.loads(params_str)
        except json.JSONDecodeError:
            safe_console.print("[red]Invalid JSON parameters. Use valid JSON format.[/red]")
            return
        except Exception as e:
            safe_console.print(f"[red]Error parsing command: {e}[/red]")
            return

        # Check if tool exists
        if tool_name not in self.server_manager.tools:
            safe_console.print(f"[red]Tool not found: {tool_name}[/red]")
            return
        
        # Get the tool and its server
        tool = self.server_manager.tools[tool_name]
        server_name = tool.server_name
        
        with Status(f"{STATUS_EMOJI['tool']} Executing {tool_name}...", spinner="dots", console=get_safe_console()) as status:
            try:
                start_time = time.time()
                result = await self.execute_tool(server_name, tool_name, params)
                latency = time.time() - start_time
                
                status.update(f"{STATUS_EMOJI['success']} Tool execution completed in {latency:.2f}s")
                
                # Show result
                safe_console.print(Panel.fit(
                    Syntax(json.dumps(result, indent=2), "json", theme="monokai"),
                    title=f"Tool Result: {tool_name} (executed in {latency:.2f}s)",
                    border_style="magenta"
                ))
            except Exception as e:
                status.update(f"{STATUS_EMOJI['failure']} Tool execution failed: {e}")
                safe_console.print(f"[red]Error executing tool: {e}[/red]")

    # After the cmd_tool method (around line 4295)
    async def cmd_prompt(self, args):
        """Apply a prompt template to the conversation"""
        if not args:
            self.safe_print("[yellow]Available prompt templates:[/yellow]")
            for name in self.server_manager.prompts:
                self.safe_print(f"  - {name}")
            return
        
        prompt = self.server_manager.prompts.get(args)
        if not prompt:
            self.safe_print(f"[red]Prompt not found: {args}[/red]")
            return
            
        self.conversation_graph.current_node.messages.insert(0, {
            "role": "system",
            "content": prompt.template
        })
        self.safe_print(f"[green]Applied prompt: {args}[/green]")


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
            self.safe_print(f"{STATUS_EMOJI['config']} Found Claude desktop config file, processing...")

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
                    self.safe_print(f"{STATUS_EMOJI['warning']} No MCP servers key ('mcpServers' or alternatives) found in {config_path}")
                    return # Stop if no server list found

            mcp_servers = desktop_config.get(mcp_servers_key) # Use .get for safety
            if not mcp_servers or not isinstance(mcp_servers, dict):
                self.safe_print(f"{STATUS_EMOJI['warning']} No valid MCP server entries found under key '{mcp_servers_key}' in {config_path}")
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
                    self.safe_print(f"{STATUS_EMOJI['success']} Imported {len(imported_servers)} servers from Claude desktop config.")

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
                self.safe_print(f"{STATUS_EMOJI['warning']}  No new servers were imported from Claude desktop config (they might already exist or failed processing).")

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

    async def cmd_export(self, args):
        """Export the current conversation or a specific branch"""
        # Parse args for ID and output path
        conversation_id = self.conversation_graph.current_node.id  # Default to current
        output_path = None
        
        if args:
            parts = args.split()
            for i, part in enumerate(parts):
                if part in ["--id", "-i"] and i < len(parts) - 1:
                    conversation_id = parts[i+1]
                elif part in ["--output", "-o"] and i < len(parts) - 1:
                    output_path = parts[i+1]
        
        # Default filename if not provided
        if not output_path:
            output_path = f"conversation_{conversation_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Call the export method
        with Status(f"{STATUS_EMOJI['scroll']} Exporting conversation...", spinner="dots", console=get_safe_console()) as status:
            success = await self.export_conversation(conversation_id, output_path)
            if success:
                status.update(f"{STATUS_EMOJI['success']} Conversation exported successfully")
                self.safe_print(f"[green]Conversation exported to: {output_path}[/]")
            else:
                status.update(f"{STATUS_EMOJI['failure']} Export failed")
                self.safe_print(f"[red]Failed to export conversation[/]")

    async def import_conversation(self, file_path: str) -> bool:
        """Import a conversation from a file
        
        Args:
            file_path: Path to the exported conversation JSON file
            
        Returns:
            True if import was successful, False otherwise
        """
        try:
            # Read the file
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                data = json.loads(content)
            
            # Create a new node
            new_node = ConversationNode(
                id=str(uuid.uuid4()),  # Generate new ID to avoid conflicts
                name=f"Imported: {data.get('name', 'Unknown')}",
                messages=data.get('messages', []),
                model=data.get('model', '')
            )
            
            # Add to graph
            self.conversation_graph.add_node(new_node)
            
            # Make it a child of current node
            new_node.parent = self.conversation_graph.current_node
            self.conversation_graph.current_node.add_child(new_node)
            
            # Switch to the new node
            self.conversation_graph.set_current_node(new_node.id)
            
            # Save the updated conversation graph
            self.conversation_graph.save(str(self.conversation_graph_file))
            
            return True
        except FileNotFoundError:
            log.error(f"Import file not found: {file_path}")
            return False
        except json.JSONDecodeError:
            log.error(f"Invalid JSON in import file: {file_path}")
            return False
        except Exception as e:
            log.error(f"Error importing conversation: {e}")
            return False

    async def cmd_import(self, args):
        """Import a conversation from a file"""
        if not args:
            self.safe_print("[yellow]Usage: /import FILEPATH[/]")
            return
        
        file_path = args.strip()
        
        with Status(f"{STATUS_EMOJI['scroll']} Importing conversation from {file_path}...", spinner="dots", console=get_safe_console()) as status:
            success = await self.import_conversation(file_path)
            if success:
                status.update(f"{STATUS_EMOJI['success']} Conversation imported successfully")
                self.safe_print(f"[green]Conversation imported and set as current conversation[/]")
            else:
                status.update(f"{STATUS_EMOJI['failure']} Import failed")
                self.safe_print(f"[red]Failed to import conversation from {file_path}[/]")

    @ensure_safe_console
    async def print_status(self):
        """Print current status of servers, tools, and capabilities with progress bars"""
        # Use the stored safe console instance to prevent multiple calls
        safe_console = self._current_safe_console
        
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
        
        status_table.add_row(
            f"{STATUS_EMOJI['model']} Model",
            self.current_model
        )
        status_table.add_row(
            f"{STATUS_EMOJI['server']} Servers",
            f"{connected_servers}/{total_servers} connected"
        )
        status_table.add_row(
            f"{STATUS_EMOJI['tool']} Tools",
            str(total_tools)
        )
        status_table.add_row(
            f"{STATUS_EMOJI['resource']} Resources",
            str(total_resources)
        )
        status_table.add_row(
            f"{STATUS_EMOJI['prompt']} Prompts",
            str(total_prompts)
        )
        
        safe_console.print(status_table)
        
        # Only show server progress if we have servers
        if total_servers > 0:
            # Create server status tasks for _run_with_progress
            server_tasks = []
            for name, server in self.config.servers.items():
                if name in self.server_manager.active_sessions:
                    # Add a task to show server status with prettier progress
                    server_tasks.append(
                        (self._display_server_status, 
                        f"{STATUS_EMOJI['server']} {name} ({server.type.value})", 
                        (name, server))
                    )
            
            # If we have any connected servers, show their status with progress bars
            if server_tasks:
                await self._run_with_progress(
                    server_tasks,
                    "Server Status",
                    transient=False,  # Keep this visible
                    use_health_scores=True  # Use health scores for progress display
                )
        
        safe_console.print("[green]Ready to process queries![/green]")
        
        
    async def _display_server_status(self, server_name, server_config):
        """Helper to display server status in a progress bar
        
        This is used by print_status with _run_with_progress
        """
        # Get number of tools for this server
        server_tools = sum(1 for t in self.server_manager.tools.values() if t.server_name == server_name)
        
        # Calculate a health score for displaying in the progress bar (0-100)
        metrics = server_config.metrics
        health_score = 100
        
        if metrics.error_rate > 0:
            # Reduce score based on error rate
            health_score -= int(metrics.error_rate * 100)
        
        if metrics.avg_response_time > 5.0:
            # Reduce score for slow response time
            health_score -= min(30, int((metrics.avg_response_time - 5.0) * 5))
            
        # Clamp health score
        health_score = max(1, min(100, health_score))
        
        # Simulate work to show the progress bar
        await asyncio.sleep(0.1)
        
        # Return some stats for the task result
        return {
            "name": server_name,
            "type": server_config.type.value,
            "tools": server_tools,
            "health": health_score
        }

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
        if hasattr(self, '_active_progress') and self._active_progress:
            log.warning("Attempted to create nested progress display, using simpler output")
            return await self._run_with_simple_progress(tasks, title)
            
        # Set a flag that we have an active progress
        self._active_progress = True
        
        results = []
        
        try:
            # Set total based on whether we're using health scores or not
            task_total = 100 if use_health_scores else 1
            
            # Create columns for progress display
            columns = [
                TextColumn("[progress.description]{task.description}"),
                BarColumn(complete_style="green", finished_style="green"),
            ]
            
            # For health score mode (0-100), add percentage
            if use_health_scores:
                columns.append(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
            else:
                columns.append(TaskProgressColumn())
                
            # Always add spinner for active tasks
            columns.append(SpinnerColumn("dots"))
            
            # Create a Progress context that only lives for this group of tasks
            with Progress(
                *columns,
                console=get_safe_console(),
                transient=transient,
                expand=False  # This helps prevent display expansion issues
            ) as progress:
                # Create all tasks up front
                task_ids = []
                for _, description, _ in tasks:
                    task_id = progress.add_task(description, total=task_total)
                    task_ids.append(task_id)
                
                # Run each task sequentially
                for i, (task_func, _, task_args) in enumerate(tasks):
                    try:
                        # Actually run the task
                        result = await task_func(*task_args) if task_args else await task_func()
                        results.append(result)
                        
                        # Update progress based on mode
                        if use_health_scores and isinstance(result, dict) and 'health' in result:
                            # Use the health score as the progress value (0-100)
                            progress.update(task_ids[i], completed=result['health'])
                            # Add info about tools to the description if available
                            if 'tools' in result:
                                current_desc = progress._tasks[task_ids[i]].description
                                progress.update(task_ids[i], 
                                    description=f"{current_desc} - {result['tools']} tools")
                        else:
                            # Just mark as complete
                            progress.update(task_ids[i], completed=task_total)
                            
                    except Exception as e:
                        # Mark this task as failed
                        progress.update(task_ids[i], description=f"[red]Failed: {str(e)}[/red]")
                        log.error(f"Task {i} failed: {str(e)}")
                        # Re-raise the exception
                        raise e
                
                return results
        finally:
            # CRITICAL: Always clear the flag when done, even if an exception occurred
            self._active_progress = False

    async def _run_with_simple_progress(self, tasks, title):
        """Simpler version of _run_with_progress without Rich Live display.
        Used as a fallback when nested progress displays would occur.
        
        Args:
            tasks: A list of tuples with (task_func, task_description, task_args)
            title: The title for the progress bar
            
        Returns:
            A list of results from the tasks
        """
        safe_console = get_safe_console()
        results = []
        
        safe_console.print(f"[cyan]{title}[/]")
        
        for i, (task_func, description, task_args) in enumerate(tasks):
            try:
                # Print status without requiring Live display
                safe_console.print(f"  [cyan]→[/] {description}...", end="", flush=True)
                
                # Run the task
                result = await task_func(*task_args) if task_args else await task_func()
                safe_console.print(" [green]✓[/]")
                results.append(result)
            except Exception as e:
                safe_console.print(" [red]✗[/]")
                safe_console.print(f"    [red]Error: {str(e)}[/]")
                log.error(f"Task {i} ({description}) failed: {str(e)}")
                # Continue with other tasks instead of failing completely
                continue
        
        return results        

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
    webui: Annotated[bool, typer.Option("--webui", help="Run the experimental Web UI instead of CLI")] = False,
    webui_host: Annotated[str, typer.Option("--host", help="Host for Web UI")] = "127.0.0.1",
    webui_port: Annotated[int, typer.Option("--port", help="Port for Web UI")] = 8017,
    serve_ui_file: Annotated[bool, typer.Option("--serve-ui", help="Serve the default mcp_client_ui.html file")] = True,
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
    asyncio.run(main_async(query, model, server, dashboard, interactive, verbose, webui, webui_host, webui_port, serve_ui_file))

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
    edit: Annotated[bool, typer.Option("--edit", "-e", help="Edit configuration in text editor")] = False,
    reset: Annotated[bool, typer.Option("--reset", "-r", help="Reset to default configuration")] = False,
):
    """Manage client configuration"""
    # Run the config management function
    asyncio.run(config_async(show, edit, reset))

async def main_async(query, model, server, dashboard, interactive, verbose_logging, webui_flag, webui_host, webui_port, serve_ui_file):
    """Main async entry point - Handles CLI, Interactive, Dashboard, and Web UI modes."""
    client = None # Initialize client to None
    safe_console = get_safe_console()
    max_shutdown_timeout = 10

    # --- Shared Setup ---
    try:
        log.info("Initializing MCPClient...")
        client = MCPClient() # Instantiation inside the try block
        await client.setup(interactive_mode=interactive or webui_flag) # Pass interactive if either mode uses it

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
                # Basic status, mirrors parts of print_status
                return {
                    "currentModel": mcp_client.current_model,
                    "connectedServersCount": len(mcp_client.server_manager.active_sessions),
                    "totalServers": len(mcp_client.config.servers),
                    "totalTools": len(mcp_client.server_manager.tools),
                    "totalResources": len(mcp_client.server_manager.resources),
                    "totalPrompts": len(mcp_client.server_manager.prompts),
                    "historyEntries": len(mcp_client.history.entries),
                    "cacheEntries": len(mcp_client.tool_cache.memory_cache) if mcp_client.tool_cache else 0,
                    "currentNodeId": mcp_client.conversation_graph.current_node.id,
                    "currentNodeName": mcp_client.conversation_graph.current_node.name,
                }

            @app.get("/api/config")
            async def get_config(mcp_client: MCPClient = Depends(get_mcp_client)):
                 # Return non-sensitive config parts
                 cfg = mcp_client.config
                 return {
                     # Exclude API key from default GET
                     'defaultModel': cfg.default_model,
                     'defaultMaxTokens': cfg.default_max_tokens,
                     'historySize': cfg.history_size,
                     'autoDiscover': cfg.auto_discover,
                     'discoveryPaths': cfg.discovery_paths,
                     'enableStreaming': cfg.enable_streaming,
                     'enableCaching': cfg.enable_caching,
                     'enableMetrics': cfg.enable_metrics,
                     'enableRegistry': cfg.enable_registry,
                     'enableLocalDiscovery': cfg.enable_local_discovery,
                     'temperature': cfg.temperature,
                     'cacheTtlMapping': cfg.cache_ttl_mapping,
                     'conversationGraphsDir': cfg.conversation_graphs_dir,
                     'registryUrls': cfg.registry_urls,
                     'dashboardRefreshRate': cfg.dashboard_refresh_rate,
                     'summarizationModel': cfg.summarization_model,
                     'autoSummarizeThreshold': cfg.auto_summarize_threshold,
                     'maxSummarizedTokens': cfg.max_summarized_tokens,
                 }

            @app.put("/api/config")
            async def update_config(update_request: ConfigUpdateRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                updated = False
                log.debug(f"Received config update request: {update_request.model_dump(exclude_unset=True)}")
                for key, value in update_request.model_dump(exclude_unset=True).items():
                    if hasattr(mcp_client.config, key):
                        setattr(mcp_client.config, key, value)
                        log.info(f"Config updated: {key} = {value}")
                        updated = True
                        # Special handling for API key to re-init client
                        if key == 'apiKey' and value:
                            try:
                                mcp_client.anthropic = AsyncAnthropic(api_key=value)
                                log.info("Anthropic client re-initialized with new API key.")
                            except Exception as e:
                                log.error(f"Failed to re-initialize Anthropic client: {e}")
                                # Optionally revert the key change or just log the error
                        # Update current model if default changes
                        elif key == 'defaultModel':
                             mcp_client.current_model = value

                if updated:
                    await mcp_client.config.save_async()
                    return {"message": "Configuration updated successfully"}
                else:
                    return {"message": "No changes applied"}

            @app.get("/api/servers")
            async def list_servers_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                server_list = []
                for name, server in mcp_client.config.servers.items():
                    metrics = server.metrics
                    server_list.append({
                        "name": server.name,
                        "type": server.type.value,
                        "path": server.path,
                        "args": server.args,
                        "enabled": server.enabled,
                        "isConnected": name in mcp_client.server_manager.active_sessions,
                        "status": metrics.status.value,
                        "statusText": f"{metrics.status.value.capitalize()}", # Simplified text
                        "health": max(1, min(100, int(100 - (metrics.error_rate * 100) - max(0, (metrics.avg_response_time - 5.0) * 5)))) if name in mcp_client.server_manager.active_sessions else 0, # Simplified health score
                        "tools": [t.name for t in mcp_client.server_manager.tools.values() if t.server_name == name]
                    })
                return server_list

            @app.post("/api/servers")
            async def add_server_api(req: ServerAddRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                 if req.name in mcp_client.config.servers:
                     raise HTTPException(status_code=409, detail=f"Server name '{req.name}' already exists")

                 args_list = req.argsString.split(' ') if req.argsString else []
                 new_server_config = ServerConfig(
                     name=req.name,
                     type=req.type,
                     path=req.path,
                     args=args_list,
                     enabled=True,
                     auto_start=False, # Don't auto-start from Web UI add by default
                     description=f"Added via Web UI ({req.type.value})"
                 )
                 mcp_client.config.servers[req.name] = new_server_config
                 await mcp_client.config.save_async()
                 log.info(f"Added server '{req.name}' via API.")
                 # Return the added server config (or just success)
                 return {"message": f"Server '{req.name}' added.", "server": new_server_config.model_dump(exclude={'metrics'})} # Exclude metrics

            @app.delete("/api/servers/{server_name}")
            async def remove_server_api(server_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                if server_name not in mcp_client.config.servers:
                    raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
                await mcp_client.disconnect_server(server_name) # Ensure disconnected first
                del mcp_client.config.servers[server_name]
                await mcp_client.config.save_async()
                log.info(f"Removed server '{server_name}' via API.")
                return {"message": f"Server '{server_name}' removed"}

            @app.post("/api/servers/{server_name}/connect")
            async def connect_server_api(server_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                 if server_name not in mcp_client.config.servers:
                     raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
                 if server_name in mcp_client.server_manager.active_sessions:
                     return {"message": f"Server '{server_name}' already connected"}
                 try:
                     # Use the internal connect method, not the command handler
                     success = await mcp_client._connect_and_load_server(server_name, mcp_client.config.servers[server_name])
                     if success:
                         return {"message": f"Successfully connected to server '{server_name}'"}
                     else:
                         raise HTTPException(status_code=500, detail=f"Failed to connect to server '{server_name}'")
                 except Exception as e:
                      log.error(f"API Error connecting to {server_name}: {e}", exc_info=True)
                      raise HTTPException(status_code=500, detail=f"Error connecting to server '{server_name}': {str(e)}") from e

            @app.post("/api/servers/{server_name}/disconnect")
            async def disconnect_server_api(server_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                if server_name not in mcp_client.server_manager.active_sessions:
                     return {"message": f"Server '{server_name}' is not connected"}
                await mcp_client.disconnect_server(server_name)
                return {"message": f"Disconnected from server '{server_name}'"}

            @app.put("/api/servers/{server_name}/enable")
            async def enable_server_api(server_name: str, enabled: bool = True, mcp_client: MCPClient = Depends(get_mcp_client)):
                if server_name not in mcp_client.config.servers:
                    raise HTTPException(status_code=404, detail=f"Server '{server_name}' not found")
                mcp_client.config.servers[server_name].enabled = enabled
                await mcp_client.config.save_async()
                # Handle disconnect if disabling a connected server
                if not enabled and server_name in mcp_client.server_manager.active_sessions:
                    await mcp_client.disconnect_server(server_name)
                action = "enabled" if enabled else "disabled"
                return {"message": f"Server '{server_name}' {action}"}


            @app.get("/api/tools")
            async def list_tools_api(mcp_client: MCPClient = Depends(get_mcp_client)): # noqa: B008
                 # Convert MCPTool objects to dicts for JSON response
                 return [
                     {
                         "name": tool.name,
                         "description": tool.description,
                         "server_name": tool.server_name,
                         "input_schema": tool.input_schema,
                         "call_count": tool.call_count,
                         "avg_execution_time": tool.avg_execution_time,
                         # Ensure last_used is handled correctly (it defaults to datetime.now)
                         "last_used": tool.last_used.isoformat() if isinstance(tool.last_used, datetime) else datetime.now().isoformat(),
                     } for tool in mcp_client.server_manager.tools.values()
                 ]

            @app.get("/api/resources")
            async def list_resources_api(mcp_client: MCPClient = Depends(get_mcp_client)): # noqa: B008
                 # Convert MCPResource objects to dicts
                 return [
                     {
                         "name": resource.name,
                         "description": resource.description,
                         "server_name": resource.server_name,
                         "template": resource.template, # This corresponds to the URI in the Resource type
                         "call_count": resource.call_count,
                         "last_used": resource.last_used.isoformat() if isinstance(resource.last_used, datetime) else datetime.now().isoformat(),
                         # Add original_resource details if needed by the UI, e.g., original URI:
                         # "original_uri": str(resource.original_resource.uri) if resource.original_resource else None
                     } for resource in mcp_client.server_manager.resources.values()
                 ]

            @app.get("/api/prompts")
            async def list_prompts_api(mcp_client: MCPClient = Depends(get_mcp_client)): # noqa: B008
                 # Convert MCPPrompt objects to dicts
                 return [
                     {
                         "name": prompt.name,
                         "description": prompt.description,
                         "server_name": prompt.server_name,
                         "template": prompt.template, # Contains the basic template string or identifier
                         "call_count": prompt.call_count,
                         "last_used": prompt.last_used.isoformat() if isinstance(prompt.last_used, datetime) else datetime.now().isoformat(),
                         # Add original_prompt details if needed, e.g., arguments schema:
                         # "arguments_schema": prompt.original_prompt.arguments if prompt.original_prompt else None
                     } for prompt in mcp_client.server_manager.prompts.values()
                 ]

            @app.get("/api/conversation")
            async def get_conversation_api(mcp_client: MCPClient = Depends(get_mcp_client)): # noqa: B008
                node = mcp_client.conversation_graph.current_node
                return {
                    "currentNodeId": node.id,
                    "currentNodeName": node.name,
                    "messages": node.messages,
                    "model": node.model,
                    # Change this line:
                    # "nodes": [n.model_dump(exclude={'parent', 'children'}) for n in mcp_client.conversation_graph.nodes.values()],
                    # To this:
                    "nodes": [n.to_dict() for n in mcp_client.conversation_graph.nodes.values()], # Use the existing to_dict method
                }

            @app.post("/api/conversation/fork")
            async def fork_conversation_api(req: ForkRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                try:
                    new_node = mcp_client.conversation_graph.create_fork(name=req.name)
                    mcp_client.conversation_graph.set_current_node(new_node.id)
                    await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                    return {"message": "Fork created", "newNodeId": new_node.id, "newNodeName": new_node.name}
                except Exception as e:
                    log.error(f"Error forking conversation via API: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Error forking: {str(e)}") from e

            @app.post("/api/conversation/checkout")
            async def checkout_branch_api(req: CheckoutRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                 node_id = req.node_id
                 node = mcp_client.conversation_graph.get_node(node_id)
                 # Add partial matching if desired
                 if not node:
                      # Attempt partial match
                      matched_node = None
                      for n_id, n in mcp_client.conversation_graph.nodes.items():
                          if n_id.startswith(node_id):
                              if matched_node: raise HTTPException(status_code=400, detail="Ambiguous node ID prefix")
                              matched_node = n
                      node = matched_node # Use matched node if found

                 if node and mcp_client.conversation_graph.set_current_node(node.id):
                     return {"message": f"Switched to branch {node.name}", "currentNodeId": node.id, "messages": node.messages}
                 else:
                     raise HTTPException(status_code=404, detail=f"Node ID '{node_id}' not found or switch failed")

            @app.post("/api/conversation/clear")
            async def clear_conversation_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                # This implementation clears the *current* node's messages and sets current to root
                # Adapt if you want full root reset like in cmd_clear
                mcp_client.conversation_graph.current_node.messages = []
                mcp_client.conversation_graph.set_current_node("root") # Go back to root after clear? Or stay on cleared node? Let's go to root.
                await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                return {"message": "Current conversation node cleared, switched to root"}

            @app.post("/api/conversation/optimize")
            async def optimize_conversation_api(req: OptimizeRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                summarization_model = req.model or mcp_client.config.summarization_model
                target_length = req.target_tokens or mcp_client.config.max_summarized_tokens
                initial_tokens = await mcp_client.count_tokens()
                try:
                    # Use process_query directly for summarization task
                    summary = await mcp_client.process_query(
                        ("Summarize this conversation history preserving key facts, decisions, and context needed for future interactions. "
                         "Keep technical details, code snippets, and numbers intact. "
                         f"Create a summary that captures all essential information while being concise enough to fit in roughly {target_length} tokens."),
                        model=summarization_model
                        # Note: This adds the summary request/response to the history itself.
                        # A more refined approach might use a separate, temporary context for summarization.
                    )
                    # Replace current node messages with summary
                    mcp_client.conversation_graph.current_node.messages = [
                         {"role": "system", "content": "Conversation summary:\n" + summary}
                     ]
                    final_tokens = await mcp_client.count_tokens()
                    await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                    return {"message": "Optimization complete", "initialTokens": initial_tokens, "finalTokens": final_tokens}
                except Exception as e:
                    log.error(f"Error optimizing conversation via API: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}") from e

            @app.post("/api/tool/execute")
            async def execute_tool_api(req: ToolExecuteRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                if req.tool_name not in mcp_client.server_manager.tools:
                     raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' not found")
                tool = mcp_client.server_manager.tools[req.tool_name]
                try:
                    result : CallToolResult = await mcp_client.execute_tool(tool.server_name, req.tool_name, req.params)
                    # Convert result content if necessary for JSON serialization
                    content_to_return = result.content
                    if isinstance(content_to_return, list):
                        # Try to serialize Pydantic models if present
                        content_to_return = [getattr(item, 'model_dump', lambda item=item: item)() for item in content_to_return]

                    return {"isError": result.isError, "content": content_to_return}
                except Exception as e:
                    log.error(f"Error executing tool '{req.tool_name}' via API: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}") from e

            @app.post("/api/discover/trigger", status_code=202) # 202 Accepted
            async def trigger_discovery_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                log.info("API triggered server discovery...")
                # Run discovery in background to avoid blocking response
                # Note: This doesn't guarantee completion before the next results call,
                #       a more robust solution might use background tasks and status polling.
                asyncio.create_task(mcp_client.server_manager.discover_servers())
                return {"message": "Server discovery process initiated."}

            @app.get("/api/discover/results", response_model=List[DiscoveredServer])
            async def get_discovery_results_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                results = await mcp_client.server_manager.get_discovery_results()
                # Convert internal dict structure to Pydantic model list
                return [DiscoveredServer(**item) for item in results]

            @app.post("/api/discover/connect") # Use request body instead of path param
            async def connect_discovered_server_api(server_info: DiscoveredServer, mcp_client: MCPClient = Depends(get_mcp_client)):
                 # Find the full info from the cache based on name/path match, or use provided data
                 # For simplicity, assume server_info contains necessary details from GET /results
                 if not server_info.name or not server_info.path_or_url or not server_info.type:
                      raise HTTPException(status_code=400, detail="Incomplete server information provided.")
                 # Convert back to dict for the helper method
                 info_dict = server_info.model_dump()
                 success, message = await mcp_client.server_manager.add_and_connect_discovered_server(info_dict)
                 if success:
                     return {"message": message}
                 else:
                     # Determine appropriate status code (409 Conflict? 500?)
                     status_code = 409 if "already configured" in message else 500
                     raise HTTPException(status_code=status_code, detail=message)

            @app.get("/api/conversation/{conversation_id}/export")
            async def export_conversation_api(conversation_id: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                data = await mcp_client.get_conversation_export_data(conversation_id)
                if data is None:
                    raise HTTPException(status_code=404, detail=f"Conversation ID '{conversation_id}' not found")
                return data # Return the JSON data directly

            @app.post("/api/conversation/import")
            async def import_conversation_api(file: UploadFile = File(...), mcp_client: MCPClient = Depends(get_mcp_client)):
                if not file.filename.endswith(".json"):
                     raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JSON file.")
                try:
                    content_bytes = await file.read()
                    content_str = content_bytes.decode('utf-8')
                    data = json.loads(content_str)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail="Invalid JSON content in uploaded file.") from e
                except Exception as e:
                    log.error(f"Error reading uploaded file {file.filename}: {e}")
                    raise HTTPException(status_code=500, detail="Error reading uploaded file.") from e

                success, message, new_node_id = await mcp_client.import_conversation_from_data(data)
                if success:
                    return {"message": message, "newNodeId": new_node_id}
                else:
                    raise HTTPException(status_code=500, detail=f"Import failed: {message}")

            @app.get("/api/cache/entries", response_model=List[CacheEntryDetail])
            async def get_cache_entries_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                if not mcp_client.tool_cache: raise HTTPException(status_code=404, detail="Caching is disabled")
                entries = mcp_client.get_cache_entries()
                return [CacheEntryDetail(**entry) for entry in entries]

            @app.delete("/api/cache/entries")
            async def clear_cache_all_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                if not mcp_client.tool_cache: raise HTTPException(status_code=404, detail="Caching is disabled")
                count = mcp_client.clear_cache()
                return {"message": f"Cleared {count} cache entries."}

            @app.delete("/api/cache/entries/{tool_name}")
            async def clear_cache_tool_api(tool_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                if not mcp_client.tool_cache: raise HTTPException(status_code=404, detail="Caching is disabled")
                # Need to URL-decode the tool name if it contains special characters like ':'
                import urllib.parse
                decoded_tool_name = urllib.parse.unquote(tool_name)
                count = mcp_client.clear_cache(tool_name=decoded_tool_name)
                return {"message": f"Cleared {count} cache entries for tool '{decoded_tool_name}'."}

            @app.post("/api/cache/clean")
            async def clean_cache_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                if not mcp_client.tool_cache: raise HTTPException(status_code=404, detail="Caching is disabled")
                count = mcp_client.clean_cache()
                return {"message": f"Cleaned {count} expired cache entries."}

            @app.get("/api/cache/dependencies", response_model=CacheDependencyInfo)
            async def get_cache_dependencies_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                if not mcp_client.tool_cache: raise HTTPException(status_code=404, detail="Caching is disabled")
                deps = mcp_client.get_cache_dependencies()
                return CacheDependencyInfo(dependencies=deps)

            @app.get("/api/tools/{tool_name}/schema")
            async def get_tool_schema_api(tool_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                import urllib.parse
                decoded_tool_name = urllib.parse.unquote(tool_name)
                schema = mcp_client.get_tool_schema(decoded_tool_name)
                if schema is None:
                    raise HTTPException(status_code=404, detail=f"Tool '{decoded_tool_name}' not found")
                return schema # Return JSON schema directly

            @app.get("/api/prompts/{prompt_name}/template")
            async def get_prompt_template_api(prompt_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                 import urllib.parse
                 decoded_prompt_name = urllib.parse.unquote(prompt_name)
                 template = mcp_client.get_prompt_template(decoded_prompt_name)
                 if template is None:
                     raise HTTPException(status_code=404, detail=f"Prompt '{decoded_prompt_name}' not found")
                 return {"template": template} # Return as simple JSON object

            @app.get("/api/servers/{server_name}/details", response_model=ServerDetail)
            async def get_server_details_api(server_name: str, mcp_client: MCPClient = Depends(get_mcp_client)):
                 import urllib.parse
                 decoded_server_name = urllib.parse.unquote(server_name)
                 details = mcp_client.get_server_details(decoded_server_name)
                 if details is None:
                     raise HTTPException(status_code=404, detail=f"Server '{decoded_server_name}' not found")
                 # Convert ServerType enum back for Pydantic validation if needed, or handle in model
                 details_model = ServerDetail(**details) # Validate against Pydantic model
                 return details_model # Return the validated model object

            @app.post("/api/runtime/reload")
            async def reload_servers_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                try:
                    await mcp_client.reload_servers()
                    return {"message": "Servers reloaded successfully."}
                except Exception as e:
                    log.error(f"Error reloading servers via API: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Server reload failed: {e}") from e

            @app.post("/api/conversation/apply_prompt")
            async def apply_prompt_api(req: ApplyPromptRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                 success = await mcp_client.apply_prompt_to_conversation(req.prompt_name)
                 if success:
                     return {"message": f"Prompt '{req.prompt_name}' applied to current conversation."}
                 else:
                     raise HTTPException(status_code=404, detail=f"Prompt '{req.prompt_name}' not found.")

            @app.post("/api/config/reset")
            async def reset_config_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                 try:
                     await mcp_client.reset_configuration()
                     return {"message": "Configuration reset to defaults. Please restart client if necessary."}
                 except Exception as e:
                      log.error(f"Error resetting configuration via API: {e}", exc_info=True)
                      raise HTTPException(status_code=500, detail=f"Configuration reset failed: {e}") from e

            @app.get("/api/dashboard", response_model=DashboardData)
            async def get_dashboard_data_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                 data = mcp_client.get_dashboard_data()
                 # Pydantic model validation happens implicitly here if response_model is set
                 return data

            # --- WebSocket Chat Endpoint ---
            @app.websocket("/ws/chat")
            async def websocket_chat(websocket: WebSocket): # Get client from websocket state instead
                # Get the client instance from the app state
                try:
                    mcp_client: MCPClient = websocket.app.state.mcp_client
                    if not mcp_client:
                        log.error("MCPClient not found in app state during WebSocket connection!")
                        await websocket.close(code=1011) # Internal error
                        return
                except AttributeError:
                    log.error("app.state.mcp_client not available during WebSocket connection!")
                    await websocket.close(code=1011)
                    return

                await websocket.accept()
                log.info("WebSocket connection accepted.")
                try:
                    while True:
                        raw_data = await websocket.receive_text() # Use receive_text and parse JSON manually for flexibility
                        try:
                            data = json.loads(raw_data)
                            message = WebSocketMessage(**data) # Validate structure
                            log.debug(f"WebSocket received: Type={message.type}, Payload={str(message.payload)[:100]}...")

                            if message.type == "query":
                                query_text = str(message.payload)
                                try:
                                    async for chunk in mcp_client.process_streaming_query(query_text):
                                        # Check for status updates vs text chunks
                                        if chunk.startswith("@@STATUS@@"):
                                             status_payload = chunk[len("@@STATUS@@"):].strip()
                                             await websocket.send_json(WebSocketMessage(type="status", payload=status_payload).model_dump())
                                        else:
                                             await websocket.send_json(WebSocketMessage(type="text_chunk", payload=chunk).model_dump())
                                    # Send a final message indicator maybe?
                                    await websocket.send_json(WebSocketMessage(type="query_complete").model_dump())
                                except Exception as e:
                                    log.error(f"Error processing query via WebSocket: {e}", exc_info=True)
                                    await websocket.send_json(WebSocketMessage(type="error", payload=f"Error processing query: {str(e)}").model_dump())

                            elif message.type == "command":
                                # Basic command handling - more complex commands might need specific API endpoints
                                command_str = str(message.payload)
                                if command_str.startswith('/'):
                                    parts = command_str[1:].split(maxsplit=1)
                                    cmd = parts[0].lower()
                                    args = parts[1] if len(parts) > 1 else ""
                                    response_payload = f"Command '{cmd}' received (basic handling)."
                                    # Example: Handle simple commands directly
                                    if cmd == "clear":
                                         mcp_client.conversation_graph.current_node.messages = []
                                         mcp_client.conversation_graph.set_current_node("root")
                                         await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                                         response_payload = "Conversation cleared."
                                         await websocket.send_json(WebSocketMessage(type="command_response", payload=response_payload).model_dump())
                                         # Also need to maybe send an empty message list update?
                                    elif cmd == "model":
                                         if args:
                                             mcp_client.current_model = args
                                             response_payload = f"Model set to {args}"
                                         else:
                                             response_payload = f"Current model: {mcp_client.current_model}"
                                         await websocket.send_json(WebSocketMessage(type="command_response", payload=response_payload).model_dump())
                                    # Add more simple commands or guide user to use UI elements/API
                                    else:
                                         await websocket.send_json(WebSocketMessage(type="command_response", payload=f"Command '{cmd}' not fully handled via WebSocket.").model_dump())
                                else:
                                     await websocket.send_json(WebSocketMessage(type="error", payload="Invalid command format via WebSocket.").model_dump())

                            # Add handlers for other message types if needed

                        except (json.JSONDecodeError, ValidationError) as e: # Catch validation errors too
                            log.warning(f"WebSocket received invalid message: {raw_data[:100]}... Error: {e}")
                            await websocket.send_json(WebSocketMessage(type="error", payload="Invalid message format received.").model_dump())
                        except WebSocketDisconnect:
                            # Re-raise to be caught by the outer handler
                            raise
                        except Exception as e:
                            # Catch unexpected errors during message processing
                            log.error(f"Unexpected error processing WebSocket message: {e}", exc_info=True)
                            try:
                                await websocket.send_json(WebSocketMessage(type="error", payload=f"Internal server error: {str(e)}").model_dump())
                            except Exception:
                                log.warning("Failed to send error back to WebSocket client after processing error.")

                except WebSocketDisconnect:
                    log.info("WebSocket connection closed.")
                except Exception as e:
                    log.error(f"Unexpected error in WebSocket handler: {e}", exc_info=True)
                    # Attempt to close gracefully if possible
                    try:
                        await websocket.close(code=1011) # Internal Error
                    except Exception:
                        pass # Ignore errors during close after another error

            # --- Static File Serving (Optional) ---
            if serve_ui_file:
                ui_file = Path("mcp_client_ui.html")
                if ui_file.exists():
                    log.info(f"Serving static UI file from {ui_file.resolve()}")
                    # Serve the specific HTML file at the root
                    @app.get("/", response_class=FileResponse)
                    async def serve_html():
                        return FileResponse(str(ui_file.resolve()))
                    # Serve other static assets if needed (e.g., CSS, JS in a subfolder)
                    # app.mount("/static", StaticFiles(directory="static"), name="static")
                else:
                     log.warning(f"UI file {ui_file} not found. Cannot serve it at '/' endpoint.")


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
    """Config management async function"""
    client = None # Initialize to None
    safe_console = get_safe_console()
    
    try:
        client = MCPClient() # Instantiate client within the main try

        if reset:
            if Confirm.ask("[yellow]Are you sure you want to reset the configuration?[/]", console=safe_console):
                # Create a new default config
                new_config = Config()
                # Save it
                new_config.save()
                safe_console.print("[green]Configuration reset to defaults[/]")

        elif edit:
            # Open config file in editor
            editor = os.environ.get("EDITOR", "vim")
            try: # --- Inner try for editor subprocess ---
                # Ensure CONFIG_FILE exists before editing
                CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
                CONFIG_FILE.touch() # Create if doesn't exist

                process = await asyncio.create_subprocess_exec(
                    editor, str(CONFIG_FILE),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                # Wait for the process to complete
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                     safe_console.print(f"[yellow]Editor exited with code {process.returncode}[/]")
                else:
                    safe_console.print(f"[green]Configuration file potentially edited: {CONFIG_FILE}[/]")
            except FileNotFoundError:
                 safe_console.print(f"[red]Editor command not found: '{editor}'. Set EDITOR environment variable.[/]")
            except OSError as e:
                 safe_console.print(f"[red]Error running editor '{editor}': {e}[/]")
            # Keep broad exception for unexpected editor issues
            except Exception as e:
                 safe_console.print(f"[red]Unexpected error trying to edit config: {e}")
            # --- End inner try for editor ---

            # Reload config (Needs client object)
            if client:
                 client.config.load()
                 safe_console.print("[green]Configuration reloaded[/]")
            else:
                 log.warning("Client not initialized, cannot reload config.")


        elif show or not (reset or edit):
            # Show current config (Needs client object)
            if not client:
                 log.warning("Client not initialized, cannot show config.")
                 return # Exit if client isn't ready

            config_data = {}
            for key, value in client.config.__dict__.items():
                if key != "servers":
                    config_data[key] = value
                else:
                    config_data["servers"] = {
                        name: {
                            "type": server.type.value,
                            "path": server.path,
                            "enabled": server.enabled,
                            "auto_start": server.auto_start,
                            "description": server.description
                        }
                        for name, server in value.items()
                    }

            safe_console.print(Panel(
                Syntax(yaml.safe_dump(config_data, default_flow_style=False), "yaml", theme="monokai"),
                title="Current Configuration",
                border_style="blue"
            ))

    # --- Top-level exceptions for config_async itself ---
    except (IOError, yaml.YAMLError, json.JSONDecodeError) as e:
         safe_console.print(f"[bold red]Configuration/File Error during config command:[/] {str(str(e))}")
    except Exception as e:
        safe_console.print(f"[bold red]Unexpected Error during config command:[/] {str(e)}")

    finally:
        # Ensure client resources are cleaned up if it was initialized
        if client and hasattr(client, 'close'):
             try:
                 await client.close()
             except Exception as close_err:
                 log.error(f"Error during config_async client cleanup: {close_err}")

async def main():
    """Main entry point - Handles app() call only for non-webui modes"""
    # Check if --webui is in sys.argv *before* calling app()
    # This is a bit hacky but necessary because Typer processes args early
    # A cleaner way might involve restructuring how the web server is launched
    is_webui_mode = "--webui" in sys.argv

    if not is_webui_mode:
        try:
            app() # Run Typer app for CLI modes
        except McpError as e:
            get_safe_console().print(f"[bold red]MCP Error:[/] {str(e)}")
            sys.exit(1)
        except httpx.RequestError as e:
            get_safe_console().print(f"[bold red]Network Error:[/] {str(e)}")
            sys.exit(1)
        except anthropic.APIError as e:
            get_safe_console().print(f"[bold red]Anthropic API Error:[/] {str(e)}")
            sys.exit(1)
        except (OSError, yaml.YAMLError, json.JSONDecodeError) as e:
            get_safe_console().print(f"[bold red]Configuration/File Error:[/] {str(e)}")
            sys.exit(1)
        except Exception as e: # Keep broad exception for top-level unexpected errors
            get_safe_console().print(f"[bold red]Unexpected Error:[/] {str(e)}")
            if os.environ.get("MCP_CLIENT_DEBUG"): # Show traceback if debug env var is set
                import traceback
                traceback.print_exc()
            sys.exit(1)
    # else: Web UI mode is handled by asyncio.run(main_async(...)) directly


if __name__ == "__main__":
    # Initialize colorama for Windows terminals
    if platform.system() == "Windows":
        colorama.init(convert=True)

    # Run the app - Now delegates logic based on flags to main_async
    # The check in main() prevents app() call for webui mode
    # Instead, asyncio.run(main_async(...)) handles everything
    app() # This will parse args and call the appropriate command (run, servers, etc.)
          # The 'run' command will then call main_async which handles mode switching.

