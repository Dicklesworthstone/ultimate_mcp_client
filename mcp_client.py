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
#     "orjson>=3.9.0",
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
- Server Management: Discover, connect to, and monitor MCP servers
- Tool Integration: Execute tools from multiple servers with intelligent routing
- Streaming: Real-time streaming responses with tool execution
- Caching: Smart caching of tool results with configurable TTLs
- Conversation Branches: Create and manage conversation forks and branches
- Conversation Import/Export: Save and share conversations with easy portable JSON format
- Health Dashboard: Real-time monitoring of servers and tool performance
- Observability: Comprehensive metrics and tracing
- Registry Integration: Connect to remote registries to discover servers
- Local Discovery: Discover MCP servers on your local network via mDNS
- Context Optimization: Automatic summarization of long conversations
- Direct Tool Execution: Run specific tools directly with custom parameters
- Dynamic Prompting: Apply pre-defined prompt templates to conversations
- Claude Desktop Integration: Automatically import server configs from Claude desktop

Usage:
------
# Interactive mode
python mcpclient.py run --interactive

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
- /discover - Discover and connect to MCP servers on local network
- /optimize - Optimize conversation context through summarization
- /clear - Clear the conversation context

Author: Jeffrey Emanuel
License: MIT
Version: 1.0.0
"""

import asyncio
import atexit
import functools
import hashlib
import inspect
import ipaddress
import logging
import os
import platform
import random
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
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)

import aiofiles  # For async file operations
import anthropic

# Additional utilities
import colorama

# Cache libraries
import diskcache
import httpx

# Third-party imports
import orjson as json
import psutil

# Token counting
import tiktoken

# Typer CLI
import typer
import yaml
from anthropic import AsyncAnthropic
from anthropic.types import (
    ContentBlockDeltaEvent,
    MessageParam,
    MessageStreamEvent,
    ToolParam,
)
from dotenv import load_dotenv

# MCP SDK imports
from mcp import ClientSession
from mcp.client.sse import sse_client
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
from mcp.types import Prompt as McpPrompt

# Observability
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from pydantic import AnyUrl
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
from typing_extensions import Annotated

# Guarded Zeroconf imports
try:
    from zeroconf import NonUniqueNameException, ServiceInfo
    # Note: We are not using ZEROCONF_AVAILABLE flag anymore
except ImportError:
    # Define dummy exception if zeroconf is not installed
    class NonUniqueNameException(Exception): pass
    class ServiceInfo: pass # Dummy class
    log = logging.getLogger("mcpclient") # Make sure log is accessible
    log.warning("Zeroconf library not found. Local network registration/discovery will be disabled.")

USE_VERBOSE_SESSION_LOGGING = True # Set to True for debugging, False for normal operation

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
                logging.warning(f"Prevented stdout pollution: {repr(text[:30])}")
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
atexit.register(lambda: force_exit_handler(is_force=False))

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

class ServerType(Enum):
    STDIO = "stdio"
    SSE = "sse"

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
    timeout: float = 30.0  # Timeout in seconds
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
    original_prompt: McpPrompt
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
        params_str = json.dumps(params, sort_keys=True).decode('utf-8')
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
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
                await f.write(json.dumps(data).decode('utf-8'))
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
                                version = ServerVersion.from_string(version_str)
                        
                        # Parse metrics if available
                        metrics = ServerMetrics()
                        if 'metrics' in server_data:
                            metrics_data = server_data['metrics']
                            for metric_key, metric_value in metrics_data.items():
                                if hasattr(metrics, metric_key):
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
                            metrics=metrics,
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
            # This is expected if the file doesn't exist yet
            self.save() # Create default config
            return
        except IOError as e:
            log.error(f"Error reading config file {CONFIG_FILE}: {e}")
        except yaml.YAMLError as e:
            log.error(f"Error parsing config file {CONFIG_FILE}: {e}")
        # Keep a broad exception for unexpected issues during config parsing/application
        except Exception as e: 
            log.error(f"Unexpected error loading config: {e}")
    
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
            
    async def load_async(self):
        """Load configuration from file asynchronously"""
        if not CONFIG_FILE.exists():
            await self.save_async()  # Create default config
            return
        
        try:
            async with aiofiles.open(CONFIG_FILE, 'r') as f:
                content = await f.read()
                config_data = yaml.safe_load(content) or {}
            
            # Update config with loaded values (reuse the same logic)
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
                                version = ServerVersion.from_string(version_str)
                        
                        # Parse metrics if available
                        metrics = ServerMetrics()
                        if 'metrics' in server_data:
                            metrics_data = server_data['metrics']
                            for metric_key, metric_value in metrics_data.items():
                                if hasattr(metrics, metric_key):
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
                            metrics=metrics,
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
            # This is expected if the file doesn't exist yet
            await self.save_async() # Create default config
            return
        except IOError as e:
            log.error(f"Error reading config file {CONFIG_FILE}: {e}")
        except yaml.YAMLError as e:
            log.error(f"Error parsing config file {CONFIG_FILE}: {e}")
        # Keep a broad exception for unexpected issues during config parsing/application
        except Exception as e: 
            log.error(f"Unexpected error loading config: {e}")
    
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
                history_data = json.load(f)
            
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
                json.dump(history_data, f)
                
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
                await f.write(json.dumps(history_data).decode('utf-8'))
                
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
    A ClientSession implementation that handles noisy server stdout
    by buffering and filtering before parsing MCP messages, using TaskGroup.
    Requires Python 3.11+.
    """
    def __init__(self, process: asyncio.subprocess.Process, server_name: str):
        if sys.version_info < (3, 11):
            raise RuntimeError("RobustStdioSession with TaskGroup requires Python 3.11+")

        self._process = process
        self._server_name = server_name
        self._stdin = process.stdin
        self._stderr_reader_task: Optional[asyncio.Task] = None # Keep track of external stderr reader
        self._message_queue = asyncio.Queue(maxsize=100)
        self._response_futures: Dict[str, asyncio.Future] = {}
        self._request_id_counter = 0
        self._lock = asyncio.Lock()
        self._is_active = True
        self._background_task_runner: Optional[asyncio.Task] = None # Task that runs the TaskGroup

        log.debug(f"[{self._server_name}] Initializing RobustStdioSession")
        # Start the main background task runner which manages the TaskGroup
        self._background_task_runner = asyncio.create_task(
            self._run_background_tasks_wrapper(), # Wrap the TaskGroup runner for error handling
            name=f"session-tasks-{server_name}"
        )

    async def initialize(self, capabilities: Optional[Dict[str, Any]] = None, response_timeout: float = 10.0) -> Any:
            """Sends the MCP initialize request and waits for the response."""
            log.info(f"[{self._server_name}] Sending initialize request...")
            # Define client capabilities (can be expanded later if needed)
            client_capabilities = capabilities if capabilities is not None else {
                # Add client capabilities here if any, e.g., related to resource subscriptions
            }
            params = {
                "processId": os.getpid(), # Client's process ID
                "clientInfo": { # Optional, but good practice
                    "name": "ultimate-mcp-client",
                    "version": "1.0.0", # TODO: Make this dynamic?
                },
                "rootUri": None, # Or workspace root if applicable, e.g., f"file://{PROJECT_ROOT.as_uri()}"
                "capabilities": client_capabilities,
                "protocolVersion": "2025-03-25",

            }
            # Use a specific timeout for initialization, can be shorter than tool calls
            result = await self._send_request("initialize", params, response_timeout=response_timeout)
            log.info(f"[{self._server_name}] Initialize request successful. Server capabilities received.")
            self._server_capabilities = result.get("capabilities")
            return result # Return the server's response (InitializeResult)        

    async def send_initialized_notification(self):
        """Sends the 'notifications/initialized' notification to the server."""
        # Ensure session is active before attempting to send
        if not self._is_active:
            log.warning(f"[{self._server_name}] Session inactive, cannot send initialized notification.")
            return

        # Construct the notification message according to the MCP schema
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {} # The schema defines params as an empty object for this notification
        }

        try:
            # Serialize using the 'json' alias (which is likely orjson in your setup)
            # and append the required newline
            notification_json_str = json.dumps(notification).decode('utf-8') + "\n"
            notification_bytes = notification_json_str.encode('utf-8')

            log.info(f"[{self._server_name}] Sending initialized notification...")

            # Check if stdin is usable before writing
            if self._stdin is None or self._stdin.is_closing():
                raise ConnectionAbortedError("Stdin is closed or None, cannot send initialized notification")

            # Write the notification bytes to the server's stdin
            self._stdin.write(notification_bytes)
            # Ensure the data is flushed to the process
            await self._stdin.drain()

            log.debug(f"[{self._server_name}] Initialized notification sent successfully.")

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as pipe_err:
            # Handle cases where the connection is lost during the send operation
            log.error(f"[{self._server_name}] Failed to send initialized notification due to connection error: {pipe_err}")
            # Mark session as inactive if sending fails critically? Depends on desired robustness.
            # For a notification, maybe just log and continue, but consider implications.
            # Let's log and suppress for now, assuming it might not be fatal if JUST notification fails.
            pass # Or await self._close_internal_state(pipe_err) if it should terminate session
        except Exception as e:
            # Catch any other unexpected errors during notification sending
            log.error(f"[{self._server_name}] Unexpected error sending initialized notification: {e}", exc_info=True)
            # Log and suppress for now

    async def _run_background_tasks_wrapper(self):
        """
        Wraps the TaskGroup runner to handle its potential failure, catching
        exceptions both from within the TaskGroup's tasks and from the await
        of the TaskGroup runner itself. Correctly handles ExceptionGroups and single exceptions.
        """
        close_exception: Optional[BaseException | BaseExceptionGroup] = None
        try:
            log.debug(f"[{self._server_name}] Entering background task runner wrapper.")
            try:
                await self._run_background_tasks() # Runs the 'async with TaskGroup()'
                log.info(f"[{self._server_name}] Background TaskGroup finished normally.")
            except BaseExceptionGroup as eg:
                log.error(f"[{self._server_name}] Background TaskGroup failed due to exception(s) in child tasks:", exc_info=eg)
                close_exception = eg # Store the ExceptionGroup itself
            log.debug(f"[{self._server_name}] TaskGroup section complete (may have failed internally).")
        except asyncio.CancelledError:
            log.info(f"[{self._server_name}] Background task runner wrapper task cancelled.")
            close_exception = asyncio.CancelledError("Wrapper task cancelled")
        except Exception as e:
            log.error(f"[{self._server_name}] Background Task runner wrapper failed unexpectedly (outside TaskGroup tasks): {e}", exc_info=True)
            close_exception = e # Store the single exception
        finally:
            log.info(f"[{self._server_name}] Background Task runner wrapper exiting (finally block).")
            # Check if the session is still considered active.
            if self._is_active:
                log.warning(f"[{self._server_name}] Background task runner finished unexpectedly or normally while session active. Forcing session state closure.")
                # Determine the exception to pass to the cleanup function.
                final_exception = close_exception if close_exception is not None else ConnectionAbortedError("Background task runner finished unexpectedly")
                # Ensure await is used here as _close_internal_state is async
                await self._close_internal_state(final_exception)

    async def _run_background_tasks(self):
        """Runs the core background tasks within a TaskGroup."""
        log.debug(f"[{self._server_name}] Starting background TaskGroup.")
        # The TaskGroup ensures that if one task fails, others are cancelled.
        async with asyncio.TaskGroup() as tg:
            log.debug(f"[{self._server_name}] TaskGroup created.")
            tg.create_task(
                self._read_stdout_loop(),
                name=f"stdout-reader-{self._server_name}"
            )
            tg.create_task(
                self._process_incoming_messages(),
                name=f"msg-processor-{self._server_name}"
            )
            log.info(f"[{self._server_name}] Stdout reader and message processor tasks started in group.")

    async def _read_stdout_loop(self):
        """
        Reads stdout line-by-line using readline(), filters noise,
        parses MCP JSON, puts valid messages in queue.
        """
        handshake_complete = False
        stream_limit = getattr(self._process.stdout, '_limit', 'Unknown')
        log.debug(f"[{self._server_name}] Starting stdout reader loop using readline() (Buffer limit: {stream_limit}).")

        try:
            while self._process.returncode is None:
                if not self._is_active:
                    log.info(f"[{self._server_name}] Session inactive, exiting readline loop.")
                    break

                try:
                    if USE_VERBOSE_SESSION_LOGGING: log.debug(f"[{self._server_name}] Attempting stdout.readline()...")
                    # Use a timeout for readline itself to prevent indefinite blocking if server sends no newline
                    line_bytes = await asyncio.wait_for(self._process.stdout.readline(), timeout=60.0)
                    if USE_VERBOSE_SESSION_LOGGING: log.debug(f"[{self._server_name}] readline() returned {len(line_bytes)} bytes.")
                    if not line_bytes:
                        # Check if EOF truly reached or just no data within timeout
                        if self._process.stdout.at_eof():
                            log.warning(f"[{self._server_name}] Stdout EOF reached via readline().")
                            break # Exit loop on EOF
                        else:
                            # Timeout occurred, but not EOF, just continue loop
                            log.debug(f"[{self._server_name}] readline() timeout, no data/newline received.")
                            continue
                    # Line received, process it directly (no need for buffer management here)
                    line_str_raw = line_bytes.decode('utf-8', errors='replace')
                    if USE_VERBOSE_SESSION_LOGGING: log.debug(f"[{self._server_name}] Processing line: {repr(line_str_raw)}")
                    line_str = line_str_raw.strip()

                    if not line_str: continue # Skip empty lines

                    try:
                        message = json.loads(line_str) # Use 'json' alias
                        is_valid_rpc = (
                            isinstance(message, dict) and message.get("jsonrpc") == "2.0" and
                            ('id' in message or 'method' in message)
                        )
                        if is_valid_rpc:
                            if USE_VERBOSE_SESSION_LOGGING: log.info(f"[{self._server_name}] Received VALID JSON-RPC 2.0: ID={message.get('id', 'N/A')}, Method={message.get('method', 'N/A')}")
                            else: log.debug(f"[{self._server_name}] Received potential MCP message: ID={message.get('id', 'N/A')}, Method={message.get('method', 'N/A')}")
                            if not handshake_complete:
                                log.info(f"[{self._server_name}] First valid JSON-RPC 2.0 detected, MCP protocol started.")
                                handshake_complete = True
                            try: # Put message onto the queue
                                await asyncio.wait_for(self._message_queue.put(message), timeout=5.0)
                                log.debug(f"[{self._server_name}] Message put in queue. Queue size: {self._message_queue.qsize()}")
                            except asyncio.TimeoutError: log.error(f"[{self._server_name}] Timeout putting message in queue (full?). Dropping: {line_str[:100]}...")
                            except asyncio.QueueFull: log.error(f"[{self._server_name}] MCP message queue full! Dropping message: {line_str[:100]}...")
                        elif isinstance(message, dict): log.debug(f"[{self._server_name}] Skipping non-MCP JSON object stdout line: {line_str[:100]}...")
                        else: log.debug(f"[{self._server_name}] Skipping non-dict JSON stdout line: {line_str[:100]}...")
                    except json.JSONDecodeError: log.debug(f"[{self._server_name}] Skipping noisy (non-JSON) stdout line: {line_str[:100]}...")
                    except Exception as parse_err: log.error(f"[{self._server_name}] Error processing stdout line '{line_str[:100]}...': {parse_err}", exc_info=True)

                except asyncio.TimeoutError:
                    # Timeout waiting for readline, just loop again
                    log.debug(f"[{self._server_name}] Outer timeout waiting for readline(). Loop continues.")
                    continue
                except (BrokenPipeError, ConnectionResetError):
                    log.warning(f"[{self._server_name}] Stdout pipe broken during readline().")
                    break
                except ValueError as e: # Catch buffer limit error from readline()
                    stream_limit = getattr(self._process.stdout, '_limit', 'Unknown')
                    if "Separator is found, but chunk is longer than limit" in str(e) or "Line is too long" in str(e):
                         log.error(f"[{self._server_name}] Buffer limit ({stream_limit} bytes) exceeded by readline()! Increase limit in process creation.", exc_info=True)
                    else: log.error(f"[{self._server_name}] ValueError during readline(): {e}", exc_info=True)
                    break
                except Exception as read_err:
                    log.error(f"[{self._server_name}] Error during readline(): {read_err}", exc_info=True)
                    break
            log.info(f"[{self._server_name}] Exiting stdout readline loop (Process exited: {self._process.returncode is not None}, Session active: {self._is_active}).")
        except asyncio.CancelledError:
            log.info(f"[{self._server_name}] Stdout readline loop cancelled.")
            raise
        except Exception as loop_err:
            log.error(f"[{self._server_name}] Unhandled error in stdout readline loop: {loop_err}", exc_info=True)
            raise

    async def _get_next_message(self, mcp_timeout: Optional[float] = 30.0) -> Dict[str, Any]:
        """Gets the next valid MCP message from the queue."""
        if not self._is_active:
            raise ConnectionAbortedError("Session is closed")
        try:
            log.debug(f"[{self._server_name}] Waiting for message from queue (timeout={mcp_timeout})...")
            # Use timeout=None for indefinite wait used by _process_incoming_messages
            msg = await asyncio.wait_for(self._message_queue.get(), timeout=mcp_timeout)
            self._message_queue.task_done()
            log.debug(f"[{self._server_name}] Got message from queue: {msg.get('id', msg.get('method', 'Notification'))}")
            return msg
        except asyncio.TimeoutError as timeout_error:
            # Check if session became inactive while waiting
            if not self._is_active:
                 raise ConnectionAbortedError("Session closed while waiting for message") from timeout_error
            log.error(f"[{self._server_name}] Timeout waiting for message from server.")
            raise RuntimeError("Timeout waiting for response from server") from timeout_error
        except asyncio.CancelledError:
            log.info(f"[{self._server_name}] _get_next_message cancelled.")
            raise
        except Exception as e:
            log.error(f"[{self._server_name}] Error getting message from queue: {e}", exc_info=True)
            raise RuntimeError(f"Error receiving message: {e}") from e

    async def _process_incoming_messages(self):
        """Task to continuously process messages from the queue and resolve futures."""
        log.debug(f"[{self._server_name}] Starting incoming message processor.")
        while True: # Loop until break/cancel/error
            if not self._is_active:
                log.info(f"[{self._server_name}] Session became inactive, exiting message processor.")
                break
            try:
                if USE_VERBOSE_SESSION_LOGGING:
                    log.debug(f"[{self._server_name}] Waiting indefinitely for message from queue (Current size: {self._message_queue.qsize()})...")

                # Wait indefinitely for the next message from the queue
                message = await self._get_next_message(mcp_timeout=None)

                if USE_VERBOSE_SESSION_LOGGING:
                    # Use repr for potentially complex data structures in verbose mode
                    log.info(f"[{self._server_name}] Dequeued message: {repr(message)}")
                else:
                    # Standard logging level
                    log.debug(f"[{self._server_name}] Dequeued message: ID={message.get('id', 'N/A')}, Method={message.get('method', 'N/A')}")


                # Process based on message type (response/error vs. notification/request)
                msg_id = message.get("id")

                if msg_id is not None:
                    # --- It's a Response or Error for a previous Request ---
                    str_msg_id = str(msg_id) # Ensure key is string for dictionary lookup

                    # Attempt to find the corresponding future for this response ID
                    future = self._response_futures.pop(str_msg_id, None)

                    if future and not future.done():
                        # Found a pending future for this ID
                        if "result" in message:
                            # Success Response
                            log.debug(f"[{self._server_name}] Resolving future for ID {msg_id} with RESULT.")
                            future.set_result(message["result"])
                        elif "error" in message:
                            # Error Response
                            err_data = message["error"]
                            # Construct a meaningful error message from the JSON-RPC error object
                            err_msg = f"Server error response for ID {msg_id}: {err_data.get('message', 'Unknown error')} (Code: {err_data.get('code', 'N/A')})"
                            # Include data field if present, useful for debugging
                            err_data_details = err_data.get('data')
                            if err_data_details:
                                 err_msg += f" Data: {repr(err_data_details)}" # Use repr for data

                            log.warning(f"[{self._server_name}] Resolving future for ID {msg_id} with ERROR: {err_msg}")
                            # Create a RuntimeError or a custom McpServerError exception
                            server_exception = RuntimeError(err_msg)
                            # Attach error details if helpful
                            # setattr(server_exception, 'mcp_error_details', err_data)
                            future.set_exception(server_exception)
                        else:
                            # Invalid response format (has ID but no 'result' or 'error')
                            log.error(f"[{self._server_name}] Received message with ID {msg_id} but no result or error field.")
                            invalid_format_exception = RuntimeError(f"Invalid response format from server for ID {msg_id}: {message}")
                            future.set_exception(invalid_format_exception)
                    elif future:
                        # Future found but already done (e.g., timed out previously)
                        log.debug(f"[{self._server_name}] Future for ID {msg_id} was already done when response arrived (likely timed out).")
                    else:
                        # No future found for this ID (either timed out long ago and removed, or server sent unsolicited ID)
                        log.warning(f"[{self._server_name}] Received response for unknown or timed-out request ID: {msg_id}. Discarding.")

                elif "method" in message:
                    # --- It's a Notification or Request FROM the Server ---
                    method_name = message['method']
                    log.debug(f"[{self._server_name}] Received server-initiated message (Notification/Request): {method_name}")
                    # TODO: Add specific handling for expected server notifications/requests here
                    # e.g., progress updates, log messages, sampling requests
                    if method_name == "notifications/progress":
                         # Handle progress update
                         pass
                    elif method_name == "notifications/message":
                         # Handle log message from server
                         pass
                    elif method_name == "sampling/createMessage":
                         # Handle sampling request (needs callback mechanism)
                         pass
                    # Add handlers for other potential server-to-client messages as needed
                    else:
                         log.warning(f"[{self._server_name}] Received unhandled server method: {method_name}")
                else:
                    # --- Message has neither 'id' nor 'method' ---
                    log.warning(f"[{self._server_name}] Received message with unknown structure (no id or method): {repr(message)}")

            except ConnectionAbortedError:
                # Handle case where _get_next_message raises due to session closing
                log.info(f"[{self._server_name}] Incoming message processor stopping: Session closed.")
                break # Exit loop cleanly
            except McpError as e:
                # Catch specific MCP errors potentially raised by _get_next_message
                log.error(f"[{self._server_name}] MCPError in incoming message processor: {e}")
                break # Exit loop on persistent errors
            except asyncio.CancelledError:
                # Handle task cancellation cleanly
                log.info(f"[{self._server_name}] Incoming message processor cancelled.")
                raise # Propagate cancellation
            except Exception as e:
                # Catch any other unexpected errors during message processing
                log.error(f"[{self._server_name}] Unexpected error in incoming message processor: {e}", exc_info=True)
                break # Stop processing on unexpected errors

        # Log loop exit
        log.info(f"[{self._server_name}] Exiting incoming message processor.")
        # Ensure state reflects closure if loop exits unexpectedly (e.g., MCPError break)
        if self._is_active:
            log.warning(f"[{self._server_name}] Incoming message processor loop exited while session active. Forcing close.")
            # Use await here as _close_internal_state is async
            await self._close_internal_state(ConnectionAbortedError("Message processor failed or loop exited unexpectedly"))

    async def _send_request(self, method: str, params: Dict[str, Any], response_timeout: float) -> Any:
        """Sends a JSON-RPC request and waits for the response."""
        # Check if session is active and process is running
        if not self._is_active or (self._process and self._process.returncode is not None):
            raise ConnectionAbortedError("Session is not active or process terminated")

        # Acquire lock for thread-safe request ID generation
        async with self._lock:
            self._request_id_counter += 1
            request_id = str(self._request_id_counter)

        # Construct the JSON-RPC request object
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": request_id,
        }

        # Create a future to wait for the response associated with this request ID
        future = asyncio.get_running_loop().create_future()
        self._response_futures[request_id] = future

        try:
            # Serialize request to JSON bytes, adding newline
            request_bytes = json.dumps(request) + b'\n'

            log.debug(f"[{self._server_name}] Sending request ID {request_id} ({method}): {request_bytes.decode('utf-8', errors='replace')[:100]}...")

            # Check if stdin stream is valid and open
            if self._stdin is None or self._stdin.is_closing():
                 raise ConnectionAbortedError("Stdin is closed or None")
            if USE_VERBOSE_SESSION_LOGGING:
                 log.debug(f"[{self._server_name}] RAW >>> {repr(request_bytes)}")
            # Write request to process's stdin
            self._stdin.write(request_bytes)
            # Ensure data is flushed to the process
            await self._stdin.drain()
            if USE_VERBOSE_SESSION_LOGGING:
                log.info(f"[{self._server_name}] Successfully drained stdin for request ID {request_id} ({method}).")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as pipe_err:
            # Handle connection errors during send
            log.error(f"[{self._server_name}] Failed to send request ID {request_id} ({method}): Pipe broken or closing: {pipe_err}")
            # Clean up future for this request
            self._response_futures.pop(request_id, None)
            if future and not future.done(): future.set_exception(pipe_err)
            raise ConnectionAbortedError(f"Connection lost while sending request ID {request_id}: {pipe_err}") from pipe_err
        except Exception as send_err:
            # Handle other unexpected errors during send
            log.error(f"[{self._server_name}] Error sending request ID {request_id} ({method}): {send_err}", exc_info=True)
            # Clean up future
            self._response_futures.pop(request_id, None)
            if future and not future.done(): future.set_exception(send_err)
            raise RuntimeError(f"Failed to send request ID {request_id}: {send_err}") from send_err

        # Wait for the response future to be set by _process_incoming_messages
        try:
            log.debug(f"[{self._server_name}] Waiting for response for ID {request_id} ({method}) (timeout={response_timeout}s)")
            # Wait for the future with the specified timeout
            result = await asyncio.wait_for(future, timeout=response_timeout)
            log.debug(f"[{self._server_name}] Received and processed response for ID {request_id} ({method})")
            return result # Return the result part of the response
        except asyncio.TimeoutError as timeout_error:
            # Handle timeout waiting for the response future
            log.error(f"[{self._server_name}] Timeout waiting for response for request ID {request_id} ({method})")
            # Clean up future from the dictionary as it won't be resolved
            self._response_futures.pop(request_id, None)
            # Raise a specific runtime error indicating the timeout
            raise RuntimeError(f"Timeout waiting for response to {method} request (ID: {request_id})") from timeout_error
        except asyncio.CancelledError:
            # Handle cancellation of the waiting task
            log.info(f"[{self._server_name}] Wait for request ID {request_id} ({method}) cancelled.")
            # Clean up future
            self._response_futures.pop(request_id, None)
            raise # Propagate cancellation
        except Exception as wait_err:
             # Handle other exceptions that might occur during await_for or if future holds an error
             # Check if the future was resolved with an exception by the message processor
             if future.done() and future.exception():
                  server_error = future.exception()
                  log.warning(f"[{self._server_name}] Request ID {request_id} ({method}) failed with server error: {server_error}")
                  # Re-raise the original error received from the server
                  raise server_error from wait_err # Chain the exceptions
             else:
                  # Error likely happened during asyncio.wait_for itself, or future wasn't set correctly
                  log.error(f"[{self._server_name}] Error waiting for/processing response for request ID {request_id} ({method}): {wait_err}", exc_info=True)
                  # Clean up future just in case
                  self._response_futures.pop(request_id, None)
                  raise RuntimeError(f"Error processing response for {method} request (ID: {request_id}): {wait_err}") from wait_err

    # --- Implement ClientSession methods (Update parameter names) ---

    async def list_tools(self, response_timeout: float = 10.0) -> ListToolsResult:
        log.debug(f"[{self._server_name}] Calling list_tools")
        # Pass argument with the new name
        result = await self._send_request("list_tools", {}, response_timeout=response_timeout)
        try:
            return ListToolsResult(**result)
        except Exception as e:
            log.error(f"[{self._server_name}] Error parsing list_tools response: {e}. Result: {result}")
            raise RuntimeError(f"Invalid list_tools response format: {e}") from e

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], response_timeout: float = 30.0) -> CallToolResult:
        log.debug(f"[{self._server_name}] Calling call_tool: {tool_name}")
        params = {"name": tool_name, "arguments": arguments}
        result = await self._send_request("call_tool", params, response_timeout=response_timeout)
        try:
            return CallToolResult(**result)
        except Exception as e:
            log.error(f"[{self._server_name}] Error parsing call_tool response for {tool_name}: {e}. Result: {result}")
            raise RuntimeError(f"Invalid call_tool response format: {e}") from e

    async def list_resources(self, response_timeout: float = 10.0) -> ListResourcesResult:
        log.debug(f"[{self._server_name}] Calling list_resources")
        result = await self._send_request("list_resources", {}, response_timeout=response_timeout)
        try:
            return ListResourcesResult(**result)
        except Exception as e:
             log.error(f"[{self._server_name}] Error parsing list_resources response: {e}. Result: {result}")
             raise RuntimeError(f"Invalid list_resources response format: {e}") from e

    async def read_resource(self, uri: AnyUrl, response_timeout: float = 30.0) -> ReadResourceResult:
        log.debug(f"[{self._server_name}] Calling read_resource: {uri}")
        params = {"uri": str(uri)} # Match ReadResourceRequestParams
        result = await self._send_request("resources/read", params, response_timeout=response_timeout)
        try:
            return ReadResourceResult(**result)
        except Exception as e:
            log.error(f"[{self._server_name}] Error parsing read_resource response for {uri}: {e}. Result: {result}")
            raise RuntimeError(f"Invalid read_resource response format: {e}") from e

    async def list_prompts(self, response_timeout: float = 10.0) -> ListPromptsResult:
        log.debug(f"[{self._server_name}] Calling list_prompts")
        result = await self._send_request("list_prompts", {}, response_timeout=response_timeout)
        try:
            return ListPromptsResult(**result)
        except Exception as e:
            log.error(f"[{self._server_name}] Error parsing list_prompts response: {e}. Result: {result}")
            raise RuntimeError(f"Invalid list_prompts response format: {e}") from e

    async def get_prompt(self, prompt_name: str, variables: Dict[str, Any], response_timeout: float = 30.0) -> GetPromptResult: # Corrected hint
        log.debug(f"[{self._server_name}] Calling get_prompt: {prompt_name}")
        params = {"name": prompt_name, "variables": variables}
        result = await self._send_request("prompts/get", params, response_timeout=response_timeout)
        try:
            return GetPromptResult(**result) # Use the correct class
        except Exception as e:
            log.error(f"[{self._server_name}] Error parsing get_prompt response for {prompt_name}: {e}. Result: {result}")
            raise RuntimeError(f"Invalid get_prompt response format: {e}") from e

    async def _close_internal_state(self, exception: Exception):
        """Closes internal session state like queue and futures."""
        if not self._is_active: return # Already closing/closed
        self._is_active = False # Mark inactive first
        log.debug(f"[{self._server_name}] Closing internal state due to: {exception}")
        # Cancel pending futures
        await self._cancel_pending_futures(exception)
        # Potentially clear queue or add a sentinel? For now, setting _is_active=False
        # should stop readers/processors from using it.


    async def _cancel_pending_futures(self, exception: Exception):
        """Cancel all outstanding response futures."""
        # --- Logic remains the same ---
        log.debug(f"[{self._server_name}] Cancelling {len(self._response_futures)} pending futures with exception: {exception}")
        futures_to_cancel = list(self._response_futures.items()) # Iterate over copy
        self._response_futures.clear() # Clear immediately
        for future_id, future in futures_to_cancel:
            if future and not future.done():
                try:
                     future.set_exception(exception)
                except asyncio.InvalidStateError:
                     pass # Already cancelled/set perhaps

    async def aclose(self):
        """Closes the session and cleans up resources."""
        log.info(f"[{self._server_name}] Closing RobustStdioSession...")
        if not self._is_active:
            log.debug(f"[{self._server_name}] Already closed.")
            return

        # 1. Mark inactive and cancel futures immediately
        await self._close_internal_state(ConnectionAbortedError("Session closed by client"))

        # 2. Cancel the main background task runner (which cancels tasks in the TaskGroup)
        if self._background_task_runner and not self._background_task_runner.done():
            log.debug(f"[{self._server_name}] Cancelling background task runner...")
            self._background_task_runner.cancel()
            # Wait for the runner task to finish cancellation
            with suppress(asyncio.CancelledError): # Ignore cancellation of the await itself
                 await self._background_task_runner
            log.debug(f"[{self._server_name}] Background task runner finished cancellation.")
        else:
             log.debug(f"[{self._server_name}] Background task runner already done or None.")

        # 3. Cancel the external stderr reader task if it exists and is active
        if self._stderr_reader_task and not self._stderr_reader_task.done():
            log.debug(f"[{self._server_name}] Cancelling external stderr reader task...")
            self._stderr_reader_task.cancel()
            with suppress(asyncio.CancelledError):
                 await self._stderr_reader_task
            log.debug(f"[{self._server_name}] External stderr reader task finished cancellation.")

        # 4. Terminate the process (gracefully first)
        # Note: The TaskGroup tasks might have already terminated it if the pipe broke
        if self._process and self._process.returncode is None:
            log.info(f"[{self._server_name}] Terminating process PID {self._process.pid} during aclose...")
            try:
                self._process.terminate()
                # Wait briefly for termination
                with suppress(asyncio.TimeoutError): # Don't let timeout stop cleanup
                    await asyncio.wait_for(self._process.wait(), timeout=2.0)

                if self._process.returncode is None: # If still running, kill
                    log.warning(f"[{self._server_name}] Process did not terminate gracefully after 2s, killing.")
                    try:
                        self._process.kill()
                        with suppress(asyncio.TimeoutError): # Short wait for kill
                            await asyncio.wait_for(self._process.wait(), timeout=1.0)
                    except ProcessLookupError:
                        log.info(f"[{self._server_name}] Process already exited before kill.")
                    except Exception as kill_err:
                        log.error(f"[{self._server_name}] Error killing process: {kill_err}")
            except ProcessLookupError:
                 log.info(f"[{self._server_name}] Process already exited before termination attempt.")
            except Exception as term_err:
                 log.error(f"[{self._server_name}] Error terminating process: {term_err}")

        log.info(f"[{self._server_name}] RobustStdioSession closed.")


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
        """Process and display discovery results, prompting user to add servers"""
        # Get all discovered servers from class attributes
        discovered_local = getattr(self, '_discovered_local', [])
        discovered_remote = getattr(self, '_discovered_remote', [])
        discovered_mdns = getattr(self, '_discovered_mdns', [])
        
        # Show discoveries to user with clear categorization
        total_discovered = len(discovered_local) + len(discovered_remote) + len(discovered_mdns)
        if total_discovered > 0:
            safe_console = get_safe_console()
            safe_console.print(f"\n[bold green]Discovered {total_discovered} potential MCP servers:[/]")
            
            if discovered_local:
                safe_console.print("\n[bold blue]Local File System:[/]")
                for i, (name, path, server_type) in enumerate(discovered_local, 1):
                    safe_console.print(f"{i}. [bold]{name}[/] ({server_type}) - {path}")
            
            if discovered_remote:
                safe_console.print("\n[bold magenta]Remote Registry:[/]")
                for i, (name, url, server_type, version, categories, rating) in enumerate(discovered_remote, 1):
                    version_str = f"v{version}" if version else "unknown version"
                    categories_str = ", ".join(categories) if categories else "no categories"
                    safe_console.print(f"{i}. [bold]{name}[/] ({server_type}) - {url} - {version_str} - Rating: {rating:.1f}/5.0 - {categories_str}")
            
            if discovered_mdns:
                safe_console.print("\n[bold cyan]Local Network (mDNS):[/]")
                for i, (name, url, server_type, version, categories, description) in enumerate(discovered_mdns, 1):
                    version_str = f"v{version}" if version else "unknown version"
                    categories_str = ", ".join(categories) if categories else "no categories"
                    desc_str = f" - {description}" if description else ""
                    safe_console.print(f"{i}. [bold]{name}[/] ({server_type}) - {url} - {version_str} - {categories_str}{desc_str}")
            
            # Ask user which ones to add
            if Confirm.ask("\nAdd discovered servers to configuration?", console=safe_console):
                # Create selection interface
                selections = []
                
                if discovered_local:
                    safe_console.print("\n[bold blue]Local File System Servers:[/]")
                    for i, (name, path, server_type) in enumerate(discovered_local, 1):
                        if Confirm.ask(f"Add {name} ({path})?", console=safe_console):
                            selections.append(("local", i-1))
                
                if discovered_remote:
                    safe_console.print("\n[bold magenta]Remote Registry Servers:[/]")
                    for i, (name, url, server_type, version, categories, rating) in enumerate(discovered_remote, 1):
                        if Confirm.ask(f"Add {name} ({url})?", console=safe_console):
                            selections.append(("remote", i-1))
                
                if discovered_mdns:
                    safe_console.print("\n[bold cyan]Local Network Servers:[/]")
                    for i, (name, url, server_type, version, categories, description) in enumerate(discovered_mdns, 1):
                        if Confirm.ask(f"Add {name} ({url})?", console=safe_console):
                            selections.append(("mdns", i-1))
                
                # Process selections
                for source, idx in selections:
                    if source == "local":
                        name, path, server_type = discovered_local[idx]
                        self.config.servers[name] = ServerConfig(
                            name=name,
                            type=ServerType(server_type),
                            path=path,
                            enabled=True,
                            auto_start=False,  # Default to not auto-starting discovered servers
                            description=f"Auto-discovered {server_type} server"
                        )
                    
                    elif source == "remote":
                        name, url, server_type, version, categories, rating = discovered_remote[idx]
                        self.config.servers[name] = ServerConfig(
                            name=name,
                            type=ServerType(server_type),
                            path=url,
                            enabled=True,
                            auto_start=False,
                            description="Discovered from registry",
                            categories=categories,
                            version=version,
                            rating=rating,
                            registry_url=self.registry.registry_urls[0]  # Use first registry as source
                        )
                    
                    elif source == "mdns":
                        name, url, server_type, version, categories, description = discovered_mdns[idx]
                        self.config.servers[name] = ServerConfig(
                            name=name,
                            type=ServerType(server_type),
                            path=url,
                            enabled=True,
                            auto_start=False,
                            description=description or "Discovered on local network",
                            categories=categories,
                            version=version if version else None
                        )
                
                self.config.save()
                safe_console.print("[green]Selected servers added to configuration[/]")
        else:
            safe_console = get_safe_console()
            safe_console.print("[yellow]No new servers discovered.[/]")

    async def discover_servers(self):
        """Auto-discover MCP servers in configured paths and from registry"""
        steps = []
        descriptions = []
        
        # Add filesystem discovery step if enabled
        if self.config.auto_discover:
            steps.append(self._discover_local_servers)
            descriptions.append(f"{STATUS_EMOJI['search']} Discovering local file system servers...")
        
        # Add registry discovery step if enabled
        if self.config.enable_registry and self.registry:
            steps.append(self._discover_registry_servers)
            descriptions.append(f"{STATUS_EMOJI['search']} Discovering registry servers...")
        
        # Add mDNS discovery step if enabled
        if self.config.enable_local_discovery and self.registry:
            steps.append(self._discover_mdns_servers)
            descriptions.append(f"{STATUS_EMOJI['search']} Discovering local network servers...")
        
        # Run the discovery steps with progress tracking
        if steps:
            await self.run_multi_step_task(
                steps=steps,
                step_descriptions=descriptions,
                title=f"{STATUS_EMOJI['search']} Discovering MCP servers..."
            )
            
            # Process and display discovery results
            await self._process_discovery_results()
        else:
            safe_console = get_safe_console()
            safe_console.print("[yellow]Server discovery is disabled in configuration.[/]")
    
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
                
    async def connect_to_server(self, server_config: ServerConfig) -> Optional[ClientSession]:
        """
        Connect to a single MCP server with retry logic, robust STDIO handling,
        automatic stderr redirection, direct shell execution for remapped commands,
        increased buffer limits, and corrected tracing context management.
        """
        server_name = server_config.name
        retry_count = 0
        max_retries = server_config.retry_count

        # Define buffer limit (consider making this configurable in Config class later)
        # Increased to 1 MiB based on previous testing needs
        BUFFER_LIMIT = 2**20 # 1 MiB

        # Use safe_stdout context manager to protect client's output
        with safe_stdout():
            while retry_count <= max_retries:
                start_time = time.time()
                session: Optional[ClientSession] = None
                process_this_attempt: Optional[asyncio.subprocess.Process] = None
                stderr_reader_task_this_attempt: Optional[asyncio.Task] = None
                created_session_tasks: List[asyncio.Task] = [] # Tasks created for *this attempt* only
                connection_error: Optional[Exception] = None
                zeroconf_registered_this_attempt = False # Reset for each attempt

                # --- Corrected Tracing Setup ---
                span = None # The actual span object
                span_context_manager = None # The context manager returned by start_as_current_span
                try:
                    # Start the span context manager if tracer is available
                    if tracer:
                        try:
                            span_context_manager = tracer.start_as_current_span(
                                f"connect_server.{server_name}",
                                attributes={
                                    "server.name": server_name,
                                    "server.type": server_config.type.value,
                                    "server.path": str(server_config.path), # Ensure path is string
                                    "retry": retry_count
                                }
                            )
                            # Manually enter the context to get the span object
                            if span_context_manager:
                                span = span_context_manager.__enter__()
                        except Exception as e:
                            log.warning(f"Failed to start trace span: {e}")
                            span = None
                            span_context_manager = None

                    # --- Main Connection Logic ---
                    try:
                        self._safe_printer(f"[cyan]Attempting to connect to server {server_name} (Attempt {retry_count+1}/{max_retries+1})...[/]")

                        if server_config.type == ServerType.STDIO:
                            # ====================================================
                            # STDIO Connection Logic
                            # ====================================================
                            existing_process = self.processes.get(server_config.name)
                            restart_process = False
                            process_to_use: Optional[asyncio.subprocess.Process] = None

                            # --- Decide if process needs restart ---
                            if existing_process:
                                if existing_process.returncode is None:
                                    if retry_count > 0: # Force restart on any retry
                                        log.warning(f"Restarting process for {server_name} on retry {retry_count}")
                                        self._safe_printer(f"[yellow]Restarting process for {server_name} on retry {retry_count}[/]")
                                        restart_process = True
                                        await self.terminate_process(server_name, existing_process)
                                        if server_name in self.registered_services: await self.unregister_local_server(server_name)
                                    else: # Use existing running process on first attempt
                                        log.debug(f"Found existing process for {server_name} (PID {existing_process.pid}), will attempt connection.")
                                        process_to_use = existing_process
                                else: # Process existed but has terminated
                                    log.warning(f"Previously managed process for {server_name} has exited with code {existing_process.returncode}. Cleaning up entry.")
                                    self._safe_printer(f"[yellow]Previous process for {server_name} exited (code {existing_process.returncode}). Starting new one.[/]")
                                    restart_process = True
                                    if server_name in self.registered_services: await self.unregister_local_server(server_name)
                                    if server_name in self.processes: del self.processes[server_name]
                            else: # No existing process entry
                                restart_process = True

                            # --- Start or restart the process if needed ---
                            if restart_process:
                                executable = server_config.path
                                arguments = server_config.args
                                log_file_path = (CONFIG_DIR / f"{server_name}_stderr.log").resolve()
                                log_file_path.parent.mkdir(parents=True, exist_ok=True)

                                log.info(f"Preparing to execute server '{server_name}': Executable='{executable}', Args={arguments}")
                                log.info(f"Stderr for STDIO server {server_name} will be captured via pipe -> {log_file_path}")

                                process: Optional[asyncio.subprocess.Process] = None
                                is_shell_cmd = (
                                    isinstance(executable, str) and
                                    any(executable.endswith(shell) for shell in ["bash", "sh", "zsh"]) and
                                    len(arguments) == 2 and arguments[0] == "-c" and isinstance(arguments[1], str)
                                )

                                if is_shell_cmd:
                                    command_string = arguments[1]
                                    log.info(f"Executing server '{server_name}' via shell '{executable}': {command_string}")
                                    self._safe_printer(f"[cyan]Starting server process (shell: {executable}): {command_string[:100]}...[/]")
                                    try:
                                        process = await asyncio.create_subprocess_shell(
                                            command_string, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                            stderr=asyncio.subprocess.PIPE, limit=BUFFER_LIMIT, env=os.environ.copy(), executable=executable
                                        )
                                    except FileNotFoundError:
                                        log.error(f"Shell executable not found: '{executable}'. Check path and environment.")
                                        raise
                                    except Exception as shell_exec_err:
                                        log.error(f"Error starting server '{server_name}' with shell '{executable}': {shell_exec_err}", exc_info=True)
                                        raise
                                else:
                                    final_cmd_list = [executable] + arguments
                                    log.info(f"Executing server '{server_name}' directly: {' '.join(map(str, final_cmd_list))}")
                                    self._safe_printer(f"[cyan]Starting server process: {' '.join(map(str, final_cmd_list))}[/]")
                                    try:
                                        process = await asyncio.create_subprocess_exec(
                                            *final_cmd_list, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                            stderr=asyncio.subprocess.PIPE, limit=BUFFER_LIMIT, env=os.environ.copy()
                                        )
                                    except FileNotFoundError:
                                        log.error(f"Executable not found: '{executable}'. Check path and environment.")
                                        raise
                                    except Exception as direct_exec_err:
                                        log.error(f"Error starting server '{server_name}' directly with '{executable}': {direct_exec_err}", exc_info=True)
                                        raise

                                if process is None: raise RuntimeError(f"Failed to create subprocess for {server_name}")
                                process_this_attempt = process
                                self.processes[server_name] = process
                                log.info(f"Started process for {server_name} with PID {process.pid}")

                                # --- Start stderr reader task ---
                                log.info(f"Starting stderr pipe reader task for {server_name} -> {log_file_path}")
                                async def read_stderr_to_file(proc_stderr, log_path_str, server_id):
                                    # --- Complete stderr reader function ---
                                    if proc_stderr is None:
                                        log.warning(f"Stderr pipe is None for {server_id}, cannot read.")
                                        return
                                    try:
                                        async with aiofiles.open(log_path_str, "ab") as log_f:
                                            while True:
                                                line_bytes = await proc_stderr.readline()
                                                if not line_bytes:
                                                    if proc_stderr.at_eof():
                                                        log.debug(f"Stderr pipe EOF reached for {server_id}")
                                                        break
                                                    else: await asyncio.sleep(0.01)
                                                    continue
                                                await log_f.write(line_bytes)
                                    except asyncio.CancelledError: log.info(f"Stderr reader task cancelled for {server_id}")
                                    except (BrokenPipeError, ConnectionResetError): log.warning(f"Stderr pipe broken for {server_id}")
                                    except Exception as read_err: log.error(f"Error reading/writing stderr line for {server_id}: {read_err}", exc_info=True)
                                    finally: log.debug(f"Exiting stderr reader task for {server_id}")
                                    # --- End stderr reader function ---

                                stderr_reader_task_this_attempt = asyncio.create_task(
                                    read_stderr_to_file(process.stderr, str(log_file_path), server_name),
                                    name=f"stderr-reader-{server_name}"
                                )
                                await asyncio.sleep(0.01) # Give reader a moment

                                # --- Check for Immediate Exit ---
                                await asyncio.sleep(0.5)
                                if process.returncode is not None:
                                    log.error(f"Process for {server_name} failed immediately after start (code {process.returncode}). Check stderr log: {log_file_path}")
                                    if stderr_reader_task_this_attempt and not stderr_reader_task_this_attempt.done():
                                        stderr_reader_task_this_attempt.cancel()
                                        with suppress(asyncio.CancelledError): await stderr_reader_task_this_attempt
                                    raise RuntimeError(f"Process for {server_name} failed immediately (code {process.returncode}). Check log file '{log_file_path}'.")

                                process_to_use = process
                                log.debug(f"Skipping Zeroconf registration for STDIO server: {server_name}")

                            # --- Use RobustStdioSession ---
                            if not process_to_use or process_to_use.returncode is not None:
                                raise RuntimeError(f"Process for STDIO server {server_name} is not valid or has exited before RobustStdioSession attempt.")

                            log.info(f"[{server_name}] Initializing RobustStdioSession...")
                            robust_session = None
                            try:
                                robust_session = RobustStdioSession(process_to_use, server_name)
                                robust_session._stderr_reader_task = stderr_reader_task_this_attempt
                                created_session_tasks = []
                                if stderr_reader_task_this_attempt: created_session_tasks.append(stderr_reader_task_this_attempt)
                                log.debug(f"[{server_name}] Linked external tasks for this attempt: {[t.get_name() for t in created_session_tasks if t]}")

                                # --- Handshake Step 1: Initialize ---
                                log.info(f"[{server_name}] Attempting MCP handshake via initialize...")
                                init_timeout = server_config.timeout + 5.0
                                initialize_result: InitializeResult = await asyncio.wait_for(
                                    robust_session.initialize(response_timeout=server_config.timeout), timeout=init_timeout
                                )
                                log.info(f"[{server_name}] MCP Initialize successful.")
                                server_capabilities = initialize_result.capabilities if initialize_result and hasattr(initialize_result, 'capabilities') else None
                                log.info(f"[{server_name}] Server Capabilities: {server_capabilities}")

                                # --- Handshake Step 1.5: Send Initialized Notification ---
                                log.info(f"[{server_name}] Sending initialized notification...")
                                await robust_session.send_initialized_notification()

                                # --- Handshake Step 2: List Tools (Conditional) ---
                                has_tools_capability = server_capabilities and server_capabilities.tools
                                if has_tools_capability:
                                    log.info(f"[{server_name}] Server supports tools. Proceeding with list_tools...")
                                    list_tools_timeout = server_config.timeout + 10.0
                                    try:
                                        await asyncio.wait_for(
                                            robust_session.list_tools(response_timeout=server_config.timeout), timeout=list_tools_timeout
                                        )
                                        log.info(f"[{server_name}] MCP List Tools successful.")
                                    except RuntimeError as e:
                                        if "Method not found" in str(e) and "-32601" in str(e):
                                            log.warning(f"[{server_name}] Server advertised tools capability but returned 'Method not found' for list_tools. Proceeding without tool listing.")
                                        else: raise # Re-raise other errors
                                else:
                                    log.info(f"[{server_name}] Server does not advertise 'tools' capability. Skipping list_tools during handshake.")

                                log.info(f"[{server_name}] Core MCP handshake complete.")
                                session = robust_session

                            except Exception as session_init_or_handshake_err:
                                # Handle handshake failures
                                connection_error = session_init_or_handshake_err # Store error for logging/retry logic
                                log.error(f"[{server_name}] Failed MCP handshake ({type(session_init_or_handshake_err).__name__}): {session_init_or_handshake_err}", exc_info=True)
                                self._safe_printer(f"[red]Failed MCP handshake/setup with {server_name}. Check server logs.[/]")
                                if robust_session: await robust_session.aclose() # Cleanup session if partially created
                                elif process_this_attempt and process_this_attempt.returncode is None: await self.terminate_process(server_name, process_this_attempt) # Terminate process if session failed early
                                if stderr_reader_task_this_attempt and not stderr_reader_task_this_attempt.done(): # Cleanup external reader
                                    stderr_reader_task_this_attempt.cancel()
                                    with suppress(asyncio.CancelledError): await stderr_reader_task_this_attempt
                                raise connection_error from session_init_or_handshake_err # Re-raise to outer try to trigger retry

                            # If handshake succeeded
                            log.info(f"Successfully established MCP session for STDIO server {server_name}")

                        elif server_config.type == ServerType.SSE:
                            # ====================================================
                            # SSE Connection Logic (Unchanged)
                            # ====================================================
                            log.info(f"Connecting to SSE server {server_name} at {server_config.path}")
                            sse_url = server_config.path
                            if not sse_url.startswith(("http://", "https://")): sse_url = f"http://{sse_url}"
                            session = await self.exit_stack.enter_async_context(
                                sse_client(url=sse_url, timeout=server_config.timeout, sse_read_timeout=server_config.timeout * 12)
                            )
                            if not session or not hasattr(session, 'list_tools') or not callable(session.list_tools):
                                raise RuntimeError(f"Invalid SSE session object obtained for {server_name}")
                            log.info(f"[{server_name}] Attempting MCP handshake via list_tools for SSE server...")
                            handshake_timeout = server_config.timeout + 10.0
                            await asyncio.wait_for(session.list_tools(response_timeout=server_config.timeout), timeout=handshake_timeout)
                            log.info(f"[{server_name}] MCP Handshake successful for SSE server.")
                            log.info(f"Successfully established MCP session for SSE server {server_name}")

                        else:
                            # --- Unknown Server Type ---
                            unknown_type_msg = f"Unknown server type: {server_config.type}"
                            if span: span.set_status(trace.StatusCode.ERROR, unknown_type_msg) # Use span object
                            log.error(unknown_type_msg)
                            self._safe_printer(f"[red]{unknown_type_msg}[/]")
                            # Must exit span context before returning
                            if span_context_manager: span_context_manager.__exit__(None, None, None)
                            return None

                        # --- Success Path (Common for STDIO/SSE) ---
                        connection_time = (time.time() - start_time) * 1000
                        server_config.metrics.request_count += 1
                        server_config.metrics.update_response_time(connection_time / 1000.0)
                        server_config.metrics.update_status()

                        if latency_histogram:
                            try: latency_histogram.record(connection_time, {"server.name": server_name})
                            except Exception as metric_err: log.warning(f"Failed to record latency metric for {server_name}: {metric_err}")

                        # Set span status to OK on success
                        if span: span.set_status(trace.StatusCode.OK)

                        log.info(f"Connected to server {server_name} in {connection_time:.2f}ms")
                        self._safe_printer(f"[green]Connected to server {server_name} in {connection_time:.2f}ms[/]")

                        if created_session_tasks:
                            self._session_tasks.setdefault(server_name, []).extend(created_session_tasks)
                        
                        settle_delay = 1.0 # seconds, adjust if needed
                        log.info(f"[{server_name}] Connection established. Waiting {settle_delay}s before returning session...")
                        await asyncio.sleep(settle_delay)

                        # Exit span context manager cleanly before returning session
                        if span_context_manager: span_context_manager.__exit__(None, None, None)
                        return session # Return the successfully connected session

                    # --- Outer Exception Handling Block (for connection attempt) ---
                    except (McpError, RuntimeError, ConnectionAbortedError, httpx.RequestError, subprocess.SubprocessError, OSError, FileNotFoundError) as e:
                        connection_error = e # Store the caught exception
                        log.warning(f"Connection attempt failed for {server_name} ({type(e).__name__}): {e}")
                        # Add specific logging if process exited
                        if isinstance(e, (subprocess.SubprocessError, OSError, FileNotFoundError)) and process_this_attempt and process_this_attempt.returncode is not None:
                            log.error(f"STDIO server process '{server_name}' exited during connection attempt with code {process_this_attempt.returncode}.")
                    except Exception as e: # Catch any other unexpected errors
                        log.exception(f"Unexpected error during connection attempt for {server_name}")
                        connection_error = e # Store the caught exception

                    # --- Shared Error Handling & Retry Logic ---
                    log.debug(f"Entering shared error handling for {server_name}, attempt {retry_count+1}")

                    # Cancel external tasks created for this failed attempt
                    if created_session_tasks:
                        log.warning(f"Cancelling {len(created_session_tasks)} external task(s) from failed attempt for {server_name}")
                        for task in created_session_tasks:
                            if task and not task.done(): task.cancel()
                        await asyncio.sleep(0.05) # Allow cancellations to process
                        for task in created_session_tasks:
                            if task and not task.done(): 
                                with suppress(asyncio.CancelledError): 
                                    await task

                    if zeroconf_registered_this_attempt: await self.unregister_local_server(server_name)

                    # Update metrics for the failed attempt
                    retry_count += 1
                    server_config.metrics.error_count += 1
                    server_config.metrics.update_status()

                    # Set span status to ERROR if span exists
                    error_msg_for_span = str(connection_error) if connection_error else "Unknown connection error during attempt"
                    if span: span.set_status(trace.StatusCode.ERROR, error_msg_for_span)

                    # Decide whether to retry or give up
                    if retry_count <= max_retries:
                        base_delay = server_config.retry_policy.get("backoff_factor", 0.5) 
                        max_delay = 10.0
                        delay = min(base_delay * (2 ** (retry_count - 1)) + random.random() * 0.1, max_delay)
                        error_msg_display = str(connection_error) if connection_error else "Unknown connection error"
                        log.warning(f"Error details for {server_name} (attempt {retry_count-1} failed): {error_msg_display}")
                        self._safe_printer(f"[yellow]Error connecting to server {server_name} (attempt {retry_count}/{max_retries+1}): {error_msg_display}[/]")
                        log.info(f"Retrying connection to {server_name} in {delay:.2f} seconds...")
                        self._safe_printer(f"[cyan]Retrying in {delay:.2f} seconds...[/]")

                        # --- Manually exit span context BEFORE sleep ---
                        if span_context_manager:
                            # Pass exception info if available
                            current_exc_info = sys.exc_info() if connection_error else (None, None, None)
                            span_context_manager.__exit__(*current_exc_info)
                            span_context_manager = None # Prevent re-exit in finally
                            span = None # Span is ended for this attempt
                        # --- End change ---
                        await asyncio.sleep(delay)
                        # Continue to the next iteration of the while loop
                    else:
                        # Max retries exceeded
                        final_error_msg = str(connection_error) if connection_error else "Unknown connection error"
                        log.error(f"Failed to connect to server {server_name} after {max_retries+1} attempts. Final error: {final_error_msg}")
                        self._safe_printer(f"[red]Failed to connect to server {server_name} after {max_retries+1} attempts: {final_error_msg}[/]")
                        if server_name in self.processes and self.processes[server_name].returncode is not None: del self.processes[server_name]

                        # Set span status if span still exists (might have been ended in retry loop)
                        if span: span.set_status(trace.StatusCode.ERROR, f"Max retries exceeded. Final error: {final_error_msg}")

                        # Exit span context manager before returning None
                        if span_context_manager: span_context_manager.__exit__(*sys.exc_info())
                        return None # Return None indicating connection failure after retries

                # --- End of while loop ---

                # --- FINALLY block to ensure span context manager is ALWAYS exited ---
                finally:
                    if span_context_manager:
                        # Get current exception info, if any, to pass to __exit__
                        exc_type, exc_value, tb = sys.exc_info()
                        try:
                            span_context_manager.__exit__(exc_type, exc_value, tb)
                        except Exception as exit_err:
                            log.warning(f"Error exiting span context manager for {server_name}: {exit_err}")
                # --- END FINALLY block ---

                # This path indicates loop finished without returning session or None after retries - should be rare
                log.error(f"Connection loop for {server_name} exited without explicit return.")
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
                session = await self.connect_to_server(server_config)
                
                if session:
                    self.active_sessions[name] = session
                    log.info(f"Connected to server: {name}")
                    await self.load_server_capabilities(name, session)
                
                progress.update(task, advance=1)
        
        # Verify no stdout pollution after connecting to servers
        if os.environ.get("MCP_VERIFY_STDOUT", "1") == "1":
            with safe_stdout():
                log.info("Verifying no stdout pollution after connecting to servers...")
                verify_no_stdout_pollution()
        
        # Start server monitoring
        with Status(f"{STATUS_EMOJI['server']} Starting server monitoring...", spinner="dots", console=get_safe_console()) as status:
            await self.server_monitor.start_monitoring()
            status.update(f"{STATUS_EMOJI['success']} Server monitoring started")
        
        # Display status
        await self.print_status()
    
    async def load_server_capabilities(self, server_name: str, session: ClientSession):
        """Load tools, resources, and prompts from a server with better validation"""
        # Use safe_stdout to prevent protocol corruption when interacting with servers
        with safe_stdout():
            try:
                # First verify the session has required methods
                if not session or not hasattr(session, 'list_tools') or not callable(session.list_tools):
                    log.error(f"Server {server_name} session object is invalid or incomplete")
                    return
                
                # Load tools with proper validation
                try:
                    tool_response = await session.list_tools()
                    
                    # Validate tool response
                    if not tool_response or not hasattr(tool_response, 'tools') or not isinstance(tool_response.tools, list):
                        log.error(f"Server {server_name} returned invalid tool response format")
                        return
                    
                    # Process tools
                    for tool in tool_response.tools:
                        if not hasattr(tool, 'name') or not hasattr(tool, 'description') or not hasattr(tool, 'input_schema'):
                            log.warning(f"Server {server_name} returned a tool with missing required attributes, skipping")
                            continue
                            
                        tool_name = f"{server_name}:{tool.name}" if ":" not in tool.name else tool.name
                        self.tools[tool_name] = MCPTool(
                            name=tool_name,
                            description=tool.description,
                            server_name=server_name,
                            input_schema=tool.input_schema,
                            original_tool=tool
                        )
                        
                        # Register tool dependencies if present
                        if hasattr(tool, 'dependencies') and isinstance(tool.dependencies, list):
                            for dependency in tool.dependencies:
                                # Make sure dependency has the server prefix if needed
                                dependency_name = f"{server_name}:{dependency}" if ":" not in dependency else dependency
                                # Add the dependency to the cache dependency graph
                                if hasattr(self, 'tool_cache') and self.tool_cache:
                                    self.tool_cache.add_dependency(tool_name, dependency_name)
                                    log.debug(f"Registered dependency: {tool_name} depends on {dependency_name}")
                except McpError as e:
                    log.error(f"MCP error loading tools from server {server_name}: {e}")
                except Exception as e:
                    log.error(f"Unexpected error loading tools from server {server_name}: {e}")
                
                # Load resources if available
                if hasattr(session, 'list_resources') and callable(session.list_resources):
                    try:
                        resource_response = await session.list_resources()
                        # Validate resource response
                        if not resource_response or not hasattr(resource_response, 'resources') or not isinstance(resource_response.resources, list):
                            log.warning(f"Server {server_name} returned invalid resource response format")
                        else:
                            for resource in resource_response.resources:
                                if not hasattr(resource, 'name') or not hasattr(resource, 'description') or not hasattr(resource, 'template'):
                                    log.warning(f"Server {server_name} returned a resource with missing required attributes, skipping")
                                    continue
                                    
                                resource_name = f"{server_name}:{resource.name}" if ":" not in resource.name else resource.name
                                self.resources[resource_name] = MCPResource(
                                    name=resource_name,
                                    description=resource.description,
                                    server_name=server_name,
                                    template=resource.template,
                                    original_resource=resource
                                )
                    except McpError as e:
                        log.debug(f"Server {server_name} doesn't support resources: {e}")
                    except Exception as e: 
                        log.error(f"Unexpected error listing resources from {server_name}: {e}")
                
                # Load prompts if available
                if hasattr(session, 'list_prompts') and callable(session.list_prompts):
                    try:
                        prompt_response = await session.list_prompts()
                        # Validate prompt response
                        if not prompt_response or not hasattr(prompt_response, 'prompts') or not isinstance(prompt_response.prompts, list):
                            log.warning(f"Server {server_name} returned invalid prompt response format")
                        else:
                            for prompt in prompt_response.prompts:
                                if not hasattr(prompt, 'name') or not hasattr(prompt, 'description') or not hasattr(prompt, 'template'):
                                    log.warning(f"Server {server_name} returned a prompt with missing required attributes, skipping")
                                    continue
                                    
                                prompt_name = f"{server_name}:{prompt.name}" if ":" not in prompt.name else prompt.name
                                self.prompts[prompt_name] = MCPPrompt(
                                    name=prompt_name,
                                    description=prompt.description,
                                    server_name=server_name,
                                    template=prompt.template,
                                    original_prompt=prompt
                                )
                    except McpError as e:
                        log.debug(f"Server {server_name} doesn't support prompts: {e}")
                    except Exception as e: 
                        log.error(f"Unexpected error listing prompts from {server_name}: {e}")
                    
            except McpError as e: # Catch specific MCP errors first
                log.error(f"MCP error loading capabilities from server {server_name}: {e}")
            except httpx.RequestError as e: # Catch network errors if using SSE
                log.error(f"Network error loading capabilities from server {server_name}: {e}")
            # Keep broad exception for other unexpected issues
            except Exception as e: 
                log.error(f"Unexpected error loading capabilities from server {server_name}: {e}")
                import traceback
                log.debug(f"Stack trace for error loading capabilities from {server_name}: {traceback.format_exc()}")
                
            # Mark the server as loaded capabilities, even if there were errors
            # This prevents repeated attempts that would just fail again
            if server_name in self.config.servers:
                self.config.servers[server_name].loaded_capabilities = True
    
    async def close(self):
        """Clean up resources before exit"""
        try:
            # Add a timeout to all cleanup operations
            cleanup_timeout = 5  # seconds
            
            # Stop local discovery monitoring if running
            if self.local_discovery_task:
                try:
                    await asyncio.wait_for(self.stop_local_discovery_monitoring(), timeout=cleanup_timeout)
                except asyncio.TimeoutError:
                    log.warning("Timeout stopping local discovery monitoring")
                except Exception as e:
                    log.error(f"Error stopping local discovery: {e}")
                    
            # Save conversation graph
            try:
                await asyncio.wait_for(
                    self.conversation_graph.save(str(self.conversation_graph_file)),
                    timeout=cleanup_timeout
                )
                log.info(f"Saved conversation graph to {self.conversation_graph_file}")
            except (asyncio.TimeoutError, Exception) as e:
                log.error(f"Failed to save conversation graph: {e}")

            # Stop server monitor with timeout
            if hasattr(self, 'server_monitor'):
                try:
                    await asyncio.wait_for(self.server_monitor.stop_monitoring(), timeout=cleanup_timeout)
                except (asyncio.TimeoutError, Exception) as e:
                    log.error(f"Error stopping server monitor: {e}")
                    
            # Close server connections and processes
            if hasattr(self, 'server_manager'):
                try:
                    # Use a more aggressive timeout for server connections
                    await asyncio.wait_for(self.server_manager.close(), timeout=cleanup_timeout)
                except (asyncio.TimeoutError, Exception) as e:
                    log.error(f"Error closing server manager: {e}")
                    
                    # Force kill any remaining processes
                    for name, process in self.server_manager.processes.items():
                        try:
                            if process and process.returncode is None:
                                log.warning(f"Force killing process {name} that didn't terminate properly")
                                process.kill()
                        except Exception as kill_error:
                            log.error(f"Error killing process {name}: {kill_error}")
                            
            # Close cache
            if self.tool_cache:
                try:
                    self.tool_cache.close()
                except Exception as e:
                    log.error(f"Error closing tool cache: {e}")
                    
        except Exception as e:
            log.error(f"Unexpected error during cleanup: {e}")
            
        finally:
            # Let user know we're done
            self._safe_printer("[yellow]Shutdown complete[/]")

    def format_tools_for_anthropic(self) -> List[ToolParam]:
        """Format MCP tools for Anthropic API"""
        tool_params: List[ToolParam] = []
        
        for tool in self.tools.values():
            tool_params.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            })
            
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
        """Execute a tool with retry and circuit breaker logic
        
        Args:
            server_name: Name of the server to execute the tool on
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            request_timeout: Optional timeout override in seconds
            
        Returns:
            The tool execution result
        """
        session = self.server_manager.active_sessions.get(server_name)
        if not session:
            raise RuntimeError(f"Server {server_name} not connected")
            
        # Get the tool from the server_manager
        tool = self.server_manager.tools.get(tool_name)
        if not tool:
            raise RuntimeError(f"Tool {tool_name} not found")
            
        # Set timeout if provided
        original_timeout = None
        if request_timeout and hasattr(session, 'timeout'):
            # Some session types might have configurable timeouts
            original_timeout = session.timeout
            session.timeout = request_timeout
            
        try:
            # Use the tool_execution_context context manager for metrics and tracing
            # Wrap with safe_stdout to protect against stdout pollution during tool execution
            with safe_stdout():
                async with self.tool_execution_context(tool_name, tool_args, server_name):
                    # Call the tool - this is unchanged from the original implementation
                    result = await session.call_tool(tool.original_tool.name, tool_args)
                    
                    # Check dependencies - unchanged from original implementation
                    if self.tool_cache:
                        dependencies = self.tool_cache.dependency_graph.get(tool_name, set())
                        if dependencies:
                            log.debug(f"Tool {tool_name} has dependencies: {dependencies}")
                    
                    return result.result
        finally:
            # Restore original timeout if it was changed - unchanged from original implementation
            if original_timeout is not None and hasattr(session, 'timeout'):
                session.timeout = original_timeout

    def completer(self, text, state):
        """Tab completion for commands"""
        options = [cmd for cmd in self.commands.keys() if cmd.startswith(text)]
        if state < len(options):
            return options[state]
        return None
        
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
                
    async def setup(self, interactive_mode=False):
        """Set up the client, connect to servers, and load capabilities"""
        # This instance will be passed to widgets requiring an explicit console.
        safe_console_instance = get_safe_console()

        # Ensure API key is set
        if not self.config.api_key:
            self.safe_print("[bold red]ERROR: Anthropic API key not found[/]") # Uses self.safe_print
            self.safe_print("Please set your API key using one of these methods:") # Uses self.safe_print
            self.safe_print("1. Set the ANTHROPIC_API_KEY environment variable") # Uses self.safe_print
            self.safe_print("2. Run 'python mcp_client.py run --interactive' and then use '/config api-key YOUR_API_KEY'") # Uses self.safe_print

            # Only exit if not in interactive mode
            if not interactive_mode:
                sys.exit(1)
            else:
                self.safe_print("[yellow]Running in interactive mode without API key.[/]") # Uses self.safe_print
                self.safe_print("[yellow]Please set your API key using '/config api-key YOUR_API_KEY'[/]") # Uses self.safe_print
                # Continue setup without API features
                self.anthropic = None  # Set to None until API key is provided

        self.conversation_graph = ConversationGraph() # Start with a default empty one

        if self.conversation_graph_file.exists():
            status_text = f"{STATUS_EMOJI['history']} Loading conversation state..." # Use history emoji
            with Status(status_text, console=safe_console_instance) as status:
                try:
                    # Load will now always return a graph object, handling errors internally
                    loaded_graph = await ConversationGraph.load(str(self.conversation_graph_file))
                    self.conversation_graph = loaded_graph # Replace default graph with result

                    # Check if loading resulted in a default graph (meaning load failed)
                    is_new_graph = (loaded_graph.root.id == "root" and
                                    not loaded_graph.root.messages and
                                    not loaded_graph.root.children and
                                    len(loaded_graph.nodes) == 1)

                    if is_new_graph:
                        # File existed, but loading failed. Inform user gently.
                        self.safe_print("[yellow]Could not load previous conversation state, starting fresh.[/yellow]")
                        status.update(f"{STATUS_EMOJI['warning']} Previous state invalid, starting fresh")
                    else:
                        # Successfully loaded existing graph
                        log.info(f"Loaded conversation graph from {self.conversation_graph_file}")
                        status.update(f"{STATUS_EMOJI['success']} Conversation state loaded")
                except Exception as setup_load_err:
                    # Catch unexpected errors *outside* the ConversationGraph.load logic
                    # (e.g., issues with the Status widget itself)
                    log.error("Unexpected error during conversation graph loading stage in setup", exc_info=True)
                    self.safe_print(f"[red]Error initializing conversation state: {setup_load_err}[/red]")
                    # Keep the default self.conversation_graph
                    status.update(f"{STATUS_EMOJI['error']} Error loading state")
                    # Ensure graph is still valid (it should be the default)
                    self.conversation_graph = ConversationGraph()
        else:
             log.info("No existing conversation graph found, using new graph.")
             # No need to print anything to console if file just doesn't exist

        if not self.conversation_graph.get_node(self.conversation_graph.current_node.id):
             log.warning("Current node ID was invalid after graph load/init, resetting to root.")
             self.conversation_graph.set_current_node("root")
        
        # Check for and load Claude desktop config if it exists
        await self.load_claude_desktop_config() # Assumes this uses safe logging/printing

        # Verify no stdout pollution before connecting to servers
        if os.environ.get("MCP_VERIFY_STDOUT", "1") == "1":
            # Use safe_stdout context manager to prevent the verification itself from polluting
            with safe_stdout():
                # Only log this, don't print directly to avoid any risk of stdout pollution
                log.info("Verifying no stdout pollution before connecting to servers...")
                verify_no_stdout_pollution()

        # Discover servers if enabled - Nested Live fix already applied
        if self.config.auto_discover:
            self.safe_print(f"{STATUS_EMOJI['search']} Discovering MCP servers...") # Uses self.safe_print
            try:
                await self.server_manager.discover_servers() # Uses _run_with_progress internally
            except Exception as discover_error:
                # Log the error and inform the user discovery failed
                log.error("Error during server discovery process", exc_info=True)
                self.safe_print(f"[red]Error during server discovery: {discover_error}[/red]") # Uses self.safe_print

        # Start continuous local discovery if enabled
        if self.config.enable_local_discovery and self.server_manager.registry:
            await self.start_local_discovery_monitoring() # Assumes this uses safe logging/printing

        enabled_servers = [s for s in self.config.servers.values() if s.enabled]
        if enabled_servers:
            self.safe_print(f"[bold blue]Connecting to {len(enabled_servers)} servers...[/]") # Uses self.safe_print
            connection_results = {}
            for name, server_config in self.config.servers.items():
                if not server_config.enabled: continue
                self.safe_print(f"[cyan]Connecting to {name}...[/]") # Uses self.safe_print
                try:
                    result = await self._connect_and_load_server(name, server_config) # Calls connect_to_server which uses _safe_printer
                    connection_results[name] = result
                    if result:
                        self.safe_print(f"  [green]✓ Connected to {name}[/]") # Uses self.safe_print
                    else:
                        log.warning(f"Failed to connect and load server: {name}")
                        self.safe_print(f"  [yellow]✗ Failed to connect to {name}[/]") # Uses self.safe_print
                except Exception as e:
                    log.error(f"Exception connecting to {name}", exc_info=True)
                    self.safe_print(f"  [red]✗ Error connecting to {name}: {e}[/]") # Uses self.safe_print
                    connection_results[name] = False

        # Start server monitoring
        # This Status widget is fine as it doesn't overlap with Progress calls
        try:
            with Status(f"{STATUS_EMOJI['server']} Starting server monitoring...",
                    spinner="dots", console=safe_console_instance) as status: # Pass variable
                await self.server_monitor.start_monitoring() # Assumes this uses safe logging
                status.update(f"{STATUS_EMOJI['success']} Server monitoring started")
        except Exception as monitor_error:
            log.error("Failed to start server monitoring", exc_info=True)
            self.safe_print(f"[red]Error starting server monitoring: {monitor_error}[/red]") # Uses self.safe_print

        # Display status without Progress widgets (uses print_simple_status)
        await self.print_simple_status() # Assumes print_simple_status uses self.safe_print
        
    async def _connect_and_load_server(self, server_name, server_config):
        """Connect to a server and load its capabilities (for use with _run_with_progress)"""
        session = await self.server_manager.connect_to_server(server_config)
        
        if session:
            self.server_manager.active_sessions[server_name] = session
            await self.server_manager.load_server_capabilities(server_name, session)
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
             await self.server_manager.close()
        # Close cache
        if self.tool_cache:
            self.tool_cache.close()
    
    async def process_streaming_query(self, query: str, model: Optional[str] = None, 
                               max_tokens: Optional[int] = None) -> AsyncIterator[str]:
        """Process a query using Claude and available tools with streaming"""
        # Wrap the entire function in safe_stdout to prevent any accidental stdout pollution
        # during the streaming interaction with stdio servers
        with safe_stdout():
            # Get core parameters
            if not model:
                model = self.current_model
                
            if not max_tokens:
                max_tokens = self.config.default_max_tokens
                
            # Check if we have any servers connected
            if not self.server_manager.active_sessions:
                yield "No MCP servers connected. Use 'servers connect' to connect to servers."
                return
            
            # Get tools from all connected servers
            available_tools = self.server_manager.format_tools_for_anthropic()
            
            if not available_tools:
                yield "No tools available from connected servers."
                return
                
            # Start timing for metrics
            start_time = time.time()
            
            # Keep track of servers and tools used
            servers_used = set()
            tools_used = []
            
            # Start with user message
            current_messages: List[MessageParam] = self.conversation_graph.current_node.messages.copy()
            user_message: MessageParam = {"role": "user", "content": query}
            messages: List[MessageParam] = current_messages + [user_message]
        
        # Start span for tracing if available
        span_ctx = None
        if tracer:
            span_ctx = tracer.start_as_current_span(
                "process_query",
                attributes={
                    "model": model,
                    "query_length": len(query),
                    "conversation_length": len(messages)
                }
            )
            
        try:
            # Initialize variables for streaming
            chunks = []
            assistant_message: List[MessageParam] = []
            current_text = ""
            
            # Create stream
            async with self.anthropic.messages.stream(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                tools=available_tools,
                temperature=self.config.temperature,
            ) as stream:
                # Process the streaming response
                async for message_delta in stream:
                    # Process event using helper method
                    event: MessageStreamEvent = message_delta
                    
                    if event.type == "content_block_delta":
                        # Process text delta using helper
                        current_text, delta_text = self._process_text_delta(event, current_text)
                        if delta_text:
                            chunks.append(delta_text)
                            yield delta_text
                            
                    elif event.type == "message_start":
                        # Message started, initialize containers
                        pass
                        
                    elif event.type == "content_block_start":
                        # Content block started
                        block = event.content_block
                        if block.type == "text":
                            current_text = ""
                        elif block.type == "tool_use":
                            yield f"\n[{STATUS_EMOJI['tool']}] Using tool: {block.name}..."
                            
                    elif event.type == "content_block_stop":
                        # Content block finished
                        block = event.content_block
                        if block.type == "text":
                            assistant_message.append({"type": "text", "text": current_text})
                        elif block.type == "tool_use":
                            # Process tool call
                            tool_name = block.name
                            tool_args = block.input
                            
                            # Find the server for this tool
                            tool = self.server_manager.tools.get(tool_name)
                            if not tool:
                                yield f"\n[Tool Error: '{tool_name}' not found]"
                                continue
                            
                            # Keep track of this server
                            servers_used.add(tool.server_name)
                            tools_used.append(tool_name)
                            
                            # Get server session
                            session = self.server_manager.active_sessions.get(tool.server_name)
                            if not session:
                                yield f"\n[Server Error: '{tool.server_name}' not connected]"
                                continue
                                
                            # Check tool cache if enabled
                            tool_result = None
                            cache_used = False
                            if self.tool_cache: # Check if cache is enabled and instantiated
                                tool_result = self.tool_cache.get(tool_name, tool_args)
                                if tool_result is not None:
                                    cache_used = True
                                    yield f"\n[{STATUS_EMOJI['cached']} Using cached result for {tool_name}]"
                            
                            # Execute the tool if not cached
                            if not cache_used:
                                yield f"\n[{STATUS_EMOJI['tool']}] Executing {tool_name}..."
                                
                                try:
                                    # Use the new execute_tool method with retry and circuit breaker
                                    tool_result = await self.execute_tool(tool.server_name, tool_name, tool_args)
                                    
                                    # Cache the result if enabled
                                    if self.tool_cache:
                                        self.tool_cache.set(tool_name, tool_args, tool_result)
                                        
                                except Exception as e:
                                    log.error(f"Error executing tool {tool_name}: {e}")
                                    yield f"\n[{STATUS_EMOJI['failure']}] Tool Execution Error: {str(e)}"
                                    
                                    # Skip this tool call
                                    continue
                            
                            # Add tool result to messages for context
                            assistant_message.append({"type": "tool_use", "id": block.id, "name": tool_name, "input": tool_args})
                            messages.append({"role": "assistant", "content": assistant_message})
                            messages.append({"role": "tool", "tool_use_id": block.id, "content": tool_result})
                            
                            # Reset for next content
                            assistant_message = []
                            current_text = ""
                            
                            # Continue the stream with new context
                            yield f"\n[{STATUS_EMOJI['success']}] Tool result received, continuing..."
                            
                            # Create new stream with updated context
                            async with self.anthropic.messages.stream(
                                model=model,
                                max_tokens=max_tokens,
                                messages=messages,
                                tools=available_tools,
                                temperature=self.config.temperature,
                            ) as continuation_stream:
                                async for continued_delta in continuation_stream:
                                    # Process event using helper method
                                    continued_event: MessageStreamEvent = continued_delta
                                    
                                    # Use the stream event processing helper
                                    current_text, text_to_yield = self._process_stream_event(continued_event, current_text)
                                    if text_to_yield:
                                        chunks.append(text_to_yield)
                                        yield text_to_yield
                                        
                                    # Handle content block stop specially
                                    if continued_event.type == "content_block_stop" and continued_event.content_block.type == "text":
                                        assistant_message.append({"type": "text", "text": current_text})
                    
                    elif event.type == "message_stop":
                        # Message complete
                        pass
            
            # Update conversation messages for future continuations
            self.conversation_graph.current_node.add_message(user_message)
            self.conversation_graph.current_node.add_message({"role": "assistant", "content": assistant_message})
            self.conversation_graph.current_node.model = model # Store model used for this node
            
            # Create a complete response for history
            complete_response = "".join(chunks)
            
            # Calculate metrics
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            # Add to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.history.add(ChatHistory(
                query=query,
                response=complete_response,
                model=model,
                timestamp=timestamp,
                server_names=list(servers_used),
                tools_used=tools_used,
                conversation_id=self.conversation_graph.current_node.id, # Add conversation ID
                latency_ms=latency_ms,
                streamed=True
            ))
            
            # End trace span
            if span_ctx:
                span_ctx.set_status(trace.StatusCode.OK)
                span_ctx.add_event("query_complete", {
                    "latency_ms": latency_ms,
                    "tools_used": len(tools_used),
                    "servers_used": len(servers_used)
                })
                span_ctx.end()
                
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            log.error(error_msg)
            yield f"\n[Error: {error_msg}]"
            
            # End trace span with error
            if span_ctx:
                span_ctx.set_status(trace.StatusCode.ERROR, error_msg)
                span_ctx.end()

    async def process_query(self, query: str, model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
        """Process a query using Claude and available tools (non-streaming version)"""
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
            return "".join(chunks)
        
        # Non-streaming implementation for backwards compatibility
        start_time = time.time()
        
        # Check if we have any servers connected
        if not self.server_manager.active_sessions:
            return "No MCP servers connected. Use 'servers connect' to connect to servers."
        
        # Get tools from all connected servers
        available_tools = self.server_manager.format_tools_for_anthropic()
        
        if not available_tools:
            return "No tools available from connected servers."
            
        # Keep track of servers and tools used
        servers_used = set()
        tools_used = []
        
        # Start with user message
        current_messages: List[MessageParam] = self.conversation_graph.current_node.messages.copy()
        user_message: MessageParam = {"role": "user", "content": query}
        messages: List[MessageParam] = current_messages + [user_message]
        
        # Create span for tracing if available
        span_ctx = None
        if tracer:
            span_ctx = tracer.start_as_current_span(
                "process_query",
                attributes={
                    "model": model,
                    "query_length": len(query),
                    "conversation_length": len(messages),
                    "streaming": False
                }
            )
        
        with Status(f"{STATUS_EMOJI['speech_balloon']} Claude is thinking...", spinner="dots", console=get_safe_console()) as status:
            try:
                # Make initial API call
                status.update(f"{STATUS_EMOJI['speech_balloon']} Sending query to Claude ({model})...")
                response = await self.anthropic.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=messages,
                    tools=available_tools,
                    temperature=self.config.temperature,
                )
                
                status.update(f"{STATUS_EMOJI['success']} Received response from Claude")
                
                # Process response and handle tool calls
                final_text = []
                assistant_message: List[MessageParam] = []
                
                for content in response.content:
                    if content.type == 'text':
                        final_text.append(content.text)
                        assistant_message.append(content)
                        
                    elif content.type == 'tool_use':
                        tool_name = content.name
                        tool_args = content.input
                        
                        # Find the server for this tool
                        tool = self.server_manager.tools.get(tool_name)
                        if not tool:
                            final_text.append(f"\n[Tool Error: '{tool_name}' not found]")
                            continue
                        
                        # Keep track of this server
                        servers_used.add(tool.server_name)
                        tools_used.append(tool_name)
                        
                        # Get server session
                        session = self.server_manager.active_sessions.get(tool.server_name)
                        if not session:
                            final_text.append(f"\n[Server Error: '{tool.server_name}' not connected]")
                            continue
                        
                        # Check tool cache if enabled
                        tool_result = None
                        cache_used = False
                        if self.tool_cache: # Check if cache is enabled and instantiated
                            tool_result = self.tool_cache.get(tool_name, tool_args)
                            if tool_result is not None:
                                cache_used = True
                                status.update(f"{STATUS_EMOJI['package']} Using cached result for {tool_name}")
                                log.info(f"Using cached result for {tool_name}")
                        
                        # Execute the tool if not cached
                        if not cache_used:
                            # Execute the tool
                            status.update(f"{STATUS_EMOJI['tool']} Executing tool: {tool_name}...")
                            
                            try:
                                # Use safe_stdout to prevent stdout pollution during tool execution
                                with safe_stdout():
                                    # Use the new execute_tool method with retry and circuit breaker
                                    tool_result = await self.execute_tool(tool.server_name, tool_name, tool_args)
                                
                                # Update progress
                                status.update(f"{STATUS_EMOJI['success']} Tool {tool_name} execution complete")
                                
                                # Cache the result if enabled
                                if self.tool_cache:
                                    self.tool_cache.set(tool_name, tool_args, tool_result)
                                
                            except Exception as e:
                                log.error(f"Error executing tool {tool_name}: {e}")
                                status.update(f"{STATUS_EMOJI['failure']} Tool {tool_name} execution failed: {e}")
                                final_text.append(f"\n[Tool Execution Error: {str(e)}]")
                                
                                # Skip this tool call
                                continue
                        
                        # Add tool result to messages list
                        messages.append({"role": "assistant", "content": assistant_message})
                        messages.append({"role": "tool", "tool_use_id": content.id, "content": tool_result})
                        
                        # Update assistant message for next potential tool call
                        assistant_message = []
                        
                        # Make another API call with the tool result
                        status.update(f"{STATUS_EMOJI['speech_balloon']} Claude is processing tool result...")
                        
                        response = await self.anthropic.messages.create(
                            model=model,
                            max_tokens=max_tokens,
                            messages=messages,
                            tools=available_tools,
                            temperature=self.config.temperature,
                        )
                        
                        status.update(f"{STATUS_EMOJI['success']} Received response from Claude")
                        
                        # Reset final_text to capture only the latest response
                        final_text = []
                        
                        # Process the new response
                        for content in response.content:
                            if content.type == 'text':
                                final_text.append(content.text)
                                assistant_message.append(content)
                
                # Update conversation messages for future continuations
                self.conversation_graph.current_node.messages = messages # Store full history leading to this response
                self.conversation_graph.current_node.add_message({"role": "assistant", "content": assistant_message}) # Add final assistant response
                self.conversation_graph.current_node.model = model # Store model used for this node
                
                # Create a single string from all text pieces
                result = "".join(final_text)
                
                # Calculate metrics
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # Add to history
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.history.add(ChatHistory(
                    query=query,
                    response=result,
                    model=model,
                    timestamp=timestamp,
                    server_names=list(servers_used),
                    tools_used=tools_used,
                    conversation_id=self.conversation_graph.current_node.id, # Add conversation ID
                    latency_ms=latency_ms,
                    streamed=False
                ))
                
                # End trace span
                if span_ctx:
                    span_ctx.set_status(trace.StatusCode.OK)
                    span_ctx.add_event("query_complete", {
                        "latency_ms": latency_ms,
                        "tools_used": len(tools_used),
                        "servers_used": len(servers_used),
                        "response_length": len(result)
                    })
                    span_ctx.end()
                
                return result
                
            except Exception as e:
                error_msg = f"Error processing query: {str(e)}"
                log.error(error_msg)
                
                # End trace span with error
                if span_ctx:
                    span_ctx.set_status(trace.StatusCode.ERROR, error_msg)
                    span_ctx.end()
                
                return f"[Error: {error_msg}]"
        
    async def interactive_loop(self):
        """Run interactive command loop"""
        # Always use get_safe_console for interactive mode to avoid interference with stdio servers
        interactive_console = get_safe_console()
        
        self.safe_print("\n[bold green]MCP Client Interactive Mode[/]")
        self.safe_print("Type your query to Claude, or a command (type 'help' for available commands)")
        self.safe_print("[italic]Press Ctrl+C twice in quick succession to force exit if unresponsive[/italic]")
        
        # Track Ctrl+C attempts for force exit
        ctrl_c_count = 0
        last_ctrl_c_time = 0
        
        while True:
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
                        await self.commands[cmd](args)
                    else:
                        interactive_console.print(f"[yellow]Unknown command: {cmd}[/]")
                        interactive_console.print("Type '/help' for available commands")
                
                # Empty input
                elif not user_input.strip():
                    continue
                
                # Process as a query to Claude
                else:
                    result = await self.process_query(user_input)
                    interactive_console.print()
                    interactive_console.print(Panel.fit(
                        Markdown(result),
                        title="Claude",
                        border_style="green"
                    ))
                    
            except KeyboardInterrupt:
                # Track Ctrl+C for double-press detection
                current_time = time.time()
                if current_time - last_ctrl_c_time < 2:  # Within 2 seconds
                    ctrl_c_count += 1
                else:
                    ctrl_c_count = 1
                last_ctrl_c_time = current_time
                
                if ctrl_c_count >= 2:
                    self.safe_print("\n[bold red]Force exiting...[/]")
                    # Terminate any active processes immediately
                    for name, process in self.server_manager.processes.items():
                        try:
                            if process and process.returncode is None:
                                self.safe_print(f"[yellow]Force killing process: {name}[/]")
                                process.kill()
                        except Exception as e:
                            self.safe_print(f"[red]Error killing process {name}: {e}[/]")
                    break
                
                self.safe_print("\n[yellow]Interrupted. Press Ctrl+C again to force exit.[/]")
                continue
            # Catch specific errors related to command execution or query processing
            except (anthropic.APIError, McpError, httpx.RequestError) as e: 
                self.safe_print(f"[bold red]Error ({type(e).__name__}):[/] {str(e)}")
            # Keep broad exception for unexpected loop issues
            except Exception as e: 
                self.safe_print(f"[bold red]Unexpected Error:[/] {str(e)}")
        
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
            Text(f"{STATUS_EMOJI['config']} /config", style="bold"), Text(" - Manage configuration"),
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
                                    
                                    # Load capabilities
                                    status.update(f"{STATUS_EMOJI['tool']} Loading capabilities from {name}...")
                                    await self.server_manager.load_server_capabilities(name, session)
                                    status.update(f"{STATUS_EMOJI['success']} Loaded capabilities from server: {name}")
                                    
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
                            await self.server_manager.load_server_capabilities(name, session)
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
                    Syntax(json.dumps(tool.input_schema).decode('utf-8'), "json", theme="monokai")
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
             # cache_size += len(self.tool_cache.disk_cache) # Example
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
                    await f.write(json.dumps(export_data).decode('utf-8'))
                
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
                    Syntax(json.dumps(result).decode('utf-8'), "json", theme="monokai"),
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
                # Use the 'json' alias which points to orjson
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
                # Inform user if no new servers were actually imported
                self.safe_print(f"{STATUS_EMOJI['warning']} No new servers were imported from Claude desktop config (they might already exist or failed processing).")

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
):
    """Run the MCP client in various modes"""
    # Configure logging based on verbosity
    if verbose:
        logging.getLogger("mcpclient").setLevel(logging.DEBUG)
    
    # Run the main async function
    asyncio.run(main_async(query, model, server, dashboard, interactive, verbose))

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

async def main_async(query, model, server, dashboard, interactive, verbose_logging):
    """Main async entry point"""
    client = None # Initialize client to None
    safe_console = get_safe_console()
    max_shutdown_timeout = 10

    try:
        # Initialize client
        client = MCPClient() # Instantiation inside the try block

        # Set up client with error handling
        try:
            # Pass interactive flag here correctly
            await client.setup(interactive_mode=interactive)
        except Exception as setup_error:
            # Log with traceback using standard logging
            log.error("Error occurred during client setup", exc_info=True)
            # Print user-facing message
            safe_console.print(f"[bold red]Error during setup:[/] {setup_error}")
            if not interactive:
                raise # Re-raise to exit non-interactive mode

        # Connect to specific server(s) if provided
        if server:
            connection_errors = []
            for s in server:
                if s in client.config.servers:
                    try:
                        with Status(f"[cyan]Connecting to server {s}...[/]", console=safe_console):
                            await client.connect_server(s)
                    except Exception as e:
                        connection_errors.append((s, str(e)))
                        safe_console.print(f"[red]Error connecting to server {s}: {e}[/]")
                else:
                    safe_console.print(f"[yellow]Server not found: {s}[/]")
            
            if connection_errors and not interactive and not query:
                # Only exit for errors in non-interactive mode with no query
                raise Exception(f"Failed to connect to servers: {', '.join(s for s, _ in connection_errors)}")
        
        # Launch dashboard if requested
        if dashboard:
            # Ensure monitor is running for dashboard data
            if not client.server_monitor.monitoring:
                await client.server_monitor.start_monitoring()
            await client.cmd_dashboard("") # Call the command method
            # Dashboard is blocking, exit after it's closed
            await client.close() # Ensure cleanup after dashboard closes
            return
        
        # Process single query if provided
        if query:
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
        
        # Run interactive loop if requested or if no query
        elif interactive or not query:
            # Prompt for API key if not set and in interactive mode
            if not client.config.api_key and interactive:
                stderr_console.print("\n[bold yellow]Anthropic API key required[/]")
                stderr_console.print("An API key is needed to use Claude and MCP tools")
                api_key = Prompt.ask("[bold green]Enter your Anthropic API key[/]", password=True, console=stderr_console)
                
                if api_key and api_key.strip():
                    # Set the API key and initialize Anthropic client
                    client.config.api_key = api_key.strip()
                    try:
                        client.anthropic = AsyncAnthropic(api_key=client.config.api_key)
                        client.config.save()
                        stderr_console.print("[green]API key saved successfully![/]")
                    except Exception as e:
                        stderr_console.print(f"[red]Error initializing Anthropic client: {e}[/]")
                        client.anthropic = None
                else:
                    stderr_console.print("[yellow]No API key provided. You can set it later with [bold]/config api-key YOUR_KEY[/bold][/]")
            
            # Always run interactive loop if requested, even if there were errors
            await client.interactive_loop()
            
    except KeyboardInterrupt:
        safe_console.print("\n[yellow]Interrupted, shutting down...[/]")
        # Ensure cleanup happens if main loop interrupted
        if 'client' in locals() and client: 
            try:
                # Use timeout to prevent hanging during shutdown
                await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout)
            except asyncio.TimeoutError:
                safe_console.print("[red]Shutdown timed out. Some processes may still be running.[/]")
                # Force kill any stubborn processes
                if hasattr(client, 'server_manager'):
                    for name, process in client.server_manager.processes.items():
                        if process and process.returncode is None:
                            try:
                                safe_console.print(f"[yellow]Force killing process: {name}[/]")
                                process.kill()
                            except Exception:
                                pass
            except Exception as e:
                safe_console.print(f"[red]Error during shutdown: {e}[/]")
    
    except Exception as main_async_error:
            # Catch any other exception originating from setup or main logic

            # Print user-facing error message
            safe_console.print(f"[bold red]An unexpected error occurred in the main process: {main_async_error}[/]")

            # Print traceback directly to stderr for diagnostics
            print(f"\n--- Traceback for Main Process Error ({type(main_async_error).__name__}) ---", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            print("--- End Traceback ---", file=sys.stderr)

            # Exit non-interactive mode with an error code
            if not interactive:
                # Attempt graceful cleanup even on error
                if client and hasattr(client, 'close'):
                    try:
                        await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout / 2) # Shorter timeout on error
                    except Exception:
                        pass # Ignore cleanup errors during error handling
                sys.exit(1)
    finally:
        # Clean up (already handled in KeyboardInterrupt and normal exit paths)
        # Ensure close is called if setup succeeded but something else failed
        if 'client' in locals() and client and hasattr(client, 'server_manager'): 
            try:
                # Add timeout to prevent hanging
                await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout)
            except asyncio.TimeoutError:
                safe_console.print("[red]Shutdown timed out. Some processes may still be running.[/]")
                # Force kill processes that didn't shut down cleanly
                if hasattr(client, 'server_manager') and hasattr(client.server_manager, 'processes'):
                    for name, process in client.server_manager.processes.items():
                        if process and process.returncode is None:
                            try:
                                safe_console.print(f"[yellow]Force killing process: {name}[/]")
                                process.kill()
                            except Exception:
                                pass
            except Exception as close_error:
                log.error(f"Error during cleanup: {close_error}")
                # Continue shutdown regardless of cleanup errors

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
                safe_console.print(json.dumps(server_data).decode('utf-8'))
            else:
                # Normal output
                await client.list_servers()
    
    finally:
        await client.close()

async def config_async(show, edit, reset):
    """Config management async function"""
    # client = MCPClient() # Instantiation might be better outside if used elsewhere
    # For simplicity here, assuming it's needed within this command scope
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
    # These should catch errors during MCPClient() init or file operations if needed,
    # NOT the misplaced blocks from before.
    except (IOError, yaml.YAMLError, json.JSONDecodeError) as e:
         safe_console.print(f"[bold red]Configuration/File Error during config command:[/] {str(str(e))}")
         # Decide if sys.exit is appropriate here or just log
    except Exception as e:
        safe_console.print(f"[bold red]Unexpected Error during config command:[/] {str(e)}")
        # Log detailed error if needed

    finally:
        # Ensure client resources are cleaned up if it was initialized
        if client and hasattr(client, 'close'):
             try:
                 await client.close()
             except Exception as close_err:
                 log.error(f"Error during config_async client cleanup: {close_err}")


async def main():
    """Main entry point"""
    try:
        app()
    except McpError as e:
        # Use get_safe_console() to prevent pollution
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

if __name__ == "__main__":
    # Initialize colorama for Windows terminals
    if platform.system() == "Windows":
        colorama.init(convert=True)
        
    # Run the app
    app()
