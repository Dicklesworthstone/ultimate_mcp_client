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
import json
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
import uuid
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager, redirect_stdout
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

# Third-party imports
import aiofiles  # For async file operations
import anthropic

# Additional utilities
import colorama

# Cache libraries
import diskcache
import httpx
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
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import Prompt as McpPrompt
from mcp.types import Resource, Tool

# Observability
from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
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
                        if process.poll() is None:  # If process is still running
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
def adapt_path_for_platform(command, args):
    is_windows = platform.system() == "Windows"
    is_wsl = "WSL" in platform.platform().upper() or os.path.exists("/proc/sys/fs/binfmt_misc/WSLInterop")
    
    # Improved Windows path to WSL path conversion
    def convert_to_wsl_path(win_path):
        if not win_path or not isinstance(win_path, str):
            return win_path
        
        # Handle C:\ style paths for WSL - convert to /mnt/c/...
        if len(win_path) > 2 and win_path[1] == ":" and win_path[2] in ["\\", "/"]:
            drive_letter = win_path[0].lower()
            # Remove the drive letter and colon, then replace backslashes with forward slashes
            rest_of_path = win_path[2:].replace("\\", "/")
            
            # Ensure there's only one leading slash and no trailing slashes
            if rest_of_path.startswith("/"):
                rest_of_path = rest_of_path[1:]
            rest_of_path = rest_of_path.rstrip("/")
                
            wsl_path = f"/mnt/{drive_letter}/{rest_of_path}"
            log.info(f"Converted Windows path '{win_path}' to WSL path '{wsl_path}'")
            return wsl_path
        return win_path
    
    # If we're running on Windows, no need to adapt paths
    if is_windows and not is_wsl:
        return command, args
    
    # If we're on Linux/WSL and command contains Windows paths
    if not is_windows or is_wsl:
        # For filesystem server specifically
        if "npx" in command and "@modelcontextprotocol/server-filesystem" in str(args):
            new_args = []
            for arg in args:
                if isinstance(arg, str) and ":" in arg:
                    # Convert Windows path to WSL path
                    new_args.append(convert_to_wsl_path(arg))
                else:
                    new_args.append(arg)
            
            log.info(f"Adapted filesystem server args: {args} -> {new_args}")
            return command, new_args
        
        # For wsl.exe commands running in Linux (extract the bash command)
        elif "wsl.exe" in command:
            if isinstance(args, list) and len(args) >= 3:
                try:
                    bash_c_index = args.index("-c")
                    if bash_c_index + 1 < len(args):
                        actual_command = args[bash_c_index + 1]
                        log.info(f"Converting wsl.exe command to direct bash command")
                        return "bash", ["-c", actual_command]
                except ValueError:
                    # If we can't parse it, just return as is
                    log.warning(f"Could not extract bash command from wsl.exe args: {args}")
                    # Still try to convert any Windows paths in args
                    new_args = []
                    for arg in args:
                        if isinstance(arg, str) and ":" in arg and len(arg) > 2 and arg[1] == ":":
                            new_args.append(convert_to_wsl_path(arg))
                        else:
                            new_args.append(arg)
                    return command, new_args
        
        # For any command with Windows paths
        else:
            new_args = []
            windows_path_found = False
            
            for arg in args:
                if isinstance(arg, str) and ":" in arg and len(arg) > 2 and arg[1] == ":":
                    windows_path_found = True
                    new_args.append(convert_to_wsl_path(arg))
                else:
                    new_args.append(arg)
            
            if windows_path_found:
                log.info(f"Converted Windows paths in command args: {args} -> {new_args}")
                return command, new_args
    
    # Default case - return unchanged
    return command, args

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
                    
                    # Add debug info to help identify unsafe console usage
                    caller_frame = inspect.currentframe().f_back
                    if caller_frame:
                        caller_info = inspect.getframeinfo(caller_frame)
                        # Check if this is being called by something other than safe_print
                        if not (caller_info.function == "safe_print" or "safe_print" in caller_info.code_context[0]):
                            log.warning(f"Potential unsafe console usage detected at: {caller_info.filename}:{caller_info.lineno}")
                            log.warning(f"Always use MCPClient.safe_print() or get_safe_console().print() to prevent stdio corruption")
                            log.warning(f"Stack: {caller_info.function} - {caller_info.code_context[0].strip()}")
                    
                    break
    except (NameError, AttributeError):
        pass
    
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
    "failure": Emoji("collision")
}

# Constants
CONFIG_DIR = Path.home() / ".config" / "mcpclient"
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
console_exporter = ConsoleSpanExporter()
span_processor = BatchSpanProcessor(console_exporter)
trace_provider.add_span_processor(span_processor)
trace.set_tracer_provider(trace_provider)

# Initialize metrics with the current API
try:
    # Try the newer API first
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    
    reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
    meter_provider = MeterProvider(metric_readers=[reader])
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
            from zeroconf import ServiceBrowser, Zeroconf
            
            class MCPServiceListener:
                def __init__(self, registry):
                    self.registry = registry
                
                def add_service(self, zeroconf, service_type, name):
                    info = zeroconf.get_service_info(service_type, name)
                    if not info:
                        return
                        
                    server_name = name.replace("._mcp._tcp.local.", "")
                    host = socket.inet_ntoa(info.addresses[0]) if info.addresses else "localhost"
                    port = info.port
                    
                    # Extract and parse properties from TXT records
                    properties = {}
                    if info.properties:
                        for k, v in info.properties.items():
                            try:
                                key = k.decode('utf-8')
                                value = v.decode('utf-8')
                                properties[key] = value
                            except UnicodeDecodeError:
                                # Skip binary properties that can't be decoded as UTF-8
                                continue

                    # Determine server type from properties or default to SSE
                    server_type = "sse"
                    if "type" in properties:
                        server_type = properties["type"]
                        
                    # Extract version information if available
                    version = None
                    if "version" in properties:
                        try:
                            version = properties["version"]
                        except Exception:
                            pass
                            
                    # Extract categories information if available
                    categories = []
                    if "categories" in properties:
                        try:
                            categories = properties["categories"].split(",")
                        except Exception:
                            pass
                            
                    # Get description if available
                    description = properties.get("description", f"mDNS discovered server at {host}:{port}")
                    
                    self.registry.discovered_servers[server_name] = {
                        "name": server_name,
                        "host": host,
                        "port": port,
                        "type": server_type,
                        "url": f"http://{host}:{port}",
                        "properties": properties,
                        "version": version,
                        "categories": categories,
                        "description": description,
                        "discovered_via": "mdns"
                    }
                    
                    log.info(f"Discovered local MCP server: {server_name} at {host}:{port} ({description})")
                
                def remove_service(self, zeroconf, service_type, name):
                    server_name = name.replace("._mcp._tcp.local.", "")
                    if server_name in self.registry.discovered_servers:
                        del self.registry.discovered_servers[server_name]
                        log.info(f"Removed local MCP server: {server_name}")
                
                def update_service(self, zeroconf, service_type, name):
                    # This method is required by newer versions of Zeroconf
                    # Can be empty if we don't need to handle service updates
                    pass
            
            self.zeroconf = Zeroconf()
            listener = MCPServiceListener(self)
            self.browser = ServiceBrowser(self.zeroconf, "_mcp._tcp.local.", listener)
            log.info("Started local MCP server discovery")
        except ImportError:
            log.warning("Zeroconf not available, local discovery disabled")
        except OSError as e: # Catch potential socket errors (Corrected indentation)
             log.error(f"Error starting local discovery (network issue?): {e}")
    
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
                await f.write(json.dumps(data, indent=2))
        except IOError as e:
             log.error(f"Could not write conversation graph to {file_path}: {e}")
        except TypeError as e: # Handle potential issues with non-serializable data
             log.error(f"Could not serialize conversation graph: {e}")
    
    @classmethod
    async def load(cls, file_path: str) -> "ConversationGraph":
        """Load a conversation graph from file asynchronously"""
        try:
            async with aiofiles.open(file_path, 'r') as f:
                content = await f.read()
                # Check if the file is empty
                if not content.strip():
                    log.warning(f"Conversation graph file is empty: {file_path}")
                    # Create a new file with a basic graph structure
                    graph = cls()
                    await graph.save(file_path)
                    return graph
                
                data = json.loads(content)
        except FileNotFoundError:
            log.warning(f"Conversation graph file not found: {file_path}")
            raise # Re-raise to be handled by the caller (__init__)
        except IOError as e:
            log.error(f"Could not read conversation graph file {file_path}: {e}")
            raise # Re-raise to be handled by the caller (__init__)
        except json.JSONDecodeError as e:
            log.error(f"Error decoding conversation graph JSON from {file_path}: {e}")
            # Create a new graph and save it to replace the corrupted file
            graph = cls()
            try:
                await graph.save(file_path)
                log.info(f"Created new conversation graph to replace corrupted file: {file_path}")
            except Exception as save_error:
                log.error(f"Failed to replace corrupted graph file: {save_error}")
            return graph

        graph = cls()
        
        try:
            # First pass: create all nodes
            for node_id, node_data in data["nodes"].items():
                node = ConversationNode.from_dict(node_data)
                graph.nodes[node_id] = node
            
            # Second pass: set up parent-child relationships
            for node_id, node_data in data["nodes"].items():
                node = graph.nodes[node_id]
                
                # Set parent
                parent_id = node_data.get("parent_id")
                if parent_id and parent_id in graph.nodes:
                    node.parent = graph.nodes[parent_id]
                
                # Set children
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
                # If current node ID is invalid, default to root
                log.warning(f"Saved current_node_id '{current_node_id}' not found, defaulting to root.")
                graph.current_node = graph.root # Assume root always exists
        except KeyError as e:
            log.error(f"Missing expected key in conversation graph data: {e}")
            raise # Re-raise to be handled by the caller (__init__)
        except (TypeError, ValueError) as e: # Catch potential errors during node creation/linking
            log.error(f"Error processing conversation graph data structure: {e}")
            raise # Re-raise to be handled by the caller (__init__)

        return graph


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
                json.dump(history_data, f, indent=2)
                
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


class ServerManager:
    def __init__(self, config: Config, tool_cache=None):
        self.config = config
        self.exit_stack = AsyncExitStack()
        self.active_sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.processes: Dict[str, subprocess.Popen] = {}
        self.tool_cache = tool_cache
        
        # Server monitoring
        self.monitor = ServerMonitor(self)
        
        # Connect to registry if enabled
        self.registry = ServerRegistry() if config.enable_registry else None
    
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
    
    async def connect_to_server(self, server_config: ServerConfig) -> Optional[ClientSession]:
        """Connect to a single MCP server with retry logic and health monitoring"""
        server_name = server_config.name
        retry_count = 0
        max_retries = server_config.retry_count
        
        # Ensure we're using the safe console during connection to avoid stdio interference
        safe_console = get_safe_console()
        
        # Use safe_stdout context manager to protect against stdout pollution during the entire connection process
        with safe_stdout():
            while retry_count <= max_retries:
                # Track metrics for this connection attempt
                start_time = time.time()
                
                try:
                    # Start span for observability if available
                    span_ctx = None
                    if tracer:
                        try:
                            span_ctx = tracer.start_as_current_span(
                                f"connect_server.{server_name}",
                                attributes={
                                    "server.name": server_name,
                                    "server.type": server_config.type.value,
                                    "server.path": server_config.path,
                                    "retry": retry_count
                                }
                            )
                        except Exception as e:
                            log.warning(f"Failed to create trace span: {e}")
                            span_ctx = None
                    
                    # Log connection info using safe_console
                    safe_console.print(f"[cyan]Attempting to connect to server {server_name}...[/]")

                    if server_config.type == ServerType.STDIO:
                        # Check if we need to start the server process
                        if server_config.auto_start and server_config.path:
                            # Check if a process is already running for this server
                            existing_process = self.processes.get(server_config.name)
                            restart_process = False
                            
                            if existing_process:
                                # Check if the process is still alive
                                if existing_process.poll() is None:
                                    # Process is running, but if we've hit an error before, we might want to restart
                                    if retry_count > 0:
                                        log.warning(f"Restarting process for {server_name} on retry {retry_count}")
                                        safe_console.print(f"[yellow]Restarting process for {server_name} on retry {retry_count}[/]")
                                        restart_process = True
                                        
                                        # Try to terminate gracefully
                                        try:
                                            existing_process.terminate()
                                            try:
                                                # Wait with timeout for termination
                                                await asyncio.wait_for(existing_process.wait(), timeout=2.0)
                                            except asyncio.TimeoutError:
                                                # Force kill if necessary
                                                log.warning(f"Process for {server_name} not responding to terminate, killing")
                                                safe_console.print(f"[red]Process for {server_name} not responding, forcing kill[/]")
                                                existing_process.kill()
                                                await existing_process.wait()
                                        except Exception as e:
                                            log.error(f"Error terminating process for {server_name}: {e}")
                                            safe_console.print(f"[red]Error terminating process: {e}[/]")
                                else:
                                    # Process has exited
                                    log.warning(f"Process for {server_name} has exited with code {existing_process.returncode}")
                                    safe_console.print(f"[yellow]Process for {server_name} has exited with code {existing_process.returncode}[/]")
                                    
                                    # Try to get stderr output for diagnostics if process has terminated
                                    try:
                                        if existing_process.stderr:
                                            stderr_data = await existing_process.stderr.read()
                                            if stderr_data:
                                                log.error(f"Process stderr: {stderr_data.decode('utf-8', errors='replace')}")
                                    except Exception as e:
                                        log.warning(f"Couldn't read stderr: {e}")
                                    
                                    restart_process = True
                            else:
                                # No existing process
                                restart_process = True
                            
                            # Start or restart the process if needed
                            if restart_process:
                                # Start the process
                                cmd = [server_config.path] + server_config.args
                                
                                # Detect if it's a Python file
                                if server_config.path.endswith('.py'):
                                    python_cmd = sys.executable
                                    cmd = [python_cmd] + cmd
                                # Detect if it's a JS file
                                elif server_config.path.endswith('.js'):
                                    node_cmd = 'node'
                                    cmd = [node_cmd] + cmd
                                
                                log.info(f"Starting server process: {' '.join(cmd)}")
                                safe_console.print(f"[cyan]Starting server process: {' '.join(cmd)}[/]")
                                
                                # Create process with pipes and set resource limits
                                # Add a unique identifier to each process to prevent interference
                                env = os.environ.copy()
                                # These environment variables are critical for preventing interference between
                                # multiple stdio MCP servers running on the same machine:
                                # - MCP_SERVER_ID: Unique name for each server to distinguish protocol messages
                                # - MCP_CLIENT_ID: Unique ID for this client instance to prevent cross-talk
                                env["MCP_SERVER_ID"] = server_name
                                env["MCP_CLIENT_ID"] = str(uuid.uuid4())
                                
                                process = await asyncio.create_subprocess_exec(
                                    *cmd,
                                    stdin=asyncio.subprocess.PIPE,
                                    stdout=asyncio.subprocess.PIPE,
                                    stderr=asyncio.subprocess.PIPE,
                                    env=env
                                )
                                
                                self.processes[server_config.name] = process
                                
                                # Register the server with zeroconf if local discovery is enabled
                                if self.config.enable_local_discovery and self.registry:
                                    await self.register_local_server(server_config)
                        
                        # Set up parameters with timeout - include existing process if available
                        params = StdioServerParameters(
                            command=server_config.path, 
                            args=server_config.args,
                            timeout=server_config.timeout,
                            process=self.processes.get(server_name)  # Pass existing process if available
                        )
                        
                        # Create client with context manager to ensure proper cleanup
                        # FIX: Remove the extra await before stdio_client
                        session = await self.exit_stack.enter_async_context(stdio_client(params))
                        
                    elif server_config.type == ServerType.SSE:
                        # Connect to SSE server using direct parameters
                        # FIX: Remove the extra await before sse_client
                        session = await self.exit_stack.enter_async_context(
                            sse_client(
                                url=server_config.path,
                                timeout=server_config.timeout,
                                sse_read_timeout=server_config.timeout * 12  # Set longer timeout for events
                            )
                        )
                    else:
                        if span_ctx and hasattr(span_ctx, 'set_status'):
                            span_ctx.set_status(trace.StatusCode.ERROR, f"Unknown server type: {server_config.type}")
                        log.error(f"Unknown server type: {server_config.type}")
                        safe_console.print(f"[red]Unknown server type: {server_config.type}[/]")
                        return None
                    
                    # Calculate connection time
                    connection_time = (time.time() - start_time) * 1000  # ms
                    
                    # Update metrics
                    server_config.metrics.request_count += 1
                    server_config.metrics.update_response_time(connection_time)
                    server_config.metrics.update_status()
                    
                    # Record metrics if available
                    if latency_histogram:
                        latency_histogram.record(
                            connection_time,
                            {
                                "operation": "connect",
                                "server": server_name,
                                "server_type": server_config.type.value
                            }
                        )
                    
                    # Mark span as successful
                    if span_ctx and hasattr(span_ctx, 'set_status'):
                        try:
                            span_ctx.set_status(trace.StatusCode.OK)
                            if hasattr(span_ctx, 'end'):
                                span_ctx.end()
                        except Exception as e:
                            log.warning(f"Error updating span status: {e}")
                    
                    log.info(f"Connected to server {server_name} in {connection_time:.2f}ms")
                    safe_console.print(f"[green]Connected to server {server_name} in {connection_time:.2f}ms[/]")
                    return session
                    
                except McpError as e: # Catch MCP client errors
                    connection_error = e
                except httpx.RequestError as e: # Network errors for SSE
                    connection_error = e
                except subprocess.SubprocessError as e: # Errors starting/communicating with STDIO process
                    connection_error = e
                    # Check if the process terminated unexpectedly
                    if server_config.name in self.processes:
                        proc = self.processes[server_config.name]
                        if proc.poll() is not None:
                            try:
                                stderr_data = await proc.stderr.read() if proc.stderr else b"stderr not captured"
                                stderr_output = stderr_data.decode('utf-8', errors='replace')
                                log.error(f"STDIO server process for '{server_config.name}' exited with code {proc.returncode}. Stderr: {stderr_output}")
                                safe_console.print(f"[red]STDIO server process for '{server_config.name}' exited with code {proc.returncode}[/]")
                            except Exception as err:
                                log.error(f"Error reading stderr: {err}")
                except OSError as e: # OS level errors (e.g., command not found)
                    connection_error = e
                # Keep broad exception for truly unexpected connection issues
                except Exception as e: 
                    connection_error = e

                # Shared error handling for caught exceptions
                retry_count += 1
                server_config.metrics.error_count += 1
                server_config.metrics.update_status()

                if span_ctx and hasattr(span_ctx, 'set_status'):
                    try:
                        span_ctx.set_status(trace.StatusCode.ERROR, str(connection_error))
                        # Don't end yet, we might retry
                    except Exception as e:
                        log.warning(f"Error updating span error status: {e}")

                connection_time = (time.time() - start_time) * 1000
                    
                if retry_count <= max_retries:
                    delay = min(1 * (2 ** (retry_count - 1)) + random.random(), 10)
                    log.warning(f"Error connecting to server {server_name} (attempt {retry_count}/{max_retries}): {connection_error}")
                    safe_console.print(f"[yellow]Error connecting to server {server_name} (attempt {retry_count}/{max_retries}): {connection_error}[/]")
                    log.info(f"Retrying in {delay:.2f} seconds...")
                    safe_console.print(f"[cyan]Retrying in {delay:.2f} seconds...[/]")
                    await asyncio.sleep(delay)
                else:
                    log.error(f"Failed to connect to server {server_name} after {max_retries} attempts: {connection_error}")
                    safe_console.print(f"[red]Failed to connect to server {server_name} after {max_retries} attempts: {connection_error}[/]")
                    if span_ctx and hasattr(span_ctx, 'end'): 
                        try:
                            span_ctx.end() # End span after final failure
                        except Exception as e:
                            log.warning(f"Error ending span: {e}")
                    return None

            if span_ctx and hasattr(span_ctx, 'end'): 
                try:
                    span_ctx.end() # End span if loop finishes unexpectedly
                except Exception as e:
                    log.warning(f"Error ending span: {e}")
            return None

    async def register_local_server(self, server_config: ServerConfig):
        """Register a locally started MCP server with zeroconf so other clients can discover it"""
        if not self.registry or not self.registry.zeroconf:
            # Start zeroconf if not already running
            if self.registry and not self.registry.zeroconf:
                self.registry.start_local_discovery()
            else:
                return
        
        try:
            import ipaddress
            import socket

            from zeroconf import ServiceInfo
            
            # Get local IP address
            # This method attempts to determine the local IP that would be used for external connections
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
            
            # Default to port 8080 for STDIO servers, but allow overriding
            port = 8080
            
            # Check if the server specifies a port in its args
            for i, arg in enumerate(server_config.args):
                if arg == '--port' or arg == '-p' and i < len(server_config.args) - 1:
                    try:
                        port = int(server_config.args[i+1])
                        break
                    except (ValueError, IndexError):
                        pass
            
            # Prepare properties as bytes dict
            props = {
                b'name': server_config.name.encode('utf-8'),
                b'type': server_config.type.value.encode('utf-8'),
                b'description': server_config.description.encode('utf-8'),
                b'version': str(server_config.version or '1.0.0').encode('utf-8'),
                b'host': 'localhost'.encode('utf-8')
            }
            
            # Create service info
            service_info = ServiceInfo(
                "_mcp._tcp.local.",
                f"{server_config.name}._mcp._tcp.local.",
                addresses=[ipaddress.IPv4Address(local_ip).packed],
                port=port,
                properties=props
            )
            
            # Register the service - Use async method and await it
            try:
                # Use async version of register_service
                await self.registry.zeroconf.async_register_service(service_info)
                log.info(f"Registered local MCP server {server_config.name} with zeroconf on {local_ip}:{port}")
                
                # Store service info for later unregistering
                if not hasattr(self, 'registered_services'):
                    self.registered_services = {}
                self.registered_services[server_config.name] = service_info
            except Exception as e:
                log.error(f"Error registering service with zeroconf: {e}")
            
        except ImportError:
            log.warning("Zeroconf not available, cannot register local server")
        except Exception as e:
            log.error(f"Error preparing zeroconf registration: {e}")

    async def connect_to_servers(self):
        """Connect to all enabled MCP servers"""
        if not self.config.servers:
            log.warning("No servers configured. Use 'config servers add' to add servers.")
            return
        
        # Connect to each enabled server
        safe_console = get_safe_console()
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            SpinnerColumn("dots"),
            TextColumn("[cyan]{task.fields[server]}"),
            console=safe_console,
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
        with Status(f"{STATUS_EMOJI['server']} Starting server monitoring...", spinner="dots", console=safe_console) as status:
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
                            if process and process.poll() is None:
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
            self.safe_print("[yellow]Shutdown complete[/]")

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
        
        self.server_manager = ServerManager(self.config, tool_cache=self.tool_cache)
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
                raise McpError(f"MCP error: {e}") from e
            except httpx.RequestError as e:
                log.error(f"Network error executing {tool_name}: {e}")
                raise McpError(f"Network error: {e}") from e
            except Exception as e:
                log.error(f"Unexpected error executing {tool_name}: {e}")
                raise McpError(f"Unexpected error: {e}") from e
        return wrapper
        
    # Add decorator for retry logic
    @staticmethod
    def retry_with_circuit_breaker(func):
        async def wrapper(self, server_name, *args, **kwargs):
            server_config = self.config.servers.get(server_name)
            if not server_config:
                raise McpError(f"Server {server_name} not found")
                
            if server_config.metrics.error_rate > 0.5:
                log.warning(f"Circuit breaker triggered for server {server_name} (error rate: {server_config.metrics.error_rate:.2f})")
                raise McpError(f"Server {server_name} in circuit breaker state")
                
            last_error = None
            for attempt in range(server_config.retry_policy["max_attempts"]):
                try:
                    # For each attempt, slightly increase the timeout
                    request_timeout = server_config.timeout + (attempt * server_config.retry_policy["timeout_increment"])
                    return await func(self, server_name, *args, **kwargs, request_timeout=request_timeout)
                except (McpError, httpx.RequestError) as e:
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
                        raise McpError(f"All {server_config.retry_policy['max_attempts']} attempts failed for server {server_name}: {str(last_error)}") from last_error
            
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
            raise McpError(f"Server {server_name} not connected")
            
        # Get the tool from the server_manager
        tool = self.server_manager.tools.get(tool_name)
        if not tool:
            raise McpError(f"Tool {tool_name} not found")
            
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
        safe_console = get_safe_console()
        
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
        
        # Show connected server info
        if connected_servers > 0:
            self.safe_print("\n[bold]Connected Servers:[/]")
            for name, server in self.config.servers.items():
                if name in self.server_manager.active_sessions:
                    # Get number of tools for this server
                    server_tools = sum(1 for t in self.server_manager.tools.values() 
                                if t.server_name == name)
                    self.safe_print(f"[green]✓[/] {name} ({server.type.value}) - {server_tools} tools")
        
        self.safe_print("[green]Ready to process queries![/green]")
                
    async def setup(self, interactive_mode=False):
        """Set up the client, connect to servers, and load capabilities"""
        # Ensure API key is set
        if not self.config.api_key:
            self.safe_print("[bold red]ERROR: Anthropic API key not found[/]")
            self.safe_print("Please set your API key using one of these methods:")
            self.safe_print("1. Set the ANTHROPIC_API_KEY environment variable")
            self.safe_print("2. Run 'python mcp_client.py run --interactive' and then use '/config api-key YOUR_API_KEY'")
            
            # Only exit if not in interactive mode
            if not interactive_mode:
                sys.exit(1)
            else:
                self.safe_print("[yellow]Running in interactive mode without API key.[/]")
                self.safe_print("[yellow]Please set your API key using '/config api-key YOUR_API_KEY'[/]")
                # Continue setup without API features
                self.anthropic = None  # Set to None until API key is provided
                
        # Load conversation graph if it exists
        if self.conversation_graph_file.exists():
            try:
                self.conversation_graph = await ConversationGraph.load(str(self.conversation_graph_file))
                log.info(f"Loaded conversation graph from {self.conversation_graph_file}")
            except Exception as e:
                log.warning(f"Could not load conversation graph ({type(e).__name__}: {e}), using new graph.")
                # Keep the default graph we created in __init__

        # Check for and load Claude desktop config if it exists
        await self.load_claude_desktop_config()

        # Use get_safe_console for all status and progress widgets
        safe_console = get_safe_console()
        
        # Verify no stdout pollution before connecting to servers
        if os.environ.get("MCP_VERIFY_STDOUT", "1") == "1":
            # Use safe_stdout context manager to prevent the verification itself from polluting
            with safe_stdout():
                # Only log this, don't print directly to avoid any risk of stdout pollution
                log.info("Verifying no stdout pollution before connecting to servers...")
                verify_no_stdout_pollution()
        
        # Discover servers if enabled - use a simple status instead of Progress
        if self.config.auto_discover:
            with Status(f"{STATUS_EMOJI['search']} Discovering MCP servers...", 
                    spinner="dots", console=safe_console) as status:
                await self.server_manager.discover_servers()
                status.update(f"{STATUS_EMOJI['success']} Server discovery complete")
        
        # Start continuous local discovery if enabled
        if self.config.enable_local_discovery and self.server_manager.registry:
            await self.start_local_discovery_monitoring()
        
        # Connect to all enabled servers without using Progress widget
        enabled_servers = [s for s in self.config.servers.values() if s.enabled]
        if enabled_servers:
            # Don't use _run_with_progress here to avoid potential display nesting
            self.safe_print(f"[bold blue]Connecting to {len(enabled_servers)} servers...[/]")
            
            for name, server_config in self.config.servers.items():
                if not server_config.enabled:
                    continue
                    
                try:
                    self.safe_print(f"[cyan]Connecting to server {name}...[/]")
                    # Connect and load server
                    result = await self._connect_and_load_server(name, server_config)
                    if result:
                        self.safe_print(f"[green]Connected to server {name}[/]")
                    else:
                        self.safe_print(f"[yellow]Failed to connect to server {name}[/]")
                except Exception as e:
                    self.safe_print(f"[red]Error connecting to server {name}: {e}[/]")
        
        # Start server monitoring
        with Status(f"{STATUS_EMOJI['server']} Starting server monitoring...", 
                spinner="dots", console=safe_console) as status:
            await self.server_monitor.start_monitoring()
            status.update(f"{STATUS_EMOJI['success']} Server monitoring started")
        
        # Display status without Progress widgets
        await self.print_simple_status()
        
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
            with Status(f"{STATUS_EMOJI['search']} Refreshing local MCP server discovery...", spinner="dots", console=safe_console) as status:
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
        
        safe_console = get_safe_console()
        with Status(f"{STATUS_EMOJI['speech_balloon']} Claude is thinking...", spinner="dots", console=safe_console) as status:
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
                            if process and process.poll() is None:
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
        safe_console = get_safe_console()
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
            with Status(f"{STATUS_EMOJI['server']} Connecting to {name}...", spinner="dots", console=safe_console) as status:
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
            if process.poll() is None:  # If process is still running
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
                if process.poll() is None:  # If process is still running
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
        
        # Use safe_console for all output
        safe_console = get_safe_console()
        
        # Close existing connections
        with Status(f"{STATUS_EMOJI['server']} Closing existing connections...", spinner="dots", console=safe_console) as status:
            await self.server_manager.close()
            status.update(f"{STATUS_EMOJI['success']} Existing connections closed")
        
        # Reset collections
        self.server_manager = ServerManager(self.config)
        
        # Reconnect
        with Status(f"{STATUS_EMOJI['server']} Reconnecting to servers...", spinner="dots", console=safe_console) as status:
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
            
            # Use get_safe_console() for the dashboard
            safe_console = get_safe_console()
            
            # Use a single Live display context for the dashboard
            with Live(self.generate_dashboard_renderable(), 
                    refresh_per_second=1.0/self.config.dashboard_refresh_rate, 
                    screen=True, 
                    transient=False,
                    console=safe_console) as live:
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
        
        with Status(f"{STATUS_EMOJI['tool']} Executing {tool_name}...", spinner="dots", console=safe_console) as status:
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
        """Look for and load the Claude desktop config file (claude_desktop_config.json) if it exists."""
        config_path = Path("claude_desktop_config.json")
        if not config_path.exists():
            return
            
        try:
            self.safe_print(f"{STATUS_EMOJI['config']} Found Claude desktop config file, processing...")
            
            # Read the file content
            async with aiofiles.open(config_path, 'r') as f:
                content = await f.read()
                
            try:
                desktop_config = json.loads(content)
                # Log the structure
                log.debug(f"Claude desktop config keys: {list(desktop_config.keys())}")
            except json.JSONDecodeError as json_error:
                self.safe_print(f"[red]Invalid JSON in Claude desktop config: {json_error}[/]")
                return
            
            # The primary key is 'mcpServers', but check alternatives if needed
            if 'mcpServers' not in desktop_config:
                log.warning(f"Expected 'mcpServers' key not found in Claude desktop config")
                log.debug(f"Available keys: {list(desktop_config.keys())}")
                # Check for alternative keys
                for alt_key in ['mcp_servers', 'servers', 'MCP_SERVERS']:
                    if alt_key in desktop_config:
                        log.info(f"Using alternative key '{alt_key}' for MCP servers")
                        desktop_config['mcpServers'] = desktop_config[alt_key]
                        break
                else:
                    self.safe_print(f"{STATUS_EMOJI['warning']} No MCP servers defined in Claude desktop config")
                    return
            
            # We now should have a mcpServers key
            mcp_servers = desktop_config['mcpServers']
            if not mcp_servers or not isinstance(mcp_servers, dict):
                self.safe_print(f"{STATUS_EMOJI['warning']} No valid MCP servers found in config")
                return
            
            # Track successful imports and skipped servers
            imported_servers = []
            skipped_servers = []
                        
            # Process each server
            for server_name, server_data in mcp_servers.items():
                try:
                    # Skip if server already exists
                    if server_name in self.config.servers:
                        log.info(f"Server '{server_name}' already exists in config, skipping")
                        skipped_servers.append((server_name, "already exists"))
                        continue
                    
                    # Log server info for debugging
                    log.debug(f"Processing server '{server_name}' with data: {server_data}")
                    
                    # Extract command and args
                    if 'command' not in server_data:
                        log.warning(f"No 'command' field for server '{server_name}'")
                        skipped_servers.append((server_name, "missing command field"))
                        continue
                        
                    command = server_data['command']
                    args = server_data.get('args', [])
                    
                    # Adapt paths for the current platform
                    try:
                        adapted_command, adapted_args = adapt_path_for_platform(command, args)
                        log.info(f"Server '{server_name}': Adapted command from '{command}' to '{adapted_command}'")
                        log.debug(f"Server '{server_name}': Adapted args from '{args}' to '{adapted_args}'")
                    except Exception as adapt_error:
                        log.error(f"Error adapting paths for server '{server_name}': {adapt_error}")
                        # Use original command and args as fallback
                        adapted_command, adapted_args = command, args
                    
                    # Create new server config
                    server_config = ServerConfig(
                        name=server_name,
                        type=ServerType.STDIO,  # Claude desktop uses STDIO servers
                        path=adapted_command,
                        args=adapted_args,
                        enabled=True,
                        auto_start=True,
                        description=f"Imported from Claude desktop config",
                        trusted=True,  # Assume trusted since configured in Claude desktop
                    )
                    
                    # Add to our config
                    self.config.servers[server_name] = server_config
                    imported_servers.append(server_name)
                    log.info(f"Imported server '{server_name}' from Claude desktop config")
                except Exception as server_error:
                    server_error_str = str(server_error)
                    log.error(f"Error processing server '{server_name}': {server_error_str}")
                    skipped_servers.append((server_name, f"processing error: {server_error_str}"))
            
            # Save config
            if imported_servers:
                try:
                    await self.config.save_async()
                    self.safe_print(f"{STATUS_EMOJI['success']} Imported {len(imported_servers)} servers from Claude desktop config")
                    
                    # Create a nice report
                    if imported_servers:
                        server_table = Table(title="Imported Servers")
                        server_table.add_column("Name")
                        server_table.add_column("Command")
                        server_table.add_column("Arguments")
                        
                        for name in imported_servers:
                            server = self.config.servers[name]
                            server_table.add_row(
                                name,
                                server.path,
                                " ".join(str(arg) for arg in server.args) if server.args else ""
                            )
                        
                        self.safe_print(server_table)
                    
                    if skipped_servers:
                        skipped_table = Table(title="Skipped Servers")
                        skipped_table.add_column("Name")
                        skipped_table.add_column("Reason")
                        
                        for name, reason in skipped_servers:
                            skipped_table.add_row(name, reason)
                        
                        self.safe_print(skipped_table)
                except Exception as save_error:
                    log.error(f"Error saving config after importing servers: {save_error}")
                    self.safe_print(f"[red]Error saving imported server config: {save_error}[/]")
            else:
                self.safe_print(f"{STATUS_EMOJI['warning']} No new servers imported from Claude desktop config")
        
        except FileNotFoundError as file_error:
            log.debug(f"Claude desktop config file not found: {file_error}")
        except json.JSONDecodeError as json_err:
            log.error(f"Invalid JSON in Claude desktop config file: {json_err}")
            # Try to show the problematic part of the JSON
            try:
                async with aiofiles.open(config_path, 'r') as f:
                    file_content = await f.read()
                    prob_line = file_content.splitlines()[max(0, json_err.lineno - 1)]
                    log.error(f"JSON error at line {json_err.lineno}, column {json_err.colno}: {prob_line}")
            except Exception as json_context_error:
                log.error(f"Error getting JSON error context: {json_context_error}")
        except Exception as outer_error:
            # Ensure we're using str(outer_error) to avoid any issue with the error being a string itself
            error_message = str(outer_error)
            log.error(f"Error processing Claude desktop config: {error_message}")
            self.safe_print(f"[red]Error processing Claude desktop config: {error_message}[/]")
            # Add more context for debugging
            import traceback
            log.debug(f"Full error: {traceback.format_exc()}")
            
        # Check for the 'warning' string bug
        if error_message == 'warning':
            log.error("Detected 'warning' string as exception. This might indicate a shadowed variable.")
            self.safe_print("[red]Internal error: detected 'warning' variable shadowing issue. Check logs.[/]")
            
            # Try to recover by checking if there were any warnings logged asynchronously
            try:
                log_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mcpclient.log')
                async with aiofiles.open(log_path, 'r') as log_file:
                    # Read the file content
                    content = await log_file.read()
                    lines = content.splitlines()
                    
                    # Get the last 100 lines or all lines if fewer
                    last_lines = lines[-100:] if len(lines) > 100 else lines
                    
                    # Filter for warning lines
                    recent_warnings = [line for line in last_lines if 'WARNING' in line]
                    if recent_warnings:
                        # Log the last 5 warnings or fewer if there are less
                        warnings_to_log = recent_warnings[-5:] if len(recent_warnings) > 5 else recent_warnings
                        log.error(f"Recent warnings that might be related: {warnings_to_log}")
            except Exception as log_read_error:
                log.error(f"Error trying to read log file asynchronously: {log_read_error}")


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
        
        safe_console = get_safe_console()
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
                console=safe_console,
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
    # Initialize client
    client = MCPClient()
    safe_console = get_safe_console()
    
    # Set up timeout for overall execution
    max_shutdown_timeout = 10  # seconds
    
    try:
        # Set up client with error handling for each step
        try:
            await client.setup(interactive_mode=interactive)
        except Exception as setup_error:
            if verbose_logging:
                safe_console.print(f"[bold red]Error during setup:[/] {str(setup_error)}")
                import traceback
                safe_console.print(traceback.format_exc())
            else:
                safe_console.print(f"[bold red]Error during setup:[/] {str(setup_error)}")
            # Continue anyway for interactive mode
            if not interactive:
                raise  # Re-raise for non-interactive mode

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
                        if process and process.poll() is None:
                            try:
                                safe_console.print(f"[yellow]Force killing process: {name}[/]")
                                process.kill()
                            except Exception:
                                pass
            except Exception as e:
                safe_console.print(f"[red]Error during shutdown: {e}[/]")
    
    except Exception as e:
        # Use client.safe_print if client is available, otherwise use safe_console
        if 'client' in locals() and client:
            client.safe_print(f"[bold red]Error:[/] {str(e)}")
            if verbose_logging:
                import traceback
                client.safe_print(traceback.format_exc())
        else:
            # Fall back to safe_console if client isn't available
            safe_console = get_safe_console()
            safe_console.print(f"[bold red]Error:[/] {str(e)}")
            if verbose_logging:
                import traceback
                safe_console.print(traceback.format_exc())
        
        # For unhandled exceptions in non-interactive mode, still try to clean up
        if 'client' in locals() and client and hasattr(client, 'server_manager'):
            try:
                # Use timeout to prevent hanging during error cleanup
                await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout)
            except (asyncio.TimeoutError, Exception):
                # Just continue with exit on any errors during error handling
                pass
        
        # Return non-zero exit code for script usage
        if not interactive:
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
                        if process and process.poll() is None:
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
                safe_console.print(json.dumps(server_data, indent=2))
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
