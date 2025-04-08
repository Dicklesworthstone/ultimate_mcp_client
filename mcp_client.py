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
#     "aiofiles>=23.2.0"
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
import functools
import hashlib
import json
import logging
import os
import platform
import random
import readline
import socket
import subprocess
import sys
import time
import uuid
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
log = logging.getLogger("mcpclient")

# Initialize Rich console with theme
console = Console(theme=custom_theme)

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
    "https://mcp-registry.anthropic.com",
    "https://registry.modelcontextprotocol.io"
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
                    params=params
                )
                
                if response.status_code == 200:
                    servers = response.json().get("servers", [])
                    for server in servers:
                        server["registry_url"] = registry_url
                        all_servers.append(server)
                else:
                    log.warning(f"Failed to get servers from {registry_url}: {response.status_code}")
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
    
    def save(self, file_path: str):
        """Save the conversation graph to file"""
        data = {
            "current_node_id": self.current_node.id,
            "nodes": {
                node_id: node.to_dict()
                for node_id, node in self.nodes.items()
            }
        }
        
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        except IOError as e:
             log.error(f"Could not write conversation graph to {file_path}: {e}")
        except TypeError as e: # Handle potential issues with non-serializable data
             log.error(f"Could not serialize conversation graph: {e}")
    
    @classmethod
    def load(cls, file_path: str) -> "ConversationGraph":
        """Load a conversation graph from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            log.warning(f"Conversation graph file not found: {file_path}")
            raise # Re-raise to be handled by the caller (__init__)
        except IOError as e:
            log.error(f"Could not read conversation graph file {file_path}: {e}")
            raise # Re-raise to be handled by the caller (__init__)
        except json.JSONDecodeError as e:
            log.error(f"Error decoding conversation graph JSON from {file_path}: {e}")
            raise # Re-raise to be handled by the caller (__init__)

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
        
        self.load()
    
    def load(self):
        """Load configuration from file"""
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
        """Save configuration to file"""
        config_data = {
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


class History:
    def __init__(self, max_entries=MAX_HISTORY_ENTRIES):
        self.entries = deque(maxlen=max_entries)
        self.max_entries = max_entries
        self.load()
    
    def add(self, entry: ChatHistory):
        """Add a new entry to history"""
        self.entries.append(entry)
        self.save()
    
    def load(self):
        """Load history from file"""
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
    
    def save(self):
        """Save history to file"""
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
            console.print(f"\n[bold green]Discovered {total_discovered} potential MCP servers:[/]")
            
            if discovered_local:
                console.print("\n[bold blue]Local File System:[/]")
                for i, (name, path, server_type) in enumerate(discovered_local, 1):
                    console.print(f"{i}. [bold]{name}[/] ({server_type}) - {path}")
            
            if discovered_remote:
                console.print("\n[bold magenta]Remote Registry:[/]")
                for i, (name, url, server_type, version, categories, rating) in enumerate(discovered_remote, 1):
                    version_str = f"v{version}" if version else "unknown version"
                    categories_str = ", ".join(categories) if categories else "no categories"
                    console.print(f"{i}. [bold]{name}[/] ({server_type}) - {url} - {version_str} - Rating: {rating:.1f}/5.0 - {categories_str}")
            
            if discovered_mdns:
                console.print("\n[bold cyan]Local Network (mDNS):[/]")
                for i, (name, url, server_type, version, categories, description) in enumerate(discovered_mdns, 1):
                    version_str = f"v{version}" if version else "unknown version"
                    categories_str = ", ".join(categories) if categories else "no categories"
                    desc_str = f" - {description}" if description else ""
                    console.print(f"{i}. [bold]{name}[/] ({server_type}) - {url} - {version_str} - {categories_str}{desc_str}")
            
            # Ask user which ones to add
            if Confirm.ask("\nAdd discovered servers to configuration?"):
                # Create selection interface
                selections = []
                
                if discovered_local:
                    console.print("\n[bold blue]Local File System Servers:[/]")
                    for i, (name, path, server_type) in enumerate(discovered_local, 1):
                        if Confirm.ask(f"Add {name} ({path})?"):
                            selections.append(("local", i-1))
                
                if discovered_remote:
                    console.print("\n[bold magenta]Remote Registry Servers:[/]")
                    for i, (name, url, server_type, version, categories, rating) in enumerate(discovered_remote, 1):
                        if Confirm.ask(f"Add {name} ({url})?"):
                            selections.append(("remote", i-1))
                
                if discovered_mdns:
                    console.print("\n[bold cyan]Local Network Servers:[/]")
                    for i, (name, url, server_type, version, categories, description) in enumerate(discovered_mdns, 1):
                        if Confirm.ask(f"Add {name} ({url})?"):
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
                console.print("[green]Selected servers added to configuration[/]")
        else:
            console.print("[yellow]No new servers discovered.[/]")

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
            console.print("[yellow]Server discovery is disabled in configuration.[/]")
    
    async def connect_to_server(self, server_config: ServerConfig) -> Optional[ClientSession]:
        """Connect to a single MCP server with retry logic and health monitoring"""
        server_name = server_config.name
        retry_count = 0
        max_retries = server_config.retry_count
        
        while retry_count <= max_retries:
            # Track metrics for this connection attempt
            start_time = time.time()
            
            try:
                # Create span for observability if available
                span_ctx = None
                if tracer:
                    span_ctx = tracer.start_as_current_span(
                        f"connect_server.{server_name}",
                        attributes={
                            "server.name": server_name,
                            "server.type": server_config.type.value,
                            "server.path": server_config.path,
                            "retry": retry_count
                        }
                    )
                
                if server_config.type == ServerType.STDIO:
                    # Check if we need to start the server process
                    if server_config.auto_start and server_config.path:
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
                        
                        # Create process with pipes and set resource limits
                        process = await asyncio.create_subprocess_exec(
                            *cmd,
                            stdin=asyncio.subprocess.PIPE,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.PIPE,
                        )
                        
                        # Read stderr in a separate thread/task to prevent blocking? (More complex change)
                        # For now, just capture it. Check poll() status after connect attempt fails.
                        
                        self.processes[server_config.name] = process
                        
                        # Register the server with zeroconf if local discovery is enabled
                        if self.config.enable_local_discovery and self.registry:
                            await self.register_local_server(server_config)
                    
                    # Set up parameters with timeout
                    params = StdioServerParameters(
                        command=server_config.path, 
                        args=server_config.args,
                        timeout=server_config.timeout
                    )
                    
                    # Create client with context manager to ensure proper cleanup
                    session = await self.exit_stack.enter_async_context(await stdio_client(params))
                    
                elif server_config.type == ServerType.SSE:
                    # Connect to SSE server using direct parameters
                    # (sse_client takes url, headers, timeout, sse_read_timeout parameters)
                    session = await self.exit_stack.enter_async_context(
                        await sse_client(
                            url=server_config.path,
                            timeout=server_config.timeout,
                            sse_read_timeout=server_config.timeout * 12  # Set longer timeout for events
                        )
                    )
                else:
                    if span_ctx:
                        span_ctx.set_status(trace.StatusCode.ERROR, f"Unknown server type: {server_config.type}")
                    log.error(f"Unknown server type: {server_config.type}")
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
                if span_ctx:
                    span_ctx.set_status(trace.StatusCode.OK)
                    span_ctx.end()
                
                log.info(f"Connected to server {server_name} in {connection_time:.2f}ms")
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
                         stderr_output = proc.stderr.read() if proc.stderr else "stderr not captured"
                         log.error(f"STDIO server process for '{server_config.name}' exited with code {proc.returncode}. Stderr: {stderr_output}")
            except OSError as e: # OS level errors (e.g., command not found)
                 connection_error = e
            # Keep broad exception for truly unexpected connection issues
            except Exception as e: 
                 connection_error = e

            # Shared error handling for caught exceptions
            retry_count += 1
            server_config.metrics.error_count += 1
            server_config.metrics.update_status()

            if span_ctx:
                span_ctx.set_status(trace.StatusCode.ERROR, str(connection_error))
                # span_ctx.end() # Don't end yet, we might retry

            connection_time = (time.time() - start_time) * 1000
                
            if retry_count <= max_retries:
                delay = min(1 * (2 ** (retry_count - 1)) + random.random(), 10)
                log.warning(f"Error connecting to server {server_name} (attempt {retry_count}/{max_retries}): {connection_error}")
                log.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            else:
                log.error(f"Failed to connect to server {server_name} after {max_retries} attempts: {connection_error}")
                if span_ctx: span_ctx.end() # End span after final failure
                return None

        if span_ctx: span_ctx.end() # End span if loop finishes unexpectedly
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
            # In a real implementation, we'd need to determine the actual port
            # the server is listening on, which might require modification of the server
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
            
            # Register the service
            self.registry.zeroconf.register_service(service_info)
            log.info(f"Registered local MCP server {server_config.name} with zeroconf on {local_ip}:{port}")
            
            # Store service info for later unregistering
            if not hasattr(self, 'registered_services'):
                self.registered_services = {}
            self.registered_services[server_config.name] = service_info
            
        except ImportError:
            log.warning("Zeroconf not available, cannot register local server")
        except Exception as e:
            log.error(f"Error registering server with zeroconf: {e}")

    async def connect_to_servers(self):
        """Connect to all enabled MCP servers"""
        if not self.config.servers:
            log.warning("No servers configured. Use 'config servers add' to add servers.")
            return
        
        # Connect to each enabled server
        with Progress() as progress:
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
    
    async def load_server_capabilities(self, server_name: str, session: ClientSession):
        """Load tools, resources, and prompts from a server"""
        try:
            # Load tools
            tool_response = await session.list_tools()
            for tool in tool_response.tools:
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
            
            # Load resources
            try:
                resource_response = await session.list_resources()
                for resource in resource_response.resources:
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
            # Keep broad exception for unexpected issues listing resources
            except Exception as e: 
                 log.error(f"Unexpected error listing resources from {server_name}: {e}")
            
            # Load prompts
            try:
                prompt_response = await session.list_prompts()
                for prompt in prompt_response.prompts:
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
            # Keep broad exception for unexpected issues listing prompts
            except Exception as e: 
                 log.error(f"Unexpected error listing prompts from {server_name}: {e}")
                
        except McpError as e: # Catch specific MCP errors first
             log.error(f"MCP error loading capabilities from server {server_name}: {e}")
        except httpx.RequestError as e: # Catch network errors if using SSE
             log.error(f"Network error loading capabilities from server {server_name}: {e}")
        # Keep broad exception for other unexpected issues
        except Exception as e: 
            log.error(f"Unexpected error loading capabilities from server {server_name}: {e}")
    
    async def close(self):
        """Close all server connections and processes"""
        # Unregister zeroconf services if any
        if hasattr(self, 'registered_services') and self.registry and self.registry.zeroconf:
            for name, service_info in self.registered_services.items():
                try:
                    self.registry.zeroconf.unregister_service(service_info)
                    log.info(f"Unregistered zeroconf service for {name}")
                except Exception as e:
                    log.error(f"Error unregistering zeroconf service for {name}: {e}")
        
        # Close all sessions
        await self.exit_stack.aclose()
        
        # Terminate all processes
        for name, process in self.processes.items():
            try:
                if process.poll() is None:  # If process is still running
                    log.info(f"Terminating server process: {name}")
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                         log.warning(f"Process {name} did not terminate gracefully, killing.")
                         process.kill()
                         process.wait() # Wait for kill
            except OSError as e:
                log.error(f"OS error terminating process {name} (PID {process.pid if process else 'unknown'}): {e}")
            # Keep broad exception for other unexpected termination issues
            except Exception as e: 
                log.error(f"Unexpected error terminating process {name}: {e}")
                
                # Force kill as a last resort
                try:
                    if process.poll() is None:
                        process.kill()
                except Exception:
                    pass

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
            
        progress_columns = []
        if show_spinner:
            progress_columns.append(SpinnerColumn())
        
        progress_columns.extend([
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[cyan]{task.completed}/{task.total}"),
            TaskProgressColumn()
        ])
        
        with Progress(*progress_columns, console=console) as progress:
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
        
        # Instantiate Caching
        self.tool_cache = ToolCache(
            cache_dir=CACHE_DIR,
            custom_ttl_mapping=self.config.cache_ttl_mapping
        )
        
        self.server_manager = ServerManager(self.config, tool_cache=self.tool_cache)
        self.anthropic = AsyncAnthropic(api_key=self.config.api_key)  # Changed to AsyncAnthropic
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
            self.conversation_graph = ConversationGraph.load(str(self.conversation_graph_file))
            log.info(f"Loaded conversation graph from {self.conversation_graph_file}")
        except (FileNotFoundError, IOError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e: # Catch specific load errors
            log.warning(f"Could not load conversation graph ({type(e).__name__}: {e}), starting new graph.")
            self.conversation_graph = ConversationGraph()
        # Keep broad exception for truly unexpected graph loading issues
        except Exception as e: 
            log.error(f"Unexpected error initializing conversation graph: {e}")
            self.conversation_graph = ConversationGraph() # Fallback to new graph
        
        # Ensure current node is valid after loading
        if not self.conversation_graph.get_node(self.conversation_graph.current_node.id):
            log.warning("Loaded current node ID not found in graph, resetting to root.")
            self.conversation_graph.set_current_node("root")

        # For storing conversation context (Now managed by ConversationGraph)
        # self.conversation_messages = [] 

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
        
    async def setup(self):
        """Set up the client, connect to servers, and load capabilities"""
        # Ensure API key is set
        if not self.config.api_key:
            console.print("[bold red]ERROR: Anthropic API key not found[/]")
            console.print("Please set your API key using one of these methods:")
            console.print("1. Set the ANTHROPIC_API_KEY environment variable")
            console.print("2. Run 'config api-key YOUR_API_KEY'")
            sys.exit(1)
        
        # Check for and load Claude desktop config if it exists
        await self.load_claude_desktop_config()
        
        # Discover servers if enabled
        if self.config.auto_discover:
            with Status(f"{STATUS_EMOJI['search']} Discovering MCP servers...", spinner="dots") as status:
                await self.server_manager.discover_servers()
                status.update(f"{STATUS_EMOJI['success']} Server discovery complete")
        
        # Start continuous local discovery if enabled
        if self.config.enable_local_discovery and self.server_manager.registry:
            await self.start_local_discovery_monitoring()
        
        # Connect to all enabled servers - Use Progress instead of Status for better visual tracking
        enabled_servers = [s for s in self.config.servers.values() if s.enabled]
        if enabled_servers:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                SpinnerColumn("dots"),
                TextColumn("[cyan]{task.fields[server]}"),
                console=console,
                transient=True
            ) as progress:
                connect_task = progress.add_task(f"{STATUS_EMOJI['server']} Connecting to servers...", total=len(enabled_servers), server="")
                
                for name, server_config in self.config.servers.items():
                    if not server_config.enabled:
                        continue
                    
                    progress.update(connect_task, description=f"{STATUS_EMOJI['server']} Connecting to server...", server=name)
                    session = await self.server_manager.connect_to_server(server_config)
                    
                    if session:
                        self.server_manager.active_sessions[name] = session
                        progress.update(connect_task, description=f"{STATUS_EMOJI['connected']} Loading capabilities...", server=name)
                        await self.server_manager.load_server_capabilities(name, session)
                    else:
                        progress.update(connect_task, description=f"{STATUS_EMOJI['disconnected']} Connection failed", server=name)
                    
                    progress.update(connect_task, advance=1)
                
                progress.update(connect_task, description=f"{STATUS_EMOJI['success']} Server connections complete")
        
        # Start server monitoring
        with Status(f"{STATUS_EMOJI['server']} Starting server monitoring...", spinner="dots") as status:
            await self.server_monitor.start_monitoring()
            status.update(f"{STATUS_EMOJI['success']} Server monitoring started")
        
        # Display status
        await self.print_status()

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
                        console.print(f"\n[bold cyan]{STATUS_EMOJI['search']} New MCP servers discovered on local network:[/]")
                        for server_name in new_servers:
                            server_info = self.server_manager.registry.discovered_servers[server_name]
                            console.print(f"  - [bold cyan]{server_name}[/] at [cyan]{server_info.get('url', 'unknown URL')}[/]")
                        console.print("Use [bold cyan]/discover list[/] to view details and [bold cyan]/discover connect NAME[/] to connect")
                        
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
        if not self.server_manager.registry:
            console.print("[yellow]Registry not available, local discovery is disabled.[/]")
            return
            
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"
        subargs = parts[1] if len(parts) > 1 else ""
        
        if subcmd == "list":
            # List all discovered servers
            discovered_servers = self.server_manager.registry.discovered_servers
            
            if not discovered_servers:
                console.print("[yellow]No MCP servers discovered on local network.[/]")
                console.print("Try running [bold blue]/discover refresh[/] to scan again.")
                return
                
            console.print(f"\n[bold cyan]{STATUS_EMOJI['search']} Discovered Local Network Servers:[/]")
            
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
                
                server_table.add_row(name, url, server_type, description, status)
            
            console.print(server_table)
            console.print("\nUse [bold blue]/discover connect NAME[/] to connect to a server.")
            
        elif subcmd == "connect":
            if not subargs:
                console.print("[yellow]Usage: /discover connect SERVER_NAME[/]")
                return
                
            server_name = subargs
            
            # Check if server exists in discovered servers
            if server_name not in self.server_manager.registry.discovered_servers:
                console.print(f"[red]Server '{server_name}' not found in discovered servers.[/]")
                console.print("Use [bold blue]/discover list[/] to see available servers.")
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
                console.print(f"[yellow]Server with URL '{url}' already exists as '{existing_server}'.[/]")
                if existing_server not in self.server_manager.active_sessions:
                    if Confirm.ask(f"Connect to existing server '{existing_server}'?"):
                        await self.connect_server(existing_server)
                else:
                    console.print(f"[yellow]Server '{existing_server}' is already connected.[/]")
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
            console.print(f"[green]Added server '{server_name}' to configuration.[/]")
            
            # Offer to connect
            if Confirm.ask(f"Connect to server '{server_name}' now?"):
                await self.connect_server(server_name)
                
        elif subcmd == "refresh":
            # Force a refresh of the discovery
            with Status(f"{STATUS_EMOJI['search']} Refreshing local MCP server discovery...", spinner="dots") as status:
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
                    console.print(f"\n[bold cyan]Found {len(current_servers)} servers on the local network[/]")
                    console.print("Use [bold blue]/discover list[/] to see details.")
                else:
                    console.print("[yellow]No servers found on the local network.[/]")
                    
        elif subcmd == "auto":
            # Enable/disable automatic discovery
            if subargs.lower() in ("on", "yes", "true", "1"):
                self.config.enable_local_discovery = True
                self.config.save()
                console.print("[green]Automatic local discovery enabled.[/]")
                
                # Start discovery if not already running
                if not self.local_discovery_task:
                    await self.start_local_discovery_monitoring()
                    
            elif subargs.lower() in ("off", "no", "false", "0"):
                self.config.enable_local_discovery = False
                self.config.save()
                console.print("[yellow]Automatic local discovery disabled.[/]")
                
                # Stop discovery if running
                await self.stop_local_discovery_monitoring()
                
            else:
                # Show current status
                status = "enabled" if self.config.enable_local_discovery else "disabled"
                console.print(f"[cyan]Automatic local discovery is currently {status}.[/]")
                console.print("Usage: [bold blue]/discover auto [on|off][/]")
                
        else:
            console.print("[yellow]Unknown discover command. Available: list, connect, refresh, auto[/]")

    async def close(self):
        """Clean up resources before exit"""
        # Stop local discovery monitoring if running
        if self.local_discovery_task:
            await self.stop_local_discovery_monitoring()
            
        # Save conversation graph
        try:
            self.conversation_graph.save(str(self.conversation_graph_file))
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
    
    async def print_status(self):
        """Print client status summary"""
        console.print("\n[bold]MCP Client Status[/]")
        console.print(f"Model: [cyan]{self.current_model}[/]")
        
        # Server connections
        server_table = Table(title="Connected Servers")
        server_table.add_column("Name")
        server_table.add_column("Type")
        server_table.add_column("Tools")
        server_table.add_column("Resources")
        server_table.add_column("Prompts")
        
        for name, session in self.server_manager.active_sessions.items():
            server_config = self.config.servers.get(name)
            if not server_config:
                continue
                
            # Count capabilities from this server
            tools_count = sum(1 for t in self.server_manager.tools.values() if t.server_name == name)
            resources_count = sum(1 for r in self.server_manager.resources.values() if r.server_name == name)
            prompts_count = sum(1 for p in self.server_manager.prompts.values() if p.server_name == name)
            
            server_table.add_row(
                name,
                server_config.type.value,
                str(tools_count),
                str(resources_count),
                str(prompts_count)
            )
        
        console.print(server_table)
        
        # Tool summary
        if self.server_manager.tools:
            console.print(f"\nAvailable tools: [green]{len(self.server_manager.tools)}[/]")
            console.print(f"Available resources: [green]{len(self.server_manager.resources)}[/]")
            console.print(f"Available prompts: [green]{len(self.server_manager.prompts)}[/]")
        else:
            console.print("\n[yellow]No tools available. Connect to MCP servers to access tools.[/]")
    
    @asynccontextmanager
    async def tool_execution_context(self, tool_name: str, tool_args: dict, server_name: str):
        """Context manager for tool execution with metrics and tracing.
        
        This centralizes all the tool execution boilerplate like tracing,
        metrics collection, and error handling.
        
        Args:
            tool_name: Name of the tool being executed
            tool_args: Arguments for the tool
            server_name: Name of the server running the tool
            
        Yields:
            The execution time in milliseconds on successful completion
        """
        tool_start_time = time.time()
        tool_span = None
        
        if tracer:
            tool_span = tracer.start_as_current_span(
                f"tool.{tool_name}",
                attributes={
                    "tool.name": tool_name,
                    "server.name": server_name,
                    "params": str(tool_args)
                }
            )
        
        try:
            # We yield a placeholder here - we'll calculate and track the execution time after the yield
            yield None
            
            # Calculate execution time
            tool_execution_time = (time.time() - tool_start_time) * 1000  # ms
            
            # Update tool metrics in tool collection
            tool = self.server_manager.tools.get(tool_name)
            if tool:
                tool.update_execution_time(tool_execution_time)
            
            # Record metrics if available
            if tool_execution_counter:
                tool_execution_counter.add(
                    1, 
                    {"tool": tool_name, "server": server_name, "success": "true"}
                )
            
            if latency_histogram:
                latency_histogram.record(
                    tool_execution_time,
                    {"operation": "tool_execution", "tool": tool_name, "server": server_name}
                )
            
            # End span if tracking with success status
            if tool_span:
                tool_span.set_status(trace.StatusCode.OK)
            
        except Exception as e:
            # Record error metrics
            if tool_execution_counter:
                tool_execution_counter.add(
                    1, 
                    {"tool": tool_name, "server": server_name, "success": "false"}
                )
            
            # End span with error status
            if tool_span:
                tool_span.set_status(trace.StatusCode.ERROR, str(e))
                
            # Re-raise the exception
            raise
            
        finally:
            # Always end the span if it exists
            if tool_span:
                tool_span.end()
    
    async def process_streaming_query(self, query: str, model: Optional[str] = None, 
                               max_tokens: Optional[int] = None) -> AsyncIterator[str]:
        """Process a query using Claude and available tools with streaming"""
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
        
        with Status(f"{STATUS_EMOJI['speech_balloon']} Claude is thinking...", spinner="dots") as status:
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
                # self.conversation_messages = messages + [{"role": "assistant", "content": assistant_message}]
                # Update the current node in the graph instead
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
        console.print("\n[bold green]MCP Client Interactive Mode[/]")
        console.print("Type your query to Claude, or a command (type 'help' for available commands)")
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold blue]>>[/]")
                
                # Check if it's a command
                if user_input.startswith('/'):
                    cmd_parts = user_input[1:].split(maxsplit=1)
                    cmd = cmd_parts[0].lower()
                    args = cmd_parts[1] if len(cmd_parts) > 1 else ""
                    
                    if cmd in self.commands:
                        await self.commands[cmd](args)
                    else:
                        console.print(f"[yellow]Unknown command: {cmd}[/]")
                        console.print("Type '/help' for available commands")
                
                # Empty input
                elif not user_input.strip():
                    continue
                
                # Process as a query to Claude
                else:
                    result = await self.process_query(user_input)
                    console.print()
                    console.print(Panel.fit(
                        Markdown(result),
                        title="Claude",
                        border_style="green"
                    ))
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted[/]")
                break
            # Catch specific errors related to command execution or query processing
            except (anthropic.APIError, McpError, httpx.RequestError) as e: 
                console.print(f"[bold red]Error ({type(e).__name__}):[/] {str(e)}")
            # Keep broad exception for unexpected loop issues
            except Exception as e: 
                console.print(f"[bold red]Unexpected Error:[/] {str(e)}")
    
    async def cmd_exit(self, args):
        """Exit the client"""
        console.print("[yellow]Exiting...[/]")
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
        console.print("\n[bold]Available Commands:[/]")
        
        console.print(Panel(
            Group(*general_commands),
            title="General Commands",
            border_style="blue"
        ))
        
        console.print(Panel(
            Group(*config_commands),
            title="Configuration Commands",
            border_style="cyan"
        ))
        
        console.print(Panel(
            Group(*server_commands),
            title="Server & Tools Commands",
            border_style="magenta"
        ))
        
        console.print(Panel(
            Group(*conversation_commands),
            title="Conversation Commands",
            border_style="green"
        ))
        
        console.print(Panel(
            Group(*monitoring_commands),
            title="Monitoring Commands",
            border_style="yellow"
        ))
    
    async def cmd_config(self, args):
        """Handle configuration commands"""
        if not args:
            # Show current config
            console.print("\n[bold]Current Configuration:[/]")
            console.print(f"API Key: {'*' * 8 + self.config.api_key[-4:] if self.config.api_key else 'Not set'}")
            console.print(f"Default Model: {self.config.default_model}")
            console.print(f"Max Tokens: {self.config.default_max_tokens}")
            console.print(f"History Size: {self.config.history_size}")
            console.print(f"Auto-Discovery: {'Enabled' if self.config.auto_discover else 'Disabled'}")
            console.print(f"Discovery Paths: {', '.join(self.config.discovery_paths)}")
            return
        
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower()
        subargs = parts[1] if len(parts) > 1 else ""
        
        if subcmd == "api-key":
            if not subargs:
                console.print("[yellow]Usage: /config api-key YOUR_API_KEY[/]")
                return
                
            self.config.api_key = subargs
            self.anthropic = AsyncAnthropic(api_key=self.config.api_key)  # Changed to AsyncAnthropic
            self.config.save()
            console.print("[green]API key updated[/]")
            
        elif subcmd == "model":
            if not subargs:
                console.print("[yellow]Usage: /config model MODEL_NAME[/]")
                return
                
            self.config.default_model = subargs
            self.current_model = subargs
            self.config.save()
            console.print(f"[green]Default model updated to {subargs}[/]")
            
        elif subcmd == "max-tokens":
            if not subargs or not subargs.isdigit():
                console.print("[yellow]Usage: /config max-tokens NUMBER[/]")
                return
                
            self.config.default_max_tokens = int(subargs)
            self.config.save()
            console.print(f"[green]Default max tokens updated to {subargs}[/]")
            
        elif subcmd == "history-size":
            if not subargs or not subargs.isdigit():
                console.print("[yellow]Usage: /config history-size NUMBER[/]")
                return
                
            self.config.history_size = int(subargs)
            self.history.max_entries = int(subargs)
            self.config.save()
            console.print(f"[green]History size updated to {subargs}[/]")
            
        elif subcmd == "auto-discover":
            if subargs.lower() in ("true", "yes", "on", "1"):
                self.config.auto_discover = True
            elif subargs.lower() in ("false", "no", "off", "0"):
                self.config.auto_discover = False
            else:
                console.print("[yellow]Usage: /config auto-discover [true|false][/]")
                return
                
            self.config.save()
            console.print(f"[green]Auto-discovery {'enabled' if self.config.auto_discover else 'disabled'}[/]")
            
        elif subcmd == "discovery-path":
            parts = subargs.split(maxsplit=1)
            action = parts[0].lower() if parts else ""
            path = parts[1] if len(parts) > 1 else ""
            
            if action == "add" and path:
                if path not in self.config.discovery_paths:
                    self.config.discovery_paths.append(path)
                    self.config.save()
                    console.print(f"[green]Added discovery path: {path}[/]")
                else:
                    console.print(f"[yellow]Path already exists: {path}[/]")
                    
            elif action == "remove" and path:
                if path in self.config.discovery_paths:
                    self.config.discovery_paths.remove(path)
                    self.config.save()
                    console.print(f"[green]Removed discovery path: {path}[/]")
                else:
                    console.print(f"[yellow]Path not found: {path}[/]")
                    
            elif action == "list" or not action:
                console.print("\n[bold]Discovery Paths:[/]")
                for i, path in enumerate(self.config.discovery_paths, 1):
                    console.print(f"{i}. {path}")
                    
            else:
                console.print("[yellow]Usage: /config discovery-path [add|remove|list] [PATH][/]")
                
        else:
            console.print("[yellow]Unknown config command. Available: api-key, model, max-tokens, history-size, auto-discover, discovery-path[/]")
    
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
            console.print("[yellow]Unknown servers command. Available: list, add, remove, connect, disconnect, enable, disable, status[/]")
    
    async def list_servers(self):
        """List all configured servers"""
        if not self.config.servers:
            console.print(f"{STATUS_EMOJI['warning']} [yellow]No servers configured[/]")
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
            
        console.print(server_table)
    
    async def add_server(self, args):
        """Add a new server to configuration"""
        parts = args.split(maxsplit=3)
        if len(parts) < 3:
            console.print("[yellow]Usage: /servers add NAME TYPE PATH [ARGS...][/]")
            console.print("Example: /servers add github stdio /path/to/github-server.js")
            console.print("Example: /servers add github sse https://github-mcp-server.example.com")
            return
            
        name, type_str, path = parts[0], parts[1], parts[2]
        extra_args = parts[3].split() if len(parts) > 3 else []
        
        # Validate inputs
        if name in self.config.servers:
            console.print(f"[red]Server with name '{name}' already exists[/]")
            return
            
        try:
            server_type = ServerType(type_str.lower())
        except ValueError:
            console.print(f"[red]Invalid server type: {type_str}. Use 'stdio' or 'sse'[/]")
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
        console.print(f"[green]Server '{name}' added to configuration[/]")
        
        # Ask if user wants to connect now
        if Confirm.ask("Connect to server now?"):
            await self.connect_server(name)
    
    async def remove_server(self, name):
        """Remove a server from configuration"""
        if not name:
            console.print("[yellow]Usage: /servers remove SERVER_NAME[/]")
            return
            
        if name not in self.config.servers:
            console.print(f"[red]Server '{name}' not found[/]")
            return
            
        # Disconnect if connected
        if name in self.server_manager.active_sessions:
            await self.disconnect_server(name)
            
        # Remove from config
        del self.config.servers[name]
        self.config.save()
        
        console.print(f"[green]Server '{name}' removed from configuration[/]")
    
    async def connect_server(self, name):
        """Connect to a specific server"""
        if not name:
            console.print("[yellow]Usage: /servers connect SERVER_NAME[/]")
            return
            
        if name not in self.config.servers:
            console.print(f"[red]Server '{name}' not found[/]")
            return
            
        if name in self.server_manager.active_sessions:
            console.print(f"[yellow]Server '{name}' is already connected[/]")
            return
            
        # Connect to server using the context manager
        server_config = self.config.servers[name]
        
        with Status(f"{STATUS_EMOJI['server']} Connecting to {name}...", spinner="dots") as status:
            async with self.server_manager.connect_server_session(server_config) as session:
                if session:
                    status.update(f"{STATUS_EMOJI['connected']} Connected to server: {name}")
                    
                    # Load capabilities
                    status.update(f"{STATUS_EMOJI['tool']} Loading capabilities from {name}...")
                    await self.server_manager.load_server_capabilities(name, session)
                    status.update(f"{STATUS_EMOJI['success']} Loaded capabilities from server: {name}")
                    
                    console.print(f"[green]Connected to server: {name}[/]")
                else:
                    status.update(f"{STATUS_EMOJI['error']} Failed to connect to server: {name}")
                    console.print(f"[red]Failed to connect to server: {name}[/]")
    
    async def disconnect_server(self, name):
        """Disconnect from a specific server"""
        if not name:
            console.print("[yellow]Usage: /servers disconnect SERVER_NAME[/]")
            return
            
        if name not in self.server_manager.active_sessions:
            console.print(f"[yellow]Server '{name}' is not connected[/]")
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
            
        console.print(f"[green]Disconnected from server: {name}[/]")
    
    async def enable_server(self, name, enable=True):
        """Enable or disable a server"""
        if not name:
            action = "enable" if enable else "disable"
            console.print(f"[yellow]Usage: /servers {action} SERVER_NAME[/]")
            return
            
        if name not in self.config.servers:
            console.print(f"[red]Server '{name}' not found[/]")
            return
            
        # Update config
        self.config.servers[name].enabled = enable
        self.config.save()
        
        action = "enabled" if enable else "disabled"
        console.print(f"[green]Server '{name}' {action}[/]")
        
        # Connect or disconnect if needed
        if enable and name not in self.server_manager.active_sessions:
            if Confirm.ask(f"Connect to server '{name}' now?"):
                await self.connect_server(name)
        elif not enable and name in self.server_manager.active_sessions:
            if Confirm.ask(f"Disconnect from server '{name}' now?"):
                await self.disconnect_server(name)
    
    async def server_status(self, name):
        """Show detailed status for a server"""
        if not name:
            console.print("[yellow]Usage: /servers status SERVER_NAME[/]")
            return
            
        if name not in self.config.servers:
            console.print(f"[red]Server '{name}' not found[/]")
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
        
        console.print(Panel(basic_info, title=f"Server Status: {name}", border_style="blue"))
        
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
            
            console.print(Panel(capability_info, title="Capabilities", border_style="green"))
            
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
                        
                        console.print(Panel(process_info, title="Process Information", border_style="yellow"))
                    except Exception:
                        console.print(Panel(f"Process ID: {pid} (stats unavailable)", 
                                           title="Process Information", 
                                           border_style="yellow"))
    
    async def cmd_tools(self, args):
        """List available tools"""
        if not self.server_manager.tools:
            console.print(f"{STATUS_EMOJI['warning']} [yellow]No tools available from connected servers[/]")
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
            
        console.print(tool_table)
        
        # Offer to show schema for a specific tool
        if not args:
            tool_name = Prompt.ask("Enter tool name to see schema (or press Enter to skip)")
            if tool_name in self.server_manager.tools:
                tool = self.server_manager.tools[tool_name]
                
                # Use Group to combine the title and schema
                schema_display = Group(
                    Text(f"Schema for {tool_name}:", style="bold"),
                    Syntax(json.dumps(tool.input_schema, indent=2), "json", theme="monokai")
                )
                
                console.print(Panel(
                    schema_display, 
                    title=f"Tool: {tool_name}", 
                    border_style="magenta"
                ))
    
    async def cmd_resources(self, args):
        """List available resources"""
        if not self.server_manager.resources:
            console.print(f"{STATUS_EMOJI['warning']} [yellow]No resources available from connected servers[/]")
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
            
        console.print(resource_table)
    
    async def cmd_prompts(self, args):
        """List available prompts"""
        if not self.server_manager.prompts:
            console.print(f"{STATUS_EMOJI['warning']} [yellow]No prompts available from connected servers[/]")
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
            
        console.print(prompt_table)
        
        # Offer to show template for a specific prompt
        if not args:
            prompt_name = Prompt.ask("Enter prompt name to see template (or press Enter to skip)")
            if prompt_name in self.server_manager.prompts:
                prompt = self.server_manager.prompts[prompt_name]
                console.print(f"\n[bold]Template for {prompt_name}:[/]")
                console.print(prompt.template)
    
    async def cmd_history(self, args):
        """View conversation history"""
        if not self.history.entries:
            console.print("[yellow]No conversation history[/]")
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
            console=console,
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
        
        console.print(f"\n[bold]Recent Conversations (last {entries_to_show}):[/]")
        
        for i, entry in enumerate(recent_entries, 1):
            console.print(f"\n[bold cyan]{i}. {entry.timestamp}[/] - Model: {entry.model}")
            console.print(f"Servers: {', '.join(entry.server_names) if entry.server_names else 'None'}")
            console.print(f"Tools: {', '.join(entry.tools_used) if entry.tools_used else 'None'}")
            console.print(f"[bold blue]Q:[/] {entry.query[:100]}..." if len(entry.query) > 100 else f"[bold blue]Q:[/] {entry.query}")
            console.print(f"[bold green]A:[/] {entry.response[:100]}..." if len(entry.response) > 100 else f"[bold green]A:[/] {entry.response}")
    
    async def cmd_model(self, args):
        """Change the current model"""
        if not args:
            console.print(f"Current model: [cyan]{self.current_model}[/]")
            console.print("Usage: /model MODEL_NAME")
            console.print("Example models: claude-3-7-sonnet-20250219, claude-3-5-sonnet-latest")
            return
            
        self.current_model = args
        console.print(f"[green]Model changed to: {args}[/]")
    
    async def cmd_clear(self, args):
        """Clear the conversation context"""
        # self.conversation_messages = []
        self.conversation_graph.set_current_node("root")
        # Optionally clear the root node's messages too
        if Confirm.ask("Reset conversation to root? (This clears root messages too)"):
             root_node = self.conversation_graph.get_node("root")
             if root_node:
                 root_node.messages = []
                 root_node.children = [] # Also clear children if resetting completely? Discuss.
                 # Need to prune orphaned nodes from self.conversation_graph.nodes if we clear children
                 # For now, just reset messages and current node
                 root_node.messages = []
             console.print("[green]Conversation reset to root node.[/]")
        else:
             console.print("[yellow]Clear cancelled. Still on root node, messages preserved.[/]")

    async def cmd_reload(self, args):
        """Reload servers and capabilities"""
        console.print("[yellow]Reloading servers and capabilities...[/]")
        
        # Close existing connections
        with Status(f"{STATUS_EMOJI['server']} Closing existing connections...", spinner="dots") as status:
            await self.server_manager.close()
            status.update(f"{STATUS_EMOJI['success']} Existing connections closed")
        
        # Reset collections
        self.server_manager = ServerManager(self.config)
        
        # Reconnect
        with Status(f"{STATUS_EMOJI['server']} Reconnecting to servers...", spinner="dots") as status:
            await self.server_manager.connect_to_servers()
            status.update(f"{STATUS_EMOJI['success']} Servers reconnected")
        
        console.print("[green]Servers and capabilities reloaded[/]")
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
            console.print("[yellow]Caching is disabled.[/]")
            return

        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"
        subargs = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            console.print("\n[bold]Cached Tool Results:[/]")
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
                 console.print("[yellow]Cache is empty.[/]")
                 return
            
            # Use progress bar for loading cache entries - especially useful for large caches
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
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
            
            console.print(cache_table)
            console.print(f"Total entries: {len(entries)}")

        elif subcmd == "clear":
            if not subargs or subargs == "--all":
                if Confirm.ask("Are you sure you want to clear the entire cache?"):
                    # Use Progress for cache clearing - especially useful for large caches
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        BarColumn(),
                        TaskProgressColumn(),
                        console=console,
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
                    
                    console.print("[green]Cache cleared.[/]")
                else:
                    console.print("[yellow]Cache clear cancelled.[/]")
            else:
                tool_name_to_clear = subargs
                # Invalidate based on tool name prefix
                with Status(f"{STATUS_EMOJI['package']} Clearing cache for {tool_name_to_clear}...", spinner="dots") as status:
                    self.tool_cache.invalidate(tool_name=tool_name_to_clear)
                    status.update(f"{STATUS_EMOJI['success']} Cache entries for {tool_name_to_clear} cleared")
                console.print(f"[green]Cleared cache entries for tool: {tool_name_to_clear}[/]")
        
        
        elif subcmd == "clean":
            # Use Progress for cache cleaning - especially useful for large caches
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
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
            
            console.print(f"[green]Expired cache entries cleaned. Removed {removed_count} entries.[/]")
        
        elif subcmd == "dependencies" or subcmd == "deps":
            # Show dependency graph
            console.print("\n[bold]Tool Dependency Graph:[/]")
            
            if not self.tool_cache.dependency_graph:
                console.print("[yellow]No dependencies registered.[/]")
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
            
            console.print(dependency_table)
            console.print(f"Total tools with dependencies: {len(self.tool_cache.dependency_graph)}")
            
            # Process specific tool's dependencies
            if subargs:
                tool_name = subargs
                dependencies = self.tool_cache.dependency_graph.get(tool_name, set())
                
                if dependencies:
                    # Show the tool's dependencies in a tree
                    tree = Tree(f"[bold cyan]{tool_name}[/]")
                    for dep in dependencies:
                        tree.add(f"[magenta]{dep}[/]")
                    
                    console.print("\n[bold]Dependencies for selected tool:[/]")
                    console.print(tree)
                else:
                    console.print(f"\n[yellow]Tool '{tool_name}' has no dependencies or was not found.[/]")
        
        else:
            console.print("[yellow]Unknown cache command. Available: list, clear [tool_name | --all], clean, dependencies[/]")

    async def cmd_fork(self, args):
        """Create a new conversation fork/branch"""
        fork_name = args if args else None
        try:
            new_node = self.conversation_graph.create_fork(name=fork_name)
            self.conversation_graph.set_current_node(new_node.id)
            console.print(f"[green]Created and switched to new branch:[/]")
            console.print(f"  ID: [cyan]{new_node.id}[/]" )
            console.print(f"  Name: [yellow]{new_node.name}[/]")
            console.print(f"Branched from node: [magenta]{new_node.parent.id if new_node.parent else 'None'}[/]")
        except Exception as e:
            console.print(f"[red]Error creating fork: {e}[/]")

    async def cmd_branch(self, args):
        """Manage conversation branches"""
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "list"
        subargs = parts[1] if len(parts) > 1 else ""

        if subcmd == "list":
            console.print("\n[bold]Conversation Branches:[/]")
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
            console.print(branch_tree)

        elif subcmd == "checkout":
            if not subargs:
                console.print("[yellow]Usage: /branch checkout NODE_ID[/]")
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
                             console.print(f"[red]Ambiguous node ID prefix: {node_id}. Multiple matches found.[/]")
                             return # Ambiguous prefix
                        matched_node = node
            
            if matched_node:
                if self.conversation_graph.set_current_node(matched_node.id):
                    console.print(f"[green]Switched to branch:[/]")
                    console.print(f"  ID: [cyan]{matched_node.id}[/]")
                    console.print(f"  Name: [yellow]{matched_node.name}[/]")
                else:
                    # Should not happen if matched_node is valid
                    console.print(f"[red]Failed to switch to node {node_id}[/]") 
            else:
                console.print(f"[red]Node ID '{node_id}' not found.[/]")

        # Add other subcommands like rename, delete later if needed
        # elif subcmd == "rename": ...
        # elif subcmd == "delete": ...

        else:
            console.print("[yellow]Unknown branch command. Available: list, checkout NODE_ID[/]")

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
            with Live(self.generate_dashboard_renderable(), refresh_per_second=1.0/self.config.dashboard_refresh_rate, screen=True, transient=True) as live:
                while True:
                    await asyncio.sleep(self.config.dashboard_refresh_rate)
                    live.update(self.generate_dashboard_renderable())
        except KeyboardInterrupt:
            console.print("\n[yellow]Dashboard stopped.[/]")
        except Exception as e:
            log.error(f"Dashboard error: {e}")
            console.print(f"\n[red]Dashboard encountered an error: {e}[/]")

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
            console.print(f"[red]Conversation ID '{conversation_id}' not found[/]")
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
            console=console
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
                console.print(f"[red]Failed to export conversation: {e}[/]")
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
                        console.print(f"[yellow]Invalid token count: {parts[i+1]}[/]")
        
        console.print(f"[yellow]Optimizing conversation context...[/]")
        
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
            console.print(f"[green]Conversation optimized: {current_tokens} → {new_tokens} tokens[/]")
        else:
            console.print(f"[red]Failed to optimize conversation.[/]")
    
    async def auto_prune_context(self):
        """Auto-prune context based on token count"""
        token_count = await self.count_tokens()
        if token_count > self.config.auto_summarize_threshold:
            console.print(f"[yellow]Context size ({token_count} tokens) exceeds threshold "
                          f"({self.config.auto_summarize_threshold}). Auto-summarizing...[/]")
            await self.cmd_optimize(f"--tokens {self.config.max_summarized_tokens}")

    async def cmd_tool(self, args):
        """Directly execute a tool with parameters"""
        if not args:
            console.print("[yellow]Usage: /tool NAME {JSON_PARAMS}[/yellow]")
            return
            
        # Split into tool name and params
        try:
            parts = args.split(" ", 1)
            tool_name = parts[0]
            params_str = parts[1] if len(parts) > 1 else "{}"
            params = json.loads(params_str)
        except json.JSONDecodeError:
            console.print("[red]Invalid JSON parameters. Use valid JSON format.[/red]")
            return
        except Exception as e:
            console.print(f"[red]Error parsing command: {e}[/red]")
            return

        # Check if tool exists
        if tool_name not in self.server_manager.tools:
            console.print(f"[red]Tool not found: {tool_name}[/red]")
            return
        
        # Get the tool and its server
        tool = self.server_manager.tools[tool_name]
        server_name = tool.server_name
        
        with Status(f"{STATUS_EMOJI['tool']} Executing {tool_name}...", spinner="dots") as status:
            try:
                start_time = time.time()
                result = await self.execute_tool(server_name, tool_name, params)
                latency = time.time() - start_time
                
                status.update(f"{STATUS_EMOJI['success']} Tool execution completed in {latency:.2f}s")
                
                # Show result
                console.print(Panel.fit(
                    Syntax(json.dumps(result, indent=2), "json", theme="monokai"),
                    title=f"Tool Result: {tool_name} (executed in {latency:.2f}s)",
                    border_style="magenta"
                ))
            except Exception as e:
                status.update(f"{STATUS_EMOJI['failure']} Tool execution failed: {e}")
                console.print(f"[red]Error executing tool: {e}[/red]")

    # After the cmd_tool method (around line 4295)
    async def cmd_prompt(self, args):
        """Apply a prompt template to the conversation"""
        if not args:
            console.print("[yellow]Available prompt templates:[/yellow]")
            for name in self.server_manager.prompts:
                console.print(f"  - {name}")
            return
        
        prompt = self.server_manager.prompts.get(args)
        if not prompt:
            console.print(f"[red]Prompt not found: {args}[/red]")
            return
            
        self.conversation_graph.current_node.messages.insert(0, {
            "role": "system",
            "content": prompt.template
        })
        console.print(f"[green]Applied prompt: {args}[/green]")

    async def load_claude_desktop_config(self):
        """Look for and load the Claude desktop config file (claude_desktop_config.json) if it exists.
        
        This enables users to easily import MCP server configurations from Claude desktop.
        """
        config_path = Path("claude_desktop_config.json")
        if not config_path.exists():
            return
            
        try:
            with Status(f"{STATUS_EMOJI['config']} Found Claude desktop config file, processing...", spinner="dots") as status:
                async with aiofiles.open(config_path, 'r') as f:
                    content = await f.read()
                    desktop_config = json.loads(content)
                
                if not desktop_config.get('mcpServers'):
                    status.update(f"{STATUS_EMOJI['warning']} No MCP servers defined in Claude desktop config")
                    return
                
                # Track successful imports and skipped servers
                imported_servers = []
                skipped_servers = []
                
                for server_name, server_data in desktop_config['mcpServers'].items():
                    # Skip if server already exists
                    if server_name in self.config.servers:
                        log.info(f"Server '{server_name}' already exists in config, skipping")
                        skipped_servers.append((server_name, "already exists"))
                        continue
                    
                    # Convert to our server config format
                    command = server_data.get('command', '')
                    args = server_data.get('args', [])
                    
                    if not command:
                        log.warning(f"No command specified for server '{server_name}', skipping")
                        skipped_servers.append((server_name, "missing command"))
                        continue
                    
                    # Create new server config
                    server_config = ServerConfig(
                        name=server_name,
                        type=ServerType.STDIO,  # Claude desktop only supports STDIO servers
                        path=command,
                        args=args,
                        enabled=True,
                        auto_start=True,
                        description=f"Imported from Claude desktop config",
                        trusted=True,  # Assume trusted since user configured it in Claude desktop
                    )
                    
                    # Add to our config
                    self.config.servers[server_name] = server_config
                    imported_servers.append(server_name)
                    log.info(f"Imported server '{server_name}' from Claude desktop config")
                
                # Save config
                if imported_servers:
                    self.config.save()
                    status.update(f"{STATUS_EMOJI['success']} Imported {len(imported_servers)} servers from Claude desktop config")
                    
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
                                " ".join(server.args) if server.args else ""
                            )
                        
                        console.print(server_table)
                    
                    if skipped_servers:
                        skipped_table = Table(title="Skipped Servers")
                        skipped_table.add_column("Name")
                        skipped_table.add_column("Reason")
                        
                        for name, reason in skipped_servers:
                            skipped_table.add_row(name, reason)
                        
                        console.print(skipped_table)
                else:
                    status.update(f"{STATUS_EMOJI['warning']} No new servers imported from Claude desktop config")
        
        except FileNotFoundError:
            log.debug("Claude desktop config file not found")
        except json.JSONDecodeError:
            log.error("Invalid JSON in Claude desktop config file")
        except Exception as e:
            log.error(f"Error processing Claude desktop config: {e}")

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
        with Status(f"{STATUS_EMOJI['scroll']} Exporting conversation...", spinner="dots") as status:
            success = await self.export_conversation(conversation_id, output_path)
            if success:
                status.update(f"{STATUS_EMOJI['success']} Conversation exported successfully")
                console.print(f"[green]Conversation exported to: {output_path}[/]")
            else:
                status.update(f"{STATUS_EMOJI['failure']} Export failed")
                console.print(f"[red]Failed to export conversation[/]")

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
            console.print("[yellow]Usage: /import FILEPATH[/]")
            return
        
        file_path = args.strip()
        
        with Status(f"{STATUS_EMOJI['scroll']} Importing conversation from {file_path}...", spinner="dots") as status:
            success = await self.import_conversation(file_path)
            if success:
                status.update(f"{STATUS_EMOJI['success']} Conversation imported successfully")
                console.print(f"[green]Conversation imported and set as current conversation[/]")
            else:
                status.update(f"{STATUS_EMOJI['failure']} Import failed")
                console.print(f"[red]Failed to import conversation from {file_path}[/]")

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
    try:
        # Get current conversation if not specified
        if not conversation_id:
            conversation_id = client.conversation_graph.current_node.id
            
        # Default filename if not provided
        if not output:
            output = f"conversation_{conversation_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        success = await client.export_conversation(conversation_id, output)
        if success:
            console.print(f"[green]Conversation exported to: {output}[/]")
        else:
            console.print(f"[red]Failed to export conversation.[/]")
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
    try:
        success = await client.import_conversation(file_path)
        if success:
            console.print(f"[green]Conversation imported successfully from: {file_path}[/]")
        else:
            console.print(f"[red]Failed to import conversation.[/]")
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
    
    try:
        # Set up client and connect to servers
        await client.setup()
        
        # Connect to specific server(s) if provided
        if server:
            for s in server:
                if s in client.config.servers:
                    await client.connect_server(s)
        else:
                    console.print(f"[yellow]Server not found: {s}[/]")
        
        # Launch dashboard if requested
        if dashboard:
            # Ensure monitor is running for dashboard data
            if not client.server_monitor.monitoring:
                await client.server_monitor.start_monitoring()
            await client.cmd_dashboard("") # Call the command method
            # Dashboard is blocking, exit after it's closed
            # We need to ensure cleanup happens, maybe move setup/close logic
            await client.close() # Ensure cleanup after dashboard closes
            return
        
        # Process single query if provided
        if query:
            result = await client.process_query(query, model=model)
            console.print()
            console.print(Panel.fit(
                Markdown(result),
                title=f"Claude ({client.current_model})",
                border_style="green"
            ))
        elif interactive or not query:
            # Run interactive loop
            await client.interactive_loop()
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted[/]")
        # Ensure cleanup happens if main loop interrupted
        if 'client' in locals() and client: 
             await client.close()
    
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        if verbose_logging:
            import traceback
            console.print(traceback.format_exc())
    
    finally:
        # Clean up (already handled in KeyboardInterrupt and normal exit paths)
        # Ensure close is called if setup succeeded but something else failed
        if 'client' in locals() and client and hasattr(client, 'server_manager'): # Check if client partially initialized
            # Check if already closed (e.g. after dashboard)
            # This logic might need refinement depending on exact exit paths
            pass # await client.close() # Re-closing might cause issues
        pass # Final cleanup logic might be needed here

async def servers_async(search, list_all, json_output):
    """Server management async function"""
    client = MCPClient()
    
    try:
        if search:
            # Discover servers
            with console.status("[cyan]Searching for servers...[/]"):
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
                console.print(json.dumps(server_data, indent=2))
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
    try:
        client = MCPClient() # Instantiate client within the main try

        if reset:
            if Confirm.ask("[yellow]Are you sure you want to reset the configuration?[/]"):
                # Create a new default config
                new_config = Config()
                # Save it
                new_config.save()
                console.print("[green]Configuration reset to defaults[/]")

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
                     console.print(f"[yellow]Editor exited with code {process.returncode}[/]")
                else:
                    console.print(f"[green]Configuration file potentially edited: {CONFIG_FILE}[/]")
            except FileNotFoundError:
                 console.print(f"[red]Editor command not found: '{editor}'. Set EDITOR environment variable.[/]")
            except OSError as e:
                 console.print(f"[red]Error running editor '{editor}': {e}[/]")
            # Keep broad exception for unexpected editor issues
            except Exception as e:
                 console.print(f"[red]Unexpected error trying to edit config: {e}")
            # --- End inner try for editor ---

            # Reload config (Needs client object)
            if client:
                 client.config.load()
                 console.print("[green]Configuration reloaded[/]")
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

            console.print(Panel(
                Syntax(yaml.safe_dump(config_data, default_flow_style=False), "yaml", theme="monokai"),
                title="Current Configuration",
                border_style="blue"
            ))

    # --- Top-level exceptions for config_async itself ---
    # These should catch errors during MCPClient() init or file operations if needed,
    # NOT the misplaced blocks from before.
    except (IOError, yaml.YAMLError, json.JSONDecodeError) as e:
         console.print(f"[bold red]Configuration/File Error during config command:[/] {str(str(e))}")
         # Decide if sys.exit is appropriate here or just log
    except Exception as e:
        console.print(f"[bold red]Unexpected Error during config command:[/] {str(e)}")
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
        # Correct indentation (aligned with try)
        console.print(f"[bold red]MCP Error:[/] {str(e)}")
        sys.exit(1)
    except httpx.RequestError as e:
        # Correct indentation
        console.print(f"[bold red]Network Error:[/] {str(e)}")
        sys.exit(1)
    except anthropic.APIError as e:
        # Correct indentation
        console.print(f"[bold red]Anthropic API Error:[/] {str(e)}")
        sys.exit(1)
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as e:
        # Correct indentation
        console.print(f"[bold red]Configuration/File Error:[/] {str(e)}")
        sys.exit(1)
    except Exception as e: # Keep broad exception for top-level unexpected errors
        # Correct indentation
        console.print(f"[bold red]Unexpected Error:[/] {str(e)}")
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
