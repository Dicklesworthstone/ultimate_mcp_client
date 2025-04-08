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
#     "asyncio>=3.4.3"
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
- Health Dashboard: Real-time monitoring of servers and tool performance
- Observability: Comprehensive metrics and tracing
- Registry Integration: Connect to remote registries to discover servers
- Local Discovery: Discover MCP servers on your local network via mDNS

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

# Configuration
python mcpclient.py config --show

Command Reference:
-----------------
Interactive mode commands:
- /help - Show available commands
- /servers - Manage MCP servers (list, add, connect, etc.)
- /tools - List and inspect available tools
- /resources - List available resources
- /prompts - List available prompts
- /model - Change AI model
- /fork - Create a conversation branch
- /branch - Manage conversation branches
- /cache - Manage tool caching
- /dashboard - Open health monitoring dashboard
- /monitor - Control server monitoring
- /registry - Manage server registry connections

Author: Jeffrey Emanuel
License: MIT
Version: 1.0.0
"""

import asyncio
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
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
)

# Typer CLI
import typer

# Anthropic API
import anthropic
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import (
    ToolParam,
)

# MCP SDK imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import Prompt, Resource, Tool
from rich import box

# Rich UI components
from rich.console import Console
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
from rich.prompt import Confirm, Prompt as RichPrompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.tree import Tree
from typing_extensions import Annotated

# Observability
try:
    from opentelemetry import metrics, trace
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    HAS_OPENTELEMETRY = True
except ImportError:
    HAS_OPENTELEMETRY = False

# Additional utilities
import httpx
import psutil
import yaml

# Cache libraries
try:
    import diskcache
    HAS_DISKCACHE = True
except ImportError:
    HAS_DISKCACHE = False

# Set up Typer app
app = typer.Typer(help="🔌 Ultimate MCP Client for Anthropic API")

# Configure Rich theme
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
    "healthy": Emoji("✅"),
    "degraded": Emoji("⚠️"),
    "error": Emoji("❌"),
    "connected": Emoji("🟢"),
    "disconnected": Emoji("🔴"),
    "cached": Emoji("📦"),
    "streaming": Emoji("🌊"),
    "forked": Emoji("🔱"),
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

# Initialize OpenTelemetry if available
if HAS_OPENTELEMETRY:
    trace_provider = TracerProvider()
    console_exporter = ConsoleSpanExporter()
    span_processor = BatchSpanProcessor(console_exporter)
    trace_provider.add_span_processor(span_processor)
    trace.set_tracer_provider(trace_provider)
    
    meter_provider = MeterProvider()
    metric_reader = PeriodicExportingMetricReader(ConsoleMetricExporter())
    meter_provider.add_metric_reader(metric_reader)
    metrics.set_meter_provider(meter_provider)
    
    tracer = trace.get_tracer("mcpclient")
    meter = metrics.get_meter("mcpclient")
    
    # Create instruments
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
else:
    tracer = None
    meter = None
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
    response_times: deque[float] = field(default_factory=deque)
    error_rate: float = 0.0
    
    def update_response_time(self, response_time: float) -> None:
        """Add a new response time and recalculate average"""
        self.response_times.append(response_time)
        # Keep only the last 100 responses
        if len(self.response_times) > 100:
            self.response_times.popleft()
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
    execution_times: deque[float] = field(default_factory=deque)
    last_used: datetime = field(default_factory=datetime.now)
    
    def update_execution_time(self, time_ms: float) -> None:
        """Update execution time metrics"""
        self.execution_times.append(time_ms)
        if len(self.execution_times) > 100:
            self.execution_times.popleft()
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
    original_prompt: Prompt
    call_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)

@dataclass
class ConversationNode:
    id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    parent: Optional["ConversationNode"] = None
    children: List["ConversationNode"] = field(default_factory=list)
    name: str = "Root"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    model: str = ""
    
    def add_message(self, message: Dict[str, Any]) -> None:
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
                def __init__(self, registry: "ServerRegistry") -> None:
                    self.registry = registry
                
                def add_service(self, zeroconf, service_type, name):
                    info = zeroconf.get_service_info(service_type, name)
                    if not info:
                        return
                        
                    server_name = name.replace("._mcp._tcp.local.", "")
                    host = socket.inet_ntoa(info.addresses[0]) if info.addresses else "localhost"
                    port = info.port
                    
                    self.registry.discovered_servers[server_name] = {
                        "name": server_name,
                        "host": host,
                        "port": port,
                        "type": "sse",
                        "url": f"http://{host}:{port}",
                        "properties": {
                            k.decode('utf-8'): v.decode('utf-8') for k, v in info.properties.items()
                        }
                    }
                    
                    log.info(f"Discovered local MCP server: {server_name} at {host}:{port}")
                
                def remove_service(self, zeroconf, service_type, name):
                    server_name = name.replace("._mcp._tcp.local.", "")
                    # Ensure zeroconf is available before using it
                    if not self.registry.zeroconf:
                         return
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
        self.disk_cache = None
        
        # Set up disk cache if available
        if HAS_DISKCACHE:
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
        self.entries: List[ChatHistory] = []
        self.max_entries = max_entries
        self.load()
    
    def add(self, entry: ChatHistory):
        """Add a new entry to history"""
        self.entries.append(entry)
        
        # Trim if needed
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]
        
        self.save()
    
    def load(self):
        """Load history from file"""
        if not HISTORY_FILE.exists():
            return
        
        try:
            with open(HISTORY_FILE, 'r') as f:
                history_data = json.load(f)
            
            self.entries = []
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
                
            # Respect max_entries
            self.entries = self.entries[-self.max_entries:]
            
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


if __name__ == "__main__":
    # Initialize colorama for Windows terminals
    if platform.system() == "Windows":
        try:
            import colorama
            colorama.init(convert=True)
        except ImportError:
            log.warning("Colorama not found, colors might not work correctly on Windows.")
        except Exception as e:
            log.error(f"Error initializing colorama: {e}")
    
    # Run the app
    app()

