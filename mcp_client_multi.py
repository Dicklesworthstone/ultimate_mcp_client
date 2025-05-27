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
#     "python-multipart>=0.0.6",
#     "json5"
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
- /model [MODEL_NAME] - Change AI model (e.g., /model gpt-4o, /model claude-3-7-sonnet-20250219)
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

import ast
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
import textwrap
import time
import traceback
import uuid
from collections import deque
from contextlib import AsyncExitStack, asynccontextmanager, contextmanager, redirect_stdout, suppress
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
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
from fastapi.encoders import jsonable_encoder
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
    TextContent,  # noqa: F401
    Tool,
)
from mcp.types import (
    InitializeResult as MCPInitializeResult,  # Alias to avoid confusion with provider results
)
from mcp.types import Prompt as McpPromptType
from openai import APIConnectionError as OpenAIAPIConnectionError
from openai import APIError as OpenAIAPIError
from openai import APIStatusError as OpenAIAPIStatusError
from openai import (
    AsyncOpenAI,  # For OpenAI, Grok, DeepSeek, Mistral, Groq, Cerebras, Gemini
    AsyncStream,
)
from openai import AuthenticationError as OpenAIAuthenticationError
from openai import BadRequestError as OpenAIBadRequestError
from openai import NotFoundError as OpenAINotFoundError
from openai import PermissionDeniedError as OpenAIPermissionDeniedError
from openai import RateLimitError as OpenAIRateLimitError
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

from agent_master_loop import (
    MISTRAL_NATIVE_MODELS_SUPPORTING_SCHEMA,
    MODELS_CONFIRMED_FOR_OPENAI_JSON_SCHEMA_FORMAT,
    MODELS_SUPPORTING_OPENAI_JSON_OBJECT_FORMAT,
)

if TYPE_CHECKING:
    from agent_master_loop import AgentMasterLoop, PlanStep
else:
    AgentMasterLoop = "AgentMasterLoop"  # Placeholder for AgentMasterLoop
    PlanStep = "PlanStep"  # Placeholder for PlanStep

decouple_config = DecoupleConfig(RepositoryEnv(".env"))

# =============================================================================
# Constants Integration
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


class TaskType(str, Enum):  # Keep as is
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


class LogLevel(str, Enum):  # Keep as is
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Cost estimates (Updated with prefixed model names for clarity)
COST_PER_MILLION_TOKENS: Dict[str, Dict[str, float]] = {
    # OpenAI models
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gpt-4o-mini": {"input": 0.15, "output": 0.6},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Claude models (Anthropic)
    "claude-3-7-sonnet-20250219": {"input": 3.0, "output": 15.0},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.0},
    "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    "claude-3-sonnet-20240229": {"input": 3.0, "output": 15.0},
    # DeepSeek models
    "deepseek-chat": {"input": 0.27, "output": 1.10},
    "deepseek-reasoner": {"input": 0.55, "output": 2.19},
    # Gemini models (Google)
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash": {"input": 0.35, "output": 1.05},
    "gemini-2.0-flash-thinking-exp-01-21": {"input": 0.0, "output": 0.0},  # Assuming free tier
    "gemini-2.5-pro-exp-03-25": {"input": 1.25, "output": 10.0},
    # OpenRouter models (Prefixed)
    "openrouter/mistralai/mistral-nemo": {"input": 0.035, "output": 0.08},
    "openrouter/tngtech/deepseek-r1t-chimera:free": {"input": 0.00001, "output": 0.00001},
    # Grok models
    "grok-3-latest": {"input": 3.0, "output": 15.0},
    "grok-3-fast-latest": {"input": 5.0, "output": 25.0},
    "grok-3-mini-latest": {"input": 0.30, "output": 0.50},
    "grok-3-mini-fast-latest": {"input": 0.60, "output": 4.0},
    # Mistral models (Native Mistral API)
    "mistral-large-latest": {"input": 2.0, "output": 6.0},
    "mistral-small-latest": {"input": 0.1, "output": 0.3},
    # Groq models (Prefixed)
    "groq/llama-3.3-70b-versatile": {"input": 0.0, "output": 0.0},  # Groq often has very low/free pricing during beta
    "groq/meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.0, "output": 0.0},
    "groq/mistral-saba-24b": {"input": 0.0, "output": 0.0},
    "groq/qwen-qwq-32b": {"input": 0.0, "output": 0.0},
    "groq/gemma2-9b-it": {"input": 0.0, "output": 0.0},
    "groq/compound-beta": {"input": 0.0, "output": 0.0},
    "groq/compound-beta-mini": {"input": 0.0, "output": 0.0},
    # Cerebras models (Prefixed)
    "cerebras/llama-4-scout-17b-16e-instruct": {"input": 0.0001, "output": 0.0001},  # Using placeholder cost
    "cerebras/llama-3.3-70b": {"input": 0.0001, "output": 0.0001},  # Using placeholder cost
}

# Default models by provider (Updated with prefixed names)
DEFAULT_MODELS = {
    Provider.OPENAI: "gpt-4o-mini",  # Changed to newer mini model
    Provider.ANTHROPIC: "claude-3-5-haiku-20241022",
    Provider.DEEPSEEK: "deepseek-chat",
    Provider.GEMINI: "gemini-2.0-flash-lite",  # Changed to cheaper flash model
    Provider.OPENROUTER: "openrouter/tngtech/deepseek-r1t-chimera:free",
    Provider.GROK: "grok-3-mini-latest",  # Changed to cheaper Grok model
    Provider.MISTRAL: "mistral-large-latest",
    Provider.GROQ: "groq/compound-beta",
    Provider.CEREBRAS: "cerebras/llama-4-scout-17b-16e-instruct",
}

# It's important that this schema is accurate and matches agent_master_loop.PlanStep
AGENT_UPDATE_PLAN_ARGUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "plan": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string", "default": "planned"},
                    "depends_on": {"type": "array", "items": {"type": "string"}, "default": []},
                    "assigned_tool": {"type": ["string", "null"]},
                    "tool_args": {
                        "type": ["object", "null"], 
                        "description": "Arguments for the assigned tool. Can be any valid JSON object."
                        # No "properties" and no "additionalProperties: False" here means it accepts any object.
                    },
                    "result_summary": {"type": ["string", "null"]},
                    "is_parallel_group": {"type": ["string", "null"]}
                },
                "required": ["id", "description"],
                "additionalProperties": False # For the PlanStep object itself
            },
            "description": "The new complete list of plan steps for the agent."
        }
    },
    "required": ["plan"],
    "additionalProperties": False # For the top-level arguments object
}


OPENAI_MAX_TOOL_COUNT = 128  # Maximum tools to send to OpenAI-compatible APIs

# Emoji mapping
EMOJI_MAP = {
    "start": "🚀",
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "info": "ℹ️",
    "debug": "🔍",
    "critical": "🔥",
    "server": "🖥️",
    "cache": "💾",
    "provider": "🔌",
    "request": "📤",
    "response": "📥",
    "processing": "⚙️",
    "model": "🧠",
    "config": "🔧",
    "token": "🔢",
    "cost": "💰",
    "time": "⏱️",
    "tool": "🛠️",
    "cancel": "🛑",
    "database": "🗄️",
    "browser": "🌐",
    "completion": "✍️",
    "chat": "💬",
    "summarization": "📝",
    "extraction": "💡",
    "generation": "🎨",
    "analysis": "📊",
    "classification": "🏷️",
    "query": "❓",
    "download": "⬇️",
    "upload": "⬆️",
    "document_processing": "📄",
    "document": "📄",
    "translation": "🔄",
    "qa": "❓",
    "history": "📜",
    "search": "🔎",
    "port": "🔌",
    "package": "📦",
    "resource": "📚",
    "prompt": "💬",
    "trident_emblem": "🔱",
    "desktop_computer": "🖥️",
    "gear": "⚙️",
    "scroll": "📜",
    "magnifying_glass_tilted_right": "🔎",
    "electric_plug": "🔌",
    "party_popper": "🎉",
    "collision": "💥",
    "robot": "🤖",
    "water_wave": "🌊",
    "green_circle": "🟢",
    "red_circle": "🔴",
    "white_check_mark": "✅",
    "cross_mark": "❌",
    "question_mark": "❓",
    "cached": "📦",
    Provider.OPENAI.value: "🟢",
    Provider.ANTHROPIC.value: "🟣",
    Provider.DEEPSEEK.value: "🐋",
    Provider.GEMINI.value: "♊",
    Provider.GROK.value: "⚡",
    Provider.MISTRAL.value: "🌫️",
    Provider.GROQ.value: "🚅",
    Provider.CEREBRAS.value: "🧠",
    Provider.OPENROUTER.value: "🔄",
    "status_healthy": "✅",
    "status_degraded": "⚠️",
    "status_error": "❌",
    "status_unknown": "❓",
}

# Add Emojis for status.healthy etc if needed, or map directly in Rich styles
# Example mapping for status consistency:
EMOJI_MAP["status_healthy"] = EMOJI_MAP["white_check_mark"]
EMOJI_MAP["status_degraded"] = EMOJI_MAP["warning"]
EMOJI_MAP["status_error"] = EMOJI_MAP["cross_mark"]
EMOJI_MAP["status_unknown"] = EMOJI_MAP["question_mark"]

AGENT_TOOL_UPDATE_PLAN = "agent:update_plan"

# ==========================================================================
# JSON SCHEMAS FOR STRUCTURED OUTPUT QUERIES
# ==========================================================================

# Schema for memory similarity analysis
MEMORY_SIMILARITY_SCHEMA = {
    "type": "object",
    "properties": {
        "similarity_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Semantic similarity score from 0-100"
        },
        "decision": {
            "type": "string",
            "enum": ["store_new", "store_with_note", "skip_similar", "skip_duplicate"],
            "description": "Decision on how to handle the memory"
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation for the decision"
        }
    },
    "required": ["similarity_score", "decision", "reason"],
    "additionalProperties": False
}

# Schema for semantic similarity scoring
SEMANTIC_SIMILARITY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Semantic similarity score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for consolidation decision
CONSOLIDATION_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Consolidation necessity score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for unique content analysis
UNIQUE_CONTENT_SCHEMA = {
    "type": "object",
    "properties": {
        "unique_items": {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 1
            },
            "description": "List of item numbers (1-based) that contain unique information"
        }
    },
    "required": ["unique_items"],
    "additionalProperties": False
}

# Schema for complexity scoring
COMPLEXITY_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Task complexity score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for research intensity scoring
RESEARCH_INTENSITY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Research intensity score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for creation focus scoring
CREATION_FOCUS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Creation focus score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for deliverable type prediction
DELIVERABLE_TYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "report": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Likelihood of producing analytical reports or research summaries"
        },
        "html_file": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Likelihood of producing web pages or HTML documents"
        },
        "interactive_quiz": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Likelihood of producing quizzes, tests, or assessments"
        },
        "code_file": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Likelihood of producing programming code or scripts"
        },
        "document": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Likelihood of producing general text documents or papers"
        },
        "general_artifact": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Likelihood of producing other types of deliverables"
        }
    },
    "required": ["report", "html_file", "interactive_quiz", "code_file", "document", "general_artifact"],
    "additionalProperties": False
}

# Schema for multi-part task analysis
MULTI_PART_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Multi-part task score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for integration needs analysis
INTEGRATION_NEEDS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "External integration needs score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for action phase analysis
ACTION_PHASE_SCHEMA = {
    "type": "object",
    "properties": {
        "phase": {
            "type": "string",
            "enum": ["initialization", "planning", "research", "development", "testing", "refinement", "completion", "error_recovery"],
            "description": "Current phase of work based on action analysis"
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Confidence in phase classification"
        }
    },
    "required": ["phase", "confidence"],
    "additionalProperties": False
}

# Schema for progress reasonableness assessment
PROGRESS_REASONABLENESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Progress reasonableness score from 0-100"
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the assessment"
        }
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False
}

# Schema for goal classification
GOAL_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["research", "development", "analysis", "creation", "optimization", "maintenance", "learning", "communication"],
            "description": "Primary goal category"
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Confidence in classification"
        },
        "secondary_categories": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["research", "development", "analysis", "creation", "optimization", "maintenance", "learning", "communication"]
            },
            "description": "Additional relevant categories"
        }
    },
    "required": ["category", "confidence"],
    "additionalProperties": False
}

# Schema for activity category assessment
ACTIVITY_CATEGORY_SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Activity category name"
                    },
                    "score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Relevance score for this category"
                    }
                },
                "required": ["name", "score"],
                "additionalProperties": False
            }
        }
    },
    "required": ["categories"],
    "additionalProperties": False
}

# Schema for replan necessity assessment
REPLAN_NECESSITY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Replan necessity score from 0-100"
        },
        "reasoning": {
            "type": "string",
            "description": "Explanation for the replan assessment"
        }
    },
    "required": ["score", "reasoning"],
    "additionalProperties": False
}

# Schema for file creation assessment
FILE_CREATION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "File creation likelihood score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for error planning assessment
ERROR_PLANNING_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Error planning-relatedness score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for plan step vagueness assessment
PLAN_VAGUENESS_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Plan step vagueness score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for workflow type matching
WORKFLOW_MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Workflow type match score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for markdown format detection
MARKDOWN_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "is_markdown": {
            "type": "boolean",
            "description": "Whether the text is markdown formatted"
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Confidence in the assessment"
        }
    },
    "required": ["is_markdown", "confidence"],
    "additionalProperties": False
}

# Schema for content quality assessment
CONTENT_QUALITY_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Content quality score from 0-100"
        },
        "aspects": {
            "type": "object",
            "properties": {
                "clarity": {"type": "integer", "minimum": 0, "maximum": 100},
                "completeness": {"type": "integer", "minimum": 0, "maximum": 100},
                "accuracy": {"type": "integer", "minimum": 0, "maximum": 100},
                "relevance": {"type": "integer", "minimum": 0, "maximum": 100}
            },
            "required": ["clarity", "completeness", "accuracy", "relevance"],
            "additionalProperties": False
        }
    },
    "required": ["score", "aspects"],
    "additionalProperties": False
}

# Schema for simple goal classification (just category name)
SIMPLE_GOAL_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["creation", "analysis", "planning", "communication", "evaluation", "problem_solving", "general_task"],
            "description": "Single goal category classification"
        }
    },
    "required": ["category"],
    "additionalProperties": False
}

# Schema for completion readiness analysis
COMPLETION_READINESS_SCHEMA = {
    "type": "object",
    "properties": {
        "completion_readiness": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Goal completion readiness score from 0-100"
        },
        "force_completion": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Force completion necessity score from 0-100"
        },
        "has_deliverable": {
            "type": "boolean",
            "description": "Whether deliverable exists or has been created"
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation of the completion assessment"
        }
    },
    "required": ["completion_readiness", "force_completion", "has_deliverable", "reason"],
    "additionalProperties": False
}

# Schema for simple scoring (0-100)
SIMPLE_SCORE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Simple numerical score from 0-100"
        }
    },
    "required": ["score"],
    "additionalProperties": False
}

# Schema for boolean analysis with confidence
BOOLEAN_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {
            "type": "boolean",
            "description": "Boolean result of the analysis"
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Confidence in the boolean assessment"
        }
    },
    "required": ["result", "confidence"],
    "additionalProperties": False
}

# Schema for error classification scoring
ERROR_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Error classification confidence score from 0-100"
        },
        "classification": {
            "type": "string",
            "enum": ["memory_type_error", "artifact_type_error", "missing_fields_error", "workflow_id_error", "file_access_error", "planning_error", "other"],
            "description": "Type of error classification"
        }
    },
    "required": ["score", "classification"],
    "additionalProperties": False
}

# Schema for enhanced goal classification with multiple scores
ENHANCED_GOAL_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "creation": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Creation work likelihood score"
        },
        "analysis": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Analysis work likelihood score"
        },
        "planning": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Planning work likelihood score"
        },
        "communication": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Communication work likelihood score"
        },
        "evaluation": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Evaluation work likelihood score"
        },
        "problem_solving": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Problem solving work likelihood score"
        }
    },
    "required": ["creation", "analysis", "planning", "communication", "evaluation", "problem_solving"],
    "additionalProperties": False
}

# Schema for JSON parsing extraction
JSON_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "completion_readiness": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Extracted completion readiness score"
        },
        "force_completion": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Extracted force completion score"
        },
        "has_deliverable": {
            "type": "boolean",
            "description": "Extracted deliverable existence flag"
        },
        "reason": {
            "type": "string",
            "description": "Extracted reason explanation"
        }
    },
    "required": ["completion_readiness", "force_completion", "has_deliverable", "reason"],
    "additionalProperties": False
}

# Schema for tool type classification
TOOL_TYPE_SCHEMA = {
    "type": "object",
    "properties": {
        "tool_type": {
            "type": "string",
            "enum": ["artifact_tool", "research_tool", "analysis_tool", "creation_tool", "storage_tool", "processing_tool", "other"],
            "description": "Classification of tool type"
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Confidence in the classification"
        }
    },
    "required": ["tool_type", "confidence"],
    "additionalProperties": False
}

# Schema for multi-category activity analysis
ACTIVITY_CATEGORIES_SCHEMA = {
    "type": "object",
    "properties": {
        "categories": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["information_gathering", "knowledge_storage", "content_creation", "analysis_thinking", "data_processing", "collaboration", "general_work"]
            },
            "description": "List of activity categories that score 40+ points"
        }
    },
    "required": ["categories"],
    "additionalProperties": False
}

# Schema for tool list parsing
TOOL_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "tools": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of recommended tools"
        }
    },
    "required": ["tools"],
    "additionalProperties": False
}

# Schema for improvement indicator analysis
IMPROVEMENT_INDICATORS_SCHEMA = {
    "type": "object",
    "properties": {
        "indicators": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of improvement indicators identified"
        }
    },
    "required": ["indicators"],
    "additionalProperties": False
}

# Schema for text format detection
TEXT_FORMAT_SCHEMA = {
    "type": "object",
    "properties": {
        "format": {
            "type": "string",
            "enum": ["markdown", "html", "plain_text", "json", "xml", "code", "other"],
            "description": "Detected text format"
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Confidence in format detection"
        }
    },
    "required": ["format", "confidence"],
    "additionalProperties": False
}

# Schema for raw text response
RAW_TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {
            "type": "string",
            "description": "Raw text result from analysis"
        }
    },
    "required": ["result"],
    "additionalProperties": False
}

# ==========================================================================
# Model -> Provider Mapping
# =============================================================================
# Maps known model identifiers (or prefixes) to their provider's enum value string.
MODEL_PROVIDER_MAP: Dict[str, str] = {}


def _infer_provider(model_name: str) -> Optional[str]:
    """
    Infers the provider based on the model name string using known prefixes.
    This is a best-effort guess, especially for models shared across providers.
    Priority: Explicit prefixes (e.g., 'openrouter/', 'groq/').
    """
    if not model_name:
        return None
    lname = model_name.lower()

    # 1. Explicit Provider Prefixes (Highest Priority)
    prefixes = {
        "openrouter/": Provider.OPENROUTER.value,
        "groq/": Provider.GROQ.value,
        "cerebras/": Provider.CEREBRAS.value,
        "openai/": Provider.OPENAI.value,
        "anthropic/": Provider.ANTHROPIC.value,
        "google/": Provider.GEMINI.value,  # Keep google/ for Gemini
        "gemini/": Provider.GEMINI.value,  # Allow gemini/ too
        "grok/": Provider.GROK.value,
        "deepseek/": Provider.DEEPSEEK.value,
        "mistralai/": Provider.MISTRAL.value,  # Keep mistralai/ -> Mistral as default
    }
    for prefix, provider_val in prefixes.items():
        if lname.startswith(prefix):
            # For OpenRouter, Groq, Cerebras, we *want* this prefix match
            if provider_val in [Provider.OPENROUTER.value, Provider.GROQ.value, Provider.CEREBRAS.value]:
                return provider_val
            # For others like mistralai/, it *could* be OpenRouter/Groq, but default to the main provider if no other prefix matches
            # This logic becomes less critical as the main flow uses the explicit provider.
            # Let's refine this: If it starts with a known base model provider but ISN'T explicitly openrouter/groq/cerebras, assume base provider.
            if provider_val not in [Provider.MISTRAL.value]:  # Example: Don't return Mistral if it might be OpenRouter/Groq
                return provider_val

    # 2. Common Model Name Prefixes (Lower Priority)
    if lname.startswith("gpt-") or lname.startswith("o1-") or lname.startswith("o3-"):
        return Provider.OPENAI.value
    if lname.startswith("claude-"):
        return Provider.ANTHROPIC.value
    # Handle Gemini models that don't start with google/ or gemini/
    if lname.startswith("gemini-"):  # Ensure this check happens
        return Provider.GEMINI.value
    if lname.startswith("grok-"):
        return Provider.GROK.value
    if lname.startswith("deepseek-"):
        return Provider.DEEPSEEK.value
    # Llama models might be Groq or Cerebras - cannot reliably infer from name alone.
    # Mistral models might be Mistral, OpenRouter, etc. - cannot reliably infer.
    # If it starts with mistral- but NOT mistralai/, it's ambiguous.

    # 3. Check if the name *exactly matches* a key in COST_PER_MILLION_TOKENS
    # This is useful for models like `mistralai/mistral-nemo` if they haven't been prefixed
    if model_name in COST_PER_MILLION_TOKENS:
        # Try inferring based *solely* on COST_PER_MILLION_TOKENS keys (less reliable)
        if model_name.startswith("mistralai/"):
            return Provider.OPENROUTER.value  # Heuristic: Nemo is likely OpenRouter
        if model_name.startswith("llama-"):  # Heuristic: Llama 3.3 is Groq, Llama 4 is Cerebras
            if "3.3" in model_name:
                return Provider.GROQ.value
            if "4-scout" in model_name:
                return Provider.CEREBRAS.value
    return None  # Cannot reliably infer


# Populate the map (used for informational purposes, e.g., /model list)
# The runtime logic relies on get_provider_from_model
for model_key in COST_PER_MILLION_TOKENS.keys():
    # Attempt inference using the refined logic
    inferred_provider = _infer_provider(model_key)
    if inferred_provider:
        MODEL_PROVIDER_MAP[model_key] = inferred_provider

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
    content: Union[str, List[Dict]]  # Content can be simple string or richer structure
    # Optional fields for provider-specific needs or errors
    is_error: NotRequired[bool]
    # Used internally to differentiate from regular user message after tool execution
    _is_tool_result: NotRequired[bool]


# Define the content type alias
InternalContent: TypeAlias = Union[str, List[Union[TextContentBlock, ToolUseContentBlock, ToolResultContentBlock]]]


# Define the main message structure
class InternalMessage(TypedDict):
    role: Literal["user", "assistant", "system"]  # Keep roles simple internally
    content: InternalContent
    # Optional: Add fields used during processing, like tool_use_id for linking
    # tool_use_id: NotRequired[str] # Example if needed for processing state


InternalMessageList = List[InternalMessage]  # Represents a list of messages
ContentDict = Dict[str, Any]
PartDict = Dict[str, Any]
FunctionResponseDict = Dict[str, Any]
FunctionCallDict = Dict[str, Any]
# =============================================================================


# Global flag for verbose logging (can be set by --verbose)
USE_VERBOSE_SESSION_LOGGING = False

# --- Set up Typer app ---
app = typer.Typer(help="🔌 Ultimate MCP Client - Multi-Provider Edition", context_settings={"help_option_names": ["--help", "-h"]})


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
                if len(self._buffer) > 100:
                    self._buffer.pop(0)
        else:
            self.original_stdout.write(text)

    def flush(self):
        if not self.active_stdio_servers:
            self.original_stdout.flush()
        else:
            sys.stderr.flush()

    def isatty(self):
        return self.original_stdout.isatty()

    def fileno(self):
        return self.original_stdout.fileno()

    def readable(self):
        return self.original_stdout.readable()

    def writable(self):
        return self.original_stdout.writable()


# Apply the protection wrapper
sys.stdout = StdioProtectionWrapper(sys.stdout)

# --- Rich Theme and Consoles ---
custom_theme = Theme(
    {
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
    }
)
console = Console(theme=custom_theme)
stderr_console = Console(theme=custom_theme, stderr=True, highlight=False)

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, format="%(message)s", datefmt="[%X]", handlers=[RichHandler(rich_tracebacks=True, markup=True, console=stderr_console)]
)
log = logging.getLogger("mcpclient_multi")  # Use a unique logger name
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)


# --- Signal Handling (force_exit_handler, atexit_handler, sigint_handler) ---
def force_exit_handler(is_force=False):
    print("\nForcing exit and cleaning up resources...")
    if is_force:
        print("Emergency shutdown initiated!")
        if "app" in globals() and hasattr(app, "mcp_client"):
            if hasattr(app.mcp_client, "server_manager"):
                for name, process in app.mcp_client.server_manager.processes.items():
                    try:
                        if process.returncode is None:
                            print(f"Force killing process {name} (PID {process.pid})")
                            process.kill()
                    except Exception:
                        pass
        os._exit(1)
    sys.exit(1)


def atexit_handler():
    print("\nShutting down and cleaning resources...")
    # Ensure cleanup happens via MCPClient.close() called in main_async finally block


# Signal handler for SIGINT (Ctrl+C)
def sigint_handler(signum, frame):
    print("\nCtrl+C detected.")
    client_instance = getattr(app, "mcp_client", None)
    active_query_task = getattr(client_instance, "current_query_task", None) if client_instance else None

    if active_query_task and not active_query_task.done():
        print("Attempting to abort current request... (Press Ctrl+C again to force exit)")
        try:
            active_query_task.cancel()
        except Exception as e:
            print(f"Error trying to cancel task: {e}")
        return  # Don't increment counter or exit yet

    sigint_handler.counter += 1
    if sigint_handler.counter >= 2:
        print("Multiple interrupts detected. Forcing immediate exit!")
        force_exit_handler(is_force=True)

    print("Shutting down...")
    try:
        sys.exit(1)  # Triggers atexit
    except SystemExit:
        pass
    except Exception as e:
        print(f"Error during clean shutdown attempt: {e}. Forcing exit!")
        force_exit_handler(is_force=True)


sigint_handler.counter = 0
signal.signal(signal.SIGINT, sigint_handler)
atexit.register(atexit_handler)


def _fmt_id(val: Any, length: int = 8) -> str:
    s = str(val) if val is not None else "?"
    return s[:length] if len(s) >= length else s


# --- Pydantic Models (ServerAddRequest, ConfigUpdateRequest, etc. - Update ConfigUpdateRequest) ---
class WebSocketMessage(BaseModel):
    type: str
    payload: Any = None


class SetModelRequest(BaseModel):
    model: str = Field(..., min_length=1)


class ServerType(Enum):
    STDIO = "stdio"
    SSE = "sse"


class ServerAddRequest(BaseModel):
    name: str
    type: ServerType
    path: str
    argsString: Optional[str] = ""


class AgentStartRequest(BaseModel):
    goal: str
    max_loops: int = Field(default=50, gt=0)
    llm_model: Optional[str] = None  # Optional model override for this run


class AgentInjectThoughtRequest(BaseModel):
    content: str
    thought_type: str = Field(default="user_guidance")


# Model for the GET /api/config response (excluding sensitive data)
class ConfigGetResponse(BaseModel):
    default_model: str = Field(..., alias="defaultModel")
    default_cheap_and_fast_model: str = Field(..., alias="defaultCheapAndFastModel")
    default_max_tokens: int = Field(..., alias="defaultMaxTokens")
    history_size: int = Field(..., alias="historySize")
    auto_discover: bool = Field(..., alias="autoDiscover")
    discovery_paths: List[str] = Field(..., alias="discoveryPaths")
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
        populate_by_name = True  # Allow using aliases in responses


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
    default_cheap_and_fast_model: Optional[str] = Field(None, alias="defaultCheapAndFastModel")
    default_max_tokens: Optional[int] = Field(None, alias="defaultMaxTokens")
    history_size: Optional[int] = Field(None, alias="historySize")
    auto_discover: Optional[bool] = Field(None, alias="autoDiscover")
    discovery_paths: Optional[List[str]] = Field(None, alias="discoveryPaths")
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
        populate_by_name = True  # Allow using aliases in requests
        extra = "ignore"  # Ignore extra fields in the request


class ToolExecuteRequest(BaseModel):
    tool_name: str
    params: Dict[str, Any]


class GraphNodeData(BaseModel):
    id: str
    name: str
    parent_id: Optional[str] = Field(None, alias="parentId")  # Use alias for JS
    model: Optional[str] = None  # Model used for this node/branch
    created_at: str = Field(..., alias="createdAt")
    modified_at: str = Field(..., alias="modifiedAt")
    message_count: int = Field(..., alias="messageCount")

    class Config:
        populate_by_name = True  # Allow population by alias


class NodeRenameRequest(BaseModel):
    new_name: str = Field(..., min_length=1)


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
        if isinstance(path_str, str) and len(path_str) > 2 and path_str[1] == ":" and path_str[2] in ["\\", "/"] and path_str[0].isalpha():
            try:  # Added try-except for robustness during conversion
                drive_letter = path_str[0].lower()
                # path_str[2:] correctly gets the part after 'C:'
                # .replace("\\", "/") correctly handles the single literal backslash in the Python string
                rest_of_path = path_str[3:].replace("\\", "/")  # Use index 3 to skip ':\' or ':/'
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
    adapted_command = command  # Default to original command
    if isinstance(command, str) and ":" in command and ("\\" in command or "/" in command):
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
            adapted_args.append(arg)  # Keep non-string args (like numbers, bools) as is

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
    except (NameError, AttributeError):
        pass
    if has_stdio_servers:
        with redirect_stdout(sys.stderr):
            yield
    else:
        yield


def get_safe_console():
    has_stdio_servers = False
    try:
        if hasattr(app, "mcp_client") and app.mcp_client and hasattr(app.mcp_client, "server_manager"):
            for name, server in app.mcp_client.server_manager.config.servers.items():
                if server.type == ServerType.STDIO and name in app.mcp_client.server_manager.active_sessions:
                    has_stdio_servers = True
                    # Optional: Add back warning logic if needed
                    break
    except (NameError, AttributeError):
        pass
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


###############################################################################
# 1) JSON repair helper
###############################################################################
_single_quote_kv = re.compile(r"([{,]\s*)\'([^\']+)\'\s*:")
_single_quote_val = re.compile(r":\s*\'([^\']*)\'")
_unquoted_key = re.compile(r"([{,]\s*)([A-Za-z_][\w\-]*)(\s*:)")
_trailing_commas = re.compile(r",\s*([}\]])")
_bool_none = re.compile(r"\b(?:True|False|None)\b")
_unterminated_qstr = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)(?=,|\s*[}\]]|$)')
_BRACE_PAIRS = {"{": "}", "[": "]"}


def _balance_brackets(s: str) -> str:
    stack: List[str] = []
    for ch in s:
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack and _BRACE_PAIRS[stack[-1]] == ch:
            stack.pop()
    return s + "".join(_BRACE_PAIRS[c] for c in reversed(stack))


def _repair_json(text: str, aggressive: bool = False) -> str:
    """Attempt to coerce *text* into valid JSON; optionally turn on aggressive mode."""
    if not text:
        return text
    s = text.strip()
    # Fast-path: looks like valid JSON already
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    # ---- basic regex fixes --------------------------------------------------
    s = _trailing_commas.sub(r"\1", s)  # remove trailing commas
    s = _single_quote_kv.sub(r'\1"\2":', s)  # keys in single quotes
    s = _single_quote_val.sub(r': "\1"', s)  # values in single quotes
    s = _unquoted_key.sub(r'\1"\2"\3', s)  # bare keys
    s = _bool_none.sub(lambda m: m.group(0).lower().replace("none", "null"), s)  # Python -> JSON
    # ---- early retry --------------------------------------------------------
    try:
        json.loads(s)
        return s
    except Exception:
        pass
    if not aggressive:
        return s  # caller may decide to escalate
    # ---- aggressive fixes ---------------------------------------------------
    # balance quotes
    if s.count('"') % 2:
        s += '"'
    s = _unterminated_qstr.sub(r'"\1"', s)
    # ensure overall structural integrity
    s = _balance_brackets(s)
    # final resorts: json5, ast.literal_eval
    try:
        import json5

        return json.dumps(json5.loads(s))  # noqa: I001
    except Exception:
        pass
    try:
        return json.dumps(ast.literal_eval(s))
    except Exception:
        pass
    return s  # best effort


###############################################################################
# 2) Extract JSON from markdown fences
###############################################################################
_FENCE_RX = re.compile(
    r"(?:```|~~~)\s*(?:json)?\s*\n(.*?)\n(?:```|~~~)",
    re.IGNORECASE | re.DOTALL,
)


def extract_json_from_markdown(text: str) -> str:
    """Return the first reparable JSON-looking blob found in *text*."""
    if not text:
        return ""
    cleaned = text.strip()
    # 1. prefer fenced code blocks
    for blob in _FENCE_RX.findall(cleaned):
        for mode in (False, True):
            repaired = _repair_json(blob, aggressive=mode)
            try:
                json.loads(repaired)
                return repaired
            except Exception:
                pass
    # 2. fall back to first brace/bracket onward
    m = re.search(r"[{\[]", cleaned)
    if m:
        candidate = cleaned[m.start() :]
        for mode in (False, True):
            repaired = _repair_json(candidate, aggressive=mode)
            try:
                json.loads(repaired)
                return repaired
            except Exception:
                pass
    return cleaned  # give caller the raw text as last resort


###############################################################################
# 3) Robustly parse MCP tool results
###############################################################################


def _strip_markdown_fences(txt: str) -> str:
    txt = txt.strip()
    if txt.startswith("```"):
        first_newline = txt.find("\n")
        if first_newline != -1 and txt.rstrip().endswith("```"):
            return txt[first_newline + 1 : txt.rfind("```")].strip()
    return txt


def robust_parse_mcp_tool_content(content_from_call_tool_result: Any) -> Dict[str, Any]:
    """
    Extracts a JSON string from CallToolResult.content (expected to be List[TextContent]
    from mcp.py SSE client) and parses it.
    Returns a dictionary: successfully parsed JSON, or an error structure.
    """
    log.debug(
        f"ROBUST_PARSE_V_FINAL: Received raw content of type: {type(content_from_call_tool_result)}, "
        f"Preview: {str(content_from_call_tool_result)[:500]}"
    )

    extracted_json_string: Optional[str] = None

    # Primary expected path: content is a list, first item is TextContent
    if isinstance(content_from_call_tool_result, list) and content_from_call_tool_result:
        first_block = content_from_call_tool_result[0]
        log.debug(f"ROBUST_PARSE_V_FINAL: Processing first block from list. Type: {type(first_block)}, Value: {str(first_block)[:200]}")

        # Explicitly check if it's an mcp.types.TextContent instance
        if isinstance(first_block, TextContent):  # Requires TextContent to be imported
            if first_block.type == "text" and isinstance(first_block.text, str):
                extracted_json_string = first_block.text
                log.debug(
                    f"ROBUST_PARSE_V_FINAL: Extracted via isinstance(TextContent): '{extracted_json_string[:100]}...' (Length: {len(extracted_json_string)})"
                )
            else:
                log.warning(f"ROBUST_PARSE_V_FINAL: Is TextContent, but type is not 'text' or .text not string. Type: {first_block.type}")
        # Fallback check using hasattr (duck-typing) if isinstance fails or TextContent not imported
        elif (
            dataclasses.is_dataclass(first_block)
            and hasattr(first_block, "type")
            and getattr(first_block, "type") == "text" # noqa: B009
            and hasattr(first_block, "text")
            and isinstance(getattr(first_block, "text"), str) # noqa: B009
        ):  # noqa: B009
            extracted_json_string = getattr(first_block, "text")  # noqa: B009
            log.debug(
                f"ROBUST_PARSE_V_FINAL: Extracted via hasattr (dataclass): '{extracted_json_string[:100]}...' (Length: {len(extracted_json_string)})"
            )
        elif isinstance(first_block, dict) and first_block.get("type") == "text" and isinstance(first_block.get("text"), str):
            extracted_json_string = first_block.get("text")
            log.debug(
                f"ROBUST_PARSE_V_FINAL: Extracted from dict-like TextContent: '{extracted_json_string[:100]}...' (Length: {len(extracted_json_string)})"
            )
        else:
            log.warning(
                f"ROBUST_PARSE_V_FINAL: First block in list not recognized TextContent. Type: {type(first_block)}. Content: {str(first_block)[:150]}"
            )

    elif isinstance(content_from_call_tool_result, TextContent):  # Handle single TextContent not in a list
        if content_from_call_tool_result.type == "text" and isinstance(content_from_call_tool_result.text, str):
            extracted_json_string = content_from_call_tool_result.text
            log.debug(
                f"ROBUST_PARSE_V_FINAL: Extracted from standalone TextContent object: '{extracted_json_string[:100]}...' (Length: {len(extracted_json_string)})"
            )
        else:
            log.warning(
                f"ROBUST_PARSE_V_FINAL: Standalone TextContent object, but type is not 'text' or .text not string. Type: {content_from_call_tool_result.type}"
            )

    elif isinstance(content_from_call_tool_result, str):  # If mcp.py client somehow already gave a string
        extracted_json_string = content_from_call_tool_result
        log.debug(f"ROBUST_PARSE_V_FINAL: Content was already a string: '{extracted_json_string[:100]}...' (Length: {len(extracted_json_string)})")

    elif isinstance(content_from_call_tool_result, dict):  # If it's already a dict (e.g. from a direct non-SSE tool, or if wrapper already parsed)
        log.info(f"ROBUST_PARSE_V_FINAL: Content is already a dict. Returning as is. Keys: {list(content_from_call_tool_result.keys())}")
        # Ensure it has a "success" key if it's meant to be a UMS-style response for the agent
        if "success" not in content_from_call_tool_result and not (
            "error" in content_from_call_tool_result and "error_type" in content_from_call_tool_result
        ):
            log.warning(
                "ROBUST_PARSE_V_FINAL: Input dict lacks 'success' key. Adding 'success: True' assuming it's data, not an error structure from this func."
            )
            content_from_call_tool_result["success"] = True  # Agent expects this
        return content_from_call_tool_result

    elif content_from_call_tool_result is None:
        log.warning("ROBUST_PARSE_V_FINAL: Received None as input content.")
        return {"success": False, "error": "Tool content was None.", "error_type": "NoneContentReceived_MCPC"}

    else:  # Fallback for completely unexpected types
        log.error(
            f"ROBUST_PARSE_V_FINAL: Unhandled content type: {type(content_from_call_tool_result)}. Attempting str(). Content: {str(content_from_call_tool_result)[:200]}"
        )
        extracted_json_string = str(content_from_call_tool_result)  # Last resort

    if not extracted_json_string:  # After all attempts, if still no string
        error_msg = "Could not extract any usable JSON string from tool result content."
        log.error(f"ROBUST_PARSE_V_FINAL: {error_msg} Original raw content preview: {str(content_from_call_tool_result)[:200]}")
        return {
            "success": False,
            "error": error_msg,
            "error_type": "NoJSONStringExtracted_MCPC_Final",
            "original_content_preview": str(content_from_call_tool_result)[:200],
        }

    # Log the string that will be attempted for JSON parsing
    log.info(
        f"ROBUST_PARSE_V_FINAL: Attempting to parse JSON from extracted string (length: {len(extracted_json_string)}): "
        f"'{extracted_json_string[:200]}{'...[TRUNCATED PREVIEW]' if len(extracted_json_string) > 200 else ''}'"
    )
    # If this log shows `{"wo...` with a very short length, the problem is upstream (MCP server/library).

    candidate_for_json = _strip_markdown_fences(extracted_json_string)  # For LLM-generated JSON that might include fences
    candidate_for_json = textwrap.dedent(candidate_for_json).strip()

    if not candidate_for_json:
        error_msg = "Extracted JSON string became empty after stripping markdown/dedenting."
        log.error(f"ROBUST_PARSE_V_FINAL: {error_msg} Original extracted string: '{extracted_json_string[:200]}...'")
        return {
            "success": False,
            "error": error_msg,
            "error_type": "EmptyAfterCleaning_MCPC",
            "original_extracted_string_preview": extracted_json_string[:200],
        }

    try:
        parsed_data = json.loads(candidate_for_json)

        if not isinstance(parsed_data, dict):
            log.info(f"ROBUST_PARSE_V_FINAL: Parsed to valid non-dictionary JSON (type: {type(parsed_data)}). Wrapping.")
            return {
                "success": True,
                "data": parsed_data,
                "_mcp_client_non_dict_payload_note": "Original tool payload was valid JSON but not a dictionary.",
            }

        if "success" not in parsed_data:
            log.warning(
                f"ROBUST_PARSE_V_FINAL: Parsed dict has no 'success' key. Assuming UMS tool success or non-UMS payload. Adding 'success: True'. Keys: {list(parsed_data.keys())}"
            )
            if not ("error" in parsed_data and "error_type" in parsed_data):
                parsed_data["success"] = True

        log.debug(f"ROBUST_PARSE_V_FINAL: Successfully parsed to dict. Success key from payload: {parsed_data.get('success', 'N/A')}")
        return parsed_data

    except json.JSONDecodeError as e:
        error_msg = f"JSONDecodeError: {e}. Input to json.loads (first 200 chars): '{candidate_for_json[:200]}...'"
        log.error(f"ROBUST_PARSE_V_FINAL: {error_msg}")
        if len(candidate_for_json) < 2000:
            log.error(f"ROBUST_PARSE_V_FINAL: Full candidate string that failed json.loads: {candidate_for_json}")
        else:
            log.error(f"ROBUST_PARSE_V_FINAL: Candidate string (first 2000 chars) for failed json.loads: {candidate_for_json[:2000]}...")
        return {
            "success": False,
            "error": error_msg,
            "error_type": "JSONDecodeError_MCPC_Final",
            "final_candidate_preview": candidate_for_json[:200],
            "original_extracted_string_preview": extracted_json_string[:200],
        }
    except Exception as e_final_parse:
        error_msg = f"Unexpected error during final parsing attempt: {e_final_parse}"
        log.error(f"ROBUST_PARSE_V_FINAL: {error_msg}. Candidate: '{candidate_for_json[:200]}...'", exc_info=True)
        return {"success": False, "error": error_msg, "error_type": "GenericFinalParseError_MCPC_Final"}


# --- Directory Constants ---
PROJECT_ROOT = Path(__file__).parent.resolve()
CONFIG_DIR = PROJECT_ROOT / ".mcpclient_multi_config"  # New config dir name
CONFIG_FILE = CONFIG_DIR / "config.yaml"
HISTORY_FILE = CONFIG_DIR / "history.json"
SERVER_DIR = CONFIG_DIR / "servers"
CACHE_DIR = CONFIG_DIR / "cache"
REGISTRY_DIR = CONFIG_DIR / "registry"
MAX_HISTORY_ENTRIES = 300
REGISTRY_URLS = []
BLACKLIST_FILE = CONFIG_DIR / "tool_blacklist.json"

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
    meter_provider = MeterProvider()  # Dummy
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
        if len(self.response_times) > 100:
            self.response_times = self.response_times[-100:]
        self.avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.0

    def update_status(self) -> None:
        self.error_rate = self.error_count / max(1, self.request_count)
        if self.error_rate > 0.5 or self.avg_response_time > 10.0:
            self.status = ServerStatus.ERROR
        elif self.error_rate > 0.1 or self.avg_response_time > 5.0:
            self.status = ServerStatus.DEGRADED
        elif self.request_count > 0:
            self.status = ServerStatus.HEALTHY  # Only healthy if requests made
        else:
            self.status = ServerStatus.UNKNOWN


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
        if self.execution_times:
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


# --- ConversationNode Class (Using Type Alias) ---
@dataclass
class ConversationNode:
    id: str
    messages: InternalMessageList = field(default_factory=list)  # Use type alias
    parent: Optional["ConversationNode"] = None
    children: List["ConversationNode"] = field(default_factory=list)
    name: str = "Root"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    model: str = ""  # Store model name used for the node

    def add_message(self, message: InternalMessage) -> None:  # Use type alias
        self.messages.append(message)
        self.modified_at = datetime.now()

    def add_child(self, child: "ConversationNode") -> None:
        self.children.append(child)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the node to a dictionary suitable for JSON serialization."""
        # Directly return the messages list. We'll handle potential non-serializable
        # content within this list using json.dumps(default=...) later.
        # The TypedDicts within the list behave like dicts at runtime.
        return {
            "id": self.id,
            "name": self.name,
            "messages": self.messages,  # Pass the list directly
            "parent_id": self.parent.id if self.parent else None,
            "children_ids": [child.id for child in self.children],
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "model": self.model,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationNode":
        return cls(
            id=data["id"],
            messages=data["messages"],
            name=data["name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            modified_at=datetime.fromisoformat(data["modified_at"]),
            model=data.get("model", ""),
        )  # Load model if present


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


class ToolBlacklistManager:
    def __init__(self, filepath: Path = BLACKLIST_FILE):
        self.filepath = filepath
        self.blacklist: Dict[str, List[str]] = {}  # provider -> list of blocked tool names
        self.load()

    def load(self):
        """Loads the blacklist from the JSON file."""
        try:
            if self.filepath.exists():
                with open(self.filepath, "r", encoding="utf-8") as f:
                    self.blacklist = json.load(f)
                log.info(f"Loaded tool blacklist from {self.filepath}")
            else:
                self.blacklist = {}
        except (json.JSONDecodeError, IOError, Exception) as e:
            log.error(f"Error loading tool blacklist {self.filepath}: {e}", exc_info=True)
            self.blacklist = {}  # Start fresh on error

    def save(self):
        """Saves the current blacklist to the JSON file."""
        try:
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump(self.blacklist, f, indent=2)
            log.debug(f"Saved tool blacklist to {self.filepath}")
        except (IOError, Exception) as e:
            log.error(f"Error saving tool blacklist {self.filepath}: {e}", exc_info=True)

    def add_to_blacklist(self, provider: str, tool_names: List[str]):
        """Adds tool names to the blacklist for a specific provider."""
        if not tool_names:
            return
        current_list = self.blacklist.setdefault(provider, [])
        added = False
        for tool_name in tool_names:
            if tool_name not in current_list:
                current_list.append(tool_name)
                added = True
        if added:
            log.info(f"Added tools to {provider} blacklist: {tool_names}")
            self.save()  # Save immediately after adding

    def is_blacklisted(self, provider: str, tool_name: str) -> bool:
        """Checks if a tool is blacklisted for a specific provider."""
        return tool_name in self.blacklist.get(provider, [])

    def get_blacklisted_tools(self, provider: str) -> List[str]:
        """Gets the list of blacklisted tools for a provider."""
        return self.blacklist.get(provider, []).copy()


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
            if min_rating > 0.0:  # Only add if > 0
                params["min_rating"] = min_rating

            try:
                response = await self.http_client.get(f"{registry_url}/servers", params=params, timeout=5.0)
                if response.status_code == 200:
                    try:
                        servers_data = response.json()
                        servers = servers_data.get("servers", [])
                        for server in servers:
                            server["registry_url"] = registry_url  # Add source registry URL
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
                self.log.error(f"Unexpected error querying registry {registry_url}: {e}", exc_info=True)  # Include traceback for unexpected errors
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
                response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx
                server_details = response.json()
                server_details["registry_url"] = url  # Add source
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
            _ = ServiceBrowser  # Check if imported
            _ = Zeroconf
        except NameError:
            log.warning("Zeroconf library not available. Install 'zeroconf'. Local discovery disabled.")
            self.zeroconf = None
            self.browser = None
            return

        try:
            # Nested class for the listener
            class MCPServiceListener:
                def __init__(self, registry_instance: "ServerRegistry"):
                    self.registry = registry_instance
                    self.log = registry_instance.log  # Use parent's logger

                def add_service(self, zeroconf_obj: Zeroconf, service_type: str, name: str):
                    """Callback when a service is added or updated."""
                    self.log.debug(f"mDNS Add Service Triggered: type={service_type}, name={name}")
                    info: Optional[ServiceInfo] = None
                    try:
                        # Use timeout for get_service_info
                        info = zeroconf_obj.get_service_info(service_type, name, timeout=1000)  # 1 second timeout
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
                                    key = k_bytes.decode("utf-8")
                                    value = v_bytes.decode("utf-8")
                                    props[key] = value
                                except UnicodeDecodeError:
                                    self.log.warning(f"Skipping non-UTF8 property key/value for {name}")
                                    continue

                        # Extract details from properties
                        server_protocol = props.get("type", "sse").lower()  # Default to sse
                        version_str = props.get("version")
                        version_obj = None
                        if version_str:
                            try:
                                version_obj = ServerVersion.from_string(version_str)  # Use the dataclass if defined
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
                            "type": server_protocol,  # 'sse' or 'stdio' etc.
                            "url": f"http://{host}:{port}" if server_protocol == "sse" else f"mDNS:{server_name}",  # Adjust URL based on type
                            "properties": props,
                            "version": version_obj,  # Store the object or None
                            "categories": categories,
                            "description": description,
                            "discovered_via": "mdns",
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
            self.zeroconf = None  # Ensure reset on error
            self.browser = None
        except Exception as e:
            # Catch other potential setup errors
            self.log.error(f"Unexpected error during Zeroconf setup: {e}", exc_info=True)
            self.zeroconf = None  # Ensure reset on error
            self.browser = None

    def stop_local_discovery(self):
        """Stops local mDNS discovery."""
        if self.browser:
            self.log.info("Stopping Zeroconf ServiceBrowser.")
            # Note: ServiceBrowser doesn't have an explicit stop, closing Zeroconf handles it.
            self.browser = None  # Clear the browser reference
        if self.zeroconf:
            self.log.info("Closing Zeroconf instance.")
            try:
                self.zeroconf.close()
            except Exception as e:
                self.log.error(f"Error closing Zeroconf: {e}", exc_info=True)
            finally:
                self.zeroconf = None  # Ensure zeroconf is None after close attempt
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
        server_info = self.servers.get(server_id)  # Check configured first
        if not server_info and server_id in self.discovered_servers:
            server_info = self.discovered_servers[server_id]

        registry_url_to_use = None
        if server_info and isinstance(server_info, dict):  # Check if it's a dict before accessing keys
            registry_url_to_use = server_info.get("registry_url")
        elif hasattr(server_info, "registry_url"):  # Handle dataclass case (ServerConfig)
            registry_url_to_use = server_info.registry_url

        if not registry_url_to_use:
            self.log.warning(f"Cannot rate server '{server_id}': Registry URL unknown.")
            return False

        rate_url = f"{registry_url_to_use}/servers/{server_id}/rate"
        payload = {"rating": rating}

        try:
            response = await self.http_client.post(rate_url, json=payload)
            response.raise_for_status()  # Check for HTTP errors
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
        self.stop_local_discovery()  # Ensure mDNS stops
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
            self.disk_cache = None  # Disable disk cache if init fails

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
        return 3600  # Default 1 hour

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
            raise  # Re-raise the TypeError so caller knows key generation failed

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
                del self.memory_cache[key]  # Remove expired from memory

        # Check disk cache if enabled
        if self.disk_cache:
            try:
                # Check existence *before* getting to potentially avoid errors
                if key in self.disk_cache:
                    entry = self.disk_cache.get(key)  # Use get for safer retrieval
                    if isinstance(entry, CacheEntry) and not entry.is_expired():
                        log.debug(f"Cache HIT (Disk): {key}")
                        self.memory_cache[key] = entry  # Promote to memory
                        return entry.result
                    else:
                        # Entry is expired or invalid type
                        log.debug(f"Cache STALE/INVALID (Disk): {key}")
                        # Safely delete expired/invalid entry
                        with suppress(KeyError, Exception):  # Suppress errors during delete
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

        expires_at = (datetime.now() + timedelta(seconds=ttl)) if ttl >= 0 else None  # Allow ttl=0 for non-expiring? Changed to >=0
        if ttl < 0:
            log.warning(f"Negative TTL ({ttl}) provided for caching tool '{tool_name}'. Cache entry will not expire.")
            expires_at = None  # Treat negative TTL as non-expiring

        entry = CacheEntry(result=result, created_at=datetime.now(), expires_at=expires_at, tool_name=tool_name, parameters_hash=key.split(":")[-1])

        # Set memory cache
        self.memory_cache[key] = entry
        log.debug(f"Cache SET (Memory): {key} (TTL: {ttl}s)")

        # Set disk cache if enabled
        if self.disk_cache:
            try:
                self.disk_cache.set(key, entry, expire=ttl if ttl >= 0 else None)  # Use diskcache expire param
                log.debug(f"Cache SET (Disk): {key} (TTL: {ttl}s)")
            except Exception as e:
                log.warning(f"Disk cache SET error for key '{key}': {e}")

    # --- invalidate Method (Synchronous Fix) ---
    def invalidate(self, tool_name=None, params=None):
        if tool_name and params:
            try:
                key = self.generate_key(tool_name, params)
            except TypeError:
                key = None  # Cannot invalidate if key cannot be generated
            if key:
                log.debug(f"Invalidating specific cache key: {key}")
                if key in self.memory_cache:
                    del self.memory_cache[key]
                if self.disk_cache:
                    # Use synchronous suppress for KeyError and potentially others
                    with suppress(KeyError, OSError, EOFError, diskcache.Timeout, Exception):
                        if key in self.disk_cache:  # Check before deleting
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
                    with suppress(Exception):  # Suppress errors during iteration itself
                        for key in self.disk_cache.iterkeys():  # iterkeys is sync
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
        for key in list(self.memory_cache.keys()):  # Iterate over a copy
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
        fork_id = str(uuid.uuid4())
        fork_name = name or f"Fork {len(self.current_node.children) + 1}"
        new_node = ConversationNode(
            id=fork_id, name=fork_name, parent=self.current_node, messages=self.current_node.messages.copy(), model=self.current_node.model
        )
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

    @staticmethod
    def _json_serializer_default(obj):
        """Custom serializer for objects json.dumps doesn't handle by default."""
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        # Add handling for other specific types if needed
        # ...
        # Fallback: Convert any other unknown object to its string representation
        # This prevents TypeErrors for most custom objects or complex types.
        log.debug(f"Using string fallback for non-serializable type: {type(obj)}")
        return str(obj)

    async def save(self, file_path: str):
        """Saves the conversation graph to a JSON file."""
        # Prepare the main data structure
        data_to_save = {
            "current_node_id": self.current_node.id,
            # Call the node's to_dict method, which now returns the messages list directly
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
        }

        try:
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                # Use the custom default handler for non-serializable objects
                json_string = json.dumps(
                    data_to_save,
                    indent=2,
                    ensure_ascii=False,
                    default=self._json_serializer_default,  # Use the custom handler
                )
                await f.write(json_string)
            log.debug(f"Saved graph to {file_path} using simplified serialization.")
        except (IOError, TypeError) as e:
            # TypeError might still occur if the default handler fails unexpectedly
            log.error(f"Could not save graph {file_path} (Serialization Error?): {e}", exc_info=True)
        except Exception as e:
            log.error(f"Unexpected error saving graph {file_path}: {e}", exc_info=True)

    @classmethod
    async def load(cls, file_path: str) -> "ConversationGraph":
        file_path_obj = Path(file_path)
        try:
            async with aiofiles.open(file_path_obj, "r") as f:
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
    Provider.GEMINI.value: "GEMINI_API_KEY",  # Use GEMINI_API_KEY
    Provider.GROK.value: "GROK_API_KEY",
    Provider.DEEPSEEK.value: "DEEPSEEK_API_KEY",
    Provider.MISTRAL.value: "MISTRAL_API_KEY",
    Provider.GROQ.value: "GROQ_API_KEY",
    Provider.CEREBRAS.value: "CEREBRAS_API_KEY",
    Provider.OPENROUTER.value: "OPENROUTER_API_KEY",
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
    Provider.OPENROUTER.value: "openrouter_api_key",
}

PROVIDER_CONFIG_URL_ATTR_MAP = {
    Provider.OPENAI.value: "openai_base_url",
    Provider.GEMINI.value: "gemini_base_url",
    Provider.GROK.value: "grok_base_url",
    Provider.DEEPSEEK.value: "deepseek_base_url",
    Provider.MISTRAL.value: "mistral_base_url",
    Provider.GROQ.value: "groq_base_url",
    Provider.CEREBRAS.value: "cerebras_base_url",
    Provider.OPENROUTER.value: "openrouter_base_url",
}

PROVIDER_ENV_URL_MAP = {
    Provider.OPENAI.value: "OPENAI_BASE_URL",
    Provider.GEMINI.value: "GEMINI_BASE_URL",
    Provider.GROK.value: "GROK_BASE_URL",
    Provider.DEEPSEEK.value: "DEEPSEEK_BASE_URL",
    Provider.MISTRAL.value: "MISTRAL_BASE_URL",
    Provider.GROQ.value: "GROQ_BASE_URL",
    Provider.CEREBRAS.value: "CEREBRAS_BASE_URL",
    Provider.OPENROUTER.value: "OPENROUTER_BASE_URL",
}

# Mapping for simple settings <-> Env Var Names
SIMPLE_SETTINGS_ENV_MAP = {
    "DEFAULT_MODEL": "default_model",
    "DEFAULT_CHEAP_AND_FAST_MODEL": "default_cheap_and_fast_model",
    "DEFAULT_MAX_TOKENS": "default_max_tokens",
    "HISTORY_SIZE": "history_size",
    "AUTO_DISCOVER": "auto_discover",
    "DISCOVERY_PATHS": "discovery_paths",
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
        self.agent_state_file_path: str = str(CONFIG_DIR / "agent_loop_state_v4.2_direct_mcp_names.json")  # Default agent state file
        self._set_defaults()
        self.dotenv_path = find_dotenv(raise_error_if_not_found=False, usecwd=True)
        if self.dotenv_path:
            log.info(f"Located .env file at: {self.dotenv_path}")
            load_dotenv(dotenv_path=self.dotenv_path, override=True)  # Load .env into os.environ
            log.debug(f"Loaded environment variables from {self.dotenv_path}")
        else:
            # Decide where a .env file *should* go if created later
            self.dotenv_path = str(Path.cwd() / ".env")  # Default to current working dir
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
        self.gemini_base_url: Optional[str] = "https://generativelanguage.googleapis.com/v1beta/openai/"  # Special URL for openai-compatible API
        self.grok_base_url: Optional[str] = "https://api.x.ai/v1"
        self.deepseek_base_url: Optional[str] = "https://api.deepseek.com/v1"
        self.mistral_base_url: Optional[str] = "https://api.mistral.ai/v1"
        self.groq_base_url: Optional[str] = "https://api.groq.com/openai/v1"
        self.cerebras_base_url: Optional[str] = "https://api.cerebras.ai/v1"
        self.openrouter_base_url: Optional[str] = "https://openrouter.ai/api/v1"

        # Default model
        self.default_model: str = "gpt-4.1"
        self.default_cheap_and_fast_model: str = "gemini-2.5-flash-preview-05-20"  # Default cheap model, will be overridden by .env

        # Other settings
        self.default_max_tokens: int = 8000
        self.history_size: int = MAX_HISTORY_ENTRIES
        self.auto_discover: bool = True
        default_discovery_paths: List[str] = [
            str(SERVER_DIR),
            os.path.expanduser("~/mcp-servers"),
            os.path.expanduser("~/modelcontextprotocol/servers"),
        ]
        self.discovery_paths: List[str] = default_discovery_paths
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
        self.agent_state_file_path: str = str(CONFIG_DIR / "agent_loop_state_v4.2_direct_mcp_names.json")

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
                        parsed_value = env_value.lower() in ("true", "1", "t", "yes", "y")
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
                            continue  # Skip setting attribute if parsing fails

                    else:  # Default to string
                        parsed_value = env_value

                    # Only update if the value actually changed from the default/YAML loaded value
                    current_val = getattr(self, attr_name, None)
                    # Need careful comparison for lists/dicts
                    if isinstance(current_val, list) and isinstance(parsed_value, list):
                        if set(current_val) != set(parsed_value):  # Compare sets for lists
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
        skip_attributes = {"decouple_instance", "dotenv_path"}  # Attributes to skip saving

        for attr_name, attr_value in self.__dict__.items():
            if attr_name.startswith("_") or callable(attr_value) or attr_name in skip_attributes:
                continue

            if attr_name == "servers":
                serialized_servers: Dict[str, Dict[str, Any]] = {}
                for name, server_config in self.servers.items():
                    # Simplified server data for YAML (excluding volatile metrics unless needed)
                    server_data = {
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
                        "retry_policy": server_config.retry_policy,
                        "registry_url": server_config.registry_url,
                        "capabilities": server_config.capabilities,
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
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
            log.info(f"Loading configuration from YAML file: {CONFIG_FILE}")
        except Exception as e:
            log.error(f"Error loading YAML config {CONFIG_FILE}: {e}", exc_info=True)
            return

        # Iterate through loaded data and update self attributes
        for key, value in config_data.items():
            if key == "servers":
                self.servers = {}  # Reset before loading
                if isinstance(value, dict):
                    for server_name, server_data in value.items():
                        if not isinstance(server_data, dict):
                            continue
                        try:
                            srv_type = ServerType(server_data.get("type", "stdio"))
                            version_str = server_data.get("version")
                            version = ServerVersion.from_string(version_str) if version_str else None
                            # Initialize metrics, don't load from YAML by default unless needed
                            metrics = ServerMetrics()
                            # Create ServerConfig, handle missing keys gracefully
                            default_config = ServerConfig(name=server_name, type=srv_type, path="")
                            server_kwargs = {
                                k: v for k, v in server_data.items() if hasattr(default_config, k) and k not in ["type", "version", "metrics"]
                            }
                            server_kwargs["type"] = srv_type
                            server_kwargs["version"] = version
                            server_kwargs["metrics"] = metrics
                            server_kwargs["name"] = server_name
                            self.servers[server_name] = ServerConfig(**server_kwargs)
                        except Exception as server_load_err:
                            log.warning(f"Skipping server '{server_name}' from YAML: {server_load_err}", exc_info=True)
                else:
                    log.warning("'servers' key in YAML is not a dict.")
            elif key == "cache_ttl_mapping":
                if isinstance(value, dict):
                    valid_mapping = {}
                    for k, v in value.items():
                        if isinstance(k, str) and isinstance(v, int):
                            valid_mapping[k] = v
                        else:
                            log.warning(f"Invalid YAML cache_ttl_mapping: K='{k}', V='{v}'.")
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
                    elif current_type in [int, float, bool, list, str]:  # Attempt basic type conversion
                        if current_type is bool:
                            converted_value = str(value).lower() in ("true", "1", "t", "yes", "y")
                        elif current_type is list and isinstance(value, list):  # Ensure list conversion takes list
                            converted_value = value  # Assume YAML list is correct
                        else:
                            converted_value = current_type(value)  # Try direct conversion
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
            async with aiofiles.open(CONFIG_FILE, "w", encoding="utf-8") as f:
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
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
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
        if not HISTORY_FILE.exists():
            return
        try:
            with open(HISTORY_FILE, "r") as f:
                data_str = f.read()
            self._load_history_data(data_str)
        except (IOError, json.JSONDecodeError, Exception) as e:
            log.error(f"Error loading history {HISTORY_FILE}: {e}")

    def save_sync(self):
        try:
            data_str = self._prepare_history_data()
            with open(HISTORY_FILE, "w") as f:
                f.write(data_str)
        except (IOError, TypeError, Exception) as e:
            log.error(f"Error saving history {HISTORY_FILE}: {e}")

    async def load(self):
        if not HISTORY_FILE.exists():
            return
        try:
            async with aiofiles.open(HISTORY_FILE, "r") as f:
                data_str = await f.read()
            self._load_history_data(data_str)
        except (IOError, json.JSONDecodeError, Exception) as e:
            log.error(f"Error async loading history {HISTORY_FILE}: {e}")

    async def save(self):
        try:
            data_str = self._prepare_history_data()
            async with aiofiles.open(HISTORY_FILE, "w") as f:
                await f.write(data_str)
        except (IOError, TypeError, Exception) as e:
            log.error(f"Error async saving history {HISTORY_FILE}: {e}")

    def search(self, query: str, limit: int = 5) -> List[ChatHistory]:
        results = []
        query_lower = query.lower()
        for entry in reversed(self.entries):
            if (
                query_lower in entry.query.lower()
                or query_lower in entry.response.lower()
                or any(query_lower in tool.lower() for tool in entry.tools_used)
                or any(query_lower in server.lower() for server in entry.server_names)
            ):
                results.append(entry)
                if len(results) >= limit:
                    break
        return results


# --- ServerMonitor Class ---
class ServerMonitor:
    def __init__(self, server_manager: "ServerManager"):
        self.server_manager = server_manager
        self.monitoring = False
        self.monitor_task = None
        self.health_check_interval = 30

    async def start_monitoring(self):
        if self.monitoring:
            return
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        log.info("Server health monitoring started")

    async def stop_monitoring(self):
        if not self.monitoring:
            return
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
            try:
                await self._check_server_health(name, session)
            except McpError as e:
                log.error(f"MCP error check health {name}: {e}")
            except httpx.RequestError as e:
                log.error(f"Net error check health {name}: {e}")
            except Exception as e:
                log.error(f"Unexpected error check health {name}: {e}")

    async def _check_server_health(self, server_name: str, session: ClientSession):
        if server_name not in self.server_manager.config.servers:
            return
        server_config = self.server_manager.config.servers[server_name]
        metrics = server_config.metrics
        metrics.uptime += self.health_check_interval / 60  # minutes
        start_time = time.time()
        try:
            # Use lightweight health check instead of list_tools() to avoid spamming servers
            await self._lightweight_health_check(session, server_name)
            response_time = time.time() - start_time
            metrics.update_response_time(response_time)
        except Exception as e:
            metrics.error_count += 1
            log.warning(f"Health check fail {server_name}: {e}")
        metrics.update_status()
        if metrics.status == ServerStatus.ERROR:
            await self._recover_server(server_name)

    async def _lightweight_health_check(self, session: ClientSession, server_name: str):
        """
        Perform a lightweight health check that doesn't spam the server with list_tools requests.
        For STDIO sessions, check if process is alive and session is active.
        For SSE sessions, perform minimal ping-like check.
        """
        if isinstance(session, RobustStdioSession):
            # For STDIO sessions: check process and session state
            if not session._is_active:
                raise ConnectionAbortedError(f"Session {server_name} is not active")
            if session._process and session._process.returncode is not None:
                raise ConnectionAbortedError(f"Process {server_name} has terminated (returncode: {session._process.returncode})")
            # Session appears healthy if we reach here
            log.debug(f"[{server_name}] STDIO health check passed (process alive, session active)")
        else:
            # For other session types (SSE, etc.), fall back to a minimal MCP method
            # Use initialize as a lightweight test - it should be cached/fast
            # If this proves problematic, we can make it even lighter
            try:
                # Just check if we can send a basic request - using ping-like mechanism
                # Since most MCP implementations don't have a ping, we'll just verify the session is responsive
                # by checking internal state rather than making network calls
                if hasattr(session, '_is_active') and not getattr(session, '_is_active', True):
                    raise ConnectionAbortedError(f"Session {server_name} is not active")
                log.debug(f"[{server_name}] Lightweight health check passed")
            except Exception as e:
                log.debug(f"[{server_name}] Lightweight health check failed: {e}")
                raise

    async def _recover_server(self, server_name: str):
        if server_name not in self.server_manager.config.servers:
            return
        server_config = self.server_manager.config.servers[server_name]
        log.warning(f"Attempting recover server {server_name}")
        if server_config.type == ServerType.STDIO:
            await self.server_manager.restart_server(server_name)
        elif server_config.type == ServerType.SSE:
            await self.server_manager.reconnect_server(server_name)


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
        params = {
            "processId": os.getpid(),
            "clientInfo": {"name": "mcp-client-multi", "version": "2.0.0"},
            "rootUri": None,
            "capabilities": client_capabilities,
            "protocolVersion": "2025-03-25",
        }
        result = await self._send_request("initialize", params, response_timeout=response_timeout)
        log.info(f"[{self._server_name}] Initialize request successful.")
        return result

    async def send_initialized_notification(self):
        if not self._is_active:
            log.warning(f"[{self._server_name}] Session inactive, skip initialized.")
            return
        notification = {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}
        try:
            notification_str = json.dumps(notification) + "\n"
            notification_bytes = notification_str.encode("utf-8")
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
        stream_limit = getattr(self._process.stdout, "_limit", "Unknown")
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
                    line_str_raw = line_bytes.decode("utf-8", errors="replace")
                    if USE_VERBOSE_SESSION_LOGGING:
                        log.debug(f"[{self._server_name}] READ/PROC RAW <<< {repr(line_str_raw)}")
                    line_str = line_str_raw.strip()
                    if not line_str:
                        continue
                    try:
                        message = json.loads(line_str)
                        is_valid_rpc = isinstance(message, dict) and message.get("jsonrpc") == "2.0" and ("id" in message or "method" in message)
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
                                    if err_data.get("data"):
                                        err_msg += f" Data: {repr(err_data.get('data'))[:100]}..."
                                    log.warning(f"[{self._server_name}] READ/PROC: Resolving future ID {msg_id} with ERROR: {err_msg}")
                                    future.set_exception(RuntimeError(err_msg))
                                else:
                                    log.error(f"[{self._server_name}] READ/PROC: Invalid response format ID {msg_id}.")
                                    future.set_exception(RuntimeError(f"Invalid response format ID {msg_id}"))
                            elif future:
                                log.debug(f"[{self._server_name}] READ/PROC: Future for ID {msg_id} already done.")
                            else:
                                log.warning(f"[{self._server_name}] READ/PROC: Received response for unknown/timed-out ID: {msg_id}.")
                        elif "method" in message:
                            method_name = message["method"]
                            log.debug(f"[{self._server_name}] READ/PROC: Received server message: {method_name}")
                            # Handle notifications/requests here if needed
                        else:
                            log.warning(f"[{self._server_name}] READ/PROC: Unknown message structure: {repr(message)}")
                    except json.JSONDecodeError:
                        log.debug(f"[{self._server_name}] Skipping noisy line: {line_str[:100]}...")
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
        if not self._is_active or (self._process and self._process.returncode is not None):
            raise ConnectionAbortedError("Session inactive or process terminated")
        async with self._lock:
            self._request_id_counter += 1
            request_id = str(self._request_id_counter)
        request = {"jsonrpc": "2.0", "method": method, "params": params, "id": request_id}
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._response_futures[request_id] = future
        try:
            request_str = json.dumps(request) + "\n"
            request_bytes = request_str.encode("utf-8")
            log.debug(f"[{self._server_name}] SEND: ID {request_id} ({method}): {request_bytes.decode('utf-8', errors='replace')[:100]}...")
            if self._stdin is None or self._stdin.is_closing():
                raise ConnectionAbortedError("Stdin closed")
            if USE_VERBOSE_SESSION_LOGGING:
                log.debug(f"[{self._server_name}] RAW >>> {repr(request_bytes)}")
            self._stdin.write(request_bytes)
            await self._stdin.drain()
            if USE_VERBOSE_SESSION_LOGGING:
                log.info(f"[{self._server_name}] Drain complete for ID {request_id}.")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as e:
            log.error(f"[{self._server_name}] SEND FAIL ID {request_id}: Pipe broken: {e}")
            self._response_futures.pop(request_id, None)
            if not future.done():
                future.set_exception(e)
            raise ConnectionAbortedError(f"Conn lost sending ID {request_id}: {e}") from e
        except Exception as e:
            log.error(f"[{self._server_name}] SEND FAIL ID {request_id}: {e}", exc_info=True)
            self._response_futures.pop(request_id, None)
            if not future.done():
                future.set_exception(e)
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
        if not self._is_active:
            return
        self._is_active = False
        log.debug(f"[{self._server_name}] Closing internal state due to: {exception}")
        await self._cancel_pending_futures(exception)

    async def _cancel_pending_futures(self, exception: Exception):
        log.debug(f"[{self._server_name}] Cancelling {len(self._response_futures)} pending futures with: {exception}")
        futures_to_cancel = list(self._response_futures.items())
        self._response_futures.clear()
        for _, future in futures_to_cancel:
            if future and not future.done():
                await suppress(asyncio.InvalidStateError)(future.set_exception(exception))

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
            except ProcessLookupError:
                pass
            except Exception as e:
                log.error(f"Terminate/kill error: {e}")
        log.info(f"[{self._server_name}] RobustStdioSession closed.")


class ServerManager:
    def __init__(self, config: Config, tool_cache=None, safe_printer=None):
        self.config = config
        self.exit_stack = AsyncExitStack()  # Manages lifecycles of connected sessions/processes
        self.active_sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}
        self.processes: Dict[str, asyncio.subprocess.Process] = {}
        self.tool_cache = tool_cache
        self._safe_printer = safe_printer or get_safe_console().print  # Default to safe print
        self.monitor = ServerMonitor(self)
        self.registry = ServerRegistry() if config.enable_registry else None
        self.registered_services: Dict[str, ServiceInfo] = {}  # name -> ServiceInfo
        self._session_tasks: Dict[str, List[asyncio.Task]] = {}  # Tracks background tasks per session (e.g., stderr reader)
        self.sanitized_to_original: Dict[str, str] = {}  # tool name mapping
        self._port_scan_client: Optional[httpx.AsyncClient] = None
        self.discovered_servers_cache: List[Dict[str, Any]] = []  # Cache for API/CLI
        self._discovery_in_progress = asyncio.Lock()
        log.info("ServerManager initialized.")

    async def close(self):
        """Gracefully shut down the server manager and release resources."""
        log.info("Closing ServerManager...")
        # 1. Stop Monitor
        if hasattr(self, "monitor") and self.monitor:
            await self.monitor.stop_monitoring()
        # 2. Close Registry (handles mDNS unregister and HTTP client)
        if hasattr(self, "registry") and self.registry:
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
                for ext in [".py", ".js", ".sh", ".bash", ".exe"]:  # Added shell scripts and exe
                    for item in base_path.rglob(f"*{ext}"):  # Recursive glob
                        if item.is_file():
                            # Heuristic check - adjust as needed
                            fname_lower = item.name.lower()
                            if "mcp" in fname_lower or "server" in fname_lower or "agent" in fname_lower:
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
                    name = file_path.stem  # Use filename without extension as default name
                    server_type = "stdio"  # Assume stdio for local files

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
                server_type_str = server_data.get("type", "sse").lower()  # Assume sse if missing

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
                registry_url = server_data.get("registry_url")  # Get source registry

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
            path_or_url = server_info.get("url")  # Use URL for SSE, path placeholder for others?
            server_type = server_info.get("type", "sse").lower()

            if not path_or_url:
                continue

            # Skip if already configured by path/url
            if path_or_url in configured_urls:
                log.debug(f"Skipping mDNS discovery of '{name}' at '{path_or_url}' - already configured.")
                continue

            version = server_info.get("version")  # Already parsed ServerVersion object or None
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
        tcp_check_timeout = 0.25  # Slightly longer TCP timeout
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
                    server_details = {
                        "name": server_name,
                        "path": sse_url,
                        "type": ServerType.SSE,
                        "args": [],
                        "description": f"SSE server (GET /sse) on {ip_address}:{port}",
                        "source": "portscan",
                    }
                    mcp_server_found = True  # Found it, no need for POST
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
                            server_details = {
                                "name": server_name,
                                "path": base_url,
                                "type": ServerType.SSE,
                                "args": [],
                                "description": f"SSE server (POST /) on {ip_address}:{port}",
                                "source": "portscan",
                            }
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
        concurrency = max(1, self.config.port_scan_concurrency)  # Ensure at least 1
        probe_timeout = max(0.1, self.config.port_scan_timeout)  # Ensure minimum timeout
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
        client_timeout = httpx.Timeout(probe_timeout + 0.5, connect=probe_timeout)  # Slightly larger overall timeout
        self._port_scan_client = httpx.AsyncClient(verify=False, timeout=client_timeout, limits=httpx.Limits(max_connections=concurrency + 10))

        try:
            total_ports_to_scan = (end_port - start_port + 1) * len(targets)
            log.info(f"Scanning {total_ports_to_scan} total IP:Port combinations.")

            async def bound_probe(ip: str, port: int):
                async with semaphore:
                    return await self._probe_port(ip, port, probe_timeout, self._port_scan_client)  # type: ignore

            probe_coroutines = [bound_probe(ip, port) for ip in targets for port in range(start_port, end_port + 1)]
            results = await asyncio.gather(*probe_coroutines, return_exceptions=True)

        except Exception as gather_err:
            log.error(f"Error during port scan task gathering: {gather_err}", exc_info=True)
            results = []  # Ensure results is empty on error
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
                        discovered_servers_to_add.append(
                            (
                                result.get("name", "Unknown"),
                                server_path,
                                ServerType.SSE.value,  # Type is always SSE from probe
                                None,
                                [],
                                result.get("description", ""),  # No version/categories from scan
                            )
                        )
                        log.debug(f"Adding port scan result: {result.get('name')} ({server_path})")
                elif server_path and server_path in configured_urls:
                    log.debug(f"Skipping port scan discovery of '{server_path}' - already configured.")
            elif isinstance(result, Exception):
                if not isinstance(
                    result, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout, asyncio.TimeoutError, ConnectionRefusedError, OSError)
                ):
                    log.warning(f"Port probe task failed with unexpected exception: {result}")
                elif USE_VERBOSE_SESSION_LOGGING:
                    log.debug(f"Port probe task failed with expected exception: {type(result).__name__}")

        self._discovered_port_scan = discovered_servers_to_add
        log.info(f"Port Scan Discovery: Identified {len(discovered_servers_to_add)} potential new, unconfigured servers.")

    async def discover_servers(self):
        """
        Orchestrates the discovery of MCP servers via all configured methods.
        Populates self.discovered_servers_cache.
        """
        async with self._discovery_in_progress:
            log.info("Starting server discovery process...")
            self.discovered_servers_cache = []  # Reset cache
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
                            steps=discovery_steps, step_descriptions=discovery_descriptions, title=f"{EMOJI_MAP['search']} Discovering MCP servers..."
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
                    return  # Already added this path/url in this discovery run
                processed_paths_this_run.add(path_or_url)
                is_conf = path_or_url in configured_paths
                item = {
                    "name": name,
                    "type": type_str,
                    "path_or_url": path_or_url,
                    "source": source,
                    "description": description,
                    "is_configured": is_conf,
                    "version": str(version) if version else None,
                    "categories": categories or [],
                }
                self.discovered_servers_cache.append(item)

            # Process Local
            for name, path, type_str in getattr(self, "_discovered_local", []):
                add_to_cache("filesystem", name, type_str, path, description=f"Discovered {type_str} server")
            # Process Remote
            for name, url, type_str, version, categories, rating, registry_url in getattr(self, "_discovered_remote", []):
                desc = f"From registry (Rating: {rating:.1f})" + (f" @ {registry_url}" if registry_url else "")
                add_to_cache("registry", name, type_str, url, version, categories, desc)
            # Process mDNS
            for name, url, type_str, version, categories, description in getattr(self, "_discovered_mdns", []):
                add_to_cache("mdns", name, type_str, url, version, categories, description or "Discovered on local network")
            # Process Port Scan
            for name, url, type_str, version, categories, description in getattr(self, "_discovered_port_scan", []):
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
        discovered_servers = self.discovered_servers_cache  # Use the cached results
        newly_discovered = [s for s in discovered_servers if not s["is_configured"]]

        if not newly_discovered:
            log.info("No *new* unconfigured servers found during discovery processing.")
            if discovered_servers:  # Still log if *any* were found, even if configured
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
            table.add_row(f"({i})", server["name"], server["source"], server["type"], server["path_or_url"], server["description"] or "")
        safe_console.print(table)

        # --- Add Logic ---
        added_count = 0
        servers_added_this_run = []  # Track names of successfully added servers

        if interactive_mode:
            if Confirm.ask("\nAdd selected discovered servers to configuration?", default=True, console=safe_console):
                servers_to_add_indices = set()  # Initialize empty set

                # --- Auto-select if only one server ---
                if len(newly_discovered) == 1:
                    safe_console.print("[dim]Only one server found, selecting index 1 automatically.[/dim]")
                    servers_to_add_indices = {0}  # Auto-select the first (and only) index
                # --- End Auto-select ---
                else:  # More than one server, ask for selection
                    while True:
                        choice = Prompt.ask("Enter number(s) to add (e.g., 1, 3-5), 'all', or 'none'", default="none", console=safe_console)
                        if choice.lower() == "none":
                            break  # Exit loop if user chooses none
                        servers_to_add_indices = set()  # Reset for each attempt inside the loop
                        try:
                            if choice.lower() == "all":
                                servers_to_add_indices = set(range(len(newly_discovered)))
                            else:
                                parts = choice.replace(" ", "").split(",")
                                for part in parts:
                                    if "-" in part:
                                        range_parts = part.split("-")
                                        if len(range_parts) != 2:
                                            raise ValueError("Invalid range format")
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
                            break  # Valid selection processed, exit the while True loop
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

        else:  # Non-interactive: Just log
            log.info(f"Non-interactive mode. Found {len(newly_discovered)} new servers. Add manually or via API.")

        # --- Save config and offer to connect *after* processing all selections ---
        if added_count > 0:
            await self.config.save_async()  # Save YAML changes
            safe_console.print(f"\n[green]{EMOJI_MAP['success']} Added {added_count} server(s) to configuration.[/]")

            # --- Offer to connect to the group ---
            if servers_added_this_run and Confirm.ask(
                f"Connect to the {len(servers_added_this_run)} newly added server(s) now?", default=True, console=safe_console
            ):
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
                version_obj = ServerVersion.from_string(server_info["version"])  # type: ignore

            new_config = ServerConfig(
                name=name,
                type=server_type,
                path=path_or_url,
                enabled=True,
                auto_start=False,  # Defaults for discovered
                description=server_info.get("description", f"Discovered via {server_info.get('source', 'unknown')}"),
                categories=server_info.get("categories", []),
                version=version_obj,
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
            return (
                success,
                f"Connected to existing server '{existing_server.name}'."
                if success
                else f"Failed to connect to existing server '{existing_server.name}'.",
            )

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
        await self.config.save_async()  # Save config *after* adding, before connecting
        log.info(f"Attempting to connect to newly added server '{final_added_name}'...")
        connect_success = await self.connect_to_server_by_name(final_added_name)

        return (
            connect_success,
            f"Server '{final_added_name}' added and connected." if connect_success else f"Server '{final_added_name}' added but failed to connect.",
        )

    # --- Connection Logic ---

    @asynccontextmanager
    async def _manage_process_lifetime(self, process: asyncio.subprocess.Process, server_name: str):
        """Async context manager to ensure a process is terminated."""
        try:
            yield process
        finally:
            log.debug(f"[{server_name}] Cleaning up process context...")
            await self.terminate_process(server_name, process)  # Use helper

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
                if retry_count > 0:  # Only restart on retries
                    log.warning(f"[{current_server_name}] Restarting process on retry {retry_count}")
                    await self.terminate_process(current_server_name, existing_process)
                    start_new_process = True
                else:  # First attempt, try to reuse existing RUNNING process
                    log.debug(f"[{current_server_name}] Reusing existing process PID {existing_process.pid}")
                    process_to_use = existing_process
            else:  # Process existed but already terminated
                log.warning(f"[{current_server_name}] Previous process exited code {existing_process.returncode}. Cleaning up.")
                if current_server_name in self.processes:
                    del self.processes[current_server_name]
                start_new_process = True
        else:  # No process found, start new one
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
                log_file_handle = await self.exit_stack.enter_async_context(aiofiles.open(log_file_path, "ab"))  # type: ignore
                stderr_fileno = log_file_handle.fileno()

                # Check for shell execution hint (simple check, might need refinement)
                is_shell_cmd = (
                    isinstance(executable, str) and executable in ["bash", "sh", "zsh"] and len(current_args) == 2 and current_args[0] == "-c"
                )

                if is_shell_cmd:
                    command_string = current_args[1]
                    log.debug(f"[{current_server_name}] Starting via shell: {executable} -c '{command_string[:50]}...'")
                    process = await asyncio.create_subprocess_shell(
                        command_string,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=stderr_fileno,
                        limit=2**22,
                        env=os.environ.copy(),
                        executable=executable,
                    )
                else:
                    final_cmd_list = [executable] + current_args
                    log.debug(f"[{current_server_name}] Starting via exec: {final_cmd_list}")
                    process = await asyncio.create_subprocess_exec(
                        *final_cmd_list,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=stderr_fileno,
                        limit=2**22,
                        env=os.environ.copy(),
                    )

                process_this_attempt = process
                self.processes[current_server_name] = process  # Track the newly started process
                log.info(f"[{current_server_name}] Started process PID {process.pid}")

                await asyncio.sleep(0.5)  # Allow startup time
                if process.returncode is not None:
                    raise RuntimeError(f"Process failed immediately (code {process.returncode}). Check log '{log_file_path}'.")
                process_to_use = process

            except FileNotFoundError:
                log.error(f"[{current_server_name}] Executable not found: {executable}")
                raise
            except Exception as proc_start_err:
                log.error(f"[{current_server_name}] Error starting process: {proc_start_err}", exc_info=True)
                if process and process.returncode is None:
                    await self.terminate_process(current_server_name, process)  # Cleanup failed process
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
            init_timeout = server_config.timeout + 10.0  # Slightly longer timeout for handshake
            initialize_result_obj = await asyncio.wait_for(session.initialize(response_timeout=server_config.timeout), timeout=init_timeout)
            log.info(f"[{current_server_name}] Initialize successful.")

            await session.send_initialized_notification()
            log.info(f"[{current_server_name}] Initialized notification sent.")

            # Return session, result, and the process that was actually used/started
            return session, initialize_result_obj, process_this_attempt

        except Exception as setup_err:
            log.warning(f"[{current_server_name}] Failed STDIO setup/handshake: {setup_err}", exc_info=True)
            # Ensure session is closed if created
            if session and hasattr(session, "aclose"):
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
        capability_timeout = 30.0  # Timeout for individual capability list calls

        # Determine if capabilities are advertised
        has_tools = False
        has_resources = False
        has_prompts = False

        server_caps_obj = None
        if isinstance(initialize_result, dict):  # From STDIO direct response
            caps_data = initialize_result.get("capabilities")
            if isinstance(caps_data, dict):
                # If STDIO returns a dict, we might need to map its keys
                # Assuming simple boolean flags for now based on previous code
                has_tools = bool(caps_data.get("tools"))
                has_resources = bool(caps_data.get("resources"))
                has_prompts = bool(caps_data.get("prompts"))
        elif isinstance(initialize_result, MCPInitializeResult):  # From standard ClientSession result object
            # Access the capabilities attribute directly from the MCPInitializeResult object
            server_caps_obj = initialize_result.capabilities  # This is the ServerCapabilities object

        # Check the ServerCapabilities object if obtained from MCPInitializeResult
        if server_caps_obj and hasattr(server_caps_obj, "tools") and hasattr(server_caps_obj, "resources") and hasattr(server_caps_obj, "prompts"):
            has_tools = bool(server_caps_obj.tools)
            has_resources = bool(server_caps_obj.resources)
            has_prompts = bool(server_caps_obj.prompts)
        elif server_caps_obj:
            # Log if capabilities object exists but lacks expected fields
            log.warning(
                f"[{server_name}] InitializeResult.capabilities object present but lacks expected fields (tools, resources, prompts). Type: {type(server_caps_obj)}"
            )
        else:  # Add an else to catch the case where initialize_result was neither dict nor MCPInitializeResult
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
            if has_tools and hasattr(session, "list_tools"):
                log.info(f"[{server_name}] Loading tools...")
                try:
                    res = await asyncio.wait_for(session.list_tools(), timeout=capability_timeout)
                    # Use helper to process results, handle potential None or incorrect type
                    self._process_list_result(server_name, getattr(res, "tools", None), self.tools, MCPTool, "tool")
                except asyncio.TimeoutError:
                    log.error(f"[{server_name}] Timeout loading tools.")
                except McpError as e:  # Catch specific MCP errors
                    log.error(f"[{server_name}] MCP Error loading tools: {e}")
                except Exception as e:
                    log.error(f"[{server_name}] Unexpected error loading tools: {e}", exc_info=True)
                    raise  # Re-raise other errors to potentially cancel the group
            else:
                log.info(f"[{server_name}] Skipping tools (not supported or method unavailable).")

        async def load_resources_task():
            if has_resources and hasattr(session, "list_resources"):
                log.info(f"[{server_name}] Loading resources...")
                try:
                    res = await asyncio.wait_for(session.list_resources(), timeout=capability_timeout)
                    self._process_list_result(server_name, getattr(res, "resources", None), self.resources, MCPResource, "resource")
                except asyncio.TimeoutError:
                    log.error(f"[{server_name}] Timeout loading resources.")
                except McpError as e:
                    log.error(f"[{server_name}] MCP Error loading resources: {e}")
                except Exception as e:
                    log.error(f"[{server_name}] Unexpected error loading resources: {e}", exc_info=True)
                    raise  # Re-raise other errors
            else:
                log.info(f"[{server_name}] Skipping resources (not supported or method unavailable).")

        async def load_prompts_task():
            if has_prompts and hasattr(session, "list_prompts"):
                log.info(f"[{server_name}] Loading prompts...")
                try:
                    res = await asyncio.wait_for(session.list_prompts(), timeout=capability_timeout)
                    self._process_list_result(server_name, getattr(res, "prompts", None), self.prompts, MCPPrompt, "prompt")
                except asyncio.TimeoutError:
                    log.error(f"[{server_name}] Timeout loading prompts.")
                except McpError as e:
                    log.error(f"[{server_name}] MCP Error loading prompts: {e}")
                except Exception as e:
                    log.error(f"[{server_name}] Unexpected error loading prompts: {e}", exc_info=True)
                    raise  # Re-raise other errors
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
            if isinstance(e, BaseExceptionGroup):  # Handle potential ExceptionGroup from anyio
                log.error(f"[{server_name}] Multiple errors during concurrent capability loading:")
                for i, sub_exc in enumerate(e.exceptions):
                    log.error(f"  Error {i + 1}: {type(sub_exc).__name__} - {sub_exc}", exc_info=False)  # Log basic info
                    log.debug(f"  Traceback {i + 1}:", exc_info=sub_exc)  # Log full traceback at debug
            else:
                # Log single exception
                log.error(f"[{server_name}] Error during concurrent capability loading task group: {e}", exc_info=True)
        # --- End anyio usage ---

        log.info(f"[{server_name}] Capability loading attempt finished.")

    def _process_list_result(
        self, server_name: str, result_list: Optional[List[Any]], target_dict: Dict[str, Any], item_class: type, item_type_name: str
    ):
        """Helper to populate tools/resources/prompts dictionaries from list results."""
        items_added = 0
        items_skipped = 0

        # Clear existing items for this server first
        for key in list(target_dict.keys()):
            if hasattr(target_dict[key], "server_name") and target_dict[key].server_name == server_name:
                del target_dict[key]

        if result_list is None:
            log.warning(f"[{server_name}] Received None instead of a list for {item_type_name}s.")
            return

        for item in result_list:
            try:
                if not hasattr(item, "name") or not isinstance(item.name, str) or not item.name:
                    log.warning(f"[{server_name}] Skipping {item_type_name} item lacking valid 'name': {item}")
                    items_skipped += 1
                    continue

                # Specific checks (example for Tool)
                correct_input_schema = None
                if item_class is MCPTool:
                    schema = getattr(item, "inputSchema", getattr(item, "input_schema", None))
                    if schema is None or not isinstance(schema, dict):
                        log.warning(f"[{server_name}] Skipping tool '{item.name}' lacking valid schema. Item data: {item}")
                        items_skipped += 1
                        continue
                    correct_input_schema = schema

                # Construct name and create instance
                item_name_full = f"{server_name}:{item.name}" if ":" not in item.name else item.name
                instance_data = {
                    "name": item_name_full,
                    "description": getattr(item, "description", "") or "",
                    "server_name": server_name,
                }
                # Add type-specific data
                if item_class is MCPTool:
                    instance_data["input_schema"] = correct_input_schema
                    instance_data["original_tool"] = item
                elif item_class is MCPResource:
                    instance_data["template"] = getattr(item, "uri", "")
                    instance_data["original_resource"] = item
                elif item_class is MCPPrompt:
                    instance_data["template"] = f"Prompt: {item.name}"
                    instance_data["original_prompt"] = item  # Placeholder/Adapt

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

        # process_to_use is scoped outside the loop to track the *running* process if reused
        process_to_use: Optional[asyncio.subprocess.Process] = self.processes.get(current_server_name)

        with safe_stdout():  # Protect stdout during connection attempts
            while retry_count <= max_retries:
                start_time = time.time()
                session: Optional[ClientSession] = None
                initialize_result_obj: Optional[Union[MCPInitializeResult, Dict[str, Any]]] = None
                # process_this_attempt tracks if a *new* process was started in this specific attempt
                process_this_attempt: Optional[asyncio.subprocess.Process] = None
                connection_error: Optional[BaseException] = None
                span = None
                span_context_manager = None
                attempt_exit_stack = AsyncExitStack()  # Stack for THIS attempt's resources

                try:  # Trace span setup
                    if tracer:
                        span_context_manager = tracer.start_as_current_span(
                            f"connect_server.{current_server_name}",
                            attributes={"server.name": initial_server_name, "server.type": current_server_config.type.value, "retry": retry_count},
                        )
                        if span_context_manager:
                            span = span_context_manager.__enter__()
                except Exception as e:
                    log.warning(f"Failed to start trace span: {e}")
                    span = span_context_manager = None
                finally:
                    if span_context_manager and hasattr(span_context_manager, "__exit__"):
                        with suppress(Exception):
                            span_context_manager.__exit__(*sys.exc_info())

                try:
                    # --- Main Connection Attempt ---
                    self._safe_printer(
                        f"[cyan]Connecting to {initial_server_name} (as '{current_server_name}', Attempt {retry_count + 1}/{max_retries + 1})...[/]"
                    )

                    # --- Connection & Handshake ---
                    if current_server_config.type == ServerType.STDIO:
                        # Helper handles process start/reuse and handshake
                        # _connect_stdio_server returns the active session, the initialize result, and the process *if* it started a new one
                        session, initialize_result_obj, process_this_attempt = await self._connect_stdio_server(
                            current_server_config, current_server_name, retry_count
                        )
                        # If a new process was started, manage its lifetime for this attempt
                        if process_this_attempt:
                            await attempt_exit_stack.enter_async_context(
                                self._manage_process_lifetime(process_this_attempt, f"attempt-{current_server_name}-{retry_count}")
                            )
                            process_to_use = process_this_attempt  # Track the newly started process for later use
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
                        persistent_read_timeout_secs = timedelta(days=1).total_seconds()  # Effectively infinite
                        handshake_timeout = connect_timeout + 15.0  # Allow extra time for handshake

                        log.info(f"[{current_server_name}] Creating persistent SSE context for {connect_url}...")
                        sse_ctx = sse_client(url=connect_url, headers=None, timeout=connect_timeout, sse_read_timeout=persistent_read_timeout_secs)
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
                        log.info(
                            f"[{current_server_name}] Initialize successful (SSE). Server reported: {getattr(initialize_result_obj, 'serverInfo', 'N/A')}"
                        )
                        process_this_attempt = None  # No process for SSE

                    else:
                        raise RuntimeError(f"Unknown server type: {current_server_config.type}")

                    # Check if session was successfully created
                    if not session:
                        raise RuntimeError(f"Session object is None after connection attempt for {current_server_name}")

                    # --- Rename Logic (Start - Placed *after* initialize_result_obj is obtained) ---
                    original_name_before_rename = current_server_name
                    actual_server_name_from_init: Optional[str] = None

                    # CORRECTED: Check the type of the result object before accessing
                    if isinstance(initialize_result_obj, dict):  # STDIO Result (raw dict)
                        server_info = initialize_result_obj.get("serverInfo", {})
                        if isinstance(server_info, dict):
                            actual_server_name_from_init = server_info.get("name")
                        else:
                            log.warning(f"[{current_server_name}] STDIO initialize result 'serverInfo' is not a dict: {server_info}")
                        # Capabilities handled by _load_server_capabilities using the dict

                    elif isinstance(initialize_result_obj, MCPInitializeResult):  # Standard Result (Pydantic model)
                        if initialize_result_obj.serverInfo:
                            actual_server_name_from_init = initialize_result_obj.serverInfo.name
                        # Capabilities handled by _load_server_capabilities using the Pydantic model

                    else:
                        # Handle case where initialize_result_obj is None or unexpected type
                        log.warning(
                            f"[{current_server_name}] Unexpected type or None for initialize_result_obj: {type(initialize_result_obj)}. Cannot extract server name."
                        )

                    # Perform rename if needed
                    if actual_server_name_from_init and actual_server_name_from_init != current_server_name:
                        if actual_server_name_from_init in self.config.servers and actual_server_name_from_init != initial_server_name:
                            log.warning(
                                f"Server '{initial_server_name}' reported name '{actual_server_name_from_init}', which conflicts with another existing server. Keeping original name '{current_server_name}'."
                            )
                            # Keep current_server_name, do not rename
                        else:
                            log.info(f"Server '{initial_server_name}' identified as '{actual_server_name_from_init}'. Renaming config.")
                            self._safe_printer(
                                f"[yellow]Server '{initial_server_name}' identified as '{actual_server_name_from_init}', updating config.[/]"
                            )
                            try:
                                if initial_server_name in self.config.servers:
                                    # Remove old entry, update config object, add new entry
                                    original_config_entry = self.config.servers.pop(initial_server_name)
                                    original_config_entry.name = actual_server_name_from_init  # Update the object itself
                                    self.config.servers[actual_server_name_from_init] = original_config_entry  # Add with new key

                                    # Update process mapping if STDIO
                                    if current_server_config.type == ServerType.STDIO:
                                        existing_proc_for_old_name = self.processes.pop(original_name_before_rename, None)
                                        if existing_proc_for_old_name:
                                            self.processes[actual_server_name_from_init] = existing_proc_for_old_name
                                            # Ensure process_to_use reflects the final name mapping
                                            process_to_use = existing_proc_for_old_name
                                        else:
                                            log.warning(f"Could not find process entry for old name '{original_name_before_rename}' during rename.")

                                    # Update the current loop variables
                                    current_server_name = actual_server_name_from_init
                                    current_server_config = original_config_entry  # Use the updated config object
                                    config_updated_by_rename = True
                                    if span:
                                        span.set_attribute("server.name", current_server_name)
                                    # Update session's internal name if it's the custom type
                                    if isinstance(session, RobustStdioSession):
                                        session._server_name = current_server_name
                                    log.info(f"Renamed config entry '{initial_server_name}' -> '{current_server_name}'.")
                                else:
                                    log.warning(f"Cannot rename: Original name '{initial_server_name}' no longer in config (maybe already renamed?).")
                            except Exception as rename_err:
                                log.error(f"Failed rename '{initial_server_name}' to '{actual_server_name_from_init}': {rename_err}", exc_info=True)
                                raise RuntimeError(f"Failed during server rename: {rename_err}") from rename_err
                    # --- Rename Logic (End) ---

                    # --- Load Capabilities (Uses the *final* current_server_name) ---
                    await self._load_server_capabilities(current_server_name, session, initialize_result_obj)

                    # --- Success Path ---
                    connection_time_ms = (time.time() - start_time) * 1000
                    # Use the potentially updated server name to find the config entry for metrics
                    server_config_in_dict = self.config.servers.get(current_server_name)
                    if server_config_in_dict:
                        metrics = server_config_in_dict.metrics
                        metrics.request_count += 1  # Count connection attempt
                        metrics.update_response_time(connection_time_ms / 1000.0)
                        metrics.update_status()
                    else:
                        # This should ideally not happen if rename logic is correct
                        log.error(f"Could not find server config '{current_server_name}' after connection success for metrics.")

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
                    # process_to_use holds the reference to the process that was actually used (reused or newly started)
                    if process_to_use and current_server_config.type == ServerType.STDIO:
                        # Ensure it's tracked under the potentially renamed server name
                        if current_server_name not in self.processes or self.processes[current_server_name] != process_to_use:
                            self.processes[current_server_name] = process_to_use

                    # Save config YAML if rename occurred
                    if config_updated_by_rename:
                        log.info(f"Saving configuration YAML after server rename to {current_server_name}...")
                        await self.config.save_async()

                    # Advertise via Zeroconf if it's a newly connected STDIO server
                    if current_server_config.type == ServerType.STDIO:
                        # Pass the potentially updated server config object
                        await self.register_local_server(current_server_config)

                    # Exit span context if it exists
                    if span_context_manager and hasattr(span_context_manager, "__exit__"):
                        span_context_manager.__exit__(None, None, None)

                    return session  # Return the successful session

                # --- Outer Exception Handling for Connection Attempt ---
                except (
                    McpError,
                    RuntimeError,
                    ConnectionAbortedError,
                    httpx.RequestError,
                    subprocess.SubprocessError,
                    OSError,
                    FileNotFoundError,
                    asyncio.TimeoutError,
                    BaseExceptionGroup,
                ) as e:
                    connection_error = e
                    error_type_name = type(e).__name__
                    log.warning(
                        f"[{initial_server_name}] Connection attempt {retry_count + 1} failed: {error_type_name} - {e}", exc_info=False
                    )  # Log less verbosely initially
                    log.debug(f"Traceback for {initial_server_name} connection error:", exc_info=True)  # Full traceback at debug level
                # --- End Connection Attempt Try/Except ---
                finally:
                    # Ensure attempt-specific resources are cleaned up *before* retry/failure
                    # This will close the session and terminate the process *if* it was started in this attempt
                    await attempt_exit_stack.aclose()
                    # Close the OpenTelemetry span if connection failed within the attempt
                    if span_context_manager and hasattr(span_context_manager, "__exit__"):
                        try:
                            exc_type, exc_val, exc_tb = sys.exc_info()
                            # Only exit if it wasn't already exited successfully
                            if session is None:  # Check if success path was reached
                                span_context_manager.__exit__(exc_type, exc_val, exc_tb)
                        except Exception as span_exit_err:
                            log.warning(f"Error closing OpenTelemetry span context on failure: {span_exit_err}")

                # --- Shared Error Handling & Retry Logic ---
                retry_count += 1
                # Update metrics even on failure attempt
                config_for_metrics = self.config.servers.get(current_server_name)  # Use potentially renamed server
                if config_for_metrics:
                    config_for_metrics.metrics.error_count += 1
                    config_for_metrics.metrics.update_status()
                else:
                    # Log error if config entry cannot be found (should not happen after rename logic fix)
                    log.error(f"Could not find server config '{current_server_name}' for error metric update after failed attempt.")

                if span:
                    # Set span status to error if an error occurred in this attempt
                    if connection_error:
                        span.set_status(trace.StatusCode.ERROR, f"Attempt {retry_count - 1} failed: {type(connection_error).__name__}")
                        # Optionally record the exception details
                        if hasattr(span, "record_exception"):
                            span.record_exception(connection_error)

                if retry_count <= max_retries:
                    delay = min(backoff_factor * (2 ** (retry_count - 1)) + random.random() * 0.1, 10.0)
                    error_msg_display = str(connection_error or "Unknown error")[:150] + "..."
                    self._safe_printer(
                        f"[yellow]{EMOJI_MAP['warning']} Error connecting {initial_server_name} (as '{current_server_name}'): {error_msg_display}[/]"
                    )
                    self._safe_printer(f"[cyan]Retrying connection in {delay:.2f}s...[/]")
                    await asyncio.sleep(delay)
                    # Continue main while loop
                else:  # Max retries exceeded
                    final_error_msg = str(connection_error or "Unknown connection error")
                    log.error(
                        f"Failed to connect to {initial_server_name} (as '{current_server_name}') after {max_retries + 1} attempts. Final error: {final_error_msg}"
                    )
                    self._safe_printer(
                        f"[red]{EMOJI_MAP['error']} Failed to connect: {initial_server_name} (as '{current_server_name}') after {max_retries + 1} attempts.[/]"
                    )
                    if span:
                        span.set_status(trace.StatusCode.ERROR, f"Max retries exceeded. Final: {type(connection_error).__name__}")
                    if config_updated_by_rename:  # Save config if rename happened but connection ultimately failed
                        log.info(f"Saving configuration YAML after failed connection for renamed server {current_server_name}...")
                        await self.config.save_async()
                    return None  # Connection failed

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
        start_emoji = EMOJI_MAP.get("processing", "⚙️")
        safe_console.print(f"[bold cyan]{start_emoji} {title}[/]")

        for i, (task_func, description, task_args) in enumerate(tasks):
            task_idx = i + 1
            # Print start message for the task using a gear or processing emoji
            processing_emoji = EMOJI_MAP.get("gear", "⚙️")
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
                success_emoji = EMOJI_MAP.get("success", "✅")
                safe_console.print(f"  ({task_idx}/{total_tasks}) {success_emoji} Finished: {clean_description}")

            except Exception as e:
                # Print failure message using an error emoji
                error_emoji = EMOJI_MAP.get("error", "❌")
                safe_console.print(f"  ({task_idx}/{total_tasks}) {error_emoji} Failed: {clean_description}")

                # Print the error details concisely
                # Get first line of error message for console brevity
                error_msg_str = str(e).split("\n")[0]
                safe_console.print(f"      [red]Error: {error_msg_str}[/]")

                # Log the full error with traceback for debugging
                log.error(f"Task {task_idx} ('{clean_description}') failed", exc_info=True)

                # Append None for failed tasks to indicate failure in the results list
                # Or potentially append the exception object itself if callers need it
                results.append(None)
                # Continue with the next task
                continue

        # Final completion message
        finish_emoji = EMOJI_MAP.get("party_popper", "🎉")  # Or use 'success' emoji again
        safe_console.print(f"[cyan]{finish_emoji} Completed: {title}[/]")
        return results  # Return results (may contain None for failed tasks)

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
        if hasattr(self, "_active_live_display") and self._active_live_display:
            log.warning("Attempted to create nested progress display, using simpler output")
            # Assuming _run_with_simple_progress exists and is implemented elsewhere
            return await self._run_with_simple_progress(tasks, title)

        # Set a flag that we have an active progress
        self._active_live_display = True  # Use the correct flag name

        results = []

        try:
            # Set total based on whether we're using health scores or not
            task_total = 100 if use_health_scores else 1

            # *** Ensure Progress and Columns are used ***
            columns = [
                TextColumn("[progress.description]{task.description}"),  # Uses TextColumn
                BarColumn(complete_style="green", finished_style="green"),  # Uses BarColumn
            ]
            if use_health_scores:
                # Display health score percentage
                columns.append(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))  # Uses TextColumn
            else:
                # Display task progress (e.g., 1/1)
                columns.append(TaskProgressColumn())  # Uses TaskProgressColumn
            # Always add a spinner for visual activity
            columns.append(SpinnerColumn("dots"))  # Uses SpinnerColumn

            # Ensure Progress is used with the defined columns
            with Progress(
                *columns,
                console=get_safe_console(),  # Use the safe console
                transient=transient,  # Controls if the bar disappears on completion
                expand=False,  # Prevent the progress bar from taking full width unnecessarily
            ) as progress:  # Uses Progress
                # Create all Rich Progress tasks upfront
                task_ids = []
                for _, description, _ in tasks:
                    # Add task to Rich Progress, initially not started
                    task_id = progress.add_task(description, total=task_total, start=False)
                    task_ids.append(task_id)

                # Run each provided async task sequentially
                for i, (task_func, task_desc, task_args) in enumerate(tasks):
                    current_task_id = task_ids[i]
                    progress.start_task(current_task_id)  # Mark the task as started in Rich Progress
                    progress.update(current_task_id, description=task_desc)  # Ensure description is set

                    try:
                        # Execute the actual async function passed in the tasks list
                        result = await task_func(*task_args) if task_args else await task_func()
                        results.append(result)

                        # Update the Rich Progress bar based on the result or mode
                        if use_health_scores and isinstance(result, dict) and "health" in result:
                            # Update progress bar based on the 'health' value (0-100)
                            health_score = max(0, min(100, int(result["health"])))  # Clamp value
                            progress.update(current_task_id, completed=health_score)
                            # Optionally append more info like tool count if available
                            if "tools" in result:
                                progress.update(current_task_id, description=f"{task_desc} - {result['tools']} tools")
                        else:
                            # For non-health score tasks, mark as fully complete
                            progress.update(current_task_id, completed=task_total)

                        # Mark task as finished (stops spinner)
                        progress.stop_task(current_task_id)

                    except Exception as e:
                        # If a task function raises an error
                        progress.stop_task(current_task_id)  # Stop the spinner
                        # Update description to show failure state
                        progress.update(current_task_id, description=f"{task_desc} [bold red]Failed: {str(e)}[/]", completed=0)
                        log.error(f"Task '{task_desc}' failed: {str(e)}", exc_info=True)
                        # Re-raise the exception to potentially halt the entire sequence if needed
                        # Or handle more gracefully depending on desired behavior
                        raise e

                # All tasks completed successfully if no exception was raised
                return results  # Return the list of results from each task function

        finally:
            # CRITICAL: Always clear the flag when exiting this method
            self._active_live_display = None  # Use the correct flag name

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
                (
                    self.connect_to_server,  # The function to call
                    f"{EMOJI_MAP['server']} Connecting to {name}...",  # Description for progress
                    (server_config,),  # Args tuple for the function
                )
            )

        # Use the progress helper to run connections concurrently
        if connection_tasks:  # Only run progress if there are tasks to run
            try:
                if hasattr(app, "mcp_client") and hasattr(app.mcp_client, "_run_with_progress"):
                    results = await app.mcp_client._run_with_progress(
                        connection_tasks, "Connecting Servers", transient=False, use_health_scores=False
                    )  # Don't use health score here
                    success_count = sum(1 for r in results if r is not None)
                    log.info(
                        f"Finished connecting. Successfully connected to {success_count}/{len(connection_tasks)} servers."
                    )  # Log count based on tasks run
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
            await self.terminate_process(server_name, process)  # Use helper

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
            await asyncio.sleep(0.5)  # Brief pause before reconnect

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
        test_exit_stack = AsyncExitStack()  # Local stack for test resources

        log.debug(f"[Test:{current_server_name}] Starting connection test...")
        for retry_count in range(max_retries + 1):
            session: Optional[ClientSession] = None
            process_this_attempt: Optional[asyncio.subprocess.Process] = None
            log_file_handle = None
            last_error = None
            is_connected = False

            try:
                log.debug(f"[Test:{current_server_name}] Attempt {retry_count + 1}/{max_retries + 1}")
                current_timeout = server_config.timeout + (retry_count * timeout_increment)

                if server_config.type == ServerType.STDIO:
                    log_file_path = (CONFIG_DIR / f"{current_server_name}_test_stderr.log").resolve()
                    log_file_path.parent.mkdir(parents=True, exist_ok=True)
                    log_file_handle = await test_exit_stack.enter_async_context(aiofiles.open(log_file_path, "ab"))  # type: ignore
                    stderr_fileno = log_file_handle.fileno()

                    process_this_attempt = await asyncio.create_subprocess_exec(
                        server_config.path,
                        *server_config.args,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=stderr_fileno,
                        limit=2**22,
                        env=os.environ.copy(),
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
                    session = ClientSession(
                        read_stream=read_stream, write_stream=write_stream, read_timeout_seconds=timedelta(seconds=current_timeout + 15.0)
                    )
                    await test_exit_stack.enter_async_context(session)
                    await asyncio.wait_for(session.initialize(), timeout=current_timeout + 10.0)
                else:
                    raise RuntimeError(f"Unknown server type: {server_config.type}")

                log.debug(f"[Test:{current_server_name}] Handshake successful.")
                is_connected = True
                return True  # Success

            except (
                McpError,
                RuntimeError,
                ConnectionAbortedError,
                httpx.RequestError,
                subprocess.SubprocessError,
                OSError,
                FileNotFoundError,
                asyncio.TimeoutError,
                BaseExceptionGroup,
            ) as e:
                last_error = e
                log_level = logging.DEBUG if isinstance(e, (httpx.ConnectError, asyncio.TimeoutError, ConnectionRefusedError)) else logging.WARNING
                log.log(
                    log_level,
                    f"[Test:{current_server_name}] Attempt {retry_count + 1} failed: {type(e).__name__} - {e}",
                    exc_info=(log_level >= logging.WARNING),
                )
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
                    test_exit_stack = AsyncExitStack()  # Reset for next potential retry

            # --- Retry Logic ---
            if retry_count < max_retries:
                delay = min(backoff_factor * (2**retry_count) + random.random() * 0.05, 5.0)
                log.debug(f"[Test:{current_server_name}] Retrying in {delay:.2f}s...")
                await asyncio.sleep(delay)
            else:
                log.warning(
                    f"[Test:{current_server_name}] Final connection test failed after {max_retries + 1} attempts. Last error: {type(last_error).__name__}"
                )
                return False  # Failed after all retries

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
            local_ip = "127.0.0.1"  # Default, safer for local registration
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                try:
                    s.connect(("8.8.8.8", 80))
                    local_ip = s.getsockname()[0]
                except Exception:
                    pass  # Keep 127.0.0.1 on error

            # Determine port (simple heuristic, may need improvement)
            port = 8080  # Default
            for i, arg in enumerate(server_config.args):
                if arg in ["--port", "-p"] and i + 1 < len(server_config.args):
                    try:
                        port = int(server_config.args[i + 1])
                        break
                    except (ValueError, IndexError):
                        pass

            # Prepare properties
            props = {
                b"name": server_config.name.encode("utf-8"),
                b"type": server_config.type.value.encode("utf-8"),
                b"description": server_config.description.encode("utf-8"),
                b"version": str(server_config.version or "0.0.0").encode("utf-8"),
                # 'host' prop might not be standard, rely on address in ServiceInfo
            }

            # Create ServiceInfo
            service_type = "_mcp._tcp.local."
            service_name = f"{server_config.name}.{service_type}"
            service_info = ServiceInfo(
                type_=service_type,
                name=service_name,
                addresses=[ipaddress.ip_address(local_ip).packed],
                port=port,
                properties=props,
                server=f"{socket.gethostname()}.local.",  # Advertise server hostname
            )

            log.info(f"Registering local server '{server_config.name}' with Zeroconf ({local_ip}:{port})")
            # Use async registration directly if available (check zeroconf version)
            if hasattr(self.registry.zeroconf, "async_register_service"):
                await self.registry.zeroconf.async_register_service(service_info)
            else:  # Fallback for older zeroconf or sync context needs
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
                if hasattr(self.registry.zeroconf, "async_unregister_service"):
                    await self.registry.zeroconf.async_unregister_service(service_info)
                else:  # Fallback for older zeroconf or sync context needs
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
        self.conversation_graph = ConversationGraph()  # Start fresh

        app.mcp_client = self  # Global access

        self.tool_cache = ToolCache(cache_dir=CACHE_DIR, custom_ttl_mapping=self.config.cache_ttl_mapping)
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
        self.current_provider: Optional[str] = self.get_provider_from_model(self.current_model)
        self.logger = log  # Assign the global logger to an instance variable
        if not self.current_provider:
            self.logger.warning(f"Could not determine provider for default model '{self.current_model}'. LLM calls may fail.")  # Use self.logger
        else:
            self.logger.info(f"Initial provider set to: {self.current_provider} for model {self.current_model}")  # Use self.logger

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
        
        # Token and cost tracking for cheap/fast requests
        self.cheap_request_count = 0
        self.cheap_input_tokens = 0
        self.cheap_output_tokens = 0
        self.cheap_total_cost = 0.0

        # Agent related properties
        self.agent_loop_instance: Optional[AgentMasterLoop] = None
        self.agent_task: Optional[asyncio.Task] = None
        self.agent_goal: Optional[str] = None
        self.agent_status: str = "idle"  # idle, running, completed, failed, stopped
        self.agent_last_message: Optional[str] = None
        self.agent_current_loop: int = 0
        self.agent_max_loops: int = 0

        # Pre-compile emoji pattern
        self._emoji_chars = [re.escape(str(emoji)) for emoji in EMOJI_MAP.values()]
        self._emoji_space_pattern = re.compile(f"({'|'.join(self._emoji_chars)})" + r"(\S)")
        self._current_query_text: str = ""
        self._current_status_messages: List[str] = []
        self._active_live_display: Optional[Live] = None  # Track the Live instance
        self.tool_blacklist_manager = ToolBlacklistManager()  # Initialize blacklist manager
        # Command handler dictionary (populated later)
        self.commands: Dict[str, Callable[[str], Coroutine[Any, Any, None]]] = {}  # Type hint

        # Initialize commands after all methods are defined
        self._initialize_commands()

        # Readline setup
        if platform.system() != "Windows":  # Readline might behave differently on Windows
            try:
                readline.set_completer(self.completer)
                # Use libedit bindings for macOS compatibility
                if "libedit" in readline.__doc__:
                    readline.parse_and_bind("bind ^I rl_complete")  # Binds Tab to complete
                else:
                    readline.parse_and_bind("tab: complete")  # Standard binding

                # --- Robust History Handling ---
                histfile_path = CONFIG_DIR / ".cli_history"
                histfile = str(histfile_path.resolve())  # Use resolved absolute path

                try:
                    # Ensure parent directory exists
                    histfile_path.parent.mkdir(parents=True, exist_ok=True)
                    # Try reading history
                    if histfile_path.exists():
                        # Only read if the file exists AND has content
                        if histfile_path.stat().st_size > 0:
                            self.logger.debug(f"Attempting to load readline history from non-empty file: {histfile}")  # Use self.logger
                            try:
                                readline.read_history_file(histfile)
                            except OSError as read_err:
                                # Log specifically if the read fails, but continue
                                self.logger.warning(
                                    f"Could not load readline history from '{histfile}': {read_err}", exc_info=False
                                )  # Use self.logger
                            except Exception as read_generic_err:  # Catch other potential read errors
                                self.logger.warning(
                                    f"Unexpected error loading readline history from '{histfile}': {read_generic_err}", exc_info=False
                                )  # Use self.logger
                        else:
                            self.logger.debug(f"Readline history file exists but is empty, skipping read: {histfile}")  # Use self.logger
                    else:
                        try:
                            histfile_path.touch()  # Create if not exists
                            self.logger.debug(f"Created empty readline history file: {histfile}")  # Use self.logger
                        except OSError as touch_err:
                            self.logger.warning(f"Could not create readline history file '{histfile}': {touch_err}")  # Use self.logger
                    readline.set_history_length(1000)
                    # Register saving history at exit
                    atexit.register(readline.write_history_file, histfile)
                    self.logger.debug(f"Readline history configured using: {histfile}")  # Use self.logger
                # --- Keep the outer Exception block for other setup errors ---
                except Exception as e:
                    # Catch specific potential errors if needed (e.g., PermissionError, OSError)
                    self.logger.warning(f"Could not load/save readline history from '{histfile}': {e}", exc_info=False)  # Less verbose log
            except ImportError:
                self.logger.warning("Readline library not available, CLI history and completion disabled.")  # Use self.logger
            except Exception as e:
                self.logger.warning(f"Error setting up readline: {e}")  # Use self.logger
        else:
            self.logger.info("Readline setup skipped on Windows.")  # Use self.logger

    def _initialize_commands(self):
        """Populates the command dictionary."""
        # Map command strings to their corresponding async methods
        self.commands = {
            "help": self.cmd_help,
            "exit": self.cmd_exit,
            "quit": self.cmd_exit,  # Aliases
            "config": self.cmd_config,
            "servers": self.cmd_servers,
            "tools": self.cmd_tools,
            "tool": self.cmd_tool,
            "resources": self.cmd_resources,
            "prompts": self.cmd_prompts,
            "model": self.cmd_model,
            "cache": self.cmd_cache,
            "clear": self.cmd_clear,
            "reload": self.cmd_reload,
            "fork": self.cmd_fork,
            "branch": self.cmd_branch,
            "export": self.cmd_export,
            "import": self.cmd_import,
            "history": self.cmd_history,
            "optimize": self.cmd_optimize,
            "apply-prompt": self.cmd_apply_prompt,
            "discover": self.cmd_discover,
            "dashboard": self.cmd_dashboard,
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
            if line.startswith("/") and (len(parts) == 0 or (len(parts) == 1 and not line.endswith(" "))):
                cmd_prefix = line[1:]
                prefix_to_match = cmd_prefix  # Match against the command part
                options = sorted([f"/{cmd}" for cmd in self.commands if cmd.startswith(cmd_prefix)])

            # --- Argument Completion ---
            elif len(parts) > 0 and parts[0].startswith("/"):
                cmd = parts[0][1:]
                num_args_entered = len(parts) - 1
                # If the last character is not a space, we are completing the *last* part
                # Otherwise, we are starting a *new* part.
                is_completing_last_part = not line.endswith(" ")
                arg_index_to_complete = num_args_entered - 1 if is_completing_last_part else num_args_entered
                prefix_to_match = parts[-1] if is_completing_last_part else ""

                # --- Specific Command Argument Completion ---
                if cmd == "model" and arg_index_to_complete == 0:
                    options = sorted([m for m in COST_PER_MILLION_TOKENS if m.startswith(prefix_to_match)])
                elif cmd == "servers":
                    if arg_index_to_complete == 0:  # Completing the subcommand
                        server_subcmds = ["list", "add", "remove", "connect", "disconnect", "enable", "disable", "status"]
                        options = sorted([s for s in server_subcmds if s.startswith(prefix_to_match)])
                    elif arg_index_to_complete >= 1 and parts[1] in [
                        "remove",
                        "connect",
                        "disconnect",
                        "enable",
                        "disable",
                        "status",
                    ]:  # Completing server names
                        options = sorted([name for name in self.config.servers if name.startswith(prefix_to_match)])
                        if "all".startswith(prefix_to_match) and parts[1] in ["connect", "disconnect"]:
                            options.append("all")  # Add 'all' option
                elif cmd == "config":
                    if arg_index_to_complete == 0:  # Completing the subcommand
                        config_subcmds = ["show", "edit", "reset", "api-key", "base-url", "model", "max-tokens", "history-size", "temperature"]
                        config_subcmds.extend(
                            [
                                a.replace("enable_", "").replace("use_", "").replace("_", "-")
                                for a in SIMPLE_SETTINGS_ENV_MAP.values()
                                if type(getattr(Config(), a)) is bool
                            ]
                        )
                        config_subcmds.extend(["port-scan", "discovery-path", "registry-urls", "cache-ttl"])
                        options = sorted(set(config_subcmds))
                        options = [s for s in options if s.startswith(prefix_to_match)]
                    elif arg_index_to_complete == 1 and parts[1] in ["api-key", "base-url"]:  # Completing provider name
                        options = sorted([p.value for p in Provider if p.value.startswith(prefix_to_match)])
                    # Add more specific config completions (e.g., for port-scan sub-subcommands)
                elif cmd == "tool" and arg_index_to_complete == 0:  # Completing tool name
                    options = sorted([name for name in self.server_manager.tools if name.startswith(prefix_to_match)])
                elif cmd == "prompt" and arg_index_to_complete == 0:  # Completing prompt name
                    options = sorted([name for name in self.server_manager.prompts if name.startswith(prefix_to_match)])
                elif cmd == "branch" and arg_index_to_complete == 0:  # Completing branch subcommand
                    options = sorted([s for s in ["list", "checkout", "rename", "delete"] if s.startswith(prefix_to_match)])
                elif cmd == "branch" and parts[1] == "checkout" and arg_index_to_complete == 1:  # Completing node ID for checkout
                    options = sorted(
                        [node_id[:12] for node_id in self.conversation_graph.nodes if node_id.startswith(prefix_to_match)]
                    )  # Show prefix
                elif cmd == "cache" and arg_index_to_complete == 0:  # Completing cache subcommand
                    options = sorted([s for s in ["list", "clear", "clean", "dependencies", "deps"] if s.startswith(prefix_to_match)])
                elif cmd == "cache" and parts[1] == "clear" and arg_index_to_complete == 1:  # Completing tool name for clear
                    options = sorted([name for name in self.server_manager.tools if name.startswith(prefix_to_match)])
                    if "--all".startswith(prefix_to_match):
                        options.append("--all")
                elif cmd == "import" and arg_index_to_complete == 0:  # Basic file completion
                    options = [f for f in os.listdir(".") if f.startswith(prefix_to_match) and f.endswith(".json")]

            # --- Return matching option based on state ---
            if state < len(options):
                return options[state]
            else:
                return None

        except Exception as e:
            log.error(f"Error during completion: {e}", exc_info=True)
            return None  # Fail gracefully

        return None

    # --- Configuration Commands ---
    async def cmd_config(self, args):
        """Handle configuration commands (CLI interface)."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=1)
        subcmd = parts[0].lower() if parts else "show"
        subargs = parts[1] if len(parts) > 1 else ""

        config_changed = False  # Flag to track if we need to save

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
                for name, ttl in self.config.cache_ttl_mapping.items():
                    ttl_table.add_row(name, str(ttl))
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
            return  # Edit doesn't trigger explicit save

        # --- Reset Command ---
        elif subcmd == "reset":
            if Confirm.ask(
                "[bold yellow]Reset ALL settings to defaults and save to config.yaml? This erases current config file content. Env overrides still apply on next start.[/]",
                console=safe_console,
            ):
                try:
                    if hasattr(self, "server_manager"):
                        await self.server_manager.close()
                    self.config = Config()  # Re-initializes (defaults -> empty yaml -> env overrides)
                    await self.config.save_async()  # Save the effective state to YAML
                    self.server_manager = ServerManager(self.config, self.tool_cache, self.safe_print)
                    await self._reinitialize_provider_clients()
                    safe_console.print("[green]Configuration reset to defaults and saved to config.yaml.[/]")
                except Exception as e:
                    safe_console.print(f"[red]Error during reset: {e}[/]")
            else:
                safe_console.print("[yellow]Reset cancelled.[/]")
            return  # Reset handles its own save

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
                else:
                    safe_console.print(f"[red]Invalid provider for API key: {provider_str}[/]")
            except ValueError:
                safe_console.print(f"[red]Invalid provider: {provider_str}[/]")

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
                else:
                    safe_console.print(f"[red]Base URL config not supported/invalid provider: {provider_str}[/]")
            except ValueError:
                safe_console.print(f"[red]Invalid provider: {provider_str}[/]")

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
                if new_size <= 0:
                    raise ValueError("Must be positive")
                self.config.history_size = new_size
                self.history = History(max_entries=new_size)  # Recreate history
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
            except (ValueError, TypeError):
                safe_console.print("[yellow]Usage: /config temperature <number_between_0_and_2>[/]")

        # Boolean Flags (using helper)
        elif subcmd in [
            s.replace("enable_", "").replace("use_", "").replace("_", "-")
            for s in SIMPLE_SETTINGS_ENV_MAP.keys()
            if type(getattr(Config(), SIMPLE_SETTINGS_ENV_MAP[s])) is bool
        ]:
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
                safe_console.print(f"[red]Internal error finding boolean attribute for command '{subcmd}'[/]")  # Should not happen

        # Delegated subcommands
        elif subcmd == "port-scan":
            config_changed = await self._handle_config_port_scan(subargs)
        elif subcmd == "discovery-path":
            config_changed = await self._handle_config_discovery_path(subargs)
        elif subcmd == "registry-urls":
            config_changed = await self._handle_config_registry_urls(subargs)
        elif subcmd == "cache-ttl":
            config_changed = await self._handle_config_cache_ttl(subargs)

        else:
            safe_console.print(f"[yellow]Unknown config command: {subcmd}[/]")
            # List available simple config subcommands dynamically
            simple_bool_cmds = [
                a.replace("enable_", "").replace("use_", "").replace("_", "-")
                for a in SIMPLE_SETTINGS_ENV_MAP.values()
                if type(getattr(Config(), a)) is bool
            ]
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
            return False  # Not changed

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
                else:
                    safe_console.print("[red]Invalid port range (0-65535, start <= end).[/]")
            else:
                safe_console.print("[yellow]Usage: /config port-scan range START END[/]")
        elif action == "targets":
            targets = [t.strip() for t in value.split(",") if t.strip()]
            if targets:
                if set(self.config.port_scan_targets) != set(targets):
                    self.config.port_scan_targets = targets
                    config_changed = True
                safe_console.print(f"[green]Port scan targets set to: {', '.join(targets)}[/]")
            else:
                safe_console.print("[yellow]Usage: /config port-scan targets ip1,ip2,...[/]")
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
        else:
            safe_console.print("[yellow]Unknown port-scan command. Use: enable, range, targets, concurrency, timeout, show[/]")

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

        current_paths = self.config.discovery_paths.copy()  # Work on a copy

        if action == "add" and path:
            if path not in current_paths:
                current_paths.append(path)
                config_changed = True
                safe_console.print(f"[green]Added discovery path: {path}[/]")
            else:
                safe_console.print(f"[yellow]Path already exists: {path}[/]")
        elif action == "remove" and path:
            if path in current_paths:
                current_paths.remove(path)
                config_changed = True
                safe_console.print(f"[green]Removed discovery path: {path}[/]")
            else:
                safe_console.print(f"[yellow]Path not found: {path}[/]")
        elif action == "list":
            safe_console.print("\n[bold]Discovery Paths:[/]")
            if self.config.discovery_paths:
                for i, p in enumerate(self.config.discovery_paths, 1):
                    safe_console.print(f" {i}. {p}")
            else:
                safe_console.print("  [dim]No paths configured.[/]")
        else:
            safe_console.print("[yellow]Usage: /config discovery-path [add|remove|list] [PATH][/]")

        if config_changed:
            self.config.discovery_paths = current_paths  # Update the actual config list

        return config_changed

    # --- Helper for registry URLs subcommands ---
    async def _handle_config_registry_urls(self, args) -> bool:
        """Handles /config registry-urls ... subcommands. Returns True if config changed."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=1)
        action = parts[0].lower() if parts else "list"
        url = parts[1] if len(parts) > 1 else ""
        config_changed = False

        current_urls = self.config.registry_urls.copy()  # Work on a copy

        if action == "add" and url:
            if url not in current_urls:
                current_urls.append(url)
                config_changed = True
                safe_console.print(f"[green]Added registry URL: {url}[/]")
            else:
                safe_console.print(f"[yellow]URL already exists: {url}[/]")
        elif action == "remove" and url:
            if url in current_urls:
                current_urls.remove(url)
                config_changed = True
                safe_console.print(f"[green]Removed registry URL: {url}[/]")
            else:
                safe_console.print(f"[yellow]URL not found: {url}[/]")
        elif action == "list":
            safe_console.print("\n[bold]Registry URLs:[/]")
            if self.config.registry_urls:
                for i, u in enumerate(self.config.registry_urls, 1):
                    safe_console.print(f" {i}. {u}")
            else:
                safe_console.print("  [dim]No URLs configured.[/]")
        else:
            safe_console.print("[yellow]Usage: /config registry-urls [add|remove|list] [URL][/]")

        if config_changed:
            self.config.registry_urls = current_urls  # Update the actual config list

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
                    if ttl < 0:
                        ttl = -1  # Allow negative for never expire
                    if self.config.cache_ttl_mapping.get(tool_key) != ttl:
                        self.config.cache_ttl_mapping[tool_key] = ttl
                        if self.tool_cache:
                            self.tool_cache.ttl_mapping[tool_key] = ttl
                        config_changed = True
                    safe_console.print(f"[green]Set TTL for '{tool_key}' to {ttl} seconds.[/]")
                except ValueError:
                    safe_console.print("[red]Invalid TTL value, must be an integer.[/]")
            else:
                safe_console.print("[yellow]Usage: /config cache-ttl set <tool_category_or_name> <ttl_seconds>[/]")
        elif action == "remove" and params:
            tool_key = params
            if tool_key in self.config.cache_ttl_mapping:
                del self.config.cache_ttl_mapping[tool_key]
                if self.tool_cache and tool_key in self.tool_cache.ttl_mapping:
                    del self.tool_cache.ttl_mapping[tool_key]
                config_changed = True
                safe_console.print(f"[green]Removed custom TTL for '{tool_key}'.[/]")
            else:
                safe_console.print(f"[yellow]No custom TTL found for '{tool_key}'.[/]")
        elif action == "list":
            safe_console.print("\n[bold]Custom Cache TTLs (from config.yaml):[/]")
            if self.config.cache_ttl_mapping:
                ttl_table = Table(box=box.ROUNDED)
                ttl_table.add_column("Tool Category/Name")
                ttl_table.add_column("TTL (seconds)")
                for name, ttl in self.config.cache_ttl_mapping.items():
                    ttl_table.add_row(name, str(ttl))
                safe_console.print(ttl_table)
            else:
                safe_console.print("  [dim]No custom TTLs defined.[/]")
        else:
            safe_console.print("[yellow]Usage: /config cache-ttl [set|remove|list] [PARAMS...][/]")

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
            Provider.OPENAI.value: {
                "key_attr": "openai_api_key",
                "url_attr": "openai_base_url",
                "default_url": "https://api.openai.com/v1",
                "client_attr": "openai_client",
            },
            Provider.GROK.value: {
                "key_attr": "grok_api_key",
                "url_attr": "grok_base_url",
                "default_url": "https://api.x.ai/v1",
                "client_attr": "grok_client",
            },
            Provider.DEEPSEEK.value: {
                "key_attr": "deepseek_api_key",
                "url_attr": "deepseek_base_url",
                "default_url": "https://api.deepseek.com/v1",
                "client_attr": "deepseek_client",
            },
            Provider.MISTRAL.value: {
                "key_attr": "mistral_api_key",
                "url_attr": "mistral_base_url",
                "default_url": "https://api.mistral.ai/v1",
                "client_attr": "mistral_client",
            },
            Provider.GROQ.value: {
                "key_attr": "groq_api_key",
                "url_attr": "groq_base_url",
                "default_url": "https://api.groq.com/openai/v1",
                "client_attr": "groq_client",
            },
            Provider.CEREBRAS.value: {
                "key_attr": "cerebras_api_key",
                "url_attr": "cerebras_base_url",
                "default_url": "https://api.cerebras.ai/v1",
                "client_attr": "cerebras_client",
            },
            Provider.GEMINI.value: {
                "key_attr": "gemini_api_key",
                "url_attr": "gemini_base_url",
                "default_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "client_attr": "gemini_client",
            },
            Provider.OPENROUTER.value: {
                "key_attr": "openrouter_api_key",
                "url_attr": "openrouter_base_url",
                "default_url": "https://openrouter.ai/api/v1",
                "client_attr": "openrouter_client",
            },
        }

        # Anthropic (Special Case)
        if Provider.ANTHROPIC.value in providers_to_init:
            anthropic_key = self.config.anthropic_api_key
            emoji = EMOJI_MAP.get(Provider.ANTHROPIC.value, "")
            if anthropic_key:
                try:
                    # Close existing client if it exists and has aclose
                    if self.anthropic and hasattr(self.anthropic, "aclose"):
                        await self.anthropic.aclose()
                    self.anthropic = AsyncAnthropic(api_key=anthropic_key)
                    msgs.append(f"{emoji} Anthropic: [green]OK[/]")
                except Exception as e:
                    log.error(f"Error re-initializing Anthropic: {e}")
                    self.anthropic = None
                    msgs.append(f"{emoji} Anthropic: [red]Failed[/]")
            else:
                if self.anthropic and hasattr(self.anthropic, "aclose"):
                    await self.anthropic.aclose()  # Close if key removed
                self.anthropic = None
                msgs.append(f"{emoji} Anthropic: [yellow]No Key[/]")

        # OpenAI Compatible Providers
        for provider_value, details in provider_details_map.items():
            if provider_value in providers_to_init:
                # Close existing client instance before creating a new one
                client_attr = details["client_attr"]
                existing_client = getattr(self, client_attr, None)
                if existing_client and hasattr(existing_client, "aclose"):
                    try:
                        await existing_client.aclose()
                    except Exception as close_err:
                        log.warning(f"Error closing existing {provider_value} client: {close_err}")
                # Initialize new client
                _, status_msg = await self._initialize_openai_compatible_client(
                    provider_name=provider_value,
                    api_key_attr=details["key_attr"],
                    base_url_attr=details["url_attr"],
                    default_base_url=details["default_url"],
                    client_attr=client_attr,
                    emoji_key=provider_value,
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
        header_text = Text(f"MCP Client Dashboard - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="bold white on blue", justify="center")
        layout["header"].update(Panel(header_text, border_style="blue"))  # Simpler panel for header

        # --- Footer ---
        layout["footer"].update(Text("Press Ctrl+C to exit dashboard", style="dim", justify="center"))

        # --- Servers Panel ---
        server_table = Table(title=f"{EMOJI_MAP['server']} Servers", box=box.ROUNDED, border_style="blue", show_header=True, header_style="bold blue")
        server_table.add_column("Name", style="cyan", no_wrap=True)
        server_table.add_column("Status", justify="center")
        server_table.add_column("Type", style="dim")
        server_table.add_column("Conn", justify="center")  # Abbreviated
        server_table.add_column("Avg Resp (ms)", justify="right", style="yellow")
        server_table.add_column("Errors", justify="right", style="red")
        server_table.add_column("Reqs", justify="right", style="green")  # Abbreviated

        # Sort servers by name for consistent display
        sorted_server_names = sorted(self.config.servers.keys())

        for name in sorted_server_names:
            server_config = self.config.servers[name]
            # Optionally skip disabled servers, or show them dimmed
            if not server_config.enabled:
                # Example: show disabled server dimmed
                # server_table.add_row(f"[dim]{name}[/]", "[dim]Disabled[/]", f"[dim]{server_config.type.value}[/]", "-", "-", "-", "-")
                continue  # Skip disabled for now

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
                req_count_str,
            )
        layout["servers"].update(Panel(server_table, title="[bold blue]MCP Servers[/]", border_style="blue"))

        # --- Tools Panel (Top N) ---
        tool_table = Table(
            title=f"{EMOJI_MAP['tool']} Tool Usage", box=box.ROUNDED, border_style="magenta", show_header=True, header_style="bold magenta"
        )
        tool_table.add_column("Name", style="magenta", no_wrap=True)
        tool_table.add_column("Server", style="blue")
        tool_table.add_column("Calls", justify="right", style="green")
        tool_table.add_column("Avg Time (ms)", justify="right", style="yellow")

        # Sort tools by call count descending and take top N
        sorted_tools = sorted(self.server_manager.tools.values(), key=lambda t: t.call_count, reverse=True)[:15]

        for tool in sorted_tools:
            # Show short name (part after ':')
            tool_short_name = tool.name.split(":")[-1] if ":" in tool.name else tool.name
            avg_time_ms = tool.avg_execution_time  # Assuming this is already in ms
            tool_table.add_row(tool_short_name, tool.server_name, str(tool.call_count), f"{avg_time_ms:.1f}" if tool.call_count > 0 else "-")
        layout["tools"].update(Panel(tool_table, title="[bold magenta]Tool Usage (Top 15)[/]", border_style="magenta"))

        # --- General Stats Panel ---
        stats_lines = []
        stats_lines.append(Text.assemble((f"{EMOJI_MAP['model']} Model: ", "dim"), (self.current_model, "cyan")))
        stats_lines.append(
            Text.assemble((f"{EMOJI_MAP['server']} Connected: ", "dim"), f"{len(self.server_manager.active_sessions)} / {len(self.config.servers)}")
        )
        stats_lines.append(Text.assemble((f"{EMOJI_MAP['tool']} Tools: ", "dim"), str(len(self.server_manager.tools))))
        stats_lines.append(Text.assemble((f"{EMOJI_MAP['history']} History: ", "dim"), f"{len(self.history.entries)} entries"))

        # Cache Stats
        mem_cache_count = 0
        disk_cache_count = 0
        if self.tool_cache:
            mem_cache_count = len(self.tool_cache.memory_cache)
            if self.tool_cache.disk_cache:
                try:
                    disk_cache_count = sum(1 for _ in self.tool_cache.disk_cache.iterkeys())
                except Exception:
                    pass  # Ignore error counting disk
        stats_lines.append(Text.assemble((f"{EMOJI_MAP['cache']} Tool Cache: ", "dim"), f"{mem_cache_count} (Mem) / {disk_cache_count} (Disk)"))

        # LLM Prompt Cache Stats (Overall)
        cache_hits = getattr(self, "cache_hit_count", 0)
        cache_misses = getattr(self, "cache_miss_count", 0)
        total_lookups = cache_hits + cache_misses
        if total_lookups > 0:
            hit_rate = (cache_hits / total_lookups) * 100
            stats_lines.append(
                Text.assemble(
                    (f"{EMOJI_MAP['package']} LLM Cache: ", "dim"),
                    (f"{hit_rate:.1f}% Hits", "green"),
                    (" (", "dim"),
                    (f"{self.tokens_saved_by_cache:,}", "green bold"),
                    (" Tokens Saved)", "dim"),
                )
            )
        else:
            stats_lines.append(Text.assemble((f"{EMOJI_MAP['package']} LLM Cache: ", "dim"), ("No usage yet", "dim")))

        # Conversation Branch
        current_node = self.conversation_graph.current_node
        stats_lines.append(
            Text.assemble(
                (f"{EMOJI_MAP['trident_emblem']} Branch: ", "dim"), (current_node.name, "yellow"), (f" ({current_node.id[:8]})", "dim cyan")
            )
        )

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
            names = [name.strip() for part in names_str.split(",") for name in part.split() if name.strip()]
            return list(dict.fromkeys(names))  # Remove duplicates while preserving order

        server_names_to_action = parse_server_names(subargs)

        try:
            if subcmd == "list":
                await self.list_servers()  # Use the dedicated listing method
            elif subcmd == "add":
                # Requires multiple args: NAME TYPE PATH [ARGS...]
                await self.add_server(subargs)  # Delegate parsing to add_server
            elif subcmd == "remove":
                if not server_names_to_action:
                    safe_console.print("[yellow]Usage: /servers remove <name1> [name2...] [/]")
                    return
                removed_count = 0
                for name in server_names_to_action:
                    if await self.remove_server(name):  # remove_server returns True on success
                        removed_count += 1
                if removed_count > 0:
                    safe_console.print(f"[green]Removed {removed_count} server(s).[/]")
                    await self.config.save_async()  # Save config after successful removals
            elif subcmd == "connect":
                if not server_names_to_action:
                    safe_console.print("[yellow]Usage: /servers connect <name1> [name2...] | all [/]")
                    return
                names_to_connect = server_names_to_action
                if "all" in names_to_connect:
                    names_to_connect = [
                        name for name, cfg in self.config.servers.items() if cfg.enabled and name not in self.server_manager.active_sessions
                    ]
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
                    await self.server_status(name)  # Show status for each
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
                (EMOJI_MAP["green_circle"], "green") if is_connected else (EMOJI_MAP["red_circle"], "red"), f" {'Conn' if is_connected else 'Disc'}"
            )
            enabled_text = Text.assemble(
                (EMOJI_MAP["success"], "green") if server.enabled else (EMOJI_MAP["error"], "red"),
            )
            health_display = Text.assemble((health_emoji, health_style), f" {health_status.value.capitalize()} ({health_score}%)")

            tools_count = sum(1 for t in self.server_manager.tools.values() if t.server_name == name)

            server_table.add_row(name, server.type.value, server.path, conn_status_text, health_display, enabled_text, str(tools_count))
        safe_console.print(server_table)

    async def add_server(self, args: str):
        """Adds a new server configuration based on user input."""
        safe_console = get_safe_console()
        parts = args.split(maxsplit=2)  # Split into name, type, rest
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
            name=name,
            type=server_type,
            path=path,
            args=extra_args,
            enabled=True,
            auto_start=True,  # Defaults for new adds
            description=f"Added via CLI ({datetime.now().strftime('%Y-%m-%d')})",
        )
        self.config.servers[name] = new_config
        await self.config.save_async()  # Save YAML
        safe_console.print(f"[green]{EMOJI_MAP['success']} Server '{name}' added to configuration.[/]")

        # Offer to connect
        if Confirm.ask(f"Connect to server '{name}' now?", default=True, console=safe_console):
            await self.connect_server(name)  # Call connect helper

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
            await self.server_manager.disconnect_server(name)  # Disconnect first

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
                    servers_to_connect.append(name)  # Track servers to potentially connect
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

        return changed  # Caller should save config if True

    async def server_status(self, name: str):
        """Shows detailed status for a specific server."""
        safe_console = get_safe_console()
        if not name:
            safe_console.print("[yellow]Usage: /servers status <name>[/]")
            return

        details = self.get_server_details(name)  # Use the helper method
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
            Text(f"[dim]Registry:[/dim] {details['registry_url'] or '[None]'}"),
        )
        safe_console.print(Panel(basic_info, title=f"{EMOJI_MAP['server']} Server Info: [bold blue]{name}[/]", border_style="blue", padding=(1, 2)))

        # --- Connection & Health Panel ---
        conn_status = details["metrics"]["status"]
        conn_emoji = EMOJI_MAP.get(f"status_{conn_status}", EMOJI_MAP["question_mark"])
        conn_style = f"status.{conn_status}" if conn_status != "unknown" else "dim"
        connect_info = Group(
            Text.assemble(
                ("Connected: ", "dim"), (EMOJI_MAP["green_circle"], "green") if details["is_connected"] else (EMOJI_MAP["red_circle"], "red")
            ),
            Text.assemble(("Health:", "dim"), (conn_emoji, conn_style), f" {conn_status.capitalize()}"),
            Text(f"[dim]Avg Response:[/dim] {details['metrics']['avg_response_time_ms']:.1f} ms"),
            Text(f"[dim]Requests:[/dim] {details['metrics']['request_count']}"),
            Text(f"[dim]Errors:[/dim] {details['metrics']['error_count']} ({details['metrics']['error_rate']:.1%} rate)"),
            Text(f"[dim]Uptime:[/dim] {details['metrics']['uptime_minutes']:.1f} min"),
            Text(
                f"[dim]Last Check:[/dim] {datetime.fromisoformat(details['metrics']['last_checked']).strftime('%Y-%m-%d %H:%M:%S') if details['metrics']['last_checked'] else 'N/A'}"
            ),
        )
        safe_console.print(Panel(connect_info, title=f"{EMOJI_MAP['processing']} Connection & Health", border_style="green", padding=(1, 2)))

        # --- Capabilities Panel ---
        caps = details["capabilities"]
        cap_info = Group(
            Text(f"[dim]Tools:[/dim] {'Yes' if caps.get('tools') else 'No'}"),
            Text(f"[dim]Resources:[/dim] {'Yes' if caps.get('resources') else 'No'}"),
            Text(f"[dim]Prompts:[/dim] {'Yes' if caps.get('prompts') else 'No'}"),
        )
        safe_console.print(Panel(cap_info, title=f"{EMOJI_MAP['tool']} Capabilities", border_style="magenta", padding=(1, 2)))

        # --- Process Info Panel (for STDIO) ---
        proc_info = details.get("process_info")
        if proc_info:
            if "error" in proc_info:
                proc_content = Text(f"[yellow]Error: {proc_info['error']}[/]")
            else:
                proc_content = Group(
                    Text(f"[dim]PID:[/dim] {proc_info.get('pid', 'N/A')}"),
                    Text(f"[dim]Status:[/dim] {proc_info.get('status', 'N/A')}"),
                    Text(f"[dim]CPU:[/dim] {proc_info.get('cpu_percent', 0.0):.1f}%"),
                    Text(f"[dim]Memory (RSS):[/dim] {proc_info.get('memory_rss_mb', 0.0):.1f} MB"),
                    Text(
                        f"[dim]Create Time:[/dim] {datetime.fromisoformat(proc_info.get('create_time')).strftime('%H:%M:%S') if proc_info.get('create_time') else 'N/A'}"
                    ),
                )
            safe_console.print(Panel(proc_content, title=f"{EMOJI_MAP['gear']} Process Info", border_style="yellow", padding=(1, 2)))

    async def cmd_exit(self, args: str):
        """Exit the client."""
        self.safe_print(f"{EMOJI_MAP['cancel']} Exiting...")
        # Let the main loop's finally block handle client.close()
        sys.exit(0)  # Use sys.exit to trigger cleanup via main_async finally

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
            ],
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
                resource_table.add_row(name, resource.server_name, resource.description or "[No description]", resource.template)

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
                prompt_table.add_row(name, prompt.server_name, prompt.description or "[No description]")

        if not filtered_prompts:
            safe_console.print(f"[yellow]No prompts found" + (f" for server '{server_filter}'" if server_filter else "") + ".[/yellow]")
        else:
            safe_console.print(prompt_table)

            # Offer to show template content
            if not server_filter:
                try:
                    prompt_name_prompt = await asyncio.to_thread(
                        Prompt.ask, "Enter prompt name to view content (or press Enter to skip)", console=safe_console, default=""
                    )
                    prompt_name_to_show = prompt_name_prompt.strip()
                    if prompt_name_to_show in self.server_manager.prompts:
                        template_content = self.get_prompt_template(prompt_name_to_show)
                        if template_content:
                            safe_console.print(
                                Panel(
                                    Text(template_content),
                                    title=f"{EMOJI_MAP['prompt']} Prompt Content: {prompt_name_to_show}",
                                    border_style="yellow",
                                )
                            )
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
            await self.server_manager.discover_servers()  # Step 1: Populates cache

            # Process the results (prompts user if interactive)
            # Assuming interactive mode is True for CLI command execution
            await self.server_manager._process_discovery_results(interactive_mode=True)  # Step 2: Uses cache, interacts

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
                safe_console.print('Example: /tool filesystem:readFile \'{"path": "/tmp/myfile.txt"}\'')
                return
            tool_name = parts[0]
            params_str = parts[1] if len(parts) > 1 else "{}"  # Default to empty JSON object
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

                status_emoji = EMOJI_MAP["success"] if not result.isError else EMOJI_MAP["error"]
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
                        syntax_lang = "text"  # Treat as plain text

                # Determine panel style based on error status
                panel_border = "red" if result.isError else "magenta"
                panel_title = f"{EMOJI_MAP['tool']} Tool Result: {tool_name}"
                if result.isError:
                    panel_title += " (Error)"

                safe_console.print(
                    Panel.fit(
                        Syntax(display_content_str, syntax_lang, theme="monokai", line_numbers=True, word_wrap=True),
                        title=panel_title,
                        subtitle=f"Executed in {latency:.3f}s",
                        border_style=panel_border,
                    )
                )
            except RuntimeError as e:  # Catch errors raised by execute_tool (circuit breaker, MCP errors etc.)
                status.update(f"{EMOJI_MAP['failure']} Tool execution failed")
                safe_console.print(f"[red]Error executing tool '{tool_name}': {e}[/]")
            except Exception as e:  # Catch unexpected errors
                status.update(f"{EMOJI_MAP['failure']} Unexpected error")
                log.error(f"Unexpected error during /tool execution for {tool_name}: {e}", exc_info=True)
                safe_console.print(f"[red]Unexpected error: {e}[/]")

    # --- Model Selection ---

    async def set_active_model(self, model_name: str, source: str = "CLI") -> bool:
        """
        Sets the active model AND
        , updates the default config, and saves.

        Args:
            model_name: The name of the model to set.
            source: Where the request originated from (e.g., "CLI", "API").

        Returns:
            True if the model and provider were successfully set, False otherwise.
        """
        safe_console = get_safe_console()
        new_model = model_name.strip()
        if not new_model:
            safe_console.print(f"[{source}] [yellow]Cannot set empty model name.[/]")
            return False

        # --- Determine Provider ---
        new_provider = self.get_provider_from_model(new_model)
        if not new_provider:
            safe_console.print(f"[{source}] [red]Error: Could not determine provider for model '{new_model}'. Model not set.[/]")
            return False  # Prevent setting if provider is unknown

        # Check if model/provider combo is actually changing
        if self.current_model == new_model and self.current_provider == new_provider:
            log.debug(f"Model/Provider already set to '{new_model}'/'{new_provider}', no change needed.")
            # Optionally notify user
            # self.safe_print(f"[{source}] Model is already set to: [cyan]{new_model}[/] (Provider: {new_provider})")
            return False  # Indicate no change was made

        log.info(f"Setting active model to '{new_model}' (Provider: {new_provider}) (requested via {source})")
        self.current_model = new_model
        self.current_provider = new_provider  # <-- Set the provider attribute
        self.config.default_model = new_model  # Update persistent default
        await self.config.save_async()  # Save the updated default
        self.safe_print(
            f"[{source}] {EMOJI_MAP.get(new_provider, EMOJI_MAP['model'])} Default model set to: [cyan]{new_model}[/] (Provider: {new_provider})"
        )
        return True  # Indicate a change occurred

    async def cmd_model(self, args: str):
        """Change the current AI model (CLI command)."""
        safe_console = get_safe_console()
        new_model = args.strip()
        if not new_model:
            safe_console.print(f"Current default model: [cyan]{self.current_model}[/]")
            # List available models by provider (existing logic is fine)
            models_by_provider = {}
            for model_name, provider_value in MODEL_PROVIDER_MAP.items():
                # Optionally filter by initialized providers if desired
                # if provider_value in initialized_providers: # Get initialized_providers set
                models_by_provider.setdefault(provider_value.capitalize(), []).append(model_name)

            safe_console.print("\n[bold]Available Models (based on cost data):[/]")
            for provider, models in sorted(models_by_provider.items()):
                safe_console.print(f" [blue]{provider}:[/] {', '.join(sorted(models))}")
            safe_console.print("\nUsage: /model MODEL_NAME (e.g., /model openrouter/mistralai/mistral-nemo)")  # Added example
            return

        # Call the shared method
        await self.set_active_model(new_model, source="CLI")

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
            entries_data = self.get_cache_entries()  # Use helper
            if not entries_data:
                safe_console.print(f"{EMOJI_MAP['package']} Tool cache is empty.")
                return

            cache_table = Table(title=f"{EMOJI_MAP['package']} Cached Tool Results ({len(entries_data)} entries)", box=box.ROUNDED)
            cache_table.add_column("Tool Name", style="magenta")
            cache_table.add_column("Params Hash", style="dim")
            cache_table.add_column("Created At")
            cache_table.add_column("Expires At")

            for entry_data in entries_data:  # Assumes get_cache_entries sorts if needed
                key_parts = entry_data["key"].split(":", 1)
                params_hash = key_parts[1][:12] if len(key_parts) > 1 else "N/A"
                expires_str = entry_data["expires_at"].strftime("%Y-%m-%d %H:%M:%S") if entry_data["expires_at"] else "[Never]"
                cache_table.add_row(entry_data["tool_name"], params_hash + "...", entry_data["created_at"].strftime("%Y-%m-%d %H:%M:%S"), expires_str)
            safe_console.print(cache_table)

        elif subcmd == "clear":
            tool_name_to_clear = subargs if subargs and subargs != "--all" else None
            target_desc = f"'{tool_name_to_clear}'" if tool_name_to_clear else "ALL entries"

            if Confirm.ask(f"Clear cache for {target_desc}?", console=safe_console):
                with Status(f"{EMOJI_MAP['processing']} Clearing cache for {target_desc}...", console=safe_console) as status:
                    removed_count = self.clear_cache(tool_name=tool_name_to_clear)  # Use helper
                    status.update(f"{EMOJI_MAP['success']} Cleared {removed_count} entries.")
                safe_console.print(f"[green]Cleared {removed_count} cache entries for {target_desc}.[/]")
            else:
                safe_console.print("[yellow]Cache clear cancelled.[/]")

        elif subcmd == "clean":
            with Status(f"{EMOJI_MAP['processing']} Cleaning expired cache entries...", console=safe_console) as status:
                removed_count = self.clean_cache()  # Use helper
                status.update(f"{EMOJI_MAP['success']} Finished cleaning.")
            safe_console.print(f"[green]Cleaned {removed_count} expired cache entries.[/]")

        elif subcmd == "dependencies" or subcmd == "deps":
            deps = self.get_cache_dependencies()  # Use helper
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
                await self.reload_servers()  # Use internal helper
                status.update(f"{EMOJI_MAP['success']} Server reload complete.")
                await self.print_status()  # Show updated status
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
            await self.conversation_graph.save(str(self.conversation_graph_file))  # Save after fork
            self.safe_print(f"{EMOJI_MAP['success']} Created and switched to new branch:")
            self.safe_print(f"  ID: [cyan]{new_node.id}[/]")
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
                for n in matches:
                    safe_console.print(f"  - {n.name} ({n.id})")
                return

            if matched_node:
                if self.conversation_graph.set_current_node(matched_node.id):
                    self.safe_print(f"{EMOJI_MAP['success']} Switched to branch:")
                    self.safe_print(f"  ID: [cyan]{matched_node.id}[/]")
                    self.safe_print(f"  Name: [yellow]{matched_node.name}[/]")
                else:  # Should not happen
                    safe_console.print(f"[red]Internal error switching to node {node_id_prefix}[/]")
            else:
                safe_console.print(f"[red]Node ID prefix '{node_id_prefix}' not found.[/]")

        elif subcmd == "rename":
            rename_parts = subargs.split(maxsplit=1)
            if len(rename_parts) != 2:
                safe_console.print('[yellow]Usage: /branch rename NODE_ID "New Name"[/]')
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
                        for child in n.children:
                            find_descendants(child)

                    find_descendants(node_to_delete)

                    # Remove from graph nodes dict
                    for removed_id in nodes_to_remove:
                        if removed_id in self.conversation_graph.nodes:
                            del self.conversation_graph.nodes[removed_id]
                    # Remove from parent's children list
                    parent.children.remove(node_to_delete)
                    parent.modified_at = datetime.now()

                    await self.conversation_graph.save(str(self.conversation_graph_file))
                    safe_console.print(f"[green]Deleted branch '{node_to_delete.name}' and {len(nodes_to_remove) - 1} descendant(s).[/]")
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
            return  # No file, nothing to do

        try:
            # Use safe_print for user-facing status messages
            self.safe_print(f"{EMOJI_MAP['config']}  Found Claude desktop config file, processing...")

            # Read the file content asynchronously
            async with aiofiles.open(config_path, "r") as f:
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
                    log.error(f"Failed to parse JSON from {config_path}", exc_info=True)  # Log generic parse error with traceback
                return  # Stop processing if JSON is invalid

            # --- Find the mcpServers key ---
            mcp_servers_key = "mcpServers"
            if mcp_servers_key not in desktop_config:
                found_alt = False
                # Check alternative keys just in case
                for alt_key in ["mcp_servers", "servers", "MCP_SERVERS"]:
                    if alt_key in desktop_config:
                        log.info(f"Using alternative key '{alt_key}' for MCP servers")
                        mcp_servers_key = alt_key
                        found_alt = True
                        break
                if not found_alt:
                    self.safe_print(f"{EMOJI_MAP['warning']} No MCP servers key ('mcpServers' or alternatives) found in {config_path}")
                    return  # Stop if no server list found

            mcp_servers = desktop_config.get(mcp_servers_key)  # Use .get for safety
            if not mcp_servers or not isinstance(mcp_servers, dict):
                self.safe_print(f"{EMOJI_MAP['warning']} No valid MCP server entries found under key '{mcp_servers_key}' in {config_path}")
                return  # Stop if server list is empty or not a dictionary

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
                    if "command" not in server_data:
                        log.warning(f"Skipping server '{server_name}': Missing 'command' field.")
                        skipped_servers.append((server_name, "missing command field"))
                        continue

                    original_command = server_data["command"]
                    original_args = server_data.get("args", [])

                    # Variables to store the final executable and arguments for ServerConfig
                    final_executable = None
                    final_args = []
                    is_shell_command = False  # Flag to indicate if we need `create_subprocess_shell` later

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
                                    shell_path = arg  # Keep the original path/name provided
                                    shell_arg_index = i
                                    break

                        # Check if shell was found, followed by '-c', and the command string exists
                        if shell_path is not None and shell_arg_index + 2 < len(original_args) and original_args[shell_arg_index + 1] == "-c":
                            # The actual command string to execute inside the Linux shell
                            linux_command_str = original_args[shell_arg_index + 2]
                            log.debug(f"Extracted Linux command string for shell '{shell_path}': {linux_command_str}")

                            # Find the absolute path of the shell if possible, default to /bin/<shell_name>
                            try:
                                import shutil

                                found_path = shutil.which(shell_path)
                                final_executable = found_path if found_path else f"/bin/{os.path.basename(shell_path)}"
                            except Exception:
                                final_executable = f"/bin/{os.path.basename(shell_path)}"  # Fallback

                            final_args = ["-c", linux_command_str]
                            is_shell_command = True  # Mark that this needs shell execution later

                            log.info(f"Remapped '{server_name}' to run directly via shell: {final_executable} -c '...'")

                        else:
                            # If parsing fails (e.g., no 'bash -c' found)
                            log.warning(
                                f"Could not parse expected 'shell -c command' structure in WSL args for '{server_name}': {original_args}. Skipping."
                            )
                            skipped_servers.append((server_name, "WSL command parse failed"))
                            continue  # Skip this server
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
                        is_shell_command = False  # Will use create_subprocess_exec later
                        log.info(f"Using command directly for '{server_name}' with adapted args: {final_executable} {' '.join(map(str, final_args))}")
                    # --- End Direct Command Handling ---

                    # Create the ServerConfig if we successfully determined the command
                    if final_executable is not None:
                        server_config = ServerConfig(
                            name=server_name,
                            type=ServerType.STDIO,  # Claude desktop config implies STDIO
                            path=final_executable,  # The direct executable or the shell
                            args=final_args,  # Args for the executable, or ['-c', cmd_string] for shell
                            enabled=True,  # Default to enabled
                            auto_start=True,  # Default to auto-start
                            description=f"Imported from Claude desktop config ({'Direct Shell' if is_shell_command else 'Direct Exec'})",
                            trusted=True,  # Assume trusted if coming from local desktop config
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
                    continue  # Skip this server and continue with the next

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
                        if len(server.args) == 2 and server.args[0] == "-c":
                            # Special display for shell commands
                            args_display = f'-c "{server.args[1][:60]}{"..." if len(server.args[1]) > 60 else ""}"'
                        else:
                            args_display = " ".join(map(str, server.args))

                        server_table.add_row(name, server.path, args_display)
                    self.safe_print(server_table)

                except Exception as save_error:
                    # Handle errors during saving
                    log.error("Error saving config after importing servers", exc_info=True)
                    self.safe_print(f"[red]Error saving imported server config: {save_error}[/]")
            else:
                self.safe_print(
                    f"{EMOJI_MAP['warning']}  No new servers were imported from Claude desktop config (they might already exist or failed processing)."
                )

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

        conversation_id = self.conversation_graph.current_node.id  # Default to current
        output_path = None

        i = 0
        while i < len(parsed_args):
            arg = parsed_args[i]
            if arg in ["--id", "-i"] and i + 1 < len(parsed_args):
                conversation_id = parsed_args[i + 1]
                i += 1
            elif arg in ["--output", "-o"] and i + 1 < len(parsed_args):
                output_path = parsed_args[i + 1]
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
                async with aiofiles.open(output_path, "w", encoding="utf-8") as f:
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
            if len(parsed_args) != 1:
                raise ValueError("Requires exactly one argument: filepath")
            file_path = parsed_args[0]
        except ValueError as e:
            safe_console.print(f"[red]Error parsing arguments: {e}[/]")
            safe_console.print('[yellow]Usage: /import "/path/to/your conversation.json"[/]')
            return

        if not Path(file_path).exists():
            safe_console.print(f"[red]Import file not found: {file_path}[/]")
            return

        with Status(f"{EMOJI_MAP['scroll']} Importing conversation from {file_path}...", spinner="dots", console=safe_console) as status:
            success = await self.import_conversation(file_path)  # Use helper method
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
        limit = 10  # Default number of entries to show
        search_query = None
        if args:
            if args.isdigit():
                limit = int(args)
            else:
                search_query = args  # Treat non-digit args as search query

        entries_to_display: List[ChatHistory] = []
        title = ""

        if search_query:
            limit = 50  # Increase limit for search results
            entries_to_display = self.history.search(search_query, limit=limit)
            title = f"{EMOJI_MAP['search']} History Search Results for '[yellow]{search_query}[/]' (max {limit})"
            if not entries_to_display:
                safe_console.print(f"[yellow]No history entries found matching '{search_query}'.[/]")
                return
        else:
            entries_to_display = list(self.history.entries)[-limit:]  # Get last N entries
            entries_to_display.reverse()  # Show newest first
            total_entries = len(self.history.entries)
            num_shown = min(limit, total_entries)
            title = f"{EMOJI_MAP['history']} Recent Conversation History (last {num_shown} of {total_entries})"

        safe_console.print(f"\n[bold]{title}[/]")

        for i, entry in enumerate(entries_to_display, 1):
            entry_panel_title = Text.assemble(
                (f"{i}. ", "dim"),
                (entry.timestamp, "cyan"),
                (" | Model: ", "dim"),
                (entry.model, "magenta"),
                (" | Tools: ", "dim"),
                (f"{len(entry.tools_used)}", "blue") if entry.tools_used else ("None", "dim"),
                (" | Latency: ", "dim"),
                (f"{entry.latency_ms:.0f}ms", "yellow"),
            )
            content_group = Group(
                Text.assemble(("[Q] ", "bold blue"), Text(entry.query, overflow="ellipsis")),
                Text.assemble(("[A] ", "bold green"), Text(entry.response, overflow="ellipsis")),
            )
            safe_console.print(Panel(content_group, title=entry_panel_title, border_style="dim", padding=(0, 1)))

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
                custom_model = parsed_args[i + 1]
                i += 1
            elif arg in ["--tokens", "-t"] and i + 1 < len(parsed_args):
                try:
                    target_length = int(parsed_args[i + 1])
                    i += 1
                except ValueError:
                    safe_console.print(f"[red]Invalid token count: {parsed_args[i + 1]}[/]")
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
                summary = await self.summarize_conversation(target_tokens=target_length, model=custom_model)
                # --- END CALL ---

                if summary is None:
                    status.stop()
                    safe_console.print(f"[red]{EMOJI_MAP['error']} Summarization failed (LLM call returned no content or errored).[/]")
                    return

                # Apply the summary
                summary_system_message = f"The preceding conversation up to this point has been summarized:\n\n---\n{summary}\n---"
                self.conversation_graph.current_node.messages = [InternalMessage(role="system", content=summary_system_message)]
                self.conversation_graph.current_node.modified_at = datetime.now()
                await self.conversation_graph.save(str(self.conversation_graph_file))

                final_tokens = await self.count_tokens()
                status.update(f"{EMOJI_MAP['success']} Conversation optimized: {initial_tokens:,} -> {final_tokens:,} tokens")
                safe_console.print(f"[green]Optimization complete. Tokens reduced from {initial_tokens:,} to {final_tokens:,}.[/]")

            except Exception as e:
                status.stop()
                log.error("Error during conversation optimization command", exc_info=True)
                safe_console.print(f"[red]{EMOJI_MAP['error']} Optimization failed: {e}[/]")

    async def cmd_apply_prompt(self, args: str):  # Renamed from cmd_prompt for clarity
        """Apply a prompt template to the current conversation."""
        safe_console = get_safe_console()
        prompt_name = args.strip()
        if not prompt_name:
            safe_console.print("[yellow]Usage: /prompt PROMPT_NAME[/]")
            await self.cmd_prompts("")  # List available prompts if none given
            return

        success = await self.apply_prompt_to_conversation(prompt_name)  # Use helper
        if success:
            safe_console.print(f"[green]{EMOJI_MAP['success']} Applied prompt template '{prompt_name}' as system message.[/]")
        else:
            safe_console.print(f"[red]{EMOJI_MAP['error']} Prompt '{prompt_name}' not found or could not be applied.[/]")

    async def cmd_dashboard(self, args: str):
        """Show the live monitoring dashboard."""
        safe_console = get_safe_console()
        if hasattr(self, "_active_live_display") and self._active_live_display:
            safe_console.print("[yellow]Cannot start dashboard while another live display (e.g., query) is active.[/]")
            return

        try:
            self._active_live_display = True  # Use simple flag to block other Live instances
            # Start monitoring if not already running
            if not self.server_monitor.monitoring:
                await self.server_monitor.start_monitoring()

            # Use Live context for the dashboard
            with Live(
                self.generate_dashboard_renderable(),
                refresh_per_second=1.0 / max(0.1, self.config.dashboard_refresh_rate),  # Ensure positive rate
                screen=True,
                transient=False,
                console=safe_console,
            ) as live:
                while True:
                    await asyncio.sleep(max(0.1, self.config.dashboard_refresh_rate))
                    live.update(self.generate_dashboard_renderable())
        except KeyboardInterrupt:
            self.safe_print("\n[yellow]Dashboard stopped.[/]")
        except Exception as e:
            log.error(f"Dashboard error: {e}", exc_info=True)
            self.safe_print(f"\n[red]{EMOJI_MAP['error']} Dashboard error: {e}[/]")
        finally:
            self._active_live_display = None  # Clear the flag

    # --- Utility methods and decorators ---

    @staticmethod
    def safe_print(message, **kwargs):  # No self parameter
        """Print using the appropriate console based on active stdio servers.

        Applies automatic spacing after known emojis defined in EMOJI_MAP.

        Args:
            message: The message to print (can be string or other Rich renderable)
            **kwargs: Additional arguments to pass to print
        """
        safe_console = get_safe_console()
        processed_message = message
        # Apply spacing logic ONLY if the message is a string
        if isinstance(message, str) and message:  # Also check if message is not empty
            try:
                # Extract actual emoji characters, escaping any potential regex special chars
                emoji_chars = [re.escape(str(emoji)) for emoji in EMOJI_MAP.values()]
                if emoji_chars:  # Only compile if there are emojis
                    # Create a pattern that matches any of these emojis followed by a non-whitespace char
                    # (?:...) is a non-capturing group for the alternation
                    # Capturing group 1: the emoji | Capturing group 2: the non-whitespace character
                    emoji_space_pattern = re.compile(f"({'|'.join(emoji_chars)})" + r"(\S)")
                    # Apply the substitution
                    processed_message = emoji_space_pattern.sub(r"\1 \2", message)
            except Exception as e:
                # Log error if regex fails, but proceed with original message
                log.warning(f"Failed to apply emoji spacing regex: {e}")
                processed_message = message  # Fallback to original message
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
                        delay = server_config.retry_policy["backoff_factor"] * (2**attempt) + random.random()
                        log.warning(
                            f"Retrying tool execution for server {server_name} (attempt {attempt + 1}/{server_config.retry_policy['max_attempts']})"
                        )
                        log.warning(f"Retry will happen after {delay:.2f}s delay. Error: {str(e)}")
                        await asyncio.sleep(delay)
                    else:
                        server_config.metrics.error_count += 1
                        server_config.metrics.update_status()
                        raise RuntimeError(
                            f"All {server_config.retry_policy['max_attempts']} attempts failed for server {server_name}: {str(last_error)}"
                        ) from last_error

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
                        tool.original_tool.name,  # Use the name from the original Tool object
                        tool_args,
                    )

                    # Dependency check (unchanged)
                    if self.tool_cache:
                        dependencies = self.tool_cache.dependency_graph.get(tool_name, set())
                        if dependencies:
                            log.debug(f"Tool {tool_name} has dependencies: {dependencies}")

                    return result
        finally:
            pass  # Context managers handle exit

    async def _execute_tool_and_parse_for_agent(
        self, server_name: str, tool_name_mcp: str, arguments: Dict[str, Any], request_timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        INTERNAL USE BY AGENT MASTER LOOP.
        Executes a tool using the standard `execute_tool` method (which returns CallToolResult)
        and then robustly parses its content into a standardized dictionary envelope suitable
        for the AgentMasterLoop.

        Returns:
            A dictionary:
            {
                "success": bool,
                "data": Any (parsed UMS payload if UMS tool, or original content if non-JSON success),
                "error_type": Optional[str],
                "error_message": Optional[str],
                "status_code": Optional[int], # If available from MCP layer
                "details": Optional[Any]      # Additional error details
            }
        """
        self.logger.debug(f"MCPC._exec_tool_and_parse_FOR_AGENT: Called for '{tool_name_mcp}' on '{server_name}'")

        # Initialize a default error envelope
        standard_agent_envelope: Dict[str, Any] = {
            "success": False,
            "data": None,
            "error_type": "AgentSideWrapperError_MCPC",
            "error_message": "Initial error in _execute_tool_and_parse_for_agent wrapper.",
            "status_code": None,
            "details": None,
        }

        try:
            # 1. Call the original execute_tool.
            #    The original execute_tool handles retries, circuit breaking, and returns CallToolResult.
            raw_call_tool_result_obj: CallToolResult = await self.execute_tool(
                server_name,
                tool_name_mcp,
                arguments,
                # request_timeout is not directly passed here; original execute_tool's retry logic might handle its own timeout mechanisms
                # or the underlying session.call_tool timeout is used.
            )
            self.logger.debug(
                f"MCPC._exec_tool_and_parse_FOR_AGENT: Raw CallToolResult from self.execute_tool for '{tool_name_mcp}': "
                f"isError={raw_call_tool_result_obj.isError}, Content Type={type(raw_call_tool_result_obj.content)}, "
                f"Content Preview: {str(raw_call_tool_result_obj.content)[:200]}"
            )

            # Preserve MCP status code if available (e.g., from HTTP errors caught by MCP layer)
            # The CallToolResult might not have status_code directly, but underlying errors might.
            # For now, we rely on robust_parse to extract it if embedded in error content.

            # 2. Robustly parse the content from CallToolResult.
            #    robust_parse_mcp_tool_content returns a dict:
            #    - On successful parsing of UMS JSON: {"success": True, "actual_ums_key1": ..., "actual_ums_key2": ...}
            #    - On successful parsing of non-dict JSON (e.g. string from summarize_text): {"success": True, "_raw_non_dict_data": "string_content"}
            #    - On parsing failure: {"success": False, "error": "parse error msg", "error_type": "ParseErrorType", ...}
            
            # ===== STREAMLINED RESPONSE LOGGING =====
            import json as json_lib

            from rich.console import Console
            from rich.panel import Panel
            from rich.syntax import Syntax
            
            debug_console = Console()
            
            parsed_dict_from_robust_parse = robust_parse_mcp_tool_content(raw_call_tool_result_obj.content)
            
            # Only log the parsed result (since raw and final are nearly identical)
            try:
                if isinstance(parsed_dict_from_robust_parse, dict):
                    parsed_json = json_lib.dumps(parsed_dict_from_robust_parse, indent=2, default=str)
                    syntax = Syntax(parsed_json, "json", theme="monokai", line_numbers=True)
                else:
                    syntax = f"Non-dict result: {type(parsed_dict_from_robust_parse)} = {parsed_dict_from_robust_parse}"
                
                debug_console.print(Panel(
                    syntax,
                    title="🧩 Tool Result",
                    border_style="green"
                ))
            except Exception as e:
                debug_console.print(Panel(
                    f"Failed to format result: {e}\nRaw: {parsed_dict_from_robust_parse}",
                    title="🧩 Tool Result (Raw)",
                    border_style="yellow"
                ))
            
            self.logger.debug(
                f"MCPC._exec_tool_and_parse_FOR_AGENT: Result from robust_parse for '{tool_name_mcp}': "
                f"Success Flag in Parsed: {parsed_dict_from_robust_parse.get('success', 'N/A')}, "
                f"Error Preview in Parsed: {str(parsed_dict_from_robust_parse.get('error', 'N/A'))[:100]}"
            )

            # 3. Construct the final standard_agent_envelope for the agent
            if raw_call_tool_result_obj.isError:
                # MCP transport layer reported an error (e.g., server unavailable after retries).
                standard_agent_envelope["success"] = False
                # The `parsed_dict_from_robust_parse` might be an error dict itself if content was an error message.
                if isinstance(parsed_dict_from_robust_parse, dict) and not parsed_dict_from_robust_parse.get("success", True):
                    standard_agent_envelope["error_message"] = parsed_dict_from_robust_parse.get(
                        "error", f"Tool '{tool_name_mcp}' failed at MCP layer AND content parsing."
                    )
                    standard_agent_envelope["error_type"] = parsed_dict_from_robust_parse.get("error_type", "MCPTransportAndContentError_MCPC")
                    standard_agent_envelope["details"] = parsed_dict_from_robust_parse.get("details")
                else:  # MCP error, but content parsing itself didn't yield a specific error struct.
                    error_content_str = (
                        str(raw_call_tool_result_obj.content) if raw_call_tool_result_obj.content is not None else "No content from MCP error"
                    )
                    standard_agent_envelope["error_message"] = (
                        f"Tool '{tool_name_mcp}' failed at MCP transport layer. Error content: {error_content_str[:150]}"
                    )
                    standard_agent_envelope["error_type"] = "MCPTransportError_MCPC"
                # `data` might still contain the (potentially unparsed or error) content from MCP
                standard_agent_envelope["data"] = (
                    parsed_dict_from_robust_parse
                    if isinstance(parsed_dict_from_robust_parse, dict)
                    else {"_raw_content_on_mcp_error": str(raw_call_tool_result_obj.content)}
                )

            elif isinstance(parsed_dict_from_robust_parse, dict):
                # MCP transport OK. The result of robust_parse dictates the envelope.
                if parsed_dict_from_robust_parse.get("success") and "data" not in parsed_dict_from_robust_parse:
                    # This is a successful UMS tool payload that needs to be wrapped in the data field
                    standard_agent_envelope["success"] = True
                    standard_agent_envelope["data"] = parsed_dict_from_robust_parse
                    standard_agent_envelope["error_type"] = None
                    standard_agent_envelope["error_message"] = None
                else:
                    # This is already in the standard envelope format (either error dict or already has data field)
                    standard_agent_envelope = parsed_dict_from_robust_parse.copy()

                # If UMS payload had success=True, but robust_parse had to wrap non-dict JSON,
                # ensure `data` field holds the actual data.
                if standard_agent_envelope.get("success") and "_raw_non_dict_data" in standard_agent_envelope:
                    standard_agent_envelope["data"] = standard_agent_envelope.pop("_raw_non_dict_data")

            elif parsed_dict_from_robust_parse is not None:  # Parsed to non-dict (e.g. string summary) and robust_parse didn't make it an error.
                standard_agent_envelope["success"] = True
                standard_agent_envelope["data"] = parsed_dict_from_robust_parse
                standard_agent_envelope["error_type"] = None
                standard_agent_envelope["error_message"] = None
                self.logger.debug(
                    f"MCPC._exec_tool_and_parse_FOR_AGENT: Tool '{tool_name_mcp}' returned non-dict success payload. Type: {type(parsed_dict_from_robust_parse)}"
                )

            else:  # parsed_dict_from_robust_parse is None (robust_parse failed completely and returned None - should be rare)
                standard_agent_envelope["success"] = False
                standard_agent_envelope["error_message"] = (
                    f"Tool '{tool_name_mcp}' executed by MCP, but robust_parse_mcp_tool_content returned None. This is unexpected."
                )
                standard_agent_envelope["error_type"] = "InternalParsingWrapperFailure_MCPC"
                self.logger.error(
                    f"MCPC._exec_tool_and_parse_FOR_AGENT: CRITICAL - robust_parse_mcp_tool_content returned None for tool '{tool_name_mcp}'. Raw CallToolResult content was: {str(raw_call_tool_result_obj.content)[:200]}"
                )

            # ===== LOG PROCESSING SUMMARY =====
            try:
                # Just log a concise summary since we already showed the tool result above
                debug_console.print(Panel(
                    f"[bold]Tool:[/] {tool_name_mcp}\n"
                    f"[bold]Success:[/] {standard_agent_envelope.get('success', 'N/A')}\n"
                    f"[bold]Error:[/] {standard_agent_envelope.get('error_message', 'None')}\n"
                    f"[bold]Data Type:[/] {type(standard_agent_envelope.get('data', None)).__name__}",
                    title="📊 Processing Summary",
                    border_style="cyan"
                ))
            except Exception as e:
                self.logger.warning(f"Failed to log processing summary: {e}")

            return standard_agent_envelope

        except RuntimeError as e_rte:  # From original execute_tool (e.g. circuit breaker, retries exhausted)
            self.logger.error(
                f"MCPC._exec_tool_and_parse_FOR_AGENT: Runtime error from underlying self.execute_tool for '{tool_name_mcp}': {e_rte}", exc_info=False
            )
            standard_agent_envelope.update(
                {
                    "success": False,
                    "error_message": f"Tool execution for '{tool_name_mcp}' failed after retries or due to server issue: {e_rte}",
                    "error_type": "ToolMaxRetriesOrServerError_MCPC",
                }
            )
            return standard_agent_envelope
        except Exception as e_outer_wrapper:  # Catch-all for unexpected errors *within this wrapper itself*
            self.logger.error(
                f"MCPC._exec_tool_and_parse_FOR_AGENT: Unexpected error wrapping execute_tool for '{tool_name_mcp}': {e_outer_wrapper}", exc_info=True
            )
            standard_agent_envelope.update(
                {
                    "success": False,
                    "error_message": f"Unexpected error processing tool '{tool_name_mcp}' for agent: {e_outer_wrapper}",
                    "error_type": "AgentSideToolWrapperUnexpectedError_MCPC",
                }
            )
            return standard_agent_envelope

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
            old_console = getattr(self, "_current_safe_console", None)
            self._current_safe_console = safe_console
            try:
                return await func(self, *args, **kwargs)
            finally:
                # Restore previous value if it existed
                if old_console is not None:
                    self._current_safe_console = old_console
                else:
                    delattr(self, "_current_safe_console")

    @ensure_safe_console
    async def print_status(self):
        """Print current status of servers, tools, and capabilities with progress bars"""
        # Use the stored safe console instance to prevent multiple calls
        safe_console = self._current_safe_console

        # Helper function using the pre-compiled pattern (still needed for progress bar)
        def apply_emoji_spacing(text: str) -> str:
            if isinstance(text, str) and text and hasattr(self, "_emoji_space_pattern"):
                try:  # Add try-except for robustness
                    return self._emoji_space_pattern.sub(r"\1 \2", text)
                except Exception as e:
                    log.warning(f"Failed to apply emoji spacing regex in helper: {e}")
            return text  # Return original text if not string, empty, pattern missing, or error

        # Count connected servers, available tools/resources
        connected_servers = len(self.server_manager.active_sessions)
        total_servers = len(self.config.servers)
        total_tools = len(self.server_manager.tools)
        total_resources = len(self.server_manager.resources)
        total_prompts = len(self.server_manager.prompts)

        # Print basic info table
        status_table = Table(title="MCP Client Status", box=box.ROUNDED)  # Use a box style
        status_table.add_column("Item", style="dim")  # Apply style to column
        status_table.add_column("Status", justify="right")

        # --- Use Text.assemble for the first column ---
        status_table.add_row(
            Text.assemble(str(EMOJI_MAP["model"]), "Model"),  # Note the space before "Model"
            self.current_model,
        )
        status_table.add_row(
            Text.assemble(str(EMOJI_MAP["server"]), "Servers"),  # Note the space before "Servers"
            f"{connected_servers}/{total_servers} connected",
        )
        status_table.add_row(
            Text.assemble(str(EMOJI_MAP["tool"]), "Tools"),  # Note the space before "Tools"
            str(total_tools),
        )
        status_table.add_row(
            Text.assemble(str(EMOJI_MAP["resource"]), "Resources"),  # Note the space before "Resources"
            str(total_resources),
        )
        status_table.add_row(
            Text.assemble(str(EMOJI_MAP["prompt"]), "Prompts"),  # Note the space before "Prompts"
            str(total_prompts),
        )
        # --- End Text.assemble usage ---

        safe_console.print(status_table)  # Use the regular safe_print

        if hasattr(self, "cache_hit_count") and (self.cache_hit_count + self.cache_miss_count) > 0:
            cache_table = Table(title="Prompt Cache Statistics", box=box.ROUNDED)
            cache_table.add_column("Metric", style="dim")
            cache_table.add_column("Value", justify="right")

            hit_rate = self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count) * 100

            cache_table.add_row(Text.assemble(str(EMOJI_MAP["package"]), "Cache Hits"), str(self.cache_hit_count))
            cache_table.add_row(Text.assemble(str(EMOJI_MAP["warning"]), "Cache Misses"), str(self.cache_miss_count))
            cache_table.add_row(Text.assemble(str(EMOJI_MAP["success"]), "Hit Rate"), f"{hit_rate:.1f}%")
            cache_table.add_row(Text.assemble(str(EMOJI_MAP["speech_balloon"]), "Tokens Saved"), f"{self.tokens_saved_by_cache:,}")

            safe_console.print(cache_table)

        # Only show server progress if we have servers
        if total_servers > 0:
            server_tasks = []
            for name, server in self.config.servers.items():
                if name in self.server_manager.active_sessions:
                    # Apply spacing to progress description (keep using helper here)
                    task_description = apply_emoji_spacing(f"{EMOJI_MAP['server']} {name} ({server.type.value})")
                    server_tasks.append((self._display_server_status, task_description, (name, server)))

            if server_tasks:
                await self._run_with_progress(server_tasks, "Server Status", transient=False, use_health_scores=True)

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
            log.info(
                f"Cost Calc ({model_name}): Input={input_tokens} (${input_cost:.6f}), "
                f"Output={output_tokens} (${output_cost:.6f}), Turn Total=${turn_cost:.6f}"
            )
        else:
            log.warning(f"Cost info not found for model '{model_name}'. Cannot calculate turn cost.")

        return turn_cost

    async def count_tokens(self, messages: Optional[InternalMessageList] = None) -> int:
        """
        Estimates the number of tokens in the provided messages or current conversation context
        using tiktoken (cl100k_base encoding). This is an estimation, actual provider counts may vary.
        """
        if messages is None:
            if not hasattr(self, "conversation_graph") or not self.conversation_graph:
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
            return 0  # Or potentially raise an error or return -1

        token_count = 0
        for message in messages:
            role = message.get("role")  # noqa: F841
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
                        input_str = json.dumps(block.get("input", {}))  # Stringify input
                        input_tokens_est = len(encoding.encode(input_str))
                        # Add overhead for structure, id, name, input markers
                        msg_token_count += name_tokens + input_tokens_est + 10  # Rough overhead
                    elif block_type == "tool_result":
                        # Estimate tokens for tool result representation
                        result_content = block.get("content")
                        content_str = self._stringify_content(result_content)  # Helper to handle complex content
                        result_tokens_est = len(encoding.encode(content_str))
                        # Add overhead for structure, id, content markers
                        msg_token_count += result_tokens_est + 10  # Rough overhead
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
            return len(text) // 4  # Very rough fallback

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
            "messages": messages_export,  # Should be list of dicts
            "model": node.model or self.config.default_model,  # Include model
            "exported_at": datetime.now().isoformat(),
            "path_ids": [n.id for n in all_nodes_in_path],  # Include path for context
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
                validated_messages.append(cast("InternalMessage", msg_data))  # Cast after basic check

            if not validated_messages and data["messages"]:  # If all messages were invalid
                return False, "Import failed: No valid messages found in the import data.", None

            new_node_id = str(uuid.uuid4())
            new_node = ConversationNode(
                id=new_node_id,
                name=f"Imported: {data.get('name', f'Branch-{new_node_id[:4]}')}",
                messages=validated_messages,  # Use validated messages
                model=data.get("model", self.config.default_model),
                parent=self.conversation_graph.current_node,  # Attach to current node
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
                try:
                    entry_obj = self.tool_cache.disk_cache.get(key)
                except Exception:
                    entry_obj = None  # Skip potentially corrupted

            if entry_obj and isinstance(entry_obj, CacheEntry):  # Ensure it's the correct type
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
        if not self.tool_cache:
            return 0

        # Count keys before
        keys_before_mem = set(self.tool_cache.memory_cache.keys())
        keys_before_disk = set()
        if self.tool_cache.disk_cache:
            with suppress(Exception):
                keys_before_disk = set(self.tool_cache.disk_cache.iterkeys())
        keys_before = keys_before_mem.union(keys_before_disk)

        # Perform invalidation (synchronous)
        self.tool_cache.invalidate(tool_name=tool_name)

        # Count keys after
        keys_after_mem = set(self.tool_cache.memory_cache.keys())
        keys_after_disk = set()
        if self.tool_cache.disk_cache:
            with suppress(Exception):
                keys_after_disk = set(self.tool_cache.disk_cache.iterkeys())
        keys_after = keys_after_mem.union(keys_after_disk)

        return len(keys_before) - len(keys_after)

    def clean_cache(self) -> int:
        """Cleans expired tool cache entries. Returns count removed."""
        if not self.tool_cache:
            return 0
        # clean() method handles both memory and disk (using expire())
        # We need to calculate the count manually for more accuracy
        mem_keys_before = set(self.tool_cache.memory_cache.keys())
        disk_keys_before = set()
        if self.tool_cache.disk_cache:
            with suppress(Exception):
                disk_keys_before = set(self.tool_cache.disk_cache.iterkeys())

        self.tool_cache.clean()  # This performs the actual cleaning

        mem_keys_after = set(self.tool_cache.memory_cache.keys())
        disk_keys_after = set()
        if self.tool_cache.disk_cache:
            with suppress(Exception):
                disk_keys_after = set(self.tool_cache.disk_cache.iterkeys())

        mem_removed = len(mem_keys_before) - len(mem_keys_after)
        disk_removed = len(disk_keys_before) - len(disk_keys_after)
        return mem_removed + disk_removed

    def get_cache_dependencies(self) -> Dict[str, List[str]]:
        """Gets the tool dependency graph."""
        if not self.tool_cache:
            return {}
        # Convert sets to lists for JSON serialization
        result_dict = {}
        for k, v_set in self.tool_cache.dependency_graph.items():
            result_dict[k] = sorted(v_set)  # Sort for consistency
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
        details["type"] = server_config.type  # Keep as enum for ServerDetail model
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
            "last_checked": metrics.last_checked.isoformat(),
        }
        details["process_info"] = None  # Initialize

        # Add process info for connected STDIO servers
        if is_connected and server_config.type == ServerType.STDIO and server_name in self.server_manager.processes:
            process = self.server_manager.processes[server_name]
            if process and process.returncode is None:
                try:
                    p = psutil.Process(process.pid)
                    with p.oneshot():  # Efficiently get multiple stats
                        mem_info = p.memory_info()
                        details["process_info"] = {
                            "pid": process.pid,
                            "cpu_percent": p.cpu_percent(interval=0.1),  # Interval needed
                            "memory_rss_mb": mem_info.rss / (1024 * 1024),
                            "memory_vms_mb": mem_info.vms / (1024 * 1024),
                            "status": p.status(),
                            "create_time": datetime.fromtimestamp(p.create_time()).isoformat(),
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

        prompt_content = prompt.template  # Assuming template holds the content
        if not prompt_content:
            # Add logic here to fetch prompt content if template is just an ID
            log.warning(f"Prompt '{prompt_name}' found but has empty template content.")
            return False

        # Ensure messages list exists
        if not hasattr(self.conversation_graph.current_node, "messages") or self.conversation_graph.current_node.messages is None:
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
        await default_config.save_async()  # This saves the YAML part

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
            if not server_config.enabled:
                continue
            metrics = server_config.metrics
            is_connected = name in self.server_manager.active_sessions
            health_score = 0
            if is_connected and metrics.request_count > 0:
                health_penalty = (metrics.error_rate * 100) + max(0, (metrics.avg_response_time - 1.0) * 10)
                health_score = max(0, min(100, int(100 - health_penalty)))

            server_item = {
                "name": name,
                "type": server_config.type.value,
                "status": metrics.status.value,
                "is_connected": is_connected,
                "avg_response_ms": metrics.avg_response_time * 1000,
                "error_count": metrics.error_count,
                "request_count": metrics.request_count,
                "health_score": health_score,
            }
            servers_data.append(server_item)

        # --- Tools Data (Top N by calls) ---
        tools_data = []
        sorted_tools = sorted(self.server_manager.tools.values(), key=lambda t: t.call_count, reverse=True)[:15]
        for tool in sorted_tools:
            tool_item = {
                "name": tool.name,
                "server_name": tool.server_name,
                "call_count": tool.call_count,
                "avg_execution_time_ms": tool.avg_execution_time,
            }
            tools_data.append(tool_item)

        # --- Client Info Data ---
        # Calculate cache hit rate safely
        cache_hits = getattr(self, "cache_hit_count", 0)
        cache_misses = getattr(self, "cache_miss_count", 0)
        total_lookups = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / total_lookups * 100) if total_lookups > 0 else 0.0

        client_info = {
            "current_model": self.current_model,
            "history_entries": len(self.history.entries),
            "cache_entries_memory": len(self.tool_cache.memory_cache) if self.tool_cache else 0,
            "current_branch_id": self.conversation_graph.current_node.id,
            "current_branch_name": self.conversation_graph.current_node.name,
            "cache_hit_count": cache_hits,  # Use calculated value
            "cache_miss_count": cache_misses,  # Use calculated value
            "tokens_saved_by_cache": getattr(self, "tokens_saved_by_cache", 0),
            "cache_hit_rate": cache_hit_rate,
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
                current_text = ""  # Reset
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
            async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
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
            async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
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
        self, messages: InternalMessageList, model: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None
    ) -> Optional[str]:
        """
        Executes a non-streaming LLM call for internal purposes (like summarization)
        without updating the main conversation graph or history.

        Args:
            messages: The list of messages in InternalMessage format to send.
            model: The specific model name to use (user-facing, potentially prefixed).
            max_tokens: Maximum tokens for the response.
            temperature: Temperature for the generation.

        Returns:
            The generated text content as a string, or None if the call fails.
        """
        internal_model_name_for_cost_lookup = model  # User-facing name for logging/cost
        provider_name = self.get_provider_from_model(internal_model_name_for_cost_lookup)
        if not provider_name:
            log.error(f"Cannot execute internal call: Unknown provider for model '{internal_model_name_for_cost_lookup}'")
            return None

        provider_client = getattr(self, f"{provider_name}_client", None)
        if provider_name == Provider.ANTHROPIC.value and not provider_client:
            provider_client = self.anthropic
        if not provider_client:
            log.error(f"Cannot execute internal call: Client not initialized for provider '{provider_name}'")
            return None

        max_tokens_to_use = max_tokens or self.config.default_max_tokens
        temperature_to_use = temperature if temperature is not None else self.config.temperature

        # --- Prepare ACTUAL Model Name String for the API Call ---
        actual_model_name_string = internal_model_name_for_cost_lookup  # Default to user-facing name
        providers_to_strip = [
            Provider.OPENROUTER.value,
            Provider.GROQ.value,
            Provider.CEREBRAS.value,
        ]
        if provider_name in providers_to_strip:
            prefix_to_strip = f"{provider_name}/"
            if internal_model_name_for_cost_lookup.lower().startswith(prefix_to_strip):
                actual_model_name_string = internal_model_name_for_cost_lookup[len(prefix_to_strip) :]
                log.debug(
                    f"Internal Call: Stripped prefix for API. Original: '{internal_model_name_for_cost_lookup}', Sent: '{actual_model_name_string}'"
                )
            else:
                log.warning(
                    f"Internal Call: Provider is '{provider_name}', but model '{internal_model_name_for_cost_lookup}' doesn't start with expected prefix '{prefix_to_strip}'. Sending model name as is."
                )
        # --- End Prefix Stripping ---

        log.info(f"Executing internal LLM call (no history): Model='{actual_model_name_string}', Provider='{provider_name}'")

        # Format messages for the specific provider, using the original user-facing model name
        formatted_messages, system_prompt = self._format_messages_for_provider(messages, provider_name, internal_model_name_for_cost_lookup)

        if not formatted_messages and provider_name != Provider.ANTHROPIC.value:
            log.error(f"Internal Call Error: Formatted messages are empty for model '{internal_model_name_for_cost_lookup}'. Cannot make API call.")
            return None
        elif not formatted_messages and provider_name == Provider.ANTHROPIC.value and not system_prompt:
            log.error(
                f"Internal Call Error (Anthropic): Formatted messages and system prompt are both empty for model '{internal_model_name_for_cost_lookup}'."
            )
            return None

        try:
            response_text: Optional[str] = None
            api_client: Any = provider_client

            # --- Provider-Specific Non-Streaming Call ---
            if provider_name == Provider.ANTHROPIC.value:
                anthropic_client = cast("AsyncAnthropic", api_client)
                anthropic_params = {
                    "model": actual_model_name_string,  # Use stripped param
                    "messages": formatted_messages,
                    "max_tokens": max_tokens_to_use,
                    "temperature": temperature_to_use,
                }
                if system_prompt:
                    anthropic_params["system"] = system_prompt
                if not formatted_messages:
                    log.warning("Internal Call (Anthropic): Sending API call with only a system prompt.")
                response = await anthropic_client.messages.create(**anthropic_params)
                if response.content and isinstance(response.content, list):
                    response_text = "\n".join(block.text for block in response.content if block.type == "text").strip()
                elif isinstance(response.content, str):
                    response_text = response.content

            elif provider_name in [
                Provider.OPENAI.value,
                Provider.GROK.value,
                Provider.DEEPSEEK.value,
                Provider.GROQ.value,
                Provider.MISTRAL.value,
                Provider.CEREBRAS.value,
                Provider.GEMINI.value,
                Provider.OPENROUTER.value,
            ]:
                openai_client = cast("AsyncOpenAI", api_client)
                if not formatted_messages:
                    log.error(f"Internal Call Error ({provider_name}): No messages to send after formatting.")
                    return None
                response = await openai_client.chat.completions.create(
                    model=actual_model_name_string,  # Use stripped param
                    messages=formatted_messages,  # type: ignore
                    max_tokens=max_tokens_to_use,
                    temperature=temperature_to_use,
                    stream=False,
                )
                if response.choices and response.choices[0].message:
                    response_text = response.choices[0].message.content
            else:
                log.error(f"Internal call not implemented for provider: {provider_name}")
                return None

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

        return None

    def _log_cheap_request(
        self,
        prompt_messages: List[Dict[str, str]],
        schema_name: str,
        model_used: str,
        response_data: Optional[Dict[str, Any]] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Log cheap/fast LLM request with rich formatting and comprehensive details.
        """
        from rich.columns import Columns
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        
        # Create a rich console for this specific log message
        safe_console = get_safe_console()
        
        # Get request content (truncated)
        request_text = ""
        if prompt_messages:
            for msg in prompt_messages:
                content = msg.get("content", "")
                if len(content) > 150:
                    request_text += f"[{msg.get('role', 'unknown')}] {content[:150]}..."
                else:
                    request_text += f"[{msg.get('role', 'unknown')}] {content}"
                request_text += "\n"
        request_text = request_text.strip()
        
        # Get response content (truncated)
        response_text = ""
        if response_data:
            response_str = str(response_data)
            if len(response_str) > 200:
                response_text = f"{response_str[:200]}..."
            else:
                response_text = response_str
        
        # Create the main table
        info_table = Table(show_header=False, box=None, padding=(0, 1))
        info_table.add_column("Label", style="dim cyan", width=12)
        info_table.add_column("Value", style="white")
        
        # Add rows
        info_table.add_row("🏷️ Schema", schema_name)
        info_table.add_row("🧠 Model", f"[green]{model_used}[/green]")
        
        if input_tokens is not None:
            info_table.add_row("📥 Input", f"{input_tokens:,} tokens")
        if output_tokens is not None:
            info_table.add_row("📤 Output", f"{output_tokens:,} tokens")
        if cost_usd is not None:
            info_table.add_row("💰 Cost", f"${cost_usd:.6f}")
        
        # Update totals
        self.cheap_request_count += 1
        if input_tokens:
            self.cheap_input_tokens += input_tokens
        if output_tokens:
            self.cheap_output_tokens += output_tokens
        if cost_usd:
            self.cheap_total_cost += cost_usd
            
        info_table.add_row("📊 Session", f"{self.cheap_request_count} requests, ${self.cheap_total_cost:.4f} total")
        
        # Show request content
        if request_text:
            request_panel = Panel(
                Text(request_text, style="dim white"),
                title="📤 Request",
                border_style="blue",
                padding=(0, 1)
            )
        else:
            request_panel = Panel("[dim]No request content[/dim]", title="📤 Request", border_style="dim")
        
        # Show response content or error
        if error:
            response_panel = Panel(
                Text(error, style="red"),
                title="❌ Error",
                border_style="red",
                padding=(0, 1)
            )
        elif response_text:
            response_panel = Panel(
                Text(response_text, style="green"),
                title="📥 Response",
                border_style="green",
                padding=(0, 1)
            )
        else:
            response_panel = Panel("[dim]No response content[/dim]", title="📥 Response", border_style="dim")
        
        # Create columns layout
        columns = Columns([
            Panel(info_table, title="ℹ️ Details", border_style="cyan", padding=(0, 1)),
            request_panel,
            response_panel
        ], equal=True, expand=True)
        
        # Main panel with icon
        main_panel = Panel(
            columns,
            title="⚡ Cheap & Fast LLM Request",
            border_style="cyan",
            padding=(1, 1)
        )
        
        # Print with proper spacing
        safe_console.print()
        safe_console.print(main_panel)
        safe_console.print()

    def _calculate_token_cost(self, model_name: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate the cost in USD for a given model and token usage.
        """
        model_costs = COST_PER_MILLION_TOKENS.get(model_name, {"input": 0, "output": 0})
        
        input_cost = (input_tokens / 1_000_000) * model_costs["input"]
        output_cost = (output_tokens / 1_000_000) * model_costs["output"]
        
        return input_cost + output_cost

    def _extract_token_usage(self, response: Any) -> tuple[Optional[int], Optional[int]]:
        """
        Extract token usage from various response formats.
        Returns (input_tokens, output_tokens) or (None, None) if not available.
        """
        try:
            # Handle different response structures
            if hasattr(response, 'usage'):
                # OpenAI-style response
                usage = response.usage
                if hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                    return usage.prompt_tokens, usage.completion_tokens
                elif hasattr(usage, 'input_tokens') and hasattr(usage, 'output_tokens'):
                    return usage.input_tokens, usage.output_tokens
            
            # Handle dict-style response
            if isinstance(response, dict):
                usage = response.get('usage', {})
                if 'prompt_tokens' in usage and 'completion_tokens' in usage:
                    return usage['prompt_tokens'], usage['completion_tokens']
                elif 'input_tokens' in usage and 'output_tokens' in usage:
                    return usage['input_tokens'], usage['output_tokens']
            
            # Handle Anthropic-style response
            if hasattr(response, 'input_tokens') and hasattr(response, 'output_tokens'):
                return response.input_tokens, response.output_tokens
            
        except Exception as e:
            log.debug(f"Could not extract token usage: {e}")
        
        return None, None

    def get_session_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive session statistics including both regular and cheap requests.
        """
        total_tokens = self.session_input_tokens + self.session_output_tokens
        cheap_total_tokens = self.cheap_input_tokens + self.cheap_output_tokens
        grand_total_tokens = total_tokens + cheap_total_tokens
        grand_total_cost = self.session_total_cost + self.cheap_total_cost
        
        return {
            "regular_requests": {
                "input_tokens": self.session_input_tokens,
                "output_tokens": self.session_output_tokens,
                "total_tokens": total_tokens,
                "total_cost": self.session_total_cost,
                "cache_hits": self.cache_hit_count,
                "cache_misses": self.cache_miss_count,
                "tokens_saved": self.tokens_saved_by_cache
            },
            "cheap_requests": {
                "request_count": self.cheap_request_count,
                "input_tokens": self.cheap_input_tokens,
                "output_tokens": self.cheap_output_tokens,
                "total_tokens": cheap_total_tokens,
                "total_cost": self.cheap_total_cost
            },
            "grand_totals": {
                "total_tokens": grand_total_tokens,
                "total_cost": grand_total_cost,
                "total_requests": self.cache_hit_count + self.cache_miss_count + self.cheap_request_count
            }
        }

    def print_session_stats(self) -> None:
        """
        Print comprehensive session statistics with rich formatting.
        """
        from rich.panel import Panel
        from rich.table import Table
        
        stats = self.get_session_stats()
        safe_console = get_safe_console()
        
        # Create main stats table
        stats_table = Table(show_header=True, box=box.ROUNDED)
        stats_table.add_column("Category", style="cyan", width=15)
        stats_table.add_column("Requests", justify="right", style="white")
        stats_table.add_column("Input Tokens", justify="right", style="blue")
        stats_table.add_column("Output Tokens", justify="right", style="green")
        stats_table.add_column("Total Tokens", justify="right", style="yellow")
        stats_table.add_column("Cost (USD)", justify="right", style="red")
        
        # Regular requests
        reg = stats["regular_requests"]
        stats_table.add_row(
            "🧠 Full MCP",
            f"{reg['cache_hits'] + reg['cache_misses']:,}",
            f"{reg['input_tokens']:,}",
            f"{reg['output_tokens']:,}",
            f"{reg['total_tokens']:,}",
            f"${reg['total_cost']:.4f}"
        )
        
        # Cheap requests
        cheap = stats["cheap_requests"]
        stats_table.add_row(
            "⚡ Cheap/Fast",
            f"{cheap['request_count']:,}",
            f"{cheap['input_tokens']:,}",
            f"{cheap['output_tokens']:,}",
            f"{cheap['total_tokens']:,}",
            f"${cheap['total_cost']:.4f}"
        )
        
        # Totals
        total = stats["grand_totals"]
        stats_table.add_row(
            "📊 TOTAL",
            f"{total['total_requests']:,}",
            f"{reg['input_tokens'] + cheap['input_tokens']:,}",
            f"{reg['output_tokens'] + cheap['output_tokens']:,}",
            f"{total['total_tokens']:,}",
            f"${total['total_cost']:.4f}",
            style="bold"
        )
        
        # Cache stats
        cache_table = Table(show_header=False, box=None)
        cache_table.add_column("Metric", style="dim")
        cache_table.add_column("Value", style="white")
        cache_table.add_row("Cache Hits", f"{reg['cache_hits']:,}")
        cache_table.add_row("Cache Misses", f"{reg['cache_misses']:,}")
        cache_table.add_row("Tokens Saved", f"{reg['tokens_saved']:,}")
        
        # Print results
        safe_console.print(Panel(stats_table, title="📈 Session Statistics", border_style="green"))
        safe_console.print(Panel(cache_table, title="💾 Cache Statistics", border_style="blue"))

    async def query_llm(
        self,
        prompt_messages: List[Dict[str, str]],
        model_override: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False,
        use_cheap_model: bool = True,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Optional[Any]:
        """
        Execute a lightweight LLM query without MCP tools, optimized for simple classification/scoring tasks.
        
        This method is designed for simple LLM requests that don't need the full MCP tool payload,
        such as classification, scoring, semantic analysis, etc. It uses a cheaper, faster model
        by default to optimize costs for these simple operations.
        
        Args:
            prompt_messages: List of message dictionaries with 'role' and 'content' keys
            model_override: Specific model to use (overrides cheap model selection)
            max_tokens: Maximum tokens for response (defaults to 500 for efficiency)
            temperature: Temperature for generation (defaults to 0.1 for classification tasks)
            stream: Whether to stream the response (currently not supported, always False)
            use_cheap_model: Whether to use the cheap model (True by default)
            response_format: Optional structured output format (e.g., JSON schema)
            
        Returns:
            Response object with .content attribute containing the generated text
        """
        # Determine which model to use
        target_model = None
        if model_override:
            target_model = model_override
        elif use_cheap_model:
            # Get cheap model from config (which handles environment variables via decouple)
            cheap_model = getattr(self.config, 'default_cheap_and_fast_model', None)
            if cheap_model:
                target_model = cheap_model
                log.debug(f"query_llm: Using cheap model from config: {cheap_model}")
            else:
                target_model = self.current_model
                log.debug(f"query_llm: No cheap model configured, using current model: {target_model}")
        else:
            target_model = self.current_model
            
        # Set defaults optimized for simple queries
        max_tokens_to_use = max_tokens or 500  # Cap at 500 for cheap requests
        temperature_to_use = temperature if temperature is not None else 0.1  # Lower temp for classification
        
        # Convert to internal message format
        internal_messages = []
        for msg in prompt_messages:
            internal_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })

        log.debug(f"query_llm: Model='{target_model}', Use Cheap={use_cheap_model}, Max Tokens={max_tokens_to_use}")

        try:
            # Use process_streaming_query with tools disabled
            accumulated_text = ""
            had_error = False
            error_message = ""
            
            # Collect the streamed response
            async for chunk_type, chunk_data in self.process_streaming_query(
                query="",  # Empty since we're using messages_override
                model=target_model,
                max_tokens=max_tokens_to_use,
                temperature=temperature_to_use,
                messages_override=internal_messages,
                tools_override=[],  # Disable all tools for lightweight queries
                force_structured_output=bool(response_format),
                ums_tool_schemas=None  # No UMS tools needed
            ):
                if chunk_type == "text_chunk":
                    accumulated_text += str(chunk_data)
                elif chunk_type == "error":
                    had_error = True
                    error_message = str(chunk_data)
                    log.error(f"query_llm error: {error_message}")
                    break
                # Ignore other chunk types (status, tool_call_*, final_usage, etc.)
            
            if had_error:
                log.error(f"query_llm failed: {error_message}")
                return None

            # Create a simple response object that mimics the expected interface
            class SimpleResponse:
                def __init__(self, content: str):
                    self.content = content

            if accumulated_text:
                log.debug(f"query_llm completed successfully. Response length: {len(accumulated_text)} chars")
                return SimpleResponse(accumulated_text.strip())
            else:
                log.warning("query_llm: No content received from LLM")
                return None
                
        except Exception as e:
            log.error(f"query_llm unexpected error: {e}", exc_info=True)
            return None

    async def query_llm_structured(
        self,
        prompt_messages: List[Dict[str, str]],
        response_schema: Dict[str, Any],
        schema_name: str = "structured_response",
        model_override: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        use_cheap_model: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a structured LLM query that returns JSON according to a schema.
        
        This method uses proper OpenAI-compatible structured output formatting
        for reliable JSON schema adherence.
        
        Args:
            prompt_messages: List of message dictionaries
            response_schema: JSON schema for the expected response structure
            schema_name: Name for the schema (used by OpenAI structured output)
            model_override: Specific model to use
            max_tokens: Maximum tokens for response  
            temperature: Temperature for generation
            use_cheap_model: Whether to use the cheap model
            
        Returns:
            Parsed JSON response as a dictionary, or None if failed
        """
        # Determine model for structured output compatibility
        target_model = None
        if model_override:
            target_model = model_override
        elif use_cheap_model:
            cheap_model = getattr(self.config, 'default_cheap_and_fast_model', None)
            target_model = cheap_model if cheap_model else self.current_model
        else:
            target_model = self.current_model
            
        provider_name = self.get_provider_from_model(target_model)
        
        # Prepare structured output format based on provider and model capabilities
        structured_format = None
        
        # Check if model supports json_schema format (OpenAI structured outputs)
        actual_model_name = target_model
        providers_to_strip = [Provider.OPENROUTER.value, Provider.GROQ.value, Provider.CEREBRAS.value]
        if provider_name in providers_to_strip:
            prefix_to_strip = f"{provider_name}/"
            if target_model.lower().startswith(prefix_to_strip):
                actual_model_name = target_model[len(prefix_to_strip):]
        
        if actual_model_name in MODELS_CONFIRMED_FOR_OPENAI_JSON_SCHEMA_FORMAT:
            # Use full JSON Schema with strict mode
            structured_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "description": f"Structured response for {schema_name}",
                    "strict": True,
                    "schema": response_schema
                }
            }
        elif actual_model_name in MODELS_SUPPORTING_OPENAI_JSON_OBJECT_FORMAT or provider_name in [
            Provider.DEEPSEEK.value, Provider.GEMINI.value, Provider.GROK.value, 
            Provider.GROQ.value, Provider.CEREBRAS.value, Provider.OPENROUTER.value
        ]:
            # Fallback to json_object mode
            structured_format = {"type": "json_object"}
        else:
            # No structured output support, will rely on prompt instructions
            structured_format = None
            
        log.debug(f"query_llm_structured: Using format {structured_format} for model {target_model}")
        
        try:
            # Use the basic query_llm with structured format
            response = await self.query_llm(
                prompt_messages=prompt_messages,
                model_override=target_model,
                max_tokens=max_tokens,
                temperature=temperature,
                use_cheap_model=False,  # We already determined the model above
                response_format=structured_format
            )
            
            # Extract token usage and calculate cost
            input_tokens, output_tokens = self._extract_token_usage(response)
            cost_usd = None
            if input_tokens and output_tokens:
                cost_usd = self._calculate_token_cost(target_model, input_tokens, output_tokens)
            
            if response and hasattr(response, 'content') and response.content:
                # Try to parse as JSON
                import json
                try:
                    parsed_json = json.loads(response.content.strip())
                    
                    # Log successful cheap request
                    self._log_cheap_request(
                        prompt_messages=prompt_messages,
                        schema_name=schema_name,
                        model_used=target_model,
                        response_data=parsed_json,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost_usd
                    )
                    
                    return parsed_json
                except json.JSONDecodeError as e:
                    error_msg = f"Failed to parse JSON response: {e}"
                    log.warning(f"query_llm_structured: {error_msg}")
                    log.debug(f"Raw response: {response.content}")
                    
                    # Try to extract JSON from markdown or repair it
                    try:
                        from mcp_client_multi import extract_json_from_markdown
                        repaired_json = extract_json_from_markdown(response.content)
                        if repaired_json:
                            try:
                                parsed_json = json.loads(repaired_json)
                                
                                # Log successful repair
                                self._log_cheap_request(
                                    prompt_messages=prompt_messages,
                                    schema_name=schema_name,
                                    model_used=target_model,
                                    response_data=parsed_json,
                                    input_tokens=input_tokens,
                                    output_tokens=output_tokens,
                                    cost_usd=cost_usd
                                )
                                
                                return parsed_json
                            except json.JSONDecodeError:
                                log.warning(f"query_llm_structured: JSON repair also failed")
                    except ImportError:
                        log.debug("extract_json_from_markdown not available")
                    
                    # Log failed request
                    self._log_cheap_request(
                        prompt_messages=prompt_messages,
                        schema_name=schema_name,
                        model_used=target_model,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cost_usd=cost_usd,
                        error=error_msg
                    )
                    
                    return None
            else:
                error_msg = "No content received from LLM"
                log.warning(f"query_llm_structured: {error_msg}")
                
                # Log failed request
                self._log_cheap_request(
                    prompt_messages=prompt_messages,
                    schema_name=schema_name,
                    model_used=target_model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost_usd,
                    error=error_msg
                )
                
                return None
                
        except Exception as e:
            error_msg = f"query_llm_structured error: {e}"
            log.error(error_msg, exc_info=True)
            
            # Log failed request
            self._log_cheap_request(
                prompt_messages=prompt_messages,
                schema_name=schema_name,
                model_used=target_model,
                error=error_msg
            )
            
            return None

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
        )

        # Prepare the message list for the internal call:
        # System prompt + Current History
        messages_for_summary_call: InternalMessageList = [InternalMessage(role="system", content=summarization_prompt)]
        messages_for_summary_call.extend(current_messages_to_summarize)

        # --- Call the new internal helper ---
        try:
            # Execute the call without side effects
            summary_text = await self._execute_llm_call_no_history(
                messages=messages_for_summary_call,
                model=summarization_model,
                max_tokens=target_length + 500,  # Allow slightly more tokens for the summary itself
                temperature=0.5,  # Lower temperature might be better for summarization
            )

            if summary_text:
                log.info(f"Summarization successful. Summary length: {len(summary_text)} chars.")
                return summary_text.strip()
            else:
                log.warning("Summarization model returned no content.")
                return None  # Explicitly return None on empty response

        except Exception as e:
            # Error is logged within the helper, just return None here
            log.error(f"Summarization failed: Error occurred during internal LLM call - {e}", exc_info=False)  # Don't need full traceback here again
            return None

    def _format_messages_for_provider(
        self,
        messages: InternalMessageList,
        provider: str,
        model_name: str,  # Keep model_name for potential future provider-specific logic
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Formats internal message list into the specific structure required by the provider.
        Extracts the system prompt for providers that use a top-level parameter (Anthropic).

        Returns:
            A tuple containing:
            - formatted_messages: List of message dictionaries for the provider's API.
            - system_prompt: The extracted system prompt string (Anthropic), or None (OpenAI).
        """
        formatted_messages: List[Dict[str, Any]] = []
        system_prompt_extracted: Optional[str] = None
        messages_to_process: InternalMessageList = []  # Messages excluding the system prompt for Anthropic

        # First pass: Extract system prompt and filter messages for Anthropic
        first_system_message_found = False
        for msg in messages:
            # Use .get() for safer access, check if msg is a dict
            if isinstance(msg, dict) and msg.get("role") == "system" and not first_system_message_found:
                system_prompt_extracted = self._extract_text_from_internal_content(msg.get("content"))
                first_system_message_found = True
                # Don't add system messages to messages_to_process if Anthropic
                if provider != Provider.ANTHROPIC.value:
                    messages_to_process.append(msg)
            else:
                messages_to_process.append(msg)  # Add non-system or subsequent system messages

        log.debug(
            f"Formatting {len(messages_to_process)} non-system messages for provider '{provider}'. System prompt extracted: {bool(system_prompt_extracted)}"
        )

        # --- Anthropic Formatting (Uses messages_to_process) ---
        if provider == Provider.ANTHROPIC.value:
            # Process messages_to_process (which excludes the first system message)
            for msg in messages_to_process:
                role = msg["role"]
                content = msg["content"]
                # Anthropic system messages are handled by the top-level param, skip any remaining here
                if role == "system":
                    log.warning("Skipping subsequent system message found when formatting for Anthropic.")
                    continue

                api_role = role  # user or assistant
                api_content_list: List[Dict[str, Any]]

                if isinstance(content, str):
                    api_content_list = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    processed_blocks: List[Dict[str, Any]] = []
                    for block in content:
                        block_type = block.get("type")
                        if block_type == "text":
                            processed_blocks.append({"type": "text", "text": block.get("text", "")})
                        elif block_type == "tool_use":
                            original_tool_name = block.get("name", "unknown_tool")
                            # --- CORRECTED LOGIC ---
                            # Apply the same sanitization used when defining tools for the current provider
                            # Anthropic names: ^[a-zA-Z0-9_-]{1,64}$
                            sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", original_tool_name)[:64]
                            if not sanitized_name:  # Handle case where sanitization results in empty string
                                log.warning(
                                    f"Original tool name '{original_tool_name}' resulted in empty sanitized name during Anthropic history formatting. Skipping tool_use block."
                                )
                                continue  # Skip this block if sanitization failed

                            # Log the sanitization happening during history formatting for clarity
                            if original_tool_name != sanitized_name:
                                log.debug(
                                    f"History Formatting (Anthropic): Sanitized '{original_tool_name}' -> '{sanitized_name}' for historical tool_use block."
                                )
                            # --- END CORRECTION ---

                            processed_blocks.append(
                                {"type": "tool_use", "id": block.get("id", ""), "name": sanitized_name, "input": block.get("input", {})}
                            )
                        elif block_type == "tool_result":
                            result_content = block.get("content")
                            # Anthropic expects tool result content to be a string or list of blocks.
                            # Stringify simply for now. Richer handling might be needed.
                            stringified_content = self._stringify_content(result_content)
                            # Wrap string content in a text block if needed, or handle list of blocks directly
                            result_content_api = [{"type": "text", "text": stringified_content}]  # Simplest form
                            result_block_api = {"type": "tool_result", "tool_use_id": block.get("tool_use_id", ""), "content": result_content_api}
                            if block.get("is_error") is True:
                                result_block_api["is_error"] = True
                            processed_blocks.append(result_block_api)
                        else:
                            log.warning(f"Skipping unknown block type during Anthropic formatting: {block_type}")
                    api_content_list = processed_blocks
                elif content is None:
                    api_content_list = [{"type": "text", "text": ""}]
                else:
                    log.warning(f"Unexpected content type for Anthropic: {type(content)}. Converting to string block.")
                    api_content_list = [{"type": "text", "text": str(content)}]

                formatted_messages.append({"role": api_role, "content": api_content_list})

            # Return messages *without* system prompt, and the extracted prompt
            return formatted_messages, system_prompt_extracted

        # --- OpenAI-Compatible Formatting (Uses messages_to_process, which *includes* system message) ---
        elif provider in [
            Provider.OPENAI.value,
            Provider.GROK.value,
            Provider.DEEPSEEK.value,
            Provider.GROQ.value,
            Provider.GEMINI.value,
            Provider.MISTRAL.value,
            Provider.CEREBRAS.value,
            Provider.OPENROUTER.value,  # Added openrouter
        ]:
            # Process messages_to_process (which includes system message for OpenAI)
            for msg in messages_to_process:
                role = msg["role"]
                content = msg["content"]
                openai_role = role  # Roles map directly (system, user, assistant)

                # Handle System Prompt (already in messages_to_process)
                if openai_role == "system":
                    system_text = self._extract_text_from_internal_content(content)
                    formatted_messages.append({"role": "system", "content": system_text})

                # Handle User Messages (including internal tool results)
                elif openai_role == "user":
                    is_internal_tool_result = False
                    tool_call_id_for_result = None
                    tool_result_content = None
                    tool_is_error = False
                    if isinstance(content, list) and content and isinstance(content[0], dict) and content[0].get("type") == "tool_result":
                        is_internal_tool_result = True
                        tool_call_id_for_result = content[0].get("tool_use_id")
                        tool_result_content = content[0].get("content")
                        tool_is_error = content[0].get("is_error", False)

                    if is_internal_tool_result:
                        if tool_call_id_for_result:
                            MAX_OPENAI_TOOL_ID_LEN = 40
                            truncated_tool_call_id = tool_call_id_for_result[:MAX_OPENAI_TOOL_ID_LEN]
                            if len(tool_call_id_for_result) > MAX_OPENAI_TOOL_ID_LEN:
                                log.debug(f"Truncating tool_call_id '{tool_call_id_for_result}' to '{truncated_tool_call_id}' for OpenAI history.")
                            stringified_result = self._stringify_content(tool_result_content)
                            final_tool_content = f"Error: {stringified_result}" if tool_is_error else stringified_result
                            formatted_messages.append(
                                {"role": "tool", "tool_call_id": truncated_tool_call_id, "content": final_tool_content}
                            )  # Use truncated ID
                        else:
                            log.warning("Internal tool result missing 'tool_use_id'. Skipping.")
                    else:
                        user_text = self._extract_text_from_internal_content(content)
                        formatted_messages.append({"role": "user", "content": user_text})

                # Handle Assistant Messages
                elif openai_role == "assistant":
                    assistant_text_content: Optional[str] = None
                    tool_calls_for_api: List[Dict[str, Any]] = []

                    if isinstance(content, str):
                        # Ensure we don't send empty strings if the original was just ""
                        assistant_text_content = content if content else None
                    elif isinstance(content, list):
                        text_parts = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text_parts.append(block.get("text", ""))
                            elif isinstance(block, dict) and block.get("type") == "tool_use":
                                # --- Tool call parsing logic (remains the same) ---
                                original_tool_name_from_history = block.get("name", "unknown_tool")
                                # --- Truncate tool_call_id for OpenAI Provider ---
                                MAX_OPENAI_TOOL_ID_LEN = 40
                                tool_call_id = block.get("id", "")[:MAX_OPENAI_TOOL_ID_LEN]  # Truncate here
                                if len(block.get("id", "")) > MAX_OPENAI_TOOL_ID_LEN:
                                    log.debug(f"Truncating assistant tool_call.id '{block.get('id', '')}' to '{tool_call_id}' for OpenAI history.")
                                tool_input = block.get("input", {})
                                name_for_api = re.sub(r"[^a-zA-Z0-9_-]", "_", original_tool_name_from_history)[:64]
                                if not name_for_api:
                                    log.warning(
                                        f"Tool name '{original_tool_name_from_history}' resulted in empty sanitized name. Skipping tool call in history."
                                    )
                                    continue
                                try:
                                    arguments_str = json.dumps(tool_input)
                                except TypeError:
                                    log.error(f"Could not JSON-stringify tool input for '{name_for_api}'. Sending empty args.")
                                    arguments_str = "{}"
                                tool_calls_for_api.append(
                                    {"id": tool_call_id, "type": "function", "function": {"name": name_for_api, "arguments": arguments_str}}
                                )
                                # --- End tool call parsing ---
                        # Combine text parts, ensure None if empty after stripping
                        combined_text = "\n".join(text_parts).strip()
                        assistant_text_content = combined_text if combined_text else None
                    elif content is not None:  # Handle unexpected types
                        assistant_text_content = str(content) if str(content) else None

                    # --- Build the payload carefully ---
                    assistant_msg_payload: Dict[str, Any] = {"role": "assistant"}
                    added_something = False

                    # Add 'content' ONLY if it's a non-empty string
                    if assistant_text_content:
                        assistant_msg_payload["content"] = assistant_text_content
                        added_something = True
                        log.debug("History Formatting (OpenAI): Added assistant text content.")

                    # Add 'tool_calls' ONLY if the list is not empty
                    if tool_calls_for_api:
                        assistant_msg_payload["tool_calls"] = tool_calls_for_api
                        added_something = True
                        log.debug(f"History Formatting (OpenAI): Added {len(tool_calls_for_api)} assistant tool_calls.")

                    # Append the message ONLY if content or tool_calls were added
                    if added_something:
                        log.debug(f"History Formatting (OpenAI): Appending assistant message. Payload keys: {list(assistant_msg_payload.keys())}")
                        formatted_messages.append(assistant_msg_payload)
                    else:
                        # Log if we intended to send an assistant message but it ended up empty
                        log.warning(f"Skipping empty assistant message during formatting. Original content type: {type(content)}")
                    # --- End payload building ---

            # Return messages *with* system prompt, and None for extracted prompt
            return formatted_messages, None  # No separate system prompt for OpenAI call

        # --- Unknown Provider ---
        else:
            log.error(f"Message formatting failed: Provider '{provider}' is not supported.")
            return [], None  # Indicate failure

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
                        current_text_block = None  # Block finished
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
                    if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
                        stop_reason = event.delta.stop_reason
                    if hasattr(event, "usage") and event.usage:  # Track output tokens from delta
                        output_tokens = event.usage.output_tokens  # Anthropic provides cumulative output tokens
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

    async def _initialize_openai_compatible_client(
        self, provider_name: str, api_key_attr: str, base_url_attr: str, default_base_url: Optional[str], client_attr: str, emoji_key: str
    ) -> Tuple[Optional[AsyncOpenAI], str]:
        """
        Initializes and validates an AsyncOpenAI client for a compatible provider.
        (Enhanced Error Handling)
        """
        status_emoji = EMOJI_MAP.get(emoji_key, EMOJI_MAP["provider"])
        provider_title = provider_name.capitalize()
        api_key = getattr(self.config, api_key_attr, None)

        if not api_key:
            return None, f"{status_emoji} {provider_title}: [yellow]No Key[/]"

        client_instance = None  # Initialize to None
        try:
            base_url = getattr(self.config, base_url_attr, None) or default_base_url
            if not base_url and provider_name != Provider.OPENAI.value:  # OpenAI default is handled by SDK
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
            await client_instance.models.list()  # Raises errors on failure
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
        except OpenAIAPIConnectionError as e:  # Often wraps httpx errors
            error_msg = f"{provider_title} initialization failed: Connection Error ({e})"
            log.error(error_msg, exc_info=True)  # Include details
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Conn Error[/]"
        except openai.BadRequestError as e:  # e.g., Malformed URL during validation
            error_msg = f"{provider_title} initialization failed: Bad Request ({e})"
            log.error(error_msg, exc_info=True)
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Bad Request[/]"
        except openai.NotFoundError as e:  # e.g., Base URL valid but path incorrect
            error_msg = f"{provider_title} initialization failed: Not Found ({e})"
            log.error(error_msg, exc_info=True)
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Not Found[/]"
        except httpx.RequestError as e:  # Catch underlying network issues if not wrapped by OpenAI error
            error_msg = f"{provider_title} initialization failed: Network Error ({e})"
            log.error(error_msg, exc_info=True)
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Network Err[/]"
        except ValueError as e:  # Catch our manual ValueError for missing default URL
            error_msg = f"{provider_title} initialization failed: Configuration Error ({e})"
            log.error(error_msg)
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Config Error[/]"
        except Exception as e:  # Catch-all for unexpected init errors
            error_msg = f"{provider_title} initialization failed: Unexpected Error ({type(e).__name__})"
            log.error(error_msg, exc_info=True)
            self.safe_print(f"[bold red]{error_msg}[/]")
            setattr(self, client_attr, None)
            return None, f"{status_emoji} {provider_title}: [red]Failed[/]"

    async def setup(self, interactive_mode=False):
        """Set up the client, load configs, initialize providers, discover servers."""
        safe_console = get_safe_console()

        # Mappings for provider names, config attributes, and env variable names
        provider_keys = {  # Map provider enum value to config attribute
            Provider.ANTHROPIC.value: "anthropic_api_key",
            Provider.OPENAI.value: "openai_api_key",
            Provider.GEMINI.value: "gemini_api_key",
            Provider.GROK.value: "grok_api_key",
            Provider.DEEPSEEK.value: "deepseek_api_key",
            Provider.MISTRAL.value: "mistral_api_key",
            Provider.GROQ.value: "groq_api_key",
            Provider.CEREBRAS.value: "cerebras_api_key",
            Provider.OPENROUTER.value: "openrouter_api_key",  # Added OpenRouter
        }
        provider_env_vars = {  # Map provider enum value to expected .env var name
            Provider.ANTHROPIC.value: "ANTHROPIC_API_KEY",
            Provider.OPENAI.value: "OPENAI_API_KEY",
            Provider.GEMINI.value: "GOOGLE_API_KEY",  # Special case for Gemini - OR GEMINI_API_KEY? Check decouple logic. Assuming GOOGLE_API_KEY based on common practice.
            Provider.GROK.value: "GROK_API_KEY",
            Provider.DEEPSEEK.value: "DEEPSEEK_API_KEY",
            Provider.MISTRAL.value: "MISTRAL_API_KEY",
            Provider.GROQ.value: "GROQ_API_KEY",
            Provider.CEREBRAS.value: "CEREBRAS_API_KEY",
            Provider.OPENROUTER.value: "OPENROUTER_API_KEY",  # Added OpenRouter
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
                        log.warning(
                            f"API key for default provider '{default_provider.value}' ({default_provider_key_env_var}) is missing or empty in current config."
                        )
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
                api_key_input = Prompt.ask(f"Enter {default_provider.value.capitalize()} API Key", default="", console=safe_console, password=True)

                if api_key_input.strip():
                    entered_key = api_key_input.strip()

                    # Use the dotenv_path stored in the config object
                    target_dotenv_path = self.config.dotenv_path

                    if target_dotenv_path:  # Should always be true now
                        try:
                            # Ensure parent directory exists
                            Path(target_dotenv_path).parent.mkdir(parents=True, exist_ok=True)
                            # Use dotenv.set_key with the correct path
                            success = set_key(target_dotenv_path, default_provider_key_env_var, entered_key, quote_mode="always")
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
                    self.safe_print(
                        f"[yellow]Skipped entering key for {default_provider.value}. {default_provider.value.capitalize()} features might be unavailable.[/]"
                    )

            except Exception as prompt_err:
                self.safe_print(f"[red]Error during API key prompt/save: {prompt_err}[/]")
                log.error("Error during API key prompt/save", exc_info=True)

        # Exit if not interactive and default provider key is missing
        elif not interactive_mode and default_provider and key_missing:
            self.safe_print(
                f"[bold red]ERROR: API key for default provider '{default_provider.value}' ({default_provider_key_env_var}) not found.[/]"
            )
            self.safe_print("Please set the key in your '.env' file or as an environment variable.")
            sys.exit(1)

        # --- 2. Initialize Provider SDK Clients ---
        # This will now use the correct key value in self.config if it was updated above
        await self._reinitialize_provider_clients()  # Call the helper to init all needed clients

        # --- 3. Load Conversation Graph ---
        # (No changes needed here)
        self.conversation_graph = ConversationGraph()
        if self.conversation_graph_file.exists():
            with Status(f"{EMOJI_MAP['history']} Loading conversation state...", console=safe_console) as status:
                try:
                    loaded_graph = await ConversationGraph.load(str(self.conversation_graph_file))
                    self.conversation_graph = loaded_graph
                    is_new_graph = (
                        loaded_graph.root.id == "root"
                        and not loaded_graph.root.messages
                        and not loaded_graph.root.children
                        and len(loaded_graph.nodes) == 1
                    )
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
            if server_config.type == ServerType.STDIO:
                identifier = (server_config.type, server_config.path, frozenset(server_config.args))
            elif server_config.type == ServerType.SSE:
                identifier = (server_config.type, server_config.path)
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
            await self.config.save_async()  # Saves YAML
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
                await self.server_manager.discover_servers()  # Populates cache
                await self.server_manager._process_discovery_results(interactive_mode=interactive_mode)  # Adds to config
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
                    connection_results[name] = session is not None
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
        """
        Determine the provider based on the model name.
        Uses the refined _infer_provider function.
        """
        if not model_name:
            log.warning("get_provider_from_model called with empty model name.")
            return None
        provider = _infer_provider(model_name)  # Use the refined inference function
        if provider:
            log.debug(f"Determined provider for '{model_name}': {provider}")
            return provider
        else:
            log.warning(
                f"Could not determine provider for model: '{model_name}'. Ensure it has a known prefix (e.g., 'openrouter/', 'groq/') or is in MODEL_PROVIDER_MAP."
            )
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
            return ""  # Return empty string for None content
        else:
            # Fallback for unexpected content types
            log.warning(f"Unexpected content type in _extract_text: {type(content)}. Converting to string.")
            return str(content)

    # --- Main Formatting Function ---
    def _format_tools_for_provider(self, provider: str) -> Optional[List[Dict[str, Any]]]:
        """
        Formats the available MCP tools into the specific structure required by the target LLM provider's API.

        Handles sanitization of tool names and validation of input schemas according to provider requirements.
        Ensures that the sanitized names returned in the formatted list are unique FOR THIS CALL.
        It also updates the shared `self.server_manager.sanitized_to_original` map with the
        sanitized_name -> original_mcp_name mapping for the tools included in the returned list.

        Args:
            provider: The string identifier of the target provider (e.g., "openai", "anthropic").

        Returns:
            A list of dictionaries representing the tools in the provider's expected format,
            or None if no tools are available or an error occurs.
        """
        # Use a distinct logger prefix for this critical formatting function
        log_prefix = f"MCPC_FMT_TOOLS ({provider})"

        mcp_tools_all = list(self.server_manager.tools.values())  # Get all tools known to ServerManager
        if not mcp_tools_all:
            log.debug(f"{log_prefix}: No MCP tools registered in ServerManager to format.")
            return None

        # 1. Filter tools based on the blacklist for the current provider
        blacklisted_for_provider = self.tool_blacklist_manager.get_blacklisted_tools(provider)
        mcp_tools_filtered = [
            tool
            for tool in mcp_tools_all
            if not self.tool_blacklist_manager.is_blacklisted(provider, tool.name)  # tool.name is original MCP name
        ]

        if len(mcp_tools_all) != len(mcp_tools_filtered):
            excluded_count = len(mcp_tools_all) - len(mcp_tools_filtered)
            log.info(f"{log_prefix}: Excluded {excluded_count} blacklisted tools. Blacklist for '{provider}': {blacklisted_for_provider}")
        else:
            log.debug(f"{log_prefix}: No tools blacklisted for provider '{provider}'. Formatting all {len(mcp_tools_filtered)} tools.")

        if not mcp_tools_filtered:
            log.info(f"{log_prefix}: No tools remaining after blacklist filtering.")
            return None

        # This set tracks sanitized names *generated during this specific call* to ensure uniqueness
        # in the list being returned *by this function*.
        used_sanitized_names_this_call: Set[str] = set()
        formatted_tools_for_provider: List[Dict[str, Any]] = []

        log.info(f"{log_prefix}: Starting formatting for {len(mcp_tools_filtered)} tools.")
        # Note: self.server_manager.sanitized_to_original will be updated by this function.
        # It's crucial to handle existing mappings carefully if a tool's sanitized name changes.

        provider_enum_val = provider  # Assumes provider is already validated and lowercased

        # --- Provider-Specific Formatting ---

        # ANTHROPIC
        if provider_enum_val == Provider.ANTHROPIC.value:
            log.debug(f"{log_prefix}: Formatting for Anthropic.")
            for tool_idx, tool_obj in enumerate(sorted(mcp_tools_filtered, key=lambda t: t.name)):
                original_mcp_name = tool_obj.name  # This is the "Server:Function" name

                # Sanitize original_mcp_name according to Anthropic rules: ^[a-zA-Z0-9_-]{1,64}$
                base_sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", original_mcp_name)[:64]
                if not base_sanitized_name:  # Handle empty string after sanitization
                    base_sanitized_name = f"tool_{str(uuid.uuid4())[:8]}"  # Fallback unique name
                    log.warning(
                        f"{log_prefix}: Original name '{original_mcp_name}' for Anthropic sanitized to empty, using fallback '{base_sanitized_name}'."
                    )

                # Ensure uniqueness for this provider call
                final_sanitized_name_for_llm = base_sanitized_name
                counter = 1
                while final_sanitized_name_for_llm in used_sanitized_names_this_call:
                    suffix = f"_v{counter}"
                    if len(base_sanitized_name) + len(suffix) > 64:
                        final_sanitized_name_for_llm = base_sanitized_name[: 64 - len(suffix)] + suffix
                    else:
                        final_sanitized_name_for_llm = base_sanitized_name + suffix
                    counter += 1

                used_sanitized_names_this_call.add(final_sanitized_name_for_llm)

                # Update the shared sanitized_to_original map in ServerManager
                # If the sanitized name changed from the base (due to _vX suffix),
                # and an old mapping for base_sanitized_name existed for THIS original_mcp_name, remove it.
                if (
                    base_sanitized_name != final_sanitized_name_for_llm
                    and base_sanitized_name in self.server_manager.sanitized_to_original
                    and self.server_manager.sanitized_to_original[base_sanitized_name] == original_mcp_name
                ):
                    del self.server_manager.sanitized_to_original[base_sanitized_name]
                    log.debug(
                        f"{log_prefix}: Removed prior shared mapping for '{base_sanitized_name}' as it's being updated to '{final_sanitized_name_for_llm}' for '{original_mcp_name}'."
                    )

                # Add/Update the mapping for the name being sent to the LLM
                self.server_manager.sanitized_to_original[final_sanitized_name_for_llm] = original_mcp_name
                log.debug(
                    f"{log_prefix}: Anthropic Tool {tool_idx + 1}: Original='{original_mcp_name}', FinalSanitized='{final_sanitized_name_for_llm}'. Mapping updated in ServerManager."
                )

                input_schema = tool_obj.input_schema
                if not isinstance(input_schema, dict):
                    log.warning(
                        f"{log_prefix}: Tool '{original_mcp_name}' for Anthropic has invalid schema type ({type(input_schema)}), expected dict. Sending empty schema."
                    )
                    input_schema = {"type": "object", "properties": {}, "required": []}
                elif not input_schema:  # Ensure empty dict if it's falsey but not None (e.g. empty string)
                    input_schema = {"type": "object", "properties": {}, "required": []}

                formatted_tools_for_provider.append(
                    {
                        "name": final_sanitized_name_for_llm,
                        "description": tool_obj.description or "No description provided.",
                        "input_schema": input_schema,
                    }
                )

            if formatted_tools_for_provider:  # Add cache_control for Anthropic
                formatted_tools_for_provider[-1]["cache_control"] = {"type": "ephemeral"}
                log.debug(f"{log_prefix}: Added ephemeral cache_control to the last Anthropic tool.")

        # OPENAI-COMPATIBLE (OpenAI, Grok, DeepSeek, Groq, Mistral, Cerebras, Gemini, OpenRouter)
        elif provider_enum_val in [
            Provider.OPENAI.value,
            Provider.GROK.value,
            Provider.DEEPSEEK.value,
            Provider.GROQ.value,
            Provider.MISTRAL.value,
            Provider.CEREBRAS.value,
            Provider.GEMINI.value,
            Provider.OPENROUTER.value,
        ]:
            log.debug(f"{log_prefix}: Formatting for OpenAI-compatible provider: {provider_enum_val}")

            temp_formatted_list_for_truncation: List[Dict[str, Any]] = []
            for tool_idx, tool_obj in enumerate(sorted(mcp_tools_filtered, key=lambda t: t.name)):
                original_mcp_name = tool_obj.name

                # Sanitize original_mcp_name: ^[a-zA-Z0-9_-]{1,64}$
                base_sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", original_mcp_name)[:64]
                if not base_sanitized_name:
                    base_sanitized_name = f"tool_{str(uuid.uuid4())[:8]}"
                    log.warning(
                        f"{log_prefix}: Original name '{original_mcp_name}' for {provider_enum_val} sanitized to empty, using fallback '{base_sanitized_name}'."
                    )

                # Ensure uniqueness for this provider call
                final_sanitized_name_for_llm = base_sanitized_name
                counter = 1
                while final_sanitized_name_for_llm in used_sanitized_names_this_call:
                    suffix = f"_v{counter}"
                    if len(base_sanitized_name) + len(suffix) > 64:
                        final_sanitized_name_for_llm = base_sanitized_name[: 64 - len(suffix)] + suffix
                    else:
                        final_sanitized_name_for_llm = base_sanitized_name + suffix
                    counter += 1

                used_sanitized_names_this_call.add(final_sanitized_name_for_llm)

                # Update ServerManager's shared map
                if (
                    base_sanitized_name != final_sanitized_name_for_llm
                    and base_sanitized_name in self.server_manager.sanitized_to_original
                    and self.server_manager.sanitized_to_original[base_sanitized_name] == original_mcp_name
                ):
                    del self.server_manager.sanitized_to_original[base_sanitized_name]
                    log.debug(
                        f"{log_prefix}: Removed prior shared mapping for '{base_sanitized_name}' as it's being updated to '{final_sanitized_name_for_llm}' for '{original_mcp_name}'."
                    )

                self.server_manager.sanitized_to_original[final_sanitized_name_for_llm] = original_mcp_name
                log.debug(
                    f"{log_prefix}: OpenAI-like Tool {tool_idx + 1}: Original='{original_mcp_name}', FinalSanitized='{final_sanitized_name_for_llm}'. Mapping updated in ServerManager."
                )

                input_schema = tool_obj.input_schema
                validated_schema: Dict[str, Any]
                if isinstance(input_schema, dict) and input_schema.get("type") == "object":
                    validated_schema = {
                        "type": "object",
                        "properties": input_schema.get("properties", {}),  # Default to empty dict if missing
                        "required": input_schema.get("required", []),  # Default to empty list if missing
                    }
                    if not isinstance(validated_schema["properties"], dict):
                        log.warning(
                            f"{log_prefix}: Tool '{original_mcp_name}' schema 'properties' is not a dict ({type(validated_schema['properties'])}). Using empty dict."
                        )
                        validated_schema["properties"] = {}
                    if not isinstance(validated_schema["required"], list):
                        log.warning(
                            f"{log_prefix}: Tool '{original_mcp_name}' schema 'required' is not a list ({type(validated_schema['required'])}). Using empty list."
                        )
                        validated_schema["required"] = []
                else:
                    log.warning(
                        f"{log_prefix}: Tool '{original_mcp_name}' for {provider_enum_val} has invalid schema root type ({input_schema.get('type') if isinstance(input_schema, dict) else type(input_schema)}), expected 'object'. Using empty schema."
                    )
                    validated_schema = {"type": "object", "properties": {}, "required": []}

                temp_formatted_list_for_truncation.append(
                    {
                        "type": "function",
                        "function": {
                            "name": final_sanitized_name_for_llm,
                            "description": tool_obj.description or "No description provided.",
                            "parameters": validated_schema,
                        },
                    }
                )

            # Apply OpenAI_MAX_TOOL_COUNT truncation if needed
            if len(temp_formatted_list_for_truncation) > OPENAI_MAX_TOOL_COUNT:
                original_tool_count = len(temp_formatted_list_for_truncation)
                formatted_tools_for_provider = temp_formatted_list_for_truncation[:OPENAI_MAX_TOOL_COUNT]
                truncated_tool_count = len(formatted_tools_for_provider)

                # Identify and log excluded tools, and remove their mappings from the shared map
                kept_sanitized_names_in_final_list = {t["function"]["name"] for t in formatted_tools_for_provider}
                excluded_original_names_log: List[str] = []

                for tool_dict_from_full_list in temp_formatted_list_for_truncation:
                    sanitized_name_in_full_list = tool_dict_from_full_list["function"]["name"]
                    if sanitized_name_in_full_list not in kept_sanitized_names_in_final_list:
                        # This tool was truncated from the list sent to LLM
                        original_mcp_name_of_truncated = self.server_manager.sanitized_to_original.get(sanitized_name_in_full_list)
                        excluded_original_names_log.append(
                            original_mcp_name_of_truncated or f"UnknownOriginal(Sanitized:{sanitized_name_in_full_list})"
                        )
                        # Remove its mapping from the shared ServerManager map as it won't be sent
                        if sanitized_name_in_full_list in self.server_manager.sanitized_to_original:
                            del self.server_manager.sanitized_to_original[sanitized_name_in_full_list]
                            log.debug(
                                f"{log_prefix}: Removed mapping for TRUNCATED tool '{sanitized_name_in_full_list}' -> '{original_mcp_name_of_truncated}'."
                            )

                log.warning(
                    f"{log_prefix}: Tool list for {provider_enum_val} ({original_tool_count} tools) exceeded limit ({OPENAI_MAX_TOOL_COUNT}). Truncated to {truncated_tool_count} tools. Excluded: {', '.join(sorted(excluded_original_names_log))}"
                )
            else:
                formatted_tools_for_provider = temp_formatted_list_for_truncation

        else:
            log.error(f"{log_prefix}: Tool formatting not implemented or provider '{provider}' unknown. Returning no tools.")
            return None

        final_tool_count = len(formatted_tools_for_provider)
        log.info(
            f"{log_prefix}: Successfully formatted {final_tool_count} tools. Sanitized names generated this call: {len(used_sanitized_names_this_call)}."
        )
        # Log final state of shared map for debugging if it changed significantly
        # log.debug(f"{log_prefix}: Final self.server_manager.sanitized_to_original map sample: {str(dict(list(self.server_manager.sanitized_to_original.items())[:10]))}")
        return formatted_tools_for_provider if formatted_tools_for_provider else None

    # --- Streaming Handlers (_handle_*_stream) ---
    async def _handle_anthropic_stream(self, stream: AsyncMessageStream) -> AsyncGenerator[Tuple[str, Any], None]:
        current_text_block = None
        # current_tool_use_block stores {'id': tool_id, 'name': original_tool_name}
        current_tool_use_block_info: Optional[Dict[str, str]] = None  # Store full info
        current_tool_input_json_accumulator = ""
        input_tokens = 0
        output_tokens = 0
        stop_reason = "unknown"

        try:
            async for event in stream:
                try:
                    event_type = event.type
                    if event_type == "message_start":
                        input_tokens = event.message.usage.input_tokens
                    elif event_type == "content_block_start":
                        block_type = event.content_block.type
                        if block_type == "text":
                            current_text_block = {"type": "text", "text": ""}
                        elif block_type == "tool_use":
                            tool_id = event.content_block.id
                            sanitized_tool_name_from_llm = event.content_block.name  # Name LLM used
                            # Map back to original MCP name
                            original_tool_name = self.server_manager.sanitized_to_original.get(
                                sanitized_tool_name_from_llm, sanitized_tool_name_from_llm
                            )
                            if sanitized_tool_name_from_llm != original_tool_name:
                                log.debug(
                                    f"Anthropic Stream: LLM used tool '{sanitized_tool_name_from_llm}', mapped to original '{original_tool_name}'."
                                )
                            else:
                                log.debug(f"Anthropic Stream: LLM used tool '{original_tool_name}' (no change from sanitized).")

                            current_tool_use_block_info = {"id": tool_id, "name": original_tool_name}  # Store original name
                            current_tool_input_json_accumulator = ""
                            yield ("tool_call_start", {"id": tool_id, "name": original_tool_name})  # Yield original name
                    elif event_type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            if current_text_block is not None:
                                yield ("text_chunk", delta.text)
                        elif delta.type == "input_json_delta":
                            if current_tool_use_block_info is not None:  # Check the info dict
                                current_tool_input_json_accumulator += delta.partial_json
                                yield ("tool_call_input_chunk", {"id": current_tool_use_block_info["id"], "json_chunk": delta.partial_json})
                    elif event_type == "content_block_stop":
                        if current_text_block is not None:
                            current_text_block = None
                        elif current_tool_use_block_info is not None:  # Check the info dict
                            parsed_input = {}
                            try:
                                parsed_input = json.loads(current_tool_input_json_accumulator) if current_tool_input_json_accumulator else {}
                            except json.JSONDecodeError as e:
                                log.error(
                                    f"Anthropic JSON parse failed for tool {current_tool_use_block_info['name']} (ID: {current_tool_use_block_info['id']}): {e}. Raw: '{current_tool_input_json_accumulator}'"
                                )
                                parsed_input = {"_tool_input_parse_error": f"Failed: {e}"}

                            yield (
                                "tool_call_end",
                                {
                                    "id": current_tool_use_block_info["id"],
                                    "name": current_tool_use_block_info["name"],  # Include the name
                                    "parsed_input": parsed_input,
                                },
                            )
                            current_tool_use_block_info = None  # Reset after processing
                    elif event_type == "message_delta":
                        if hasattr(event.delta, "stop_reason") and event.delta.stop_reason:
                            stop_reason = event.delta.stop_reason
                        if hasattr(event, "usage") and event.usage:
                            output_tokens = event.usage.output_tokens
                    elif event_type == "message_stop":
                        final_message = await stream.get_final_message()
                        stop_reason = final_message.stop_reason
                        output_tokens = final_message.usage.output_tokens
                        break
                    elif event_type == "error":
                        stream_error = getattr(event, "error", {})
                        error_message = stream_error.get("message", "Unknown stream error")
                        log.error(f"Anthropic stream reported error event: {error_message}")
                        yield ("error", f"Anthropic Stream Error: {error_message}")
                        stop_reason = "error"
                        break
                except Exception as event_proc_err:
                    log.error(
                        f"Error processing Anthropic stream event ({getattr(event, 'type', 'UNKNOWN_EVENT_TYPE')}): {event_proc_err}", exc_info=True
                    )
                    yield ("error", f"Error processing stream event: {event_proc_err}")
                    stop_reason = "error"
                    break

        except anthropic.APIConnectionError as e:
            log.error(f"Anthropic stream connection error: {e}")
            yield ("error", f"Anthropic Conn Error: {e}")
            stop_reason = "error"
        except anthropic.RateLimitError as e:
            log.warning(f"Anthropic stream rate limit error: {e}")
            yield ("error", f"Anthropic Rate Limit: {e}")
            stop_reason = "rate_limit"
        except anthropic.APIStatusError as e:
            log.error(f"Anthropic stream API status error ({e.status_code}): {e}")
            yield ("error", f"Anthropic API Error ({e.status_code}): {e}")
            stop_reason = "error"
        except anthropic.APIError as e:  # Catch other Anthropic API errors
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

    async def _handle_openai_compatible_stream(
        self, stream: AsyncStream[ChatCompletionChunk], provider_name: str
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        current_tool_calls: Dict[int, Dict] = {}
        input_tokens: Optional[int] = None
        output_tokens: Optional[int] = None
        stop_reason = "stop"
        finish_reason = None
        final_chunk_usage: Optional[Any] = None

        try:
            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if chunk.usage:
                    final_chunk_usage = chunk.usage
                    log.debug(f"Received usage data in stream chunk for {provider_name}: {final_chunk_usage}")
                if not choice and chunk.usage:
                    continue
                elif not choice:
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
                            sanitized_name = tc_chunk.function.name  # This is the name OpenAI/LLM returned
                            original_name = self.server_manager.sanitized_to_original.get(sanitized_name, sanitized_name)  # Map back

                            # Initialize current_tool_calls[idx] if it's a new tool call
                            if idx not in current_tool_calls:
                                current_tool_calls[idx] = {"id": tool_id, "name": original_name, "args_acc": ""}
                                yield ("tool_call_start", {"id": tool_id, "name": original_name})
                            # If it's a subsequent chunk for an existing tool call, ensure name is consistent (though it should be)
                            elif current_tool_calls[idx]["name"] != original_name:
                                log.warning(
                                    f"Tool name mismatch for index {idx}: expected {current_tool_calls[idx]['name']}, got {original_name}. Using original."
                                )
                                # Potentially update or just log, current logic assumes it's part of the same tool.

                        if tc_chunk.function and tc_chunk.function.arguments:
                            args_chunk = tc_chunk.function.arguments
                            if idx in current_tool_calls:  # Check if the tool call was properly started
                                current_tool_calls[idx]["args_acc"] += args_chunk
                                yield ("tool_call_input_chunk", {"id": current_tool_calls[idx]["id"], "json_chunk": args_chunk})
                            else:
                                log.warning(f"Args chunk for unknown tool index {idx} from {provider_name}. Tool ID from chunk: {tc_chunk.id}")

                if provider_name == Provider.GROQ.value and hasattr(chunk, "x_groq") and chunk.x_groq and hasattr(chunk.x_groq, "usage"):
                    usage = chunk.x_groq.usage
                    if usage:
                        if input_tokens is None:
                            input_tokens = getattr(usage, "prompt_tokens", None)
                        current_chunk_output = getattr(usage, "completion_tokens", 0)
                        output_tokens = max(output_tokens or 0, current_chunk_output)
                        log.debug(f"Groq chunk usage: In={input_tokens}, Out={output_tokens} (Chunk Out={current_chunk_output})")

            for idx, tool_data in current_tool_calls.items():
                accumulated_args = tool_data["args_acc"]
                parsed_input = {}
                try:
                    if accumulated_args:
                        parsed_input = json.loads(accumulated_args)
                except json.JSONDecodeError as e:
                    log.error(
                        f"{provider_name} JSON parse failed for tool {tool_data['name']} (ID: {tool_data['id']}): {e}. Raw: '{accumulated_args}'"
                    )
                    parsed_input = {"_tool_input_parse_error": f"Failed: {e}"}

                # Include "name" in the yielded event_data for "tool_call_end"
                yield (
                    "tool_call_end",
                    {
                        "id": tool_data["id"],
                        "name": tool_data["name"],  # Add the original_name here
                        "parsed_input": parsed_input,
                    },
                )

            stop_reason = finish_reason if finish_reason else "stop"
            if stop_reason == "tool_calls":
                stop_reason = "tool_use"

        except (OpenAIAPIError, OpenAIAPIConnectionError, OpenAIAuthenticationError) as e:
            log.error(f"{provider_name} stream API error: {e}")
            yield ("error", f"{provider_name} API Error: {e}")
            stop_reason = "error"
        except Exception as e:
            log.error(f"Unexpected error in {provider_name} stream handler: {e}", exc_info=True)
            yield ("error", f"Unexpected stream processing error: {e}")
            stop_reason = "error"
        finally:
            if final_chunk_usage:
                input_tokens = final_chunk_usage.prompt_tokens
                output_tokens = final_chunk_usage.completion_tokens
                log.debug(f"Using final usage from stream options for {provider_name}: In={input_tokens}, Out={output_tokens}")
            elif provider_name != Provider.GROQ.value:
                log.warning(f"Final usage chunk not received or parsed for {provider_name}. Usage will be 0 or based on Groq accumulation.")
                if input_tokens is None:
                    input_tokens = 0
                if output_tokens is None:
                    output_tokens = 0

            final_input_tokens = input_tokens if input_tokens is not None else 0
            final_output_tokens = output_tokens if output_tokens is not None else 0

            if final_input_tokens == 0 or final_output_tokens == 0:
                if final_chunk_usage:
                    log.warning(f"{provider_name} usage details reported as zero (Input: {final_input_tokens}, Output: {final_output_tokens}).")
                elif provider_name != Provider.GROQ.value:
                    log.warning(f"{provider_name} usage details unavailable. Cannot calculate cost accurately.")

            yield ("final_usage", {"input_tokens": final_input_tokens, "output_tokens": final_output_tokens})
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
        # Use the more specific error signature
        client_error_signature = "_tool_input_parse_error"  # From the stream handler

        skipped_indices = set()  # To track indices of messages to skip

        log.debug(f"Filtering history ({len(messages_in)} messages) for known client tool result parse errors...")

        # First pass: identify indices of faulty interactions to skip
        assistant_tool_uses_to_check: Dict[int, Set[str]] = {}

        for idx, msg in enumerate(messages_in):
            # Ensure msg is a dict before accessing keys
            if not isinstance(msg, dict):
                log.warning(f"Skipping non-dict message at index {idx} in history filtering.")
                continue

            # Use .get() for safer access
            role = msg.get("role")
            content = msg.get("content")

            if not role or content is None:  # Skip if essential keys missing
                continue

            if role == "assistant":
                # Check if content is a list (it should be for tool_use)
                if isinstance(content, list):
                    tool_use_ids: Set[str] = set()
                    for block in content:
                        # Check block type correctly
                        if isinstance(block, dict) and block.get("type") == "tool_use":
                            tool_use_id = block.get("id")
                            if tool_use_id:
                                tool_use_ids.add(tool_use_id)

                    if tool_use_ids:
                        assistant_tool_uses_to_check[idx] = tool_use_ids

            elif role == "user":
                # Check if this user message corresponds to a preceding assistant tool use
                prev_idx = idx - 1
                if prev_idx in assistant_tool_uses_to_check:
                    # Ensure content is a list (it should be for tool_result)
                    if isinstance(content, list):
                        corresponding_ids = assistant_tool_uses_to_check[prev_idx]
                        found_faulty_result = False
                        for block in content:
                            # Check block type correctly
                            if isinstance(block, dict) and block.get("type") == "tool_result":
                                block_tool_use_id = block.get("tool_use_id")
                                if block_tool_use_id in corresponding_ids:
                                    result_content = block.get("content")
                                    # Check if the result_content itself is the error dict
                                    error_found_in_content = False
                                    if isinstance(result_content, dict) and client_error_signature in result_content:
                                        error_found_in_content = True
                                    # Less likely, but check if it's a string containing the key
                                    elif isinstance(result_content, str) and client_error_signature in result_content:
                                        error_found_in_content = True

                                    if error_found_in_content:
                                        found_faulty_result = True
                                        log.warning(
                                            f"Found faulty client tool result for tool_use_id {block_tool_use_id} "
                                            f"at history index {idx}. Marking preceding assistant request "
                                            f"(index {prev_idx}) and this user result for filtering."
                                        )
                                        break  # Found one faulty result for this user message turn

                        if found_faulty_result:
                            # Mark both the assistant request and the user result for skipping
                            skipped_indices.add(prev_idx)
                            skipped_indices.add(idx)
                            del assistant_tool_uses_to_check[prev_idx]  # Avoid re-matching

        # Second pass: build the filtered list
        for idx, msg in enumerate(messages_in):
            if idx not in skipped_indices:
                messages_to_send.append(msg)
            else:
                # More informative log about *what* is being skipped
                role = msg.get("role", "UnknownRole") if isinstance(msg, dict) else "NonDict"
                content_preview = repr(msg.get("content", "NoContent"))[:50] + "..." if isinstance(msg, dict) else repr(msg)[:50] + "..."
                log.debug(
                    f"Skipping message at index {idx} (Role: {role}, Content Preview: {content_preview}) due to client tool result parse error linkage."
                )

        if len(messages_in) != len(messages_to_send):
            log.info(f"Filtered {len(messages_in) - len(messages_to_send)} messages due to client tool result parse errors.")
        else:
            log.debug("No client tool result parse errors found requiring filtering.")
        return messages_to_send

    async def auto_prune_context(self):
        """Auto-prune context based on token count if enabled."""
        if not self.use_auto_summarization:
            return
        try:
            token_count = await self.count_tokens()
            if token_count > self.config.auto_summarize_threshold:
                self.safe_print(
                    f"[yellow]{EMOJI_MAP['warning']} Context size ({token_count:,} tokens) exceeds threshold "
                    f"({self.config.auto_summarize_threshold:,}). Auto-summarizing...[/]"
                )
                # Use cmd_optimize which now calls the correct summarize_conversation
                # We pass the target token argument specifically
                await self.cmd_optimize(f"--tokens {self.config.max_summarized_tokens}")
        except Exception as e:
            log.error(f"Error during auto-pruning: {e}", exc_info=True)
            self.safe_print(f"[red]Error during automatic context pruning: {e}[/]")

    async def _relay_agent_llm_event_to_ui(self, sender: Callable[[str, Any], Coroutine], event_type: str, event_data: Any):
        """Safely sends an agent's LLM event to the UI via the provided sender."""
        if not sender:
            return
        try:
            # Prefix event types for agent's LLM stream to distinguish from user's query stream
            prefixed_event_type = f"agent_llm_{event_type}" if not event_type.startswith("agent_llm_") else event_type

            # For 'status' events, ensure data is a string or simple dict for JSON serialization
            payload_to_send = event_data
            if prefixed_event_type == "agent_llm_status" and not isinstance(event_data, (str, dict)):
                payload_to_send = {"text": str(event_data)}
            elif prefixed_event_type == "agent_llm_error" and not isinstance(event_data, (str, dict)):
                payload_to_send = {"message": str(event_data)}

            await sender(prefixed_event_type, payload_to_send)
        except Exception as e:
            log.warning(f"MCPC: Error relaying agent LLM event '{prefixed_event_type}' to UI: {e}", exc_info=False)

    def _detect_ums_tool_in_request(
        self, 
        formatted_tools_for_api: List[Dict[str, Any]], 
        ums_tool_schemas: Dict[str, Dict[str, Any]]
    ) -> Optional[str]:
        """
        Detect if exactly one UMS tool is available for structured output.
        
        Returns the LLM-seen tool name if exactly one UMS tool is available,
        None otherwise (to avoid schema conflicts with multiple UMS tools).
        """
        ums_tools_found = []
        
        for tool_def in formatted_tools_for_api:
            tool_name = None
            if isinstance(tool_def, dict):
                if tool_def.get("type") == "function" and isinstance(tool_def.get("function"), dict):
                    tool_name = tool_def["function"].get("name")
                else:
                    tool_name = tool_def.get("name")
            
            if tool_name and tool_name in ums_tool_schemas:
                ums_tools_found.append(tool_name)
        
        # Only apply structured output if exactly one UMS tool is available
        # This avoids schema conflicts and ensures the right schema is used
        if len(ums_tools_found) == 1:
            log.info(f"Single UMS tool detected for structured output: {ums_tools_found[0]}")
            return ums_tools_found[0]
        elif len(ums_tools_found) > 1:
            log.warning(f"Multiple UMS tools available ({ums_tools_found}), skipping structured output to avoid schema conflicts")
            return None
        else:
            return None

    def _apply_ums_structured_output(
        self, 
        completion_params: Dict[str, Any], 
        schema_for_tool: Dict[str, Any], 
        model_name: str, 
        provider_name: str
    ) -> None:
        """
        Apply structured output for UMS tools using OpenAI-compatible providers.
        """
        try:
            # Check if model supports json_schema format
            use_json_schema_format = False
            if provider_name == Provider.OPENAI.value:
                if any(prefix in model_name.lower() for prefix in MODELS_CONFIRMED_FOR_OPENAI_JSON_SCHEMA_FORMAT):
                    use_json_schema_format = True
            elif provider_name == Provider.MISTRAL.value:
                if model_name.lower() in MISTRAL_NATIVE_MODELS_SUPPORTING_SCHEMA:
                    use_json_schema_format = True
            
            if use_json_schema_format:
                completion_params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": f"ums_tool_structured_output",
                        "description": "Structured output for UMS tool call",
                        "strict": True,
                        "schema": schema_for_tool
                    }
                }
                log.info(f"[{provider_name}] Applied json_schema structured output for UMS tool on model {model_name}")
            elif any(prefix in model_name.lower() for prefix in MODELS_SUPPORTING_OPENAI_JSON_OBJECT_FORMAT) or \
                 provider_name in [Provider.DEEPSEEK.value, Provider.GEMINI.value, Provider.GROK.value, 
                                  Provider.GROQ.value, Provider.CEREBRAS.value, Provider.OPENROUTER.value]:
                completion_params["response_format"] = {"type": "json_object"}
                log.info(f"[{provider_name}] Applied json_object structured output for UMS tool on model {model_name}")
            else:
                log.warning(f"[{provider_name}] Model {model_name} does not support structured output for UMS tools")
        except Exception as e:
            log.error(f"[{provider_name}] Error applying UMS structured output: {e}", exc_info=False)
            # Continue without structured output rather than failing the request

    def _apply_ums_structured_output_anthropic(
        self, 
        stream_params: Dict[str, Any], 
        schema_for_tool: Dict[str, Any], 
        tool_name: str
    ) -> None:
        """
        Apply structured output for UMS tools using Anthropic by modifying the system prompt.
        """
        try:
            structured_instruction = f"\n\nVERY IMPORTANT: When calling the tool '{tool_name}', you MUST respond with valid JSON that exactly matches this schema:\n"
            structured_instruction += f"```json\n{json.dumps(schema_for_tool, indent=2)}\n```\n"
            structured_instruction += "Ensure all required fields are included and follow the exact structure shown above."
            
            if stream_params.get("system"):
                stream_params["system"] = stream_params["system"] + structured_instruction
            else:
                stream_params["system"] = structured_instruction
            
            log.info(f"[anthropic] Applied structured output instruction for UMS tool: {tool_name}")
        except Exception as e:
            log.error(f"[anthropic] Error applying UMS structured output: {e}", exc_info=False)
            # Continue without structured output rather than failing the request

    async def process_streaming_query(
        self,
        query: str,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        messages_override: Optional[InternalMessageList] = None,
        tools_override: Optional[List[Dict[str, Any]]] = None,
        ui_websocket_sender: Optional[Callable[[str, Any], Coroutine]] = None,
        force_structured_output: bool = False,
        force_tool_choice: Optional[str] = None,
        ums_tool_schemas: Optional[Dict[str, Dict[str, Any]]] = None,  # NEW: For UMS structured outputs
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        span: Optional[trace.Span] = None
        span_context_manager = None
        current_task = asyncio.current_task()
        overall_query_stop_reason: Optional[str] = "processing"
        overall_query_error_occurred = False
        retrying_without_tools = False
        is_agent_llm_turn = messages_override is not None

        if not hasattr(self, "session_input_tokens") or is_agent_llm_turn:
            self.session_input_tokens = 0
            self.session_output_tokens = 0
            self.session_total_cost = 0.0
            self.cache_hit_count = 0
            self.tokens_saved_by_cache = 0

        cache_hits_this_query_for_tools: int = 0
        tokens_saved_by_tool_cache_this_query: int = 0
        tool_execution_details_for_client_log: List[Dict] = []

        with safe_stdout():
            start_time_overall_query = time.time()
            internal_model_name_for_cost_lookup = model or self.current_model
            if not max_tokens:
                max_tokens = self.config.default_max_tokens

            provider_name = self.get_provider_from_model(internal_model_name_for_cost_lookup)

            if not provider_name:
                error_msg = f"No provider determined for model '{internal_model_name_for_cost_lookup}'. Cannot process."
                log.error(error_msg)
                yield "error", error_msg
                return

            provider_client = getattr(self, f"{provider_name}_client", None)
            if not provider_client and provider_name == Provider.ANTHROPIC.value:
                provider_client = self.anthropic
            if not provider_client:
                error_msg = f"API client for provider '{provider_name}' not initialized for model '{internal_model_name_for_cost_lookup}'."
                log.error(error_msg)
                yield "error", error_msg
                return

            actual_model_name_string_for_api = internal_model_name_for_cost_lookup
            providers_requiring_prefix_strip = [Provider.OPENROUTER.value, Provider.GROQ.value, Provider.CEREBRAS.value]
            if provider_name in providers_requiring_prefix_strip:
                prefix_to_strip = f"{provider_name}/"
                if internal_model_name_for_cost_lookup.lower().startswith(prefix_to_strip):
                    actual_model_name_string_for_api = internal_model_name_for_cost_lookup[len(prefix_to_strip) :]
            log.info(
                f"MCPC Streaming Query: Provider='{provider_name}', Model (Display/Cost)='{internal_model_name_for_cost_lookup}', Model (API)='{actual_model_name_string_for_api}'"
            )

            if tracer:
                try:
                    span_attributes = {
                        "llm.model_name": internal_model_name_for_cost_lookup,
                        "llm.provider": provider_name,
                        "query_length": len(query),
                        "streaming": True,
                        "is_agent_turn": is_agent_llm_turn,
                    }
                    span_context_manager = tracer.start_as_current_span("mcpclient.process_streaming_query", attributes=span_attributes)
                    if span_context_manager:
                        span = span_context_manager.__enter__()
                except Exception as e:
                    log.warning(f"Failed to start trace span: {e}")

            messages_to_use_for_api: InternalMessageList
            if messages_override is not None:
                messages_to_use_for_api = messages_override
            else:
                await self.auto_prune_context()
                messages_to_use_for_api = self.conversation_graph.current_node.messages.copy()
                if query:
                    user_message: InternalMessage = {"role": "user", "content": query}
                    messages_to_use_for_api.append(user_message)

            if span:
                try:
                    span.set_attribute("conversation_length_initial", len(messages_to_use_for_api))
                except Exception as e:
                    log.warning(f"Span attr error for conversation_length_initial: {e}")

            final_response_text_for_chat_history_log: str = ""
            servers_used_this_query_session: Set[str] = set()
            tools_used_this_query_session: List[str] = []

        try:
            turn_count = 0
            while True:
                turn_count += 1
                log.debug(f"[{provider_name}] Starting LLM Interaction Turn {turn_count} for query.")

                if current_task and current_task.cancelled():
                    log.info(f"[{provider_name}] Query processing cancelled by client before API call in turn {turn_count}.")
                    raise asyncio.CancelledError(f"Query cancelled before API turn {turn_count}")

                accumulated_text_this_llm_turn: str = ""
                llm_requested_tool_calls_this_turn: List[Dict[str, Any]] = []
                current_llm_turn_input_tokens = 0
                current_llm_turn_output_tokens = 0
                current_llm_turn_stop_reason: Optional[str] = "unknown_llm_stop_reason"
                current_llm_turn_api_error: Optional[str] = None

                messages_to_send_to_llm_api = self._filter_faulty_client_tool_results(messages_to_use_for_api)
                if span:
                    try:
                        span.set_attribute(f"llm_turn_{turn_count}.messages_sent_count", len(messages_to_send_to_llm_api))
                    except Exception as e:
                        log.warning(f"Span attr error for messages_sent_count: {e}")

                formatted_messages_for_api, system_prompt_for_api = self._format_messages_for_provider(
                    messages_to_send_to_llm_api, provider_name, internal_model_name_for_cost_lookup
                )

                formatted_tools_for_api: Optional[List[Dict[str, Any]]]
                if tools_override is not None and is_agent_llm_turn:
                    formatted_tools_for_api = tools_override
                elif retrying_without_tools:
                    formatted_tools_for_api = None
                else:
                    formatted_tools_for_api = self._format_tools_for_provider(provider_name)

                if provider_name == Provider.DEEPSEEK.value and "deepseek-reasoner" in internal_model_name_for_cost_lookup:
                    if formatted_tools_for_api is not None:
                        log.warning(
                            f"Model '{internal_model_name_for_cost_lookup}' (DeepSeek Reasoner) does not support tools. Sending request without tools."
                        )
                    formatted_tools_for_api = None

                log.debug(
                    f"[{provider_name}] LLM Turn {turn_count} START: Messages to LLM API={len(formatted_messages_for_api)}, Tools to LLM API={len(formatted_tools_for_api) if formatted_tools_for_api else 'None'}"
                )

                logger_for_llm_prompt = getattr(self, "logger", log)
                prompt_origin_log_tag = (
                    f"AGENT LLM TURN {getattr(self.agent_loop_instance.state, 'current_loop', 'N/A') if self.agent_loop_instance and is_agent_llm_turn else 'N/A'}"
                    if is_agent_llm_turn
                    else f"USER QUERY TURN {turn_count}"
                )
                logger_for_llm_prompt.info(
                    f"MCPC {prompt_origin_log_tag}: Sending to LLM Provider '{provider_name}', Model '{actual_model_name_string_for_api}'."
                )
                if system_prompt_for_api:
                    logger_for_llm_prompt.debug(f"MCPC {prompt_origin_log_tag}: SYSTEM PROMPT FOR LLM:\n----\n{system_prompt_for_api}\n----")
                try:
                    messages_json_log = json.dumps(formatted_messages_for_api, indent=2, ensure_ascii=False)
                    logger_for_llm_prompt.debug(
                        f"MCPC {prompt_origin_log_tag}: MESSAGES FOR LLM (count {len(formatted_messages_for_api)}):\n----\n{messages_json_log}\n----"
                    )
                except Exception as json_log_err:
                    logger_for_llm_prompt.error(f"MCPC {prompt_origin_log_tag}: Could not serialize messages for LLM logging: {json_log_err}")
                if formatted_tools_for_api:
                    try:
                        tools_json_log = json.dumps(formatted_tools_for_api, indent=2, ensure_ascii=False)
                        logger_for_llm_prompt.debug(
                            f"MCPC {prompt_origin_log_tag}: TOOLS FOR LLM (count {len(formatted_tools_for_api)}):\n----\n{tools_json_log[:3000]}...\n----"
                        )
                    except Exception as json_log_err_tools:
                        logger_for_llm_prompt.error(f"MCPC {prompt_origin_log_tag}: Could not serialize tools for LLM logging: {json_log_err_tools}")
                else:
                    logger_for_llm_prompt.debug(f"MCPC {prompt_origin_log_tag}: NO TOOLS sent to LLM for this turn.")

                if not formatted_messages_for_api and provider_name != Provider.ANTHROPIC.value:
                    current_llm_turn_api_error = "Internal Error: No messages to send to LLM after formatting."
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}")
                    break
                elif not formatted_messages_for_api and provider_name == Provider.ANTHROPIC.value and not system_prompt_for_api:
                    current_llm_turn_api_error = "Internal Error (Anthropic): No messages or system prompt to send to LLM."
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}")
                    break

                stream_start_time_llm_api_call = time.time()
                
                try:
                    stream_iterator: Optional[AsyncGenerator[Tuple[str, Any], None]] = None
                    api_client_for_call: Any = provider_client

                    if provider_name == Provider.ANTHROPIC.value:
                        anthropic_sdk_client = cast("AsyncAnthropic", api_client_for_call)
                        stream_params_anthropic: Dict[str, Any] = {
                            "model": actual_model_name_string_for_api,
                            "messages": formatted_messages_for_api,
                            "max_tokens": max_tokens or self.config.default_max_tokens,
                            "temperature": temperature if temperature is not None else self.config.temperature,
                        }
                        if formatted_tools_for_api:
                            stream_params_anthropic["tools"] = formatted_tools_for_api
                        if system_prompt_for_api:
                            stream_params_anthropic["system"] = system_prompt_for_api

                        # Handle UMS tool structured outputs BEFORE generic structured output
                        if ums_tool_schemas and formatted_tools_for_api:
                            ums_tool_being_called = self._detect_ums_tool_in_request(formatted_tools_for_api, ums_tool_schemas)
                            if ums_tool_being_called:
                                schema_for_ums_tool = ums_tool_schemas[ums_tool_being_called]
                                self._apply_ums_structured_output_anthropic(
                                    stream_params_anthropic, schema_for_ums_tool, ums_tool_being_called
                                )
                                log.info(f"[{provider_name}] Applied structured output for UMS tool: {ums_tool_being_called}")

                        # Add structured output forcing for Anthropic
                        if force_structured_output:
                            # For Anthropic, we use the system prompt to enforce JSON response with specific format
                            plan_example = [
                                {
                                    "id": "step-1",
                                    "description": "Research the impact of exercise on mental health using scientific studies",
                                    "status": "planned",
                                    "depends_on": [],
                                },
                                {
                                    "id": "step-2",
                                    "description": "Analyze and summarize key findings about exercise benefits",
                                    "status": "planned",
                                    "depends_on": [],
                                },
                            ]

                            json_instruction = "\nVERY IMPORTANT: You MUST respond with valid JSON in EXACTLY this format (with your plan steps):\n"
                            json_instruction += (
                                f'```json\n{{"tool": "agent_update_plan", "tool_input": {{"plan": {json.dumps(plan_example, indent=2)}}}}}```\n'
                            )
                            json_instruction += (
                                "\nDo not include any text outside the JSON structure. Make sure to include all required fields for each step."
                            )

                            if force_tool_choice:
                                json_instruction += f"\nMust use the tool named: {force_tool_choice}"

                            if system_prompt_for_api:
                                stream_params_anthropic["system"] = system_prompt_for_api + json_instruction
                            else:
                                stream_params_anthropic["system"] = json_instruction
                            log.info(f"[{provider_name}] Forcing JSON response format via system prompt with detailed example")

                        # Add tool choice for Anthropic
                        if force_tool_choice and formatted_tools_for_api:
                            stream_params_anthropic["tool_choice"] = {"type": "tool", "name": force_tool_choice}
                            log.info(f"[{provider_name}] Forcing tool choice: {force_tool_choice}")

                        anthropic_stream_manager = anthropic_sdk_client.messages.stream(**stream_params_anthropic)
                        async with anthropic_stream_manager as api_stream_obj_anthropic:
                            stream_iterator = self._handle_anthropic_stream(api_stream_obj_anthropic)

                    elif provider_name in [
                        Provider.OPENAI.value,
                        Provider.GROK.value,
                        Provider.DEEPSEEK.value,
                        Provider.GROQ.value,
                        Provider.MISTRAL.value,
                        Provider.CEREBRAS.value,
                        Provider.GEMINI.value,
                        Provider.OPENROUTER.value,
                    ]:
                        openai_sdk_client = cast("AsyncOpenAI", api_client_for_call)
                        completion_params_openai: Dict[str, Any] = {
                            "model": actual_model_name_string_for_api,
                            "messages": formatted_messages_for_api,
                            "max_tokens": max_tokens or self.config.default_max_tokens,
                            "temperature": temperature if temperature is not None else self.config.temperature,
                            "stream": True,
                        }  # type: ignore

                        if formatted_tools_for_api:
                            completion_params_openai["tools"] = formatted_tools_for_api

                        # Handle UMS tool structured outputs BEFORE generic structured output
                        if ums_tool_schemas and formatted_tools_for_api:
                            ums_tool_being_called = self._detect_ums_tool_in_request(formatted_tools_for_api, ums_tool_schemas)
                            if ums_tool_being_called:
                                schema_for_ums_tool = ums_tool_schemas[ums_tool_being_called]
                                self._apply_ums_structured_output(
                                    completion_params_openai, schema_for_ums_tool, actual_model_name_string_for_api, provider_name
                                )
                                log.info(f"[{provider_name}] Applied structured output for UMS tool: {ums_tool_being_called}")

                        if force_structured_output:
                            llm_seen_agent_update_plan_name = AGENT_TOOL_UPDATE_PLAN 
                            if formatted_tools_for_api:
                                for tool_schema_item in formatted_tools_for_api:
                                    if isinstance(tool_schema_item, dict) and tool_schema_item.get("type") == "function":
                                        func_details = tool_schema_item.get("function", {})
                                        if isinstance(func_details, dict) and \
                                        self.server_manager.sanitized_to_original.get(func_details.get("name")) == AGENT_TOOL_UPDATE_PLAN:
                                            llm_seen_agent_update_plan_name = func_details.get("name")
                                            break
                            
                            # This schema describes the desired direct JSON output from the LLM when it's asked to provide a plan.
                            # It should match the structure of a tool call to agent:update_plan.
                            agent_plan_direct_output_schema = {
                                "type": "object",
                                "properties": {
                                    "tool": {"type": "string", "const": llm_seen_agent_update_plan_name},
                                    "tool_input": AGENT_UPDATE_PLAN_ARGUMENT_SCHEMA 
                                },
                                "required": ["tool", "tool_input"],
                                "additionalProperties": False
                            }
                            
                            # Check if the current MODEL (actual_model_name_string_for_api) supports "json_schema" type.
                            # actual_model_name_string_for_api is the name sent to the API (e.g. "gpt-4o", "mistral-large-latest" if using Mistral native)
                            
                            use_json_schema_format = False
                            if provider_name == Provider.OPENAI.value:
                                if any(prefix in actual_model_name_string_for_api.lower() for prefix in MODELS_CONFIRMED_FOR_OPENAI_JSON_SCHEMA_FORMAT):
                                    use_json_schema_format = True
                            elif provider_name == Provider.MISTRAL.value: # Assuming native Mistral API if provider is Mistral
                                if actual_model_name_string_for_api.lower() in MISTRAL_NATIVE_MODELS_SUPPORTING_SCHEMA:
                                    use_json_schema_format = True
                            # Add elif for other providers if they explicitly support json_schema type in response_format

                            if use_json_schema_format:
                                completion_params_openai["response_format"] = {
                                    "type": "json_schema",
                                    "json_schema": { # This is the new OpenAI API structure for response_format json_schema
                                        "name": "agent_plan_update_direct_output", 
                                        "description": "Strict schema for the agent's plan update, structured as a tool call to agent:update_plan.",
                                        "strict": False, 
                                        "schema": agent_plan_direct_output_schema
                                    }
                                }
                                log.info(f"[{provider_name}] Forcing response_format={{'type': 'json_schema', strict:true}} with agent_plan_tool_call_schema for model {actual_model_name_string_for_api}.")
                            elif any(prefix in actual_model_name_string_for_api.lower() for prefix in MODELS_SUPPORTING_OPENAI_JSON_OBJECT_FORMAT) or \
                                provider_name in [Provider.DEEPSEEK.value, Provider.GEMINI.value, Provider.GROK.value, Provider.GROQ.value, Provider.CEREBRAS.value, Provider.OPENROUTER.value]:
                                # Fallback to json_object for broader compatibility if json_schema not confirmed
                                completion_params_openai["response_format"] = {"type": "json_object"}
                                log.info(f"[{provider_name}] Forcing response_format={{'type': 'json_object'}} for model {actual_model_name_string_for_api}.")
                            else:
                                log.warning(f"[{provider_name}] Model {actual_model_name_string_for_api} not explicitly listed for json_schema or json_object support when force_structured_output=True. Not setting response_format. Relaying on prompt.")
                        
                        if force_tool_choice and formatted_tools_for_api:
                            # Ensure force_tool_choice is the LLM-seen name for agent:update_plan if that's the intent
                            if force_structured_output: # If forcing structure, assume it's for the plan tool
                                llm_seen_agent_update_plan_name_for_choice = AGENT_TOOL_UPDATE_PLAN
                                for tool_schema_item_choice in formatted_tools_for_api:
                                    if isinstance(tool_schema_item_choice, dict) and tool_schema_item_choice.get("type") == "function":
                                        func_details_choice = tool_schema_item_choice.get("function", {})
                                        if isinstance(func_details_choice, dict) and \
                                            self.server_manager.sanitized_to_original.get(func_details_choice.get("name")) == AGENT_TOOL_UPDATE_PLAN:
                                            llm_seen_agent_update_plan_name_for_choice = func_details_choice.get("name")
                                            break
                                # Only set tool_choice to agent_update_plan if that was the original intent of force_tool_choice
                                if force_tool_choice == llm_seen_agent_update_plan_name_for_choice:
                                    completion_params_openai["tool_choice"] = {"type": "function", "function": {"name": force_tool_choice}}
                                    log.info(f"[{provider_name}] Forcing OpenAI-compatible tool choice: {force_tool_choice} (for agent:update_plan)")
                                elif force_tool_choice: # If some other tool was forced by the caller
                                    completion_params_openai["tool_choice"] = {"type": "function", "function": {"name": force_tool_choice}}
                                    log.info(f"[{provider_name}] Forcing OpenAI-compatible tool choice (non-plan): {force_tool_choice}")
                            elif force_tool_choice: # force_tool_choice set, but not necessarily for structured plan output
                                completion_params_openai["tool_choice"] = {"type": "function", "function": {"name": force_tool_choice}}
                                log.info(f"[{provider_name}] Forcing OpenAI-compatible tool choice: {force_tool_choice}")
                            
                        providers_not_supporting_stream_options = {Provider.MISTRAL.value, Provider.CEREBRAS.value}
                        if provider_name not in providers_not_supporting_stream_options:
                            completion_params_openai["stream_options"] = {"include_usage": True}
                        api_stream_obj_openai = await openai_sdk_client.chat.completions.create(**completion_params_openai)
                        stream_iterator = self._handle_openai_compatible_stream(api_stream_obj_openai, provider_name)
                    else:
                        current_llm_turn_api_error = f"Internal Error: Streaming not implemented for provider '{provider_name}'."
                        log.critical(current_llm_turn_api_error)
                        break

                    if stream_iterator:
                        async for std_event_type, std_event_data in stream_iterator:
                            if current_task and current_task.cancelled():
                                log.info(f"[{provider_name}] Query cancelled by client during LLM stream consumption in turn {turn_count}.")
                                raise asyncio.CancelledError("Query cancelled during LLM stream consumption")
                            yield std_event_type, std_event_data  # Pass through all standardized events
                            if std_event_type == "error":
                                current_llm_turn_api_error = str(std_event_data)
                                log.error(f"[{provider_name}] Error event from stream handler: {current_llm_turn_api_error}")
                                break
                            elif std_event_type == "text_chunk":
                                accumulated_text_this_llm_turn += str(std_event_data)
                            elif std_event_type == "tool_call_end":  # LLM requested a tool
                                if isinstance(std_event_data, dict) and std_event_data.get("id") and std_event_data.get("name"):
                                    llm_requested_tool_calls_this_turn.append(
                                        {"id": std_event_data["id"], "name": std_event_data["name"], "input": std_event_data.get("parsed_input", {})}
                                    )
                                else:
                                    log.warning(f"[{provider_name}] Received malformed 'tool_call_end' event: {std_event_data}")
                            elif std_event_type == "final_usage":
                                current_llm_turn_input_tokens = std_event_data.get("input_tokens", 0)
                                current_llm_turn_output_tokens = std_event_data.get("output_tokens", 0)
                                self.session_input_tokens += current_llm_turn_input_tokens
                                self.session_output_tokens += current_llm_turn_output_tokens
                                turn_cost = self._calculate_and_log_cost(
                                    internal_model_name_for_cost_lookup, current_llm_turn_input_tokens, current_llm_turn_output_tokens
                                )
                                self.session_total_cost += turn_cost
                            elif std_event_type == "stop_reason":
                                current_llm_turn_stop_reason = str(std_event_data)
                    else:
                        if not current_llm_turn_api_error:
                            current_llm_turn_api_error = f"Internal Error: Stream iterator was not properly created for provider {provider_name}."
                        log.error(f"[{provider_name}] {current_llm_turn_api_error}")

                except OpenAIBadRequestError as e_br_openai:
                    error_detail_br_openai = str(e_br_openai)
                    is_tool_error_br_openai = (
                        "tool" in error_detail_br_openai.lower()
                        or "function" in error_detail_br_openai.lower()
                        or "parameter" in error_detail_br_openai.lower()
                    )
                    failing_tool_name_br_openai: Optional[str] = None
                    if (
                        provider_name == Provider.GEMINI.value
                        and "Failed to parse parameters of tool declaration with name" in error_detail_br_openai
                    ):
                        match_br_openai = re.search(r"with name (\S+)", error_detail_br_openai)
                        if match_br_openai:
                            failing_tool_name_br_openai = self.server_manager.sanitized_to_original.get(
                                match_br_openai.group(1), match_br_openai.group(1)
                            )
                    if is_tool_error_br_openai and not retrying_without_tools and formatted_tools_for_api:
                        log.warning(f"[{provider_name}] Caught tool-related BadRequestError: {e_br_openai}")
                        tools_to_blacklist_now_br_openai = []
                        if failing_tool_name_br_openai:
                            tools_to_blacklist_now_br_openai.append(failing_tool_name_br_openai)
                        else:
                            tools_to_blacklist_now_br_openai = [
                                self.server_manager.sanitized_to_original.get(tf["function"]["name"], tf["function"]["name"])
                                for tf in formatted_tools_for_api
                                if tf.get("type") == "function" and isinstance(tf.get("function"), dict) and tf["function"].get("name")
                            ]
                        if tools_to_blacklist_now_br_openai:
                            log.warning(f"[{provider_name}] Adding to blacklist and retrying LLM call without: {tools_to_blacklist_now_br_openai}")
                            self.tool_blacklist_manager.add_to_blacklist(provider_name, list(set(tools_to_blacklist_now_br_openai)))
                            retrying_without_tools = True
                            yield "status", f"[yellow]Tool schema error with {provider_name}. Retrying without affected tool(s)...[/]"
                            await asyncio.sleep(0.1)
                            continue
                        else:
                            current_llm_turn_api_error = (
                                f"BadRequestError (Tool related, but no specific tool identified to blacklist): {e_br_openai}"
                            )
                        log.error(f"[{provider_name}] {current_llm_turn_api_error}", exc_info=False)
                    else:
                        current_llm_turn_api_error = f"BadRequestError: {e_br_openai}"
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}", exc_info=True)
                except (anthropic.APIConnectionError, OpenAIAPIConnectionError, httpx.RequestError) as e_conn:
                    current_llm_turn_api_error = f"Connection Error: {e_conn}"
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}", exc_info=True)
                except (anthropic.AuthenticationError, OpenAIAuthenticationError) as e_auth:
                    current_llm_turn_api_error = f"Authentication Error (Check API Key for {provider_name}): {e_auth}"
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}")
                except (anthropic.PermissionDeniedError, OpenAIPermissionDeniedError) as e_perm:
                    current_llm_turn_api_error = f"Permission Denied by {provider_name}: {e_perm}"
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}")
                except (anthropic.NotFoundError, OpenAINotFoundError) as e_nf:
                    current_llm_turn_api_error = f"API Endpoint or Model Not Found for {provider_name}: {e_nf}"
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}")
                except (anthropic.RateLimitError, OpenAIRateLimitError) as e_rl:
                    current_llm_turn_api_error = f"Rate Limit Exceeded for {provider_name}: {e_rl}"
                    log.warning(f"[{provider_name}] {current_llm_turn_api_error}")
                except (anthropic.APIStatusError, OpenAIAPIStatusError) as e_stat:
                    current_llm_turn_api_error = f"API Status Error ({e_stat.status_code}) from {provider_name}: {getattr(e_stat, 'message', e_stat)}"
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}", exc_info=True)
                except (anthropic.APIError, OpenAIAPIError) as e_api:
                    current_llm_turn_api_error = f"Generic API Error from {provider_name}: {e_api}"
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}", exc_info=True)
                except asyncio.CancelledError:
                    log.debug(f"[{provider_name}] API call/stream consumption cancelled during turn {turn_count}.")
                    raise
                except NotImplementedError as e_ni:
                    current_llm_turn_api_error = str(e_ni)
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}")
                except Exception as api_err_outer:
                    current_llm_turn_api_error = f"Unexpected API/Stream Error for {provider_name}: {api_err_outer}"
                    log.error(f"[{provider_name}] {current_llm_turn_api_error}", exc_info=True)
                finally:
                    log.debug(
                        f"[{provider_name}] LLM API Call for Turn {turn_count} finished. Duration: {(time.time() - stream_start_time_llm_api_call):.2f}s. API Error this turn: {current_llm_turn_api_error}. LLM Stop Reason this turn: {current_llm_turn_stop_reason}"
                    )

                if current_llm_turn_api_error:
                    overall_query_error_occurred = True
                    overall_query_stop_reason = "error"
                    log.error(
                        f"[{provider_name}] Exiting interaction loop for this query due to API error in LLM turn {turn_count}: {current_llm_turn_api_error}"
                    )
                    yield "error", current_llm_turn_api_error
                    break

                retrying_without_tools = False
                assistant_response_blocks_for_history_this_turn: List[Union[TextContentBlock, ToolUseContentBlock]] = []
                if accumulated_text_this_llm_turn:
                    assistant_response_blocks_for_history_this_turn.append({"type": "text", "text": accumulated_text_this_llm_turn})
                    final_response_text_for_chat_history_log += accumulated_text_this_llm_turn
                for llm_tool_call_obj in llm_requested_tool_calls_this_turn:  # LLM requested these tools
                    assistant_response_blocks_for_history_this_turn.append(
                        {
                            "type": "tool_use",
                            "id": llm_tool_call_obj.get("id", f"missing_id_{str(uuid.uuid4())[:4]}"),
                            "name": llm_tool_call_obj.get("name", "unknown_tool_name"),
                            "input": llm_tool_call_obj.get("input", {}),
                        }
                    )
                if assistant_response_blocks_for_history_this_turn:
                    messages_to_use_for_api.append({"role": "assistant", "content": assistant_response_blocks_for_history_this_turn})

                overall_query_stop_reason = current_llm_turn_stop_reason
                agent_update_plan_requested_this_turn = False
                if is_agent_llm_turn and llm_requested_tool_calls_this_turn:
                    if any(tc.get("name") == AGENT_TOOL_UPDATE_PLAN for tc in llm_requested_tool_calls_this_turn):
                        agent_update_plan_requested_this_turn = True
                if agent_update_plan_requested_this_turn and overall_query_stop_reason == "tool_use":
                    log.info(f"[{provider_name}] LLM requested '{AGENT_TOOL_UPDATE_PLAN}'. Overriding stop reason to 'agent_internal_tool_request'.")
                    overall_query_stop_reason = "agent_internal_tool_request"
                elif agent_update_plan_requested_this_turn and overall_query_stop_reason != "agent_internal_tool_request":
                    log.info(
                        f"[{provider_name}] LLM requested '{AGENT_TOOL_UPDATE_PLAN}'. Ensuring stop reason is 'agent_internal_tool_request' (was '{overall_query_stop_reason}')."
                    )
                    overall_query_stop_reason = "agent_internal_tool_request"

                if overall_query_stop_reason == "tool_use" or (is_agent_llm_turn and overall_query_stop_reason == "agent_internal_tool_request"):
                    if is_agent_llm_turn and overall_query_stop_reason == "agent_internal_tool_request":
                        if not llm_requested_tool_calls_this_turn:
                            log.warning(f"[{provider_name}] Stop reason 'agent_internal_tool_request' but no tool_calls parsed. Breaking.")
                            yield "status", "[yellow]Warning: Model indicated agent tool use, but no tools identified.[/]"
                            overall_query_stop_reason = "error_no_agent_tool_parsed"
                            overall_query_error_occurred = True
                            break
                        log.info(f"[{provider_name}] LLM requested agent-internal tool(s). Breaking to pass to agent formulation.")
                        break
                    if not llm_requested_tool_calls_this_turn and overall_query_stop_reason == "tool_use":
                        log.warning(f"[{provider_name}] LLM stop reason 'tool_use' but no tool requests parsed. Breaking.")
                        yield "status", "[yellow]Warning: Model indicated tool use, but no tools identified.[/]"
                        overall_query_stop_reason = "error_no_tool_parsed"
                        overall_query_error_occurred = True
                        break

                    tool_results_for_next_llm_api_turn: List[InternalMessage] = []
                    yield "status", f"{EMOJI_MAP['tool']} MCPClient processing {len(llm_requested_tool_calls_this_turn)} UMS/server tool call(s)..."
                    all_tool_executions_successful_this_round = True

                    for tool_call_request_obj_from_llm in llm_requested_tool_calls_this_turn:
                        if current_task and current_task.cancelled():
                            log.info(f"[{provider_name}] Query cancelled during tool execution phase in turn {turn_count}.")
                            raise asyncio.CancelledError("Cancelled during tool processing phase")

                        original_mcp_tool_name_to_execute = tool_call_request_obj_from_llm["name"]  # This is original_mcp_name
                        tool_args_for_execution = tool_call_request_obj_from_llm["input"]
                        tool_use_id_from_llm_for_result = tool_call_request_obj_from_llm["id"]
                        tool_execution_output_content: Dict[str, Any] = {"error": "MCPClient: Tool exec failed by default.", "success": False}
                        tool_execution_is_error_flag = True
                        is_agent_internal_tool_call_for_aml = original_mcp_tool_name_to_execute == AGENT_TOOL_UPDATE_PLAN

                        if not is_agent_internal_tool_call_for_aml:
                            yield "status", f"MCPClient executing server tool: {original_mcp_tool_name_to_execute}..."
                            mcp_tool_object_from_manager = self.server_manager.tools.get(original_mcp_tool_name_to_execute)
                            if not mcp_tool_object_from_manager:
                                tool_execution_output_content = {
                                    "error": f"Tool '{original_mcp_tool_name_to_execute}' not found by MCPClient.",
                                    "success": False,
                                }
                                tool_execution_is_error_flag = True
                                log.error(f"MCPClient: Tool '{original_mcp_tool_name_to_execute}' requested by LLM not found.")
                            else:
                                server_name_for_tool_exec = mcp_tool_object_from_manager.server_name
                                servers_used_this_query_session.add(server_name_for_tool_exec)
                                tools_used_this_query_session.append(original_mcp_tool_name_to_execute)
                                cached_tool_result = None
                                if self.tool_cache and self.config.enable_caching:
                                    try:
                                        cached_tool_result = self.tool_cache.get(original_mcp_tool_name_to_execute, tool_args_for_execution)
                                    except TypeError:
                                        log.warning(f"Cache key gen failed for {original_mcp_tool_name_to_execute}, skipping cache.")
                                is_cached_result_an_error = isinstance(cached_tool_result, dict) and (
                                    cached_tool_result.get("error") or cached_tool_result.get("success") is False
                                )

                                if cached_tool_result is not None and not is_cached_result_an_error:
                                    tool_execution_output_content = cached_tool_result
                                    tool_execution_is_error_flag = False
                                    cache_hits_this_query_for_tools += 1
                                    content_str_cache = self._stringify_content(cached_tool_result)
                                    tokens_saved_by_this_tool_cache_hit = self._estimate_string_tokens(content_str_cache)
                                    tokens_saved_by_tool_cache_this_query += tokens_saved_by_this_tool_cache_hit
                                    yield (
                                        "status",
                                        f"{EMOJI_MAP['cached']} MCPClient: Cache for {original_mcp_tool_name_to_execute} ({tokens_saved_by_this_tool_cache_hit:,} est. tokens saved)",
                                    )
                                else:
                                    if cached_tool_result is not None and is_cached_result_an_error:
                                        log.info(f"MCPClient: Ignoring cached error for {original_mcp_tool_name_to_execute}.")
                                    try:
                                        mcp_call_tool_result_object: CallToolResult = await self.execute_tool(
                                            server_name_for_tool_exec, original_mcp_tool_name_to_execute, tool_args_for_execution
                                        )
                                        raw_content_from_mcp_call = mcp_call_tool_result_object.content
                                        log.debug(
                                            f"MCPC: Raw content from CallToolResult for '{original_mcp_tool_name_to_execute}': Type {type(raw_content_from_mcp_call)}, Preview: {str(raw_content_from_mcp_call)[:200]}"
                                        )

                                        # Use the integrated robust_parse_mcp_tool_content
                                        parsed_ums_payload_dict = robust_parse_mcp_tool_content(raw_content_from_mcp_call)
                                        tool_execution_output_content = parsed_ums_payload_dict  # This is the UMS dict or error dict

                                        # Determine error flag based on parsed_ums_payload_dict and mcp_call_tool_result_object.isError
                                        if mcp_call_tool_result_object.isError:
                                            tool_execution_is_error_flag = True
                                            if isinstance(tool_execution_output_content, dict) and tool_execution_output_content.get("success", True):
                                                tool_execution_output_content.setdefault(
                                                    "error", f"Tool '{original_mcp_tool_name_to_execute}' reported error by MCP layer."
                                                )
                                                tool_execution_output_content["success"] = False
                                        elif (
                                            isinstance(tool_execution_output_content, dict) and tool_execution_output_content.get("success") is False
                                        ):
                                            tool_execution_is_error_flag = True
                                        elif not isinstance(tool_execution_output_content, dict):
                                            tool_execution_is_error_flag = True  # Parsing failed
                                        else:
                                            tool_execution_is_error_flag = False  # Assume success

                                        log.debug(
                                            f"MCPC: Parsed UMS payload for '{original_mcp_tool_name_to_execute}': Success={not tool_execution_is_error_flag}, Content Keys: {list(tool_execution_output_content.keys()) if isinstance(tool_execution_output_content, dict) else 'N/A'}"
                                        )

                                        if not tool_execution_is_error_flag and self.tool_cache and self.config.enable_caching:
                                            try:
                                                self.tool_cache.set(
                                                    original_mcp_tool_name_to_execute, tool_args_for_execution, tool_execution_output_content
                                                )
                                            except TypeError:
                                                log.warning(f"Cache set failed for {original_mcp_tool_name_to_execute}: unhashable.")
                                    except Exception as tool_exec_exception_mcpc:
                                        tool_execution_output_content = {
                                            "success": False,
                                            "error": f"MCPC exception during tool exec/parse for '{original_mcp_tool_name_to_execute}': {str(tool_exec_exception_mcpc)}",
                                        }
                                        tool_execution_is_error_flag = True
                                        log.error(
                                            f"MCPC exception executing/parsing '{original_mcp_tool_name_to_execute}': {tool_exec_exception_mcpc}",
                                            exc_info=True,
                                        )
                            # Yield "mcp_tool_executed_for_agent" ONLY for tools MCPC executed.
                            if is_agent_llm_turn:  # Still only yield this if it's an agent turn
                                yield (
                                    "mcp_tool_executed_for_agent",
                                    {
                                        "id": tool_use_id_from_llm_for_result,
                                        "name": original_mcp_tool_name_to_execute,
                                        "input": tool_args_for_execution,
                                        "result": tool_execution_output_content,  # The result from self.execute_tool
                                    },
                                )
                            else:
                                # For agent-internal tools like AGENT_TOOL_UPDATE_PLAN,
                                # MCPClient does *not* execute them here.
                                # It also should *not* yield "mcp_tool_executed_for_agent".
                                # The "tool_call_end" event yielded by the stream handler
                                # (e.g., _handle_openai_compatible_stream) is what
                                # process_agent_llm_turn will use to populate its
                                # llm_requested_tool_calls list for these agent-internal tools.
                                log.debug(
                                    f"MCPC Query Stream: LLM requested agent-internal tool '{original_mcp_tool_name_to_execute}'. It will be handled by AML based on 'tool_call_end' event."
                                )
                                # No yield "mcp_tool_executed_for_agent" for this case.
                                # tool_execution_is_error_flag remains True by default for agent internal tools *here*,
                                # but this won't matter as we don't yield its "execution result" from MCPC.
                                # The actual success/failure will be determined by AML when it executes it.
                                pass

                            if tool_execution_is_error_flag and not is_agent_internal_tool_call_for_aml:  # Only if MCPC executed it and it failed
                                all_tool_executions_successful_this_round = False

                        # Add tool result to history for the *next* LLM turn
                        # ONLY IF it's NOT an agent-internal tool.
                        if not is_agent_internal_tool_call_for_aml:
                            tool_result_block_for_llm_api: ToolResultContentBlock = {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id_from_llm_for_result,
                                "content": tool_execution_output_content,  # This is the UMS payload dict
                                "is_error": tool_execution_is_error_flag,
                                "_is_tool_result": True,
                            }
                            tool_result_message_for_llm_api: InternalMessage = {"role": "user", "content": [tool_result_block_for_llm_api]}
                            tool_results_for_next_llm_api_turn.append(tool_result_message_for_llm_api)

                        tool_execution_details_for_client_log.append(
                            {
                                "tool_name": original_mcp_tool_name_to_execute,
                                "tool_use_id": tool_use_id_from_llm_for_result,
                                "content": tool_execution_output_content
                                if not is_agent_internal_tool_call_for_aml
                                else {"note": "Agent internal tool, not executed by MCPC."},
                                "is_error": tool_execution_is_error_flag
                                if not is_agent_internal_tool_call_for_aml
                                else False,  # Assume not error for agent internal here
                                "cache_used": False,  # Placeholder, add real logic if applicable
                            }
                        )
                    if not all_tool_executions_successful_this_round and is_agent_llm_turn:
                        yield "status", "[yellow]MCPClient: One or more UMS/server tools failed for agent's turn."
                    if tool_results_for_next_llm_api_turn:
                        messages_to_use_for_api.extend(tool_results_for_next_llm_api_turn)
                        log.info(
                            f"[{provider_name}] Added {len(tool_results_for_next_llm_api_turn)} UMS/Server tool results. Continuing LLM loop for turn {turn_count + 1}."
                        )
                        retrying_without_tools = False
                        continue
                    elif llm_requested_tool_calls_this_turn and not tool_results_for_next_llm_api_turn:  # Only agent-internal tools were requested
                        log.info(
                            f"[{provider_name}] LLM requested only agent-internal tools (e.g., {llm_requested_tool_calls_this_turn[0]['name']}). Breaking LLM loop."
                        )
                        overall_query_stop_reason = "agent_internal_tool_request"
                        break
                    else:  # No UMS/server tools were requested or executed by MCPC this turn, or no results to send back
                        log.debug(
                            f"[{provider_name}] No UMS/Server tool results to send. Breaking LLM loop. LLM Stop Reason: {overall_query_stop_reason}"
                        )
                        break

                elif overall_query_stop_reason == "error":
                    log.error(f"[{provider_name}] Exiting query loop due to API error in LLM turn {turn_count}.")
                    break
                else:  # "stop", "max_tokens", "end_turn", etc.
                    log.info(f"[{provider_name}] LLM interaction finished (turn {turn_count}). LLM Stop Reason: {overall_query_stop_reason}")
                    break

        except asyncio.CancelledError:
            log.info(f"[{provider_name}] Query processing was cancelled by client (asyncio.CancelledError).")
            overall_query_stop_reason = "cancelled"
            overall_query_error_occurred = True
            yield "status", "[yellow]Request cancelled by user.[/]"
        except Exception as e_outer_loop:
            error_msg_outer_loop = f"Unexpected error during MCPClient query processing loop: {str(e_outer_loop)}"
            log.error(error_msg_outer_loop, exc_info=True)
            if span:
                span.set_status(trace.StatusCode.ERROR, description=error_msg_outer_loop)
            if hasattr(span, "record_exception"):
                span.record_exception(e_outer_loop)  # type: ignore
            yield "error", f"Unexpected Error: {error_msg_outer_loop}"
            overall_query_stop_reason = "error_outer_loop"
            overall_query_error_occurred = True
        finally:
            if span:
                final_span_status_code = trace.StatusCode.ERROR if overall_query_error_occurred else trace.StatusCode.OK
                final_span_description = (
                    f"Query failed/cancelled: {overall_query_stop_reason}"
                    if overall_query_error_occurred
                    else f"Query finished: {overall_query_stop_reason or 'completed'}"
                )
                span.set_status(final_span_status_code, description=final_span_description)
                try:
                    span.set_attribute("final_llm_stop_reason_overall", overall_query_stop_reason or "unknown")
                    span.set_attribute("total_input_tokens_session_accumulated", self.session_input_tokens)
                    span.set_attribute("total_output_tokens_session_accumulated", self.session_output_tokens)
                    span.set_attribute("total_estimated_cost_session_accumulated", self.session_total_cost)
                    span.set_attribute("tool_cache_hits_this_query_total", cache_hits_this_query_for_tools)
                    span.set_attribute("tokens_saved_by_tool_cache_this_query_total", tokens_saved_by_tool_cache_this_query)
                except Exception as span_final_attr_err:
                    log.warning(f"Failed to set final span attributes: {span_final_attr_err}")
            if span_context_manager and hasattr(span_context_manager, "__exit__"):
                with suppress(Exception):
                    span_context_manager.__exit__(*sys.exc_info())

            if not is_agent_llm_turn and not overall_query_error_occurred and not (current_task and current_task.cancelled()):
                try:
                    self.conversation_graph.current_node.messages = messages_to_use_for_api
                    self.conversation_graph.current_node.model = internal_model_name_for_cost_lookup
                    await self.conversation_graph.save(str(self.conversation_graph_file))
                    end_time_for_chat_history_log = time.time()
                    latency_ms_for_chat_history_log = (end_time_for_chat_history_log - start_time_overall_query) * 1000
                    tokens_used_for_chat_history_log = self.session_input_tokens + self.session_output_tokens
                    timestamp_for_chat_history_log = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(self, "history") and hasattr(self.history, "add_async"):
                        history_entry_for_log = ChatHistory(
                            query=query,
                            response=final_response_text_for_chat_history_log,
                            model=internal_model_name_for_cost_lookup,
                            timestamp=timestamp_for_chat_history_log,
                            server_names=list(servers_used_this_query_session),
                            tools_used=tools_used_this_query_session,
                            conversation_id=self.conversation_graph.current_node.id,
                            latency_ms=latency_ms_for_chat_history_log,
                            tokens_used=tokens_used_for_chat_history_log,
                            streamed=True,
                            cached=(cache_hits_this_query_for_tools > 0),
                        )
                        await self.history.add_async(history_entry_for_log)
                except Exception as final_update_error:
                    log.error(f"Error during final graph/history update for user query: {final_update_error}", exc_info=True)
                    yield "error", f"Failed to save history for user query: {final_update_error}"

            final_usage_payload_to_yield_overall = {
                "input_tokens": self.session_input_tokens,
                "output_tokens": self.session_output_tokens,
                "total_cost": self.session_total_cost,
                "tool_cache_hits_this_query": cache_hits_this_query_for_tools,
                "tokens_saved_by_tool_cache_this_query": tokens_saved_by_tool_cache_this_query,
            }
            yield "final_usage", final_usage_payload_to_yield_overall
            yield "stop_reason", overall_query_stop_reason or "unknown_overall_stop"
        log.info(
            f"MCPC Streaming Query processing finished. Overall Stop Reason: {overall_query_stop_reason}. Overall Query Latency: {(time.time() - start_time_overall_query) * 1000:.0f}ms"
        )

    async def _stream_wrapper(self, stream_generator: AsyncGenerator[Any, None]) -> AsyncGenerator[Tuple[str, Any], None]:
        """Wraps the query generator to extract specific event types and final stats."""
        final_stats = {}
        try:
            async for chunk in stream_generator:
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    event_type, event_data = chunk
                    if event_type == "text_chunk":
                        yield "text", event_data
                    elif event_type == "status":
                        yield "status", event_data
                    elif event_type == "tool_call_start":
                        yield (
                            "status",
                            f"{EMOJI_MAP['tool']} Preparing tool: [bold]{event_data.get('name', 'unknown').split(':')[-1]}[/] (ID: {event_data.get('id', '?')[:8]})...",
                        )
                    elif event_type == "tool_call_input_chunk":
                        pass  # UI might want this, for now, pass
                    elif event_type == "tool_call_end":
                        tool_name = event_data.get("name", "unknown")
                        tool_short_name = tool_name.split(":")[-1]
                        yield "status", f"{EMOJI_MAP['tool']} Executing tool: [bold]{tool_short_name}[/]"
                    # START ADDED BLOCK
                    elif event_type == "mcp_tool_executed_for_agent":
                        # This event is for internal processing by process_agent_llm_turn.
                        # Pass it through directly.
                        yield "mcp_tool_executed_for_agent", event_data
                    # END ADDED BLOCK
                    elif event_type == "error":
                        log.error(f"Stream wrapper received error event: {event_data}")
                        yield "error", str(event_data)
                    elif event_type == "final_usage":
                        final_stats.update(event_data)
                        log.debug(f"Stream wrapper captured final usage: {event_data}")
                    elif event_type == "stop_reason":
                        final_stats["stop_reason"] = event_data
                        log.debug(f"Stream wrapper captured stop reason: {event_data}")
                    else:
                        # This is where the original warning came from
                        log.warning(f"Stream wrapper: Unhandled standardized event tuple type: {event_type}. Data: {str(event_data)[:100]}")
                        # Optionally, yield it as a generic event if some consumer might handle it
                        # yield event_type, event_data

                elif isinstance(chunk, str) and chunk.startswith("@@STATUS@@"):
                    yield "status", chunk[len("@@STATUS@@") :].strip()
                elif isinstance(chunk, str):
                    yield "text", chunk
                else:
                    log.warning(f"Unexpected chunk type from stream generator: {type(chunk)} Data: {repr(chunk)[:100]}")

        except asyncio.CancelledError:
            log.debug("Stream wrapper detected cancellation during iteration.")
            raise  # Propagate cancellation
        except Exception as e:
            log.error(f"Error in stream wrapper iteration: {e}", exc_info=True)
            yield "error", str(e)  # Yield error message
        finally:
            # Yield the combined stats at the very end
            if final_stats:
                yield "final_stats", final_stats  # Yield collected stats
            else:
                log.warning("Stream wrapper did not capture final stats/reason.")

    def _should_attempt_plan_parsing(self, full_text_response: str, llm_requested_tool_calls: List[Dict[str, Any]], 
                                     force_structured_output: bool, force_tool_choice: Optional[str], llm_stop_reason: str) -> bool:
        """Determine if we should attempt plan parsing based on various signals."""
        return (force_structured_output or llm_requested_tool_calls or 
                llm_stop_reason == "agent_internal_tool_request" or
                any(keyword in full_text_response.lower() for keyword in ["plan", "steps", "```json", "agent_update_plan"]) or
                full_text_response.lower().strip().startswith("goal achieved")) and full_text_response.strip()

    def _validate_plan_steps(self, steps) -> bool:
        """Validate that plan steps have the required structure."""
        return (isinstance(steps, list) and len(steps) > 0 and 
               all(isinstance(step, dict) and "description" in step and step.get("description", "").strip() 
                   for step in steps))

    def _process_multiple_tool_calls_atomically(self, llm_requested_tool_calls: List[Dict[str, Any]], 
                                               executed_tool_details_for_agent: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process multiple tool calls from a single LLM turn atomically.
        
        This handles the case where LLMs (especially Anthropic) make multiple tool calls
        in a single turn. We need to process them all consistently or fail atomically.
        """
        logger_to_use = self.logger
        
        if not llm_requested_tool_calls and not executed_tool_details_for_agent:
            return {"decision": "error", "message": "No tool calls to process", "error_type_for_agent": "NoToolCalls"}
        
        # Categorize tool calls
        plan_update_calls = []
        other_tool_calls = []
        executed_tools = executed_tool_details_for_agent  # Already a list
        
        # Get LLM-seen name for agent:update_plan
        agent_llm_seen_name = None
        for llm_name, original_name in self.server_manager.sanitized_to_original.items():
            if original_name == AGENT_TOOL_UPDATE_PLAN:
                agent_llm_seen_name = llm_name
                break
        
        # Categorize requested tool calls
        for tool_call in llm_requested_tool_calls:
            if not isinstance(tool_call, dict):
                continue
            tool_name = tool_call.get("name", "")
            
            if tool_name in [AGENT_TOOL_UPDATE_PLAN, agent_llm_seen_name]:
                plan_update_calls.append(tool_call)
            else:
                other_tool_calls.append(tool_call)
        
        logger_to_use.info(f"MCPC Atomic Processing: Plan updates: {len(plan_update_calls)}, Other tools: {len(other_tool_calls)}, Executed: {len(executed_tools)}")
        
        # RULE 1: Multiple plan updates in one turn is invalid
        if len(plan_update_calls) > 1:
            logger_to_use.warning(f"MCPC Atomic Processing: Multiple plan updates detected ({len(plan_update_calls)}). Invalid.")
            return {
                "decision": "thought_process",
                "content": f"LLM Error: Multiple plan updates in one turn ({len(plan_update_calls)} calls). Only one allowed.",
                "_mcp_client_force_replan_after_thought_": True
            }
        
        # RULE 2: If executed tools AND requested tools, prefer executed (already happened)
        if executed_tools and (plan_update_calls or other_tool_calls):
            logger_to_use.info(f"MCPC Atomic Processing: Both executed and requested tools found. Prioritizing executed.")
            if len(executed_tools) == 1:
                # Single executed tool - return as before
                return {
                    "decision": "tool_executed_by_mcp",
                    "tool_name": executed_tools[0]["tool_name"],
                    "arguments": executed_tools[0]["arguments"],
                    "result": executed_tools[0]["result"],
                    "deferred_tool_calls": plan_update_calls + other_tool_calls,
                    "total_tools_this_turn": len(executed_tools) + len(plan_update_calls) + len(other_tool_calls)
                }
            else:
                # Multiple executed tools - return as multi-tool execution
                return {
                    "decision": "multiple_tools_executed_by_mcp",
                    "executed_tools": executed_tools,
                    "deferred_tool_calls": plan_update_calls + other_tool_calls,
                    "total_tools_this_turn": len(executed_tools) + len(plan_update_calls) + len(other_tool_calls),
                    "agent_message": f"MCP executed {len(executed_tools)} tools: {', '.join(t['tool_name'] for t in executed_tools)}"
                }
        
        # RULE 3: Process plan update if present (highest priority)
        if plan_update_calls:
            plan_call = plan_update_calls[0]
            plan_steps = plan_call.get("input", {}).get("plan")
            
            if self._validate_plan_steps(plan_steps):
                decision = {
                    "decision": "call_tool",
                    "tool_name": AGENT_TOOL_UPDATE_PLAN,
                    "arguments": {"plan": plan_steps},
                    "tool_use_id": plan_call.get("id"),
                    "deferred_tool_calls": other_tool_calls,
                    "total_tools_this_turn": len(plan_update_calls) + len(other_tool_calls)
                }
                
                if other_tool_calls:
                    logger_to_use.info(f"MCPC Atomic Processing: Processing plan update, deferring {len(other_tool_calls)} other tools")
                    decision["agent_message"] = f"Plan updated. {len(other_tool_calls)} other tool calls deferred to next turn."
                
                return decision
            else:
                logger_to_use.warning(f"MCPC Atomic Processing: Plan update call had invalid structure")
                return {
                    "decision": "thought_process", 
                    "content": f"LLM Error: Plan update call had invalid structure. Plan: {plan_steps}",
                    "_mcp_client_force_replan_after_thought_": True
                }
        
        # RULE 4: Enhanced multi-tool processing for other tool calls
        if other_tool_calls:
            # Check if we can execute multiple tools efficiently in this turn
            can_execute_multiple = self._can_execute_multiple_tools(other_tool_calls)
            
            if can_execute_multiple and len(other_tool_calls) <= 3:  # Cap at 3 tools per turn for safety
                # Execute multiple tools in sequence for this turn
                logger_to_use.info(f"MCPC Atomic Processing: Executing {len(other_tool_calls)} tools in sequence this turn")
                return {
                    "decision": "call_multiple_tools",
                    "tool_calls": other_tool_calls,
                    "total_tools_this_turn": len(other_tool_calls),
                    "agent_message": f"Executing {len(other_tool_calls)} tools in sequence for efficiency."
                }
            else:
                # Original single-tool processing with deferral
                first_tool = other_tool_calls[0]
                deferred_tools = other_tool_calls[1:]
                
                decision = {
                    "decision": "call_tool",
                    "tool_name": first_tool.get("name"),
                    "arguments": first_tool.get("input", {}),
                    "tool_use_id": first_tool.get("id"),
                    "deferred_tool_calls": deferred_tools,
                    "total_tools_this_turn": len(other_tool_calls)
                }
                
                if deferred_tools:
                    logger_to_use.info(f"MCPC Atomic Processing: Processing '{first_tool.get('name')}', deferring {len(deferred_tools)} tools")
                    decision["agent_message"] = f"Processed {first_tool.get('name')}. {len(deferred_tools)} other calls deferred."
                
                return decision
        
        # RULE 5: Only executed tools
        if executed_tools:
            if len(executed_tools) == 1:
                # Single executed tool
                return {
                    "decision": "tool_executed_by_mcp",
                    "tool_name": executed_tools[0]["tool_name"], 
                    "arguments": executed_tools[0]["arguments"],
                    "result": executed_tools[0]["result"]
                }
            else:
                # Multiple executed tools
                return {
                    "decision": "multiple_tools_executed_by_mcp",
                    "executed_tools": executed_tools,
                    "total_tools_this_turn": len(executed_tools),
                    "agent_message": f"MCP executed {len(executed_tools)} tools: {', '.join(t['tool_name'] for t in executed_tools)}"
                }
        
        # RULE 6: Fallback
        return {"decision": "error", "message": "No valid tool calls found", "error_type_for_agent": "NoValidToolCalls"}

    def _can_execute_multiple_tools(self, tool_calls: List[Dict[str, Any]]) -> bool:
        """
        Determine if multiple tools can be executed efficiently in one turn.
        
        ENHANCED CRITERIA for more aggressive multi-tool execution:
        1. Exclude only the most disruptive meta/planning tools
        2. Allow most productive tools to be batched together
        3. Focus on artifact creation and research efficiency
        """
        if not tool_calls or len(tool_calls) > 5:  # Increased limit to 5 tools
            return False
            
        # Explicitly prohibited patterns (tools that must run alone)
        prohibited_patterns = [
            "update_plan", "create_workflow", "update_workflow_status", 
            "record_action_start", "record_action_completion", 
            "save_cognitive_state", "get_rich_context_package"
        ]
        
        # Highly encouraged patterns for multi-execution (artifact creation focus)
        encouraged_patterns = [
            "search", "browse", "write_file", "read_file", "record_artifact",
            "record_thought", "create_goal", "store_memory", "query_memories",
            "smart_browser", "filesystem", "artifact", "mcp_browser", "web_search"
        ]
        
        prohibited_count = 0
        encouraged_count = 0
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "").lower()
            
            # Count prohibited patterns
            if any(prohibited in tool_name for prohibited in prohibited_patterns):
                prohibited_count += 1
            
            # Count encouraged patterns  
            if any(encouraged in tool_name for encouraged in encouraged_patterns):
                encouraged_count += 1
        
        # Allow multi-execution if:
        # 1. No prohibited tools, OR
        # 2. Mostly encouraged tools (at least 60% are encouraged)
        if prohibited_count == 0:
            return True
        elif encouraged_count > 0 and (encouraged_count / len(tool_calls)) >= 0.6:
            return True
        else:
            return False

    def _parse_agent_plan_unified(self, full_text_response: str, llm_requested_tool_calls: List[Dict[str, Any]], 
                                 force_structured_output: bool, force_tool_choice: Optional[str], llm_stop_reason: str) -> Dict[str, Any]:
        """
        Simplified, robust plan parsing focused on the most common cases.
        
        Priority: 
        1. Formal tool calls (highest confidence)
        2. Well-formed JSON in text (medium confidence) 
        3. Everything else is a thought (lowest complexity)
        """
        logger_to_use = self.logger
        
        # Get LLM-seen name for agent:update_plan
        agent_llm_seen_name = None
        for llm_name, original_name in self.server_manager.sanitized_to_original.items():
            if original_name == AGENT_TOOL_UPDATE_PLAN:
                agent_llm_seen_name = llm_name
                break
        
        logger_to_use.debug(f"MCPC Unified Parser: Looking for plan updates using names: '{AGENT_TOOL_UPDATE_PLAN}' or '{agent_llm_seen_name}'")
        
        def create_plan_decision(plan_steps: List[Dict[str, Any]], source: str) -> Dict[str, Any]:
            logger_to_use.info(f"MCPC Plan Parser: ✅ Found valid plan from {source} with {len(plan_steps)} steps")
            return {"decision": "call_tool", "tool_name": AGENT_TOOL_UPDATE_PLAN, 
                   "arguments": {"plan": plan_steps}, "tool_use_id": str(uuid.uuid4())}
        
        def create_thought_decision(content: str, force_replan: bool = False) -> Dict[str, Any]:
            decision = {"decision": "thought_process", "content": content}
            if force_replan:
                decision["_mcp_client_force_replan_after_thought_"] = True
            return decision
            
        def extract_plan_from_dict(data: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
            """Extract plan steps from various dict formats."""
            # Direct plan key
            if "plan" in data and self._validate_plan_steps(data["plan"]):
                return data["plan"]
                
            # Tool call format: {"tool": "agent_update_plan", "tool_input": {"plan": [...]}}
            tool_name = data.get("tool") or data.get("name")
            if tool_name in [AGENT_TOOL_UPDATE_PLAN, agent_llm_seen_name]:
                tool_args = data.get("tool_input") or data.get("arguments") or {}
                if isinstance(tool_args, dict) and self._validate_plan_steps(tool_args.get("plan")):
                    return tool_args["plan"]
                    
            return None

        def extract_tool_call_from_dict(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            """Extract a general tool call from JSON response."""
            if not isinstance(data, dict):
                return None
                
            # Standard tool call format: {"name": "tool_name", "arguments": {...}}
            tool_name = data.get("name")
            tool_args = data.get("arguments") 
            
            if tool_name and isinstance(tool_args, dict):
                # Convert LLM-seen name back to original MCP name
                original_mcp_name = self.server_manager.sanitized_to_original.get(tool_name, tool_name)
                return {
                    "decision": "call_tool",
                    "tool_name": original_mcp_name,
                    "arguments": tool_args,
                    "tool_use_id": str(uuid.uuid4())
                }
                
            # Alternative format: {"tool": "tool_name", "tool_input": {...}}
            tool_name = data.get("tool")
            tool_args = data.get("tool_input")
            
            if tool_name and isinstance(tool_args, dict):
                # Convert LLM-seen name back to original MCP name
                original_mcp_name = self.server_manager.sanitized_to_original.get(tool_name, tool_name)
                return {
                    "decision": "call_tool", 
                    "tool_name": original_mcp_name,
                    "arguments": tool_args,
                    "tool_use_id": str(uuid.uuid4())
                }
                
            return None
        

        
        # Check formal tool calls first
        if llm_requested_tool_calls:
            for tool_call in llm_requested_tool_calls:
                if isinstance(tool_call, dict) and tool_call.get("name") in [AGENT_TOOL_UPDATE_PLAN, agent_llm_seen_name]:
                    plan_steps = tool_call.get("input", {}).get("plan")
                    if self._validate_plan_steps(plan_steps):
                        return create_plan_decision(plan_steps, "formal tool call")
                    return create_thought_decision(f"LLM attempted formal plan update but invalid structure", True)
        
        # Handle forced structured output (highest reliability expected)
        if force_structured_output and force_tool_choice:
            logger_to_use.debug(f"MCPC Plan Parser: Processing forced structured output for tool: {force_tool_choice}")
            
            if (force_tool_choice == agent_llm_seen_name or 
                self.server_manager.sanitized_to_original.get(force_tool_choice) == AGENT_TOOL_UPDATE_PLAN):
                
                # Strategy 1: Try direct JSON parsing first
                try:
                    response_stripped = full_text_response.strip()
                    parsed = json.loads(response_stripped)
                    plan_steps = extract_plan_from_dict(parsed) if isinstance(parsed, dict) else (parsed if self._validate_plan_steps(parsed) else None)
                    if plan_steps:
                        logger_to_use.info(f"MCPC Plan Parser: ✅ Forced structured output parsed successfully (direct JSON)")
                        return create_plan_decision(plan_steps, "forced_structured_direct")
                except json.JSONDecodeError:
                    logger_to_use.debug("MCPC Plan Parser: Direct JSON parsing failed for forced output")
                
                # Strategy 2: Try extracting from markdown
                try:
                    json_content = extract_json_from_markdown(full_text_response)
                    if json_content.strip():
                        parsed = json.loads(json_content)
                        plan_steps = extract_plan_from_dict(parsed) if isinstance(parsed, dict) else (parsed if self._validate_plan_steps(parsed) else None)
                        if plan_steps:
                            logger_to_use.info(f"MCPC Plan Parser: ✅ Forced structured output parsed successfully (markdown extraction)")
                            return create_plan_decision(plan_steps, "forced_structured_markdown")
                except (json.JSONDecodeError, AttributeError):
                    logger_to_use.debug("MCPC Plan Parser: Markdown JSON extraction failed for forced output")
                
                # If forced structured output fails, this is a serious issue
                logger_to_use.error(f"MCPC Plan Parser: 🚨 Forced structured output completely failed to parse! Response: {full_text_response[:200]}...")
                return create_thought_decision(f"LLM forced structured output failed - response was: {full_text_response[:100]}...", True)
        
        # Parse from text content - simplified approach
        if full_text_response:
            response_stripped = full_text_response.strip()
            logger_to_use.debug(f"MCPC Plan Parser: Checking text response (length: {len(response_stripped)})")
            
            # Strategy 1: Whole response is JSON
            if response_stripped.startswith('{') and response_stripped.endswith('}'):
                try:
                    parsed = json.loads(response_stripped)
                    # First check for plan updates
                    plan_steps = extract_plan_from_dict(parsed)
                    if plan_steps:
                        return create_plan_decision(plan_steps, "whole_response_json")
                    # Then check for general tool calls
                    tool_call_result = extract_tool_call_from_dict(parsed)
                    if tool_call_result:
                        logger_to_use.debug(f"MCPC Plan Parser: Recognized tool call: {tool_call_result['tool_name']}")
                        return tool_call_result
                except json.JSONDecodeError:
                    logger_to_use.debug("MCPC Plan Parser: Whole response is not valid JSON")
            
            # Strategy 2: Extract from code blocks  
            code_block_match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", response_stripped)
            if code_block_match:
                try:
                    parsed = json.loads(code_block_match.group(1).strip())
                    # First check for plan updates
                    plan_steps = extract_plan_from_dict(parsed)
                    if plan_steps:
                        return create_plan_decision(plan_steps, "code_block_json")
                    # Then check for general tool calls
                    tool_call_result = extract_tool_call_from_dict(parsed)
                    if tool_call_result:
                        logger_to_use.debug(f"MCPC Plan Parser: Recognized tool call in code block: {tool_call_result['tool_name']}")
                        return tool_call_result
                except json.JSONDecodeError:
                    logger_to_use.debug("MCPC Plan Parser: Code block content is not valid JSON")
            
            # Strategy 3: Find balanced JSON objects
            def find_first_json_object(text: str) -> Optional[str]:
                """Find the first complete JSON object in text."""
                brace_count = 0
                start_pos = None
                
                for i, char in enumerate(text):
                    if char == '{':
                        if brace_count == 0:
                            start_pos = i
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0 and start_pos is not None:
                            return text[start_pos:i+1]
                return None
            
            json_obj = find_first_json_object(response_stripped)
            if json_obj:
                try:
                    parsed = json.loads(json_obj)
                    # First check for plan updates
                    plan_steps = extract_plan_from_dict(parsed)
                    if plan_steps:
                        return create_plan_decision(plan_steps, "extracted_json_object")
                    # Then check for general tool calls
                    tool_call_result = extract_tool_call_from_dict(parsed)
                    if tool_call_result:
                        logger_to_use.debug(f"MCPC Plan Parser: Recognized tool call in extracted JSON: {tool_call_result['tool_name']}")
                        return tool_call_result
                except json.JSONDecodeError:
                    logger_to_use.debug("MCPC Plan Parser: Extracted JSON object is not valid")
            
            logger_to_use.debug("MCPC Plan Parser: No valid plan found in text response")
            
            # Handle special cases
            if llm_stop_reason == "agent_internal_tool_request":
                return create_thought_decision(f"LLM indicated plan update but no valid structure found", True)
            
            if full_text_response.lower().strip().startswith("goal achieved"):
                summary = full_text_response.split(":", 1)[1].strip() if ":" in full_text_response else full_text_response[len("goal achieved"):].strip()
                return {"decision": "complete", "summary": summary}
            
            # Check for other tool calls
            if llm_requested_tool_calls:
                first_tool = llm_requested_tool_calls[0]
                return {"decision": "call_tool", "tool_name": first_tool.get("name"),
                       "arguments": first_tool.get("input", {}), "tool_use_id": first_tool.get("id")}
            
            return create_thought_decision(full_text_response)
        
        return {"decision": "error", "message": "No actionable content from LLM", "error_type_for_agent": "LLMOutputError"}

    async def process_agent_llm_turn(
        self,
        prompt_messages: List[Dict[str, Any]],  # This is InternalMessageList
        tool_schemas: List[Dict[str, Any]],  # LLM-formatted schemas
        model_name: str,  # User-facing model name
        ui_websocket_sender: Optional[Callable[[str, Any], Coroutine]] = None,
        force_structured_output: bool = False,
        force_tool_choice: Optional[str] = None,
        ums_tool_schemas: Optional[Dict[str, Dict[str, Any]]] = None,  # NEW: For UMS structured outputs
    ) -> Dict[str, Any]:
        self.safe_print(f"MCPC: Processing LLM turn for Agent. Model: {model_name}. Provided Tools: {len(tool_schemas) if tool_schemas else 'None'}")

        logger_to_use = getattr(self, "logger", log)

        current_agent_internal_loop = "N/A"
        if self.agent_loop_instance and hasattr(self.agent_loop_instance, "state") and hasattr(self.agent_loop_instance.state, "current_loop"):
            current_agent_internal_loop = str(self.agent_loop_instance.state.current_loop)

        logger_to_use.info(f"MCPC AGENT LLM PROMPT (AML Turn {current_agent_internal_loop}) DETAILS (BEGIN) ==================")
        logger_to_use.info(f"MCPC AGENT LLM PROMPT (AML Turn {current_agent_internal_loop}): Model for this turn: {model_name}")
        logger_to_use.info(f"MCPC AGENT LLM PROMPT (AML Turn {current_agent_internal_loop}): Type of prompt_messages: {type(prompt_messages)}")
        if isinstance(prompt_messages, list) and prompt_messages:
            logger_to_use.debug(
                f"MCPC AGENT LLM PROMPT (AML Turn {current_agent_internal_loop}): First prompt message role: {prompt_messages[0].get('role', 'N/A')}"
            )
            try:
                first_msg_content_str = json.dumps(prompt_messages[0].get("content", ""), default=str, ensure_ascii=False)
                logger_to_use.debug(
                    f"MCPC AGENT LLM PROMPT (AML Turn {current_agent_internal_loop}): First prompt message content preview (first 500 chars): {first_msg_content_str[:500]}..."
                )
            except Exception as e_log_prompt:
                logger_to_use.warning(
                    f"MCPC AGENT LLM PROMPT (AML Turn {current_agent_internal_loop}): Could not serialize first prompt message content for logging: {e_log_prompt}"
                )
        logger_to_use.info(
            f"MCPC AGENT LLM PROMPT (AML Turn {current_agent_internal_loop}): Number of tools provided: {len(tool_schemas) if tool_schemas else 0}"
        )
        if tool_schemas:
            try:
                tool_names_for_log = [
                    t.get("name") or (t.get("function", {}).get("name") if isinstance(t.get("function"), dict) else "UnknownToolName")
                    for t in tool_schemas[:5]
                ]
                logger_to_use.debug(
                    f"MCPC AGENT LLM PROMPT (AML Turn {current_agent_internal_loop}): Sample tool names for LLM: {tool_names_for_log}..."
                )
            except Exception as e_log_tools:
                logger_to_use.warning(
                    f"MCPC AGENT LLM PROMPT (AML Turn {current_agent_internal_loop}): Could not serialize sample tool names for logging: {e_log_tools}"
                )
        logger_to_use.info(f"MCPC AGENT LLM PROMPT (AML Turn {current_agent_internal_loop}) DETAILS (END)   ==================")

        original_graph_node_messages: Optional[List[Dict[str, Any]]] = None
        original_graph_node_model: Optional[str] = None

        if hasattr(self, "conversation_graph") and self.conversation_graph and self.conversation_graph.current_node:
            original_graph_node_messages = self.conversation_graph.current_node.messages.copy()
            original_graph_node_model = self.conversation_graph.current_node.model
            self.conversation_graph.current_node.messages = cast("InternalMessageList", prompt_messages)
            self.conversation_graph.current_node.model = model_name
            logger_to_use.debug(f"MCPC: Temporarily set conversation graph for agent turn (Node: {self.conversation_graph.current_node.id}).")
        else:
            logger_to_use.error("MCPC: Conversation graph or current_node not available for agent LLM turn.")
            return {
                "decision": "error",
                "message": "Internal MCPClient state error: Conversation graph unavailable for agent turn.",
                "error_type_for_agent": "InternalError",
            }

        full_text_response_accumulated: str = ""
        llm_requested_tool_calls: List[Dict[str, Any]] = []
        executed_tool_details_for_agent: List[Dict[str, Any]] = []  # Changed to list to capture multiple tools

        final_decision: Dict[str, Any] = {  # Default error state
            "decision": "error",
            "message": "MCPClient: No actionable decision derived from LLM stream for agent.",
            "error_type_for_agent": "LLMOutputError",
        }
        llm_error_message: Optional[str] = None
        llm_stop_reason: Optional[str] = "unknown_pre_stream"

        try:
            async for event_type, event_data in self._stream_wrapper(
                self.process_streaming_query(
                    query="",
                    model=model_name,
                    messages_override=cast("InternalMessageList", prompt_messages),
                    tools_override=tool_schemas,
                    ui_websocket_sender=ui_websocket_sender,
                    force_structured_output=force_structured_output,
                    force_tool_choice=force_tool_choice,
                    ums_tool_schemas=ums_tool_schemas,  # NEW: Pass UMS tool schemas
                )
            ):
                if ui_websocket_sender:
                    await self._relay_agent_llm_event_to_ui(ui_websocket_sender, event_type, event_data)
                # Clean logging - only log significant events, not every text chunk
                if event_type in ["tool_call_start", "tool_call_end", "mcp_tool_executed_for_agent", "error", "final_stats"]:
                    if isinstance(event_data, dict) and event_data.get("name"):
                        logger_to_use.debug(f"LLM Stream Event: {event_type} - {event_data.get('name')}")
                    elif event_type == "final_stats":
                        logger_to_use.debug(f"LLM Stream Event: {event_type} - Stop reason: {event_data.get('stop_reason', 'unknown')}")
                    elif event_type == "error":
                        logger_to_use.error(f"LLM Stream Event: {event_type} - {str(event_data)[:200]}")
                if event_type == "text":
                    full_text_response_accumulated += str(event_data)
                elif event_type == "tool_call_end":
                    if isinstance(event_data, dict) and event_data.get("id") and event_data.get("name"):
                        # Store the original_mcp_name which _handle_anthropic_stream now provides
                        llm_requested_tool_calls.append(
                            {
                                "id": event_data["id"],
                                "name": event_data["name"],  # This should be the original_mcp_name
                                "input": event_data.get("parsed_input", {}),
                            }
                        )
                        logger_to_use.debug(f"MCPC (Agent Turn Stream): LLM requested tool call: {event_data['name']}")
                    else:
                        logger_to_use.warning(f"MCPC (Agent Turn Stream): Malformed 'tool_call_end' event: {event_data}")

                elif event_type == "mcp_tool_executed_for_agent":
                    if isinstance(event_data, dict) and event_data.get("name"):
                        tool_details = {
                            "tool_name": event_data["name"],  # Original MCP name
                            "arguments": event_data.get("input", {}),
                            "result": event_data.get("result", {"success": False, "error": "Result missing"}),
                        }
                        executed_tool_details_for_agent.append(tool_details)
                        log.info(
                            f"MCPC (Agent Turn Stream): Captured MCPC-executed UMS tool #{len(executed_tool_details_for_agent)} for AML. "
                            f"Tool='{tool_details['tool_name']}'"
                        )
                    else:
                        logger_to_use.warning(f"MCPC (Agent Turn Stream): Malformed 'mcp_tool_executed_for_agent' event: {event_data}")

                elif event_type == "error":
                    llm_error_message = str(event_data)
                    logger_to_use.error(f"MCPC (Agent Turn Stream): Error event from LLM stream: {llm_error_message}")
                    break
                elif event_type == "final_stats":  # Note: This was "final_usage" in Agent loop, now "final_stats" from _stream_wrapper
                    llm_stop_reason = event_data.get("stop_reason", "unknown_from_final_stats")
                    logger_to_use.debug(f"MCPC (Agent Turn Stream): Final stats received. LLM Stop reason for this turn: {llm_stop_reason}")
                    # IMPORTANT: Break here as "final_stats" signals the end of the stream for this LLM turn.
                    break
            # End of stream consumption loop

            # --- Formulate the final decision for AgentMasterLoop ---
            if llm_error_message:
                final_decision = {"decision": "error", "message": llm_error_message, "error_type_for_agent": "LLMError"}
            elif executed_tool_details_for_agent or llm_requested_tool_calls:
                # ATOMIC TOOL CALL PROCESSING - handles multiple tool calls properly
                final_decision = self._process_multiple_tool_calls_atomically(
                    llm_requested_tool_calls, executed_tool_details_for_agent
                )
            else:
                # UNIFIED PLAN PARSING - for text-based content only
                final_decision = self._parse_agent_plan_unified(
                    full_text_response_accumulated, llm_requested_tool_calls, 
                    force_structured_output, force_tool_choice, llm_stop_reason
                )
                

                

        except asyncio.CancelledError:
            logger_to_use.warning("MCPC (Agent Turn): LLM turn processing stream was cancelled by MCPClient or external signal.")
            final_decision = {
                "decision": "error",
                "message": "LLM turn stream processing cancelled by MCPClient.",
                "error_type_for_agent": "CancelledError",
            }
        except Exception as e:
            logger_to_use.error(f"MCPC (Agent Turn): Unexpected error processing LLM turn stream for agent: {e}", exc_info=True)
            final_decision = {
                "decision": "error",
                "message": f"Unexpected error processing LLM stream for agent: {e}",
                "error_type_for_agent": "InternalError",
            }
        finally:
            if hasattr(self, "conversation_graph") and self.conversation_graph and self.conversation_graph.current_node:
                if original_graph_node_messages is not None:
                    self.conversation_graph.current_node.messages = original_graph_node_messages
                if original_graph_node_model is not None:
                    self.conversation_graph.current_node.model = original_graph_node_model
                logger_to_use.debug("MCPC: Restored original conversation graph state after agent LLM turn.")

        decision_log_preview_parts = []
        if final_decision.get("message"):
            decision_log_preview_parts.append(str(final_decision["message"]))
        if final_decision.get("content"):
            decision_log_preview_parts.append(str(final_decision["content"]))
        if final_decision.get("tool_name"):
            decision_log_preview_parts.append(f"Tool: {final_decision['tool_name']}")
        if final_decision.get("updated_plan_steps") and isinstance(final_decision["updated_plan_steps"], list):
            decision_log_preview_parts.append(f"Plan Steps: {len(final_decision['updated_plan_steps'])}")

        decision_log_preview = " | ".join(filter(None, decision_log_preview_parts))[:150]

        logger_to_use.info(
            f"MCPC (Agent Turn): Final decision for AgentMasterLoop: Type='{final_decision.get('decision', 'UNKNOWN DECISION TYPE')}'. Preview: '{decision_log_preview}...'"
        )
        return final_decision

    async def _initialize_agent_if_needed(self, default_llm_model_override: Optional[str] = None) -> bool:
        """
        Initializes or RE-INITIALIZES the AgentMasterLoop instance for a new agent task.
        This involves:
        1. Creating the AgentMasterLoop instance if it doesn't exist.
        2. Updating its configured LLM model if an override is provided for this task.
        3. Calling the AgentMasterLoop's own .initialize() method, which handles:
           - Loading its persistent state from its state file.
           - Validating any loaded workflow_id against UMS and a temp recovery file.
           - Resetting workflow-specific state if the loaded workflow_id is invalid.
           - Saving its (potentially reset) state back.
           - Setting up its tool schemas based on the current MCPClient's available tools.
        Returns True if initialization is successful, False otherwise.
        """
        if self.agent_loop_instance is None:
            log.info("MCPC: Initializing AgentMasterLoop instance for the first time...")
            ActualAgentMasterLoopClass = None
            try:
                # Ensure this import path is correct for your project structure
                from agent_master_loop import AgentMasterLoop as AMLClass

                ActualAgentMasterLoopClass = AMLClass
                log.debug(f"MCPC: Successfully imported AgentMasterLoop as AMLClass: {type(ActualAgentMasterLoopClass)}")
            except ImportError as ie:
                log.critical(f"MCPC: CRITICAL - Failed to import AgentMasterLoop class from 'agent_master_loop.py': {ie}", exc_info=True)
                return False  # Cannot proceed without the class

            if not callable(ActualAgentMasterLoopClass):  # Check if it's a class
                log.critical(f"MCPC: CRITICAL - AgentMasterLoop is not a callable class after import. Type: {type(ActualAgentMasterLoopClass)}")
                return False

            try:
                agent_model_to_use = default_llm_model_override or self.current_model
                self.agent_loop_instance = ActualAgentMasterLoopClass(
                    mcp_client_instance=self,  # Pass self (MCPClient instance)
                    default_llm_model_string=agent_model_to_use,
                    agent_state_file=self.config.agent_state_file_path,  # Pass state file path
                )
                log.info(f"MCPC: AgentMasterLoop instance created. Model: {agent_model_to_use}")
            except Exception as e:  # Catch errors during instantiation
                log.critical(f"MCPC: CRITICAL - Error instantiating AgentMasterLoop: {e}", exc_info=True)
                self.agent_loop_instance = None  # Ensure it's None on failure
                return False
        else:
            log.info("MCPC: Reusing existing AgentMasterLoop instance for new task. Will re-initialize its state.")
            # If reusing, ensure the model is updated if an override is provided for *this new task*
            if default_llm_model_override and self.agent_loop_instance.agent_llm_model != default_llm_model_override:
                log.info(
                    f"MCPC: Updating agent's LLM model for new task from '{self.agent_loop_instance.agent_llm_model}' to '{default_llm_model_override}'."
                )
                self.agent_loop_instance.agent_llm_model = default_llm_model_override

        # ALWAYS call .initialize() on the (new or existing) instance for EACH new task start.
        # This ensures its internal state is freshly loaded/validated from the state file,
        # and its tool schemas are up-to-date with MCPClient's current tool availability.
        if self.agent_loop_instance:
            log.info("MCPC: Calling .initialize() on AgentMasterLoop instance to prepare for new task...")
            if not await self.agent_loop_instance.initialize():  # This now contains the robust state loading and tool setup
                log.error("MCPC: AgentMasterLoop instance .initialize() method FAILED. Agent task cannot start correctly.")
                # Do not nullify self.agent_loop_instance here if it's an existing instance
                # that might be reused later. A failed initialize means the agent
                # cannot start the *current* task correctly.
                return False  # Signal failure to start

            log.info(
                f"MCPC: AgentMasterLoop initialized/re-initialized successfully for new task. Effective WF ID: {_fmt_id(self.agent_loop_instance.state.workflow_id)}"
            )
            return True
        else:  # Should not happen if creation above succeeded for a new instance
            log.critical(
                "MCPC: Agent loop instance is None after creation/reuse attempt in _initialize_agent_if_needed. This indicates a prior critical failure."
            )
            return False

    async def start_agent_task(
        self,
        goal: str,
        max_loops: int = 100,
        llm_model_override: Optional[str] = None,
        ui_websocket_sender: Optional[Callable[[str, Any], Coroutine]] = None,
    ) -> Dict[str, Any]:
        """
        Starts the self-driving agent task in the background.
        This method initializes the agent if needed, resets run-specific state,
        and then kicks off the agent's main processing loop.
        """
        if self.agent_task and not self.agent_task.done():
            log.warning("MCPC: Agent task is already running.")
            return {"success": False, "message": "Agent is already running.", "status": self.agent_status}

        # 1. Initialize AgentMasterLoop instance.
        #    _initialize_agent_if_needed will call agent_loop_instance.initialize(),
        #    which handles loading its persistent state (like UMS workflow_id)
        #    and validating/resetting it if necessary.
        if not await self._initialize_agent_if_needed(default_llm_model_override=llm_model_override):
            # Error messages are logged by _initialize_agent_if_needed or AML.initialize()
            log.error("MCPC: Agent initialization failed. Cannot start agent task.")
            self.agent_status = "error_initializing_agent"  # Update MCPClient status
            return {"success": False, "message": "Agent initialization failed.", "status": self.agent_status}

        # Ensure agent_loop_instance is available after initialization.
        if not self.agent_loop_instance:
            log.critical(
                "MCPC: CRITICAL - Agent loop instance is None after successful _initialize_agent_if_needed. This is an internal logic error."
            )
            self.agent_status = "error_internal_agent_missing"
            return {"success": False, "message": "Internal error: Agent loop instance unavailable post-initialization.", "status": self.agent_status}

        # At this point, self.agent_loop_instance.state is the loaded, validated, and potentially
        # reset (for workflow-specific parts) state from AML.initialize().

        # 2. Set/Reset MCPClient's run-specific agent properties.
        #    These are for MCPClient to track the current agent "run".
        self.agent_goal = goal  # The overall goal for *this specific run/activation*
        self.agent_status = "starting"  # MCPClient's high-level status of the agent task
        self.agent_last_message = "Agent task initiated by MCPClient."  # MCPClient's general message
        self.agent_current_loop = 0  # Loops completed by MCPClient for *this current run*
        self.agent_max_loops = max_loops  # Max loops for *this current run*

        # 3. Reset run-specific state on the AgentMasterLoop's *existing, loaded* state object.
        #    Do NOT re-assign self.agent_loop_instance.state to a new AgentState object here.
        #    The self.agent_loop_instance.state object was already prepared by AML.initialize().

        agent_internal_state = self.agent_loop_instance.state  # Get a reference for clarity

        # Reset AML's internal processing loop counter FOR THIS NEW RUN.
        agent_internal_state.current_loop = 0
        # Reset goal_achieved_flag FOR THIS NEW RUN.
        agent_internal_state.goal_achieved_flag = False
        # Reset error tracking FOR THIS NEW RUN.
        agent_internal_state.consecutive_error_count = 0
        agent_internal_state.last_error_details = None  # Cleared as the new run starts
        # Clear any meta-feedback from a previous run/task.
        agent_internal_state.last_meta_feedback = None

        # `needs_replan` should be determined by AML.initialize() or the first run_main_loop call.
        # If AML.initialize() determined the loaded workflow is invalid, it will reset workflow_id
        # and the first run_main_loop will set a plan to create a new workflow (needs_replan=False).
        # If a valid workflow IS loaded, AML.initialize() might have set needs_replan if the goal stack was problematic.
        # For a truly new task, we often want the agent to start with a fresh assessment, so setting
        # needs_replan to False here is generally correct, allowing run_main_loop to set the initial plan.
        agent_internal_state.needs_replan = False

        # Set a clear initial action summary for this new run.
        agent_internal_state.last_action_summary = f"New agent task directive from MCPClient. Goal: '{goal[:70]}...'"

        # Log the state of critical AML properties *after* MCPClient's run-specific resets.
        log.info(
            f"MCPC: Agent run-specific state (re)set for new task '{goal[:50]}...'. "
            f"AML Internal Loop (target for next turn): {agent_internal_state.current_loop + 1}. "
            f"Effective WF ID from AML state: {_fmt_id(agent_internal_state.workflow_id)}. "
            f"Effective Current UMS Goal ID from AML state: {_fmt_id(agent_internal_state.current_goal_id)}. "
            f"AML Goal Stack Depth: {len(agent_internal_state.goal_stack)}. "
            f"AML Plan Steps: {len(agent_internal_state.current_plan) if agent_internal_state.current_plan else 0}. "
            f"AML NeedsReplan: {agent_internal_state.needs_replan}"
        )

        # 4. Define the agent runner wrapper
        agent_loop_instance_ref = self.agent_loop_instance  # Closure for the wrapper

        async def agent_runner_wrapper():
            nonlocal ui_websocket_sender  # Ensure access to outer scope ui_websocket_sender
            try:
                self.agent_status = "running"  # MCPClient's status updated
                log.info(f"MCPC Agent Task (Wrapper): Starting self-driving agent for goal: '{self.agent_goal}' (Max Loops: {max_loops})")

                # `run_self_driving_agent` will now correctly use the prepared `agent_loop_instance_ref`
                await self.run_self_driving_agent(
                    agent_loop=agent_loop_instance_ref,
                    overall_goal=self.agent_goal,
                    max_agent_loops=max_loops,  # Max loops for this run
                    ui_websocket_sender=ui_websocket_sender,
                )

                # After run_self_driving_agent completes (normally or by max loops)
                # Check the agent_loop_instance_ref's state to determine final MCPClient status
                if agent_loop_instance_ref.state.goal_achieved_flag:
                    self.agent_status = "completed"
                    self.agent_last_message = "Agent achieved its overall goal."
                    log.info(f"MCPC Agent Task (Wrapper): Goal '{self.agent_goal}' achieved.")
                elif agent_loop_instance_ref._shutdown_event.is_set():
                    self.agent_status = "stopped"
                    self.agent_last_message = "Agent task was stopped or shut down (AML signal)."
                    log.info(f"MCPC Agent Task (Wrapper): Agent for goal '{self.agent_goal}' was stopped/shut down (AML signal).")
                else:  # Max loops reached or other non-error termination by AML
                    self.agent_status = "max_loops_reached"
                    self.agent_last_message = f"Agent reached max MCPC-driven loops ({max_loops}) or stopped for other reasons."
                    log.info(f"MCPC Agent Task (Wrapper): Agent for goal '{self.agent_goal}' finished (max_loops or other AML stop).")

            except asyncio.CancelledError:
                self.agent_status = "stopped"
                self.agent_last_message = "Agent task was cancelled by MCPClient."
                log.info(f"MCPC Agent Task (Wrapper): Run for goal '{self.agent_goal}' cancelled by MCPClient.")
                if agent_loop_instance_ref and not agent_loop_instance_ref._shutdown_event.is_set():
                    log.info("MCPC Agent Task (Wrapper): Explicitly calling AML shutdown due to MCPClient cancellation.")
                    await agent_loop_instance_ref.shutdown()
            except Exception as e:
                self.agent_status = "failed"
                self.agent_last_message = f"Agent task failed with exception: {str(e)[:100]}"
                log.error(f"MCPC Agent Task (Wrapper): Error running agent for goal '{self.agent_goal}': {e}", exc_info=True)
                if agent_loop_instance_ref and not agent_loop_instance_ref._shutdown_event.is_set():
                    log.info("MCPC Agent Task (Wrapper): Explicitly calling AML shutdown due to unhandled exception.")
                    await agent_loop_instance_ref.shutdown()
            finally:
                # Update MCPClient's view of the agent's loop count *after* the task finishes
                # This reflects how many loops AML *actually* ran in its last session for this task.
                self.agent_current_loop = agent_loop_instance_ref.state.current_loop if agent_loop_instance_ref else self.agent_max_loops
                log.info(
                    f"MCPC Agent Task (Wrapper): Finished. Final MCPC status: {self.agent_status}. AML loops completed: {self.agent_current_loop}."
                )

        # 5. Create and start the agent runner task
        self.agent_task = asyncio.create_task(agent_runner_wrapper(), name=f"mcp_agent_run_{self.agent_goal[:20].replace(' ', '_')}")
        log.info(f"MCPC: Agent task '{self.agent_task.get_name()}' created and started in background.")

        return {"success": True, "message": "Agent task started successfully in background.", "status": self.agent_status}

    async def stop_agent_task(self) -> Dict[str, Any]:
        """
        Stops the currently running self-driving agent task.
        """
        if not self.agent_task or self.agent_task.done():
            log.info("MCPC: No active agent task to stop.")
            return {"success": False, "message": "No active agent task to stop.", "status": self.agent_status}

        log.info("MCPC: Attempting to stop agent task...")
        self.agent_status = "stopping"
        self.agent_last_message = "Agent stop requested."

        # Signal the AgentMasterLoop to shut down first
        if self.agent_loop_instance:
            await self.agent_loop_instance.shutdown()  # This sets its internal _shutdown_event

        # Then, cancel the wrapper task in MCPClient
        self.agent_task.cancel()
        try:
            await asyncio.wait_for(self.agent_task, timeout=15.0)  # Give it time to clean up
        except asyncio.CancelledError:
            log.info("MCPC: Agent task successfully cancelled.")
            self.agent_status = "stopped"
            self.agent_last_message = "Agent task stopped successfully."
        except asyncio.TimeoutError:
            log.warning("MCPC: Timeout waiting for agent task to stop after cancellation. It might be stuck.")
            self.agent_status = "stopping_timeout"  # A special status
            self.agent_last_message = "Agent task stop timed out."
        except Exception as e:
            log.error(f"MCPC: Error during agent task stop: {e}", exc_info=True)
            self.agent_status = "error_stopping"
            self.agent_last_message = f"Error stopping agent: {str(e)[:100]}"

        return {"success": True, "message": self.agent_last_message, "status": self.agent_status}

    def get_agent_status(self) -> Dict[str, Any]:
        """
        Returns the current status of the self-driving agent, including
        details about its current plan and operational goal stack.
        This method is designed to be called by the API or UI to get a snapshot.
        """
        # --- Initialize with default/fallback values ---
        is_agent_task_active = bool(self.agent_task and not self.agent_task.done())

        # MCPClient's high-level status (this is set by start/stop_agent_task etc.)
        mcpc_agent_status_str = self.agent_status if isinstance(self.agent_status, str) else "unknown"

        # Defaults for when agent_loop_instance is not available or state is minimal
        agent_internal_loop_count = 0
        agent_max_loops_from_mcp = self.agent_max_loops  # Max loops set by MCPC when starting
        agent_last_action_summary = "Agent not initialized or no actions yet."
        agent_current_plan_steps_for_api: List[Dict[str, Any]] = []
        agent_current_plan_step_summary = "No active plan."
        agent_current_plan_step_full_desc = "No active plan."
        agent_current_plan_step_dependencies: List[str] = []

        agent_current_ums_goal_id: Optional[str] = None
        agent_current_ums_goal_desc = "No active UMS goal."
        agent_local_ums_goal_stack_summary_for_api: List[Dict[str, Any]] = []

        agent_target_model_name = "Not set."
        agent_workflow_id: Optional[str] = None
        agent_thought_chain_id: Optional[str] = None

        # --- Derive a user-friendly display status ---
        # Start with MCPClient's view, then refine if agent_loop_instance is active
        status_display = mcpc_agent_status_str.replace("_", " ").title()
        if is_agent_task_active and mcpc_agent_status_str == "running":  # Check both
            status_display = f"Running (Loop .../...)"  # Placeholder, will be updated if agent_loop_instance exists
        elif mcpc_agent_status_str == "completed":
            status_display = "Goal Achieved!"

        # --- If AgentMasterLoop instance exists, get more detailed state ---
        if self.agent_loop_instance:
            agent_state = self.agent_loop_instance.state  # Shortcut to agent's internal state
            agent_internal_loop_count = agent_state.current_loop
            agent_last_action_summary = agent_state.last_action_summary
            agent_target_model_name = self.agent_loop_instance.agent_llm_model  # Get the model AML is configured with
            agent_workflow_id = agent_state.workflow_id
            agent_thought_chain_id = agent_state.current_thought_chain_id

            # Update status_display if agent is actively running its loops
            if is_agent_task_active and mcpc_agent_status_str == "running":
                status_display = f"Running (Loop {agent_internal_loop_count}/{agent_max_loops_from_mcp})"

            # Get current plan from agent's state
            if agent_state.current_plan:
                try:
                    agent_current_plan_steps_for_api = [step.model_dump(exclude_none=True) for step in agent_state.current_plan]
                    if agent_current_plan_steps_for_api:  # Ensure plan is not empty
                        current_step_obj = agent_state.current_plan[0]
                        agent_current_plan_step_full_desc = current_step_obj.description
                        agent_current_plan_step_summary = current_step_obj.description[:100] + (
                            "..." if len(current_step_obj.description) > 100 else ""
                        )
                        agent_current_plan_step_dependencies = current_step_obj.depends_on
                    else:
                        agent_current_plan_step_summary = "Plan is empty."
                        agent_current_plan_step_full_desc = "Plan is empty."
                except Exception as e:
                    log.warning(f"MCPC: Error serializing agent plan for status: {e}")
                    agent_current_plan_step_summary = "Error getting plan."
                    agent_current_plan_step_full_desc = "Error getting plan."
                    agent_current_plan_steps_for_api = [{"error": "Could not serialize plan"}]

            # Get current UMS goal and stack summary from agent's state
            # The agent's `self.state.goal_stack` should contain full UMS goal dictionaries
            # The agent's `self.state.current_goal_id` is the ID of its current operational UMS goal
            agent_current_ums_goal_id = agent_state.current_goal_id

            if agent_state.current_goal_id and agent_state.goal_stack:
                # Find the current operational goal object in the stack
                current_op_goal_obj_from_stack = next(
                    (g for g in reversed(agent_state.goal_stack) if isinstance(g, dict) and g.get("goal_id") == agent_state.current_goal_id), None
                )
                if current_op_goal_obj_from_stack:
                    agent_current_ums_goal_desc = current_op_goal_obj_from_stack.get("description", "Unnamed UMS Operational Goal")
                else:
                    agent_current_ums_goal_desc = f"UMS Goal ID '{_fmt_id(agent_state.current_goal_id)}' (Details not in local stack)"
                    
                    # Throttle this warning to prevent log spam - only warn once per 30 seconds per goal ID
                    warning_key = f"goal_sync_warning_{agent_state.current_goal_id}"
                    import time
                    current_time = time.time()
                    
                    if not hasattr(self, '_last_warning_times'):
                        self._last_warning_times = {}
                    
                    last_warning_time = self._last_warning_times.get(warning_key, 0)
                    if current_time - last_warning_time > 30:  # Only warn every 30 seconds
                        log.warning(
                            f"MCPC: Agent's current_goal_id {_fmt_id(agent_state.current_goal_id)} not found in its local goal_stack of {len(agent_state.goal_stack)} items. "
                            f"This usually indicates a temporary sync issue that resolves automatically."
                        )
                        self._last_warning_times[warning_key] = current_time
                    else:
                        log.debug(f"MCPC: Goal sync mismatch (throttled warning) - current_goal_id {_fmt_id(agent_state.current_goal_id)} not in local stack")

                # Summarize the agent's local view of the UMS goal stack for the API
                # The stack in agent_state.goal_stack is already ordered root-to-leaf
                # Take the last N items for summary (leaf-most)
                GOAL_STACK_SUMMARY_LIMIT_FOR_API = 5  # Defined in AML, copied for consistency
                goals_to_summarize_for_api = agent_state.goal_stack[-GOAL_STACK_SUMMARY_LIMIT_FOR_API:]
                for goal_dict_from_agent_stack in goals_to_summarize_for_api:
                    if isinstance(goal_dict_from_agent_stack, dict):  # Ensure it's a dict
                        agent_local_ums_goal_stack_summary_for_api.append(
                            {
                                "goal_id": goal_dict_from_agent_stack.get("goal_id", "UnknownID"),
                                "description": goal_dict_from_agent_stack.get("description", "No description provided"),  # Full description
                                "status": goal_dict_from_agent_stack.get("status", "unknown"),
                            }
                        )
            elif not agent_state.current_goal_id and agent_state.workflow_id:
                # Workflow is active, but no specific UMS goal is the current focus (e.g., between root goals, or if root failed to set)
                agent_current_ums_goal_desc = "Workflow active, UMS operational goal not set."
                if agent_state.goal_stack:  # Should be empty if current_goal_id is None
                    log.warning("MCPC: Agent has goal_stack items but no current_goal_id. Inconsistent state.")

        # --- Construct the final status dictionary for the API ---
        status_to_return = {
            "agent_running": is_agent_task_active,
            "status": mcpc_agent_status_str,  # MCPClient's high-level status
            "status_display": status_display,  # User-friendly display string
            "overall_goal_from_mcp_client": self.agent_goal,  # The goal MCPClient was asked to start the agent with
            # Details from AgentMasterLoop's state
            "agent_workflow_id": agent_workflow_id,
            "agent_thought_chain_id": agent_thought_chain_id,
            "agent_target_model": agent_target_model_name,
            "agent_internal_loop_count": agent_internal_loop_count,
            "agent_max_loops_from_mcp_config": agent_max_loops_from_mcp,  # Max loops MCPC is running the agent for
            "agent_current_ums_goal_id": agent_current_ums_goal_id,
            "agent_current_ums_goal_description": agent_current_ums_goal_desc,  # Full description
            "agent_local_ums_goal_stack_summary": agent_local_ums_goal_stack_summary_for_api,  # Summary of agent's view
            "agent_last_action_summary": agent_last_action_summary,
            "agent_last_general_message": self.agent_last_message,  # General message from MCPClient about agent task
            "agent_current_plan_step_summary": agent_current_plan_step_summary,  # Truncated
            "agent_current_plan_step_full_description": agent_current_plan_step_full_desc,  # Full
            "agent_current_plan_step_dependencies": agent_current_plan_step_dependencies,
            "agent_current_full_plan": agent_current_plan_steps_for_api,  # List of all plan step dicts
        }

        # Log a summary of what's being returned by this status check
        log.debug(
            f"MCPC Agent Status: Running={status_to_return['agent_running']}, "
            f"MCPC Status='{status_to_return['status']}', "
            f"AML Loop={status_to_return['agent_internal_loop_count']}, "
            f"UMS Goal='{_fmt_id(status_to_return['agent_current_ums_goal_id'])}', "
            f"Plan Steps={len(status_to_return['agent_current_full_plan'])}"
        )

        return status_to_return

    async def run_self_driving_agent(
        self,
        agent_loop: AgentMasterLoop,  # Type hint AgentMasterLoop correctly
        overall_goal: str,
        max_agent_loops: int,
        ui_websocket_sender: Optional[Callable[[str, Any], Coroutine]] = None,
    ):
        self.safe_print(f"\n[bold cyan]{EMOJI_MAP.get('robot', '🤖')} MCPC Driving Agent...[/]")
        self.safe_print(f"Overall Goal (from MCPC): [yellow]{overall_goal}[/]")
        self.safe_print(f"Max Loops (from MCPC): {max_agent_loops}")
        log.info(f"MCPC: Orchestrating self-driving agent. Goal: '{overall_goal}', Max Loops: {max_agent_loops}")

        # Main loop for agent turns, managed by MCPClient
        for loop_num_mcp_managed in range(max_agent_loops):
            # Check agent's internal shutdown/completion flags first
            if agent_loop._shutdown_event.is_set():
                self.safe_print(f"MCPC: Agent shutdown signal detected at start of turn {loop_num_mcp_managed + 1}. Stopping agent.")
                log.info(f"MCPC: Agent shutdown signal detected. Loop {loop_num_mcp_managed + 1}/{max_agent_loops}.")
                self.agent_status = "stopped"
                self.agent_last_message = "Agent shutdown signal."
                break
            if agent_loop.state.goal_achieved_flag:
                self.safe_print(f"MCPC: Agent signaled overall goal achievement at start of turn {loop_num_mcp_managed + 1}. Stopping agent.")
                log.info(f"MCPC: Agent goal_achieved_flag is true. Loop {loop_num_mcp_managed + 1}/{max_agent_loops}.")
                self.agent_status = "completed"
                self.agent_last_message = "Agent achieved overall goal."
                break

            self.agent_current_loop = loop_num_mcp_managed
            current_aml_loop = agent_loop.state.current_loop + 1
            self.safe_print(
                f"\n--- MCPC: Starting Agent Turn {loop_num_mcp_managed + 1}/{max_agent_loops} (AML Internal Loop: {current_aml_loop}) ---"
            )
            log.info(f"MCPC: Beginning agent turn {loop_num_mcp_managed + 1}. AML Internal Loop: {current_aml_loop}.")
            self.agent_status = "running"
            self.agent_last_message = f"Executing turn {current_aml_loop}."

            # 1. Agent prepares data for the LLM turn.
            turn_data_for_llm_dict: Optional[Dict[str, Any]] = None  # Initialize
            prompt_messages: Optional[List[Dict[str, Any]]] = None
            tool_schemas: Optional[List[Dict[str, Any]]] = None

            try:
                turn_data_for_llm_dict = await agent_loop.run_main_loop(overall_goal, max_agent_loops)
                if turn_data_for_llm_dict is None:
                    self.safe_print("MCPC: Agent signaled termination during its data preparation phase (run_main_loop returned None).")
                    log.info("MCPC: AgentMasterLoop.run_main_loop returned None, signaling stop.")
                    self.agent_status = "stopped"
                    self.agent_last_message = "Agent run_main_loop phase signaled stop."
                    break

                if (
                    not isinstance(turn_data_for_llm_dict, dict)
                    or "prompt_messages" not in turn_data_for_llm_dict
                    or "tool_schemas" not in turn_data_for_llm_dict
                ):
                    log.error(
                        f"MCPC: CRITICAL - AgentMasterLoop.run_main_loop returned unexpected data format: {type(turn_data_for_llm_dict)}. Keys: {list(turn_data_for_llm_dict.keys()) if isinstance(turn_data_for_llm_dict, dict) else 'N/A'}. Expected dict with 'prompt_messages' and 'tool_schemas'."
                    )
                    self.agent_status = "failed"
                    self.agent_last_message = "Internal error: Agent data preparation failed."
                    if agent_loop and not agent_loop._shutdown_event.is_set():
                        await agent_loop.shutdown()
                    break

                prompt_messages = turn_data_for_llm_dict.get("prompt_messages")
                tool_schemas = turn_data_for_llm_dict.get("tool_schemas")

                if prompt_messages is None or tool_schemas is None:
                    log.error(
                        f"MCPC: CRITICAL - 'prompt_messages' or 'tool_schemas' missing from agent_loop.run_main_loop result. PM type: {type(prompt_messages)}, TS type: {type(tool_schemas)}"
                    )
                    self.agent_status = "failed"
                    self.agent_last_message = "Internal error: Agent data structure malformed."
                    if agent_loop and not agent_loop._shutdown_event.is_set():
                        await agent_loop.shutdown()
                    break

            except asyncio.CancelledError:
                self.safe_print("MCPC: Agent data preparation was cancelled. Stopping agent.")
                log.info("MCPC: AgentMasterLoop.run_main_loop was cancelled.")
                if agent_loop and not agent_loop._shutdown_event.is_set():
                    await agent_loop.shutdown()
                self.agent_status = "stopped"
                self.agent_last_message = "Agent preparation cancelled."
                break
            except Exception as prep_err:
                self.safe_print(f"[bold red]MCPC: Error during agent data preparation: {str(prep_err)[:200]}[/]")
                log.error(f"MCPC: Critical error in agent_loop.run_main_loop: {prep_err}", exc_info=True)
                self.agent_status = "failed"
                self.agent_last_message = f"Preparation error: {str(prep_err)[:100]}"
                if agent_loop and not agent_loop._shutdown_event.is_set():
                    await agent_loop.shutdown()
                break

            # Ensure prompt_messages and tool_schemas are not None before proceeding
            if prompt_messages is None or tool_schemas is None:
                # This case should have been caught by the checks above, but as a safeguard:
                log.critical("MCPC: prompt_messages or tool_schemas is None after preparation block. This indicates a logic flaw.")
                self.agent_status = "failed"
                self.agent_last_message = "Internal critical error: Malformed agent data."
                if agent_loop and not agent_loop._shutdown_event.is_set():
                    await agent_loop.shutdown()
                break

            # 2. MCPClient makes the LLM call
            self.safe_print(f"MCPC: Requesting LLM decision (Model: {agent_loop.agent_llm_model})...")
            self.agent_last_message = f"Requesting LLM decision for turn {current_aml_loop}..."

            # Detect if we're in a needs_replan state - determine if we should use structured output
            force_structured_output = False
            force_tool_choice = None
            if agent_loop.state.needs_replan:
                # When replanning is needed, we want to force structured output and tool choice
                force_structured_output = True
                # We'll use the provider name later when passing to process_streaming_query
                # Force tool choice for agent:update_plan
                sanitized_tool_name = None
                for schema in tool_schemas:
                    name = schema.get("name", "")
                    if not name:
                        continue
                    original = self.server_manager.sanitized_to_original.get(name)
                    if original == AGENT_TOOL_UPDATE_PLAN:
                        sanitized_tool_name = name
                        break
                if sanitized_tool_name:
                    force_tool_choice = sanitized_tool_name
                    log.info(f"MCPC: Agent needs replan - forcing structured output and tool choice: {sanitized_tool_name}")

            try:
                # Extract ums_tool_schemas from the turn_data_for_llm_dict if available
                ums_tool_schemas = turn_data_for_llm_dict.get("ums_tool_schemas", {})
                
                # Add timeout protection for LLM calls (3 minutes max)
                llm_decision = await asyncio.wait_for(
                    self.process_agent_llm_turn(
                        prompt_messages=prompt_messages,  # Now correctly assigned
                        tool_schemas=tool_schemas,  # Now correctly assigned
                        model_name=agent_loop.agent_llm_model,
                        ui_websocket_sender=ui_websocket_sender,
                        force_structured_output=force_structured_output,
                        force_tool_choice=force_tool_choice,
                        ums_tool_schemas=ums_tool_schemas,  # NEW: Pass UMS schemas for structured outputs
                    ),
                    timeout=180.0  # 3 minutes timeout
                )
            except asyncio.TimeoutError:
                self.safe_print(f"[bold red]MCPC: LLM call timeout (3 minutes) - agent may be stuck or model overloaded[/]")
                log.error(f"MCPC: LLM call timeout after 180 seconds for turn {loop_num_mcp_managed + 1}")
                llm_decision = {
                    "decision": "error",
                    "message": "LLM call timed out after 3 minutes. Consider using a faster model or simpler prompts.",
                    "error_type_for_agent": "LLMTimeoutError",
                }
                self.agent_status = "failed"
                self.agent_last_message = "LLM call timed out"
                if not agent_loop._shutdown_event.is_set():
                    await agent_loop.shutdown()
                break
            except Exception as llm_call_err:
                self.safe_print(f"[bold red]MCPC: Critical error calling LLM for agent: {str(llm_call_err)[:200]}[/]")
                log.error(f"MCPC: Unhandled exception in self.process_agent_llm_turn: {llm_call_err}", exc_info=True)
                llm_decision = {
                    "decision": "error",
                    "message": f"MCPClient failed to get LLM decision: {llm_call_err}",
                    "error_type_for_agent": "MCPClientError",
                }
                self.agent_status = "failed"  # Update MCPClient's status
                self.agent_last_message = f"LLM call error: {str(llm_call_err)[:100]}"
                if not agent_loop._shutdown_event.is_set():
                    await agent_loop.shutdown()
                break

            # 3. Agent executes the LLM decision.
            decision_type_log = llm_decision.get("decision", "unknown")
            self.safe_print(f"MCPC: Passing LLM decision (type: {decision_type_log}) to agent for execution...")
            log.debug(f"MCPC: LLM decision for agent: {str(llm_decision)[:300]}")
            self.agent_last_message = f"Executing LLM decision (type: {decision_type_log}) for turn {current_aml_loop}..."
            try:
                # Call execute_llm_decision directly on the agent_loop instance
                should_continue_loop = await agent_loop.execute_llm_decision(llm_decision)

                # Relay agent's internal action summary if sender available (AFTER execution)
                if ui_websocket_sender and agent_loop and agent_loop.state.last_action_summary:
                    activity_summary = agent_loop.state.last_action_summary
                    await ui_websocket_sender(
                        "agent_activity_log", {"summary": activity_summary, "timestamp": datetime.now(timezone.utc).isoformat()}
                    )

                if not should_continue_loop:
                    self.safe_print("MCPC: Agent signaled loop termination after executing decision.")
                    log.info("MCPC: AgentMasterLoop.execute_llm_decision returned False (stop).")
                    if not agent_loop.state.goal_achieved_flag:
                        self.agent_status = "stopped"
                        self.agent_last_message = "Agent decided to stop."
                    # If goal_achieved_flag is true, status will be set later based on it.
                    break
            except asyncio.CancelledError:
                self.safe_print("MCPC: Agent decision execution was cancelled. Stopping agent.")
                log.info("MCPC: AgentMasterLoop.execute_llm_decision was cancelled.")
                if not agent_loop._shutdown_event.is_set():
                    await agent_loop.shutdown()
                self.agent_status = "stopped"
                self.agent_last_message = "Agent execution cancelled."
                break
            except Exception as exec_err:
                self.safe_print(f"[bold red]MCPC: Error during agent decision execution: {str(exec_err)[:200]}[/]")
                log.error(f"MCPC: Critical error in agent_loop.execute_llm_decision: {exec_err}", exc_info=True)
                self.agent_status = "failed"
                self.agent_last_message = f"Execution error: {str(exec_err)[:100]}"
                if not agent_loop._shutdown_event.is_set():
                    await agent_loop.shutdown()
                break

            # Note: AgentMasterLoop.state.current_loop is incremented internally by agent_loop.execute_llm_decision upon success.

            if agent_loop.state.goal_achieved_flag:
                self.safe_print("MCPC: Agent goal_achieved_flag is now true after decision execution. Stopping.")
                log.info("MCPC: Agent goal_achieved_flag became true. Loop will terminate.")
                self.agent_status = "completed"
                self.agent_last_message = "Agent achieved overall goal."
                break

            await asyncio.sleep(0.1)

        # --- Loop End ---
        final_aml_loop_count = agent_loop.state.current_loop if agent_loop else 0
        self.agent_current_loop = final_aml_loop_count  # Ensure final sync for agentStatus API

        # Update final status based on how the loop exited
        if self.agent_status == "running":  # If loop finished due to max_agent_loops
            self.agent_status = "max_loops_reached"
            self.agent_last_message = f"Agent reached max MCPC-driven loops ({max_agent_loops})."

        # Print final summary message
        if self.agent_status == "completed":
            self.safe_print(
                f"[bold green]{EMOJI_MAP.get('party_popper', '🎉')} Agent achieved its overall goal after {final_aml_loop_count} internal loops![/]"
            )
            log.info(f"MCPC: Agent run finished - goal achieved after {final_aml_loop_count} internal loops.")
        elif self.agent_status == "stopped":
            self.safe_print(f"[yellow]{EMOJI_MAP.get('cancel', '🛑')} Agent task stopped/cancelled after {final_aml_loop_count} internal loops.[/]")
            log.info(f"MCPC: Agent run finished - stopped/cancelled after {final_aml_loop_count} internal loops.")
        elif self.agent_status == "failed":
            self.safe_print(f"[bold red]{EMOJI_MAP.get('collision', '💥')} Agent task failed after {final_aml_loop_count} internal loops.[/]")
            log.info(f"MCPC: Agent run finished - failed after {final_aml_loop_count} internal loops.")
        else:  # Max loops reached or other states
            self.safe_print(
                f"[yellow]{EMOJI_MAP.get('warning', '⚠️')} Agent run concluded after {final_aml_loop_count} internal loops. Final status: {self.agent_status}[/]"
            )
            log.info(f"MCPC: Agent run finished - {final_aml_loop_count} internal loops. Status: {self.agent_status}")

        if agent_loop and not agent_loop._shutdown_event.is_set():
            self.safe_print("MCPC: Ensuring final agent shutdown process...")
            await agent_loop.shutdown()
        elif not agent_loop:
            log.warning("MCPC: Agent loop instance was None at end of run_self_driving_agent.")
        else:
            self.safe_print("MCPC: Agent already processing shutdown.")

        self.safe_print(f"[bold cyan]{EMOJI_MAP.get('robot', '🤖')} Self-Driving Agent Orchestration Concluded by MCPC.[/]")

    async def interactive_loop(self):
        """Runs the main interactive command loop with direct stream handling."""
        interactive_console = get_safe_console()  # Get console instance once

        self.safe_print("\n[bold green]MCP Client Interactive Mode[/]")
        self.safe_print(f"Default Model: [cyan]{self.current_model}[/]. Type '/help' for commands.")
        self.safe_print("[italic dim]Press Ctrl+C once to abort request, twice quickly to exit[/italic dim]")

        # Define fixed layout dimensions
        RESPONSE_PANEL_HEIGHT = 55
        STATUS_PANEL_HEIGHT = 10  # Reduced status height
        # --- Throttling Config ---
        TEXT_UPDATE_INTERVAL = 0.2  # Update Markdown max ~5 times/sec

        @contextmanager
        def suppress_logs_during_live():
            """Temporarily suppress logging below WARNING during Live display."""
            logger_instance = logging.getLogger("mcpclient_multi")  # Get specific logger
            original_level = logger_instance.level
            # Suppress INFO and DEBUG during live updates
            if original_level < logging.WARNING:
                logger_instance.setLevel(logging.WARNING)
            try:
                yield
            finally:
                logger_instance.setLevel(original_level)  # Restore original level

        while True:
            # --- Reset state for this iteration ---
            live_display: Optional[Live] = None
            self.current_query_task = None
            self._current_query_text = ""
            self._current_status_messages = []
            query_error: Optional[Exception] = None
            query_cancelled = False
            final_stats_to_print = {}
            last_text_update_time = 0.0  # For throttling

            try:
                user_input = await asyncio.to_thread(Prompt.ask, "\n[bold blue]>>> [/]", console=interactive_console, default="")
                user_input = user_input.strip()

                if not user_input:
                    continue

                # --- Command Handling ---
                if user_input.startswith("/"):
                    # (Command handling logic remains the same - ensure active display check)
                    if hasattr(self, "_active_live_display") and self._active_live_display:
                        log.warning("Attempting to run command while Live display might be active. Stopping display.")
                        if live_display and live_display.is_started:
                            with suppress(Exception):
                                live_display.stop()
                        self._active_live_display = None
                        live_display = None

                    try:
                        cmd_parts = shlex.split(user_input[1:])
                        if not cmd_parts:
                            continue
                        cmd = cmd_parts[0].lower()
                        args = shlex.join(cmd_parts[1:]) if len(cmd_parts) > 1 else ""
                    except ValueError as e:
                        self.safe_print(f"[red]Error parsing command: {e}[/]")
                        continue

                    handler = self.commands.get(cmd)
                    if handler:
                        log.debug(f"Executing command: /{cmd} with args: '{args}'")
                        await handler(args)
                    else:
                        self.safe_print(f"[yellow]Unknown command: [bold]/{cmd}[/]. Type /help for list.[/yellow]")

                # --- Query Handling ---
                else:
                    query = user_input
                    status_lines = deque(maxlen=STATUS_PANEL_HEIGHT)
                    abort_message = Text("Press Ctrl+C once to abort...", style="dim yellow")
                    current_response_text = ""  # Text accumulated *during* the live update
                    needs_render_update = False  # Flag to trigger render

                    # --- Setup Live Display Placeholders ---
                    response_panel = Panel(Text("Waiting...", style="dim"), title="Response", height=RESPONSE_PANEL_HEIGHT, border_style="dim blue")
                    status_panel = Panel(
                        Group(*[Text("")] * STATUS_PANEL_HEIGHT), title="Status", height=STATUS_PANEL_HEIGHT + 2, border_style="dim blue"
                    )
                    abort_panel = Panel(abort_message, height=3, border_style="none")
                    main_group = Group(response_panel, status_panel, abort_panel)
                    live_panel = Panel(main_group, title=f"Querying {self.current_model}...", border_style="dim green")

                    self._active_live_display = True  # Set flag
                    consuming_task = asyncio.current_task()
                    self.current_query_task = consuming_task

                    # --- Start Live Display and Consume Stream ---
                    with suppress_logs_during_live():
                        with Live(live_panel, console=interactive_console, refresh_per_second=12, transient=True, vertical_overflow="crop") as live:
                            live_display = live
                            try:
                                current_time = time.time()
                                last_text_update_time = current_time  # Initialize time

                                # Directly iterate over the wrapped generator stream
                                async for chunk_type, chunk_data in self._stream_wrapper(self.process_streaming_query(query)):
                                    current_time = time.time()
                                    needs_render_update = False  # Reset flag for this chunk

                                    if consuming_task.cancelled():
                                        query_cancelled = True
                                        log.info("Query consuming loop cancelled.")
                                        break

                                    # Handle different chunk types
                                    if chunk_type == "text":
                                        current_response_text += chunk_data
                                        # --- Throttle Markdown Update ---
                                        if current_time - last_text_update_time >= TEXT_UPDATE_INTERVAL:
                                            response_panel.renderable = Markdown(current_response_text, code_theme="monokai")
                                            response_panel.border_style = "blue"  # Update style as text comes in
                                            live_panel.border_style = "green"
                                            last_text_update_time = current_time
                                            needs_render_update = True
                                        # --- End Throttle ---
                                    elif chunk_type == "status":
                                        clean_status_text = Text.from_markup(chunk_data)
                                        status_lines.append(clean_status_text)
                                        display_status = list(status_lines)[-STATUS_PANEL_HEIGHT:]
                                        if len(display_status) < STATUS_PANEL_HEIGHT:
                                            padding = [Text("")] * (STATUS_PANEL_HEIGHT - len(display_status))
                                            display_status = padding + display_status
                                        status_panel.renderable = Group(*display_status)
                                        status_panel.border_style = "blue"
                                        needs_render_update = True  # Update immediately for status
                                    elif chunk_type == "error":
                                        query_error = RuntimeError(chunk_data)
                                        status_lines.append(Text.from_markup(f"[bold red]Error: {chunk_data}[/bold red]"))
                                        display_status = list(status_lines)[-STATUS_PANEL_HEIGHT:]
                                        if len(display_status) < STATUS_PANEL_HEIGHT:
                                            padding = [Text("")] * (STATUS_PANEL_HEIGHT - len(display_status))
                                            display_status = padding + display_status
                                        status_panel.renderable = Group(*display_status)
                                        status_panel.border_style = "red"
                                        needs_render_update = True  # Update immediately for error
                                    elif chunk_type == "final_stats":
                                        final_stats_to_print = chunk_data

                                    # Update abort message visibility & title
                                    abort_panel.renderable = abort_message if not consuming_task.done() else Text("")
                                    live_panel.title = (
                                        f"Querying {self.current_model}..." if not consuming_task.done() else f"Result ({self.current_model})"
                                    )
                                    # Always need to update if title/abort message changes
                                    needs_render_update = True

                                    # Refresh the Live display if needed
                                    if needs_render_update:
                                        live.update(live_panel)

                                # --- Perform one final update after the loop ---
                                # Ensure the last text chunk is rendered
                                response_panel.renderable = Markdown(current_response_text, code_theme="monokai")
                                response_panel.border_style = (
                                    "blue" if not query_error and not query_cancelled else ("red" if query_error else "yellow")
                                )
                                # Update final status state
                                display_status_final = list(status_lines)[-STATUS_PANEL_HEIGHT:]
                                if len(display_status_final) < STATUS_PANEL_HEIGHT:
                                    padding = [Text("")] * (STATUS_PANEL_HEIGHT - len(display_status_final))
                                    display_status_final = padding + display_status_final
                                status_panel.renderable = Group(*display_status_final)
                                status_panel.border_style = (
                                    "blue" if not query_error and not query_cancelled else ("red" if query_error else "yellow")
                                )
                                # Update final title/abort message
                                abort_panel.renderable = Text("")
                                live_panel.title = f"Result ({self.current_model})" + (
                                    " - Cancelled" if query_cancelled else (" - Error" if query_error else "")
                                )
                                live_panel.border_style = "green" if not query_error and not query_cancelled else ("red" if query_error else "yellow")
                                # Trigger final refresh
                                live.update(live_panel)
                                # --- End Final Update ---

                            except asyncio.CancelledError:
                                query_cancelled = True
                                log.info("Query task was cancelled (caught in Live context).")
                                status_lines.append(Text("[yellow]Request Aborted.[/yellow]", style="yellow"))
                                # Final update happens above
                            except Exception as e:
                                query_error = e
                                log.error(f"Error consuming query stream in interactive loop: {e}", exc_info=True)
                                status_lines.append(Text(f"[bold red]Error: {e}[/bold red]", style="red"))
                                # Final update happens above
                            finally:
                                if self.current_query_task == consuming_task:
                                    self.current_query_task = None

                    # --- Live display stops automatically here ---
                    self._active_live_display = None
                    live_display = None

                    # --- Print Final Result Panel (outside Live) ---
                    final_response_renderable: Union[Markdown, Text]
                    final_title = f"Result ({self.current_model})"
                    final_border = "green"

                    if query_cancelled:
                        final_response_renderable = (
                            Markdown(current_response_text, code_theme="monokai")
                            if current_response_text
                            else Text("Query aborted by user.", style="yellow")
                        )
                        final_title += " - Cancelled"
                        final_border = "yellow"
                    elif query_error:
                        final_response_renderable = (
                            Markdown(current_response_text) if current_response_text else Text(f"Error: {query_error}", style="red")
                        )
                        final_title += " - Error"
                        final_border = "red"
                    else:
                        final_response_renderable = (
                            Markdown(current_response_text, code_theme="monokai")
                            if current_response_text
                            else Text("[dim]No text content received.[/dim]", style="dim")
                        )

                    # Rebuild final status renderable using the final state of status_lines
                    display_status_final = list(status_lines)[-STATUS_PANEL_HEIGHT:]
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

                    # Print final stats panel if available and no error/cancellation
                    if final_stats_to_print and not query_error and not query_cancelled:
                        # Call the *correct* method definition
                        self._print_final_query_stats(interactive_console, final_stats_to_print)
                    elif not query_error and not query_cancelled:
                        log.warning("Final stats were not received from query processor.")

            # --- Outer Loop Error Handling ---
            except KeyboardInterrupt:
                self.safe_print("\n[yellow]Input interrupted. Type /exit or Ctrl+C again to quit.[/yellow]")
                if live_display and live_display.is_started:
                    with suppress(Exception):
                        live_display.stop()
                    self._active_live_display = None
                    live_display = None
                continue
            except EOFError:
                self.safe_print("\n[yellow]EOF received, exiting...[/]")
                break
            except Exception as loop_err:
                self.safe_print(f"\n[bold red]Unexpected Error in interactive loop:[/] {loop_err}")
                log.error("Unexpected error in interactive loop", exc_info=True)
                if live_display and live_display.is_started:
                    with suppress(Exception):
                        live_display.stop()
                    self._active_live_display = None
                    live_display = None
                await asyncio.sleep(1)

            finally:
                if live_display and live_display.is_started:
                    with suppress(Exception):
                        live_display.stop()
                self._active_live_display = None
                if self.current_query_task == asyncio.current_task():
                    self.current_query_task = None
                self._current_query_text = ""
                self._current_status_messages = []

    def _print_final_query_stats(self, target_console: Console, stats_data: Dict):
        """Helper to format and print the final query statistics panel from provided data."""
        # Extract stats from the dictionary
        session_input_tokens = stats_data.get("input_tokens", 0)
        session_output_tokens = stats_data.get("output_tokens", 0)
        session_total_cost = stats_data.get("total_cost", 0.0)
        cache_hits = stats_data.get("cache_hits", 0)
        cache_misses = stats_data.get("cache_misses", 0)
        tokens_saved = stats_data.get("tokens_saved", 0)

        # Calculate hit rate for *this query*
        hit_rate = 0.0
        total_lookups = cache_hits + cache_misses
        if total_lookups > 0:
            hit_rate = (cache_hits / total_lookups) * 100

        # Calculate cost savings for *this query*
        cost_saved = 0.0
        if tokens_saved > 0:
            model_cost_info = COST_PER_MILLION_TOKENS.get(self.current_model, {})
            input_cost_per_token = model_cost_info.get("input", 0) / 1_000_000
            # Rough estimate: Assume saved tokens were primarily input tokens
            cost_saved = tokens_saved * input_cost_per_token

        # Assemble token stats text using the extracted session variables
        token_stats_text = Text.assemble(
            "Tokens: ",
            ("Input: ", "dim cyan"),
            (f"{session_input_tokens:,}", "cyan"),
            " | ",
            ("Output: ", "dim magenta"),
            (f"{session_output_tokens:,}", "magenta"),
            " | ",
            ("Total: ", "dim white"),
            (f"{session_input_tokens + session_output_tokens:,}", "white"),
            " | ",
            ("Cost: ", "dim yellow"),
            (f"${session_total_cost:.4f}", "yellow"),
        )

        cache_stats_text = Text()
        if total_lookups > 0 or tokens_saved > 0:
            cache_stats_text = Text.assemble(
                "\nCache: ",
                ("Hits: ", "dim green"),
                (f"{cache_hits}", "green"),
                " | ",
                ("Misses: ", "dim yellow"),
                (f"{cache_misses}", "yellow"),
                " | ",
                ("Hit Rate: ", "dim green"),
                (f"{hit_rate:.1f}%", "green"),
                " | ",
                ("Tokens Saved: ", "dim green"),
                (f"{tokens_saved:,}", "green bold"),
                " | ",
                ("Cost Saved: ≈$", "dim green"),
                (f"{cost_saved:.4f}", "green bold"),  # Indicate estimate
            )

        # Combine and create panel
        stats_group = Group(token_stats_text, cache_stats_text)
        final_stats_panel = Panel(
            stats_group,
            title="Final Stats (This Query)",
            border_style="green",
            padding=(0, 1),  # Less vertical padding
        )
        target_console.print(final_stats_panel)

    async def close(self):
        """Gracefully shut down the client and release resources."""
        log.info("Closing MCPClient...")
        # Stop monitoring first
        if hasattr(self, "server_monitor"):
            await self.server_monitor.stop_monitoring()

        # Stop local discovery listener
        await self.stop_local_discovery_monitoring()  # ADDED CALL

        # Close server manager (handles sessions/processes)
        if hasattr(self, "server_manager"):
            await self.server_manager.close()

        # Close provider clients
        async def _safe_aclose(client):
            if client and hasattr(client, "aclose"):
                try:
                    await client.aclose()
                except Exception as e:
                    log.warning(f"Error closing client {type(client).__name__}: {e}")

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
        if hasattr(self, "tool_cache") and hasattr(self.tool_cache, "close"):
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
    client = None  # Initialize client to None
    safe_console = get_safe_console()
    max_shutdown_timeout = 10

    # --- Shared Setup ---
    try:
        log.info("Initializing MCPClient...")
        client = MCPClient()  # Instantiation inside the try block
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
                app.state.mcp_client = client  # Make client accessible
                log.info("MCPClient setup complete for Web UI.")
                yield  # Server runs here
                # --- Shutdown ---
                log.info("FastAPI lifespan shutdown: Closing MCPClient...")
                if app.state.mcp_client:
                    await app.state.mcp_client.close()
                log.info("MCPClient closed.")

            # Define FastAPI app within this scope
            app = FastAPI(title="Ultimate MCP Client API", lifespan=lifespan)
            global web_app  # Allow modification of global var
            web_app = app  # Assign to global var for uvicorn

            # Make client accessible to endpoints via dependency injection
            async def get_mcp_client(request: Request) -> MCPClient:
                if not hasattr(request.app.state, "mcp_client") or request.app.state.mcp_client is None:
                    # This should ideally not happen due to lifespan
                    log.error("MCPClient not found in app state during request!")
                    raise HTTPException(status_code=500, detail="MCP Client not initialized")
                return request.app.state.mcp_client

            # --- CORS Middleware ---
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Allow all for development, restrict in production
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
                        "anthropic_api_key",
                        "openai_api_key",
                        "gemini_api_key",
                        "grok_api_key",
                        "deepseek_api_key",
                        "mistral_api_key",
                        "groq_api_key",
                        "cerebras_api_key",
                        "openrouter_api_key",
                        "servers",  # Servers have their own endpoint
                        "decouple_instance",
                        "dotenv_path",  # Internal attributes
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
            async def update_config(update_request: ConfigUpdateRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """
                Updates the **running** configuration based on the request.
                Note: Simple settings (keys, URLs, flags, etc.) are NOT persisted
                    to .env by this endpoint. Only 'cache_ttl_mapping' is saved
                    to the YAML configuration file.
                """
                updated_fields = False
                providers_to_reinit = set()
                fields_updated = []
                config_yaml_needs_save = False  # Track if YAML needs saving

                log.debug(f"Received config update request via API: {update_request.model_dump(exclude_unset=True)}")

                for key_alias, value in update_request.model_dump(exclude_unset=True, by_alias=True).items():
                    # Find the corresponding attribute name in the Config class
                    # This handles the alias mapping back from camelCase to snake_case
                    attr_name = key_alias  # Default if no specific mapping needed
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
                            if attr_name == "default_model":
                                mcp_client.current_model = value
                            elif attr_name == "history_size":
                                if isinstance(value, int) and value > 0:
                                    mcp_client.history = History(max_entries=value)  # Recreate history
                                else:
                                    log.warning(f"Invalid history_size '{value}' ignored.")
                            elif attr_name == "cache_ttl_mapping":
                                if isinstance(value, dict):
                                    # Update the cache instance as well
                                    if mcp_client.tool_cache:
                                        mcp_client.tool_cache.ttl_mapping = value.copy()
                                    config_yaml_needs_save = True  # Mark YAML for saving
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
                    return {
                        "message": f"Configuration updated successfully for fields: {', '.join(fields_updated)} (Simple settings are session-only)"
                    }
                else:
                    return {"message": "No configuration changes applied."}

            @app.put("/api/model", status_code=200)
            async def set_model_api(req: SetModelRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Sets the active model for subsequent queries."""
                log.info(f"API request to set model to: {req.model}")
                try:
                    # Use the shared method in MCPClient
                    changed = await mcp_client.set_active_model(req.model, source="API")
                    if changed:
                        message = f"Active model set to {req.model}."
                    else:
                        message = f"Model was already set to {req.model}."

                    return {"message": message, "currentModel": mcp_client.current_model}
                except Exception as e:
                    log.error(f"Error setting model via API: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Failed to set model: {str(e)}") from e

            @app.get("/api/models")
            async def list_models_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Lists available models, grouped by provider, based on cost data and initialized clients."""
                models_by_provider = {}
                initialized_providers = set()

                # Check which providers have initialized clients
                if mcp_client.anthropic:
                    initialized_providers.add(Provider.ANTHROPIC.value)
                if mcp_client.openai_client:
                    initialized_providers.add(Provider.OPENAI.value)
                if mcp_client.gemini_client:
                    initialized_providers.add(Provider.GEMINI.value)
                if mcp_client.grok_client:
                    initialized_providers.add(Provider.GROK.value)
                if mcp_client.deepseek_client:
                    initialized_providers.add(Provider.DEEPSEEK.value)
                if mcp_client.mistral_client:
                    initialized_providers.add(Provider.MISTRAL.value)
                if mcp_client.groq_client:
                    initialized_providers.add(Provider.GROQ.value)
                if mcp_client.cerebras_client:
                    initialized_providers.add(Provider.CEREBRAS.value)
                # Add OpenRouter check if needed
                # if mcp_client.openrouter_client: initialized_providers.add(Provider.OPENROUTER.value)

                # Group models based on the static map and check if provider is initialized
                for model_name, provider_value in MODEL_PROVIDER_MAP.items():
                    if provider_value not in models_by_provider:
                        models_by_provider[provider_value] = []
                    cost_info = COST_PER_MILLION_TOKENS.get(model_name, {})
                    models_by_provider[provider_value].append(
                        {
                            "name": model_name,
                            "cost_input_per_million": cost_info.get("input"),
                            "cost_output_per_million": cost_info.get("output"),
                            "is_active": provider_value in initialized_providers,  # Indicate if provider client is ready
                        }
                    )

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
                        "toolsCount": tools_count,
                    }
                    server_list.append(server_data)

                # Sort alphabetically by name
                server_list.sort(key=lambda s: s["name"])
                return server_list

            @app.post("/api/servers", status_code=201)
            async def add_server_api(req: ServerAddRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Adds a new server configuration."""
                if req.name in mcp_client.config.servers:
                    raise HTTPException(status_code=409, detail=f"Server name '{req.name}' already exists")

                args_list = req.argsString.split() if req.argsString else []
                new_server_config = ServerConfig(
                    name=req.name,
                    type=req.type,
                    path=req.path,
                    args=args_list,
                    enabled=True,
                    auto_start=False,
                    description=f"Added via Web UI ({req.type.value})",
                )
                mcp_client.config.servers[req.name] = new_server_config
                await mcp_client.config.save_async()  # Save YAML
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
                await mcp_client.config.save_async()  # Save YAML
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
                        final_name = server_config.name  # Name might change during connect
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
                await mcp_client.config.save_async()  # Save YAML
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

            @app.get("/api/tools")  # Keep the original endpoint for now
            async def list_tools_api(
                server_name: Optional[str] = None,
                provider_for_format: Optional[str] = None,  # ADD THIS QUERY PARAM
                mcp_client: MCPClient = Depends(get_mcp_client),
            ):
                """Lists available tools, optionally filtered by server_name.
                If provider_for_format is given, returns tools formatted for that provider."""

                if provider_for_format:  # If a provider is specified for formatting
                    log.info(f"API: Request to list tools formatted for provider: {provider_for_format}")
                    # Ensure provider_for_format is a valid provider string if necessary
                    try:
                        Provider(provider_for_format.lower())  # Validate
                    except ValueError as e:
                        raise HTTPException(status_code=400, detail=f"Invalid provider_for_format: {provider_for_format}") from e

                    formatted_tools = mcp_client._format_tools_for_provider(provider_for_format.lower())
                    if formatted_tools is None:
                        return []  # Or an appropriate empty response
                    # Log a sample of what's being returned for verification
                    log.debug(
                        f"API: Returning {len(formatted_tools)} tools formatted for '{provider_for_format}'. Sample: {str(formatted_tools[:2])[:500]}"
                    )
                    # Also log the sanitized_to_original map that was just populated
                    s_to_o_map_snapshot = mcp_client.server_manager.sanitized_to_original.copy()
                    log.debug(f"API: Current sanitized_to_original map (size {len(s_to_o_map_snapshot)}): {str(s_to_o_map_snapshot)[:1000]}...")
                    return formatted_tools  # Return the directly formatted list
                else:
                    # Original logic for generic tool listing
                    tools_list = []
                    sorted_tools = sorted(mcp_client.server_manager.tools.values(), key=lambda t: t.name)
                    for tool in sorted_tools:
                        if server_name is None or tool.server_name == server_name:
                            tool_data = {  # ... existing generic tool_data ... }
                                "name": tool.name,  # Original MCP name
                                "description": tool.description,
                                "server_name": tool.server_name,
                                "input_schema": tool.input_schema,
                                "call_count": tool.call_count,
                                "avg_execution_time": tool.avg_execution_time,
                                "last_used": tool.last_used.isoformat() if isinstance(tool.last_used, datetime) else None,
                            }
                            tools_list.append(tool_data)
                    return tools_list

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
                server_name: Optional[str] = None,  # Add optional query parameter
                mcp_client: MCPClient = Depends(get_mcp_client),
            ):
                """Lists available resources, optionally filtered by server_name."""
                resources_list = []
                sorted_resources = sorted(mcp_client.server_manager.resources.values(), key=lambda r: r.name)
                for resource in sorted_resources:
                    if server_name is None or resource.server_name == server_name:
                        resource_data = {
                            "name": resource.name,
                            "description": resource.description,
                            "server_name": resource.server_name,
                            "template": resource.template,
                            "call_count": resource.call_count,
                            "last_used": resource.last_used.isoformat() if isinstance(resource.last_used, datetime) else None,
                        }
                        resources_list.append(resource_data)
                return resources_list

            @app.get("/api/prompts")
            async def list_prompts_api(
                server_name: Optional[str] = None,  # Add optional query parameter
                mcp_client: MCPClient = Depends(get_mcp_client),
            ):
                """Lists available prompts, optionally filtered by server_name."""
                prompts_list = []
                sorted_prompts = sorted(mcp_client.server_manager.prompts.values(), key=lambda p: p.name)
                for prompt in sorted_prompts:
                    if server_name is None or prompt.server_name == server_name:
                        prompt_data = {
                            "name": prompt.name,
                            "description": prompt.description,
                            "server_name": prompt.server_name,
                            "template": prompt.template,
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
                tool_short_name = req.tool_name.split(":")[-1]
                mcp_client.safe_print(f"{EMOJI_MAP['server']} API executing [bold]{tool_short_name}[/] via {tool.server_name}...")
                try:
                    start_time = time.time()
                    result: CallToolResult = await mcp_client.execute_tool(tool.server_name, req.tool_name, req.params)
                    latency = (time.time() - start_time) * 1000

                    # Prepare response content safely for JSON
                    content_to_return = None
                    if result.content is not None:
                        try:
                            # Attempt to serialize complex content, fallback to string
                            _ = json.dumps(result.content)  # Test serialization
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
                        mcp_client.safe_print(
                            f"{EMOJI_MAP['success']} API Tool Result [bold]{tool_short_name}[/] ({result_tokens:,} tokens, {latency:.0f}ms)"
                        )

                    return {"isError": result.isError, "content": content_to_return, "latency_ms": latency}

                except asyncio.CancelledError as e:
                    mcp_client.safe_print(f"[yellow]API Tool execution [bold]{tool_short_name}[/] cancelled.[/]")
                    raise HTTPException(status_code=499, detail="Tool execution cancelled by client") from e
                except Exception as e:
                    mcp_client.safe_print(f"{EMOJI_MAP['failure']} API Tool Execution Failed [bold]{tool_short_name}[/]: {str(e)}")
                    log.error(f"Error executing tool '{req.tool_name}' via API: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Tool execution failed: {str(e)}") from e

            @app.post("/api/agent/start", status_code=202)  # 202 Accepted for background task
            async def start_agent_api(req: AgentStartRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Starts the self-driving agent with a given goal."""
                if not hasattr(mcp_client, "start_agent_task"):  # Ensure method exists
                    log.error("API: start_agent_task method not found on mcp_client.")
                    raise HTTPException(status_code=501, detail="Agent functionality not fully implemented in server.")

                log.info(f"API: Request to start agent. Goal: '{req.goal}', Max Loops: {req.max_loops}, Model: {req.llm_model or 'default'}")
                result = await mcp_client.start_agent_task(goal=req.goal, max_loops=req.max_loops, llm_model_override=req.llm_model)
                if not result.get("success"):
                    # Determine appropriate status code based on message
                    status_code = 409 if "already running" in result.get("message", "").lower() else 500
                    log.warning(f"API: Failed to start agent: {result.get('message')}")
                    raise HTTPException(status_code=status_code, detail=result.get("message", "Failed to start agent."))
                log.info(f"API: Agent task successfully initiated. Current status: {result.get('status')}")
                # Return current agent status along with success message
                return {"message": result.get("message"), "agent_status": mcp_client.get_agent_status()}

            @app.post("/api/agent/stop", status_code=200)
            async def stop_agent_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Stops the currently running agent task."""
                if not hasattr(mcp_client, "stop_agent_task"):
                    log.error("API: stop_agent_task method not found on mcp_client.")
                    raise HTTPException(status_code=501, detail="Agent functionality not fully implemented.")

                log.info("API: Request to stop agent.")
                result = await mcp_client.stop_agent_task()
                # stop_agent_task returns success even if no task was running to stop.
                # The message indicates the outcome.
                log.info(f"API: Agent stop process finished. Result message: {result.get('message')}")
                return {"message": result.get("message"), "agent_status": mcp_client.get_agent_status()}

            @app.post("/api/agent/inject_thought", status_code=202)
            async def inject_thought_into_agent_api(req: AgentInjectThoughtRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Injects a thought (guidance/observation) into a running agent."""
                if not mcp_client.agent_loop_instance or not mcp_client.agent_task or mcp_client.agent_task.done():
                    raise HTTPException(status_code=409, detail="Agent is not currently running or available.")

                if not hasattr(mcp_client.agent_loop_instance, "inject_manual_thought"):
                    log.error("API: AgentMasterLoop instance is missing 'inject_manual_thought' method.")
                    raise HTTPException(status_code=501, detail="Agent guidance injection not supported by current agent version.")

                log.info(f"API: Injecting thought into agent: '{req.content[:50]}...' (Type: {req.thought_type})")
                try:
                    # The inject_manual_thought method in AML should handle UMS calls and potentially update agent's state or plan.
                    success_inject = await mcp_client.agent_loop_instance.inject_manual_thought(
                        content=req.content,
                        thought_type=req.thought_type if req.thought_type else "user_guidance",  # AML defaults if not provided
                    )
                    if success_inject:
                        return {"success": True, "message": "Guidance injected into agent's reasoning process."}
                    else:
                        # This else implies inject_manual_thought returned False explicitly
                        raise HTTPException(status_code=500, detail="Agent failed to process injected thought.")
                except AttributeError as ae:  # If method doesn't exist on old agent state
                    log.error(f"API: Missing 'inject_manual_thought' on agent loop: {ae}")
                    raise HTTPException(status_code=501, detail="Agent version does not support thought injection.") from ae
                except Exception as e:
                    log.error(f"API: Error injecting thought into agent: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail=f"Error injecting thought: {str(e)}") from e

            @app.get("/api/agent/status")  # No response_model needed here as it's dynamic
            async def get_agent_status_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Gets the current status of the self-driving agent, including plan and goal stack."""
                if not hasattr(mcp_client, "get_agent_status"):
                    log.error("API: get_agent_status method not found on mcp_client.")
                    raise HTTPException(status_code=501, detail="Agent functionality not fully implemented.")

                # get_agent_status in MCPClient should now be expanded to include
                # current_plan_from_agent and agent_current_operational_goal_stack_summary
                status = mcp_client.get_agent_status()
                log.debug(
                    f"API: Polled agent status. Status: {status.get('status')}, Loop: {status.get('current_loop')}/{status.get('max_loops')}, Plan steps: {len(status.get('current_plan_from_agent', []))}, Goal Stack Depth: {len(status.get('agent_current_operational_goal_stack_summary', []))}"
                )
                return status

            @app.get("/api/conversation")
            async def get_conversation_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Gets the current state of the conversation graph."""
                node = mcp_client.conversation_graph.current_node
                nodes_data_serializable = []  # Store serializable node data

                # Ensure nodes are serializable using jsonable_encoder
                for n_id, n_obj in mcp_client.conversation_graph.nodes.items():
                    try:
                        # Convert node object to dict first using its method
                        node_dict = n_obj.to_dict()
                        # Then encode that dict using jsonable_encoder
                        serializable_node_data = jsonable_encoder(node_dict)
                        nodes_data_serializable.append(serializable_node_data)
                    except Exception as e:
                        log.error(f"Node data for {n_id} not serializable via jsonable_encoder: {e}", exc_info=True)
                        # Skip adding problematic node data

                try:
                    # Explicitly encode the messages list using jsonable_encoder
                    # This should correctly handle TypedDicts within the list structure
                    serializable_messages = jsonable_encoder(node.messages)

                    # Return the fully encoded data structure
                    return {
                        "currentNodeId": node.id,
                        "currentNodeName": node.name,
                        "messages": serializable_messages,  # Use the encoded list
                        "model": node.model or mcp_client.config.default_model,
                        "nodes": nodes_data_serializable,  # Use the encoded list of nodes
                    }
                except Exception as e:  # Catch potential errors during jsonable_encoder itself
                    log.error(f"Conversation data not serializable via jsonable_encoder: {e}", exc_info=True)
                    raise HTTPException(status_code=500, detail="Internal error: Conversation state not serializable") from e

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
                            if matched_node:  # Ambiguous prefix
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
                cleared_node = mcp_client.conversation_graph.get_node(current_node_id)  # Get potentially updated node
                return {
                    "message": f"Messages cleared for node {cleared_node.name}",
                    "currentNodeId": cleared_node.id,
                    "messages": cleared_node.messages,
                }

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
                        "messageCount": len(node_obj.messages),
                    }
                    # Validate and potentially add using the Pydantic model
                    try:
                        graph_nodes.append(GraphNodeData(**node_data))
                    except ValidationError as e:
                        log.warning(f"Skipping invalid graph node data for node {node_id}: {e}")

                return graph_nodes  # Return the list of validated Pydantic models

            @app.put("/api/conversation/nodes/{node_id}/rename")
            async def rename_conversation_node_api(node_id: str, req: NodeRenameRequest, mcp_client: MCPClient = Depends(get_mcp_client)):
                """Renames a specific conversation node/branch."""
                node = mcp_client.conversation_graph.get_node(node_id)
                if not node:
                    raise HTTPException(status_code=404, detail=f"Node ID '{node_id}' not found.")
                if node.id == "root":  # Prevent renaming root
                    raise HTTPException(status_code=400, detail="Cannot rename the root node.")

                old_name = node.name
                new_name = req.new_name.strip()
                if not new_name:
                    raise HTTPException(status_code=400, detail="New name cannot be empty.")

                if old_name == new_name:
                    return {"message": "Node name unchanged.", "node_id": node_id, "new_name": new_name}

                node.name = new_name
                node.modified_at = datetime.now()  # Update modification time
                await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                log.info(f"Renamed conversation node '{node_id}' from '{old_name}' to '{new_name}' via API.")
                return {"message": f"Node '{node_id}' renamed successfully.", "node_id": node_id, "new_name": new_name}

            @app.get("/api/usage")
            async def get_token_usage(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Gets the token usage and cost statistics for the current session."""
                hit_rate = 0.0
                cache_hits = getattr(mcp_client, "cache_hit_count", 0)
                cache_misses = getattr(mcp_client, "cache_miss_count", 0)
                total_cache_lookups = cache_hits + cache_misses
                if total_cache_lookups > 0:
                    hit_rate = (cache_hits / total_cache_lookups) * 100

                cost_saved = 0.0
                tokens_saved = getattr(mcp_client, "tokens_saved_by_cache", 0)
                if tokens_saved > 0:
                    model_cost_info = COST_PER_MILLION_TOKENS.get(mcp_client.current_model, {})
                    input_cost_per_token = model_cost_info.get("input", 0) / 1_000_000
                    cost_saved = tokens_saved * input_cost_per_token * 0.9  # 90% saving estimate

                usage_data = {
                    "input_tokens": getattr(mcp_client, "session_input_tokens", 0),
                    "output_tokens": getattr(mcp_client, "session_output_tokens", 0),
                    "total_tokens": getattr(mcp_client, "session_input_tokens", 0) + getattr(mcp_client, "session_output_tokens", 0),
                    "total_cost": getattr(mcp_client, "session_total_cost", 0.0),
                    "cache_metrics": {
                        "hit_count": cache_hits,
                        "miss_count": cache_misses,
                        "hit_rate_percent": hit_rate,
                        "tokens_saved": tokens_saved,
                        "estimated_cost_saved": cost_saved,
                    },
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
                    summary = await mcp_client.summarize_conversation(target_tokens=target_length, model=summarization_model)
                    if summary is None:  # Check if summarization failed internally
                        raise RuntimeError("Summarization failed or returned no content.")

                    # Replace messages with summary
                    summary_system_message = f"The preceding conversation up to this point has been summarized:\n\n---\n{summary}\n---"
                    mcp_client.conversation_graph.current_node.messages = [InternalMessage(role="system", content=summary_system_message)]
                    final_tokens = await mcp_client.count_tokens()
                    await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))

                    log.info(f"Optimization complete. Tokens: {initial_tokens} -> {final_tokens}")
                    return {"message": "Optimization complete", "initialTokens": initial_tokens, "finalTokens": final_tokens}

                except Exception as e:
                    log.error(f"Error optimizing conversation via API: {e}", exc_info=True)
                    mcp_client.conversation_graph.current_node.messages = original_messages  # Restore on failure
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
                    content_str = content_bytes.decode("utf-8")
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
                q: str,  # Query parameter 'q'
                limit: int = 10,  # Optional limit parameter
                mcp_client: MCPClient = Depends(get_mcp_client),
            ):
                """Searches the conversation history."""
                if not q:
                    raise HTTPException(status_code=400, detail="Search query 'q' cannot be empty.")
                if limit <= 0:
                    limit = 10  # Default limit if invalid

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
                        except Exception:
                            pass  # Ignore errors counting disk cache

                # Get LLM prompt cache stats (session-level)
                prompt_cache_stats = await get_token_usage(mcp_client)  # Reuses the logic from /api/usage
                prompt_cache_stats = prompt_cache_stats.get("cache_metrics", {})  # Extract cache part

                return {"tool_cache": tool_cache_stats, "prompt_cache": prompt_cache_stats}

            @app.post("/api/cache/reset_stats")
            async def reset_cache_stats_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Resets the session-level prompt cache statistics."""
                # Explicitly check and reset attributes
                if hasattr(mcp_client, "cache_hit_count"):
                    mcp_client.cache_hit_count = 0
                if hasattr(mcp_client, "cache_miss_count"):
                    mcp_client.cache_miss_count = 0
                if hasattr(mcp_client, "tokens_saved_by_cache"):
                    mcp_client.tokens_saved_by_cache = 0
                log.info("Reset LLM prompt cache statistics via API.")
                return {"message": "LLM prompt cache statistics reset successfully"}

            @app.get("/api/cache/entries", response_model=List[CacheEntryDetail])
            async def get_cache_entries_api(mcp_client: MCPClient = Depends(get_mcp_client)):
                """Lists entries currently in the tool cache."""
                if not mcp_client.tool_cache:
                    return []  # Return empty list if caching disabled
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
                    await mcp_client.reload_servers()  # Assumes this method exists and works
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
                    new_config_state = await get_config(mcp_client)  # Reuse the GET endpoint logic
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
                            aborted = True  # Consider it "aborted" from API perspective
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
                    # Use suppress for closing on error
                    with suppress(Exception):
                        await websocket.close(code=1011)
                    return

                await websocket.accept()
                connection_id = str(uuid.uuid4())[:8]
                log.info(f"WebSocket connection accepted (ID: {connection_id}).")
                active_query_task: Optional[asyncio.Task] = None  # Track task specific to this WS connection

                async def send_ws_message(msg_type: str, payload: Any):
                    """Safely send a JSON message over the WebSocket."""
                    try:
                        # Use model_dump() for Pydantic v2 compatibility
                        await websocket.send_json(WebSocketMessage(type=msg_type, payload=payload).model_dump())
                    except (WebSocketDisconnect, RuntimeError) as send_err:  # More specific exceptions
                        # Ignore errors if the socket is already closing/closed
                        log.debug(f"WS-{connection_id}: Failed send (Type: {msg_type}) - likely disconnected: {send_err}")
                    except Exception as send_err:
                        # Log other unexpected send errors
                        log.warning(f"WS-{connection_id}: Unexpected error sending WS message (Type: {msg_type}): {send_err}")

                async def send_command_response(success: bool, message: str, data: Optional[Dict] = None):
                    """Sends a standardized response to a command."""
                    payload = {"success": success, "message": message}
                    payload.update(data or {})
                    await send_ws_message("command_response", payload)

                async def send_error_response(message: str, cmd: Optional[str] = None):
                    """Logs an error and sends an error message to the client."""
                    log_msg = f"WS-{connection_id} Error: {message}" + (f" (Cmd: /{cmd})" if cmd else "")
                    log.warning(log_msg)
                    await send_ws_message("error", {"message": message})
                    # Also send a command_response failure if it was triggered by a command
                    if cmd is not None:
                        await send_command_response(False, message)

                # --- Consumer coroutine for processing the query stream ---
                async def _consume_query_stream(query_text: str):
                    """Consumes the query stream and sends updates over WebSocket."""
                    nonlocal active_query_task  # Allow modification of outer scope variable
                    try:
                        # Iterate over the standardized events yielded by the stream wrapper
                        # Call process_streaming_query without model - it uses mcp_client.current_model
                        async for chunk_type, chunk_data in mcp_client._stream_wrapper(mcp_client.process_streaming_query(query_text)):
                            # Forward relevant events to the WebSocket client
                            if chunk_type == "text":
                                await send_ws_message("text_chunk", chunk_data)
                            elif chunk_type == "status":
                                # Convert potentially marked-up string to plain text for basic WS status
                                plain_status_text = Text.from_markup(str(chunk_data)).plain
                                await send_ws_message("status", plain_status_text)
                            elif chunk_type == "error":
                                await send_ws_message("error", {"message": str(chunk_data)})
                            elif chunk_type == "final_stats":
                                # Send final usage and completion status after stream completes
                                # Use the get_token_usage function to get current session stats
                                usage_data = await get_token_usage(mcp_client)
                                await send_ws_message("token_usage", usage_data)
                                await send_ws_message("query_complete", {"stop_reason": chunk_data.get("stop_reason", "unknown")})
                            elif chunk_type == "tool_call_start":
                                # Forward tool start event if UI needs it
                                await send_ws_message("tool_call_start", chunk_data)
                            elif chunk_type == "tool_call_input_chunk":
                                # Forward input chunks if UI needs them
                                await send_ws_message("tool_call_input_chunk", chunk_data)
                            elif chunk_type == "tool_call_end":
                                # Forward tool end event if UI needs it
                                await send_ws_message("tool_call_end", chunk_data)

                    except asyncio.CancelledError:
                        log.info(f"WS-{connection_id}: Query task cancelled.")
                        # Send status update indicating cancellation
                        await send_ws_message("status", "[yellow]Request Aborted by User.[/]")
                    except Exception as e:
                        error_msg = f"Error processing query stream: {str(e)}"
                        log.error(f"WS-{connection_id}: {error_msg}", exc_info=True)
                        # Send error message to the client
                        await send_ws_message("error", {"message": error_msg})
                    finally:
                        # --- Cleanup specific to this query task ---
                        task_being_cleaned = active_query_task  # Capture current task locally
                        active_query_task = None  # Clear task reference for this WS connection

                        # Clear global task reference ONLY if it was THIS task
                        # Check the task object itself for safety before clearing
                        if mcp_client.current_query_task and mcp_client.current_query_task is task_being_cleaned:
                            mcp_client.current_query_task = None
                        log.debug(f"WS-{connection_id}: Query consuming task finished.")
                        # query_complete message is sent via 'final_stats' event handling now

                # --- Main WebSocket Receive Loop ---
                try:
                    while True:
                        raw_data = await websocket.receive_text()
                        try:
                            data = json.loads(raw_data)
                            # Use model_validate for Pydantic v2
                            message = WebSocketMessage.model_validate(data)
                            log.debug(f"WS-{connection_id} Received: Type={message.type}, Payload={str(message.payload)[:100]}...")

                            # --- Handle Query ---
                            if message.type == "query":
                                query_text = str(message.payload or "").strip()
                                if not query_text:
                                    continue  # Ignore empty queries
                                if active_query_task and not active_query_task.done():
                                    await send_error_response("Previous query still running.")
                                    continue

                                # Reset session stats before starting the new query task
                                mcp_client.session_input_tokens = 0
                                mcp_client.session_output_tokens = 0
                                mcp_client.session_total_cost = 0.0
                                mcp_client.cache_hit_count = 0
                                mcp_client.tokens_saved_by_cache = 0

                                # Create the task using the consumer coroutine
                                query_task = asyncio.create_task(_consume_query_stream(query_text), name=f"ws_query_{connection_id}")
                                active_query_task = query_task

                                # Link to client instance ONLY if no other task is running (for global abort)
                                if not mcp_client.current_query_task or mcp_client.current_query_task.done():
                                    mcp_client.current_query_task = query_task
                                else:
                                    log.warning("Another query task is already running globally, global abort might affect wrong task.")

                                # Task runs in background, loop continues to listen for more messages/commands

                            # --- Handle Command ---
                            elif message.type == "command":
                                command_str = str(message.payload).strip()
                                if command_str.startswith("/"):
                                    try:
                                        # Use shlex to handle potential quoting in args
                                        parts = shlex.split(command_str[1:])
                                        if not parts:
                                            continue  # Ignore empty command
                                        cmd = parts[0].lower()
                                        args = shlex.join(parts[1:]) if len(parts) > 1 else ""
                                    except ValueError as e:
                                        await send_error_response(f"Error parsing command: {e}", cmd=command_str[:10])  # Pass partial command
                                        continue

                                    log.info(f"WS-{connection_id} processing command: /{cmd} {args}")
                                    try:
                                        # --- Implement Command Logic ---
                                        if cmd == "clear":
                                            mcp_client.conversation_graph.current_node.messages = []
                                            # Optionally switch back to root, or stay on cleared node
                                            # mcp_client.conversation_graph.set_current_node("root")
                                            await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                                            await send_command_response(
                                                True, "Conversation branch cleared.", {"messages": []}
                                            )  # Send cleared messages
                                        elif cmd == "model":
                                            if args:
                                                # Use the set_active_model method which handles provider logic
                                                success = await mcp_client.set_active_model(args, source="WebSocket")
                                                if success:
                                                    await send_command_response(True, f"Model set to: {args}", {"currentModel": args})
                                                else:
                                                    # Error message printed by set_active_model if provider unknown
                                                    await send_command_response(
                                                        False,
                                                        f"Failed to set model to {args} (check logs).",
                                                        {"currentModel": mcp_client.current_model},
                                                    )
                                            else:
                                                await send_command_response(
                                                    True, f"Current model: {mcp_client.current_model}", {"currentModel": mcp_client.current_model}
                                                )
                                        elif cmd == "fork":
                                            new_node = mcp_client.conversation_graph.create_fork(name=args if args else None)
                                            mcp_client.conversation_graph.set_current_node(new_node.id)
                                            await mcp_client.conversation_graph.save(str(mcp_client.conversation_graph_file))
                                            await send_command_response(
                                                True,
                                                f"Created and switched to branch: {new_node.name}",
                                                {"newNodeId": new_node.id, "newNodeName": new_node.name, "messages": new_node.messages},
                                            )
                                        elif cmd == "checkout":
                                            if not args:
                                                await send_error_response("Usage: /checkout NODE_ID_or_Prefix", cmd)
                                                continue
                                            node_id_prefix = args
                                            node_to_checkout = None
                                            matches = [
                                                n for n_id, n in mcp_client.conversation_graph.nodes.items() if n_id.startswith(node_id_prefix)
                                            ]
                                            if len(matches) == 1:
                                                node_to_checkout = matches[0]
                                            elif len(matches) > 1:
                                                await send_error_response(f"Ambiguous node ID prefix '{node_id_prefix}'", cmd)
                                                continue

                                            if node_to_checkout and mcp_client.conversation_graph.set_current_node(node_to_checkout.id):
                                                await send_command_response(
                                                    True,
                                                    f"Switched to branch: {node_to_checkout.name}",
                                                    {"currentNodeId": node_to_checkout.id, "messages": node_to_checkout.messages},
                                                )
                                            else:
                                                await send_error_response(f"Node ID prefix '{node_id_prefix}' not found.", cmd)
                                        elif cmd == "apply_prompt":
                                            if not args:
                                                await send_error_response("Usage: /apply_prompt <prompt_name>", cmd)
                                                continue
                                            success = await mcp_client.apply_prompt_to_conversation(args)
                                            if success:
                                                await send_command_response(
                                                    True, f"Applied prompt: {args}", {"messages": mcp_client.conversation_graph.current_node.messages}
                                                )
                                            else:
                                                await send_error_response(f"Prompt not found: {args}", cmd)
                                        elif cmd == "abort":
                                            log.info(f"WS-{connection_id} received abort command.")
                                            task_to_cancel = active_query_task  # Target the task specific to this connection
                                            if task_to_cancel and not task_to_cancel.done():
                                                was_cancelled = task_to_cancel.cancel()  # Request cancellation  # noqa: F841
                                                await send_command_response(True, "Abort signal sent.")
                                            else:
                                                await send_command_response(False, "No active query for this connection.")
                                        # Add other command handlers here as needed
                                        # Example:
                                        # elif cmd == "some_other_command":
                                        #      # ... logic ...
                                        #      await send_command_response(True, "Did something.")
                                        else:
                                            await send_command_response(False, f"Command '/{cmd}' not supported via WebSocket.")
                                    except Exception as cmd_err:
                                        # Catch errors during command execution
                                        await send_error_response(f"Error executing '/{cmd}': {cmd_err}", cmd)
                                        log.error(f"WS-{connection_id} Cmd Error /{cmd}: {cmd_err}", exc_info=True)
                                else:
                                    await send_error_response("Invalid command format (must start with '/').", None)

                            # Ignore other message types for now

                        except (json.JSONDecodeError, ValidationError) as e:
                            log.warning(f"WS-{connection_id} invalid message: {raw_data[:100]}... Error: {e}")
                            await send_ws_message("error", {"message": "Invalid message format."})
                        except WebSocketDisconnect:
                            raise  # Re-raise to be caught by the outer handler
                        except Exception as e:
                            log.error(f"WS-{connection_id} error processing message: {e}", exc_info=True)
                            # Use suppress correctly when sending error back
                            with suppress(Exception):
                                await send_ws_message("error", {"message": f"Internal error: {str(e)}"})

                # --- Outer Exception Handling for the WebSocket Connection ---
                except WebSocketDisconnect:
                    log.info(f"WebSocket connection closed (ID: {connection_id}).")
                except Exception as e:
                    log.error(f"WS-{connection_id} unexpected handler error: {e}", exc_info=True)
                    # Use suppress correctly when trying to close the socket on error
                    with suppress(Exception):
                        await websocket.close(code=1011)
                finally:
                    # --- Final Cleanup for this WebSocket Connection ---
                    log.debug(f"WS-{connection_id}: Cleaning up WebSocket handler resources.")
                    # Ensure any active query task *for this specific connection* is cancelled
                    if active_query_task and not active_query_task.done():
                        log.warning(f"WS-{connection_id} closing, cancelling associated active query task.")
                        active_query_task.cancel()
                        # Optionally await cancellation with timeout, or just signal and move on
                        with suppress(asyncio.TimeoutError, Exception):
                            await asyncio.wait_for(active_query_task, timeout=0.5)

                    # Clear global task reference ONLY if it was the task associated with this connection
                    if mcp_client.current_query_task and mcp_client.current_query_task is active_query_task:
                        log.debug(f"WS-{connection_id}: Clearing global query task reference.")
                        mcp_client.current_query_task = None
                    log.debug(f"WS-{connection_id}: WebSocket cleanup complete.")

            # --- Static File Serving ---
            if serve_ui_file:
                ui_file = Path(__file__).parent / "mcp_client_ui.html"  # Look relative to script
                if ui_file.exists():
                    log.info(f"Serving static UI file from {ui_file.resolve()}")

                    @app.get("/", response_class=FileResponse, include_in_schema=False)
                    async def serve_html():
                        # Added cache control headers to encourage browser reloading for development
                        headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache", "Expires": "0"}
                        return FileResponse(str(ui_file.resolve()), headers=headers)
                else:
                    log.warning(f"UI file {ui_file} not found. Cannot serve.")

            log.info("Starting Uvicorn server...")
            config = uvicorn.Config(app, host=webui_host, port=webui_port, log_level="info")
            server_instance = uvicorn.Server(config)
            try:
                # This blocks until the server is stopped
                await server_instance.serve()
                log.info("Web UI server shut down normally.")
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    safe_console.print(f"[bold red]ERROR: Could not start Web UI. Port {webui_port} is already in use.[/]")
                    safe_console.print(f"[yellow]Please stop the other process using port {webui_port} or choose a different port using --port.[/]")
                    sys.exit(1)  # Exit directly
                else:
                    log.error(f"Uvicorn server failed with OS error: {e}", exc_info=True)
                    safe_console.print(f"[bold red]Web UI server failed to start (OS Error): {e}[/]")
                    sys.exit(1)  # Exit on other OS errors too
            except Exception as e:  # Catch other potential server errors during serve()
                log.error(f"Uvicorn server failed: {e}", exc_info=True)
                safe_console.print(f"[bold red]Web UI server failed to start: {e}[/]")
                sys.exit(1)  # Exit on other startup errors

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
            query_text = query
            model_to_use = model or client.current_model
            provider_name = client.get_provider_from_model(model_to_use)
            provider_title = provider_name.capitalize() if provider_name else "Unknown Provider"

            safe_console.print(f"[cyan]Processing single query with {provider_title} ({model_to_use})...[/]")
            final_result_text = ""
            query_error: Optional[Exception] = None
            query_cancelled = False
            status_updates_internal: List[str] = []
            client.session_input_tokens = 0
            client.session_output_tokens = 0
            client.session_total_cost = 0.0
            client.cache_hit_count = 0
            client.tokens_saved_by_cache = 0  # Reset stats

            with Status(f"{EMOJI_MAP['processing']} Processing query with {model_to_use}...", console=safe_console, spinner="dots") as status:
                try:
                    query_task = asyncio.create_task(
                        client.process_streaming_query(query_text, model=model_to_use), name=f"single_query-{query_text[:20].replace(' ', '_')}"
                    )
                    client.current_query_task = query_task
                    async for chunk in query_task:
                        if chunk.startswith("@@STATUS@@"):
                            status_updates_internal.append(chunk[len("@@STATUS@@") :].strip())
                        else:
                            final_result_text += chunk
                    await query_task  # Ensure completion/catch errors
                except asyncio.CancelledError:
                    query_cancelled = True
                    log.info("Single query processing cancelled.")
                except Exception as e:
                    query_error = e
                    log.error(f"Error processing single query: {e}", exc_info=True)
                finally:
                    status.stop()
                    client.current_query_task = None

            safe_console.print()  # Blank line
            if query_cancelled:
                if final_result_text:
                    safe_console.print(
                        Panel.fit(Markdown(final_result_text), title=f"Partial Result ({model_to_use}) - Cancelled", border_style="yellow")
                    )
                else:
                    safe_console.print("[yellow]Query cancelled, no result generated.[/yellow]")
            elif query_error:
                safe_console.print(f"[bold red]Error processing query:[/] {str(query_error)}")
                if status_updates_internal:
                    safe_console.print("\n[dim]Status updates during failed query:[/dim]")
                    for line in status_updates_internal:
                        safe_console.print(f"[dim]- {line}[/dim]")
            elif not final_result_text:
                safe_console.print(
                    Panel.fit("[dim]Model returned no text content.[/dim]", title=f"Result ({model_to_use}) - Empty", border_style="yellow")
                )
            else:
                safe_console.print(Panel.fit(Markdown(final_result_text), title=f"Result ({model_to_use})", border_style="green"))

            if not query_error:
                client._print_final_query_stats(safe_console)
            # No explicit return needed, finally block handles cleanup

        elif interactive:  # Check interactive explicitly last
            # --- Run Interactive Mode ---
            await client.interactive_loop()
            # No explicit return needed, finally block handles cleanup

        # --- Specific Server Connection (if requested via --server) ---
        # This should run REGARDLESS of the mode, as it's a startup connection request
        if server:  # Check if the --server argument was provided
            log.info(f"Attempting specific connection to server '{server}' due to --server flag.")
            if server in client.config.servers:
                if server not in client.server_manager.active_sessions:
                    safe_console.print(f"[cyan]Attempting to connect to specific server: {server}...[/]")
                    connected = await client.connect_server(server)  # Use existing method
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
            if client and hasattr(client, "close"):
                try:
                    await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout / 2)
                except Exception:
                    pass
            sys.exit(1)  # Exit with error code
    finally:
        # Ensure cleanup doesn't run if webui failed early OR if lifespan is handling it
        should_cleanup_main = not webui_flag  # Default: cleanup if not webui
        # If webui mode, assume lifespan handles cleanup *unless* we explicitly
        # exited early (e.g., due to OSError above, which calls sys.exit).
        # If sys.exit was called, this finally block might not execute fully anyway.
        # The goal is to avoid double cleanup if lifespan is responsible.
        if webui_flag:
            should_cleanup_main = False  # Lifespan is responsible in normal webui operation/shutdown
            log.info("Web UI mode: Skipping final client cleanup in main_async (handled by lifespan or exited early).")

        if should_cleanup_main and client and hasattr(client, "close"):
            log.info("Performing final cleanup...")
            try:
                # Shorten timeout slightly to avoid long waits on hang
                await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout - 1)
            except asyncio.TimeoutError:
                safe_console.print("[red]Shutdown timed out. Some processes may still be running.[/]")
                if hasattr(client, "server_manager") and hasattr(client.server_manager, "processes"):
                    for name, process in client.server_manager.processes.items():
                        if process and process.returncode is None:
                            try:
                                safe_console.print(f"[yellow]Force killing process: {name}[/]")
                                process.kill()
                            except Exception:
                                pass
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
                        "categories": server.categories,
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
        if reset:
            args = "reset"
        elif edit:
            args = "edit"
        elif show:
            args = "show"
        else:
            args = "show"  # Default action if no flags are given

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
    safe_console = get_safe_console()  # Get safe console here
    try:
        client = MCPClient()
        # Minimal setup needed? Just loading graph might be enough.
        # Load graph if not loaded by constructor
        if not client.conversation_graph.nodes:
            client.conversation_graph = await ConversationGraph.load(str(client.conversation_graph_file))

        target_id = conversation_id or client.conversation_graph.current_node.id
        final_output_path = output_path  # Need default logic here if None

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
        if client and hasattr(client, "close"):  # Check if close method exists
            await client.close()  # Close client if setup was performed


async def _import_conv_async(file_path_str: str):
    """Async implementation for importing."""
    client = None
    safe_console = get_safe_console()  # Get safe console here
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
        if client and hasattr(client, "close"):  # Check if close method exists
            await client.close()  # Close client if setup was performed


async def main_async_agent_mode(
    initial_goal: str,
    max_loops: int,
    llm_model_for_agent: Optional[str],
    server_to_connect: Optional[str],
    verbose_logging_flag: bool,
    cleanup_servers_flag: bool,
):
    """
    Async entry point specifically for running the self-driving agent mode.
    """
    client = None
    safe_console = get_safe_console()
    max_shutdown_timeout = 10

    try:
        log.info("Initializing MCPClient for Agent Mode...")
        client = MCPClient()
        # Perform essential MCPClient setup (connect to UMS server, etc.)
        # Interactive_mode=False as agent is self-driving
        await client.setup(interactive_mode=False)

        # Connect to a specific server if requested (e.g., the UMS server)
        if server_to_connect:
            log.info(f"Agent Mode: Attempting specific connection to server '{server_to_connect}'...")
            if server_to_connect in client.config.servers:
                if server_to_connect not in client.server_manager.active_sessions:
                    connected = await client.connect_server(server_to_connect)
                    if not connected:
                        safe_console.print(f"[bold red]Agent Mode Error: Failed to connect to specified server '{server_to_connect}'. Aborting.[/]")
                        return  # Cannot proceed if essential server connection fails
                else:
                    log.info(f"Agent Mode: Specified server '{server_to_connect}' is already connected.")
            else:
                safe_console.print(f"[bold red]Agent Mode Error: Specified server '{server_to_connect}' not found in configuration. Aborting.[/]")
                return

        # Initialize the agent loop instance (now done within start_agent_task)
        # Start the agent task using the MCPClient method
        start_result = await client.start_agent_task(
            goal=initial_goal,
            max_loops=max_loops,
            llm_model_override=llm_model_for_agent,  # Pass model if specified
        )

        if not start_result.get("success"):
            safe_console.print(f"[bold red]Failed to start agent: {start_result.get('message')}[/]")
            return

        safe_console.print(f"[green]Agent task started for goal: '{initial_goal}'. Max loops: {max_loops}. Check logs/API for progress.[/]")

        # Keep the main_async_agent_mode alive while the agent_task runs,
        # allowing for graceful shutdown via Ctrl+C or API stop.
        if client.agent_task:
            try:
                await client.agent_task  # Wait for the agent task to complete
                log.info("Agent task completed its run.")
            except asyncio.CancelledError:
                log.info("Agent task wait was cancelled (likely due to client.stop_agent_task or main shutdown).")
            except Exception as e:
                log.error(f"Agent task finished with an unexpected error: {e}", exc_info=True)
        else:
            log.error("Agent task was not successfully created or started.")

    except KeyboardInterrupt:
        safe_console.print("\n[yellow]Agent Mode Interrupted, shutting down...[/]")
        if client and client.agent_task and not client.agent_task.done():
            log.info("KeyboardInterrupt: Stopping agent task...")
            await client.stop_agent_task()
    except Exception as main_agent_err:
        safe_console.print(f"[bold red]An unexpected error occurred in agent mode: {main_agent_err}[/]")
        # Log full traceback for unexpected errors
        log.error(f"Unexpected error in main_async_agent_mode: {main_agent_err}", exc_info=True)
    finally:
        if client:
            log.info("Performing final cleanup for agent mode...")
            try:
                # Ensure agent is stopped if it was running and client is shutting down
                if client.agent_task and not client.agent_task.done():
                    log.warning("Client shutting down, ensuring agent task is stopped...")
                    await client.stop_agent_task()
                await asyncio.wait_for(client.close(), timeout=max_shutdown_timeout - 1)
            except asyncio.TimeoutError:
                safe_console.print("[red]Agent mode shutdown timed out. Some processes may still be running.[/]")
            except Exception as close_error:
                log.error(f"Error during final agent mode cleanup: {close_error}", exc_info=True)
        log.info("Agent mode application shutdown complete.")


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
    webui_flag: Annotated[bool, typer.Option("--webui", "-w", help="Launch the Web UI instead of CLI.")] = False,
    webui_host: Annotated[str, typer.Option("--host", "-h", help="Host for the Web UI server.")] = "127.0.0.1",
    webui_port: Annotated[int, typer.Option("--port", "-p", help="Port for the Web UI server.")] = 8880,
    serve_ui_file: Annotated[bool, typer.Option("--serve-ui", help="Serve the default HTML UI file from the current directory.")] = True,
    cleanup_servers: Annotated[bool, typer.Option("--cleanup-servers", help="Test and remove unreachable servers on startup.")] = False,
    agent_mode: Annotated[bool, typer.Option("--agent", "-a", help="Run in self-driving agent mode.")] = False,
    agent_goal: Annotated[Optional[str], typer.Option("--agent-goal", "-g", help="The overall goal for the agent.")] = None,
    agent_max_loops: Annotated[int, typer.Option("--agent-max-loops", "-l", help="Max loops for the agent to run.")] = 50,  # Default to 50
):
    """
    Run the MCP client (Interactive, Single Query, Web UI, Dashboard, or Self-Driving Agent).
    If no mode flag is provided, defaults to interactive mode.
    For agent mode, --agent-goal is required.
    """
    global USE_VERBOSE_SESSION_LOGGING
    if verbose_logging:
        USE_VERBOSE_SESSION_LOGGING = True
        log.setLevel(logging.DEBUG)
        stderr_console.print("[dim]Verbose logging enabled.[/dim]")

    modes_selected = sum([dashboard, interactive, webui_flag, query is not None, agent_mode])
    actual_interactive = interactive

    if modes_selected > 1:
        stderr_console.print(
            "[bold red]Error: Please specify only one primary mode: --interactive, --query, --dashboard, --webui, or --agent.[/bold red]"
        )
        raise typer.Exit(code=1)
    elif modes_selected == 0:
        stderr_console.print("[dim]No mode specified, defaulting to interactive mode.[/dim]")
        actual_interactive = True  # Set interactive if no other mode chosen

    if agent_mode and not agent_goal:
        stderr_console.print("[bold red]Error: --agent-goal is required when using --agent mode.[/bold red]")
        raise typer.Exit(code=1)

    # Determine if we should run the agent_master_loop variant of main_async
    if agent_mode:
        # Call a new async entry point for agent mode
        asyncio.run(
            main_async_agent_mode(
                initial_goal=agent_goal,  # Will not be None due to check above
                max_loops=agent_max_loops,
                llm_model_for_agent=model,  # Pass the --model if specified
                # Pass other relevant flags if agent_master_loop needs them indirectly
                # For now, these are mainly for MCPClient setup
                server_to_connect=server,
                verbose_logging_flag=verbose_logging,
                cleanup_servers_flag=cleanup_servers,
            )
        )
    else:
        # Call the existing main_async for other modes
        asyncio.run(
            main_async(
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
            )
        )


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
        actual_show = True  # Default to showing if no other action specified

    # Execute the async config logic
    asyncio.run(config_async(show=actual_show, edit=edit, reset=reset))


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
        actual_list_all = True  # Default to listing if no other action

    # Execute the async server logic
    asyncio.run(servers_async(search=search, list_all=actual_list_all, json_output=json_output))


@app.command("export")
def export_conv_command(
    conversation_id: Annotated[Optional[str], typer.Option("--id", "-i", help="ID of the conversation branch to export (default: current).")] = None,
    output_path: Annotated[Optional[str], typer.Option("--output", "-o", help="Path to save the exported JSON file.")] = None,
):
    """Export a conversation branch to a JSON file."""
    asyncio.run(_export_conv_async(conversation_id, output_path))


@app.command("import-conv")
def import_conv_command(
    file_path: Annotated[
        Path, typer.Argument(exists=True, file_okay=True, dir_okay=False, readable=True, help="Path to the JSON conversation file to import.")
    ],
):
    """Import a conversation from a JSON file as a new branch."""
    asyncio.run(_import_conv_async(str(file_path)))


# =============================================================================
# Main execution block
# =============================================================================

if __name__ == "__main__":
    if platform.system() == "Windows":
        colorama.init(convert=True)  # Initialize colorama for Windows

    # Directly call the Typer app instance. Typer handles argument parsing
    # and calling the appropriate command function.
    app()
