"""
EideticEngine Agent Master Loop (AML) - v4.3 - Simplified & Robust
====================================================================

This module implements the core orchestration logic for the EideticEngine
AI agent. It manages the primary think-act cycle, interacts with the
Unified Memory System (UMS) via MCPClient, leverages an LLM
for decision-making and planning.

** V4.3 FIXES - SIMPLIFIED & ROBUST AGENT BEHAVIOR **
- **FIXED Infinite Loop Prevention**: Actually raises InfiniteLoopError instead of just logging
- **FIXED Auto-Fix Complexity**: Replaced complex semantic auto-fixing with simple error handling  
- **FIXED Duplicate Prevention**: Actually prevents duplicates instead of just warning
- **SIMPLIFIED Background Tasks**: Removed complex task proliferation, kept only essential logging
- **SIMPLIFIED Error Classification**: Uses simple pattern matching instead of LLM classification
- **FAIL FAST APPROACH**: Clear errors instead of silent "clever" fixes that hide problems

Key Functionalities:
--------------------
*   **Workflow & Context Management:**
    - Creates, manages, and tracks progress within structured workflows (via UMS).
    - Supports sub-workflow execution via a workflow stack (agent-managed).
    - Manages an explicit Goal Stack (agent's view of UMS-managed goals).
    - Gathers rich, multi-faceted context for the LLM decision-making process.
    - Implements structure-aware context truncation.

*   **Planning & Execution:**
    - Maintains an explicit, modifiable plan (agent-managed).
    - Allows the LLM to propose plan updates.
    - Includes a heuristic fallback mechanism to update plan steps.
    - Validates plan steps and detects dependency cycles.
    - Checks action prerequisites (dependencies) before execution (via UMS).
    - Executes tools via the MCPClient, using original MCP tool names.
    - Records detailed action history in UMS.

*   **LLM Interaction & Reasoning:**
    - Constructs detailed prompts for the LLM (LLM sees sanitized tool names).
    - Parses LLM responses (which contain sanitized tool names that MCPClient de-sanitizes
      back to original MCP names before passing to AML).
    - Manages dedicated thought chains in UMS for recording reasoning.

*   **Cognitive & Meta-Cognitive Processes:**
    - All UMS interactions use original MCP tool names directly.
    - Background Cognitive Tasks: Initiates asynchronous UMS tasks.
    - Periodic Meta-cognition: Runs scheduled UMS tasks.
    - Adaptive Thresholds for meta-cognition.
    - Maintenance via UMS.

*   **State & Error Handling:**
    - Persists the complete agent runtime state to JSON.
    - Implements retry logic for tool failures.
    - Tracks consecutive errors.
    - Provides detailed error information to the LLM.
    - Handles graceful shutdown.

────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations  # VERY IMPORTANT: MUST BE THE FIRST LINE

import asyncio
import contextlib
import copy
import dataclasses
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import sys
import tempfile
import time
import uuid
from asyncio import CancelledError
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import aiofiles
import httpx
from anthropic import APIConnectionError, APIStatusError, AsyncAnthropic, RateLimitError
from pydantic import BaseModel, Field, ValidationError

try:
    from mcp.types import CallToolResult
except ImportError:
    if TYPE_CHECKING:
        from mcp.types import CallToolResult
    else:
        CallToolResult = "CallToolResult"  # Fallback to string for runtime if mcp not in sys.path properly

if TYPE_CHECKING:
    from mcp_client_multi import MCPClient  # This will be the actual class at runtime
else:
    MCPClient = "MCPClient"  # Placeholder for static analysis

# Module-level logger for utility functions
log = logging.getLogger(__name__)

# Models explicitly supporting "json_schema" type in response_format for direct structured output
# (primarily newer OpenAI models or native APIs with this feature)
MODELS_CONFIRMED_FOR_OPENAI_JSON_SCHEMA_FORMAT = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o1-preview",
    "o3-mini",
    "deepseek-chat",
    "deepseek-reasoner",
    "gemini-2.5-pro-exp-03-25",
    "gemini-2.5-pro-exp-05-06",
}

# Models that should robustly support "json_object" type in response_format
# (includes older OpenAI and many OpenAI-compatible providers)
MODELS_SUPPORTING_OPENAI_JSON_OBJECT_FORMAT = {
    "gpt-4.1", # Assuming this is like gpt-4-turbo
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o1-preview", # OpenAI, likely json_object
    "o3-mini",    # OpenAI, likely json_object
    "deepseek-chat",
    "deepseek-reasoner",
    "gemini-2.0-flash-lite", # Via OpenAI compatible endpoint
    "gemini-2.0-flash",      # Via OpenAI compatible endpoint
    "gemini-2.0-flash-thinking-exp-01-21", # Via OpenAI compatible endpoint
    "gemini-2.5-pro-exp-03-25",          # Via OpenAI compatible endpoint
    "gemini-2.5-pro-exp-05-06",          # Via OpenAI compatible endpoint
    "grok-3-latest",
    "grok-3-fast-latest",
    "grok-3-mini-latest",
    "grok-3-mini-fast-latest",
    # Mistral native API can take a schema, but if using via generic OpenAI client, json_object is safer unless specific handling for Mistral native schema is done.
    # For now, if Mistral is called via the generic OpenAI client block, json_object is the more general assumption.
    # If you have specific logic for Mistral native API client that sets response_format with a schema, that's different.
    # The current code routes Mistral to the OpenAI-compatible block if its client is an AsyncOpenAI instance.
    "mistral-large-latest", # If accessed via an OpenAI-compatible endpoint for it
    "mistral-small-latest", # If accessed via an OpenAI-compatible endpoint for it
    # Groq models are OpenAI compatible
    "groq/llama-3.3-70b-versatile",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/mistral-saba-24b",
    "groq/qwen-qwq-32b",
    "groq/gemma2-9b-it",
    "groq/compound-beta",
    "groq/compound-beta-mini",
    # Cerebras models are OpenAI compatible
    "cerebras/llama-4-scout-17b-16e-instruct",
    "cerebras/llama-3.3-70b",
    # OpenRouter - support depends on the underlying model.
    # Forcing json_object here is a safe bet if OpenRouter passes it.
    # Add OpenRouter model prefixes/names if known to support at least json_object
    "openrouter/mistralai/mistral-nemo",
    "openrouter/tngtech/deepseek-r1t-chimera:free",
}
# Add specific known Mistral native model names if you have a separate path for them supporting schema directly
MISTRAL_NATIVE_MODELS_SUPPORTING_SCHEMA = {
    "mistral-large-latest", # Add other native Mistral model names that support schema in response_format
    # "mistral-medium", "mistral-small" etc. if they do via native API
}

# --- Enums (Mirrored from UMS for consistency, or defined if agent-specific) ---
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


class WorkflowStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ActionStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ActionType(str, Enum):
    TOOL_USE = "tool_use"
    REASONING = "reasoning"
    PLANNING = "planning"
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    SUMMARY = "summary"
    CONSOLIDATION = "consolidation"
    MEMORY_OPERATION = "memory_operation"


class ArtifactType(str, Enum):
    FILE = "file"
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CHART = "chart"
    CODE = "code"
    DATA = "data"
    JSON = "json"
    URL = "url"


class ThoughtType(str, Enum):
    GOAL = "goal"
    QUESTION = "question"
    HYPOTHESIS = "hypothesis"
    INFERENCE = "inference"
    EVIDENCE = "evidence"
    CONSTRAINT = "constraint"
    PLAN = "plan"
    DECISION = "decision"
    REASONING = "reasoning"
    REFLECTION = "reflection"
    CRITIQUE = "critique"
    SUMMARY = "summary"
    USER_GUIDANCE = "user_guidance"
    INSIGHT = "insight"
    ANALYSIS = "analysis"


class MemoryLevel(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


class MemoryType(str, Enum):
    OBSERVATION = "observation"
    ACTION_LOG = "action_log"
    TOOL_OUTPUT = "tool_output"
    ARTIFACT_CREATION = "artifact_creation"
    REASONING_STEP = "reasoning_step"
    FACT = "fact"
    INSIGHT = "insight"
    PLAN = "plan"
    QUESTION = "question"
    SUMMARY = "summary"
    REFLECTION = "reflection"
    SKILL = "skill"
    PROCEDURE = "procedure"
    PATTERN = "pattern"
    CODE = "code"
    JSON = "json"
    URL = "url"
    USER_INPUT = "user_input"
    TEXT = "text"


class LinkType(str, Enum):
    RELATED = "related"
    CAUSAL = "causal"
    SEQUENTIAL = "sequential"
    HIERARCHICAL = "hierarchical"
    CONTRADICTS = "contradicts"
    SUPPORTS = "supports"
    GENERALIZES = "generalizes"
    SPECIALIZES = "specializes"
    FOLLOWS = "follows"
    PRECEDES = "precedes"
    TASK = "task"
    REFERENCES = "references"


class GoalStatus(str, Enum):
    ACTIVE = "active"
    PLANNED = "planned"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    ABANDONED = "abandoned"

class _Decision(str, Enum):
    TOOL_EXECUTED = "tool_executed_by_mcp"
    THOUGHT = "thought_process"
    CALL_TOOL = "call_tool"
    COMPLETE = "complete"
    COMPLETE_ART = "complete_with_artifact"
    PLAN_UPDATE = "plan_update"

# ==========================================================================
# CONSTANTS
# ==========================================================================
AGENT_STATE_FILE = "agent_loop_state_v4.3_simplified_robust.json"
AGENT_NAME = "EidenticEngine4.3-SimplifiedRobust"

AGENT_LOOP_TEMP_DIR = Path(".") / ".agent_loop_tmp"
TEMP_WORKFLOW_ID_FILE = AGENT_LOOP_TEMP_DIR / "current_workflow_id.txt"

BASE_REFLECTION_THRESHOLD = int(os.environ.get("BASE_REFLECTION_THRESHOLD", "15"))  # Increased from 7
BASE_CONSOLIDATION_THRESHOLD = int(os.environ.get("BASE_CONSOLIDATION_THRESHOLD", "25"))  # Increased from 12
MIN_REFLECTION_THRESHOLD = 8  # Increased from 3
MAX_REFLECTION_THRESHOLD = 30  # Increased from 15
MIN_CONSOLIDATION_THRESHOLD = 15  # Increased from 5
MAX_CONSOLIDATION_THRESHOLD = 50  # Increased from 25
THRESHOLD_ADAPTATION_DAMPENING = float(os.environ.get("THRESHOLD_DAMPENING", "0.75"))
MOMENTUM_THRESHOLD_BIAS_FACTOR = 1.2

# Significantly increased intervals to reduce interruptions during productive work
OPTIMIZATION_LOOP_INTERVAL = int(os.environ.get("OPTIMIZATION_INTERVAL", "50"))
MEMORY_PROMOTION_LOOP_INTERVAL = int(os.environ.get("PROMOTION_INTERVAL", "75"))
STATS_ADAPTATION_INTERVAL = int(os.environ.get("STATS_ADAPTATION_INTERVAL", "60"))
MAINTENANCE_INTERVAL = int(os.environ.get("MAINTENANCE_INTERVAL", "100"))

# LLM-based scoring thresholds (all scores are 0-100)
LLM_SIMILARITY_CONSOLIDATION_THRESHOLD = 60    # Consolidate memories if similarity >= 60
LLM_PROGRESS_REASONABLENESS_THRESHOLD = 51     # Progress is reasonable if score >= 51
LLM_GOAL_CLASSIFICATION_THRESHOLD = 40         # Valid goal classification if confidence >= 40
LLM_ACTIVITY_CATEGORY_THRESHOLD = 40           # Include activity category if score >= 40
LLM_COMPLEXITY_HIGH_THRESHOLD = 70             # Task is highly complex if score >= 70
LLM_RESEARCH_INTENSIVE_THRESHOLD = 60          # Task is research-intensive if score >= 60
LLM_CREATION_FOCUSED_THRESHOLD = 60            # Task is creation-focused if score >= 60
LLM_REPLAN_NECESSITY_THRESHOLD = 65            # Replan if feedback score >= 65
LLM_FILE_CREATION_THRESHOLD = 60               # Task involves file creation if score >= 60
LLM_ERROR_PLANNING_THRESHOLD = 70              # Error is planning-related if score >= 70
LLM_VAGUENESS_THRESHOLD = 60                   # Plan step is vague if score >= 60
LLM_WORKFLOW_MATCH_THRESHOLD = 70              # Workflow type matches if score >= 70

# Focus mode thresholds - when agent is actively creating artifacts, reduce meta-cognition
FOCUS_MODE_REFLECTION_MULTIPLIER = 2.5  # Multiply thresholds by this when in focus mode
FOCUS_MODE_CONSOLIDATION_MULTIPLIER = 2.0

AUTO_LINKING_DELAY_SECS: Tuple[float, float] = (1.5, 3.0)
DEFAULT_PLAN_STEP = "Execute immediate action: Use available tools to make progress toward the goal."
CONTEXT_RECENT_ACTIONS_FETCH_LIMIT = 30  # **CRITICAL FIX**: Increased from 10 to 30 to fetch more recent actions
CONTEXT_IMPORTANT_MEMORIES_FETCH_LIMIT = 25  # **CRITICAL FIX**: Increased from 7 to 25 to avoid losing critical info  
CONTEXT_KEY_THOUGHTS_FETCH_LIMIT = 20  # **CRITICAL FIX**: Increased from 7 to 20 for better reasoning continuity
CONTEXT_PROCEDURAL_MEMORIES_FETCH_LIMIT = 8  # **CRITICAL FIX**: Increased from 3 to 8
CONTEXT_PROACTIVE_MEMORIES_FETCH_LIMIT = 12  # **CRITICAL FIX**: Increased from 5 to 12
CONTEXT_LINK_TRAVERSAL_FETCH_LIMIT = 12  # **CRITICAL FIX**: Increased from 5 to 12
CONTEXT_GOAL_DETAILS_FETCH_LIMIT = 8  # **CRITICAL FIX**: Increased from 3 to 8
CONTEXT_RECENT_ACTIONS_SHOW_LIMIT = 20  # **CRITICAL FIX**: Increased from 7 to 20 so agent can see what it just did
CONTEXT_IMPORTANT_MEMORIES_SHOW_LIMIT = 15  # **CRITICAL FIX**: Increased from 5 to 15 to avoid losing recent memories
CONTEXT_KEY_THOUGHTS_SHOW_LIMIT = 12  # **CRITICAL FIX**: Increased from 5 to 12 for better continuity
CONTEXT_PROCEDURAL_MEMORIES_SHOW_LIMIT = 5  # **CRITICAL FIX**: Increased from 2 to 5
CONTEXT_PROACTIVE_MEMORIES_SHOW_LIMIT = 8  # **CRITICAL FIX**: Increased from 3 to 8  
CONTEXT_WORKING_MEMORY_SHOW_LIMIT = 20  # **CRITICAL FIX**: Increased from 10 to 20
CONTEXT_LINK_TRAVERSAL_SHOW_LIMIT = 8  # **CRITICAL FIX**: Increased from 3 to 8
CONTEXT_GOAL_STACK_SHOW_LIMIT = 10  # **CRITICAL FIX**: Increased from 5 to 10
CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD = 15_000
CONTEXT_COMPRESSION_TARGET_TOKENS = 5_000
MAX_CONSECUTIVE_ERRORS = 3

# --- Agent-Internal Tools ---
AGENT_TOOL_UPDATE_PLAN = "agent:update_plan"

# --- UMS Server Name ---
# This is now the primary way the agent knows how to construct full UMS tool names.
UMS_SERVER_NAME = "Ultimate MCP Server"  # ASSUMPTION: Matches your MCP Server's registered name

# --- UMS Base Function Names (used for internal logic mapping and checks) ---
# These are the actual function names as defined in unified_memory_system.py
UMS_FUNC_CREATE_WORKFLOW = "create_workflow"
UMS_FUNC_UPDATE_WORKFLOW_STATUS = "update_workflow_status"
UMS_FUNC_RECORD_ACTION_START = "record_action_start"
UMS_FUNC_RECORD_ACTION_COMPLETION = "record_action_completion"
UMS_FUNC_GET_ACTION_DETAILS = "get_action_details"
UMS_FUNC_ADD_ACTION_DEPENDENCY = "add_action_dependency"
UMS_FUNC_GET_ACTION_DEPENDENCIES = "get_action_dependencies"
UMS_FUNC_RECORD_ARTIFACT = "record_artifact"
UMS_FUNC_GET_ARTIFACTS = "get_artifacts"
UMS_FUNC_GET_ARTIFACT_BY_ID = "get_artifact_by_id"
UMS_FUNC_RECORD_THOUGHT = "record_thought"
UMS_FUNC_CREATE_THOUGHT_CHAIN = "create_thought_chain"
UMS_FUNC_GET_THOUGHT_CHAIN = "get_thought_chain"
UMS_FUNC_STORE_MEMORY = "store_memory"
UMS_FUNC_GET_MEMORY_BY_ID = "get_memory_by_id"
UMS_FUNC_CREATE_LINK = "create_memory_link"
UMS_FUNC_SEARCH_SEMANTIC_MEMORIES = "search_semantic_memories"
UMS_FUNC_QUERY_MEMORIES = "query_memories"
UMS_FUNC_HYBRID_SEARCH = "hybrid_search_memories"
UMS_FUNC_UPDATE_MEMORY = "update_memory"
UMS_FUNC_GET_LINKED_MEMORIES = "get_linked_memories"
UMS_FUNC_GET_WORKING_MEMORY = "get_working_memory"
UMS_FUNC_FOCUS_MEMORY = "focus_memory"
UMS_FUNC_OPTIMIZE_WM = "optimize_working_memory"
UMS_FUNC_SAVE_COGNITIVE_STATE = "save_cognitive_state"
UMS_FUNC_LOAD_COGNITIVE_STATE = "load_cognitive_state"
UMS_FUNC_AUTO_FOCUS = "auto_update_focus"
UMS_FUNC_PROMOTE_MEM = "promote_memory_level"
UMS_FUNC_CONSOLIDATION = "consolidate_memories"
UMS_FUNC_REFLECTION = "generate_reflection"
UMS_FUNC_SUMMARIZE_TEXT = "summarize_text"
UMS_FUNC_SUMMARIZE_CONTEXT_BLOCK = "summarize_context_block"
UMS_FUNC_DELETE_EXPIRED_MEMORIES = "delete_expired_memories"
UMS_FUNC_COMPUTE_STATS = "compute_memory_statistics"
UMS_FUNC_LIST_WORKFLOWS = "list_workflows"
UMS_FUNC_GET_WORKFLOW_DETAILS = "get_workflow_details"
UMS_FUNC_GET_RECENT_ACTIONS = "get_recent_actions"
UMS_FUNC_VISUALIZE_REASONING_CHAIN = "visualize_reasoning_chain"
UMS_FUNC_VISUALIZE_MEMORY_NETWORK = "visualize_memory_network"
UMS_FUNC_GENERATE_WORKFLOW_REPORT = "generate_workflow_report"
UMS_FUNC_GET_RICH_CONTEXT_PACKAGE = "get_rich_context_package"
UMS_FUNC_CREATE_GOAL = "create_goal"
UMS_FUNC_UPDATE_GOAL_STATUS = "update_goal_status"
UMS_FUNC_GET_GOAL_DETAILS = "get_goal_details"
UMS_FUNC_GET_GOAL_STACK = "get_goal_stack"
UMS_FUNC_DIAGNOSE_FILE_ACCESS = "diagnose_file_access_issues"
UMS_FUNC_GET_MULTI_TOOL_GUIDANCE = "get_multi_tool_guidance"


BACKGROUND_TASK_TIMEOUT_SECONDS = 60.0
MAX_CONCURRENT_BG_TASKS = 10
_PAYLOAD_TRUNCATE_AT = 8_192          # characters
_RECORD_TIMEOUT_SEC   = 30            # asyncio.wait_for guard
_MAX_RETRIES          = 1             # one retry after first failure

_TASK_TIMEOUT_SEC              = 30       # hard stop for any single periodic task
_INTERNAL_PROMO_SENTINEL       = "CHECK_PROMOTIONS_INTERNAL_TASK"
_PROMO_JITTER_RANGE = (0.10, 0.40)          # seconds
_PROMO_CALL_TIMEOUT = 12                    # seconds – safeguard for hung UMS call
_PROMO_TOOL_NAME    = UMS_FUNC_PROMOTE_MEM  # keep constants in one place

# Lowest number ⇒ highest priority in execution order
_TASK_PRIORITY: Dict[str, int] = {
    UMS_FUNC_DELETE_EXPIRED_MEMORIES : 0,
    UMS_FUNC_COMPUTE_STATS          : 1,
    _INTERNAL_PROMO_SENTINEL        : 2,
    UMS_FUNC_OPTIMIZE_WM            : 3,
    UMS_FUNC_AUTO_FOCUS             : 4,
    UMS_FUNC_CONSOLIDATION          : 5,
    UMS_FUNC_REFLECTION             : 6,
}


# --- LOCAL CUSTOM EXCEPTIONS ---
class ToolError(Exception):
    pass


class ToolInputError(ToolError):
    pass


class InfiniteLoopError(Exception):
    """Raised when an infinite loop is detected and should be prevented."""
    pass


# ==========================================================================
# LOCAL UTILITY CLASSES & HELPERS
# ==========================================================================

class LRUCache:
    """
    Simple LRU (Least Recently Used) cache implementation with size limits.
    
    When the cache reaches maxsize, the least recently used items are evicted.
    This prevents indefinite growth of cache memory usage.
    """
    
    def __init__(self, maxsize: int = 1000):
        self.maxsize = maxsize
        self._cache: OrderedDict = OrderedDict()
    
    def __contains__(self, key) -> bool:
        return key in self._cache
    
    def __getitem__(self, key):
        if key in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        raise KeyError(key)
    
    def __setitem__(self, key, value):
        if key in self._cache:
            # Update existing item and move to end
            self._cache[key] = value
            self._cache.move_to_end(key)
        else:
            # Add new item
            self._cache[key] = value
            # Evict oldest if over capacity
            if len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)  # Remove oldest (first) item
    
    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default
    
    def clear(self):
        """Clear all cached items."""
        self._cache.clear()
    
    def __len__(self):
        return len(self._cache)

from contextlib import asynccontextmanager


class LLMClassificationHelper:
    """
    Unified helper for LLM-based classification tasks with caching and robust error handling.
    Eliminates duplicate patterns throughout the codebase.
    """
    
    def __init__(self, agent_master_loop: "AgentMasterLoop"):
        self.aml = agent_master_loop
        self.logger = agent_master_loop.logger
        self.mcp_client = agent_master_loop.mcp_client
        
        # Unified cache for all LLM classifications
        self._cache: Dict[str, Any] = {}
        self._cache_max_size = 1000
        
    async def _llm_classify(
        self, 
        prompt: str, 
        schema: Dict[str, Any], 
        cache_key: str,
        schema_name: str = "classification",
        max_tokens: int = 50,
        temperature: float = 0.1,
        fallback_value: Any = None,
        extract_key: Optional[str] = None
    ) -> Optional[Any]:
        """
        Unified LLM call with caching, error handling, and consistent patterns.
        
        Args:
            prompt: The prompt to send to the LLM
            schema: JSON schema for structured response
            cache_key: Unique key for caching this classification
            schema_name: Name for the schema (for logging)
            max_tokens: Maximum tokens for response
            temperature: Temperature for generation
            fallback_value: Value to return if LLM fails
            extract_key: If provided, extract this key from response
            
        Returns:
            LLM response data or fallback_value if failed
        """
        # Check cache first
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Manage cache size
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest 25% of entries
            items_to_remove = list(self._cache.keys())[:self._cache_max_size // 4]
            for key in items_to_remove:
                del self._cache[key]
        
        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": prompt}],
                response_schema=schema,
                schema_name=schema_name,
                use_cheap_model=True,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            if response:
                result = response.get(extract_key) if extract_key else response
                self._cache[cache_key] = result
                return result
                
        except Exception as e:
            self.logger.debug(f"LLM {schema_name} failed: {e}")
        
        # Cache the fallback to avoid repeated failures
        self._cache[cache_key] = fallback_value
        return fallback_value
    
    async def semantic_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity score (0.0-1.0)"""
        if not content1 or not content2:
            return 0.0
            
        cache_key = f"similarity:{hash(content1[:300] + content2[:300])}"
        
        prompt = f"""Rate the semantic similarity between these two pieces of content on a scale of 0-100:

Content 1: "{content1[:500]}"
Content 2: "{content2[:500]}"

Scoring criteria:
0-10: Completely different topics
11-25: Loosely related but different
26-40: Some shared concepts
41-60: Moderately similar
61-80: Highly similar
81-95: Nearly identical
96-100: Semantically identical

Respond with just the numerical score (0-100)."""

        schema = {
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
        
        result = await self._llm_classify(
            prompt=prompt,
            schema=schema,
            cache_key=cache_key,
            schema_name="semantic_similarity",
            max_tokens=5,
            extract_key="score",
            fallback_value=0
        )
        
        return result / 100.0 if result is not None else 0.0


class ToolServerCache:
    """
    Cache for tool server resolution to avoid repeated complex lookups.
    Addresses the inefficient _find_tool_server pattern.
    """
    
    def __init__(self, agent_master_loop: "AgentMasterLoop"):
        self.aml = agent_master_loop
        self.logger = agent_master_loop.logger
        self._cache: Dict[str, Optional[str]] = {}
        self._cache_max_size = 500
        
    def get_server(self, tool_identifier: str) -> Optional[str]:
        """Get cached server name for tool identifier"""
        return self._cache.get(tool_identifier)
    
    def set_server(self, tool_identifier: str, server_name: Optional[str]):
        """Cache server name for tool identifier"""
        # Manage cache size
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest 25% of entries
            items_to_remove = list(self._cache.keys())[:self._cache_max_size // 4]
            for key in items_to_remove:
                del self._cache[key]
        
        self._cache[tool_identifier] = server_name
    
    def invalidate(self, tool_identifier: Optional[str] = None):
        """Invalidate cache entry or entire cache"""
        if tool_identifier:
            self._cache.pop(tool_identifier, None)
        else:
            self._cache.clear()


class SimpleBackgroundTaskManager:
    """
    Simplified background task manager to replace the overly complex _start_background_task.
    Provides basic task management without excessive overhead.
    """
    
    def __init__(self, agent_master_loop: "AgentMasterLoop"):
        self.aml = agent_master_loop
        self.logger = agent_master_loop.logger
        self._tasks: Set[asyncio.Task] = set()
        self._max_tasks = MAX_CONCURRENT_BG_TASKS
        
    def start_task(self, coro_fn, *args, **kwargs) -> Optional[asyncio.Task]:
        """Start a background task with simple management"""
        # Check task limit
        if len(self._tasks) >= self._max_tasks:
            self.logger.warning(f"Background task limit reached ({self._max_tasks}). Skipping task.")
            return None
        
        # Create task
        task_name = f"bg_{coro_fn.__name__}_{random.randint(100, 999)}"
        
        # Build coroutine
        if inspect.ismethod(coro_fn) and coro_fn.__self__ is not None:
            coro = coro_fn(*args, **kwargs)
        else:
            coro = coro_fn(self.aml, *args, **kwargs)
        
        # Create and track task
        task = asyncio.create_task(coro, name=task_name)
        self._tasks.add(task)
        
        # Add cleanup callback
        task.add_done_callback(self._task_done)
        
        self.logger.debug(f"Started background task: {task_name}")
        return task
    
    def _task_done(self, task: asyncio.Task):
        """Clean up completed task"""
        self._tasks.discard(task)
        
        if task.cancelled():
            self.logger.debug(f"Background task {task.get_name()} was cancelled")
        elif task.exception():
            self.logger.error(f"Background task {task.get_name()} failed: {task.exception()}")
        else:
            self.logger.debug(f"Background task {task.get_name()} completed successfully")
    
    async def cleanup_all(self):
        """Cancel and cleanup all background tasks"""
        if not self._tasks:
            return
        
        self.logger.info(f"Cancelling {len(self._tasks)} background tasks")
        
        # Cancel all tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        
        # Wait for cancellation
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self._tasks.clear()


class ErrorHandlerMixin:
    """
    Provides consistent error handling patterns to replace silent exception catching.
    """
    
    def safe_execute(self, operation_name: str, func, *args, **kwargs):
        """
        Execute a function with consistent error handling and logging.
        Returns (success: bool, result: Any, error: Optional[Exception])
        """
        try:
            result = func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            self.logger.warning(f"Operation '{operation_name}' failed: {type(e).__name__}: {e}")
            return False, None, e
    
    async def safe_execute_async(self, operation_name: str, coro_func, *args, **kwargs):
        """
        Execute an async function with consistent error handling and logging.
        Returns (success: bool, result: Any, error: Optional[Exception])
        """
        try:
            result = await coro_func(*args, **kwargs)
            return True, result, None
        except Exception as e:
            self.logger.warning(f"Async operation '{operation_name}' failed: {type(e).__name__}: {e}")
            return False, None, e
    
    def handle_tool_error(self, tool_name: str, error: Exception, fallback_result: Any = None):
        """
        Handle tool execution errors with consistent logging and fallback behavior.
        """
        error_type = type(error).__name__
        
        # Log based on error severity
        if isinstance(error, (ConnectionError, TimeoutError)):
            self.logger.warning(f"Tool '{tool_name}' failed due to connectivity: {error}")
        elif isinstance(error, (ValueError, TypeError)):
            self.logger.error(f"Tool '{tool_name}' failed due to invalid input: {error}")
        else:
            self.logger.error(f"Tool '{tool_name}' failed unexpectedly: {error_type}: {error}")
        
        return {
            "success": False,
            "error_type": error_type,
            "error_message": str(error),
            "data": fallback_result
        }


class StateTransactionManager:
    """
    Atomic state management for AgentMasterLoop.
    
    Ensures that state changes and UMS operations either all succeed
    or all are rolled back, preventing inconsistent state.
    """
    
    def __init__(self, agent_master_loop: "AgentMasterLoop"):
        self.aml = agent_master_loop
        self.logger = agent_master_loop.logger
        self._in_transaction = False
        self._original_state: Optional["AgentState"] = None
        self._transaction_id: Optional[str] = None
        
    @asynccontextmanager
    async def transaction(self, description: str = "State Transaction"):
        """
        Start an atomic transaction for state changes.
        
        Usage:
            async with state_manager.transaction("Update goal"):
                # Make state changes
                # Call UMS operations  
                # If any operation fails, everything is rolled back
        """
        if self._in_transaction:
            raise RuntimeError("Cannot start nested state transactions")
            
        self._transaction_id = f"txn-{uuid.uuid4().hex[:8]}"
        self._in_transaction = True
        
        # Take a deep copy of current state as backup
        self._original_state = copy.deepcopy(self.aml.state)
        
        self.logger.debug(f"🔄 State Transaction [{self._transaction_id}] STARTED: {description}")
        
        try:
            yield self
            # If we reach here, commit the transaction
            await self._commit_transaction()
            self.logger.info(f"✅ State Transaction [{self._transaction_id}] COMMITTED: {description}")
            
        except Exception as e:
            # Rollback on any exception
            await self._rollback_transaction()
            self.logger.warning(f"🔙 State Transaction [{self._transaction_id}] ROLLED BACK: {description} - Error: {e}")
            raise
            
        finally:
            self._cleanup_transaction()
    
    async def _commit_transaction(self):
        """Commit the transaction by saving state to disk."""
        try:
            await self.aml._save_agent_state()
        except Exception as e:
            self.logger.error(f"Failed to save state during transaction commit: {e}")
            raise
    
    async def _rollback_transaction(self):
        """Rollback by restoring the original state."""
        if self._original_state is not None:
            self.aml.state = self._original_state
            self.logger.debug(f"State rolled back to pre-transaction state")
        else:
            self.logger.warning("No original state to rollback to")
    
    def _cleanup_transaction(self):
        """Clean up transaction state."""
        self._in_transaction = False
        self._original_state = None
        self._transaction_id = None


class StateRecoveryManager:
    """
    Handles state recovery operations instead of emergency replanning.
    
    Attempts to fix state inconsistencies gracefully before resorting
    to replanning as a last resort.
    """
    
    def __init__(self, agent_master_loop: "AgentMasterLoop"):
        self.aml = agent_master_loop
        self.logger = agent_master_loop.logger
        
    async def attempt_state_recovery(self, issue_description: str, error_details: Optional[Dict[str, Any]] = None) -> bool:
        """
        Attempt to recover from state issues instead of immediate replanning.
        Enhanced with completion-aware recovery strategies.
        
        Returns:
            True if recovery was successful, False if replanning is needed
        """
        self.logger.info(f"🔧 Attempting state recovery for: {issue_description}")
        
        # **COMPLETION-AWARE RECOVERY**: Check if we should prioritize completion over complex recovery
        completion_recovery_result = await self._attempt_completion_recovery(issue_description, error_details)
        if completion_recovery_result:
            return True
        
        recovery_attempts = [
            self._recover_goal_stack_sync,
            self._recover_workflow_context_sync,
            self._recover_thought_chain,
            self._recover_plan_validation,
            self._recover_infinite_loop_state,  # New: Handle loop states
            self._recover_to_completion_track,  # New: Direct to completion
        ]
        
        for recovery_func in recovery_attempts:
            try:
                if await recovery_func(error_details):
                    self.logger.info(f"✅ State recovery successful via {recovery_func.__name__}")
                    return True
            except Exception as e:
                self.logger.warning(f"Recovery attempt {recovery_func.__name__} failed: {e}")
                continue
        
        self.logger.warning(f"❌ All state recovery attempts failed for: {issue_description}")
        return False
    
    async def _recover_goal_stack_sync(self, error_details: Optional[Dict[str, Any]]) -> bool:
        """Try to sync goal stack with UMS."""
        if not self.aml.state.current_goal_id:
            return False
            
        try:
            # Fetch current goal stack from UMS
            ums_stack = await self.aml._fetch_goal_stack_from_ums(self.aml.state.current_goal_id)
            if ums_stack:
                # Update local goal stack to match UMS
                self.aml.state.goal_stack = ums_stack
                self.logger.info("🔄 Goal stack synced with UMS")
                return True
        except Exception as e:
            self.logger.debug(f"Goal stack sync failed: {e}")
            
        return False
    
    async def _recover_workflow_context_sync(self, error_details: Optional[Dict[str, Any]]) -> bool:
        """Try to fix workflow/context inconsistencies."""
        if not self.aml.state.workflow_id:
            return False
            
        try:
            # Re-validate workflow and context
            if await self.aml._validate_agent_workflow_and_context():
                self.logger.info("🔄 Workflow/context validation recovered")
                return True
        except Exception as e:
            self.logger.debug(f"Workflow/context recovery failed: {e}")
            
        return False
    
    async def _recover_thought_chain(self, error_details: Optional[Dict[str, Any]]) -> bool:
        """Try to recover thought chain."""
        if not self.aml.state.current_thought_chain_id:
            try:
                await self.aml._set_default_thought_chain_id()
                if self.aml.state.current_thought_chain_id:
                    self.logger.info("🔄 Thought chain recovered")
                    return True
            except Exception as e:
                self.logger.debug(f"Thought chain recovery failed: {e}")
                
        return False
    
    async def _recover_plan_validation(self, error_details: Optional[Dict[str, Any]]) -> bool:
        """Try to fix plan validation issues."""
        if not self.aml.state.current_plan or not self.aml.state.current_plan[0].description:
            return False
            
        # Check if plan is just the default
        first_step = self.aml.state.current_plan[0]
        if first_step.description == DEFAULT_PLAN_STEP:
            return False  # This needs actual replanning
            
        # Validate plan steps structure
        valid_plan = True
        for step in self.aml.state.current_plan:
            if not step.description or not step.description.strip():
                valid_plan = False
                break
                
        if valid_plan:
            self.logger.info("🔄 Plan structure is valid")
            return True
            
        return False
    
    async def _attempt_completion_recovery(self, issue_description: str, error_details: Optional[Dict[str, Any]]) -> bool:
        """Attempt recovery by directing toward goal completion rather than complex recovery"""
        try:
            # Check if we have a current goal and can assess completion
            if not self.aml.state.current_goal_id or not self.aml.state.goal_stack:
                return False
            
            current_goal = self.aml.state.goal_stack[-1]
            goal_desc = current_goal.get("description", "")
            
            if not goal_desc or goal_desc == "Overall UMS Workflow Goal or Initial Task":
                return False
            
            # Check if goal is ready for completion
            goal_type = self.aml._classify_goal_type(goal_desc)
            completion_status = await self.aml._analyze_goal_completion_semantically(goal_desc, goal_type)
            
            # If ready for completion or should be forced, direct to completion
            if completion_status["ready_for_completion"] or completion_status["force_completion"]:
                self.logger.info("🎯 COMPLETION RECOVERY: Goal ready for completion - directing to completion instead of complex recovery")
                
                # Reset error tracking state
                self.aml.state.consecutive_error_count = 0
                self.aml.state.last_error_details = None
                
                # Set completion-focused plan
                criteria = completion_status["completion_criteria"]
                completion_plan = PlanStep(
                    description=f"Create {criteria['expected_deliverable']} to complete the goal",
                    assigned_tool=self.aml._get_ums_tool_mcp_name("record_artifact")
                )
                self.aml.state.current_plan = [completion_plan]
                self.aml.state.needs_replan = False
                
                return True
            
            # If we've exceeded reasonable limits, force completion recovery
            if (self.aml.state.search_attempts_count >= 3 or 
                self.aml.state.turns_since_artifact_creation >= 8 or
                self.aml.state.consecutive_same_tool_count >= 3):
                
                self.logger.info("🚨 COMPLETION RECOVERY: Exceeded limits - forcing completion recovery")
                
                # Reset problematic state
                self.aml.state.search_attempts_count = 1
                self.aml.state.consecutive_same_tool_count = 0
                self.aml.state.last_tool_executed = None
                self.aml.state.consecutive_error_count = 0
                
                # Force to completion stage
                criteria = completion_status["completion_criteria"]
                completion_plan = PlanStep(
                    description=f"Complete goal by creating {criteria['expected_deliverable']} with available information",
                    assigned_tool=self.aml._get_ums_tool_mcp_name("record_artifact")
                )
                self.aml.state.current_plan = [completion_plan]
                self.aml.state.plan_progression_stage = "completion"
                self.aml.state.needs_replan = False
                
                return True
                
        except Exception as e:
            self.logger.debug(f"Completion recovery failed: {e}")
        
        return False
    
    async def _recover_infinite_loop_state(self, error_details: Optional[Dict[str, Any]]) -> bool:
        """Recover from infinite loop states"""
        try:
            if self.aml.state.consecutive_same_tool_count >= 3:
                self.logger.info("🔄 LOOP RECOVERY: Resetting infinite loop state")
                
                # Reset loop tracking
                self.aml.state.consecutive_same_tool_count = 0
                self.aml.state.last_tool_executed = None
                self.aml.state.recent_tool_sequence = []
                
                # Force progression to different stage
                if self.aml.state.plan_progression_stage == "research":
                    self.aml.state.plan_progression_stage = "creation"
                elif self.aml.state.plan_progression_stage == "initial":
                    self.aml.state.plan_progression_stage = "analysis"
                else:
                    self.aml.state.plan_progression_stage = "completion"
                
                return True
                
        except Exception as e:
            self.logger.debug(f"Infinite loop recovery failed: {e}")
        
        return False
    
    async def _recover_to_completion_track(self, error_details: Optional[Dict[str, Any]]) -> bool:
        """Direct recovery to completion track when other approaches fail"""
        try:
            # If we have goal info, try to direct to completion
            if self.aml.state.goal_stack and self.aml.state.goal_stack[-1]:
                current_goal = self.aml.state.goal_stack[-1]
                goal_desc = current_goal.get("description", "")
                
                if goal_desc and goal_desc != "Overall UMS Workflow Goal or Initial Task":
                    goal_type = self.aml._classify_goal_type(goal_desc)
                    criteria = self.aml._get_goal_completion_criteria(goal_type, goal_desc)
                    
                    self.logger.info("🎯 COMPLETION TRACK RECOVERY: Directing to deliverable creation")
                    
                    # Create completion-focused plan
                    completion_plan = PlanStep(
                        description=f"Recover by creating {criteria['expected_deliverable']} to complete {goal_type} goal",
                        assigned_tool=self.aml._get_ums_tool_mcp_name("record_artifact")
                    )
                    
                    # Replace plan with completion focus
                    self.aml.state.current_plan = [completion_plan]
                    self.aml.state.plan_progression_stage = "completion"
                    self.aml.state.needs_replan = False
                    
                    # Reset error states
                    self.aml.state.consecutive_error_count = 0
                    self.aml.state.last_error_details = None
                    
                    return True
                    
        except Exception as e:
            self.logger.debug(f"Completion track recovery failed: {e}")
        
        return False


class IntelligentMemoryManager:
    """
    Handles intelligent memory consolidation and duplicate prevention
    to avoid storing the same information repeatedly.
    """
    
    def __init__(self, agent_master_loop: "AgentMasterLoop"):
        self.aml = agent_master_loop
        self.logger = agent_master_loop.logger
    
    async def should_store_memory(self, content: str, memory_type: str, description: str = "") -> Dict[str, Any]:
        """
        Determine if a memory should be stored or if similar content already exists.
        Returns decision with reasoning and suggested actions.
        """
        try:
            # Check for similar existing memories
            similar_memories = await self._find_similar_memories(content, memory_type)
            
            if similar_memories:
                # Analyze the similarity and determine action
                decision = self._analyze_memory_similarity(content, similar_memories, memory_type)
                return decision
            
            # No similar memories found - safe to store
            return {
                "should_store": True,
                "reason": "No similar memories found",
                "action": "store_new",
                "similar_count": 0
            }
            
        except Exception as e:
            self.logger.debug(f"Memory duplication check failed: {e}")
            # Default to allowing storage if check fails
            return {
                "should_store": True,
                "reason": "Duplication check failed - defaulting to store",
                "action": "store_new",
                "similar_count": 0
            }
    
    async def _find_similar_memories(self, content: str, memory_type: str) -> List[Dict[str, Any]]:
        """Find existing memories similar to the proposed content"""
        if not self.aml.state.workflow_id:
            return []
        
        try:
            # Use semantic search to find similar content
            search_tool = self.aml._get_ums_tool_mcp_name("search_semantic_memories")
            if not self.aml._find_tool_server(search_tool):
                return []
            
            # Search for semantically similar content
            search_result = await self.aml._execute_tool_call_internal(
                search_tool,
                {
                    "workflow_id": self.aml.state.workflow_id,
                    "query": content[:500],  # Limit search query length
                    "memory_types": [memory_type],
                    "limit": 10,
                    "similarity_threshold": 0.7  # High threshold for duplicates
                },
                record_action=False
            )
            
            if search_result.get("success"):
                data = search_result.get("data", {})
                memories = data.get("memories", [])
                return memories
                
        except Exception as e:
            self.logger.debug(f"Similar memory search failed: {e}")
        
        return []
    
    async def _analyze_memory_similarity(self, new_content: str, existing_memories: List[Dict[str, Any]], memory_type: str) -> Dict[str, Any]:
        """Analyze similarity between new content and existing memories using LLM-based semantic understanding"""
        
        similarity_analysis = await self._analyze_content_similarity_with_llm(new_content, existing_memories)
        
        return similarity_analysis if similarity_analysis else {
            "should_store": True,
            "reason": "LLM analysis failed - defaulting to store",
            "action": "store_new",
            "similar_count": 0
        }
    
    async def _analyze_content_similarity_with_llm(self, new_content: str, existing_memories: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze content similarity for memory deduplication"""
        
        if not existing_memories:
            return {
                "should_store": True,
                "reason": "No existing memories to compare against",
                "action": "store_new",
                "similar_count": 0
            }
        
        # Prepare content comparison for LLM
        existing_content_samples = []
        for i, memory in enumerate(existing_memories[:5]):  # Limit to 5 for LLM context
            content = memory.get("content", "")[:200]  # Truncate for LLM
            existing_content_samples.append(f"Memory {i+1}: {content}")
        
        comparison_prompt = f"""
Analyze whether this new content should be stored or if it's too similar to existing memories.

New Content: "{new_content[:500]}"

Existing Memories:
{chr(10).join(existing_content_samples)}

Rate similarity and provide decision:
- Rate overall similarity to existing content on 0-100 scale
- If 0-40: Store new (sufficiently different)
- If 41-70: Store with note (some similarity but adds value)  
- If 71-85: Consider skipping (high similarity)
- If 86-100: Skip duplicate (too similar/redundant)

Respond in this format:
similarity_score: [0-100]
decision: [store_new/store_with_note/skip_similar/skip_duplicate]
reason: [brief explanation]
"""

        try:
            memory_similarity_schema = {
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
            
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": comparison_prompt}],
                response_schema=memory_similarity_schema,
                schema_name="memory_similarity_analysis",
                use_cheap_model=True,  # Use cheap/fast model
                max_tokens=50,
                temperature=0.1
            )
            
            if response:
                similarity_score = response.get("similarity_score", 0)
                decision = response.get("decision", "store_new")
                reason = response.get("reason", "LLM analysis completed")
                
                # Convert LLM decision to memory analysis format
                should_store = decision in ["store_new", "store_with_note"]
                
                return {
                    "should_store": should_store,
                    "reason": f"LLM analysis (similarity={similarity_score}): {reason}",
                    "action": decision,
                    "similar_count": len(existing_memories),
                    "llm_similarity_score": similarity_score
                }
                
        except Exception as e:
            self.logger.debug(f"LLM memory similarity analysis failed: {e}")
        
        return None
    

    
    async def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity using LLM analysis with intelligent fallback"""
        if not content1 or not content2:
            return 0.0
        
        # Cache to avoid repeated LLM calls
        content_hash = hash(f"{content1[:300]}|{content2[:300]}")
        if content_hash in self._content_similarity_cache:
            return self._caches['content_similarity'][content_hash]
        
        # Use LLM for semantic similarity analysis
        similarity_score = await self._get_semantic_similarity_with_llm(content1, content2)
        self._caches['content_similarity'][content_hash] = similarity_score
        
        return similarity_score
    
    async def _get_semantic_similarity_with_llm(self, content1: str, content2: str) -> float:
        """Use LLM to calculate semantic similarity between content"""
        
        if not content1 or not content2:
            return 0.0
        
        # Use the new LLM helper for consistency and caching
        return await self.llm_helper.semantic_similarity(content1, content2)


    async def consolidate_similar_memories(self, memory_type: str = None) -> int:
        """
        Consolidate very similar memories to reduce redundancy.
        Returns number of memories consolidated.
        """
        try:
            if not self.aml.state.workflow_id:
                return 0
            
            # Query recent memories for consolidation
            query_tool = self.aml._get_ums_tool_mcp_name("query_memories")
            if not self.aml._find_tool_server(query_tool):
                return 0
            
            query_args = {
                "workflow_id": self.aml.state.workflow_id,
                "limit": 50,
                "include_content": True,
                "sort_by": "created_at",
                "sort_order": "desc"
            }
            
            if memory_type:
                query_args["memory_type"] = memory_type
            
            result = await self.aml._execute_tool_call_internal(
                query_tool, query_args, record_action=False
            )
            
            if not result.get("success"):
                return 0
            
            memories = result.get("data", {}).get("memories", [])
            if len(memories) < 2:
                return 0
            
            # Find groups of similar memories
            consolidation_groups = await self._find_consolidation_groups(memories)
            
            consolidated_count = 0
            for group in consolidation_groups:
                if await self._consolidate_memory_group(group):
                    consolidated_count += len(group) - 1  # -1 because we keep one
            
            if consolidated_count > 0:
                self.logger.info(f"📚 MEMORY CONSOLIDATION: Consolidated {consolidated_count} duplicate memories")
            
            return consolidated_count
            
        except Exception as e:
            self.logger.debug(f"Memory consolidation failed: {e}")
            return 0
    
    async def _find_consolidation_groups(self, memories: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Find groups of memories that should be consolidated using LLM-based semantic analysis"""
        consolidation_groups = []
        used_indices = set()
        
        for i, memory1 in enumerate(memories):
            if i in used_indices:
                continue
            
            content1 = memory1.get("content", "")
            if not content1:
                continue
            
            group = [memory1]
            used_indices.add(i)
            
            for j, memory2 in enumerate(memories[i+1:], i+1):
                if j in used_indices:
                    continue
                
                content2 = memory2.get("content", "")
                if not content2:
                    continue
                
                # Use LLM-based semantic similarity with intelligent scoring
                similarity_score_float = await self._get_semantic_similarity_with_llm(content1, content2)
                similarity_score = int(similarity_score_float * 100) if similarity_score_float else None
                
                # Use LLM-based decision for consolidation instead of primitive threshold
                should_consolidate = await self._should_consolidate_memories_llm(content1, content2, similarity_score)
                
                if should_consolidate:
                    group.append(memory2)
                    used_indices.add(j)
            
            # Only consider groups with 2 or more similar memories
            if len(group) >= 2:
                consolidation_groups.append(group)
        
        return consolidation_groups
    
    async def _should_consolidate_memories_llm(self, content1: str, content2: str, similarity_score: Optional[int]) -> bool:
        """Use LLM to decide if memories should be consolidated instead of primitive thresholds"""
        
        if similarity_score is None:
            return False  # No LLM score = no consolidation
        
        # Use the new LLM helper for consistency and caching
        return await self.llm_helper.consolidation_decision(content1, content2, similarity_score)
    
    async def _consolidate_memory_group(self, group: List[Dict[str, Any]]) -> bool:
        """Consolidate a group of similar memories into one enhanced memory"""
        try:
            if len(group) < 2:
                return False
            
            # Keep the most recent memory as the base
            base_memory = max(group, key=lambda m: m.get("created_at", ""))
            
            # Create consolidated content
            all_content = [mem.get("content", "") for mem in group]
            unique_points = await self._extract_unique_points_semantically(all_content)
            
            if len(unique_points) <= 1:
                # Not enough unique content to warrant keeping separate
                # Delete the older memories
                for memory in group:
                    if memory["memory_id"] != base_memory["memory_id"]:
                        await self._delete_memory(memory["memory_id"])
                return True
            
            return False  # Don't consolidate if there's significant unique content
            
        except Exception as e:
            self.logger.debug(f"Memory group consolidation failed: {e}")
            return False
    
    async def _extract_unique_points_semantically(self, content_list: List[str]) -> List[str]:
        """Extract unique information points using LLM-based semantic analysis"""
        
        if not content_list or len(content_list) <= 1:
            return content_list
        
        # Use LLM to identify unique vs redundant information
        unique_analysis_prompt = f"""
Analyze this list of content items and identify which contain unique information vs. redundant content.

Content Items:
{chr(10).join([f"{i+1}. {content[:200]}" for i, content in enumerate(content_list)])}

Identify which items contain substantially unique information that would be lost if removed.
List the item numbers that contain unique, non-redundant information.

Respond with just the numbers separated by commas (e.g., "1,3,4").
"""

        try:
            unique_content_schema = {
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
            
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": unique_analysis_prompt}],
                response_schema=unique_content_schema,
                schema_name="unique_content_analysis",
                use_cheap_model=True,  # Use cheap/fast model
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                unique_numbers = []
                unique_items = response.get("unique_items", [])
                for num in unique_items:
                    if isinstance(num, int) and 1 <= num <= len(content_list):
                        unique_numbers.append(num - 1)  # Convert to 0-based index
                
                # Return unique content items
                if unique_numbers:
                    return [content_list[i] for i in unique_numbers]
                    
        except Exception as e:
            self.logger.debug(f"LLM unique points analysis failed: {e}")
        
        # Intelligent fallback - assume all items are unique if analysis fails
        return content_list
    
    async def _delete_memory(self, memory_id: str) -> bool:
        """Delete a memory (placeholder - implement if UMS supports deletion)"""
        # Note: This would require a delete_memory function in UMS
        # For now, just log the consolidation
        self.logger.debug(f"Would delete memory {memory_id} during consolidation")
        return True


class EnhancedStateValidator:
    """
    Comprehensive state validation that checks all aspects of agent state
    consistency with UMS and attempts recovery before falling back to replanning.
    """
    
    def __init__(self, agent_master_loop: "AgentMasterLoop"):
        self.aml = agent_master_loop
        self.logger = agent_master_loop.logger
        self.recovery_manager = StateRecoveryManager(agent_master_loop)
        
    async def validate_and_recover_state(self, context: str = "general") -> bool:
        """
        Perform comprehensive state validation and attempt recovery if needed.
        
        Returns:
            True if state is valid or was successfully recovered
            False if replanning is needed
        """
        self.logger.debug(f"🔍 Comprehensive state validation ({context})")
        
        validation_checks = [
            ("workflow_exists", self._validate_workflow_exists),
            ("context_valid", self._validate_context_valid), 
            ("goal_stack_consistent", self._validate_goal_stack_consistent),
            ("thought_chain_exists", self._validate_thought_chain_exists),
            ("plan_structure_valid", self._validate_plan_structure_valid),
            ("goal_completion_state", self._validate_goal_completion_state),
            ("progress_tracking_state", self._validate_progress_tracking_state),
        ]
        
        issues_found = []
        
        for check_name, check_func in validation_checks:
            try:
                is_valid, issue_desc = await check_func()
                if not is_valid:
                    issues_found.append((check_name, issue_desc))
            except Exception as e:
                issues_found.append((check_name, f"Validation error: {e}"))
        
        if not issues_found:
            self.logger.debug("✅ All state validation checks passed")
            return True
        
        # Attempt recovery for each issue
        self.logger.info(f"🔧 Found {len(issues_found)} state issues, attempting recovery...")
        
        recovery_success = True
        for check_name, issue_desc in issues_found:
            # Use specific recovery methods for our new validation types
            if check_name == "goal_completion_state":
                recovered = await self._recover_goal_completion_state()
            elif check_name == "progress_tracking_state":
                recovered = await self._recover_progress_tracking_state()
            else:
                # Use the general recovery manager for other issues
                recovered = await self.recovery_manager.attempt_state_recovery(
                    f"{check_name}: {issue_desc}"
                )
            
            if not recovered:
                recovery_success = False
                self.logger.warning(f"❌ Could not recover from: {check_name}: {issue_desc}")
        
        return recovery_success
    
    async def _validate_workflow_exists(self) -> Tuple[bool, str]:
        """Validate that workflow exists in UMS."""
        if not self.aml.state.workflow_id:
            return False, "No workflow_id set"
            
        exists = await self.aml._check_workflow_exists(self.aml.state.workflow_id)
        if not exists:
            return False, f"Workflow {self.aml.state.workflow_id} not found in UMS"
            
        return True, ""
    
    async def _validate_context_valid(self) -> Tuple[bool, str]:
        """Validate that context is consistent with UMS."""
        if not self.aml.state.context_id:
            return False, "No context_id set"
            
        # This leverages the existing validation logic
        valid = await self.aml._validate_agent_workflow_and_context()
        if not valid:
            return False, "Context validation failed"
            
        return True, ""
    
    async def _validate_goal_stack_consistent(self) -> Tuple[bool, str]:
        """Validate that goal stack is consistent with UMS and repair if needed."""
        if not self.aml.state.current_goal_id:
            return True, ""  # No goal is a valid state
            
        try:
            # Check if current goal is in local stack
            current_goal_found = any(
                isinstance(g, dict) and g.get("goal_id") == self.aml.state.current_goal_id
                for g in self.aml.state.goal_stack
            )
            
            if not current_goal_found:
                # Try to fetch and synchronize with UMS
                ums_stack = await self.aml._fetch_goal_stack_from_ums(self.aml.state.current_goal_id)
                if ums_stack:
                    # Verify the current goal is in the UMS stack
                    current_goal_in_ums = any(
                        isinstance(g, dict) and g.get("goal_id") == self.aml.state.current_goal_id
                        for g in ums_stack
                    )
                    if current_goal_in_ums:
                        self.aml.state.goal_stack = ums_stack
                        self.aml.logger.info(f"🔄 RECOVERY: Synchronized goal stack with UMS - goal {self.aml.state.current_goal_id} now found")
                        return True, "Recovered by synchronizing with UMS"
                    else:
                        return False, "Current goal not found in UMS stack either"
                else:
                    return False, "Could not fetch goal stack from UMS"
            
            # Additional check - ensure UMS and local are consistent
            ums_stack = await self.aml._fetch_goal_stack_from_ums(self.aml.state.current_goal_id)
            if ums_stack and len(ums_stack) != len(self.aml.state.goal_stack):
                # Stack size mismatch - update local to match UMS
                self.aml.state.goal_stack = ums_stack
                self.aml.logger.info(f"🔄 SYNC: Updated local goal stack to match UMS size ({len(ums_stack)} items)")
                
        except Exception as e:
            return False, f"Goal stack validation error: {e}"
            
        return True, ""
    
    async def _validate_thought_chain_exists(self) -> Tuple[bool, str]:
        """Validate that thought chain exists."""
        if not self.aml.state.current_thought_chain_id:
            return False, "No thought chain ID set"
        return True, ""
    
    async def _validate_plan_structure_valid(self) -> Tuple[bool, str]:
        """Validate that plan structure is valid."""
        if not self.aml.state.current_plan:
            return False, "No plan steps"
            
        for i, step in enumerate(self.aml.state.current_plan):
            if not step.description or not step.description.strip():
                return False, f"Plan step {i} has empty description"
                
        return True, ""
    
    async def _validate_goal_completion_state(self) -> Tuple[bool, str]:
        """Validate that goal completion state is consistent and reasonable."""
        if not self.aml.state.current_goal_id or not self.aml.state.goal_stack:
            return True, ""  # No goal to validate
        
        current_goal = self.aml.state.goal_stack[-1] if self.aml.state.goal_stack else {}
        goal_desc = current_goal.get("description", "")
        
        if not goal_desc or goal_desc == "Overall UMS Workflow Goal or Initial Task":
            return True, ""  # Skip generic goals
        
        try:
            # Check if completion readiness is working properly
            goal_type = self.aml._classify_goal_type(goal_desc)
            completion_status = await self.aml._analyze_goal_completion_semantically(goal_desc, goal_type)
            
            # Validate completion state consistency
            if completion_status["force_completion"] and not completion_status["ready_for_completion"]:
                if self.aml.state.search_attempts_count >= 3 or self.aml.state.turns_since_artifact_creation >= 5:
                    # Should be ready for completion if force is triggered
                    return False, f"Goal should be ready for completion (force={completion_status['force_completion']}, ready={completion_status['ready_for_completion']})"
            
            # Check for stuck patterns
            if (self.aml.state.search_attempts_count >= 4 and 
                self.aml.state.turns_since_artifact_creation >= 8 and
                not completion_status["force_completion"]):
                return False, "Goal appears stuck without triggering force completion"
                
        except Exception as e:
            return False, f"Goal completion validation error: {e}"
        
        return True, ""
    
    async def _validate_progress_tracking_state(self) -> Tuple[bool, str]:
        """Validate progress tracking using LLM-based assessment instead of primitive thresholds."""
        
        # Use LLM to assess if progress state is reasonable
        progress_assessment = await self._assess_progress_reasonableness_with_llm()
        
        if progress_assessment["is_reasonable"]:
            return True, ""
        else:
            return False, progress_assessment["issue_description"]
    
    async def _assess_progress_reasonableness_with_llm(self) -> Dict[str, Any]:
        """Use LLM to assess if current progress state is reasonable instead of primitive thresholds"""
        
        # Build context for LLM assessment
        current_goal = self.aml.state.goal_stack[-1] if self.aml.state.goal_stack else {}
        goal_description = current_goal.get("description", "Unknown goal")
        
        progress_context = f"""
Current Goal: "{goal_description}"
Progression Stage: {self.aml.state.plan_progression_stage}
Search Attempts: {self.aml.state.search_attempts_count}
Turns Since Artifact Creation: {self.aml.state.turns_since_artifact_creation}
Consecutive Same Tool Count: {self.aml.state.consecutive_same_tool_count}
Current Loop: {self.aml.state.current_loop}
Last Tool: {self.aml.state.last_tool_executed or "None"}
"""

        assessment_prompt = f"""
Assess if this agent's progress state is reasonable and healthy.

{progress_context}

Rate the reasonableness of this progress state on 0-100 scale:

0-25: Severely problematic state (stuck, infinite loops, excessive attempts)
26-50: Concerning state (inefficient patterns, potential issues)
51-75: Acceptable state (minor concerns but generally reasonable)
76-100: Healthy state (good progress, reasonable metrics)

Consider:
- Is the agent making reasonable progress toward the goal?
- Are search attempts reasonable for this type of goal?
- Is the time since artifact creation appropriate?
- Are there signs of being stuck or in infinite loops?
- Does the progression stage match the activity pattern?

Respond in this format:
reasonableness_score: [0-100]
primary_concern: [main issue if score < 51, or "none" if reasonable]
recommendation: [brief suggestion for improvement]
"""

        try:
            progress_assessment_schema = {
                "type": "object",
                "properties": {
                    "reasonableness_score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 100,
                        "description": "Progress reasonableness score from 0-100"
                    },
                    "primary_concern": {
                        "type": "string",
                        "description": "Main issue if score < 51, or 'none' if reasonable"
                    },
                    "recommendation": {
                        "type": "string",
                        "description": "Brief suggestion for improvement"
                    }
                },
                "required": ["reasonableness_score", "primary_concern", "recommendation"],
                "additionalProperties": False
            }
            
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": assessment_prompt}],
                response_schema=progress_assessment_schema,
                schema_name="progress_assessment",
                use_cheap_model=True,  # Use cheap/fast model
                max_tokens=100,
                temperature=0.1
            )
            
            if response:
                reasonableness_score = response.get("reasonableness_score", 60)
                primary_concern = response.get("primary_concern", "none")
                recommendation = response.get("recommendation", "continue current approach")
                
                is_reasonable = reasonableness_score >= LLM_PROGRESS_REASONABLENESS_THRESHOLD
                
                if is_reasonable:
                    return {
                        "is_reasonable": True,
                        "score": reasonableness_score,
                        "issue_description": ""
                    }
                else:
                    return {
                        "is_reasonable": False,
                        "score": reasonableness_score,
                        "issue_description": f"Progress state issue (score={reasonableness_score}): {primary_concern}. {recommendation}"
                    }
                    
        except Exception as e:
            self.logger.debug(f"LLM progress assessment failed: {e}")
            # No LLM = assume reasonable progress
            return {"is_reasonable": True, "score": 50, "issue_description": ""}
    

    
    async def _recover_goal_completion_state(self) -> bool:
        """Attempt to recover from goal completion state issues."""
        try:
            # Reset completion tracking to reasonable values
            if self.aml.state.search_attempts_count > 10:
                self.aml.state.search_attempts_count = 2
                self.aml.logger.info("🔧 RECOVERY: Reset excessive search attempts count")
            
            if self.aml.state.turns_since_artifact_creation > 15:
                self.aml.state.turns_since_artifact_creation = 5
                self.aml.logger.info("🔧 RECOVERY: Reset excessive turns without artifacts")
            
            if self.aml.state.consecutive_same_tool_count > 5:
                self.aml.state.consecutive_same_tool_count = 0
                self.aml.state.last_tool_executed = None
                self.aml.logger.info("🔧 RECOVERY: Reset infinite tool loop")
            
            # Force progression if stuck
            if (self.aml.state.search_attempts_count >= 3 and 
                self.aml.state.turns_since_artifact_creation >= 5):
                self.aml.state.plan_progression_stage = "creation"
                self.aml.logger.info("🔧 RECOVERY: Forced progression to creation stage")
            
            return True
            
        except Exception as e:
            self.aml.logger.error(f"❌ RECOVERY: Failed to recover goal completion state: {e}")
            return False
    
    async def _recover_progress_tracking_state(self) -> bool:
        """Attempt to recover from progress tracking state issues."""
        try:
            # Validate and fix progression stage
            valid_stages = {"initial", "research", "analysis", "creation", "completion"}
            if self.aml.state.plan_progression_stage not in valid_stages:
                self.aml.state.plan_progression_stage = "initial"
                self.aml.logger.info("🔧 RECOVERY: Reset invalid progression stage to 'initial'")
            
            # Reset tool sequence tracking if corrupted
            if len(self.aml.state.recent_tool_sequence) > 10:
                self.aml.state.recent_tool_sequence = self.aml.state.recent_tool_sequence[-5:]
                self.aml.logger.info("🔧 RECOVERY: Trimmed excessive tool sequence history")
            
            # Fix stage-progression inconsistencies
            if (self.aml.state.plan_progression_stage == "completion" and 
                self.aml.state.turns_since_artifact_creation > 5):
                self.aml.state.plan_progression_stage = "creation"
                self.aml.logger.info("🔧 RECOVERY: Fixed completion stage inconsistency")
            
            return True
            
        except Exception as e:
            self.aml.logger.error(f"❌ RECOVERY: Failed to recover progress tracking state: {e}")
            return False


class MemoryUtils:
    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())

def _fmt_id(val: Any, length: int = 8) -> str:
    s = str(val) if val is not None else "?"
    return s[:length] if len(s) >= length else s

def _utf8_safe_slice(s: str, max_len: int) -> str:
    if not isinstance(s, str):
        s = str(s)
    return s.encode("utf‑8")[:max_len].decode("utf‑8", "ignore")

def _safe_json_compatible(obj: Any, *, limit: int = _PAYLOAD_TRUNCATE_AT) -> Any:
    """
    Make sure an object can be JSON-serialised and is not absurdly large.
    - dict / list: recurse
    - primitives/None: leave as-is
    - everything else: repr()-string
    Strings longer than *limit* are truncated with an ellipsis.
    """
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return (obj[: limit] + "…") if len(obj) > limit else obj
    if isinstance(obj, list):
        return [_safe_json_compatible(i, limit=limit) for i in obj]
    if isinstance(obj, dict):
        return {k: _safe_json_compatible(v, limit=limit) for k, v in obj.items()}
    # fallback – e.g. datetime, custom class …
    s = repr(obj)
    return (s[: limit] + "…") if len(s) > limit else s


def _safe_json_dumps(obj: Any) -> str:
    """
    json.dumps wrapper that never raises on unserializable types.
    Falls back to `str()` for unknown objects.
    """
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        # Force full stringify fallback (rare but contracts against explosions)
        # Convert the entire object to string and then wrap in JSON
        return json.dumps(str(obj), indent=2, ensure_ascii=False)


def _truncate_context(context: Dict[str, Any], max_len: int = 25_000) -> str:  # noqa: C901 – intentionally complex
    """
    Serialize *context* to a JSON string no longer than *max_len* characters,
    applying the same structure-aware truncation strategy as the original
    implementation while fixing edge-cases, improving logging clarity, and
    guaranteeing the byte-length contract.

    Behaviour preserved:
    • Deep-copy of the object before mutation
    • Addition of " _truncation_applied" marker
    • Notes appended to truncated lists
    • Ordered two-phase strategy (shrink lists ➜ drop low-priority keys ➜ slice)
    • All existing constant limits & path semantics
    """
    # Get logger for this function
    _log = logging.getLogger(__name__)
    
    # ── 1. initial pretty dump ─────────────────────────────────────────────────
    full_json: str = _safe_json_dumps(context)
    if len(full_json) <= max_len:
        return full_json

    _log.debug("Context length %s exceeds max %s. Structured truncation begins.", len(full_json), max_len)

    # ── 2. work on a deep copy so original object is untouched ────────────────
    ctx_copy: Dict[str, Any] = copy.deepcopy(context)
    ctx_copy["_truncation_applied"] = "structure-aware_agent_side"
    original_length: int = len(full_json)

    # -------------------------------------------------------------------------
    # Helper lambdas
    # -------------------------------------------------------------------------
    def _current_size(candidate: Dict[str, Any]) -> Tuple[str, int]:
        """
        Serialize candidate and return (json_str, length) – avoids
        repeating json.dumps twice at every size check.
        """
        dumped = _safe_json_dumps(candidate)
        return dumped, len(dumped)

    def _truncate_list_at_path(path: Sequence[str | None], limit: int) -> bool:
        """
        Locate list by following *path* (skip None placeholders). If list size
        exceeds *limit*, shrink it in-place and append a note. Returns True if
        modification happened.
        """
        container = ctx_copy
        for key in path[:-1]:
            if key is None:
                continue
            if not (isinstance(container, dict) and key in container):
                return False
            container = container[key]

        target_key = path[-1]
        if not (isinstance(container, dict) and isinstance(container.get(target_key), list)):
            return False

        lst: List[Any] = container[target_key]
        if len(lst) <= limit:
            return False

        omitted = len(lst) - limit
        container[target_key] = lst[:limit]
        # Add truncation note
        container[target_key].append(
            {"truncated_note": f"{omitted} items omitted from '{'/'.join(str(p) for p in path if p)}'"}
        )
        _log.debug("Truncated list '%s' from %s to %s items.", "/".join(str(p) for p in path if p), len(lst), limit)
        return True

    def _remove_key_at_path(path: Sequence[str]) -> bool:
        """
        Remove *path[-1]* key from its container if present. Returns True when a
        key is actually removed.
        """
        container = ctx_copy
        for key in path[:-1]:
            if not (isinstance(container, dict) and key in container):
                return False
            container = container[key]
        if path[-1] in container:
            container.pop(path[-1], None)
            _log.debug("Removed low-priority key '%s' for truncation.", "/".join(map(str, path)))
            return True
        return False

    # -------------------------------------------------------------------------
    # 3A. shrink long lists first (highest information retention) -------------
    # -------------------------------------------------------------------------
    list_paths_to_truncate: Tuple[Tuple[str | None, ...], ...] = (
        ("ums_context_package", "core_context", "recent_actions", "recent_actions", CONTEXT_RECENT_ACTIONS_SHOW_LIMIT),
        ("ums_context_package", "core_context", "important_memories", "important_memories", CONTEXT_IMPORTANT_MEMORIES_SHOW_LIMIT),
        ("ums_context_package", "core_context", "key_thoughts", "key_thoughts", CONTEXT_KEY_THOUGHTS_SHOW_LIMIT),
        ("ums_context_package", "proactive_memories", "memories", "memories", CONTEXT_PROACTIVE_MEMORIES_SHOW_LIMIT),
        ("ums_context_package", "current_working_memory", "working_memories", "working_memories", CONTEXT_WORKING_MEMORY_SHOW_LIMIT),
        ("ums_context_package", "relevant_procedures", "procedures", "procedures", CONTEXT_PROCEDURAL_MEMORIES_SHOW_LIMIT),
        ("agent_assembled_goal_context", None, "goal_stack_summary_from_agent_state", "goal_stack_summary_from_agent_state", CONTEXT_GOAL_STACK_SHOW_LIMIT),
        (None, "current_plan_snapshot", None, "current_plan_snapshot", 5),
    )

    # iterate and truncate until size target met or all paths tried
    for *path, key_name, limit in list_paths_to_truncate:
        # Build real path list, replacing None placeholders
        real_path: List[str | None] = [p for p in path if p is not None] + [key_name]
        if _truncate_list_at_path(real_path, limit):
            new_json, new_len = _current_size(ctx_copy)
            if new_len <= max_len:
                _log.info("Context truncated via list reduction to %s bytes (was %s).", new_len, original_length)
                return new_json

    # -------------------------------------------------------------------------
    # 3B. remove low-priority blocks completely if still too large ------------
    # -------------------------------------------------------------------------
    keys_to_remove_low_priority: Tuple[Tuple[str, ...], ...] = (
        ("ums_context_package", "contextual_links"),
        ("ums_context_package", "relevant_procedures"),
        ("ums_context_package", "proactive_memories"),
        ("ums_context_package", "core_context", "key_thoughts"),
        ("ums_context_package", "core_context", "important_memories"),
        ("ums_context_package", "core_context", "recent_actions"),
        ("ums_context_package", "core_context"),
        ("ums_context_package", "current_working_memory"),
        ("agent_assembled_goal_context",),
        ("ums_context_package",),
    )

    for path in keys_to_remove_low_priority:
        if _remove_key_at_path(path):
            new_json, new_len = _current_size(ctx_copy)
            if new_len <= max_len:
                _log.info("Context truncated via key removal to %s bytes (was %s).", new_len, original_length)
                return new_json

    # -------------------------------------------------------------------------
    # 4. final fallback – naive byte clipping ----------------------------------
    # -------------------------------------------------------------------------
    _log.warning(
        "Structured truncation insufficient (length still %s). Falling back to raw slicing.",
        _current_size(ctx_copy)[1],
    )

    # use original *full_json* because it preserves formatting and comments
    clipped: str = _utf8_safe_slice(full_json, max_len - 50)  # keep room for marker
    last_brace = max(clipped.rfind("}"), clipped.rfind("]"))
    if last_brace > 0:
        final_str = (
            clipped[: last_brace + 1]
            + "\n// ... (CONTEXT TRUNCATED BY BYTE LIMIT) ...\n}"
        )
    else:
        final_str = clipped + "... (CONTEXT TRUNCATED)"

    # guarantee we never exceed hard limit
    final_str = _utf8_safe_slice(final_str, max_len)
    _log.error("Context severely truncated from %s to %s bytes (fallback).", original_length, len(final_str))
    return final_str


# ==========================================================================
# Dataclass & pydantic models (LOCAL DEFINITIONS)
# ==========================================================================
class PlanStep(BaseModel):
    id: str = Field(default_factory=lambda: f"step-{MemoryUtils.generate_id()[:8]}")
    description: str
    status: str = Field(default="planned")
    depends_on: List[str] = Field(default_factory=list)
    assigned_tool: Optional[str] = None  # This will be the original MCP Name (e.g., "UMS_SERVER_NAME:function_name")
    tool_args: Optional[Dict[str, Any]] = None
    result_summary: Optional[str] = None
    is_parallel_group: Optional[str] = None
    failure_count: int = Field(default=0)  # **CRITICAL FIX**: Track failures to enable plan advancement


def _default_tool_stats() -> Dict[str, Dict[str, Union[int, float]]]:
    return defaultdict(lambda: {"success": 0, "failure": 0, "latency_ms_total": 0.0})


@dataclass
class AgentState:
    workflow_id: Optional[str] = None
    context_id: Optional[str] = None
    workflow_stack: List[str] = field(default_factory=list)
    goal_stack: List[Dict[str, Any]] = field(default_factory=list)  # Stores UMS Goal objects
    current_goal_id: Optional[str] = None  # ID of the current UMS goal
    current_plan: List[PlanStep] = field(default_factory=lambda: [PlanStep(description=DEFAULT_PLAN_STEP)])
    current_thought_chain_id: Optional[str] = None
    last_action_summary: str = "Loop initialized."
    current_loop: int = 0
    goal_achieved_flag: bool = False  # For overall workflow goal achievement
    consecutive_error_count: int = 0
    needs_replan: bool = False
    last_error_details: Optional[Dict[str, Any]] = None
    # Atomic tool call processing state
    deferred_tool_calls: List[Dict[str, Any]] = field(default_factory=list)  # Tool calls deferred to next turn
    last_atomic_decision_info: Optional[Dict[str, Any]] = None  # Info about last multi-tool decision
    successful_actions_since_reflection: float = 0.0
    successful_actions_since_consolidation: float = 0.0
    loops_since_optimization: int = 0
    loops_since_promotion_check: int = 0
    loops_since_stats_adaptation: int = 0
    loops_since_maintenance: int = 0
    reflection_cycle_index: int = 0
    last_meta_feedback: Optional[str] = None
    current_reflection_threshold: int = BASE_REFLECTION_THRESHOLD
    current_consolidation_threshold: int = BASE_CONSOLIDATION_THRESHOLD
    tool_usage_stats: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=_default_tool_stats)  # Key is original MCP Name
    background_tasks: Set[asyncio.Task] = field(default_factory=set, init=False, repr=False)
    # Focus mode tracking for artifact creation
    artifact_focus_mode: bool = False  # True when actively working on deliverables
    consecutive_artifact_actions: int = 0  # Track artifact-related actions
    last_context_cache: Optional[Dict[str, Any]] = None
    context_cache_timestamp: float = 0.0
    context_cache_plan_hash: str = ""
    successful_patterns: Dict[str, List[List[str]]] = field(default_factory=dict)  # goal_type -> list of tool_sequences  
    pattern_success_count: Dict[str, int] = field(default_factory=dict)  # pattern_hash -> success_count
    last_workflow_tools: List[str] = field(default_factory=list)  # Track tools used in current workflow
    
    # NEW: Loop detection and plan progression tracking
    recent_tool_sequence: List[str] = field(default_factory=list)  # Last 5 tools executed
    consecutive_same_tool_count: int = 0  # Count of same tool executed consecutively
    last_tool_executed: Optional[str] = None  # Last tool name executed
    plan_progression_stage: str = "initial"  # Track: initial, research, analysis, creation, completion
    turns_since_artifact_creation: int = 1  # Count turns since last artifact was created (starts at 1, resets to 0 only when artifact is actually created)
    search_attempts_count: int = 0  # Count search attempts to prevent endless searching

# =====================================================================
# Agent Master Loop
# =====================================================================
class AgentMasterLoop(ErrorHandlerMixin):
    # This set should contain the BASE FUNCTION NAMES of UMS tools considered meta/internal.
    # Example: "record_action_start", "get_workflow_details", etc.
    _INTERNAL_OR_META_TOOLS_BASE_NAMES: Set[str] = {
        UMS_FUNC_RECORD_ACTION_START,
        UMS_FUNC_RECORD_ACTION_COMPLETION,
        UMS_FUNC_CREATE_WORKFLOW,  # Added: create_workflow doesn't record actions (no workflow context yet)
        "get_workflow_context",
        UMS_FUNC_GET_RICH_CONTEXT_PACKAGE,
        UMS_FUNC_GET_WORKING_MEMORY,
        UMS_FUNC_SEARCH_SEMANTIC_MEMORIES,
        UMS_FUNC_HYBRID_SEARCH,
        UMS_FUNC_QUERY_MEMORIES,
        UMS_FUNC_GET_MEMORY_BY_ID,
        UMS_FUNC_GET_LINKED_MEMORIES,
        UMS_FUNC_GET_ACTION_DETAILS,
        UMS_FUNC_GET_ARTIFACTS,
        UMS_FUNC_GET_ARTIFACT_BY_ID,
        UMS_FUNC_GET_ACTION_DEPENDENCIES,
        UMS_FUNC_GET_THOUGHT_CHAIN,
        UMS_FUNC_GET_WORKFLOW_DETAILS,
        UMS_FUNC_GET_GOAL_DETAILS,
        UMS_FUNC_LIST_WORKFLOWS,
        UMS_FUNC_COMPUTE_STATS,
        UMS_FUNC_SUMMARIZE_TEXT,
        UMS_FUNC_SUMMARIZE_CONTEXT_BLOCK,
        UMS_FUNC_OPTIMIZE_WM,
        UMS_FUNC_AUTO_FOCUS,
        UMS_FUNC_PROMOTE_MEM,
        UMS_FUNC_REFLECTION,
        UMS_FUNC_CONSOLIDATION,
        UMS_FUNC_DELETE_EXPIRED_MEMORIES,
        UMS_FUNC_GET_RECENT_ACTIONS,
        # Note: AGENT_TOOL_UPDATE_PLAN is handled separately by its full name.
    }
    # Tools that the agent calls which represent direct actions towards its goal,
    # even if they are UMS tools, might NOT be in the above set if they should be recorded as main actions.
    # E.g., store_memory, record_artifact, create_goal when LLM decides.

    def __init__(
        self,
        mcp_client_instance: "MCPClient",
        default_llm_model_string: str,
        agent_state_file: str = AGENT_STATE_FILE,
    ):
        """
        Parameters
        ----------
        mcp_client_instance : MCPClient
            Fully-configured MCPClient that already owns an `AsyncAnthropic` instance and any
            tool-schema metadata the agent will need.
        default_llm_model_string : str
            The model name the agent should request from Anthropic if a prompt does not explicitly
            override it (e.g. ``"claude-3-opus-20240229"``).
        agent_state_file : str, optional
            Path to a JSON file for persisting incremental state across runs.  Defaults to the
            module-level constant ``AGENT_STATE_FILE``.
        """
        # ------------------------------------------------------------------ #
        # 0. Guard-rail validation                                           #
        # ------------------------------------------------------------------ #
        if not isinstance(default_llm_model_string, str) or not default_llm_model_string.strip():
            raise ValueError("`default_llm_model_string` must be a non-empty string.")

        # MCPClient typing check (duck-typing is OK, but a hard fail early is clearer for users)
        if type(mcp_client_instance).__name__ != "MCPClient":
            raise TypeError(
                f"`mcp_client_instance` expected MCPClient by name; got {type(mcp_client_instance).__name__!s}"
            )

        self.mcp_client: MCPClient = mcp_client_instance
        self.agent_llm_model: str = default_llm_model_string.strip()

        # ------------------------------------------------------------------ #
        # 1. Logger / Paths                                                  #
        # ------------------------------------------------------------------ #
        self.logger = logging.getLogger("AgentMasterLoop.AgentMasterLoop")
        self.agent_state_file = Path(agent_state_file).expanduser().resolve()

        # ------------------------------------------------------------------ #
        # 2. Anthropic client presence check                                 #
        # ------------------------------------------------------------------ #
        from anthropic import AsyncAnthropic  # postponed import for optional dep
        if not hasattr(self.mcp_client, "anthropic") or not isinstance(
            self.mcp_client.anthropic, AsyncAnthropic
        ):
            self.logger.critical("CRITICAL: MCPClient instance missing valid `AsyncAnthropic` client.")
            raise ValueError("MCPClient instance missing valid `AsyncAnthropic` client.")
        self.anthropic_client: AsyncAnthropic = self.mcp_client.anthropic

        # ------------------------------------------------------------------ #
        # 3. Tunable heuristics & thresholds (can be tweaked by reflection)  #
        # ------------------------------------------------------------------ #
        self.consolidation_memory_level: str = MemoryLevel.EPISODIC.value
        self.consolidation_max_sources: int = 10
        self.auto_linking_threshold: float = 0.7
        self.auto_linking_max_links: int = 3
        # Make a *copy* so external mutations to the module-level constant don't affect runtime
        self.reflection_type_sequence: list[str] = [
            "summary",
            "progress",
            "gaps",
            "strengths",
            "plan",
        ]

        # ------------------------------------------------------------------ #
        # 4. Core runtime state & concurrency primitives                     #
        # ------------------------------------------------------------------ #
        self.state = AgentState()
        
        # State management components for Fix C: Atomic State Management
        self.state_transaction_manager = StateTransactionManager(self)
        self.state_validator = EnhancedStateValidator(self)
        self.memory_manager = IntelligentMemoryManager(self)
        
        self._shutdown_event = asyncio.Event()
        self._bg_tasks_lock = asyncio.Lock()
        self._bg_task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BG_TASKS)

        # ------------------------------------------------------------------ #
        # 4a. Initialize LRU caches with proper size limits                  #
        # ------------------------------------------------------------------ #
        self._caches = {
            'complexity_score': LRUCache(maxsize=1000),
            'research_intensity': LRUCache(maxsize=1000),
            'creation_focus': LRUCache(maxsize=1000),
            'deliverable_type': LRUCache(maxsize=1000),
            'multi_part': LRUCache(maxsize=1000),
            'integration': LRUCache(maxsize=1000),
            'action_phase': LRUCache(maxsize=1000),
            'goal_classification': LRUCache(maxsize=1000),
            'tool_suggestion': LRUCache(maxsize=500),
            'memory_type': LRUCache(maxsize=1000),
            'artifact_type': LRUCache(maxsize=1000),
            'similarity': LRUCache(maxsize=1000),
            'error_classification': LRUCache(maxsize=500),
            'tool_analysis': LRUCache(maxsize=500),
            'content_format': LRUCache(maxsize=1000),
            'replan_feedback': LRUCache(maxsize=500),
            'file_creation': LRUCache(maxsize=1000),
            'error_context': LRUCache(maxsize=500),
            'complex_goal': LRUCache(maxsize=1000),
            'vague_plan': LRUCache(maxsize=1000),
            'workflow_analysis': LRUCache(maxsize=500),
            'content_similarity': LRUCache(maxsize=1000),
        }

        # Populated during `initialize()` by MCPClient; keeping list here avoids attribute-errors
        self.tool_schemas: list[dict[str, Any]] = []

        # ------------------------------------------------------------------ #
        # 4b. Artifact file management                                       #
        # ------------------------------------------------------------------ #
        self._task_artifact_directories: Dict[str, str] = {}  # workflow_id -> directory_path
        self._artifact_base_dir = "/home/ubuntu/ultimate_mcp_server/storage/generated_agent_artifacts"

        # ------------------------------------------------------------------ #
        # 5. Initialize helper classes for refactored patterns               #
        # ------------------------------------------------------------------ #
        self.llm_helper = LLMClassificationHelper(self)
        self.tool_cache = ToolServerCache(self)
        self.simple_bg_manager = SimpleBackgroundTaskManager(self)
        
        # ------------------------------------------------------------------ #
        # 6. Build quick-lookup set of all UMS base-function names           #
        #    (e.g., {"create_workflow", "save_memory", …})                   #
        # ------------------------------------------------------------------ #
        self.all_ums_base_function_names: set[str] = {
            v
            for k, v in globals().items()
            if k.startswith("UMS_FUNC_") and isinstance(v, str)
        }

        # ------------------------------------------------------------------ #
        # 6. Final diagnostics                                               #
        # ------------------------------------------------------------------ #
        self.logger.info(
            "AgentMasterLoop initialised. "
            "LLM=%s | UMS Server='%s' | known_ums_funcs=%d | state_file=%s",
            self.agent_llm_model,
            UMS_SERVER_NAME,
            len(self.all_ums_base_function_names),
            self.agent_state_file,
        )
        self.logger.debug(
            "First five UMS base-function names: %s",
            sorted(self.all_ums_base_function_names)[:5],
        )

    def clear_caches(self) -> None:
        """Clear all LRU caches to free memory."""
        for cache_name, cache in self._caches.items():
            cache.clear()
        self.logger.debug("Cleared all LRU caches")

    async def shutdown(self) -> None:
        """
        Gracefully shut the AgentMasterLoop down.

        Guarantees
        ----------
        • *Idempotent* – subsequent calls are no-ops once shutdown completes.  
        • `_shutdown_event` is set immediately so any in-flight coroutines can
        observe it and exit early.  
        • All background tasks spawned through the agent are awaited (with a
        timeout) to avoid orphaned tasks.  
        • Agent state is *always* persisted, even if cleanup raises.  
        • If the current workflow is in a **terminal** condition
        (goal achieved, max errors hit, or loop budget exhausted) the workflow
        state is cleared so that the next run starts cleanly.  No existing
        functionality is removed.
        """
        # ------------------------------------------------------------------
        # 0. Fast-path: ignore duplicate shutdown requests
        # ------------------------------------------------------------------
        if getattr(self, "_shutdown_completed", False):
            self.logger.debug("shutdown(): already completed – duplicate call ignored.")
            return

        # ------------------------------------------------------------------
        # 1. Ensure only one coroutine can perform the shutdown sequence
        # ------------------------------------------------------------------
        if not hasattr(self, "_shutdown_lock"):
            self._shutdown_lock = asyncio.Lock()
        async with self._shutdown_lock:
            if getattr(self, "_shutdown_completed", False):
                return  # someone else just finished while we awaited the lock

            self.logger.info("Shutdown requested.")
            self._shutdown_event.set()

            # ------------------------------------------------------------------
            # 2. Clean up background tasks with a defensive timeout
            # ------------------------------------------------------------------
            try:
                timeout_sec = getattr(self, "SHUTDOWN_CLEANUP_TIMEOUT_SEC", 15)
                await asyncio.wait_for(
                    asyncio.shield(self._cleanup_background_tasks()),
                    timeout=timeout_sec,
                )
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"shutdown(): cleanup timed-out after {timeout_sec}s – background tasks may be cancelled."
                )
            except Exception as e:
                self.logger.exception(f"shutdown(): unexpected error during background-task cleanup – {e!r}")

            # ------------------------------------------------------------------
            # 3. Decide whether the active workflow should be cleared
            # ------------------------------------------------------------------
            try:
                max_loops_this_run = getattr(self, "_current_max_loops_for_run", float("inf"))
                is_workflow_terminal = (
                    self.state.goal_achieved_flag
                    or self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS
                    or self.state.current_loop >= max_loops_this_run
                )

                if self.state.workflow_id and is_workflow_terminal:
                    wf_fmt = _fmt_id(self.state.workflow_id)
                    self.logger.info(
                        f"AML Shutdown: Workflow '{wf_fmt}' is terminal – extracting database artifacts to files."
                    )
                    
                    # Extract text/data artifacts from database to physical files before clearing state
                    try:
                        await self._extract_database_artifacts_to_files(self.state.workflow_id)
                    except Exception as extract_err:
                        self.logger.error(f"Error during artifact extraction: {extract_err}", exc_info=True)
                    
                    # Purge workflow-scoped artefacts
                    self.state.workflow_id = None
                    self.state.context_id = None
                    self.state.workflow_stack.clear()
                    self.state.goal_stack.clear()
                    self.state.current_goal_id = None
                    self.state.current_thought_chain_id = None
                    self.state.current_plan = [PlanStep(description=DEFAULT_PLAN_STEP)]
                    self.state.last_action_summary = "Agent shut down; workflow state cleared."
                    self.state.needs_replan = False
                    self.state.last_error_details = None
                elif self.state.workflow_id:
                    self.logger.info(
                        f"AML Shutdown: Workflow '{_fmt_id(self.state.workflow_id)}' not terminal – state persisted for later resumption."
                    )
                else:
                    self.logger.info("AML Shutdown: No active workflow to clear.")
            finally:
                # ------------------------------------------------------------------
                # 4. Persist agent state unconditionally
                # ------------------------------------------------------------------
                try:
                    await self._save_agent_state()
                    self.logger.info("shutdown(): agent state successfully persisted.")
                except Exception as e_save:
                    # Do not propagate – shutdown must always finish.
                    self.logger.exception(f"shutdown(): failed to persist state – {e_save!r}")

            # ------------------------------------------------------------------
            # 5. Mark shutdown complete (idempotency flag) and log
            # ------------------------------------------------------------------
            self._shutdown_completed = True
            self.logger.info("Agent loop shutdown complete.")

    def _get_ums_tool_mcp_name(self, base_function_name: str) -> str:
        """Constructs the full original MCP tool name for a UMS base function."""
        return f"{UMS_SERVER_NAME}:{base_function_name}"
    
    async def _extract_database_artifacts_to_files(self, workflow_id: str) -> None:
        """
        Extract text/data artifacts stored in UMS database to physical files.
        
        This ensures all user-accessible content is available as files, not just in the database.
        Called during terminal workflow shutdown to preserve all created artifacts.
        """
        if not workflow_id:
            self.logger.warning("Cannot extract artifacts: no workflow_id provided")
            return
        
        # Query UMS for artifacts in this workflow that are stored in database (not as files)
        get_artifacts_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_GET_ARTIFACTS)
        if not self._find_tool_server(get_artifacts_mcp):
            self.logger.error("Cannot extract artifacts: get_artifacts tool unavailable")
            return
        
        try:
            # Get all artifacts for this workflow
            artifacts_result = await self._execute_tool_call_internal(
                get_artifacts_mcp,
                {
                    "workflow_id": workflow_id,
                    "include_content": True,  # We need the actual content
                    "artifact_types": ["text", "data", "json"]  # Types that might be stored in DB
                },
                record_action=False
            )
            
            if not artifacts_result.get("success"):
                self.logger.warning(f"Failed to query artifacts for extraction: {artifacts_result.get('error_message')}")
                return
            
            artifacts = artifacts_result.get("data", {}).get("artifacts", [])
            if not artifacts:
                self.logger.info("No database artifacts to extract to files")
                return
            
            # Determine the output directory from existing file artifacts
            output_dir = self._get_artifacts_output_directory(workflow_id, artifacts)
            if not output_dir:
                output_dir = "/home/ubuntu/ultimate_mcp_server/storage"
            
            extracted_count = 0
            for artifact in artifacts:
                try:
                    if await self._extract_single_artifact_to_file(artifact, output_dir):
                        extracted_count += 1
                except Exception as e:
                    self.logger.error(f"Failed to extract artifact {artifact.get('artifact_id', 'unknown')}: {e}")
                    continue
            
            if extracted_count > 0:
                self.logger.info(f"📁 Extracted {extracted_count} database artifacts to files in {output_dir}")
            else:
                self.logger.info("No database artifacts needed extraction (all already available as files)")
                
        except Exception as e:
            self.logger.error(f"Error during artifact extraction process: {e}", exc_info=True)
    
    def _get_artifacts_output_directory(self, workflow_id: str, artifacts: List[Dict]) -> Optional[str]:
        """Determine output directory by finding where file artifacts were already saved."""
        for artifact in artifacts:
            artifact_type = artifact.get("artifact_type", "").lower()
            file_path = artifact.get("file_path", "")
            if artifact_type == "file" and file_path:
                # Extract directory from existing file path
                return os.path.dirname(file_path)
        
        # Default fallback
        return "/home/ubuntu/ultimate_mcp_server/storage"
    
    async def _extract_single_artifact_to_file(self, artifact: Dict, output_dir: str) -> bool:
        """
        Extract a single artifact to a physical file.
        
        Returns True if extraction was performed, False if skipped.
        """
        artifact_id = artifact.get("artifact_id", "unknown")
        artifact_type = artifact.get("artifact_type", "").lower()
        name = artifact.get("name", f"artifact_{artifact_id}")
        content = artifact.get("content", "")
        file_path = artifact.get("file_path", "")
        
        # Skip if already a file or has no content
        if artifact_type == "file" or file_path or not content:
            return False
        
        # Generate appropriate filename and extension
        safe_filename = self._sanitize_filename(name)
        file_extension = self._get_file_extension_for_artifact_type(artifact_type, content)
        
        if not safe_filename.endswith(file_extension):
            safe_filename += file_extension
        
        output_path = os.path.join(output_dir, safe_filename)
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle potential filename conflicts
        if os.path.exists(output_path):
            base_name, ext = os.path.splitext(safe_filename)
            counter = 1
            while os.path.exists(output_path):
                new_filename = f"{base_name}_extracted_{counter}{ext}"
                output_path = os.path.join(output_dir, new_filename)
                counter += 1
        
        try:
            # Write content to file using async file operations
            async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            self.logger.info(f"📄 Extracted '{name}' ({artifact_type}) → {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to write artifact {artifact_id} to {output_path}: {e}")
            return False
    
    def _sanitize_filename(self, name: str) -> str:
        """Convert artifact name to safe filename."""
        # Remove or replace unsafe characters
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', name)
        safe_name = re.sub(r'[^\w\s\-_.]', '_', safe_name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name.strip('._')
        
        # Ensure reasonable length
        if len(safe_name) > 100:
            safe_name = safe_name[:100]
        
        return safe_name or "unnamed_artifact"
    
    def _get_file_extension_for_artifact_type(self, artifact_type: str, content: str) -> str:
        """Determine appropriate file extension based on artifact type and content."""
        artifact_type = artifact_type.lower()
        
        # Type-based extensions
        if artifact_type == "json":
            return ".json"
        elif artifact_type == "code":
            # Use semantic content analysis instead of hardcoded patterns
            content_analysis = self._analyze_content_format_semantically(content)
            if content_analysis["primary_format"] == "html":
                return ".html"
            elif content_analysis["primary_format"] == "python":
                return ".py"
            elif content_analysis["primary_format"] == "javascript":
                return ".js"
            elif content_analysis["primary_format"] == "c_cpp":
                return ".c"
            else:
                return ".txt"
        elif artifact_type == "data":
            # Use semantic content analysis for data format detection
            content_analysis = self._analyze_content_format_semantically(content)
            data_format = content_analysis["primary_format"]
            if data_format == "json":
                return ".json"
            elif data_format == "csv":
                return ".csv"
            elif data_format == "xml":
                return ".xml"
            else:
                return ".txt"
        elif artifact_type == "text":
            # Use LLM semantic understanding to detect markdown content
            format_analysis = self._analyze_content_format_semantically(content)
            if format_analysis.get("primary_format") == "markdown":
                return ".md"
            else:
                return ".txt"
        else:
            return ".txt"  # Default fallback

    def _create_task_artifact_directory(self, workflow_id: str) -> str:
        """
        Create and return the task-specific directory path for artifacts.
        
        Directory structure: /home/ubuntu/ultimate_mcp_server/storage/generated_agent_artifacts/[task_name]_[task_id]/
        
        Returns the full path to the created directory.
        """
        if workflow_id in self._task_artifact_directories:
            return self._task_artifact_directories[workflow_id]
        
        try:
            # Get the task name from the current goal
            task_name = "agent_task"
            if self.state.goal_stack and self.state.goal_stack[0]:
                root_goal = self.state.goal_stack[0]
                if isinstance(root_goal, dict) and root_goal.get("description"):
                    task_name = root_goal["description"]
                elif isinstance(root_goal, dict) and root_goal.get("title"):
                    task_name = root_goal["title"]
            
            # Sanitize task name for use as directory name
            safe_task_name = self._sanitize_task_name_for_directory(task_name)
            
            # Create directory name with task name and truncated workflow ID
            truncated_id = workflow_id[:8] if workflow_id else "unknown"
            dir_name = f"{safe_task_name}_{truncated_id}"
            
            # Full directory path
            full_dir_path = os.path.join(self._artifact_base_dir, dir_name)
            
            # Create the directory
            os.makedirs(full_dir_path, exist_ok=True)
            
            # Cache the directory path
            self._task_artifact_directories[workflow_id] = full_dir_path
            
            self.logger.info(f"📁 Created task artifact directory: {full_dir_path}")
            return full_dir_path
            
        except Exception as e:
            self.logger.error(f"Failed to create task artifact directory: {e}")
            # Fallback to a safe directory
            fallback_dir = os.path.join(self._artifact_base_dir, f"fallback_{workflow_id[:8] if workflow_id else 'unknown'}")
            os.makedirs(fallback_dir, exist_ok=True)
            self._task_artifact_directories[workflow_id] = fallback_dir
            return fallback_dir
    
    def _sanitize_task_name_for_directory(self, task_name: str) -> str:
        """Convert a task description into a safe directory name."""
        if not task_name:
            return "unnamed_task"
        
        # Convert to lowercase and replace spaces with underscores
        safe_name = task_name.lower().replace(" ", "_")
        
        # Remove or replace unsafe characters for directory names
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', safe_name)
        safe_name = re.sub(r'[^\w\-_.]', '_', safe_name)
        safe_name = re.sub(r'_+', '_', safe_name)  # Collapse multiple underscores
        safe_name = safe_name.strip('._')  # Remove leading/trailing dots and underscores
        
        # Truncate to reasonable length
        if len(safe_name) > 50:
            safe_name = safe_name[:50].rstrip('_')
        
        return safe_name or "unnamed_task"
    
    async def _save_artifact_to_file(self, artifact_data: Dict[str, Any], workflow_id: str) -> Optional[str]:
        """
        Save an artifact to a file in the task-specific directory.
        
        Args:
            artifact_data: The artifact data from UMS (should include content, name, artifact_type, etc.)
            workflow_id: The workflow ID to determine the directory
            
        Returns:
            The file path where the artifact was saved, or None if saving failed
        """
        try:
            if not artifact_data or not isinstance(artifact_data, dict):
                self.logger.warning("Invalid artifact data for file saving")
                return None
            
            content = artifact_data.get("content", "")
            if not content:
                self.logger.warning("Artifact has no content to save")
                return None
            
            # Get artifact details
            artifact_name = artifact_data.get("name", "unnamed_artifact")
            artifact_type = artifact_data.get("artifact_type", "text")
            artifact_id = artifact_data.get("artifact_id", "unknown")  # noqa: F841
            
            # Get or create the task directory
            task_dir = self._create_task_artifact_directory(workflow_id)
            
            # Generate filename
            safe_name = self._sanitize_filename(artifact_name)
            file_extension = self._get_file_extension_for_artifact_type(artifact_type, content)
            
            if not safe_name.endswith(file_extension):
                safe_name += file_extension
            
            # Handle filename conflicts by checking existing files
            base_name, ext = os.path.splitext(safe_name)
            counter = 1
            final_filename = safe_name
            final_path = os.path.join(task_dir, final_filename)
            
            while os.path.exists(final_path):
                final_filename = f"{base_name}_v{counter}{ext}"
                final_path = os.path.join(task_dir, final_filename)
                counter += 1
            
            # Write the file
            async with aiofiles.open(final_path, 'w', encoding='utf-8') as f:
                await f.write(content)
            
            self.logger.info(f"💾 Saved artifact '{artifact_name}' ({artifact_type}) → {final_path}")
            return final_path
            
        except Exception as e:
            self.logger.error(f"Failed to save artifact to file: {e}", exc_info=True)
            return None
    
    async def _detect_and_update_focus_mode(self, tool_name: str, success: bool) -> None:
        """
        Detect if the agent is in artifact creation focus mode and adjust thresholds accordingly.
        
        Focus mode is activated when:
        - Agent is actively creating, writing, or working on deliverables
        - Plan contains artifact creation steps
        - Recent actions suggest artifact work
        - Research workflows with web searching (ENHANCED)
        """

        
        # Use semantic classification for tool and workflow analysis
        is_artifact_tool = await self._analyze_tool_type_with_llm(tool_name, "artifact_tool")
        is_research_tool = await self._analyze_tool_type_with_llm(tool_name, "research_tool")
        
        # Check if goal suggests research workflow using semantic understanding
        goal_suggests_research = False
        if self.state.goal_stack and self.state.goal_stack[-1]:
            goal_desc = self.state.goal_stack[-1].get("description", "")
            goal_suggests_research = await self._analyze_workflow_type_with_llm(goal_desc, "research_workflow")
        
        # Check if plan suggests artifact work OR research work using semantic analysis
        plan_suggests_artifacts = False
        plan_suggests_research = False
        if self.state.current_plan:
            current_step_desc = self.state.current_plan[0].description
            plan_suggests_artifacts = await self._analyze_workflow_type_with_llm(current_step_desc, "artifact_work")
            plan_suggests_research = await self._analyze_workflow_type_with_llm(current_step_desc, "research_work")
        
        # NEW: Enter focus mode for research workflows earlier
        if success and (is_research_tool or goal_suggests_research or plan_suggests_research):
            self.state.consecutive_artifact_actions += 1
            # Enter focus mode after 2 research actions (less aggressive)
            if self.state.consecutive_artifact_actions >= 2 and not self.state.artifact_focus_mode:
                self.state.artifact_focus_mode = True
                self.logger.info("🎯 RESEARCH FOCUS MODE ACTIVATED: Agent is actively researching/gathering information")
        
        # Original artifact focus logic (keep threshold at 2 for actual artifacts)
        elif success and (is_artifact_tool or plan_suggests_artifacts):
            self.state.consecutive_artifact_actions += 1
            # Enter focus mode after 2 consecutive artifact actions
            if self.state.consecutive_artifact_actions >= 2:
                if not self.state.artifact_focus_mode:
                    self.state.artifact_focus_mode = True
                    self.logger.info("🎯 ARTIFACT FOCUS MODE ACTIVATED: Agent is actively working on deliverables")
        else:
            # Reset counter if not doing productive work
            if self.state.consecutive_artifact_actions > 0:
                self.state.consecutive_artifact_actions = max(0, self.state.consecutive_artifact_actions - 1)
            
            # Exit focus mode if we haven't done productive work for a while OR periodically check for completion
            should_exit_focus = (
                self.state.consecutive_artifact_actions == 0 or
                (self.state.current_loop > 0 and self.state.current_loop % 7 == 0)  # Check every 7 loops
            )
            if should_exit_focus and self.state.artifact_focus_mode:
                self.state.artifact_focus_mode = False
                self.logger.info("🎯 FOCUS MODE DEACTIVATED: Returning to normal meta-cognition levels")
    
    async def _get_effective_thresholds(self) -> Tuple[int, int]:
        """Get the current effective reflection and consolidation thresholds, adjusted for focus mode."""
        if self.state.artifact_focus_mode:
            # Be even more aggressive about reducing meta-cognition during research
            is_research_workflow = False
            if self.state.goal_stack and self.state.goal_stack[-1]:
                goal_desc = self.state.goal_stack[-1].get("description", "")
                is_research_workflow = await self._analyze_workflow_type_with_llm(goal_desc, "research_workflow")
            
            if is_research_workflow:
                # Higher multipliers for research workflows
                reflection_threshold = int(self.state.current_reflection_threshold * (FOCUS_MODE_REFLECTION_MULTIPLIER * 1.5))
                consolidation_threshold = int(self.state.current_consolidation_threshold * (FOCUS_MODE_CONSOLIDATION_MULTIPLIER * 1.5))
            else:
                reflection_threshold = int(self.state.current_reflection_threshold * FOCUS_MODE_REFLECTION_MULTIPLIER)
                consolidation_threshold = int(self.state.current_consolidation_threshold * FOCUS_MODE_CONSOLIDATION_MULTIPLIER)
            return reflection_threshold, consolidation_threshold
        else:
            return self.state.current_reflection_threshold, self.state.current_consolidation_threshold
    
    async def _construct_agent_prompt(self, current_task_goal_desc: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build the gigantic system+user prompt that steers the LLM each turn.
        """
        dbg = self.logger.isEnabledFor(logging.DEBUG)

        # ---------- 1. High-level logging about current state ----------------
        current_turn_for_log_prompt = self.state.current_loop
        self.logger.info(f"AML CONSTRUCT_PROMPT: Building prompt for turn {current_turn_for_log_prompt}")

        if dbg:
            self.logger.debug(f"AML CONSTRUCT_PROMPT: self.state.workflow_id = {_fmt_id(self.state.workflow_id)}")
            self.logger.debug(f"AML CONSTRUCT_PROMPT: self.state.current_goal_id = {_fmt_id(self.state.current_goal_id)}")
            goal_stack_summary = [
                {
                    "id": _fmt_id(g.get("goal_id")),
                    "desc": (g.get("description") or "")[:30] + "...",
                    "status": g.get("status"),
                }
                for g in self.state.goal_stack
                if isinstance(g, dict)
            ]
            self.logger.debug(
                "AML CONSTRUCT_PROMPT: self.state.goal_stack (summary) = %s",
                goal_stack_summary,
            )
            self.logger.debug(f"AML CONSTRUCT_PROMPT: current_task_goal_desc (param for this prompt) = {current_task_goal_desc[:100]}...")

        # ---------- 2. Helper lambdas ---------------------------------------
        def _schema_name(schema_item: Any) -> str | None:
            if not isinstance(schema_item, dict):
                return None
            if schema_item.get("type") == "function" and isinstance(schema_item.get("function"), dict):
                return schema_item["function"].get("name")
            return schema_item.get("name")

        def _safe_json(obj: Any, **dump_kwargs) -> str:
            try:
                return json.dumps(obj, ensure_ascii=False, **dump_kwargs)
            except Exception:
                return repr(obj)

        # ---------- 3. Discover the LLM alias for agent:update_plan ----------
        llm_seen_agent_update_plan_name_for_instr = AGENT_TOOL_UPDATE_PLAN  # default
        try:
            for schema_item in self.tool_schemas or []:
                llm_seen_name = _schema_name(schema_item)
                if not llm_seen_name:
                    continue
                original_mcp = self.mcp_client.server_manager.sanitized_to_original.get(llm_seen_name)
                if original_mcp == AGENT_TOOL_UPDATE_PLAN:
                    llm_seen_agent_update_plan_name_for_instr = llm_seen_name
                    break
        except Exception as e:
            self.logger.warning(f"AML CONSTRUCT_PROMPT: Exception while resolving update_plan alias: {e}", exc_info=dbg)
        if not llm_seen_agent_update_plan_name_for_instr:
            llm_seen_agent_update_plan_name_for_instr = re.sub(r"[^a-zA-Z0-9_-]", "_", AGENT_TOOL_UPDATE_PLAN)[:64] or "agent_update_plan_fallback"

        # ---------- 4. Assemble SYSTEM prompt blocks (with tweaks) -----------
        # Check if we're in focus mode
        focus_mode_prefix = ""
        if context.get("artifact_focus_mode"):
            focus_mode_prefix = "🎯 **FOCUS MODE ACTIVE** - " + context.get("focus_mode_message", "Maintaining productivity flow") + "\n\n"

        system_blocks: list[str] = [
            focus_mode_prefix + f"You are '{AGENT_NAME}', an AI agent orchestrator using a Unified Memory System (UMS) provided by the '{UMS_SERVER_NAME}' server.",
            "",
            "🎯 **CRITICAL OUTPUT FORMAT REQUIREMENTS:**",
            "• Your response MUST be either:",
            "  1. A single, valid JSON object for a tool call (no extra text, no markdown wrapping)",
            "  2. OR the exact text 'Goal Achieved...' (for workflow completion)",
            "• Examples of CORRECT tool call format:",
            f'  {{"name": "{llm_seen_agent_update_plan_name_for_instr}", "arguments": {{"plan": [...]}}}}',
            f'  {{"name": "ums_record_thought", "arguments": {{"content": "..."}}}}',
            "• INCORRECT formats that will cause failures:",
            "  - Markdown code blocks: ```json {...} ```",
            "  - Extra text before/after JSON: 'Here is my response: {...}'",
            "  - Multiple JSON objects or tool calls in one response",
            "",
            "🚀 **EFFICIENCY PRIORITIES:**",
            "• FOCUS ON DELIVERABLES: Prioritize creating artifacts (reports, code, files) over meta-analysis",
            "• MULTI-TOOL EXECUTION: The system can execute multiple related tools in one turn - be decisive!", 
            "• MINIMIZE TURNS: Aim to complete tasks in 3-5 turns instead of 20+",
            "• BATCH RELATED ACTIONS: Group search → analyze → write operations together",
            "",
        ]

        agent_status_message = context.get("status_message_from_agent", "Status unknown.")
        if dbg:
            self.logger.debug(f"AML CONSTRUCT_PROMPT: context['status_message_from_agent'] = {agent_status_message}")
            ctx_goal_det = context.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
            self.logger.debug(
                "AML CONSTRUCT_PROMPT: context['agent_assembled_goal_context']['current_goal_details_from_ums'] = "
                f"{({'id': _fmt_id(ctx_goal_det.get('goal_id')), 'desc': (ctx_goal_det.get('description') or '')[:50] + '...'} if isinstance(ctx_goal_det, dict) else ctx_goal_det)}"
            )

        if not self.state.workflow_id:
            system_blocks.append(
                f"Current State: NO ACTIVE UMS WORKFLOW. (Agent Status: {agent_status_message}) Your immediate primary objective is to establish one using the task description below."
            )
            system_blocks.append(f"Initial Overall Task Description (from user/MCPClient): {current_task_goal_desc}")
            system_blocks.append(
                f"**Action Required: You MUST first call the UMS tool whose base function is '{UMS_FUNC_CREATE_WORKFLOW}'.** Use the 'Initial Overall Task Description' above as the 'goal' parameter for this tool. Provide a suitable 'title'."
            )
        else:
            ums_workflow_goal_from_context = context.get("ums_context_package", {}).get("core_context", {}).get("workflow_goal", "N/A")
            system_blocks.append(f"Active UMS Workflow ID: {_fmt_id(self.state.workflow_id)} (on server '{UMS_SERVER_NAME}')")
            system_blocks.append(f"Overall UMS Workflow Goal: {ums_workflow_goal_from_context}")

            current_operational_goal_details = context.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
            if (
                current_operational_goal_details
                and isinstance(current_operational_goal_details, dict)
                and current_operational_goal_details.get("goal_id")
            ):
                desc = current_operational_goal_details.get("description", "N/A")
                gid = _fmt_id(current_operational_goal_details.get("goal_id"))
                status = current_operational_goal_details.get("status", "N/A")
                system_blocks.append(f"Current Operational UMS Goal: {desc} (ID: {gid}, Status: {status})")
            else:
                system_blocks += [
                    f"Current State: UMS Workflow '{_fmt_id(self.state.workflow_id)}' is ACTIVE, but NO specific UMS operational goal is currently set in agent's focus. (Agent Status: {agent_status_message})",
                    f"The Overall UMS Workflow Goal is: {ums_workflow_goal_from_context}",
                    f"**Action Required: Your next step should be to establish the primary UMS operational goal for this workflow.**",
                    f"   - If the Overall UMS Workflow Goal ('{ums_workflow_goal_from_context[:50]}...') is suitable as the first operational UMS goal, use the UMS tool with base function '{UMS_FUNC_CREATE_GOAL}' to create it. Set `parent_goal_id` to `null` or omit it. Use the Overall UMS Workflow Goal as the description for this new UMS goal.",
                    f"   - Then, update your plan using the tool named `{llm_seen_agent_update_plan_name_for_instr}` to reflect steps towards this new UMS goal.",
                ]

        system_blocks += [
            "",
            "Available Tools (Use ONLY these for UMS/Agent actions; format arguments per schema. Refer to tools by 'Name LLM Sees'):",
        ]

        if not self.tool_schemas:
            system_blocks.append("- CRITICAL WARNING: No tools loaded into agent's schema list. Cannot function effectively.")
        else:
            essential_tool_base_names = {
                UMS_FUNC_ADD_ACTION_DEPENDENCY,
                UMS_FUNC_RECORD_ARTIFACT,
                UMS_FUNC_HYBRID_SEARCH,
                UMS_FUNC_STORE_MEMORY,
                UMS_FUNC_UPDATE_MEMORY,
                UMS_FUNC_CREATE_LINK,
                UMS_FUNC_CREATE_THOUGHT_CHAIN,
                UMS_FUNC_GET_THOUGHT_CHAIN,
                UMS_FUNC_RECORD_THOUGHT,
                UMS_FUNC_REFLECTION,
                UMS_FUNC_CONSOLIDATION,
                UMS_FUNC_CREATE_GOAL,
                UMS_FUNC_UPDATE_GOAL_STATUS,
                UMS_FUNC_GET_GOAL_DETAILS,
                UMS_FUNC_GET_GOAL_STACK,
                UMS_FUNC_CREATE_WORKFLOW,
                UMS_FUNC_GET_RICH_CONTEXT_PACKAGE,
                UMS_FUNC_SUMMARIZE_TEXT,
                UMS_FUNC_DIAGNOSE_FILE_ACCESS,
                UMS_FUNC_GET_MULTI_TOOL_GUIDANCE,
            }
            essential_agent_tool_mcp_names = {AGENT_TOOL_UPDATE_PLAN}

            for schema in self.tool_schemas:
                llm_seen_name = _schema_name(schema)
                if not llm_seen_name:
                    if dbg:
                        self.logger.debug(f"AML CONSTRUCT_PROMPT: Skipping tool schema with no discernible name: {str(schema)[:100]}")
                    continue
                original_mcp_name = self.mcp_client.server_manager.sanitized_to_original.get(llm_seen_name, llm_seen_name)
                base_function_name = self._get_base_function_name(original_mcp_name)
                is_essential = original_mcp_name in essential_agent_tool_mcp_names or base_function_name in essential_tool_base_names
                prefix = "**" if is_essential else ""
                desc = schema.get("description", "No description.")
                input_schema_obj = (
                    schema.get("input_schema")
                    if schema.get("input_schema") is not None
                    else (schema.get("function", {}).get("parameters") if schema.get("type") == "function" else None)
                )
                # [MOVE] Schema details above description for salience:
                tool_entry_lines = [
                    f"\n- {prefix}Name LLM Sees: `{llm_seen_name}`{prefix}\n  (Base Function: `{base_function_name}`, Original MCP: `{original_mcp_name}`)",
                    f"  Schema: {_safe_json(input_schema_obj or {'type':'object','properties':{}}, indent=None)}"
                ]
                tool_entry_lines.append(f"  Desc: {desc}")
                if is_essential:
                    tool_entry_lines.append("  (CRITICAL — unexpected failure of this tool usually triggers needs_replan)")  # [ADDED]
                system_blocks.extend(tool_entry_lines)
        system_blocks.append("")

        # ---- 4c. STRATEGIC TOOL SELECTION GUIDANCE ----
        system_blocks.extend(
            [
                "🧠 **STRATEGIC TOOL SELECTION** (Leverage your complete MCP toolkit):",
                "",
                "**STEP 1: ANALYZE YOUR CURRENT SITUATION**",
                "• What type of work am I doing? (research, analysis, creation, problem-solving)",
                "• What information do I already have vs. what do I need?",
                "• What deliverable am I ultimately trying to create?",
                "• What data sources, documents, or systems might I need to access?",
                "• What constraints or requirements must I consider?",
                "",
                "**STEP 2: REVIEW ALL AVAILABLE TOOLS** (Look beyond just UMS tools)",
                "• **Information Gathering**: Web search, document search, database query tools",
                "• **Document Processing**: PDF/DOC/PPT converters, text extractors, format transformers", 
                "• **Data Analysis**: Database tools, spreadsheet processors, statistical analyzers",
                "• **Content Creation**: Document generators, code writers, report builders",
                "• **External Services**: API connectors, specialized service integrations",
                "• **UMS Memory/Organization**: Knowledge storage, linking, reflection tools",
                "• **Workflow Management**: Task delegation, multi-step orchestration tools",
                "",
                "**STEP 3: MATCH EXTERNAL TOOLS TO TASK REQUIREMENTS**",
                "• **Research Tasks**: Consider web search tools AND document search tools AND database query tools",
                "• **Document Analysis**: Look for PDF/DOC converters, text extractors, content analyzers", 
                "• **Data Tasks**: Check for database connectors, spreadsheet tools, statistical processors",
                "• **Integration Needs**: Review API tools, service connectors, external system interfaces",
                "• **Automation Opportunities**: Consider task delegation tools, workflow orchestrators",
                "",
                "**STEP 4: DESIGN STRATEGIC TOOL WORKFLOWS**",
                "• **External → Internal**: Gather from external sources → store in UMS → analyze → create",
                "• **Process → Transform**: Convert documents → extract data → analyze → synthesize",
                "• **Query → Analyze**: Database/API calls → process results → store insights → report",
                "• **Delegate → Review**: Send subtasks to specialized tools → integrate results → finalize",
                "",
                "**STEP 5: OPTIMIZE FOR EFFICIENCY**",
                "• **Parallel Processing**: Use multiple external tools simultaneously when possible",
                "• **Tool Chaining**: Combine complementary tools (search→convert→analyze→store→create)",
                "• **Delegation**: Use task delegation tools for routine/specialized work",
                "• **Caching**: Store processed results to avoid re-processing",
                "",
            ]
        )

        # ---- 4d. ACTION-FIRST Process (replaces planning-heavy instructions) ----
        system_blocks.extend(
            [
                "🎯 **ACTION-FIRST PROCESS** (Complete tasks in 2-3 turns, not 20+):",
                "1. **QUICK SCAN**: Check for errors in `last_error_details`, current goal, available tools",
                "2. **STRATEGIC SELECTION**: Use the tool selection guidance above to pick the right tool(s)",
                "3. **IMMEDIATE ACTION**: Execute your chosen tool(s) with appropriate arguments",
                "",
                "🚀 **TOOL USAGE PRIORITIES** (DELIVERABLE-FIRST APPROACH):",
                f"• **CREATE FIRST**: Use `{UMS_FUNC_RECORD_ARTIFACT}` IMMEDIATELY if you can create any deliverable",
                f"  - Don't wait for 'perfect' information - create with what you have!",
                f"  - Valid artifact_type values: 'text', 'file', 'code', 'data', 'json', 'image', 'table', 'chart', 'url'",
                f"  - For creation tasks: Produce the requested deliverable",
                f"  - For reports: Create comprehensive analysis with available knowledge",
                f"• **LIMITED SEARCH**: Use `{UMS_FUNC_HYBRID_SEARCH}` ONLY if absolutely essential (MAX 2 searches)",
                f"  - Search for SPECIFIC missing pieces, not general research",
                f"  - After ANY search, IMMEDIATELY create the deliverable",
                f"• **NO ENDLESS RESEARCH**: If you've searched twice, STOP and CREATE the deliverable",
                f"• **Store Key Facts**: Use `{UMS_FUNC_STORE_MEMORY}` for critical findings only",
                f"• **Brief Thoughts**: Use `{UMS_FUNC_RECORD_THOUGHT}` for quick insights (not lengthy analysis)",
                f"• **Multi-Tool Execution**: Preferred pattern: search → create (NOT search → search → search)",
                f"• **Multi-Tool Planning**: Use `{UMS_FUNC_GET_MULTI_TOOL_GUIDANCE}` for complex operations only",
                "",
                            "📁 **FILE STORAGE BEST PRACTICES** (CRITICAL for file creation):",
            "• **Safe Locations**: Always use writable directories like:",
            "  - `/home/ubuntu/ultimate_mcp_server/storage/` (preferred)",
            "  - `~/.ultimate_mcp_server/artifacts/`", 
            "  - Project workspace subdirectories",
            "• **Avoid System Directories**: Never write to `/usr/`, `/etc/`, `/var/`, `/root/`",
            "• **File Permissions**: Use `diagnose_file_access_issues` tool if you encounter permission errors",
            "• **Auto-Recovery**: The system will auto-fix many file path issues, but use safe paths from the start",
            "",
            "🔍 **DUPLICATE PREVENTION** (CRITICAL before storing memories or creating artifacts):",
            f"• **Before storing memories**: ALWAYS query existing memories first with `{UMS_FUNC_QUERY_MEMORIES}` or `{UMS_FUNC_HYBRID_SEARCH}` to check if similar information already exists",
            f"• **Before creating artifacts**: Query for similar existing ones with `{UMS_FUNC_QUERY_MEMORIES}`", 
            "• **Avoid Redundancy**: Don't store the same facts multiple times or create nearly-identical artifacts",
            "• **Build on Existing**: If similar memories/artifacts exist, reference them instead of duplicating",
            "• **Clear Purpose**: Each new memory/artifact should have a distinct, well-defined purpose",
            "• **CRITICAL**: If you find existing similar information, use it rather than storing duplicates",
                "",
                "🔧 **MULTI-TOOL OPERATIONS** (Execute multiple tools in single turns):",
                f"• **UMS Support**: The UMS fully supports multiple tool calls per turn - no batching required",
                f"• **Common Patterns**: search→check_existing→store_memory→create_link, query_memories→analyze→record_artifact",
                f"• **Get Guidance**: Use `{UMS_FUNC_GET_MULTI_TOOL_GUIDANCE}` for complex operations or when unsure about tool combinations",
                f"• **No Delays**: Tools are optimized for concurrent use - execute immediately without artificial delays",
                f"• **Tool Synergies**: Many tools work better together (e.g., store_memory + create_memory_link)",
                "",
                "❌ **AVOID ANALYSIS PARALYSIS:**",
                "• Don't spend multiple turns 'developing strategies' or 'clarifying requirements'",
                "• Don't create elaborate multi-step plans that span many turns",
                "• Don't overthink - if you need info, search for it; if you have content, create the artifact",
                "• Don't write thoughts about what you 'should do' - just do it",
                "",
                "📋 **WORKFLOW-SPECIFIC TOOL PATTERNS** (Choose pattern that matches your scenario):",
                "",
                "**RESEARCH & ANALYSIS WORKFLOWS:**",
                f"• Pattern: `{UMS_FUNC_QUERY_MEMORIES}` → `{UMS_FUNC_HYBRID_SEARCH}` → check duplicates → `{UMS_FUNC_STORE_MEMORY}` → `{UMS_FUNC_RECORD_ARTIFACT}`",
                f"• When: Need to gather information and create deliverable",  
                f"• Critical: Check existing knowledge BEFORE external search, avoid storing duplicate facts",
                f"• Limit: Max 2 search calls, then MUST create artifact",
                "",
                "**CONTENT CREATION WORKFLOWS:**",
                f"• Pattern: `{UMS_FUNC_QUERY_MEMORIES}` → `{UMS_FUNC_RECORD_ARTIFACT}` → `{UMS_FUNC_CREATE_LINK}`",
                f"• When: Building on existing knowledge and connecting to related work",
                f"• Focus: Create first, enhance later",
                "",
                "**PROBLEM SOLVING WORKFLOWS:**",
                f"• Pattern: `{UMS_FUNC_RECORD_THOUGHT}` → `{UMS_FUNC_DIAGNOSE_FILE_ACCESS}` → `{UMS_FUNC_RECORD_ARTIFACT}`",
                f"• When: Debugging issues or working through complex problems",
                f"• Strategy: Think, investigate, then implement solution",
                "",
                "**COMPLEX MULTI-STEP WORKFLOWS:**",
                f"• Pattern: `{UMS_FUNC_GET_MULTI_TOOL_GUIDANCE}` → [guided tools] → `{UMS_FUNC_REFLECTION}`",
                f"• When: Unclear how to proceed or multiple approaches possible",
                f"• Benefit: Get strategic guidance before committing to approach",
                "",
                "✅ **DECISION LOGIC** (SYSTEMATIC TOOL SELECTION):",
                f"    *   **NO WORKFLOW**: Call `{UMS_FUNC_CREATE_WORKFLOW}` with the task description",
                f"    *   **NO GOAL SET**: Call `{UMS_FUNC_CREATE_GOAL}` to establish the root goal",
                f"    *   **INFORMATION TASKS**: Use research pattern above - search → store → create",
                f"    *   **CREATION TASKS**: Check existing knowledge first, then create deliverable immediately",
                f"    *   **ANALYSIS TASKS**: Think through problem, gather data if needed, analyze and create findings",
                f"    *   **PROBLEM-SOLVING**: Diagnose issue, apply fix, document solution",
                f"    *   **COMPLEX/UNCLEAR**: Start with `{UMS_FUNC_GET_MULTI_TOOL_GUIDANCE}` for strategy",
                f"    *   **ALREADY SEARCHED 2x**: STOP researching - create deliverable with current info",
                f"    *   **GOAL COMPLETE**: Call `{UMS_FUNC_UPDATE_GOAL_STATUS}` then signal completion",
                f"    *   **ERROR/REPLAN**: Call `{llm_seen_agent_update_plan_name_for_instr}` with corrected plan",
                "    *   **IN DOUBT**: Choose the tool that produces tangible progress toward deliverable",
                "",
                "🏁 **OUTPUT**: Respond with ONLY a valid JSON tool call or 'Goal Achieved...' text",
            ]
        )

        system_blocks.extend(
            [
                "\nKey Considerations:",
                "*   Goal Focus: Always work towards the Current Operational UMS Goal. Use UMS goal tools by their base function names.",
                "*   Immediate Action: Prioritize tool usage over extensive analysis - search when you need info, create when you have content.",
                "*   Token Budget: Keep outputs under 3500 tokens; if content risk exceeds, summarise memories with summarize_text first.*",
                "*   Efficiency Target: Complete simple tasks in 2-3 turns by using tools immediately rather than planning extensively.",
                "*   Dependencies & Cycles: Ensure `depends_on` actions (in plan or UMS) are complete. Avoid cycles.",
                "*   UMS Context: Leverage the `ums_context_package` (core, working, proactive, procedural memories, links).",
                "*   Errors: Prioritize error analysis based on `last_error_details.type` and `last_action_summary`.",
                "*   User Guidance: Pay close attention to thoughts of type 'user_guidance' or memories of type 'user_input'. These are direct inputs from the operator and will likely require plan adjustments.",
                f"*   Final Output: If your task involves creating a deliverable (report, file, etc.), ensure it's saved as a UMS artifact using the UMS tool with base function '{UMS_FUNC_RECORD_ARTIFACT}' (passing `is_output=True`) *before* signaling overall workflow completion with its ID.",
                "*   Idempotency Keys: For UMS creation tools (like create_workflow, record_action_start, store_memory, create_goal, record_artifact, record_thought, create_thought_chain), you may see an optional `idempotency_key` parameter in their schemas. You (the LLM) generally do NOT need to provide this key. The agent system will manage idempotency for retries internally. Only provide it if explicitly instructed to reuse a specific key from a previous failed attempt.",
            ]
        )
        system_blocks.extend(
            [  # Recovery Strategies – text preserved verbatim + one new error class
                "\nRecovery Strategies based on `last_error_details.type`:",
                f"*   `InvalidInputError`: Review tool schema, args, context. Correct args and retry OR choose different tool/step.",
                f"*   `DependencyNotMetError`: Choose a different action that doesn't require dependencies, or use simpler approach.",
                f"*   `ServerUnavailable` / `NetworkError`: Tool's server might be down. Try different tool, wait, or adjust plan.",
                f"*   `UMSMalformedPayload`: Tool returned unexpected schema; record the raw payload to memory, wait one turn, then attempt again or escalate.*",
                f"*   `APILimitError` / `RateLimitError`: External API busy. Plan to wait (record thought) before retry.",
                f"*   `ToolExecutionError` / `ToolInternalError` / `UMSError`: Tool failed. Analyze message. Try different args, alternative tool, or adjust plan.",
                f"*   `PlanUpdateError`: Proposed plan structure was invalid when agent tried to apply it. Re-examine plan and dependencies, try `{llm_seen_agent_update_plan_name_for_instr}` again with a corrected *complete* plan.",
                f"*   `PlanValidationError`: Create a simpler 2-step plan: (1) gather info, (2) create deliverable. For complex operations, use `{UMS_FUNC_GET_MULTI_TOOL_GUIDANCE}` for planning assistance.",
                f"*   `CancelledError`: Previous action cancelled. Re-evaluate current step.",
                f"*   `GoalManagementError` / `GoalSyncError`: Error managing UMS goals or mismatch between agent and UMS state. Review `agent_assembled_goal_context` and `last_error_details.recommendation`. Use UMS goal tools to correct or re-establish goals. May need to call UMS tool with base func `{UMS_FUNC_GET_GOAL_DETAILS}`.",
                f"*   `CognitiveStateError`: Error saving or loading agent's cognitive state. This is serious. Attempt to record key information as memories and then try to re-establish state or simplify the current task.",
                f"*   `InternalStateSetupError`: Critical internal error during agent/workflow setup. Analyze error. May require `{llm_seen_agent_update_plan_name_for_instr}` to fix plan or re-initiate a step.",
                f"*   `FilePermissionError` / `FileAccessError`: Use safe file paths (e.g., `/home/ubuntu/ultimate_mcp_server/storage/`). Call `{UMS_FUNC_DIAGNOSE_FILE_ACCESS}` tool for path diagnosis and alternatives. Avoid system directories.",
                f"*   `InfiniteLoopDetected` / `InfiniteLoopDetectedOnFailure`: Agent is stuck repeating the same tool without progress. CRITICAL: You MUST break the pattern by: (1) Creating a deliverable immediately with `{UMS_FUNC_RECORD_ARTIFACT}`, (2) Using a completely different tool/approach, or (3) Taking concrete action with available information. DO NOT repeat the same action that caused the loop.",
                f"*   `DuplicateMemoryPrevention`: Tried to store information that already exists. CRITICAL: STOP storing duplicate facts and START creating deliverables. Use `{UMS_FUNC_RECORD_ARTIFACT}` to create the required output with information you already have. DO NOT store more memories - proceed to goal completion.",
                f"*   `UnknownError` / `UnexpectedExecutionError` / `AgentError` / `MCPClientError` / `LLMError` / `LLMOutputError`: Analyze error message carefully. Simplify step, use different approach, or record_thought if stuck. If related to agent state, try to save essential info and restart a simpler sub-task.",
            ]
        )
        system_prompt_str = "\n".join(system_blocks)

        # ---------- 5. Assemble USER prompt blocks (unchanged wording) -------
        context_json_str = _truncate_context(context)

        user_prompt_blocks: list[str] = [
            "Current Context:",
            "```json",
            context_json_str,
            "```",
            "",
            "Current Plan:",
            "```json",
            _safe_json([step.model_dump(exclude_none=True) for step in self.state.current_plan], indent=2),
            "```",
            "",
            f"Last Action Summary:\n{self.state.last_action_summary}\n",
        ]

        # Add deferred tool call information if present
        if self.state.deferred_tool_calls:
            user_prompt_blocks += [
                "**DEFERRED TOOL CALLS FROM PREVIOUS TURN:**",
                f"The following {len(self.state.deferred_tool_calls)} tool call(s) were deferred from the previous turn and should be considered for execution:",
                "```json",
                _safe_json(self.state.deferred_tool_calls, indent=2),
                "```",
                "",
            ]

        # Add atomic decision context if available
        if self.state.last_atomic_decision_info:
            info = self.state.last_atomic_decision_info
            user_prompt_blocks += [
                "**PREVIOUS TURN TOOL PROCESSING INFO:**",
                f"Last turn processed {info.get('tools_processed', 0)} of {info.get('total_tools_requested', 0)} requested tools. " +
                f"{info.get('tools_deferred', 0)} tools were deferred to this turn. " +
                (f"Agent Message: {info.get('agent_message')}" if info.get('agent_message') else ""),
                "",
            ]

        if self.state.last_error_details:
            # Add special context for infinite loop detection
            if self.state.last_error_details.get("type") in ["InfiniteLoopDetected", "InfiniteLoopDetectedOnFailure"]:
                loop_info = self.state.last_error_details.get("loop_info", {})
                user_prompt_blocks += [
                    "🔄 **INFINITE LOOP DETECTED - CRITICAL INTERVENTION REQUIRED**:",
                    f"The agent has been executing the same tool '{self.state.last_error_details.get('tool')}' repeatedly without making progress.",
                    "",
                    "**Loop Detection Metrics:**",
                    f"• Consecutive same tool executions: {loop_info.get('consecutive_same_tool', 0)}",
                    f"• Search attempts without artifacts: {loop_info.get('search_attempts', 0)}",
                    f"• Current progression stage: {loop_info.get('progression_stage', 'unknown')}",
                    f"• Turns since last artifact creation: {loop_info.get('turns_since_artifact', 0)}",
                    "",
                    "**REQUIRED ACTION - Choose ONE of these approaches:**",
                    f"1. **FORCE PROGRESSION**: Create a deliverable NOW using `{UMS_FUNC_RECORD_ARTIFACT}` with whatever information you have",
                    f"2. **CHANGE APPROACH**: Use a completely different tool or method than '{self.state.last_error_details.get('tool')}'",
                    f"3. **SEARCH FOR INFO**: Use `{UMS_FUNC_HYBRID_SEARCH}` to gather specific information needed for the goal",
                    "",
                    "**DO NOT repeat the same search/analysis patterns. Take CONCRETE ACTION to create progress.**",
                    "",
                ]
            
            user_prompt_blocks += [
                "**CRITICAL: Address Last Error Details (refer to Recovery Strategies in System Prompt)**:",
                "```json",
                _safe_json(self.state.last_error_details, indent=2, default=str),
                "```",
                "",
            ]

        if self.state.last_meta_feedback:
            user_prompt_blocks += [
                "**Meta-Cognitive Feedback (e.g., a suggested plan or insight from reflection/consolidation):**",
                self.state.last_meta_feedback,
                "",
            ]

        # ---------- 6. Goal reminder and final instruction -------------------
        current_goal_desc_for_reminder = "Overall UMS Workflow Goal or Initial Task"
        if self.state.workflow_id:
            current_op_goal_details_reminder = context.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
            if (
                isinstance(current_op_goal_details_reminder, dict)
                and current_op_goal_details_reminder.get("description")
            ):
                current_goal_desc_for_reminder = current_op_goal_details_reminder["description"]
            elif context.get("ums_context_package", {}).get("core_context", {}).get("workflow_goal"):
                current_goal_desc_for_reminder = context["ums_context_package"]["core_context"]["workflow_goal"]
        else:
            current_goal_desc_for_reminder = current_task_goal_desc

        user_prompt_blocks.append(f"Current Goal Reminder: {current_goal_desc_for_reminder}")
        user_prompt_blocks.append("")

        # Add goal completion status and guidance
        if current_goal_desc_for_reminder and current_goal_desc_for_reminder != "Overall UMS Workflow Goal or Initial Task":
            goal_type = self._classify_goal_type(current_goal_desc_for_reminder)
            completion_status = await self._analyze_goal_completion_semantically(current_goal_desc_for_reminder, goal_type)
            goal_type = completion_status["goal_type"]
            criteria = completion_status["completion_criteria"]
            
            user_prompt_blocks.append("🎯 **GOAL COMPLETION STATUS & GUIDANCE**:")
            user_prompt_blocks.append(f"• **Goal Type**: {goal_type.replace('_', ' ').title()}")
            user_prompt_blocks.append(f"• **Expected Deliverable**: {criteria['expected_deliverable']}")
            user_prompt_blocks.append(f"• **Deliverable Format**: {criteria['deliverable_format']}")
            user_prompt_blocks.append(f"• **Current Status**: {completion_status['completion_message']}")
            
            if completion_status["ready_for_completion"]:
                user_prompt_blocks.append("🎉 **GOAL READY FOR COMPLETION**: All criteria met. Signal completion with 'Goal Achieved...' message.")
            elif completion_status["force_completion"]:
                user_prompt_blocks.append("⚠️ **FORCE COMPLETION**: Exceeded limits. Create deliverable NOW and signal completion.")
            else:
                missing_items = []
                if not completion_status["has_deliverable"]:
                    missing_items.append("create deliverable artifact")
                if not completion_status["has_sufficient_facts"]:
                    missing_items.append(f"gather {criteria['min_facts_required'] - completion_status['fact_count']} more key facts")
                user_prompt_blocks.append(f"🔄 **NEXT STEPS**: {', '.join(missing_items)}")
            
            user_prompt_blocks.append(f"• **Completion Criteria**: {'; '.join(criteria['completion_criteria'])}")
            user_prompt_blocks.append("")

        # Add CRITICAL infinite loop prevention warnings based on current state
        loop_warnings = self._generate_loop_prevention_warnings(current_goal_desc_for_reminder)
        if loop_warnings:
            user_prompt_blocks.append(loop_warnings)

        # Add proactive tool suggestions (enhanced for research workflows)
        tool_suggestions = await self._suggest_next_tools(
            current_goal_desc_for_reminder, 
            self.state.last_action_summary,
            len(self.state.current_plan)
        )
        if tool_suggestions:
            user_prompt_blocks.append(tool_suggestions)
        
        # Add workflow-specific multi-tool suggestions
        efficient_suggestions = self._suggest_efficient_tool_patterns(current_goal_desc_for_reminder, self.state.last_action_summary)
        if efficient_suggestions:
            user_prompt_blocks.append(efficient_suggestions)

        # ---------- 7. Final instruction branch logic (plus hard constraint) --------
        instruction_append = " Remember: output MUST be either a single JSON object for a tool call or the 'Goal Achieved…' sentence - no markdown, no additional keys."  # [ADDED]

        if not self.state.workflow_id:
            final_instruction_text = (
                f"Instruction: NO ACTIVE UMS WORKFLOW. Your first action MUST be to call the UMS tool whose base function is '{UMS_FUNC_CREATE_WORKFLOW}'. "
                f"Use the 'Initial Overall Task Description' from the system prompt as the 'goal' for this tool. Provide a suitable 'title'."
                + instruction_append
            )
        elif self.state.workflow_id and not self.state.current_goal_id:
            final_instruction_text = (
                f"Instruction: UMS WORKFLOW ACTIVE, BUT NO UMS OPERATIONAL GOAL SET. Your first action MUST be to call the UMS tool with base function "
                f"'{UMS_FUNC_CREATE_GOAL}' to establish the root UMS goal for the current workflow. Use the 'Overall UMS Workflow Goal' as its description."
                + instruction_append
            )
        elif self.state.needs_replan and self.state.last_meta_feedback:
            # Create example plan format to show the LLM exactly how to structure its response
            example_plan_steps = [
                {"id": f"step-{MemoryUtils.generate_id()[:8]}", "description": "Use search tools to gather needed information", "status": "planned", "depends_on": []},
                {"id": f"step-{MemoryUtils.generate_id()[:8]}", "description": "Create the required deliverable using record_artifact", "status": "planned", "depends_on": []}
            ]
            final_instruction_text = (
                f"Instruction: **REPLANNING REQUIRED.** Meta-cognitive feedback (see 'Meta-Cognitive Feedback' in context) is available. "
                f"Your primary action MUST be to use the tool named `{llm_seen_agent_update_plan_name_for_instr}` to set a new, detailed plan. "
                f"Carefully consider the meta-feedback when formulating the new plan. After updating the plan, the agent will proceed with the first step of the new plan in the *next* turn.\n\n"
                f"VERY IMPORTANT - FORMAT YOUR RESPONSE EXACTLY LIKE THIS (with your plan steps):\n"
                f"{{\"name\": \"{llm_seen_agent_update_plan_name_for_instr}\", \"arguments\": {{\"plan\": {json.dumps(example_plan_steps, indent=2)}}}}}\n\n"
                f"Or with OpenAI function calling format:\n"
                f"```json\n"
                f'{{"tool": "{llm_seen_agent_update_plan_name_for_instr}", "tool_input": {{"plan": {json.dumps(example_plan_steps, indent=2)}}}}}\n'
                f"```"
                + instruction_append
            )
        elif self.state.needs_replan: # This block is when needs_replan is True
            example_plan_steps = [
                {"id": f"step-{MemoryUtils.generate_id()[:8]}", "description": "Research the impact of exercise on mental health using scientific studies", "status": "planned", "depends_on": []},
                {"id": f"step-{MemoryUtils.generate_id()[:8]}", "description": "Analyze and summarize key findings about exercise benefits", "status": "planned", "depends_on": []}
            ]
            
            # Determine if the current model supports the stricter json_schema format
            current_model_for_api_call = self.agent_llm_model
            provider_for_current_model = self.mcp_client.get_provider_from_model(current_model_for_api_call)
            
            model_supports_json_schema_type = False
            if provider_for_current_model == Provider.OPENAI.value:
                if any(prefix in current_model_for_api_call.lower() for prefix in MODELS_CONFIRMED_FOR_OPENAI_JSON_SCHEMA_FORMAT):
                    model_supports_json_schema_type = True
            elif provider_for_current_model == Provider.MISTRAL.value:
                 if current_model_for_api_call.lower() in MISTRAL_NATIVE_MODELS_SUPPORTING_SCHEMA: # Assuming you have this list
                    model_supports_json_schema_type = True
            # Add other providers if they support json_schema type

            if model_supports_json_schema_type:
                # Instruction for models that *do* support json_schema (OpenAI tool call format is fine)
                # The existing prompt asking for the full tool call using llm_seen_agent_update_plan_name_for_instr is okay here,
                # as json_schema mode with tool_choice should handle it.
                # Ensure `force_tool_choice` is correctly set to llm_seen_agent_update_plan_name_for_instr in MCPClient.
                final_instruction_text = (
                    f"Instruction: **REPLANNING REQUIRED.** An error occurred (`last_error_details`) or a significant state change necessitates a new plan. "
                    f"Analyze `last_error_details` and other context. Your primary action MUST be to use the tool named "
                    f"`{llm_seen_agent_update_plan_name_for_instr}` to propose a new, complete, and valid plan to address the situation and achieve the Current Operational UMS Goal. "
                    f"The response MUST be a call to the '{llm_seen_agent_update_plan_name_for_instr}' tool, formatted as a JSON object.\n\n"
                    f"VERY IMPORTANT - FORMAT YOUR RESPONSE EXACTLY LIKE THIS (with your plan steps):\n"
                    f"{{\"name\": \"{llm_seen_agent_update_plan_name_for_instr}\", \"arguments\": {{\"plan\": {json.dumps(example_plan_steps, indent=2)}}}}}\n\n"
                    f"Or with OpenAI function calling format (if applicable for the model family despite using generic client):\n"
                    f"```json\n"
                    f'{{"tool": "{llm_seen_agent_update_plan_name_for_instr}", "tool_input": {{"plan": {json.dumps(example_plan_steps, indent=2)}}}}}\n'
                    f"```"
                    + instruction_append
                )
            else:
                # Instruction for models that ONLY support json_object (like gpt-4.1)
                # Ask for ONLY THE ARGUMENTS JSON.
                direct_json_output_example = {"plan": example_plan_steps}
                final_instruction_text = (
                    f"Instruction: **REPLANNING REQUIRED.** An error occurred (`last_error_details`) or a significant state change necessitates a new plan. "
                    f"Analyze `last_error_details` and other context. "
                    f"Your response MUST BE A SINGLE VALID JSON OBJECT containing ONLY the new plan arguments. "
                    f"This JSON object must have a single top-level key named \"plan\", and its value must be an array of plan step objects. "
                    f"DO NOT wrap this JSON in a tool call structure (like 'name' or 'tool' keys at the top level). Output ONLY the JSON for the plan arguments.\n\n"
                    f"VERY IMPORTANT - YOUR ENTIRE RESPONSE MUST BE *EXACTLY* THE JSON OBJECT SHOWN IN THE EXAMPLE BELOW (with your plan steps):\n"
                    f"```json\n{json.dumps(direct_json_output_example, indent=2)}\n```\n"
                    f"No other text, no markdown, just this JSON object structure as your direct output."
                    # instruction_append is removed here because we are not asking for a tool call JSON.
                )
            
                            # Prepend meta-feedback if available
                if self.state.last_meta_feedback:
                     final_instruction_text = (
                        f"Instruction: **REPLANNING REQUIRED.** Meta-cognitive feedback (see 'Meta-Cognitive Feedback' in context) is available. "
                        + final_instruction_text.split("Instruction: **REPLANNING REQUIRED.**", 1)[-1]
                    )
        else:
            # Normal operational case: workflow and goal exist, no replanning needed
            # Add goal-type-specific completion guidance
            completion_guidance = ""
            if current_goal_desc_for_reminder and current_goal_desc_for_reminder != "Overall UMS Workflow Goal or Initial Task":
                goal_type = self._classify_goal_type(current_goal_desc_for_reminder)
                completion_status = await self._analyze_goal_completion_semantically(current_goal_desc_for_reminder, goal_type)
                
                if completion_status["ready_for_completion"]:
                    completion_guidance = (
                        f"🎉 GOAL COMPLETION READY: All criteria met for this {completion_status['goal_type'].replace('_', ' ')} goal. "
                        f"Signal completion by responding with: 'Goal Achieved: [brief summary of deliverable created]' "
                    )
                elif completion_status["force_completion"]:
                    completion_guidance = (
                        f"⚠️ FORCE COMPLETION: You've exceeded search limits or time constraints. "
                        f"Create the required {completion_status['completion_criteria']['expected_deliverable']} NOW using available information, "
                        f"then signal completion. "
                    )
                elif not completion_status["has_deliverable"]:
                    completion_guidance = (
                        f"🎯 DELIVERABLE REQUIRED: Create the expected {completion_status['completion_criteria']['expected_deliverable']} "
                        f"as an artifact with is_output=True. "
                    )
            
            final_instruction_text = (
                f"Instruction: You are actively working on the current UMS goal. "
                f"{completion_guidance}"
                f"Review your current plan and progress, then execute the next logical action. "
                f"IMPORTANT: Do NOT create sub-goals - work directly on the current goal using available tools. "
                f"Take action to search for information, create deliverables, or signal completion when done."
                + instruction_append
            )
        
        self.logger.info(f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): Final instruction: {final_instruction_text}")

        user_prompt_blocks.append(final_instruction_text)
        user_prompt_str = "\n".join(user_prompt_blocks)

        # ---------- 8. Combine SYSTEM + USER into final prompt ---------------
        constructed_prompt_messages = [
            {
                "role": "user",
                "content": system_prompt_str + "\n---\n" + user_prompt_str,
            }
        ]

        if dbg:
            self.logger.debug(
                f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages length: {len(constructed_prompt_messages)}"
            )
            if constructed_prompt_messages:
                self.logger.debug(
                    f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): prompt_messages[0] keys: {list(constructed_prompt_messages[0].keys())}"
                )
                self.logger.debug(
                    f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): prompt_messages[0]['content'] snippet: "
                    f"{constructed_prompt_messages[0]['content'][:500]}..."
                )
        return constructed_prompt_messages



    async def _background_task_done_safe(self, task: asyncio.Task) -> None:
        """
        Clean-up callback for a background `asyncio.Task`.

        Responsibilities
        ----------------
        1.  Remove the task from `self.state.background_tasks` (under `_bg_tasks_lock`).
        2.  Release exactly **one** permit on `_bg_task_semaphore` *if* the task was
            actually tracked.
        3.  Surface cancellation or exceptions through structured logging.
        4.  **Never** propagate an exception itself (ensured by the outer `try/finally`).

        The method is intentionally chatty at DEBUG level so long-running services
        can spot leaks or semaphore imbalances quickly.
        """
        try:
            # ------------------------------------------------------------------- #
            # 1. Un-register the task from our tracking set
            # ------------------------------------------------------------------- #
            async with self._bg_tasks_lock:
                was_present = task in self.state.background_tasks
                if was_present:
                    self.state.background_tasks.discard(task)

            # ------------------------------------------------------------------- #
            # 2. Release semaphore slot if we had registered this task
            # ------------------------------------------------------------------- #
            if was_present:
                try:
                    self._bg_task_semaphore.release()
                    self.logger.debug(
                        "Released background-task semaphore (value=%s) for %s",
                        getattr(self._bg_task_semaphore, "_value", "n/a"),
                        task.get_name(),
                    )
                except ValueError:
                    # More releases than acquires → log, but keep going.
                    self.logger.warning(
                        "Semaphore release imbalance for background task %s", task.get_name()
                    )
                except Exception as sem_err:  # pragma: no cover
                    self.logger.exception(
                        "Unexpected error releasing semaphore for %s: %s",
                        task.get_name(),
                        sem_err,
                    )

            # ------------------------------------------------------------------- #
            # 3. Inspect task outcome
            # ------------------------------------------------------------------- #
            if task.cancelled():
                self.logger.debug("Background task %s was cancelled.", task.get_name())
                return

            exc: Optional[BaseException]
            try:
                exc = task.exception()  # Safe now: task not cancelled.
            except asyncio.CancelledError:
                # Rare race: task.cancelled() False yet exception() raises.
                self.logger.debug("Background task %s raised CancelledError.", task.get_name())
                return
            except Exception:  # pragma: no cover
                # Shouldn't happen, but keep callback from crashing.
                self.logger.exception("Could not retrieve exception from task %s", task.get_name())
                return

            if exc:
                # Log full traceback once; service operators need stack-trace.
                self.logger.error(
                    "Background task %s failed – %s: %s",
                    task.get_name(),
                    type(exc).__name__,
                    exc,
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
        finally:
            # Ensure reference is dropped so GC can collect completed task objects.
            task = None


    def _start_background_task(self, coro_fn, *args, **kwargs) -> Optional[asyncio.Task]:
        """
        Launch `coro_fn` as a throttled background asyncio task.
        Simplified version using SimpleBackgroundTaskManager.
        """
        # Add workflow/context info to kwargs if not provided
        if "workflow_id" not in kwargs and self.state.workflow_id:
            kwargs["workflow_id"] = self.state.workflow_id
        if "context_id" not in kwargs and self.state.context_id:
            kwargs["context_id"] = self.state.context_id
        
        # Delegate to the simplified manager
        task = self.simple_bg_manager.start_task(coro_fn, *args, **kwargs)
        
        if task:
            # Still track for legacy compatibility
            asyncio.create_task(self._add_bg_task(task))
        
        return task

    async def _add_bg_task(self, task: asyncio.Task) -> None:
        async with self._bg_tasks_lock:
            self.state.background_tasks.add(task)


    async def _cleanup_background_tasks(self) -> None:
        """
        Gracefully cancel and flush all background tasks.
        Now uses simplified background task manager.
        """
        # Use the simplified manager for cleanup
        await self.simple_bg_manager.cleanup_all()
        
        # Legacy cleanup for any remaining tracked tasks
        async with self._bg_tasks_lock:
            if self.state.background_tasks:
                self.logger.info("Cleaning up %d legacy background tasks", len(self.state.background_tasks))
                for task in self.state.background_tasks:
                    if not task.done():
                        task.cancel()
                
                # Wait for cancellation
                if self.state.background_tasks:
                    await asyncio.gather(*self.state.background_tasks, return_exceptions=True)
                
                self.state.background_tasks.clear()
                self.logger.info("Legacy background tasks cleanup completed")


    async def _estimate_tokens_anthropic(self, data: Any) -> int:
        if data is None:
            return 0
        try:
            if not self.anthropic_client:
                raise RuntimeError("Anthropic client unavailable")
            text_to_count = data if isinstance(data, str) else json.dumps(data, default=str, ensure_ascii=False)
            # Anthropic's count_tokens is synchronous in some SDK versions, ensure it's awaited if async
            if asyncio.iscoroutinefunction(self.anthropic_client.count_tokens):
                token_count = await self.anthropic_client.count_tokens(text_to_count)
            else:  # Synchronous version
                token_count = self.anthropic_client.count_tokens(text_to_count)
            return int(token_count)
        except Exception as e:
            self.logger.warning(f"Token estimation via Anthropic API failed: {e}. Using fallback.")
            text_representation = data if isinstance(data, str) else json.dumps(data, default=str, ensure_ascii=False)
            return len(text_representation) // 4  # Rough fallback


    async def _with_retries(                # noqa: C901 – complex but self-contained
        self,
        coro_fun: Callable[..., Awaitable[Any]],
        *args: Any,
        max_retries: int = 3,
        retry_exceptions: Tuple[type[BaseException], ...] = (
            ToolError,
            ToolInputError,
            asyncio.TimeoutError,
            ConnectionError,
            APIConnectionError,
            RateLimitError,
            APIStatusError,
            httpx.RequestError,
        ),
        retry_backoff: float = 2.0,          # exponential base
        jitter: Tuple[float, float] = (0.1, 0.5),
        max_delay: Optional[float] = None,   # cap for any single sleep (None = no cap)
        on_retry: Optional[
            Callable[[int, BaseException, float], None]
        ] = None,                            # optional sync callback (attempt, exc, delay)
        **kwargs: Any,
    ) -> Any:
        """
        Execute *coro_fun* with automatic retries for transient errors.

        Parameters
        ----------
        coro_fun
            Awaitable callable to execute.
        max_retries
            Maximum **total** attempts (first call + retries).
        retry_exceptions
            Tuple of exception classes that should trigger a retry.
        retry_backoff
            Exponential back-off base (e.g. ``2.0`` → 1 s, 2 s, 4 s … before jitter).
        jitter
            Uniform random jitter *added* to each back-off delay.
        max_delay
            Optional upper bound for any single delay.
        on_retry
            Optional callback executed synchronously **after** a failure is caught and
            **before** sleeping. It receives ``attempt`` (1-based retry count),
            the caught ``exception`` and the calculated ``delay`` seconds.

        Raises
        ------
        Same exceptions raised by *coro_fun* after exhausting retries, or when
        ``self._shutdown_event`` is set during the retry cycle.
        """
        attempt: int = 0
        last_exception: Optional[BaseException] = None

        while attempt < max_retries:
            if self._shutdown_event.is_set():
                self.logger.warning(
                    f"Shutdown requested while executing {coro_fun.__name__}; aborting retries."
                )
                raise asyncio.CancelledError(
                    f"Shutdown during execution of {coro_fun.__name__}"
                ) from last_exception

            try:
                return await coro_fun(*args, **kwargs)

            except retry_exceptions as exc:
                attempt += 1
                last_exception = exc

                # Final attempt failed → propagate
                if attempt >= max_retries:
                    self.logger.error(
                        "%s failed after %d attempts: %s – %s",
                        coro_fun.__name__,
                        max_retries,
                        type(exc).__name__,
                        exc,
                        exc_info=True,
                    )
                    raise

                # Compute next delay
                base_delay: float = retry_backoff ** (attempt - 1)
                random_jitter: float = random.uniform(*jitter)
                delay: float = base_delay + random_jitter
                if max_delay is not None:
                    delay = min(delay, max_delay)

                self.logger.warning(
                    "%s failed (%s: %s); retry %d/%d in %.2f s",
                    coro_fun.__name__,
                    type(exc).__name__,
                    str(exc)[:100],
                    attempt,
                    max_retries,
                    delay,
                )

                # Invoke optional hook - guard against hook misbehaviour
                if on_retry is not None:
                    try:
                        on_retry(attempt, exc, delay)
                    except Exception:  # pragma: no cover
                        self.logger.debug(
                            "on_retry callback raised an exception; continuing retries",
                            exc_info=True,
                        )

                try:
                    await asyncio.sleep(delay)
                except asyncio.CancelledError:
                    self.logger.info(
                        "%s cancelled while waiting %.2f s before retry.", coro_fun.__name__, delay
                    )
                    raise

            except asyncio.CancelledError:
                # Propagate cancellation while still logging for observability
                self.logger.info("%s execution cancelled.", coro_fun.__name__)
                raise


    async def _save_agent_state(self) -> None:  # noqa: C901  (complexity is OK for a single, well-scoped helper)
        """
        Persist the *entire* ``AgentState`` object to ``self.agent_state_file`` **atomically**.
        """
        # -----------------------------------------------------------------
        # 1. Ensure we have a per-instance lock
        # -----------------------------------------------------------------
        if not hasattr(self, "_state_save_lock"):
            # noqa: Attribute is created dynamically – that is intentional
            self._state_save_lock: asyncio.Lock = asyncio.Lock()  # type: ignore[attr-defined]

        async with self._state_save_lock:
            self.logger.debug(
                "Attempting to save agent state. WF ID: %s",
                _fmt_id(self.state.workflow_id),
            )

            # --------------------------------------------------------------
            # 2. Build a *pure* dict suitable for JSON serialization
            # --------------------------------------------------------------
            def _json_default(obj: Any) -> str:
                """Fallback serializer for `json.dumps`."""
                if isinstance(obj, (datetime, uuid.UUID)):
                    return obj.isoformat() if isinstance(obj, datetime) else str(obj)
                if isinstance(obj, Enum):
                    return obj.value  # type: ignore[return-value]
                if dataclasses.is_dataclass(obj):
                    return dataclasses.asdict(obj)
                if isinstance(obj, BaseModel):
                    return obj.model_dump(mode="json")
                return str(obj)  # best-effort

            state_dict_to_save: dict[str, Any] = {}

            for fld in dataclasses.fields(AgentState):
                # Explicitly skip non-serialisable / runtime fields
                if fld.name == "background_tasks":
                    continue

                value = getattr(self.state, fld.name)

                if fld.name == "current_plan":
                    state_dict_to_save["current_plan"] = [
                        step.model_dump(exclude_none=True) for step in value
                    ] if value else []
                elif fld.name == "tool_usage_stats":
                    # Convert defaultdicts -> normal dicts for JSON
                    state_dict_to_save["tool_usage_stats"] = {
                        k: dict(v) for k, v in value.items()
                    } if value else {}
                elif fld.name == "goal_stack":
                    state_dict_to_save["goal_stack"] = [
                        g.model_dump()
                        if isinstance(g, BaseModel) else dict(g)
                        if isinstance(g, dict) else g
                        for g in (value or [])
                    ]
                else:
                    state_dict_to_save[fld.name] = value

            # Add a write-timestamp (ISO-8601, UTC)
            state_dict_to_save["timestamp"] = datetime.now(timezone.utc).isoformat()

            # --------------------------------------------------------------
            # 3. Serialize → JSON (pretty-printed, UTF-8, stable order)
            # --------------------------------------------------------------
            try:
                json_string: str = json.dumps(
                    state_dict_to_save,
                    indent=2,
                    ensure_ascii=False,
                    default=_json_default,
                    sort_keys=True,
                )
            except TypeError:
                # Pin-point which field is not serialisable
                for k, v in state_dict_to_save.items():
                    try:
                        json.dumps(v, default=_json_default)
                    except TypeError as bad:
                        self.logger.error(
                            "JSON serialisation failed for field '%s': %s", k, bad, exc_info=True
                        )
                        break
                raise  # re-raise so the caller notices

            # --------------------------------------------------------------
            # 4. Write atomically in a background thread
            # --------------------------------------------------------------
            def _write_file() -> None:
                tmp_path: Path | None = None
                try:
                    tmp_dir = self.agent_state_file.parent
                    tmp_dir.mkdir(parents=True, exist_ok=True)

                    with tempfile.NamedTemporaryFile(
                        mode="w",
                        encoding="utf-8",
                        dir=tmp_dir,
                        prefix=f"{self.agent_state_file.stem}._tmp_",
                        suffix=f".{os.getpid()}_{uuid.uuid4().hex[:4]}",
                        delete=False,
                    ) as tmp_file:
                        tmp_path = Path(tmp_file.name)
                        tmp_file.write(json_string)
                        tmp_file.flush()
                        os.fsync(tmp_file.fileno())

                    # Atomic replace
                    os.replace(tmp_path, self.agent_state_file)
                    self.logger.debug("State saved atomically → %s", self.agent_state_file)

                except Exception:
                    # Clean-up orphaned tmp file on *any* failure
                    if tmp_path and tmp_path.exists():
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except OSError:
                            pass
                    raise

            # Fix: This should be awaited properly to prevent "coroutine was never awaited" warnings
            try:
                await asyncio.to_thread(_write_file)
            except Exception as write_err:
                self.logger.error(f"_save_agent_state(): Error during asyncio.to_thread(_write_file): {write_err}", exc_info=True)


    async def _load_agent_state(self) -> None:  # noqa: C901  (complex by design)
        """
        Robustly load `AgentState` from `self.agent_state_file`.
        """
        # ── 1. Fast-path if no prior state ───────────────────────────────────────
        if not self.agent_state_file.exists():
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD,
            )
            self.logger.info("State-file missing → starting with a fresh AgentState.")
            return

        # ── 2. Read & parse JSON (fail-hard handled below) ───────────────────────
        try:
            async with aiofiles.open(self.agent_state_file, "r", encoding="utf-8") as f:
                raw_text: str = await f.read()
            data: Dict[str, Any] = json.loads(raw_text)
            if not isinstance(data, dict):
                raise TypeError(f"Top-level JSON is {type(data)}, expected dict.")
        except Exception as exc:  # JSONDecodeError, OSError, etc.
            self.logger.error(
                f"State-file load failed ({exc!s}). Reverting to defaults.",
                exc_info=True,
            )
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD,
            )
            return

        # ── 3. Build kwargs for dataclass instantiation ─────────────────────────
        kwargs: Dict[str, Any] = {}
        processed_keys: set[str] = set()

        for fld in dataclasses.fields(AgentState):
            # Skip non-init (e.g., background_tasks)
            if not fld.init:
                continue

            name = fld.name
            processed_keys.add(name)

            # ---------------------------------------------------------------------
            # A. Key present in JSON → attempt to coerce/validate
            # ---------------------------------------------------------------------
            if name in data:
                value = data[name]

                # 1️⃣  current_plan  -- ensure list[PlanStep]
                if name == "current_plan":
                    if isinstance(value, list) and value:
                        try:
                            kwargs["current_plan"] = [PlanStep(**item) if not isinstance(item, PlanStep) else item for item in value]
                        except (ValidationError, TypeError) as exc:
                            self.logger.warning(
                                f"State-file plan reload failed: {exc!s}. Falling back to default plan."
                            )
                            kwargs["current_plan"] = [PlanStep(description=DEFAULT_PLAN_STEP)]
                    else:
                        kwargs["current_plan"] = [PlanStep(description=DEFAULT_PLAN_STEP)]

                # 2️⃣  tool_usage_stats  -- normalise numeric counters
                elif name == "tool_usage_stats":
                    stats_default = _default_tool_stats()
                    if isinstance(value, dict):
                        for k, stat in value.items():
                            if isinstance(stat, dict):
                                stats_default[k]["success"] = int(stat.get("success", 0))
                                stats_default[k]["failure"] = int(stat.get("failure", 0))
                                stats_default[k]["latency_ms_total"] = float(stat.get("latency_ms_total", 0.0))
                    kwargs["tool_usage_stats"] = stats_default

                # 3️⃣  goal_stack  -- list[dict]
                elif name == "goal_stack":
                    kwargs[name] = value if isinstance(value, list) and all(isinstance(i, dict) for i in value) else []

                # 4️⃣  everything else  -- pass through verbatim
                else:
                    kwargs[name] = value

            # ---------------------------------------------------------------------
            # B. Key missing in JSON → fall back to defaults
            # ---------------------------------------------------------------------
            else:
                if fld.default_factory is not dataclasses.MISSING:  # type: ignore[attr-defined]
                    kwargs[name] = fld.default_factory()            # type: ignore[attr-defined]
                elif fld.default is not dataclasses.MISSING:
                    kwargs[name] = fld.default
                elif name == "current_reflection_threshold":
                    kwargs[name] = BASE_REFLECTION_THRESHOLD
                elif name == "current_consolidation_threshold":
                    kwargs[name] = BASE_CONSOLIDATION_THRESHOLD
                elif name == "goal_stack":
                    kwargs[name] = []
                else:
                    # Leave unset; dataclass will enforce required fields
                    pass

        # -------------------------------------------------------------------------
        # C. Warn on unknown / extra keys
        # -------------------------------------------------------------------------
        extra_keys = set(data.keys()) - processed_keys - {"timestamp", "schema_version"}
        if extra_keys:
            self.logger.warning(f"State-file contains {len(extra_keys)} unknown keys → ignored: {sorted(extra_keys)}")

        # ── 4. Instantiate & validate AgentState ────────────────────────────────
        try:
            temp_state: AgentState = AgentState(**kwargs)
        except TypeError as exc:
            # Catastrophic mismatch – fall back to safe default
            self.logger.critical(
                f"AgentState(**kwargs) failed ({exc!s}). Using default state.",
                exc_info=True,
            )
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD,
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD,
            )
            return

        # -------------------------------------------------------------------------
        # Range / integrity checks
        # -------------------------------------------------------------------------
        if not (MIN_REFLECTION_THRESHOLD <= temp_state.current_reflection_threshold <= MAX_REFLECTION_THRESHOLD):
            self.logger.warning(
                f"Reflection threshold {temp_state.current_reflection_threshold} out of bounds. "
                f"Reset → {BASE_REFLECTION_THRESHOLD}"
            )
            temp_state.current_reflection_threshold = BASE_REFLECTION_THRESHOLD

        if not (MIN_CONSOLIDATION_THRESHOLD <= temp_state.current_consolidation_threshold <= MAX_CONSOLIDATION_THRESHOLD):
            self.logger.warning(
                f"Consolidation threshold {temp_state.current_consolidation_threshold} out of bounds. "
                f"Reset → {BASE_CONSOLIDATION_THRESHOLD}"
            )
            temp_state.current_consolidation_threshold = BASE_CONSOLIDATION_THRESHOLD

        # Ensure goal_stack consistency with current_goal_id
        if (
            temp_state.current_goal_id and
            not any(isinstance(g, dict) and g.get("goal_id") == temp_state.current_goal_id for g in temp_state.goal_stack)
        ):
            self.logger.warning(
                f"current_goal_id {_fmt_id(temp_state.current_goal_id)} missing from goal_stack; "
                "auto-realigning to last stack item."
            )
            temp_state.current_goal_id = (
                temp_state.goal_stack[-1]["goal_id"]  # type: ignore[index]
                if temp_state.goal_stack and isinstance(temp_state.goal_stack[-1], dict)
                else None
            )

        # ── 5. Commit loaded state & log summary ────────────────────────────────
        self.state = temp_state
        self.logger.info(
            "State loaded: loop %d • WF %s • Goal %s",
            self.state.current_loop,
            _fmt_id(self.state.workflow_id),
            _fmt_id(self.state.current_goal_id),
        )

    def _get_base_function_name(self, tool_name_input: str) -> str:
        """
        Extracts the base function name from various tool name formats.
        e.g., "Ultimate MCP Server:create_workflow" -> "create_workflow"
              "create_workflow" -> "create_workflow"
              "agent:update_plan" -> "update_plan"
        """
        return tool_name_input.split(":")[-1]

    def _is_ums_tool(self, tool_name: str) -> bool:
        """Check if a tool name refers to a UMS (Ultimate MCP Server) tool."""
        return (tool_name.startswith(f"{UMS_SERVER_NAME}:") or 
                self._get_base_function_name(tool_name) in self.all_ums_base_function_names)

    def _validate_ums_tool_arguments(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tool arguments against the tool's schema before calling UMS."""
        # Get the tool schema
        tool_obj = self.mcp_client.server_manager.tools.get(tool_name)
        if not tool_obj or not tool_obj.input_schema:
            return {"valid": True}  # Can't validate without schema
        
        try:
            # Use basic validation since jsonschema might not be available
            schema = tool_obj.input_schema
            if not isinstance(schema, dict):
                return {"valid": True}
            
            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in arguments:
                    return {
                        "valid": False,
                        "error": f"Missing required field: {field}",
                        "details": {"missing_field": field}
                    }
            
            # Special validation for known problematic fields
            base_tool_name = self._get_base_function_name(tool_name)
            
            # Validate memory_type for store_memory
            if base_tool_name == "store_memory" and "memory_type" in arguments:
                valid_memory_types = {
                    "observation", "action_log", "tool_output", "artifact_creation", 
                    "reasoning_step", "fact", "insight", "plan", "question", "summary", 
                    "reflection", "skill", "procedure", "pattern", "code", "json", 
                    "url", "user_input", "text"
                }
                memory_type = arguments.get("memory_type", "").lower()
                if memory_type not in valid_memory_types:
                    return {
                        "valid": False,
                        "error": f"Invalid memory_type '{memory_type}'. Use one of: {', '.join(sorted(valid_memory_types))}",
                        "details": {"invalid_field": "memory_type", "provided_value": memory_type, "valid_values": list(valid_memory_types)}
                    }
            
            # Validate artifact_type for record_artifact
            if base_tool_name == "record_artifact" and "artifact_type" in arguments:
                valid_artifact_types = {
                    "file", "text", "image", "table", "chart", "code", "data", "json", "url"
                }
                artifact_type = arguments.get("artifact_type", "").lower()
                if artifact_type not in valid_artifact_types:
                    return {
                        "valid": False,
                        "error": f"Invalid artifact_type '{artifact_type}'. Use one of: {', '.join(sorted(valid_artifact_types))}",
                        "details": {"invalid_field": "artifact_type", "provided_value": artifact_type, "valid_values": list(valid_artifact_types)}
                    }
            
            # Check field types for properties
            properties = schema.get("properties", {})
            for field, value in arguments.items():
                if field in properties:
                    expected_type = properties[field].get("type")
                    if expected_type:
                        if expected_type == "string" and not isinstance(value, str):
                            return {
                                "valid": False,
                                "error": f"Field '{field}' must be string, got {type(value).__name__}",
                                "details": {"invalid_field": field, "expected_type": expected_type}
                            }
                        elif expected_type == "object" and not isinstance(value, dict):
                            return {
                                "valid": False,
                                "error": f"Field '{field}' must be object, got {type(value).__name__}",
                                "details": {"invalid_field": field, "expected_type": expected_type}
                            }
                        elif expected_type == "array" and not isinstance(value, list):
                            return {
                                "valid": False,
                                "error": f"Field '{field}' must be array, got {type(value).__name__}",
                                "details": {"invalid_field": field, "expected_type": expected_type}
                            }
                        elif expected_type == "boolean" and not isinstance(value, bool):
                            return {
                                "valid": False,
                                "error": f"Field '{field}' must be boolean, got {type(value).__name__}",
                                "details": {"invalid_field": field, "expected_type": expected_type}
                            }
                        elif expected_type in ["integer", "number"] and not isinstance(value, (int, float)):
                            return {
                                "valid": False,
                                "error": f"Field '{field}' must be {expected_type}, got {type(value).__name__}",
                                "details": {"invalid_field": field, "expected_type": expected_type}
                            }
            
            return {"valid": True}
        except Exception as e:
            return {"valid": False, "error": f"Validation error: {e}"}

    def _generate_ums_tool_schema(self, tool_name: str, tool_def: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate a structured output schema for a specific UMS tool call."""
        
        try:
            # Get the tool's input schema
            input_schema = None
            if isinstance(tool_def, dict):
                if "input_schema" in tool_def:
                    input_schema = tool_def["input_schema"]
                elif "function" in tool_def and isinstance(tool_def["function"], dict):
                    input_schema = tool_def["function"].get("parameters", {})
                elif "parameters" in tool_def:
                    input_schema = tool_def["parameters"]
            
            if not isinstance(input_schema, dict):
                self.logger.debug(f"No valid input schema found for UMS tool {tool_name}")
                return None
            
            # Basic validation that the input schema looks reasonable
            if "type" not in input_schema:
                self.logger.warning(f"UMS tool {tool_name} input schema missing 'type' field")
                return None
                
            # Create a structured output schema that enforces the tool call format
            structured_schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "const": tool_name},
                    "arguments": input_schema  # Use the UMS tool's actual schema
                },
                "required": ["name", "arguments"],
                "additionalProperties": False
            }
            
            self.logger.debug(f"Generated structured output schema for UMS tool {tool_name}")
            return structured_schema
            
        except Exception as e:
            self.logger.error(f"Error generating structured output schema for UMS tool {tool_name}: {e}")
            return None


    async def _check_workflow_exists(self, workflow_id: str) -> bool:
        """
        Robustly checks whether the specified ``workflow_id`` exists in UMS.

        Returns
        -------
        bool
            * True  – workflow is confirmed to exist.
            * False – workflow does not exist, the check failed, or the result is inconclusive.
        """
        # ------------------------------------------------------------------ #
        # 0. Quick sanity checks
        # ------------------------------------------------------------------ #
        if not workflow_id:
            self.logger.error("AML Check WF Exists: Empty workflow_id received – treating as non-existent.")
            return False
        if getattr(self, "_shutdown_event", None) and self._shutdown_event.is_set():
            self.logger.warning("AML Check WF Exists: Shutdown in progress – aborting workflow existence check.")
            return False

        # ------------------------------------------------------------------ #
        # 1. Resolve UMS tool and confirm it is reachable
        # ------------------------------------------------------------------ #
        get_details_mcp_tool_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_WORKFLOW_DETAILS)
        wf_id_fmt = _fmt_id(workflow_id)
        self.logger.debug(f"AML Check WF Exists: Verifying WF {wf_id_fmt} using '{get_details_mcp_tool_name}'.")

        tool_server = self._find_tool_server(get_details_mcp_tool_name)
        if not tool_server:
            self.logger.error(
                f"AML Check WF Exists: Required tool '{get_details_mcp_tool_name}' (for server '{UMS_SERVER_NAME}') unavailable."
            )
            return False
        self.logger.debug(f"AML Check WF Exists: Tool server for '{get_details_mcp_tool_name}' resolved to: {tool_server}")


        # ------------------------------------------------------------------ #
        # 2. Issue tool call (with graceful back-off and timeout)
        # ------------------------------------------------------------------ #
        await asyncio.sleep(0.05)  # Gentle breather

        tool_args = {
            "workflow_id": workflow_id,
            "include_actions": False,
            "include_artifacts": False,
            "include_thoughts": False,
            "include_memories": False,
            "include_cognitive_states": False,
        }
        self.logger.debug(f"AML Check WF Exists (Debug): Full payload structure for UMS call={str(tool_args)[:1000]}")
        
        result_envelope: Optional[Dict[str, Any]] = None # Initialize to None
        try:
            timeout_sec = getattr(self, "WORKFLOW_CHECK_TIMEOUT_SEC", 5.0) # Ensure float
            if not isinstance(timeout_sec, (int, float)) or timeout_sec <= 0:
                timeout_sec = 5.0 # Fallback
            
            # Call the internal execution method which returns the standardized envelope
            result_envelope = await asyncio.wait_for(
                self._execute_tool_call_internal(
                    tool_name_mcp=get_details_mcp_tool_name,
                    arguments=tool_args,
                    record_action=False, # This is a meta-check, not a primary agent action
                ),
                timeout=timeout_sec,
            )
            self.logger.debug(f"AML Check WF Exists: Raw result_envelope from UMS call for WF {wf_id_fmt}: {str(result_envelope)[:500]}")

        except asyncio.TimeoutError:
            self.logger.error(f"AML Check WF Exists: Timeout ({timeout_sec}s) while calling UMS for WF {wf_id_fmt}.")
            return False
        except Exception as exc:
            # Log unexpected errors during the call itself
            self.logger.error(
                f"AML Check WF Exists: Unexpected exception during UMS call for WF {wf_id_fmt}: {exc}",
                exc_info=True, # Full traceback for unexpected issues
            )
            return False # Treat as non-existent on unexpected error

        # ------------------------------------------------------------------ #
        # 3. Evaluate envelope and payload
        # ------------------------------------------------------------------ #
        if not isinstance(result_envelope, dict):
            self.logger.error(
                f"AML Check WF Exists: Malformed envelope (type={type(result_envelope)}) for WF {wf_id_fmt}. Envelope: {str(result_envelope)[:200]}"
            )
            return False

        # Case A: The envelope itself reports failure (e.g., tool execution error before UMS logic)
        if not result_envelope.get("success", False):
            err_type = result_envelope.get("error_type", "UnknownError")
            err_msg = result_envelope.get("error_message", "UMS tool call failed at envelope level.")
            # ToolInputError with "not found" from UMS is a valid way to say it doesn't exist
            if "ToolInputError" in str(err_type) and "not found" in str(err_msg).lower():
                 self.logger.info(f"AML Check WF Exists: Workflow {wf_id_fmt} confirmed NOT to exist (UMS ToolInputError: not found).")
                 return False
            self.logger.warning(
                f"AML Check WF Exists: Envelope for WF {wf_id_fmt} reported failure: Type='{err_type}', Msg='{err_msg[:150]}...'"
            )
            return False # Treat other envelope failures as "cannot confirm existence"

        # Case B: Envelope is success, now inspect the UMS payload.
        # The `_execute_tool_call_internal` method wraps UMS responses in an envelope structure,
        # so the actual UMS data is in the `data` field of the envelope.
        
        # Extract the actual UMS payload from the envelope's data field
        ums_payload = result_envelope.get("data", {})
        
        # Check if the UMS payload explicitly states success=false
        if ums_payload.get("success") is False:
            self.logger.warning(
                f"AML Check WF Exists: UMS payload for WF {wf_id_fmt} indicates failure despite successful envelope. "
                f"UMS Error: {ums_payload.get('error_message', ums_payload.get('error', 'No details'))}"
            )
            return False

        # Positive confirmation: the workflow_id in the UMS payload matches the one we asked for.
        returned_workflow_id = ums_payload.get("workflow_id")
        if returned_workflow_id == workflow_id:
            self.logger.debug(f"AML Check WF Exists: Workflow {wf_id_fmt} positively confirmed to exist (ID match in UMS payload).")
            return True
        
        # If the returned ID is different, it's an issue (shouldn't happen with get_workflow_details)
        if returned_workflow_id and returned_workflow_id != workflow_id:
            self.logger.error(
                f"AML Check WF Exists: UMS returned details for a DIFFERENT workflow_id ('{returned_workflow_id}') "
                f"when checking for '{workflow_id}'. This is unexpected. Treating as non-existent."
            )
            return False

        # If no workflow_id field in a successful-looking payload from get_workflow_details
        # then the workflow probably doesn't exist (or payload is malformed).
        # The UMS get_workflow_details should raise ToolInputError if not found,
        # which _execute_tool_call_internal would turn into success=False envelope, handled above.
        # So, reaching here with a successful envelope but no matching workflow_id is unusual.
        self.logger.warning(
            f"AML Check WF Exists: UMS call for WF {wf_id_fmt} succeeded at envelope level, "
            f"but the UMS payload did not contain a matching 'workflow_id'. Payload keys: {list(ums_payload.keys())}. "
            f"Treating as non-existent."
        )
        return False

    async def _validate_agent_workflow_and_context(self) -> bool:
        """
        Ensures that the agent's `workflow_id` and `context_id` (cognitive-state ID)
        are both valid **and** mutually consistent with what UMS reports.

        Validation flow
        ---------------
        1. Verify that `self.state.workflow_id` exists in UMS.
        2. If a `self.state.context_id` is present, attempt to load it and confirm:
            • it belongs to that workflow **and**
            • it is flagged `is_latest` in UMS.
        - If the check fails (or raises), the stale `context_id` is cleared.
        3. If no valid context is present, load the **latest** cognitive state for
        the workflow from UMS and store its `state_id` back onto
        `self.state.context_id`.
        4. Return **True** only when both IDs are now valid, else **False**.

        This method never raises; it logs and returns False on all errors so callers
        can decide whether to trigger a re-plan, create a new workflow, etc.
        """
        wf_id: Optional[str] = self.state.workflow_id
        ctx_id: Optional[str] = self.state.context_id

        self.logger.debug(
            "AML Validate WF/CTX: Start - WF=%s, CTX=%s",
            _fmt_id(wf_id), _fmt_id(ctx_id)
        )

        # ------------------------------------------------------------------ #
        # 0. Quick guard: missing workflow in agent state                     #
        # ------------------------------------------------------------------ #
        if not wf_id:
            self.logger.info("AML Validate WF/CTX: No workflow_id set → invalid.")
            return False

        # ------------------------------------------------------------------ #
        # 1. Verify workflow exists in UMS                                    #
        # ------------------------------------------------------------------ #
        if not await self._check_workflow_exists(wf_id):
            self.logger.warning(
                "AML Validate WF/CTX: workflow_id %s not found in UMS.",
                _fmt_id(wf_id),
            )
            self.logger.error(f"AML Validate WF/CTX: _check_workflow_exists returned False for WF {_fmt_id(wf_id)}")
            return False
        self.logger.debug("AML Validate WF/CTX: Workflow %s exists.", _fmt_id(wf_id))

        # ------------------------------------------------------------------ #
        # 2. Prepare helper for load_cognitive_state                          #
        # ------------------------------------------------------------------ #
        load_state_tool = self._get_ums_tool_mcp_name(UMS_FUNC_LOAD_COGNITIVE_STATE)
        if not self._find_tool_server(load_state_tool):
            self.logger.error(
                "AML Validate WF/CTX: required tool %s unavailable.", UMS_FUNC_LOAD_COGNITIVE_STATE
            )
            return False

        async def _load_state(state_id: Optional[str]) -> tuple[bool, dict[str, Any]]:
            """
            Wrapper around `load_cognitive_state` that normalises the envelope.

            Returns
            -------
            (success, payload) where *payload* is the `data` dict from the envelope
            (empty dict on failure).
            """
            try:
                envelope: dict[str, Any] = await self._execute_tool_call_internal(
                    load_state_tool,
                    {"workflow_id": wf_id, "state_id": state_id},
                    record_action=False,
                )
            except Exception as exc:
                self.logger.error(
                    "AML Validate WF/CTX: exception calling load_cognitive_state(%s): %s",
                    _fmt_id(state_id), exc, exc_info=False
                )
                return False, {}

            if not envelope or not envelope.get("success"):
                self.logger.warning(
                    "AML Validate WF/CTX: load_cognitive_state(%s) failed – %s",
                    _fmt_id(state_id),
                    envelope.get("error_message", "no error_message"),
                )
                return False, {}

            payload: dict[str, Any] = envelope.get("data") or {}
            return True, payload

        # ------------------------------------------------------------------ #
        # 3. If we already have a context_id, validate it                     #
        # ------------------------------------------------------------------ #
        if ctx_id:
            ok, payload = await _load_state(ctx_id)
            if (
                ok
                and payload.get("state_id") == ctx_id
                and payload.get("workflow_id") == wf_id
                and payload.get("is_latest") is True
            ):
                self.logger.info(
                    "AML Validate WF/CTX: context_id %s is valid & latest.",
                    _fmt_id(ctx_id),
                )
                return True

            self.logger.warning(
                "AML Validate WF/CTX: context_id %s invalid/out-of-date → clearing.",
                _fmt_id(ctx_id),
            )
            # Stale or bad context — clear before trying again
            self.state.context_id = None
            ctx_id = None  # for clarity

        # ------------------------------------------------------------------ #
        # 4. No valid ctx → fetch the latest from UMS                         #
        # ------------------------------------------------------------------ #
        ok, payload = await _load_state(None)  # state_id=None ⇒ latest
        if (
            ok
            and isinstance(payload.get("state_id"), str)
            and payload.get("workflow_id") == wf_id
            and payload.get("is_latest") is True
        ):
            self.state.context_id = payload["state_id"]
            self.logger.info(
                "AML Validate WF/CTX: loaded latest context_id %s for WF %s.",
                _fmt_id(self.state.context_id), _fmt_id(wf_id)
            )
            return True

        self.logger.error(
            "AML Validate WF/CTX: unable to obtain a valid cognitive state for WF %s.",
            _fmt_id(wf_id),
        )
        # Make absolutely sure we don't hold a bad ID
        self.state.context_id = None
        return False

    async def initialize(self) -> bool:
        """Fully‑featured replacement for the old ``initialize`` implementation.
        """
        # ------------------------------------------------------------------
        # 0.  Helper lambdas – purely local, do *not* touch class namespace
        # ------------------------------------------------------------------
        _fmt = _fmt_id               # local alias – read‑only closure capture
        _l   = self.logger           # micro‑optimise log access, keeps lines slim

        def _log(level: int, msg: str) -> None:
            """Helper that only formats the f‑string when log‑level is active."""
            if _l.isEnabledFor(level):
                _l.log(level, msg)

        # ------------------------------------------------------------------
        # 1. Load on‑disk state ------------------------------------------------
        # ------------------------------------------------------------------
        _l.info("🤖 AML: Initializing Agent Master Loop (initialize v2)…")
        try:
            await self._load_agent_state()  # may raise, we catch below
        except Exception as exc:
            _l.exception("🤖 AML INIT ‑ CRITICAL: Failed to load agent state – proceeding with clean state.  Error: %s", exc)
            self.state = AgentState()  # hard reset

        loaded_wf_id  = self.state.workflow_id
        loaded_ctx_id = self.state.context_id
        state_is_valid_and_ready = False   # final flag – drives reset branch

        # ------------------------------------------------------------------
        # 2. Attempt validation path -----------------------------------------
        # ------------------------------------------------------------------
        if loaded_wf_id:
            _l.info(
                "🤖 AML Initialize: Loaded state has WF='%s', CTX='%s'. Performing validation/sync with UMS…",
                _fmt(loaded_wf_id), _fmt(loaded_ctx_id)
            )
            try:
                if await self._validate_agent_workflow_and_context():
                    _l.info(
                        "🤖 AML Initialize: Loaded/Synced state is VALID. WF='%s', CTX='%s'.",
                        _fmt(self.state.workflow_id), _fmt(self.state.context_id)
                    )
                    state_is_valid_and_ready = True

                    # --- keep workflow stack sane ------------------------------------
                    if not self.state.workflow_stack or self.state.workflow_stack[-1] != self.state.workflow_id:
                        self.state.workflow_stack = [self.state.workflow_id]
                        _l.info("🤖 AML Initialize: Workflow stack reset/set to: [%s]", _fmt(self.state.workflow_id))

                    # --- goal‑stack / thought‑chain validation ----------------------
                    await self._validate_goal_stack_on_load()

                    if not self.state.current_thought_chain_id:
                        _l.info("🤖 AML Initialize: No current_thought_chain_id – selecting default…")
                        await self._set_default_thought_chain_id()

                    # If a workflow is valid but lacks an operational goal, set a plan stub.
                    if self.state.workflow_id and not self.state.current_goal_id:
                        _l.info("🤖 AML Initialize: Valid workflow '%s' lacks operational goal – injecting plan step.", _fmt(self.state.workflow_id))
                        if not self.state.current_plan or self.state.current_plan[0].description == DEFAULT_PLAN_STEP:
                            self.state.current_plan = [PlanStep(description=f"Establish root UMS goal for existing active workflow: {_fmt(self.state.workflow_id)}")]
                            self.state.needs_replan = False
                            _l.info("🤖 AML Initialize: Plan updated to establish root UMS goal.")
            except Exception as exc:
                _l.exception("🤖 AML Initialize: Exception during validation – will fall back to reset.  Error: %s", exc)
                state_is_valid_and_ready = False

        # ------------------------------------------------------------------
        # 3. Reset path (either no WF or validation failed) ------------------
        # ------------------------------------------------------------------
        if not state_is_valid_and_ready:
            _l.info("🤖 AML Initialize: Resetting workflow‑specific agent state…")

            # Preserve non‑workflow statistics so we do not lose learning.
            preserved_tool_stats  = copy.deepcopy(self.state.tool_usage_stats)
            pres_ref_thresh       = self.state.current_reflection_threshold
            pres_con_thresh       = self.state.current_consolidation_threshold

            # Fresh state object
            self.state = AgentState()

            # Re‑apply preserved, non‑workflow values
            self.state.tool_usage_stats              = preserved_tool_stats
            self.state.current_reflection_threshold  = pres_ref_thresh
            self.state.current_consolidation_threshold = pres_con_thresh

            # Prime minimal plan and meta fields
            self.state.current_plan        = [PlanStep(description=DEFAULT_PLAN_STEP)]
            self.state.last_action_summary = "Agent state reset: No valid prior workflow/context, or validation failed."
            self.state.needs_replan        = False
            _l.info("🤖 AML Initialize: Agent state fully reset; starting fresh.")

        # ------------------------------------------------------------------
        # 4. Persist whatever state we ended up with -------------------------
        # ------------------------------------------------------------------
        await self._save_agent_state()
        _l.info(
            "🤖 AML Initialize: State finalised. WF='%s', CTX='%s', Goal='%s', NeedsReplan=%s",
            _fmt(self.state.workflow_id), _fmt(self.state.context_id), _fmt(self.state.current_goal_id), self.state.needs_replan
        )

        # ------------------------------------------------------------------
        # 5. TOOL‑SCHEMA PREPARATION ----------------------------------------
        # ------------------------------------------------------------------
        _l.info("🤖 AML Initialize: Starting tool‑schema setup phase…")
        if not getattr(self.mcp_client, "server_manager", None):
            _l.critical("🤖 AML CRITICAL: MCPClient or its ServerManager is not initialised – cannot continue.")
            return False

        agent_llm_provider = self.mcp_client.get_provider_from_model(self.agent_llm_model)
        if not agent_llm_provider:
            _l.critical("🤖 AML CRITICAL: Cannot determine LLM provider for model '%s'.", self.agent_llm_model)
            return False

        # ------------------------------------------------------------------
        # 5a.  Fetch & re‑sanitise MCP tool schemas -------------------------
        # ------------------------------------------------------------------
        all_llm_formatted = self.mcp_client._format_tools_for_provider(agent_llm_provider) or []
        _l.info("🤖 AML: Received %d LLM‑formatted tool schemas from MCPClient (provider=%s).", len(all_llm_formatted), agent_llm_provider)

        self.tool_schemas = []
        final_used_names: Set[str] = set()
        s2o = self.mcp_client.server_manager.sanitized_to_original
        is_anthropic = agent_llm_provider == Provider.ANTHROPIC.value

        for idx, schema in enumerate(all_llm_formatted):
            # --- extract the sanitised name field depending on provider format
            name_field = "name" if is_anthropic else (schema.get("function", {}) if isinstance(schema, dict) else {})
            sanitized_name = schema.get("name") if is_anthropic else name_field.get("name") if isinstance(name_field, dict) else ""
            if not sanitized_name:
                _log(logging.WARNING, f"🤖 AML (Tool Init {idx+1}): Skipping schema with missing name: {str(schema)[:120]}")
                continue

            original_mcp = s2o.get(sanitized_name)
            if not original_mcp:
                _log(logging.ERROR, f"🤖 AML (Tool Init {idx+1}): Missing original‑name mapping for '{sanitized_name}'. Skipping.")
                continue

            # ensure uniqueness for the agent‑side name
            final_name = sanitized_name
            counter = 1
            while final_name in final_used_names:
                suffix = f"_agent_v{counter}"
                base = sanitized_name[:64 - len(suffix)]
                final_name = f"{base}{suffix}"
                counter += 1
            final_used_names.add(final_name)

            # deep‑copy & mutate if name changed
            working_schema = copy.deepcopy(schema)
            if final_name != sanitized_name:
                if is_anthropic:
                    working_schema["name"] = final_name
                else:
                    working_schema.setdefault("function", {})["name"] = final_name
                # update shared map so downstream mapping stays correct
                s2o.pop(sanitized_name, None)
                s2o[final_name] = original_mcp
                _log(logging.INFO, f"🤖 AML (Tool Init): Re‑sanitised '{sanitized_name}' -> '{final_name}' (orig='{original_mcp}')")

            self.tool_schemas.append(working_schema)

        # ------------------------------------------------------------------
        # 5b.  Inject AGENT_TOOL_UPDATE_PLAN --------------------------------
        # ------------------------------------------------------------------
        plan_step_schema = PlanStep.model_json_schema(); plan_step_schema.pop("title", None)
        update_plan_input = {
            "type": "object",
            "properties": {
                "plan": {
                    "type": "array",
                    "items": plan_step_schema,
                    "description": "The new complete list of plan steps."
                }
            },
            "required": ["plan"]
        }
        original_agent_tool = AGENT_TOOL_UPDATE_PLAN
        sanitized_base = re.sub(r"[^a-zA-Z0-9_-]", "_", original_agent_tool)[:64] or f"internal_tool_{uuid.uuid4().hex[:8]}"
        final_agent_name = sanitized_base
        counter = 1
        while final_agent_name in final_used_names:
            suffix = f"_agent_v{counter}"
            final_agent_name = f"{sanitized_base[:64 - len(suffix)]}{suffix}"
            counter += 1
        final_used_names.add(final_agent_name)
        s2o[final_agent_name] = original_agent_tool
        _l.info("🤖 AML (Agent Tool Init): Added mapping '%s' -> '%s'", final_agent_name, original_agent_tool)

        desc = "Replace agent's current plan. Use for significant replanning, error recovery, or fixing validation issues. Submit the ENTIRE new plan."
        if is_anthropic:
            plan_tool_schema = {"name": final_agent_name, "description": desc, "input_schema": update_plan_input}
        else:
            plan_tool_schema = {"type": "function", "function": {"name": final_agent_name, "description": desc, "parameters": update_plan_input}}
        self.tool_schemas.append(plan_tool_schema)

        _l.info("🤖 AML: Total %d tool schemas prepared.", len(self.tool_schemas))

        # ------------------------------------------------------------------
        # 6. Verify essential tools are present -----------------------------
        # ------------------------------------------------------------------
        essential = [
            self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_WORKFLOW),
            self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL),
            self._get_ums_tool_mcp_name(UMS_FUNC_GET_RICH_CONTEXT_PACKAGE),
            AGENT_TOOL_UPDATE_PLAN,
        ]
        available_original = set(s2o.values())
        missing = [tool for tool in essential if tool not in available_original]
        if missing:
            _l.critical("🤖 AML CRITICAL: Missing essential tools: %s", missing)
            return False
        _l.info("🤖 AML: All essential tools confirmed available.")

        # ------------------------------------------------------------------
        # 7.  Success! -------------------------------------------------------
        # ------------------------------------------------------------------
        _l.info("🤖 AML: Initialization complete. WF='%s', CTX='%s', Goal='%s'", _fmt(self.state.workflow_id), _fmt(self.state.context_id), _fmt(self.state.current_goal_id))
        return True

    async def run_main_loop(self, overall_goal: str, max_loops: int) -> Optional[Dict[str, Any]]:
        """
        Main loop method that MCPClient expects to exist.
        This prepares context and data for a single LLM turn.
        
        Returns:
            Dict with 'prompt_messages' and 'tool_schemas' keys, or None to signal termination
        """
        # Check termination conditions
        if (self.state.goal_achieved_flag or 
            self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS or
            self.state.current_loop >= max_loops or
            self._shutdown_event.is_set()):
            return None
        
        self.state.current_loop += 1
        
        # CRITICAL: Check if current goal is ready for completion before gathering context
        if self.state.current_goal_id and self.state.goal_stack:
            current_goal = self.state.goal_stack[-1] if self.state.goal_stack else {}
            goal_desc = current_goal.get("description", "")
            
            if goal_desc and goal_desc != "Overall UMS Workflow Goal or Initial Task":
                goal_type = self._classify_goal_type(goal_desc)
                completion_status = await self._analyze_goal_completion_semantically(goal_desc, goal_type)
                
                # Force completion if goal is ready or exceeded limits
                if completion_status["ready_for_completion"] or completion_status["force_completion"]:
                    self.logger.info(f"🎯 Goal completion detected - triggering automatic completion: {completion_status['completion_message']}")
                    
                    # Set goal achieved flag to trigger proper completion
                    self.state.goal_achieved_flag = True
                    self.state.last_action_summary = f"Goal automatically completed: {completion_status['completion_message']}"
                    
                    # Update goal status to completed
                    try:
                        update_goal_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_GOAL_STATUS)
                        if self._find_tool_server(update_goal_mcp):
                            await self._execute_tool_call_internal(
                                update_goal_mcp,
                                {
                                    "goal_id": self.state.current_goal_id,
                                    "status": GoalStatus.COMPLETED.value,
                                    "completion_message": f"Auto-completed: {completion_status['completion_message']}"
                                },
                                record_action=False
                            )
                            self.logger.info(f"✅ Updated goal {self.state.current_goal_id} status to completed")
                    except Exception as e:
                        self.logger.error(f"Failed to update goal status: {e}")
                    
                    return None  # Signal termination due to goal completion
        
        # Gather context for the LLM turn
        try:
            context_payload = await self._gather_context()
            
            # Construct prompt messages using the existing method
            prompt_messages = await self._construct_agent_prompt(overall_goal, context_payload)
            
            # Return data in the format MCPClient expects
            return {
                "prompt_messages": prompt_messages,
                "tool_schemas": self.tool_schemas,
            }
            
        except Exception as e:
            self.logger.error(f"Error in run_main_loop: {e}", exc_info=True)
            self.state.last_error_details = {
                "error": str(e),
                "type": "AgentMainLoopError"
            }
            self.state.consecutive_error_count += 1
            
            # Create minimal error context
            error_context = {
                "agent_name": AGENT_NAME,
                "current_loop": self.state.current_loop,
                "last_error_details": self.state.last_error_details,
                "status_message_from_agent": f"Error in main loop: {str(e)[:100]}",
                "needs_replan": True,
                "errors_in_context_gathering": [f"Main loop error: {str(e)}"],
            }
            
            # Construct error recovery prompt
            error_prompt_messages = await self._construct_agent_prompt(overall_goal, error_context)
            
            return {
                "prompt_messages": error_prompt_messages,
                "tool_schemas": self.tool_schemas,
            }

    def _find_tool_server(self, tool_identifier_mcp_style: str) -> Optional[str]:
        """
        Locate an **active** server that exposes the tool described by
        `tool_identifier_mcp_style`. Now with caching for improved performance.

        Parameters
        ----------
        tool_identifier_mcp_style : str
            • Either a fully–qualified MCP identifier
            ``"<ServerName>:<BaseFunctionName>"`` **or**
            • a bare function name (``"create_goal"``) **or**
            • ``AGENT_TOOL_UPDATE_PLAN`` (special in-process tool).

        Returns
        -------
        Optional[str]
            The *server name* that should run the tool, or ``None`` if a safe
            choice cannot be made.

        Behaviour & resolution strategy
        -------------------------------
        1. **Fast-path**:  
        • ``AGENT_TOOL_UPDATE_PLAN`` → `"AGENT_INTERNAL"`  
        2. Parse *expected* server hint and *target* base-function name.  
        3. Build the **candidate set** of *active* servers that expose that
        base-function.  
        4. Resolve ambiguity (≥ 2 candidates) in this order:  
        a. server-hint match (case-insensitive).  
        b. if the function is a *UMS core* function, prefer ``UMS_SERVER_NAME``  
            when present in the candidates set (case-insensitive).  
        5. If ambiguity remains, log and return ``None`` instead of guessing.
        """
        # Check cache first for performance improvement
        cached_server = self.tool_cache.get_server(tool_identifier_mcp_style)
        if cached_server is not None:
            self.logger.debug(f"AML _find_tool_server: using cached server '{cached_server}' for '{tool_identifier_mcp_style}'")
            return cached_server

        self.logger.debug(
            "AML _find_tool_server: locating server for identifier '%s'",
            tool_identifier_mcp_style,
        )

        # ------------------------------------------------------------------ guard rails
        if not tool_identifier_mcp_style:
            self.logger.warning("AML _find_tool_server: empty identifier supplied")
            self.tool_cache.set_server(tool_identifier_mcp_style, None)
            return None

        if not (self.mcp_client and getattr(self.mcp_client, "server_manager", None)):
            self.logger.warning(
                "AML _find_tool_server: MCPClient or ServerManager unavailable"
            )
            self.tool_cache.set_server(tool_identifier_mcp_style, None)
            return None

        sm = self.mcp_client.server_manager

        # ------------------------------------------------------------------ special case
        if tool_identifier_mcp_style == AGENT_TOOL_UPDATE_PLAN:
            self.logger.debug(
                "AML _find_tool_server: '%s' is internal – returning 'AGENT_INTERNAL'",
                AGENT_TOOL_UPDATE_PLAN,
            )
            self.tool_cache.set_server(tool_identifier_mcp_style, "AGENT_INTERNAL")
            return "AGENT_INTERNAL"

        # ------------------------------------------------------------------ parse parts
        # Split *once* on ':' to obtain possible server-name hint.
        server_hint, _, func_part = tool_identifier_mcp_style.partition(":")
        expected_server_name_hint = server_hint if _ else None   # '' → None
        target_base_function_name = self._get_base_function_name(func_part or server_hint)

        self.logger.debug(
            "AML _find_tool_server: target base-function='%s', server-hint='%s'",
            target_base_function_name,
            expected_server_name_hint,
        )

        # ------------------------------------------------------------------ gather candidates
        candidate_servers: Set[str] = set()

        for mcp_tool_name, mcp_tool_obj in sm.tools.items():
            # Normalise base-function for comparison
            if self._get_base_function_name(mcp_tool_name) != target_base_function_name:
                continue

            srv_name = getattr(mcp_tool_obj, "server_name", None)
            if not srv_name:
                continue

            # Check active session
            if srv_name in sm.active_sessions:
                candidate_servers.add(srv_name)
                self.logger.debug(
                    "AML _find_tool_server:   + '%s' provides '%s' (ACTIVE)",
                    srv_name,
                    target_base_function_name,
                )
            else:
                self.logger.debug(
                    "AML _find_tool_server:   - '%s' provides '%s' but is NOT active",
                    srv_name,
                    target_base_function_name,
                )

        # ------------------------------------------------------------------ evaluate candidates
        if not candidate_servers:
            self.logger.warning(
                "AML _find_tool_server: no ACTIVE server provides '%s' (identifier='%s')",
                target_base_function_name,
                tool_identifier_mcp_style,
            )
            self.tool_cache.set_server(tool_identifier_mcp_style, None)
            return None

        if len(candidate_servers) == 1:
            chosen = next(iter(candidate_servers))
            self.logger.info(
                "AML _find_tool_server: unique match – '%s' hosts '%s'",
                chosen,
                target_base_function_name,
            )
            self.tool_cache.set_server(tool_identifier_mcp_style, chosen)
            return chosen

        # ------------------------------------------------------------------ ambiguous – resolve
        self.logger.debug(
            "AML _find_tool_server: ambiguity (%s) for '%s', resolving…",
            sorted(candidate_servers),
            target_base_function_name,
        )

        # 1️⃣  server hint (case-insensitive exact match)
        if expected_server_name_hint:
            for srv in candidate_servers:
                if srv.casefold() == expected_server_name_hint.casefold():
                    self.logger.info(
                        "AML _find_tool_server: resolved via server-hint → '%s'",
                        srv,
                    )
                    self.tool_cache.set_server(tool_identifier_mcp_style, srv)
                    return srv

        # 2️⃣  prefer UMS server for core UMS functions
        if (
            target_base_function_name in self.all_ums_base_function_names
            and any(srv.casefold() == UMS_SERVER_NAME.casefold() for srv in candidate_servers)
        ):
            ums_server = next(
                srv for srv in candidate_servers if srv.casefold() == UMS_SERVER_NAME.casefold()
            )
            self.logger.info(
                "AML _find_tool_server: resolved UMS core '%s' → '%s'",
                target_base_function_name,
                UMS_SERVER_NAME,
            )
            self.tool_cache.set_server(tool_identifier_mcp_style, ums_server)
            return ums_server

        # ------------------------------------------------------------------ still ambiguous
        self.logger.warning(
            "AML _find_tool_server: cannot disambiguate server for '%s'; candidates=%s ; identifier='%s'",
            target_base_function_name,
            sorted(candidate_servers),
            tool_identifier_mcp_style,
        )
        self.tool_cache.set_server(tool_identifier_mcp_style, None)
        return None


    async def _set_default_thought_chain_id(self) -> None:
        """
        Populate ``self.state.current_thought_chain_id`` with the *first* thought-chain
        attached to the agent's current workflow, **if** the field is still unset.
        """
        # ------------------------------------------------------------------ constants
        DEFAULT_TOOL_TIMEOUT_SEC = 15           # hard stop for the UMS call
        TOOL_NAME = UMS_FUNC_GET_WORKFLOW_DETAILS  # base func name (no MCP prefix)

        # ------------------------------------------------------------------ short-circuit
        if self.state.current_thought_chain_id:
            # Nothing to do – keep the already-selected chain
            self.logger.debug(
                "Default thought-chain already set: %s",
                _fmt_id(self.state.current_thought_chain_id),
            )
            return self.state.current_thought_chain_id

        current_wf_id = (
            self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        )
        if not current_wf_id:
            self.logger.debug("Cannot set default thought chain – no active workflow.")
            return None

        # ------------------------------------------------------------------ tool discovery
        mcp_tool_name = self._get_ums_tool_mcp_name(TOOL_NAME)
        if not self._find_tool_server(mcp_tool_name):
            self.logger.warning("UMS tool '%s' not available – cannot fetch workflow details.", mcp_tool_name)
            return None

        # ------------------------------------------------------------------ call UMS
        try:
            envelope: Dict[str, Any] = await asyncio.wait_for(
                self._execute_tool_call_internal(
                    mcp_tool_name,
                    {
                        "workflow_id": current_wf_id,
                        "include_thoughts": True,
                        "include_actions": False,
                        "include_artifacts": False,
                        "include_memories": False,
                    },
                    record_action=False,
                ),
                timeout=DEFAULT_TOOL_TIMEOUT_SEC,
            )
        except asyncio.TimeoutError:
            self.logger.error(
                "Timeout (%ss) while calling '%s' for workflow %s.",
                DEFAULT_TOOL_TIMEOUT_SEC,
                mcp_tool_name,
                _fmt_id(current_wf_id),
            )
            return None
        except Exception as exc:
            self.logger.error(
                "Exception while calling '%s' for workflow %s: %s",
                mcp_tool_name,
                _fmt_id(current_wf_id),
                exc,
                exc_info=True,
            )
            return None

        # ------------------------------------------------------------------ validate envelope
        if not envelope.get("success"):
            self.logger.error(
                "UMS tool '%s' failed for workflow %s: %s",
                mcp_tool_name,
                _fmt_id(current_wf_id),
                envelope.get("error_message") or envelope.get("error"),
            )
            return None

        data = envelope.get("data", {})
        if not isinstance(data, dict):
            self.logger.error(
                "UMS tool '%s' returned non-dict data (%s) for workflow %s.",
                mcp_tool_name,
                type(data).__name__,
                _fmt_id(current_wf_id),
            )
            return None

        # UMS may embed the list directly or under 'workflow'
        thought_chains = (
            data.get("thought_chains")
            or data.get("workflow", {}).get("thought_chains")
            or []
        )

        if not (isinstance(thought_chains, list) and thought_chains):
            self.logger.warning(
                "No thought chains found for workflow %s.",
                _fmt_id(current_wf_id),
            )
            return None

        first_chain = thought_chains[0]
        chain_id = first_chain.get("thought_chain_id") if isinstance(first_chain, dict) else None
        if not chain_id:
            self.logger.warning(
                "First thought chain in workflow %s lacks an ID.",
                _fmt_id(current_wf_id),
            )
            return None

        # ------------------------------------------------------------------ commit to state
        self.state.current_thought_chain_id = chain_id
        self.logger.info(
            "Default thought-chain set to %s for workflow %s.",
            _fmt_id(chain_id),
            _fmt_id(current_wf_id),
        )
        return chain_id



    async def _validate_goal_stack_on_load(self) -> None:
        """
        Re-hydrates and validates the agent's in-memory goal stack after a restart.
        """
        # ------------------------------------------------------------------ #
        # Ensure we have a lock (older instances may not). This does *not*
        # change semantics in single-threaded contexts.
        # ------------------------------------------------------------------ #
        if not hasattr(self, "_state_lock"):
            self._state_lock = asyncio.Lock()

        async with self._state_lock:
            # ------ 1. No workflow → nothing to validate ------------------- #
            if not self.state.workflow_id:
                self.logger.warning(
                    "🛑 Goal-stack validation skipped: no active workflow_id."
                )
                self.state.goal_stack.clear()
                self.state.current_goal_id = None
                return

            # ------ 2. We *have* a current_goal_id ------------------------ #
            if self.state.current_goal_id:
                goal_id_fmt = _fmt_id(self.state.current_goal_id)
                self.logger.info("🔍 Validating goal stack; current_goal_id=%s", goal_id_fmt)

                FETCH_TIMEOUT_SEC = 10
                try:
                    ums_stack = await asyncio.wait_for(
                        self._fetch_goal_stack_from_ums(self.state.current_goal_id),
                        timeout=FETCH_TIMEOUT_SEC,
                    )
                except asyncio.TimeoutError:
                    self.logger.error(
                        "⏰ Timed-out (>%ss) while fetching goal stack for %s. "
                        "Clearing local goal context.",
                        FETCH_TIMEOUT_SEC,
                        goal_id_fmt,
                    )
                    self.state.goal_stack.clear()
                    self.state.current_goal_id = None
                    return

                if ums_stack:
                    ums_leaf_id = ums_stack[-1].get("goal_id")
                    if ums_leaf_id == self.state.current_goal_id:
                        # Happy path – synchronise local stack
                        self.state.goal_stack = ums_stack
                        self.logger.info(
                            "✅ Goal stack synchronised from UMS (leaf=%s, depth=%d).",
                            goal_id_fmt,
                            len(self.state.goal_stack),
                        )
                    else:
                        # Mismatch – leaf differs
                        self.logger.warning(
                            "⚠️  Local current_goal_id %s is not the leaf returned "
                            "by UMS (%s). Resetting goal context.",
                            goal_id_fmt,
                            _fmt_id(ums_leaf_id),
                        )
                        self.state.goal_stack.clear()
                        self.state.current_goal_id = None
                else:  # empty list → goal vanished or fetch failed silently
                    self.logger.warning(
                        "⚠️  UMS returned an empty stack for goal %s. "
                        "Clearing goal context.",
                        goal_id_fmt,
                    )
                    self.state.goal_stack.clear()
                    self.state.current_goal_id = None
                return  # handled all branches above

            # ------ 3. No current_goal_id but a residual stack ------------- #
            if self.state.goal_stack:
                self.logger.info(
                    "⚠️  goal_stack present (%d items) but current_goal_id is None. "
                    "Resetting stack.",
                    len(self.state.goal_stack),
                )
                self.state.goal_stack.clear()
            else:
                self.logger.debug(
                    "ℹ️  No goal information to validate – state is clean."
                )


    def _detect_plan_cycle(self, plan_steps: Iterable["PlanStep"]) -> bool:
        """
        Detects whether the given collection of PlanStep objects contains a
        dependency cycle.

        A cycle exists when, starting from any step, following `depends_on`
        edges eventually revisits the same step.

        This implementation is **iterative** (explicit stack) so it is safe
        for very deep graphs ≥ 10,000 nodes without blowing Python's call-stack.

        Args:
            plan_steps: Any iterable of PlanStep instances.  Each PlanStep is
                        expected to expose `.id` (hashable) and
                        `.depends_on` (Iterable[Hashable]) attributes.

        Returns:
            bool: True - at least one cycle detected  
                False - no cycles
        """
        # ------------------------------------------------------------------
        # Build quick-lookup table: step_id -> depends_on (as tuple for speed)
        # ------------------------------------------------------------------
        dep_map: Dict[str, tuple] = {}
        for step in plan_steps:
            try:
                dep_map[step.id] = tuple(step.depends_on or ())
            except AttributeError:
                raise TypeError(
                    "Each PlanStep must provide 'id' and 'depends_on' attributes."
                ) from None

        # Fast-exit for trivial sizes
        if len(dep_map) < 2:
            return False

        # ------------------------------------------------------------------
        # Iterative DFS with tri-color marking:
        #   white = unseen, gray = in current DFS stack, black = fully visited
        # A gray → gray edge signals a cycle.
        # ------------------------------------------------------------------
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {sid: WHITE for sid in dep_map} # noqa: C420

        for root in dep_map.keys():                       # try every component
            if color[root] != WHITE:
                continue

            stack: Deque[tuple[str, int]] = deque([(root, 0)])
            while stack:
                node, idx = stack.pop()

                if color[node] == WHITE:
                    # First time we see the node – mark gray & push back for post-visit
                    color[node] = GRAY
                    stack.append((node, 0))               # post-visit marker
                    # Push all children for pre-visit
                    for child in dep_map[node]:
                        if child not in dep_map:           # dangling dep → ignore
                            continue
                        if color[child] == GRAY:           # gray → gray  ⇒ cycle
                            return True
                        if color[child] == WHITE:
                            stack.append((child, 0))

                elif color[node] == GRAY:
                    # Post-visit – mark black so future traversals skip it
                    color[node] = BLACK
                    # (nothing else to do – children already processed)

                # color[node] == BLACK is impossible here because we never push blacks

        return False
    

    async def _check_prerequisites(self, ids: List[str]) -> Tuple[bool, str]:
        """
        Verify that every action-ID in *ids* exists **and** is already
        `ActionStatus.COMPLETED`.

        Returns
        -------
        Tuple[bool, str]
            • First element – *True* when every prerequisite is satisfied.  
            • Second element – human-readable reason/message.
        """
        # ------------------------------------------------------------------
        # 0. Fast exit – nothing to check
        # ------------------------------------------------------------------
        if not ids:
            return True, "No dependencies."

        # De-duplicate while preserving order.
        unique_ids: List[str] = list(dict.fromkeys(filter(None, ids)))
        if not unique_ids:
            return True, "No valid dependency IDs."

        # ------------------------------------------------------------------
        # 1. Resolve MCP tool
        # ------------------------------------------------------------------
        get_action_details_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_ACTION_DETAILS)
        if not self._find_tool_server(get_action_details_mcp_name):
            msg = f"Tool for '{UMS_FUNC_GET_ACTION_DETAILS}' unavailable."
            self.logger.error(msg)
            return False, msg

        # ------------------------------------------------------------------
        # 2. Parameters & helpers
        # ------------------------------------------------------------------
        CHUNK_SIZE: int = 50          # payload size safety-valve
        RETRY_ATTEMPTS: int = 3
        RETRY_BACKOFF_SEC: float = 0.8
        TOOL_TIMEOUT_SEC: float = 15.0

        async def _fetch_chunk(chunk: List[str]) -> Optional[Dict[str, Any]]:
            """
            Call UMS for a single chunk with bounded retries / timeout.
            Returns the tool envelope (dict) on success or None on failure.
            """
            for attempt in range(1, RETRY_ATTEMPTS + 1):
                try:
                    self.logger.debug(
                        f"Prereq-check: fetching details for "
                        f"{', '.join(_fmt_id(i) for i in chunk)} (try {attempt}/{RETRY_ATTEMPTS})"
                    )
                    res = await asyncio.wait_for(
                        self._execute_tool_call_internal(
                            get_action_details_mcp_name,
                            {"action_ids": chunk, "include_dependencies": False},
                            record_action=False,
                        ),
                        timeout=TOOL_TIMEOUT_SEC,
                    )
                    return res
                except asyncio.TimeoutError:
                    self.logger.warning(
                        f"Prereq-check: timeout ({TOOL_TIMEOUT_SEC}s) on attempt {attempt} "
                        f"for IDs {', '.join(_fmt_id(i) for i in chunk)}"
                    )
                except Exception as e:
                    self.logger.error(
                        f"Prereq-check: exception '{e}' on attempt {attempt} "
                        f"for IDs {', '.join(_fmt_id(i) for i in chunk)}",
                        exc_info=True,
                    )
                await asyncio.sleep(RETRY_BACKOFF_SEC * attempt)  # simple exponential back-off
            return None  # all retries exhausted

        # ------------------------------------------------------------------
        # 3. Gather all action meta in parallel (chunked)
        # ------------------------------------------------------------------
        chunks: List[List[str]] = [
            unique_ids[i : i + CHUNK_SIZE] for i in range(0, len(unique_ids), CHUNK_SIZE)
        ]
        # Parallel fetch with limited concurrency to avoid hammering the UMS.
        SEMAPHORE = asyncio.Semaphore(5)

        async def _bounded_fetch(chunk):
            async with SEMAPHORE:
                return await _fetch_chunk(chunk)

        envelopes = await asyncio.gather(*(_bounded_fetch(c) for c in chunks))

        # ------------------------------------------------------------------
        # 4. Analyse results
        # ------------------------------------------------------------------
        actions_found: List[Dict[str, Any]] = []
        for env, chunk in zip(envelopes, chunks, strict=True):
            if not env or not env.get("success"):
                err_msg = (
                    (env or {}).get("error")
                    or "UMS call failed or timed-out."
                )
                self.logger.warning(
                    f"Prereq-check: Failed to retrieve chunk "
                    f"{', '.join(_fmt_id(i) for i in chunk)} – {err_msg}"
                )
                return False, f"Failed: {err_msg}"
            # Successful envelope – merge actions
            actions_found.extend(env.get("actions", []))

        found_ids: set = {a.get("action_id") for a in actions_found}
        missing_ids: List[str] = [i for i in unique_ids if i not in found_ids]
        if missing_ids:
            self.logger.warning(f"Prereq-check: actions not found: {', '.join(_fmt_id(i) for i in missing_ids)}")
            return False, f"Not found: {', '.join(_fmt_id(i) for i in missing_ids)}"

        # Check completion status
        incomplete_descriptions: List[str] = []
        for act in actions_found:
            status = str(act.get("status", "")).upper()
            if status != ActionStatus.COMPLETED.value:
                title_or_id = act.get("title") or _fmt_id(act.get("action_id"))
                incomplete_descriptions.append(f"'{title_or_id}' (Status: {status or 'UNK'})")

        if incomplete_descriptions:
            reason = f"Not completed: {', '.join(incomplete_descriptions)}"
            self.logger.warning(f"Prereq-check: {reason}")
            return False, reason

        self.logger.debug("Prereq-check: all dependencies completed.")
        return True, "All deps completed."


    async def _record_action_start_internal(
        self,
        tool_name_mcp: str,
        tool_args: Dict[str, Any],
        planned_dependencies: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Create an **Action Start** record in UMS before executing a tool.

        Parameters
        ----------
        tool_name_mcp : str
            *Original* MCP name of the tool being executed (e.g. ``"Ultimate MCP Server:some_tool"``).
        tool_args : Dict[str, Any]
            Arguments that will be passed to the tool call (deep-copied for safety).
        planned_dependencies : Optional[List[str]]
            Memory / artifact IDs this action is expected to depend on.

        Returns
        -------
        Optional[str]
            The newly-created ``action_id`` (UMS UUID) or ``None`` if the record
            could not be created.
        """
        # ------------------------------------------------------------------ #
        # 1. Resolve the UMS tool we need and make sure a server is present. #
        # ------------------------------------------------------------------ #
        record_action_start_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_ACTION_START)

        if not self._find_tool_server(record_action_start_mcp):
            self.logger.error("Tool server for '%s' unavailable.", UMS_FUNC_RECORD_ACTION_START)
            return None

        # -------------------------------------------------------------- #
        # 2. Work out which workflow this action belongs to.             #
        # -------------------------------------------------------------- #
        current_wf_id: Optional[str] = (
            self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        )
        if not current_wf_id:
            self.logger.warning("Attempted to record action start, but no active workflow is set.")
            return None

        # -------------------------------------------------------------- #
        # 3. Build the payload – keep caller's dict untouched.           #
        # -------------------------------------------------------------- #
        safe_tool_args: Dict[str, Any] = copy.deepcopy(tool_args)

        payload: Dict[str, Any] = {
            "workflow_id": current_wf_id,
            "title": f"Execute: {tool_name_mcp.split(':')[-1]}",
            "action_type": ActionType.TOOL_USE.value,
            "tool_name": tool_name_mcp,
            "tool_args": safe_tool_args,
            "reasoning": f"Agent initiated call to tool: {tool_name_mcp}",
        }

        # -------------------------------------------------------------- #
        # 4. Call the UMS tool and handle the response.                  #
        # -------------------------------------------------------------- #
        try:
            res: Dict[str, Any] = await self._execute_tool_call_internal(
                record_action_start_mcp,
                payload,
                record_action=False,  # ← avoid recursive action logging
            )
        except Exception as exc:
            self.logger.error(
                "Exception while calling '%s' for tool '%s': %s",
                record_action_start_mcp,
                tool_name_mcp,
                exc,
                exc_info=True,
            )
            return None

        if not res.get("success"):
            self.logger.error(
                "Failed to record action start for '%s': %s",
                tool_name_mcp,
                res.get("error"),
            )
            return None

        # Try to find action_id in various possible locations
        action_id: Optional[str] = None
        if isinstance(res.get("data"), dict):
            action_id = res["data"].get("action_id")
        if not action_id:
            action_id = res.get("action_id")
        
        if not action_id:
            self.logger.warning(
                "'%s' succeeded but returned no action_id (tool '%s'). Response: %s",
                record_action_start_mcp,
                tool_name_mcp,
                str(res)[:300]
            )
            return None

        self.logger.debug("Action started (%s) for tool %s", _fmt_id(action_id), tool_name_mcp)

        # ---------------------------------------------------------------- #
        # 5. Record planned dependencies – best-effort, never fatal.       #
        # ---------------------------------------------------------------- #
        if planned_dependencies:
            # validate list contents
            if not isinstance(planned_dependencies, list) or not all(isinstance(d, str) for d in planned_dependencies):
                self.logger.warning(
                    "planned_dependencies must be List[str]; got %s – skipping dependency recording.",
                    type(planned_dependencies),
                )
            else:
                unique_deps: List[str] = [d for d in dict.fromkeys(planned_dependencies) if d]
                if unique_deps:
                    try:
                        await self._record_action_dependencies_internal(action_id, unique_deps)
                    except Exception as dep_err:
                        self.logger.error(
                            "Failed to record dependencies for action %s: %s",
                            _fmt_id(action_id),
                            dep_err,
                            exc_info=True,
                        )

        return action_id

    async def _record_action_dependencies_internal(  # noqa: C901  (complexity OK – scoped)
        self,
        source_id: str,
        target_ids: List[str],
    ) -> None:
        """
        Persist "requires" dependencies (⟂ edges) between a *source* action and
        one or more *target* actions in UMS.
        """
        # ------------------------------------------------------------------ #
        # 1. Fast exit guards                                                #
        # ------------------------------------------------------------------ #
        if not source_id or not target_ids:
            return

        valid_targets: Set[str] = {tid for tid in target_ids if tid and tid != source_id}
        if not valid_targets:
            return

        # Allow higher-level shutdown to short-circuit early
        if getattr(self, "_shutdown_event", None) and self._shutdown_event.is_set():
            return

        # ------------------------------------------------------------------ #
        # 2. Resolve MCP tool + workflow context                             #
        # ------------------------------------------------------------------ #
        add_dep_mcp_tool: str = self._get_ums_tool_mcp_name(UMS_FUNC_ADD_ACTION_DEPENDENCY)
        if not self._find_tool_server(add_dep_mcp_tool):
            self.logger.error("Tool for '%s' unavailable.", UMS_FUNC_ADD_ACTION_DEPENDENCY)
            return

        workflow_id: str | None = (
            self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        )
        if not workflow_id:
            self.logger.warning("No active WF ID while recording deps for action %s.", _fmt_id(source_id))
            return

        self.logger.debug(
            "Recording %d deps for action %s: %s",
            len(valid_targets),
            _fmt_id(source_id),
            [_fmt_id(tid) for tid in valid_targets],
        )

        # ------------------------------------------------------------------ #
        # 3. Helper to call UMS tool (w/ single retry)                       #
        # ------------------------------------------------------------------ #
        async def _call_add_dependency(target_id: str) -> Tuple[str, Union[Dict[str, Any], Exception]]:
            """Single RPC wrapper – catches & returns all exceptions."""
            payload: Dict[str, Any] = {
                "workflow_id": workflow_id,          # **new** – required by UMS
                "source_action_id": source_id,
                "target_action_id": target_id,
                "dependency_type": "requires",
            }
            try:
                result = await self._execute_tool_call_internal(
                    add_dep_mcp_tool,
                    payload,
                    record_action=False,
                )
                # Lightweight retry once on transient UMS failure
                if isinstance(result, dict) and not result.get("success", False):
                    if result.get("status_code") in {429, 500, 503}:  # transient
                        self.logger.debug(
                            "Transient failure on dep %s -> %s, retrying once…",
                            _fmt_id(source_id),
                            _fmt_id(target_id),
                        )
                        result = await self._execute_tool_call_internal(
                            add_dep_mcp_tool,
                            payload,
                            record_action=False,
                        )
                return target_id, result
            except Exception as exc:  # noqa: BLE001  (we re-emit in caller)
                return target_id, exc

        # ------------------------------------------------------------------ #
        # 4. Bounded concurrency execution                                   #
        # ------------------------------------------------------------------ #
        max_parallel: int = getattr(self, "MAX_PARALLEL_DEP_CALLS", 10)
        semaphore = asyncio.Semaphore(max_parallel)

        async def _bounded(tid: str):
            async with semaphore:
                return await _call_add_dependency(tid)

        tasks = [_bounded(tid) for tid in valid_targets]

        try:
            results = await asyncio.gather(*tasks, return_exceptions=False)
        except asyncio.CancelledError:  # propagate cancellation cleanly
            for t in tasks:
                t.cancel()
            raise

        # ------------------------------------------------------------------ #
        # 5. Post-processing + logging                                       #
        # ------------------------------------------------------------------ #
        for target_id, res in results:
            if isinstance(res, Exception):
                self.logger.error(
                    "Error recording dep %s -> %s: %s",
                    _fmt_id(source_id),
                    _fmt_id(target_id),
                    res,
                    exc_info=False,
                )
            elif isinstance(res, dict) and not res.get("success", False):
                self.logger.warning(
                    "Failed recording dep %s -> %s: %s",
                    _fmt_id(source_id),
                    _fmt_id(target_id),
                    res.get("error") or res.get("error_message"),
                )

    async def _record_action_completion_internal(
        self,
        action_id: str,
        result: Dict[str, Any],
    ) -> None:
        """
        Persist the completion/failure status of an action in UMS.
        """
        if not isinstance(action_id, str) or not action_id.strip():
            self.logger.error("record_action_completion: invalid action_id %r", action_id)
            return

        completion_mcp_tool_name: str = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_ACTION_COMPLETION)
        if not self._find_tool_server(completion_mcp_tool_name):
            self.logger.error("Tool for '%s' unavailable.", UMS_FUNC_RECORD_ACTION_COMPLETION)
            return

        status: str = (
            ActionStatus.COMPLETED.value
            if isinstance(result, dict) and result.get("success", False)
            else ActionStatus.FAILED.value
        )

        current_wf_id: Optional[str] = (
            self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        )
        if not current_wf_id:
            self.logger.warning("No active WF ID for action completion %s.", _fmt_id(action_id))
            return

        payload: Dict[str, Any] = {
            "workflow_id": current_wf_id,
            "context_id": self.state.context_id,            # may be None – UMS ignores if null
            "action_id": action_id,
            "status": status,
            "tool_result": _safe_json_compatible(result),
        }

        # Attempt + optional single retry
        for attempt in range(_MAX_RETRIES + 1):
            try:
                completion_res = await asyncio.wait_for(
                    self._execute_tool_call_internal(
                        completion_mcp_tool_name,
                        payload,
                        record_action=False,
                    ),
                    timeout=_RECORD_TIMEOUT_SEC,
                )

                if completion_res.get("success"):
                    self.logger.debug(
                        "Action completion recorded • WF=%s • Action=%s • Status=%s",
                        _fmt_id(current_wf_id),
                        _fmt_id(action_id),
                        status,
                    )
                    return  # success → done

                self.logger.error(
                    "Failed to record action completion (attempt %d/%d) for %s: %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    _fmt_id(action_id),
                    completion_res.get("error"),
                )
                if attempt < _MAX_RETRIES:
                    await asyncio.sleep(0.5 * math.pow(2, attempt))  # simple back-off
                    continue
                return  # no more retries

            except asyncio.TimeoutError:
                self.logger.error(
                    "Timeout (%ss) while recording action completion (attempt %d/%d) for %s.",
                    _RECORD_TIMEOUT_SEC,
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    _fmt_id(action_id),
                )
                if attempt < _MAX_RETRIES:
                    continue
                return
            except Exception as exc:  # noqa: BLE001 – log & bail
                self.logger.error(
                    "Exception recording action completion (attempt %d/%d) for %s: %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    _fmt_id(action_id),
                    exc,
                    exc_info=True,
                )
                if attempt < _MAX_RETRIES:
                    continue
                return

    async def _run_auto_linking(
        self,
        memory_id: str,
        *,
        workflow_id: Optional[str],
        context_id: Optional[str],
    ) -> None:
        """
        Background task that tries to create semantic/keyword links from the *source*
        `memory_id` to other memories in **the same workflow**.  It is intentionally
        conservative: it never mutates agent state (other than via UMS-side link
        creation) and it bails out immediately on workflow changes, shutdown events,
        tool unavailability, or low-quality search results.

        ── Current responsibilities ────────────────────────────────────────────────
        1.  Guard-rails & throttling (snapshot WF, shutdown flag, random delay)
        2.  Retrieve the source memory (description/content)
        3.  Choose the best available search tool (hybrid > semantic)
        4.  Run a similarity search and rank candidates
        5.  Create up to `self.auto_linking_max_links` links using UMS
        6.  Log verbosely in DEBUG, warn gracefully on any failure
        """
        # --------------------------------------------------------------------- #
        # 0.  Early exits                                                       #
        # --------------------------------------------------------------------- #
        snapshot_wf_id = workflow_id                      # value captured at schedule-time
        active_wf_id    = self.state.workflow_id          # value at execution-time

        if (
            not memory_id
            or not snapshot_wf_id
            or snapshot_wf_id != active_wf_id
            or self._shutdown_event.is_set()
        ):
            # Nothing to do (workflow changed, missing ids, or agent shutting down)
            self.logger.debug(
                "Auto-linking skipped - mem=%s, snapWF=%s, activeWF=%s, shutdown=%s",
                _fmt_id(memory_id),
                _fmt_id(snapshot_wf_id),
                _fmt_id(active_wf_id),
                self._shutdown_event.is_set(),
            )
            return

        # --------------------------------------------------------------------- #
        # 1.  Gentle back-off so we don't hammer the store immediately           #
        # --------------------------------------------------------------------- #
        try:
            await asyncio.sleep(random.uniform(*AUTO_LINKING_DELAY_SECS))
            if self._shutdown_event.is_set():
                return

            # ----------------------------------------------------------------- #
            # 2.  Fetch the *source* memory (single call)                        #
            # ----------------------------------------------------------------- #
            get_mem_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_GET_MEMORY_BY_ID)
            src_resp = await self._execute_tool_call_internal(
                get_mem_mcp,
                {
                    "memory_id": memory_id,
                    "include_links": False,
                    "workflow_id": snapshot_wf_id,
                },
                record_action=False,
            )

            # Enhanced validation with better error messages
            if not src_resp.get("success"):
                self.logger.warning(
                    "Auto-linking: failed to fetch source mem %s - tool call unsuccessful: %s",
                    _fmt_id(memory_id),
                    src_resp.get("error_message", "Unknown error")
                )
                return
            
            # Check for workflow ID mismatch (common during state transitions)
            response_wf_id = src_resp.get("workflow_id")
            if response_wf_id != snapshot_wf_id:
                self.logger.debug(
                    "Auto-linking: skipping mem %s - workflow ID mismatch (expected %s, got %s)",
                    _fmt_id(memory_id),
                    _fmt_id(snapshot_wf_id),
                    _fmt_id(response_wf_id)
                )
                return
                
            # Validate response structure
            if not isinstance(src_resp, dict):
                self.logger.warning(
                    "Auto-linking: source mem %s response is not dict: %s",
                    _fmt_id(memory_id),
                    type(src_resp)
                )
                return

            source_mem = src_resp
            query_text: str = (
                source_mem.get("description") or
                (source_mem.get("content") or "")[:200]
            )

            if not query_text:
                self.logger.debug(
                    "Auto-linking skipped for %s - no description/content.",
                    _fmt_id(memory_id),
                )
                return

            # ----------------------------------------------------------------- #
            # 3.  Decide which search tool to use                                #
            # ----------------------------------------------------------------- #
            hybrid_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_HYBRID_SEARCH)
            semantic_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_SEARCH_SEMANTIC_MEMORIES)

            if self._find_tool_server(hybrid_mcp):
                search_mcp_name  = hybrid_mcp
                score_key        = "hybrid_score"
                search_args_extra = {"semantic_weight": 0.8, "keyword_weight": 0.2}
            elif self._find_tool_server(semantic_mcp):
                search_mcp_name  = semantic_mcp
                score_key        = "similarity"
                search_args_extra = {}
            else:
                self.logger.warning("Auto-linking aborted: no search tool available.")
                return

            # ----------------------------------------------------------------- #
            # 4.  Run search                                                     #
            # ----------------------------------------------------------------- #
            search_args: Dict[str, Any] = {
                "workflow_id": snapshot_wf_id,
                "query": query_text,
                "limit": self.auto_linking_max_links + 1,  # +1 so we can ignore self
                "threshold": self.auto_linking_threshold,
                "include_content": False,
                **search_args_extra,
            }

            search_resp = await self._execute_tool_call_internal(
                search_mcp_name,
                search_args,
                record_action=False,
            )

            if not search_resp.get("success"):
                self.logger.warning(
                    "Auto-linking search failed for %s: %s",
                    _fmt_id(memory_id),
                    search_resp.get("error"),
                )
                return

            candidate_mems = search_resp.get("memories", []) or []
            if not candidate_mems:
                self.logger.debug(
                    "Auto-linking found no candidates for %s (query='%s...').",
                    _fmt_id(memory_id),
                    query_text[:60],
                )
                return

            # ----------------------------------------------------------------- #
            # 5.  Prepare for link creation                                      #
            # ----------------------------------------------------------------- #
            create_link_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_LINK)
            if not self._find_tool_server(create_link_mcp):
                self.logger.warning("Auto-linking: link-creation tool unavailable.")
                return

            src_type = source_mem.get("memory_type")
            links_created = 0

            # ----------------------------------------------------------------- #
            # 6.  Iterate through search results                                 #
            # ----------------------------------------------------------------- #
            for mem in candidate_mems:
                if links_created >= self.auto_linking_max_links:
                    break
                if self._shutdown_event.is_set():
                    break

                tgt_id   = mem.get("memory_id")
                sim_val  = mem.get(score_key, 0.0)

                if not tgt_id or tgt_id == memory_id:
                    continue  # skip self or malformed entries

                # Simple heuristic for link typing
                link_type = LinkType.RELATED.value
                tgt_type  = mem.get("memory_type")
                if (
                    src_type == MemoryType.INSIGHT.value
                    and tgt_type == MemoryType.FACT.value
                ):
                    link_type = LinkType.SUPPORTS.value

                link_args = {
                    "source_memory_id": memory_id,
                    "target_memory_id": tgt_id,
                    "link_type": link_type,
                    "strength": round(float(sim_val), 3),
                    "description": f"Auto-link ({link_type})",
                    "workflow_id": snapshot_wf_id,
                }

                link_resp = await self._execute_tool_call_internal(
                    create_link_mcp,
                    link_args,
                    record_action=False,
                )

                if link_resp.get("success"):
                    links_created += 1
                    self.logger.debug(
                        "Auto-linked %s → %s  (%s, %.3f)",
                        _fmt_id(memory_id),
                        _fmt_id(tgt_id),
                        link_type,
                        sim_val,
                    )
                else:
                    self.logger.warning(
                        "Auto-linking failed %s→%s: %s",
                        _fmt_id(memory_id),
                        _fmt_id(tgt_id),
                        link_resp.get("error"),
                    )

                # Friendly pacing between link creations
                await asyncio.sleep(0.1)

        except Exception as exc:
            self.logger.warning(
                "Error during auto-linking for %s (WF %s): %s",
                _fmt_id(memory_id),
                _fmt_id(workflow_id),
                exc,
                exc_info=False,
            )

    async def _save_artifact_to_file_background(
        self,
        artifact_data: Dict[str, Any],
        *,
        workflow_id: str,
    ) -> None:
        """
        Background task to save an artifact to file in the task-specific directory.
        
        This runs asynchronously to avoid blocking the main agent loop.
        """
        if not workflow_id or self._shutdown_event.is_set():
            return
        
        try:
            # Save the artifact to file
            file_path = await self._save_artifact_to_file(artifact_data, workflow_id)
            
            if file_path:
                self.logger.info(f"📁 Background file save completed: {file_path}")
            else:
                self.logger.warning(f"📁 Background file save failed for artifact: {artifact_data.get('name', 'unknown')}")
                
        except Exception as exc:
            self.logger.warning(
                f"Error during background artifact file save (WF {_fmt_id(workflow_id)}): {exc}",
                exc_info=False,
            )

    def _get_adaptive_context_limits(self) -> Dict[str, int]:
        """Adjust context limits based on task complexity and history"""
        base_limits = {
            "recent_actions": CONTEXT_RECENT_ACTIONS_FETCH_LIMIT,
            "important_memories": CONTEXT_IMPORTANT_MEMORIES_FETCH_LIMIT,  
            "key_thoughts": CONTEXT_KEY_THOUGHTS_FETCH_LIMIT,
            "proactive_memories": CONTEXT_PROACTIVE_MEMORIES_FETCH_LIMIT,
            "procedural_memories": CONTEXT_PROCEDURAL_MEMORIES_FETCH_LIMIT,
            "link_traversal": CONTEXT_LINK_TRAVERSAL_FETCH_LIMIT,
        }
        
        # **CRITICAL FIX**: Establish minimum limits to prevent agent blindness
        minimum_limits = {
            "recent_actions": 15,  # Always see at least 15 recent actions
            "important_memories": 10,  # Always see at least 10 key memories
            "key_thoughts": 8,  # Always see at least 8 thoughts for continuity
            "proactive_memories": 5,  # Minimum proactive context
            "procedural_memories": 3,  # Minimum procedural context
            "link_traversal": 5,  # Minimum link context
        }
        
        # Simple task optimization - but maintain minimum visibility
        if len(self.state.current_plan) <= 2:
            multiplier = 0.7  # Reduced reduction from 0.5 to 0.7
            self.logger.debug("🏃 Slightly reducing context limits for simple task")
            result = {k: max(minimum_limits[k], int(v * multiplier)) for k, v in base_limits.items()}
            return result
        
        # Error recovery mode - increase context when struggling
        if self.state.consecutive_error_count > 1:
            multiplier = 1.5
            self.logger.debug(f"🔍 Increasing context limits due to {self.state.consecutive_error_count} errors")
            return {k: max(minimum_limits[k], min(30, int(v * multiplier))) for k, v in base_limits.items()}
        
        # Focus mode - but maintain minimum visibility to prevent blindness
        if getattr(self.state, 'artifact_focus_mode', False):
            multiplier = 0.6  # Reduced reduction from 0.3 to 0.6
            self.logger.debug("🎯 Reducing context limits during focus mode but maintaining minimum visibility")  
            return {k: max(minimum_limits[k], int(v * multiplier)) for k, v in base_limits.items()}
        
        # Long-running task - increase context for complex workflows
        if self.state.current_loop > 10:
            multiplier = 1.2
            self.logger.debug("📈 Slightly increasing context limits for long-running task")
            return {k: min(15, int(v * multiplier)) for k, v in base_limits.items()}
        
        return base_limits

    async def _get_completion_biased_suggestions(self, current_goal: str) -> List[str]:
        """Get suggestions biased toward goal completion rather than endless research"""
        suggestions = []
        
        if not current_goal or current_goal == "Overall UMS Workflow Goal or Initial Task":
            return suggestions
        
        try:
            goal_type = self._classify_goal_type(current_goal)
            completion_status = await self._analyze_goal_completion_semantically(current_goal, goal_type)
            
            # Force completion if ready
            if completion_status["ready_for_completion"]:
                suggestions.append("🎉 **GOAL COMPLETION READY**: All criteria met - signal completion now!")
                suggestions.append("✅ **COMPLETE NOW**: Respond with 'Goal Achieved: [brief summary]'")
                return suggestions
            
            # Force completion if exceeded limits
            if completion_status["force_completion"]:
                suggestions.append("🚨 **FORCE COMPLETION**: You've exceeded reasonable limits!")
                suggestions.append(f"🎯 **CREATE NOW**: Use `{UMS_FUNC_RECORD_ARTIFACT}` with available information")
                suggestions.append("⚠️ **STOP RESEARCHING**: Take action with what you have")
                return suggestions
            
            # Guide toward completion
            if not completion_status["has_deliverable"]:
                expected = completion_status["completion_criteria"]["expected_deliverable"]
                suggestions.append(f"🎯 **FOCUS ON DELIVERABLE**: Create {expected} as your primary objective")
                suggestions.append(f"📋 **USE**: `{UMS_FUNC_RECORD_ARTIFACT}` with is_output=True")
            
            if not completion_status["has_sufficient_facts"] and completion_status["within_search_limits"]:
                needed = completion_status["completion_criteria"]["min_facts_required"] - completion_status["fact_count"]
                suggestions.append(f"📚 **GATHER {needed} MORE KEY FACTS**: Then immediately create deliverable")
                suggestions.append("⚠️ **NO ENDLESS RESEARCH**: Limit information gathering")
            
        except Exception as e:
            self.logger.debug(f"Error in completion bias suggestions: {e}")
        
        return suggestions
    
    async def _suggest_next_tools(self, current_goal: str, last_action: str, plan_length: int) -> str:
        """Suggest the most logical next tools using intelligent semantic analysis and LLM-based understanding"""
        suggestions = []
        
        # **COMPLETION-FIRST BIAS**: Check if goal is ready for completion
        completion_guidance = await self._get_completion_biased_suggestions(current_goal)
        if completion_guidance:
            suggestions.extend(completion_guidance)
            
            # If we have completion guidance, prioritize it over other suggestions
            if any("COMPLETION READY" in s or "FORCE COMPLETION" in s for s in completion_guidance):
                result = "**🎯 GOAL COMPLETION PRIORITY:**\n" + "\n".join(f"• {s}" for s in suggestions)
                result += "\n\n💡 **Remember:** Goal completion takes priority over additional research!"
                return result
        
        # Use intelligent semantic analysis instead of hardcoded patterns
        task_analysis = self._analyze_task_semantically(current_goal, last_action)
        progress_analysis = self._analyze_progress_semantically(last_action, current_goal)
        
        # REMOVED: Massive hardcoded task_characteristics and progress_state dictionaries
        # REPLACED: With intelligent semantic analysis functions
        
        # Generate intelligent strategic suggestions based on semantic analysis
        suggestions.extend(self._generate_strategic_suggestions_semantically(task_analysis, progress_analysis))
        
        # Generate intelligent state context and format result
        state_context = self._generate_state_context_semantically()
        loop_warnings = self._generate_loop_prevention_warnings(current_goal)
        
        result = "**🎯 STRATEGIC NEXT STEPS:**\n" + "\n".join(f"• {s}" for s in suggestions) if suggestions else ""
        
        if state_context:
            result += f"\n\n**📊 Current State:** {state_context}"
        
        result += "\n\n**💡 Remember:** Review your COMPLETE MCP toolkit - you have access to web search, document processors, database tools, API connectors, and more beyond just UMS tools!"
        
        if loop_warnings:
            result += loop_warnings
        
        # Schedule background LLM improvement
        if hasattr(self, 'mcp_client') and self.mcp_client:
            self._start_background_task(
                self._improve_tool_suggestions_with_llm,
                current_goal, last_action, suggestions
            )
        
        return result
    
    async def _analyze_task_semantically(self, goal_description: str, last_action: str) -> Dict[str, Any]:
        """Semantically analyze task characteristics using intelligent heuristics and LLM-based semantic understanding"""
        
        # Use intelligent semantic scoring instead of simple keyword matching
        analysis = {
            "primary_type": await self._classify_goal_type(goal_description),
            "complexity_score": await self._calculate_complexity_score_semantically(goal_description),
            "research_intensity": await self._assess_research_intensity_semantically(goal_description),
            "creation_focus": await self._assess_creation_focus_semantically(goal_description),
            "deliverable_type": await self._predict_deliverable_type_semantically(goal_description),
            "has_multi_part_indicators": await self._detect_multi_part_task_semantically(goal_description),
            "external_integration_needed": await self._detect_integration_needs_semantically(goal_description)
        }
        
        return analysis
        
    async def _analyze_progress_semantically(self, last_action: str, goal_description: str) -> Dict[str, Any]:
        """Semantically analyze current progress state using intelligent understanding"""
        
        # Intelligent progress phase detection
        analysis = {
            "current_phase": await self._detect_current_phase_semantically(last_action),
            "recent_activities": await self._categorize_recent_activities_semantically(last_action),
            "search_fatigue": self.state.search_attempts_count >= 2,
            "artifact_creation_gap": self.state.turns_since_artifact_creation >= 3,
            "tool_repetition_concern": self.state.consecutive_same_tool_count >= 2,
            "error_recovery_mode": self.state.consecutive_error_count >= 1
        }
        
        return analysis
    
    def _generate_strategic_suggestions_semantically(self, task_analysis: Dict, progress_analysis: Dict) -> List[str]:
        """Generate intelligent strategic suggestions based on semantic analysis"""
        
        suggestions = []
        
        # Strategic routing based on semantic understanding
        if progress_analysis["current_phase"] == "initialization":
            suggestions.extend(self._get_initialization_suggestions_semantically(task_analysis))
        elif progress_analysis["current_phase"] == "research":
            suggestions.extend(self._get_research_phase_suggestions_semantically(task_analysis, progress_analysis))
        elif progress_analysis["current_phase"] == "creation":
            suggestions.extend(self._get_creation_suggestions_semantically(task_analysis, progress_analysis))
        elif progress_analysis["current_phase"] == "error_recovery":
            suggestions.extend(self._get_error_recovery_suggestions_semantically(task_analysis, progress_analysis))
        else:
            suggestions.extend(self._get_general_suggestions_semantically(task_analysis, progress_analysis))
        
        # Add cross-cutting strategic guidance
        if task_analysis["complexity_score"] > 7:
            suggestions.append(f"📚 **STRATEGIC GUIDANCE**: Use `{UMS_FUNC_GET_MULTI_TOOL_GUIDANCE}` for complex task orchestration")
        
        if progress_analysis["search_fatigue"]:
            suggestions.append("🚨 **SEARCH LIMIT REACHED**: Move to concrete action with available information")
        
        if progress_analysis["artifact_creation_gap"]:
            suggestions.append(f"🎯 **DELIVERABLE FOCUS**: Use `{UMS_FUNC_RECORD_ARTIFACT}` to create tangible output")
        
        return suggestions
    
    def _generate_state_context_semantically(self) -> str:
        """Generate intelligent state context summary"""
        
        context_parts = []
        
        if self.state.search_attempts_count > 0:
            context_parts.append(f"Research: {self.state.search_attempts_count}/2 searches")
        
        if self.state.consecutive_same_tool_count > 1:
            tool_name = self.state.last_tool_executed or "unknown"
            context_parts.append(f"Tool repetition: {tool_name} ({self.state.consecutive_same_tool_count}x)")
        
        if self.state.turns_since_artifact_creation > 0:
            context_parts.append(f"No artifacts: {self.state.turns_since_artifact_creation} turns")
        
        if self.state.artifact_focus_mode:
            context_parts.append("Mode: Artifact Creation")
        elif self.state.research_focus_mode:
            context_parts.append("Mode: Research Focus")
        
        return ", ".join(context_parts)
    

    async def _calculate_complexity_score_semantically(self, goal_description: str) -> int:
        """Calculate semantic complexity score using LLM analysis with caching"""
        
        # Cache to avoid repeated LLM calls
        goal_hash = hash(goal_description.strip())
        
        if goal_hash in self._caches['complexity_score']:
            return self._caches['complexity_score'][goal_hash]

        complexity_prompt = f"""
    Rate the complexity of this goal/task on a scale of 0-100.

    Goal: "{goal_description}"

    Complexity factors:
    - Number of distinct steps required
    - Technical sophistication needed
    - Coordination between different types of work
    - Domain expertise required
    - Time and resource investment
    - Dependencies and sequencing challenges

    Scoring guide:
    - 0-20: Simple, single-step task requiring minimal expertise
    - 21-40: Multi-step but straightforward with clear methodology
    - 41-60: Moderately complex, requires planning and coordination
    - 61-80: Complex, requires significant expertise and coordination
    - 81-100: Highly complex, multi-domain, extensive planning

    Respond with just the numerical complexity score (0-100).
    """

        try:
            simple_score_schema = {
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
            
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": complexity_prompt}],
                response_schema=simple_score_schema,
                schema_name="complexity_scoring",
                use_cheap_model=True,
                max_tokens=5,
                temperature=0.1
            )
            
            if response:
                complexity_score = response.get("score", 50)
                if 0 <= complexity_score <= 100:
                    self._caches['complexity_score'][goal_hash] = complexity_score
                    return complexity_score
                    
        except Exception as e:
            self.logger.debug(f"LLM complexity scoring failed: {e}")
            
        # Fallback
        fallback_score = 50
        self._caches['complexity_score'][goal_hash] = fallback_score
        return fallback_score
        
    
    async def _assess_research_intensity_semantically(self, goal_description: str) -> int:
        """Assess research intensity using LLM analysis with caching"""
        
        # Cache to avoid repeated LLM calls
        goal_hash = hash(f"research_intensity:{goal_description.strip()}")
        
        if goal_hash in self._caches['research_intensity']:
            return self._caches['research_intensity'][goal_hash]

        research_prompt = f"""
    Analyze this task and rate how research-intensive it is on a scale of 0-100:

    0-20: Minimal research needed (task can be completed with existing knowledge)
    21-40: Light research (1-2 quick searches for basic information)
    41-60: Moderate research (systematic information gathering from multiple sources)
    61-80: Heavy research (extensive investigation, analysis of multiple sources)
    81-100: Very intensive research (comprehensive study requiring deep investigation)

    Task: "{goal_description}"

    Consider:
    - How much external information is needed?
    - How many sources would need to be consulted?
    - How deep does the investigation need to go?
    - Complexity and breadth of information gathering required

    Respond with just the numerical intensity score (0-100).
    """

        try:
            simple_score_schema = {
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
            
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": research_prompt}],
                response_schema=simple_score_schema,
                schema_name="research_intensity_scoring",
                use_cheap_model=True,
                max_tokens=5,
                temperature=0.1
            )
            
            if response:
                intensity_score = response.get("score", 50)
                if 0 <= intensity_score <= 100:
                    self._caches['research_intensity'][goal_hash] = intensity_score
                    return intensity_score
                    
        except Exception as e:
            self.logger.debug(f"LLM research intensity assessment failed: {e}")
            
        # Fallback to moderate intensity
        fallback_score = 50
        self._caches['research_intensity'][goal_hash] = fallback_score
        return fallback_score
        
    async def _assess_creation_focus_semantically(self, goal_description: str) -> int:
        """Assess creation focus using LLM analysis with caching"""
        
        # Cache to avoid repeated LLM calls
        goal_hash = hash(f"creation_focus:{goal_description.strip()}")
        
        if goal_hash in self._caches['creation_focus']:
            return self._caches['creation_focus'][goal_hash]

        creation_prompt = f"""
    Analyze this task and rate how creation-focused it is on a scale of 0-100:

    0-20: Minimal creation (mostly reading, understanding, analyzing existing content)
    21-40: Light creation (minor modifications, simple outputs)
    41-60: Moderate creation (substantive content creation, documents, reports)
    61-80: Heavy creation (complex deliverables, interactive content, detailed outputs)
    81-100: Primary creation focus (main goal is to build/create something significant)

    Task: "{goal_description}"

    Consider:
    - Is the primary goal to create something new?
    - How substantial is the deliverable?
    - How much original content needs to be produced?
    - Complexity and scope of creation work required

    Respond with just the numerical creation focus score (0-100).
    """

        try:
            simple_score_schema = {
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
            
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": creation_prompt}],
                response_schema=simple_score_schema,
                schema_name="creation_focus_scoring",
                use_cheap_model=True,
                max_tokens=5,
                temperature=0.1
            )
            
            if response:
                focus_score = response.get("score", 50)
                if 0 <= focus_score <= 100:
                    self._caches['creation_focus'][goal_hash] = focus_score
                    return focus_score
                    
        except Exception as e:
            self.logger.debug(f"LLM creation focus assessment failed: {e}")
            
        # Fallback to moderate creation focus
        fallback_score = 50
        self._caches['creation_focus'][goal_hash] = fallback_score
        return fallback_score

    async def _predict_deliverable_type_semantically(self, goal_description: str) -> str:
        """Predict deliverable type using LLM analysis with caching"""
        
        # Cache to avoid repeated LLM calls
        goal_hash = hash(f"deliverable_type:{goal_description.strip()}")
        
        if goal_hash in self._caches['deliverable_type']:
            return self._caches['deliverable_type'][goal_hash]

        deliverable_prompt = f"""
    Analyze this task and rate the likelihood (0-100) for each possible deliverable type.

    Task: "{goal_description}"

    Rate each deliverable type from 0-100:

    report: How likely is this to produce analytical reports, research summaries, or findings documents?
    html_file: How likely is this to produce web pages, interactive content, or HTML documents?
    interactive_quiz: How likely is this to produce quizzes, tests, assessments, or Q&A content?
    code_file: How likely is this to produce programming code, scripts, algorithms, or functions?
    document: How likely is this to produce general text documents, papers, essays, or articles?
    general_artifact: How likely is this to produce other types of deliverables not fitting above categories?

    Consider:
    - Primary output mentioned or implied
    - Format that would best serve the goal
    - Type of content being created

    Respond in this exact format:
    report: [0-100]
    html_file: [0-100]
    interactive_quiz: [0-100]
    code_file: [0-100]
    document: [0-100]
    general_artifact: [0-100]
    """

        try:
            deliverable_type_schema = {
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
            
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": deliverable_prompt}],
                response_schema=deliverable_type_schema,
                schema_name="deliverable_type_prediction",
                use_cheap_model=True,
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                # Find the highest confidence type
                scores = {k: v for k, v in response.items() if isinstance(v, int) and 0 <= v <= 100}
                
                if scores:
                    best_type = max(scores, key=scores.get)
                    best_score = scores[best_type]
                    
                    # Only use if confidence is reasonable (50+)
                    if best_score >= 50:
                        self._caches['deliverable_type'][goal_hash] = best_type
                        return best_type
                        
        except Exception as e:
            self.logger.debug(f"LLM deliverable type prediction failed: {e}")
            
        # Fall back to general_artifact if no high confidence
        fallback_type = "general_artifact"
        self._caches['deliverable_type'][goal_hash] = fallback_type
        return fallback_type

    async def _detect_multi_part_task_semantically(self, goal_description: str) -> bool:
        """Detect if task has multiple distinct parts using LLM-based semantic understanding"""
        
        # Cache to avoid repeated LLM calls
        goal_hash = hash(f"multi_part:{goal_description[:200]}")
        
        if goal_hash in self._caches['multi_part']:
            return self._caches['multi_part'][goal_hash]
        
        # Use LLM for semantic analysis
        is_multi_part = await self._analyze_multi_part_with_llm(goal_description)
        
        if is_multi_part is not None:
            self._caches['multi_part'][goal_hash] = is_multi_part
            return is_multi_part
        
        # No LLM result = assume single-part task
        self._caches['multi_part'][goal_hash] = False
        return False
    
    async def _analyze_multi_part_with_llm(self, goal_description: str) -> Optional[bool]:
        """Use LLM to analyze if task has multiple distinct parts"""
        
        multi_part_prompt = f"""
Analyze this task and determine if it has multiple distinct parts or sub-tasks.

Task: "{goal_description}"

Rate how strongly this task has multiple distinct parts on a 0-100 scale:

- 0-20: Single, unified task with one main objective
- 21-40: Primarily single task but may have minor sub-components
- 41-60: Moderate complexity, has some distinct parts
- 61-80: Multiple distinct parts or sub-tasks
- 81-100: Clearly has multiple distinct parts that need to be done separately

Consider:
1. Are there multiple distinct objectives?
2. Do different parts require different approaches/tools?
3. Could parts be done independently?
4. Are there clear sequences or dependencies?

Respond with just the numerical score (0-100).
"""

        try:
            simple_score_schema = {
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
            
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": multi_part_prompt}],
                response_schema=simple_score_schema,
                schema_name="multi_part_analysis",
                use_cheap_model=True,  # Use cheap/fast model
                max_tokens=5,
                temperature=0.1
            )
            
            if response:
                score = response.get("score", 0)
                if 0 <= score <= 100:
                    # Use threshold of 60+ for considering task multi-part
                    MULTI_PART_THRESHOLD = 60
                    is_multi_part = score >= MULTI_PART_THRESHOLD
                    self.logger.debug(f"🧠 LLM multi-part analysis: score={score}, is_multi_part={is_multi_part}")
                    return is_multi_part
                    
        except Exception as e:
            self.logger.debug(f"LLM multi-part analysis failed: {e}")
        
        return None
    

    
    async def _detect_integration_needs_semantically(self, goal_description: str) -> bool:
        """Detect if task needs external system integration using LLM-based semantic understanding"""
        
        # Cache to avoid repeated LLM calls
        goal_hash = hash(f"integration:{goal_description[:200]}")
        
        if goal_hash in self._caches['integration']:
            return self._caches['integration'][goal_hash]
        
        # Use LLM for semantic analysis
        needs_integration = await self._analyze_integration_needs_with_llm(goal_description)
        
        if needs_integration is not None:
            self._caches['integration'][goal_hash] = needs_integration
            return needs_integration
        
        # No LLM result = assume no integration needed
        self._caches['integration'][goal_hash] = False
        return False
    
    async def _analyze_integration_needs_with_llm(self, goal_description: str) -> Optional[bool]:
        """Use LLM to analyze if task needs external system integration"""
        
        integration_prompt = f"""
Analyze this task and determine if it requires integration with external systems, APIs, or services.

Task: "{goal_description}"

Rate how strongly this task needs external system integration on a 0-100 scale:

- 0-20: Self-contained task, no external integration needed
- 21-40: Minimal external needs, mostly internal processing
- 41-60: Some external data/services might be helpful
- 61-80: Likely needs external APIs, databases, or services
- 81-100: Definitely requires external system integration

Consider:
1. Does it mention specific external services/platforms?
2. Does it need real-time data from external sources?
3. Does it require connecting different systems?
4. Does it need external APIs or databases?
5. Does it involve cross-platform operations?

Respond with just the numerical score (0-100).
"""

        try:
            simple_score_schema = {
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
            
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": integration_prompt}],
                response_schema=simple_score_schema,
                schema_name="integration_needs_analysis",
                use_cheap_model=True,  # Use cheap/fast model
                max_tokens=5,
                temperature=0.1
            )
            
            if response:
                score = response.get("score", 0)
                if 0 <= score <= 100:
                    # Use threshold of 60+ for considering integration needed
                    INTEGRATION_THRESHOLD = 60
                    needs_integration = score >= INTEGRATION_THRESHOLD
                    self.logger.debug(f"🧠 LLM integration analysis: score={score}, needs_integration={needs_integration}")
                    return needs_integration
                    
        except Exception as e:
            self.logger.debug(f"LLM integration analysis failed: {e}")
        
        return None
    
    async def _detect_current_phase_semantically(self, last_action: str) -> str:
        """Detect current phase of work based on semantic analysis"""
        
        action_lower = last_action.lower()
        
        # Quick checks for obvious cases
        if not last_action or "initialized" in action_lower or "started" in action_lower:
            return "initialization"
        
        if "error" in action_lower or "failed" in action_lower:
            return "error_recovery"
        
        # Cache to avoid repeated LLM calls
        action_hash = hash(f"action_phase:{last_action[:200]}")
        
        if action_hash in self._caches['action_phase']:
            return self._caches['action_phase'][action_hash]

        phase_prompt = f"""
    Analyze this action description and determine which work phase it represents.

    Action: "{last_action}"

    Rate how strongly this action fits each phase on 0-100 scale:

    research: Information gathering, searching, investigating, browsing, studying
    creation: Creating, building, generating content, recording artifacts, writing
    processing: Converting, extracting, transforming, processing data/documents
    analysis: Analyzing, thinking, reasoning, evaluating, understanding
    general_work: Other productive work activities

    Consider:
    - Primary purpose of the action
    - Type of work being performed
    - Expected outcome or result

    Respond with just the phase name that scored highest.
    """

        try:
            phase_classification_schema = {
                "type": "object",
                "properties": {
                    "phase": {
                        "type": "string",
                        "enum": ["research", "creation", "processing", "analysis", "general_work"],
                        "description": "Current phase of work based on action analysis"
                    }
                },
                "required": ["phase"],
                "additionalProperties": False
            }
            
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": phase_prompt}],
                response_schema=phase_classification_schema,
                schema_name="action_phase_analysis",
                use_cheap_model=True,
                max_tokens=10,
                temperature=0.1
            )
            
            if response:
                phase = response.get("phase", "general_work")
                self._caches['action_phase'][action_hash] = phase
                return phase
                    
        except Exception as e:
            self.logger.debug(f"LLM action phase analysis failed: {e}")
        
        # Fallback
        fallback_phase = "general_work"
        self._caches['action_phase'][action_hash] = fallback_phase
        return fallback_phase
    
    async def _categorize_recent_activities_semantically(self, last_action: str) -> List[str]:
        """Categorize recent activities using LLM-based semantic understanding"""
        
        # Use LLM for multi-category activity analysis
        categories = await self._analyze_activity_categories_with_llm(last_action)
        
        return categories if categories else ["general_work"]
    
    async def _analyze_activity_categories_with_llm(self, action_description: str) -> Optional[List[str]]:
        """Use LLM to categorize action into multiple activity types"""
        
        categorization_prompt = f"""
Analyze this action and rate how strongly it represents each activity type on 0-100 scale.

Action: "{action_description}"

Rate each category (an action can fit multiple categories):

information_gathering: Searching, researching, browsing, finding information
knowledge_storage: Storing, saving, recording, organizing information
content_creation: Creating, building, generating, writing new content
analysis_thinking: Analyzing, reasoning, evaluating, understanding
data_processing: Converting, extracting, transforming, processing data
collaboration: Communicating, sharing, coordinating with others

For each category scoring 40+ points, include it in the response.

Respond with only the category names that score 40+ (e.g., "information_gathering,content_creation").
"""

        categorization_schema = {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["information_gathering", "knowledge_storage", "content_creation", 
                                "analysis_thinking", "data_processing", "collaboration"]
                    },
                    "description": "List of activity categories that score 40+ points"
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief explanation of the categorization"
                }
            },
            "required": ["categories"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": categorization_prompt}],
                response_schema=categorization_schema,
                schema_name="activity_categorization",
                use_cheap_model=True,
                max_tokens=100,
                temperature=0.1
            )
            
            if response:
                categories = response.get("categories", [])
                if categories:
                    return categories
                    
        except Exception as e:
            self.logger.debug(f"LLM activity categorization failed: {e}")
        
        return None
    

    
    # PHASE-SPECIFIC SUGGESTION FUNCTIONS (replacing hardcoded logic)
    
    def _get_initialization_suggestions_semantically(self, task_analysis: Dict) -> List[str]:
        """Get suggestions for task initialization phase"""
        
        suggestions = []
        
        if task_analysis["complexity_score"] > 70:
            suggestions.append(f"📚 **STRATEGIC START**: Use `{UMS_FUNC_GET_MULTI_TOOL_GUIDANCE}` for complex task orchestration")
            suggestions.append("📋 **TOOLKIT SURVEY**: Review ALL available MCP tools systematically")
        
        if task_analysis["deliverable_type"] == "html_file":
            suggestions.append("🌐 **WEB WORKFLOW**: Look for HTML generators, web creation tools, interactive builders")
        elif task_analysis["deliverable_type"] == "report":
            suggestions.append("📄 **RESEARCH WORKFLOW**: Plan information gathering → analysis → documentation chain")
        elif task_analysis["deliverable_type"] == "code_file":
            suggestions.append("💻 **DEVELOPMENT WORKFLOW**: Survey code generation and development tools")
        else:
            suggestions.append("🛠️ **TOOL DISCOVERY**: Survey complete MCP toolkit for relevant capabilities")
        
        if task_analysis["research_intensity"] > 60:
            suggestions.append("🔍 **RESEARCH STRATEGY**: Plan systematic information gathering approach")
            suggestions.append("⚠️ **EFFICIENCY NOTE**: Limit research iterations, focus on deliverable creation")
        
        return suggestions
    
    def _get_research_phase_suggestions_semantically(self, task_analysis: Dict, progress_analysis: Dict) -> List[str]:
        """Get suggestions for research phase"""
        
        suggestions = []
        
        if progress_analysis["search_fatigue"]:
            suggestions.append("🚨 **RESEARCH COMPLETE**: Stop searching, move to synthesis and creation")
            suggestions.append(f"🎯 **CREATE NOW**: Use `{UMS_FUNC_RECORD_ARTIFACT}` with gathered information")
        else:
            suggestions.append(f"📝 **ORGANIZE FINDINGS**: Use `{UMS_FUNC_STORE_MEMORY}` to structure research")
            suggestions.append("🔄 **PROCESS CHAIN**: Research → Store → Synthesize → Create deliverable")
        
        if task_analysis["external_integration_needed"]:
            suggestions.append("🔗 **INTEGRATION TOOLS**: Check for API connectors and external service tools")
        
        return suggestions
    
    def _get_creation_suggestions_semantically(self, task_analysis: Dict, progress_analysis: Dict) -> List[str]:
        """Get suggestions for creation phase"""
        
        suggestions = []
        
        if task_analysis["deliverable_type"] == "interactive_quiz":
            suggestions.append("🎯 **INTERACTIVE CONTENT**: Use HTML/JS tools to create engaging quiz")
            suggestions.append("📋 **QUIZ STRUCTURE**: Plan questions, answers, interactivity, and feedback")
        elif task_analysis["deliverable_type"] == "report":
            suggestions.append("📊 **COMPREHENSIVE REPORT**: Structure findings with analysis and conclusions")
            suggestions.append("🔗 **LINK SOURCES**: Connect content to supporting research and evidence")
        else:
            suggestions.append(f"🎯 **CREATE DELIVERABLE**: Use `{UMS_FUNC_RECORD_ARTIFACT}` for final output")
        
        suggestions.append(f"🔗 **CONNECT CONTENT**: Use `{UMS_FUNC_CREATE_LINK}` to link related information")
        
        return suggestions
    
    def _get_error_recovery_suggestions_semantically(self, task_analysis: Dict, progress_analysis: Dict) -> List[str]:
        """Get suggestions for error recovery"""
        
        suggestions = []
        
        suggestions.append("🔧 **DIAGNOSE ISSUE**: Analyze error message and context systematically")
        suggestions.append(f"🔧 **FILE ACCESS**: Use `{UMS_FUNC_DIAGNOSE_FILE_ACCESS}` for file/path issues")
        suggestions.append(f"🤔 **ANALYZE PROBLEM**: Use `{UMS_FUNC_RECORD_THOUGHT}` to understand failure")
        suggestions.append("🛠️ **ALTERNATIVE APPROACH**: Try different tools or methods")
        
        if task_analysis["complexity_score"] > 60:
            suggestions.append(f"📚 **GET GUIDANCE**: Use `{UMS_FUNC_GET_MULTI_TOOL_GUIDANCE}` for strategy")
        
        return suggestions
    
    def _get_general_suggestions_semantically(self, task_analysis: Dict, progress_analysis: Dict) -> List[str]:
        """Get general suggestions when phase is unclear"""
        
        suggestions = []
        
        if progress_analysis["artifact_creation_gap"]:
            suggestions.append(f"🎯 **IMMEDIATE ACTION**: Use `{UMS_FUNC_RECORD_ARTIFACT}` to create progress")
        
        if task_analysis["creation_focus"] > 60:
            suggestions.append("✅ **CHECK RESOURCES**: Query existing memories before creating new content")
            suggestions.append("🎯 **FOCUS ON DELIVERABLE**: Prioritize tangible output creation")
        
        suggestions.append("🛠️ **STRATEGIC CHOICE**: Select most impactful tool for current objective")
        
        return suggestions
    
    async def _improve_tool_suggestions_with_llm(self, goal_desc: str, last_action: str, current_suggestions: List[str]):
        """Background LLM improvement of tool suggestions"""
        
        suggestion_prompt = f"""
Analyze this agent situation and suggest the most strategic next tools/actions:

GOAL: {goal_desc}
LAST ACTION: {last_action}
CURRENT SUGGESTIONS: {current_suggestions[:3]}

Agent has access to:
- UMS tools (memory, artifacts, goals, reasoning)  
- Web search tools
- Document processing tools
- Database/data tools
- API integration tools
- File system tools

Focus on:
1. What's the most logical next step?
2. Which specific tools would be most effective?
3. How to avoid infinite loops and create tangible progress?

Provide 2-3 specific, actionable suggestions focusing on tool names and approach.
"""
        suggestion_schema = {
            "type": "object",
            "properties": {
                "suggestions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 3,
                    "description": "2-3 specific, actionable tool suggestions"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of why these tools are recommended"
                }
            },
            "required": ["suggestions"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": suggestion_prompt}],
                response_schema=suggestion_schema,
                schema_name="tool_suggestion_improvement",
                use_cheap_model=True,
                max_tokens=250,
                temperature=0.2
            )
            
            if response:
                suggestions = response.get("suggestions", [])
                if suggestions:
                    self.logger.debug(f"🧠 LLM suggested improved tools: {suggestions}")
                    # Could be used to improve future suggestions via caching/learning
                
        except Exception as e:
            self.logger.debug(f"Background LLM tool suggestion improvement failed: {e}")
        
    def _suggest_efficient_tool_patterns(self, current_goal: str, last_action: str) -> str:
        """Suggest efficient multi-tool patterns leveraging the full MCP toolkit"""
        suggestions = []
        
        goal_lower = current_goal.lower()
        action_lower = last_action.lower()
        
        # Determine workflow phase generically
        if ("initialized" in action_lower or "started" in action_lower or 
            not ("search" in action_lower or "store" in action_lower or "record" in action_lower)):
            # Beginning of workflow
            suggestions.append("**🚀 WORKFLOW START - STRATEGIC TOOL PATTERNS:**")
            suggestions.append("• **Survey Your Toolkit**: Review ALL available MCP tools, not just UMS tools")
            suggestions.append("• **Information Strategy**: Consider web search, document search, database tools")
            suggestions.append("• **Processing Strategy**: Look for document converters, data processors, analyzers")
            suggestions.append("• **Integration Strategy**: Check for API tools, external service connectors")
            
        elif ("search" in action_lower or "web" in action_lower or "browser" in action_lower or "query" in action_lower):
            # In information gathering phase
            suggestions.append("**🔍 INFORMATION GATHERING - NEXT PATTERNS:**")
            suggestions.append("• **Process Results**: Use document converters if you got PDFs/docs")
            suggestions.append("• **Extract Data**: Use text extraction tools for complex documents")
            suggestions.append("• **Store Findings**: Save processed information in UMS for organization")
            suggestions.append("• **Analyze Patterns**: Look for data analysis tools if you have structured data")
            
        elif "process" in action_lower or "convert" in action_lower:
            # Document/data processing phase (use simple contains check, not primitive keyword lists)
            suggestions.append("**🔄 PROCESSING PHASE - NEXT PATTERNS:**")
            suggestions.append("• **Chain Processing**: Combine multiple processors for complex transformations")
            suggestions.append("• **Store Processed Data**: Save transformed content for later use")
            suggestions.append("• **Quality Check**: Validate processed results before proceeding")
            suggestions.append("• **Analyze Results**: Use data analysis tools on processed information")
            
        elif "store" in action_lower or "memory" in action_lower:
            # Knowledge storage phase
            suggestions.append("**📚 KNOWLEDGE STORED - STRATEGIC NEXT STEPS:**")
            suggestions.append("• **Cross-Reference**: Use database tools to check for related information")
            suggestions.append("• **Validate Data**: Use external tools to verify key facts")
            suggestions.append("• **Synthesize**: Combine stored knowledge with external tools for analysis")
            suggestions.append("• **Create Deliverables**: Use your organized knowledge to build outputs")
            
        elif "record" in action_lower or "artifact" in action_lower:
            # Action completion phase
            if ("and" in goal_lower or "multiple" in goal_lower or "both" in goal_lower):
                # Potential multi-part task detected
                suggestions.append("**📋 MULTI-PART TASK - ORCHESTRATION PATTERNS:**")
                suggestions.append("• **Task Delegation**: Consider using delegation tools for subtasks")
                suggestions.append("• **Parallel Processing**: Use multiple external tools simultaneously")
                suggestions.append("• **Integration Tools**: Look for tools that can combine multiple results")
                suggestions.append("• **Quality Assurance**: Use validation tools to check completeness")
            else:
                suggestions.append("**✅ TASK COMPLETION - FINALIZATION PATTERNS:**")
                suggestions.append("• **External Validation**: Use external tools to verify your results")
                suggestions.append("• **Format Optimization**: Use document tools for final formatting")
                suggestions.append("• **Distribution**: Consider tools for sharing or publishing results")
                suggestions.append("• **Documentation**: Store process/results for future reference")
        
        # Add comprehensive efficiency tips
        if suggestions:
            suggestions.append("")
            suggestions.append("**⚡ STRATEGIC EFFICIENCY PRINCIPLES:**")
            suggestions.append("• **Tool Discovery**: Regularly review your complete MCP toolkit")
            suggestions.append("• **Workflow Chaining**: Connect external tools → UMS → external tools")
            suggestions.append("• **Automation**: Use delegation tools for routine or specialized tasks")
            suggestions.append("• **Parallel Processing**: Execute compatible external tools simultaneously")
            suggestions.append("• **Result Caching**: Store intermediate results to avoid reprocessing")
            suggestions.append("• **Integration Thinking**: Look for tools that bridge different systems/formats")
        
        if suggestions:
            return "\n".join(suggestions) + "\n"
        
        return ""

    def _generate_loop_prevention_warnings(self, current_goal_desc: str) -> str:
        """Generate critical warnings to prevent infinite loops based on current agent state"""
        warnings = []
        
        # Critical search loop warnings
        if self.state.search_attempts_count >= 2:
            warnings.append("🚨 **CRITICAL LOOP ALERT**: You have already searched 2+ times!")
            warnings.append("🚨 **STOP SEARCHING**: Take concrete action based on information gathered")
            warnings.append("🚨 **NO MORE RESEARCH**: Act with information you have")
        elif self.state.search_attempts_count >= 1:
            warnings.append("⚠️ **SEARCH WARNING**: 1 search completed - next action MUST be concrete action")
            warnings.append("⚠️ **NO MORE RESEARCH**: Act on gathered information immediately")
        
        # Consecutive tool warnings
        if self.state.consecutive_same_tool_count >= 2:
            last_tool = self.state.last_tool_executed or "unknown"
            warnings.append(f"🚨 **INFINITE LOOP DETECTED**: Tool '{last_tool}' used {self.state.consecutive_same_tool_count} times consecutively")
            warnings.append("🚨 **BREAK THE PATTERN**: Use a different approach or tool")
        
        # Turns without concrete action warnings
        if self.state.turns_since_artifact_creation >= 5:
            warnings.append(f"🚨 **{self.state.turns_since_artifact_creation} TURNS WITHOUT CONCRETE ACTION**: This is wasteful!")
            warnings.append("🚨 **IMMEDIATE ACTION**: Take concrete action toward the goal NOW")
        elif self.state.turns_since_artifact_creation >= 3:
            warnings.append(f"⚠️ **{self.state.turns_since_artifact_creation} turns without concrete action**: Act soon")
        
        # Generic action guidance (no task-specific hardcoding)
        if warnings and current_goal_desc:
            warnings.append("💡 **SOLUTION**: Take concrete action using the information and tools available")
        
        # Plan progression warnings
        if self.state.plan_progression_stage == "research" and self.state.current_loop >= 6:
            warnings.append("🚨 **STUCK IN RESEARCH STAGE**: Move to concrete action immediately")
        
        if warnings:
            warning_text = "\n".join(f"  {w}" for w in warnings)
            return f"\n\n**🚨 INFINITE LOOP PREVENTION ALERTS**:\n{warning_text}\n"
        
        return ""

    async def _classify_goal_type(self, goal_description: str) -> str:
        """Classify goal into a pattern category using LLM-based semantic understanding with caching"""
        
        # Cache classification to avoid repeated LLM calls for the same goal
        goal_hash = hash(goal_description.strip())
        if goal_hash in self._goal_classification_cache:
            return self._caches['goal_classification'][goal_hash]

        classification_prompt = f"""
    Classify this goal/task into ONE category based on its primary nature.

    Goal: "{goal_description}"

    Categories:
    - creation: Primary focus on creating, building, generating, producing something new
    - analysis: Primary focus on researching, analyzing, investigating, understanding information  
    - planning: Primary focus on creating plans, strategies, roadmaps, organizing future actions
    - communication: Primary focus on explaining, presenting, describing, communicating information
    - evaluation: Primary focus on comparing, assessing, rating, evaluating options/alternatives
    - problem_solving: Primary focus on solving problems, fixing issues, debugging, troubleshooting

    Respond with ONLY the category name (no explanation).
    """
        classification_schema = {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["creation", "analysis", "planning", "communication", "evaluation", "problem_solving"],
                    "description": "Primary goal category based on the nature of work"
                }
            },
            "required": ["category"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": classification_prompt}],
                response_schema=classification_schema,
                schema_name="goal_type_classification",
                use_cheap_model=True,
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                classification = response.get("category", "general_task")
                self._caches['goal_classification'][goal_hash] = classification
                return classification
                    
        except Exception as e:
            self.logger.debug(f"LLM goal classification failed: {e}")
            
        # Fallback
        fallback_classification = "general_task"
        self._caches['goal_classification'][goal_hash] = fallback_classification
        return fallback_classification
            
    def _get_goal_completion_criteria(self, goal_type: str, goal_description: str) -> Dict[str, Any]:
        """Get completion criteria and expected deliverable for different goal types"""
        if goal_type == "analysis":
            return {
                "expected_deliverable": "comprehensive analysis report",
                "deliverable_format": "structured document with findings, sources, and conclusions", 
                "completion_criteria": [
                    "Key findings identified and documented",
                    "Sources and evidence gathered", 
                    "Analysis report artifact created with is_output=True",
                    "Conclusions and insights summarized"
                ],
                "success_indicators": ["record_artifact with analysis findings", "comprehensive content with sources"],
                "max_search_attempts": 3,
                "min_facts_required": 5,
                "suggested_sections": ["Executive Summary", "Key Findings", "Sources", "Conclusions"]
            }
        elif goal_type == "creation":
            return {
                "expected_deliverable": "created content or artifact",
                "deliverable_format": "final deliverable as specified in goal",
                "completion_criteria": [
                    "Content created according to specifications",
                    "Final artifact recorded with is_output=True", 
                    "Quality and completeness verified"
                ],
                "success_indicators": ["record_artifact with created content", "specifications met"],
                "max_search_attempts": 1,
                "min_facts_required": 1,
                "suggested_sections": ["Content", "Specifications Met", "Quality Notes"]
            }
        elif goal_type == "problem_solving":
            return {
                "expected_deliverable": "solution documentation",
                "deliverable_format": "problem analysis and solution with implementation details",
                "completion_criteria": [
                    "Problem clearly defined and analyzed",
                    "Solution identified and documented", 
                    "Solution artifact created with is_output=True",
                    "Implementation steps provided"
                ],
                "success_indicators": ["record_artifact with solution details", "implementation guidance"],
                "max_search_attempts": 2,
                "min_facts_required": 2,
                "suggested_sections": ["Problem Definition", "Analysis", "Solution", "Implementation"]
            }
        elif goal_type == "planning":
            return {
                "expected_deliverable": "comprehensive plan",
                "deliverable_format": "structured plan with timeline and action items",
                "completion_criteria": [
                    "Planning objectives identified",
                    "Plan structure and timeline created",
                    "Plan artifact recorded with is_output=True", 
                    "Action items and milestones defined"
                ],
                "success_indicators": ["record_artifact with detailed plan", "actionable items listed"],
                "max_search_attempts": 2,
                "min_facts_required": 3,
                "suggested_sections": ["Objectives", "Timeline", "Action Items", "Milestones"]
            }
        elif goal_type == "communication":
            return {
                "expected_deliverable": "communication document",
                "deliverable_format": "clear explanation or presentation of topic",
                "completion_criteria": [
                    "Topic clearly explained with appropriate detail",
                    "Communication artifact created with is_output=True",
                    "Audience needs addressed"
                ],
                "success_indicators": ["record_artifact with explanation", "clear communication achieved"],
                "max_search_attempts": 2,
                "min_facts_required": 3,
                "suggested_sections": ["Overview", "Key Points", "Details", "Summary"]
            }
        elif goal_type == "evaluation":
            return {
                "expected_deliverable": "evaluation report",
                "deliverable_format": "comparative analysis with ratings and recommendations",
                "completion_criteria": [
                    "Items/options evaluated against criteria",
                    "Evaluation report artifact created with is_output=True",
                    "Recommendations provided"
                ],
                "success_indicators": ["record_artifact with evaluation results", "comparative analysis completed"],
                "max_search_attempts": 2,
                "min_facts_required": 3,
                "suggested_sections": ["Evaluation Criteria", "Comparison", "Ratings", "Recommendations"]
            }
        else:  # general_task
            return {
                "expected_deliverable": "task completion documentation",
                "deliverable_format": "summary of work completed with results",
                "completion_criteria": [
                    "Task requirements addressed",
                    "Work performed and documented",
                    "Results artifact created with is_output=True"
                ],
                "success_indicators": ["record_artifact with results", "task requirements met"],
                "max_search_attempts": 2,
                "min_facts_required": 2,
                "suggested_sections": ["Objectives", "Work Performed", "Results"]
            }
            


    async def _analyze_goal_completion_semantically(self, goal_description: str, goal_type: str) -> Dict[str, Any]:
        """Use LLM to semantically analyze goal completion readiness"""
        
        # Get current context for analysis
        progress_context = self._get_current_progress_summary()
        recent_actions = self._get_recent_actions_summary()
        
        completion_prompt = f"""
Analyze whether this goal is ready for completion based on the current progress context.

Goal: "{goal_description}"
Goal Type: {goal_type}
Progress Context: {progress_context}
Recent Actions: {recent_actions}

Rate on a scale of 0-100 how ready this goal is for completion:
- 0-20: Just started, needs significant work
- 21-40: Some progress made, more work needed
- 41-60: Good progress, approaching completion
- 61-80: Nearly complete, minor work remaining
- 81-100: Ready for completion/deliverable exists

Also rate 0-100 whether forced completion should occur due to:
- Excessive searching without progress
- Getting stuck in loops
- Diminishing returns from continued work

Respond in this exact format:
COMPLETION_READINESS: [0-100]
FORCE_COMPLETION: [0-100]
HAS_DELIVERABLE: [yes/no]
REASON: [brief explanation]
"""
        completion_analysis_schema = {
            "type": "object",
            "properties": {
                "completion_readiness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Score 0-100 for goal completion readiness"
                },
                "force_completion": {
                    "type": "integer", 
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Score 0-100 for whether completion should be forced"
                },
                "has_deliverable": {
                    "type": "boolean",
                    "description": "Whether deliverable exists"
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of the analysis"
                }
            },
            "required": ["completion_readiness", "force_completion", "has_deliverable", "reason"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": completion_prompt}],
                response_schema=completion_analysis_schema,
                schema_name="goal_completion_analysis",
                use_cheap_model=True,
                max_tokens=150,
                temperature=0.1
            )
            
            if response:
                completion_readiness = response.get("completion_readiness", 0)
                force_completion = response.get("force_completion", 0)
                has_deliverable = response.get("has_deliverable", False)
                reason = response.get("reason", "Structured analysis completed")
                
                ready_for_completion = completion_readiness >= 70 or force_completion >= 70
                force_needed = force_completion >= 70
                
                return {
                    "goal_type": goal_type,
                    "ready_for_completion": ready_for_completion,
                    "force_completion": force_needed,
                    "has_deliverable": has_deliverable,
                    "completion_readiness_score": completion_readiness,
                    "force_completion_score": force_completion,
                    "completion_message": f"Goal completion analysis: readiness={completion_readiness}%, force={force_completion}%, deliverable={has_deliverable}",
                    "reason": reason
                }
                
        except Exception as e:
            self.logger.debug(f"LLM goal completion analysis failed: {e}")
        
        # Intelligent fallback using semantic progress analysis
        return await self._analyze_completion_with_semantic_fallback(goal_description, goal_type)

    async def _parse_completion_analysis(self, llm_response: str, goal_type: str) -> Dict[str, Any]:
        """Parse LLM completion analysis response using intelligent JSON structure instead of primitive text parsing"""
        
        # Try to parse as JSON first (modern structured output approach)
        try:
            import json
            response_data = json.loads(llm_response.strip())
            
            completion_readiness = response_data.get("completion_readiness", 0)
            force_completion = response_data.get("force_completion", 0)
            has_deliverable = response_data.get("has_deliverable", False)
            reason = response_data.get("reason", "JSON analysis completed")
            
        except (json.JSONDecodeError, ValueError):
            # Fallback: Use LLM to extract structured data from text response
            parsed_data = await self._parse_completion_response_with_llm(llm_response)
            completion_readiness = parsed_data.get("completion_readiness", 0)
            force_completion = parsed_data.get("force_completion", 0)
            has_deliverable = parsed_data.get("has_deliverable", False)
            reason = parsed_data.get("reason", "LLM parsing completed")
        
        ready_for_completion = completion_readiness >= 70 or force_completion >= 70
        force_needed = force_completion >= 70
        
        return {
            "goal_type": goal_type,
            "ready_for_completion": ready_for_completion,
            "force_completion": force_needed,
            "has_deliverable": has_deliverable,
            "completion_readiness_score": completion_readiness,
            "force_completion_score": force_completion,
            "completion_message": f"Goal completion analysis: readiness={completion_readiness}%, force={force_completion}%, deliverable={has_deliverable}",
            "reason": reason
        }

    async def _parse_completion_response_with_llm(self, response_text: str) -> Dict[str, Any]:
        """Use LLM to extract structured data from text response instead of primitive parsing"""
        
        parsing_prompt = f"""
Extract structured information from this completion analysis response and provide it as a JSON object.

Response Text: "{response_text}"

Extract these values:
- completion_readiness: Score 0-100 for goal completion readiness
- force_completion: Score 0-100 for whether completion should be forced
- has_deliverable: Boolean for whether deliverable exists
- reason: Brief explanation text

Provide response as a JSON object:
{{"completion_readiness": 0, "force_completion": 0, "has_deliverable": false, "reason": "explanation"}}
"""
        parsing_schema = {
            "type": "object",
            "properties": {
                "completion_readiness": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Score 0-100 for goal completion readiness"
                },
                "force_completion": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Score 0-100 for whether completion should be forced"
                },
                "has_deliverable": {
                    "type": "boolean",
                    "description": "Boolean for whether deliverable exists"
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation text"
                }
            },
            "required": ["completion_readiness", "force_completion", "has_deliverable", "reason"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": parsing_prompt}],
                response_schema=parsing_schema,
                schema_name="completion_response_parsing",
                use_cheap_model=True,
                max_tokens=150,
                temperature=0.1
            )
            
            if response:
                return {
                    "completion_readiness": response.get("completion_readiness", 0),
                    "force_completion": response.get("force_completion", 0), 
                    "has_deliverable": response.get("has_deliverable", False),
                    "reason": response.get("reason", "Structured extraction completed")
                }
                
        except Exception as e:
            self.logger.debug(f"LLM response parsing failed: {e}")
        
        # Final fallback with sensible defaults
        return {
            "completion_readiness": 0,
            "force_completion": 0,
            "has_deliverable": False,
            "reason": "Response parsing failed"
        }

    async def _analyze_completion_with_semantic_fallback(self, goal_description: str, goal_type: str) -> Dict[str, Any]:
        """Semantic fallback for completion analysis when LLM fails"""
        
        # Use semantic understanding of current state
        has_recent_artifact = await self._check_recent_artifact_creation_semantically()
        has_sufficient_progress = await self._check_sufficient_progress_semantically(goal_description)
        should_force_completion = await self._check_force_completion_semantically()
        
        ready_for_completion = has_recent_artifact and (has_sufficient_progress or should_force_completion)
        
        return {
            "goal_type": goal_type,
            "ready_for_completion": ready_for_completion,
            "force_completion": should_force_completion,
            "has_deliverable": has_recent_artifact,
            "completion_readiness_score": 70 if ready_for_completion else 30,
            "force_completion_score": 80 if should_force_completion else 20,
            "completion_message": f"Semantic analysis: deliverable={has_recent_artifact}, progress={has_sufficient_progress}, force={should_force_completion}",
            "reason": "Semantic fallback analysis"
        }
    
    async def _check_recent_artifact_creation_semantically(self) -> bool:
        """Check if recent artifact creation occurred using semantic understanding"""
        return self.state.turns_since_artifact_creation <= 3
    
    async def _check_sufficient_progress_semantically(self, goal_description: str) -> bool:
        """Check if sufficient progress has been made using semantic analysis"""
        progress_context = self._get_current_progress_summary()
        
        progress_prompt = f"""
Rate how much meaningful progress has been made toward this goal.

Goal: "{goal_description}"  
Progress: {progress_context}

Rate 0-100 how much substantial progress has occurred:
- 0-30: Minimal progress, mostly searching
- 31-60: Moderate progress, some concrete work done
- 61-100: Significant progress, substantial work completed

Respond with just the numerical score (0-100).
"""
        progress_schema = {
            "type": "object",
            "properties": {
                "progress_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Progress score from 0-100"
                }
            },
            "required": ["progress_score"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": progress_prompt}],
                response_schema=progress_schema,
                schema_name="progress_scoring",
                use_cheap_model=True,
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                progress_score = response.get("progress_score", 0)
                return progress_score >= 60

        except Exception as e:
            self.logger.debug(f"LLM progress analysis failed: {e}")
        
        # Semantic fallback - check if meaningful work beyond searching has occurred
        return self.state.search_attempts_count > 0 and self.state.turns_since_artifact_creation <= 5
    
    async def _check_force_completion_semantically(self) -> bool:
        """Check if forced completion should occur using semantic understanding"""
        
        # Semantic indicators of stuck/diminishing returns patterns
        excessive_searching = self.state.search_attempts_count >= 5
        stuck_without_artifacts = self.state.turns_since_artifact_creation >= 8
        tool_repetition_loop = self.state.consecutive_same_tool_count >= 3
        
        return excessive_searching or stuck_without_artifacts or tool_repetition_loop
    
    def _get_recent_actions_summary(self) -> str:
        """Get summary of recent actions for context"""
        if hasattr(self.state, 'recent_tool_sequence') and self.state.recent_tool_sequence:
            recent_tools = self.state.recent_tool_sequence[-3:]  # Last 3 tools
            return f"Recent tools: {' → '.join(recent_tools)}"
        return "No recent actions"

    def _record_successful_pattern(self, goal_type: str, tool_sequence: List[str]):
        """Learn from successful tool sequences for future use"""
        if not tool_sequence or len(tool_sequence) < 2:
            return  # Need at least 2 tools to form a pattern
        
        # Clean tool names to base functions for pattern matching
        clean_sequence = []
        for tool in tool_sequence:
            if tool and isinstance(tool, str):
                base_name = self._get_base_function_name(tool)
                if base_name not in self._INTERNAL_OR_META_TOOLS_BASE_NAMES:
                    clean_sequence.append(base_name)
        
        if len(clean_sequence) < 2:
            return
        
        # Store the pattern
        if goal_type not in self.state.successful_patterns:
            self.state.successful_patterns[goal_type] = []
        
        # Avoid duplicates
        if clean_sequence not in self.state.successful_patterns[goal_type]:
            self.state.successful_patterns[goal_type].append(clean_sequence)
            # Keep only recent patterns (last 5)
            if len(self.state.successful_patterns[goal_type]) > 5:
                self.state.successful_patterns[goal_type] = self.state.successful_patterns[goal_type][-5:]
            
            # Track success count
            pattern_key = f"{goal_type}:{','.join(clean_sequence)}"
            self.state.pattern_success_count[pattern_key] = self.state.pattern_success_count.get(pattern_key, 0) + 1
            
            self.logger.info(f"📚 Learned successful pattern for {goal_type}: {' → '.join(clean_sequence)}")

    def _get_learned_tool_sequence(self, goal_type: str) -> List[str]:
        """Get the most successful tool sequence for a goal type"""
        if goal_type not in self.state.successful_patterns or not self.state.successful_patterns[goal_type]:
            return []
        
        # Find most successful pattern
        best_pattern = None
        best_score = 0
        
        for pattern in self.state.successful_patterns[goal_type]:
            pattern_key = f"{goal_type}:{','.join(pattern)}"
            score = self.state.pattern_success_count.get(pattern_key, 0)
            if score > best_score:
                best_score = score
                best_pattern = pattern
        
        if best_pattern:
            self.logger.info(f"📖 Using learned pattern for {goal_type}: {' → '.join(best_pattern)} (success_count: {best_score})")
            return [self._get_ums_tool_mcp_name(tool) for tool in best_pattern]
        
        return []

    def _suggest_tool_chain(self, current_goal_desc: str, last_action: str) -> List[str]:
        """Suggest logical next tools based on semantic understanding and learned patterns"""
        # First try learned patterns
        if current_goal_desc:
            goal_type = self._classify_goal_type(current_goal_desc)
            learned_sequence = self._get_learned_tool_sequence(goal_type)
            if learned_sequence:
                self.logger.info(f"🧠 Using learned pattern for {goal_type}")
                return learned_sequence[:3]  # Return first 3 tools
        
        # Use LLM-based semantic understanding for tool suggestions
        try:
            suggested_tools = self._suggest_next_tools_with_llm(current_goal_desc, last_action)
            if suggested_tools:
                return suggested_tools[:3]  # Limit to 3 tools
        except Exception as e:
            self.logger.debug(f"LLM tool suggestion failed: {e}, using fallback")
        
        # Intelligent fallback based on goal classification
        if current_goal_desc:
            goal_type = self._classify_goal_type(current_goal_desc)
            return self._get_fallback_tool_sequence_for_goal_type(goal_type)
        
        return []
    
    async def _suggest_next_tools_with_llm(self, goal_desc: str, last_action: str) -> List[str]:
        """Use LLM to suggest next tools based on semantic understanding of context"""
        
        # Cache tool suggestions to avoid repeated LLM calls
        context_hash = hash(f"{goal_desc}::{last_action}")
        if context_hash in self._tool_suggestion_cache:
            return self._caches['tool_suggestion'][context_hash]
        
        # Get current progress context
        progress_context = self._get_current_progress_summary()

        try:
            # Schedule background LLM call for better suggestions
            if hasattr(self, 'mcp_client') and self.mcp_client:
                self._start_background_task(self._get_llm_tool_suggestions_async, goal_desc, last_action, context_hash)
            
            # Return immediate intelligent fallback
            fallback_suggestions = await self._get_intelligent_tool_fallback(goal_desc, last_action, progress_context)
            self._caches['tool_suggestion'][context_hash] = fallback_suggestions
            return fallback_suggestions
            
        except Exception as e:
            self.logger.debug(f"Tool suggestion generation failed: {e}")
            return []
    
    async def _get_llm_tool_suggestions_async(self, goal_desc: str, last_action: str, context_hash: int):
        """Background task to get better tool suggestions from LLM"""
        progress_context = self._get_current_progress_summary()
        
        suggestion_prompt = f"""
Analyze the current situation and suggest the next 1-3 tools to use for making progress.

Current Goal: {goal_desc}
Last Action: {last_action or "Starting new task"}
Progress Context: {progress_context}

Available tool categories and their purposes:
- search/research: hybrid_search_memories, search_semantic_memories, browse, search
- information storage: store_memory, record_thought  
- creation: record_artifact, write_file
- analysis: get_artifacts, query_memories, get_workflow_details
- web interaction: browse, search, download

Consider:
1. What information is needed vs. what we already have
2. Whether we need to gather more data or start creating deliverables
3. The natural progression from research → analysis → creation → completion
4. Avoiding endless information gathering loops

Respond with 1-3 tool base names (like "search", "store_memory", "record_artifact") separated by commas, in logical order.
Focus on concrete progress toward the goal.
"""
        tool_suggestion_schema = {
            "type": "object",
            "properties": {
                "suggested_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of 1-3 base tool names for progress",
                    "maxItems": 3
                }
            },
            "required": ["suggested_tools"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": suggestion_prompt}],
                response_schema=tool_suggestion_schema,
                schema_name="tool_suggestion_analysis",
                use_cheap_model=True,
                max_tokens=80,
                temperature=0.2
            )
            
            if response:
                suggested_tools = []
                tool_list = response.get("suggested_tools", [])
                
                for tool_name in tool_list[:3]:  # Max 3 tools
                    if isinstance(tool_name, str):
                        # Convert to MCP names
                        mcp_tool_name = self._convert_base_to_mcp_tool_name(tool_name.strip())
                        if mcp_tool_name:
                            suggested_tools.append(mcp_tool_name)
                
                if suggested_tools:
                    # Update cache with LLM result
                    self._caches['tool_suggestion'][context_hash] = suggested_tools
                    self.logger.debug(f"🛠️ LLM suggested tools: {suggested_tools}")
                    
        except Exception as e:
            self.logger.debug(f"Background LLM tool suggestion failed: {e}")
    
    async def _get_intelligent_tool_fallback(self, goal_desc: str, last_action: str, progress_context: str) -> List[str]:
        """Intelligent fallback for tool suggestions when LLM is unavailable"""
        
        # Analyze current state using LLM semantic understanding  
        has_searched = await self._analyze_progress_state_with_llm(progress_context, "search_activity")
        has_info = await self._analyze_progress_state_with_llm(progress_context, "information_storage")
        has_artifacts = await self._analyze_progress_state_with_llm(progress_context, "artifact_creation")
        
        # Get goal classification for context
        goal_type = self._classify_goal_type(goal_desc)
        
        # Intelligent progression logic
        if goal_type == "analysis" and not has_searched:
            # Research phase - need to gather information
            return [
                self._get_ums_tool_mcp_name("hybrid_search_memories"),
                self._get_ums_tool_mcp_name("search")
            ]
        elif goal_type == "analysis" and has_searched and not has_info:
            # Information storage phase - gathered data but haven't stored it
            return [
                self._get_ums_tool_mcp_name("store_memory"),
                self._get_ums_tool_mcp_name("record_thought")
            ]
        elif goal_type == "analysis" and has_searched and has_info and not has_artifacts:
            # Analysis and creation phase - have information, now create deliverable
            return [
                self._get_ums_tool_mcp_name("record_artifact"),
                self._get_ums_tool_mcp_name("store_memory")
            ]
        elif goal_type == "creation" and not has_artifacts:
            # Direct creation phase
            return [
                self._get_ums_tool_mcp_name("record_artifact"),
                self._get_ums_tool_mcp_name("write_file")
            ]
        elif has_artifacts:
            # Completion phase
            return [
                self._get_ums_tool_mcp_name("get_artifacts"),
                self._get_ums_tool_mcp_name("update_goal_status")
            ]
        else:
            # Default progression - start with search
            return [self._get_ums_tool_mcp_name("hybrid_search_memories")]
    
    def _convert_base_to_mcp_tool_name(self, base_name: str) -> Optional[str]:
        """Convert base tool name to full MCP tool name"""
        base_name = base_name.lower().strip()
        
        # Map common base names to UMS functions
        base_to_ums = {
            "search": "hybrid_search_memories",
            "browse": "browse", 
            "store_memory": "store_memory",
            "record_thought": "record_thought",
            "record_artifact": "record_artifact", 
            "write_file": "write_file",
            "get_artifacts": "get_artifacts",
            "query_memories": "query_memories",
            "get_workflow_details": "get_workflow_details",
            "update_goal_status": "update_goal_status",
            "search_semantic_memories": "search_semantic_memories",
            "download": "download"
        }
        
        if base_name in base_to_ums:
            return self._get_ums_tool_mcp_name(base_to_ums[base_name])
        
        # Try to find partial matches
        for key, value in base_to_ums.items():
            if base_name in key or key in base_name:
                return self._get_ums_tool_mcp_name(value)
                
        return None
    
    def _get_fallback_tool_sequence_for_goal_type(self, goal_type: str) -> List[str]:
        """Get intelligent fallback tool sequence based on goal type"""
        
        sequences = {
            "analysis": [
                self._get_ums_tool_mcp_name("hybrid_search_memories"),
                self._get_ums_tool_mcp_name("store_memory"),
                self._get_ums_tool_mcp_name("record_artifact")
            ],
            "creation": [
                self._get_ums_tool_mcp_name("record_artifact"),
                self._get_ums_tool_mcp_name("write_file")
            ],
            "planning": [
                self._get_ums_tool_mcp_name("hybrid_search_memories"),
                self._get_ums_tool_mcp_name("record_artifact")
            ],
            "communication": [
                self._get_ums_tool_mcp_name("get_artifacts"),
                self._get_ums_tool_mcp_name("record_artifact")
            ],
            "evaluation": [
                self._get_ums_tool_mcp_name("hybrid_search_memories"),
                self._get_ums_tool_mcp_name("store_memory"),
                self._get_ums_tool_mcp_name("record_artifact")
            ],
            "problem_solving": [
                self._get_ums_tool_mcp_name("hybrid_search_memories"),
                self._get_ums_tool_mcp_name("record_artifact")
            ]
        }
        
        return sequences.get(goal_type, [self._get_ums_tool_mcp_name("hybrid_search_memories")])
    
    def _get_current_progress_summary(self) -> str:
        """Get a summary of current progress for context"""
        summary_parts = []
        
        if self.state.search_attempts_count > 0:
            summary_parts.append(f"Performed {self.state.search_attempts_count} searches")
        
        if self.state.turns_since_artifact_creation == 0:
            summary_parts.append("Just created an artifact")
        elif self.state.turns_since_artifact_creation < 5:
            summary_parts.append(f"Created artifact {self.state.turns_since_artifact_creation} turns ago")
            
        if self.state.plan_progression_stage != "initial":
            summary_parts.append(f"Current stage: {self.state.plan_progression_stage}")
            
        if self.state.last_action_summary:
            summary_parts.append(f"Last action: {self.state.last_action_summary[:100]}")
            
        return ". ".join(summary_parts) or "Starting new task"
    
    async def _analyze_progress_state_with_llm(self, progress_context: str, state_type: str) -> bool:
        """Use LLM to analyze progress state with 0-100 scoring"""
        
        state_descriptions = {
            "search_activity": "searching, researching, investigating, or gathering information",
            "information_storage": "storing, saving, recording, or documenting information in memory", 
            "artifact_creation": "creating, generating, building, or producing artifacts, files, or deliverables"
        }
        
        analysis_prompt = f"""
Analyze this progress context and rate how much {state_descriptions[state_type]} has occurred.

Progress Context: "{progress_context}"

Rate on a scale of 0-100:
- 0-20: No evidence of {state_descriptions[state_type]}
- 21-40: Minimal {state_descriptions[state_type]} activity  
- 41-60: Some {state_descriptions[state_type]} activity
- 61-80: Significant {state_descriptions[state_type]} activity
- 81-100: Extensive {state_descriptions[state_type]} activity

Respond with just the numerical score (0-100).
"""
        state_analysis_schema = {
            "type": "object",
            "properties": {
                "activity_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Score 0-100 for activity occurrence"
                }
            },
            "required": ["activity_score"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": analysis_prompt}],
                response_schema=state_analysis_schema,
                schema_name="progress_state_analysis",
                use_cheap_model=True,
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                progress_score = response.get("activity_score", 0)
                if 0 <= progress_score <= 100:
                    return progress_score >= 50  # Threshold for activity occurrence
        except Exception as e:
            self.logger.error(f"LLM progress analysis failed: {e}")
            
        return False  # Default if LLM fails
    
    async def _classify_memory_type_semantically(self, memory_type: str, content: str = "", description: str = "") -> str:
        """Classify memory type using LLM with structured output and caching"""
        
        # Cache to avoid repeated LLM calls
        context_hash = hash(f"{memory_type}::{content[:100]}::{description[:100]}")
        if context_hash in self._memory_type_cache:
            return self._caches['memory_type'][context_hash]
        
        # Default fallback
        classified_type = "text"
        
        if hasattr(self, 'mcp_client') and self.mcp_client:
            try:
                # Combine context for LLM analysis
                full_context = f"{memory_type} {content} {description}".strip()
                
                classification_prompt = f"""
    Classify this memory content into the most appropriate memory type based on its semantic meaning and purpose.

    Memory Type Label: {memory_type}
    Content: {content[:400]}
    Description: {description[:200]}
    Full Context: {full_context[:300]}

    Available types and their purposes:
    - fact: concrete information, data, research findings, evidence
    - insight: understanding, realizations, conclusions, key takeaways
    - reasoning_step: logical thinking, analysis process, rationale
    - observation: noticed patterns, what was observed or seen
    - procedure: methods, processes, steps, techniques
    - summary: overview, recap, condensed information
    - text: general content, notes, descriptions

    Consider:
    1. What kind of information this represents
    2. How it was obtained or created
    3. What purpose it serves in understanding
    4. The nature and structure of the content

    Respond with just the type name (e.g., "fact", "insight", "reasoning_step").
    """
                
                memory_classification_schema = {
                    "type": "object",
                    "properties": {
                        "memory_type": {
                            "type": "string",
                            "enum": ["fact", "insight", "reasoning_step", "observation", "procedure", "summary", "text"],
                            "description": "Classified memory type"
                        }
                    },
                    "required": ["memory_type"],
                    "additionalProperties": False
                }
                
                # Direct async LLM classification
                response = await self.mcp_client.query_llm_structured(
                    prompt_messages=[{"role": "user", "content": classification_prompt}],
                    response_schema=memory_classification_schema,
                    schema_name="memory_type_classification",
                    use_cheap_model=True,
                    max_tokens=30,
                    temperature=0.1
                )
                
                if response and response.get("memory_type"):
                    classified_type = response["memory_type"]
                    
            except Exception as e:
                self.logger.debug(f"LLM memory classification failed, using fallback: {e}")
        
        # Cache result
        self._caches['memory_type'][context_hash] = classified_type
        
        return classified_type
    
    async def _classify_artifact_type_semantically(self, artifact_type: str, name: str = "", description: str = "", content: str = "") -> str:
        """Classify artifact type using LLM with caching and background improvement"""
        
        # Cache classifications  
        context_hash = hash(f"{artifact_type}::{name[:50]}::{description[:50]}::{content[:100]}")
        if context_hash in self._artifact_type_cache:
            return self._caches['artifact_type'][context_hash]
        
        # Valid artifact types
        valid_types = {
            "file": "documents, reports, web pages, structured content that users can access",
            "text": "simple text content, notes, descriptions", 
            "code": "programming code, scripts, functions, algorithms",
            "data": "structured data, CSV, JSON, XML, YAML files",
            "image": "pictures, photos, graphics, diagrams",
            "chart": "graphs, plots, visualizations, charts", 
            "table": "tabular data, spreadsheets",
            "json": "JSON formatted data structures",
            "url": "web links, URLs, references to online content"
        }
        
        # Direct LLM classification with immediate fallback
        classified_type = "text"  # Safe default
        
        if hasattr(self, 'mcp_client') and self.mcp_client:
            try:
                # Combine context for LLM
                content_sample = content[:400] if content else ""
                
                prompt = f"""
    Classify this artifact into the most appropriate type based on its content and purpose.

    Type Label: {artifact_type}
    Name: {name}
    Description: {description}  
    Content Sample: {content_sample}

    Available types and purposes:
    {chr(10).join([f"- {type_name}: {purpose}" for type_name, purpose in valid_types.items()])}

    Consider:
    1. The nature of the content (text vs code vs data vs visual)
    2. How users would interact with this artifact
    3. The format and structure of the content
    4. The intended purpose and use case

    Respond with just the type name (e.g., "file", "code", "data").
    """
                
                schema = {
                    "type": "object",
                    "properties": {
                        "artifact_type": {
                            "type": "string", 
                            "enum": ["file", "text", "code", "data", "image", "chart", "table", "json", "url"],
                            "description": "Classified artifact type"
                        }
                    },
                    "required": ["artifact_type"],
                    "additionalProperties": False
                }
                
                # Direct async LLM call
                response = await self.mcp_client.query_llm_structured(
                    prompt_messages=[{"role": "user", "content": prompt}],
                    response_schema=schema,
                    schema_name="artifact_classification",
                    use_cheap_model=True,
                    max_tokens=30,
                    temperature=0.1
                )
                                
                if response and response.get("artifact_type"):
                    classified_type = response["artifact_type"]
                    
            except Exception as e:
                self.logger.debug(f"LLM artifact classification failed, using fallback: {e}")
        
        # Cache result
        self._caches['artifact_type'][context_hash] = classified_type
        
        return classified_type    

    async def _check_for_similar_artifacts(self, artifact_name: str, artifact_description: str = "") -> List[Dict[str, Any]]:
        """Check for existing similar artifacts in the current workflow to prevent duplication"""
        if not self.state.workflow_id:
            return []
        
        try:
            # Query for artifact creation memories in this workflow
            query_tool = self._get_ums_tool_mcp_name(UMS_FUNC_QUERY_MEMORIES)
            if not self._find_tool_server(query_tool):
                self.logger.debug("Cannot check for similar artifacts: query_memories tool not available")
                return []
            
            result = await self._execute_tool_call_internal(
                query_tool,
                {
                    "workflow_id": self.state.workflow_id,
                    "memory_type": MemoryType.ARTIFACT_CREATION.value,
                    "limit": 20,  # Check recent artifacts
                    "include_content": True
                },
                record_action=False
            )
            
            if not result.get("success"):
                self.logger.debug("Failed to query existing artifacts")
                return []
            
            similar_artifacts = []
            existing_memories = result.get("data", {}).get("memories", [])
            
            # Async similarity check based on name and description
            artifact_name_lower = artifact_name.lower()
            artifact_desc_lower = artifact_description.lower()
            
            for memory in existing_memories:
                if not isinstance(memory, dict):
                    continue
                    
                content = memory.get("content", "")
                if not content:
                    continue
                
                content_lower = content.lower()
                
                # Async similarity checks
                name_similar = await self._are_artifacts_similar_async(artifact_name_lower, content_lower)
                desc_similar = False
                if artifact_desc_lower:
                    desc_similar = await self._are_artifacts_similar_async(artifact_desc_lower, content_lower)
                
                if name_similar or desc_similar:
                    similar_artifacts.append(memory)
            
            if similar_artifacts:
                self.logger.info(f"🔍 Found {len(similar_artifacts)} potentially similar artifacts in current workflow")
                
            return similar_artifacts
            
        except Exception as e:
            self.logger.warning(f"Error checking for similar artifacts: {e}")
            return []

    async def _are_artifacts_similar_async(self, artifact_name: str, existing_content: str) -> bool:
        """Check if artifacts are similar using LLM semantic analysis with caching"""
        
        if not artifact_name or not existing_content:
            return False
        
        # Cache to avoid repeated LLM calls
        content_hash = hash(f"similarity:{artifact_name}:{existing_content[:200]}")
        if content_hash in self._similarity_cache:
            return self._caches['similarity'][content_hash]
        
        # Default fallback while we do LLM analysis
        fallback_result = False
        self._caches['similarity'][content_hash] = fallback_result
        
        # Do immediate LLM analysis if available
        if hasattr(self, 'mcp_client') and self.mcp_client:
            try:
                content_sample = existing_content[:500]
                
                similarity_prompt = f"""
    Analyze the semantic similarity between this artifact name and existing content.

    New Artifact Name: "{artifact_name}"
    Existing Content Sample: "{content_sample}"

    Rate if they are similar enough to be considered duplicates or redundant:

    Scoring criteria:
    - Are they about the same topic/subject matter?
    - Do they serve the same or very similar purposes?
    - Would both existing together be redundant?

    Consider similar if there's significant conceptual overlap that suggests redundancy.

    Respond with true if they are similar/redundant, false if they are different enough to both be useful.
    """
                
                similarity_schema = {
                    "type": "object",
                    "properties": {
                        "is_similar": {
                            "type": "boolean",
                            "description": "Whether the artifacts are similar enough to be considered redundant"
                        }
                    },
                    "required": ["is_similar"],
                    "additionalProperties": False
                }
                
                # Proper async LLM call
                response = await self.mcp_client.query_llm_structured(
                    prompt_messages=[{"role": "user", "content": similarity_prompt}],
                    response_schema=similarity_schema,
                    schema_name="similarity_analysis",
                    use_cheap_model=True,
                    max_tokens=20,
                    temperature=0.1
                )
                
                if response and "is_similar" in response:
                    is_similar = response["is_similar"]
                    self._caches['similarity'][content_hash] = is_similar
                    return is_similar
                    
            except Exception as e:
                self.logger.debug(f"LLM similarity analysis failed: {e}")
        
        return fallback_result

    def _is_in_loop(self, tool_name: str) -> bool:
        """Detect if the agent is stuck in an infinite loop with the same tool"""
        base_tool = self._get_base_function_name(tool_name)
        
        # Update loop detection state
        if self.state.last_tool_executed == tool_name:
            self.state.consecutive_same_tool_count += 1
        else:
            self.state.consecutive_same_tool_count = 1
            self.state.last_tool_executed = tool_name
        
        # Update recent tool sequence (keep last 5)
        self.state.recent_tool_sequence.append(base_tool)
        if len(self.state.recent_tool_sequence) > 5:
            self.state.recent_tool_sequence = self.state.recent_tool_sequence[-5:]
        
        # Count search attempts for search tools
        search_tools = {"hybrid_search_memories", "search_semantic_memories", "web_search", "smart_browser"}
        if base_tool in search_tools:
            self.state.search_attempts_count += 1
        
        # Detect various loop patterns
        
        # 1. Same tool executed 3+ times consecutively
        if self.state.consecutive_same_tool_count >= 3:
            return True
        
        # 2. Repetitive pattern in recent sequence (e.g., A-B-A-B-A)
        if len(self.state.recent_tool_sequence) >= 4:
            recent = self.state.recent_tool_sequence[-4:]
            if len(set(recent)) <= 2 and recent[0] == recent[2]:  # A-B-A-B pattern
                return True
        
        # 3. Excessive search attempts without progress (3+ searches without creating artifacts)
        if (self.state.search_attempts_count >= 3 and 
            self.state.turns_since_artifact_creation > 5):
            return True
        
        # 4. Check if we've been in the same progression stage too long
        if (self.state.plan_progression_stage == "research" and 
            self.state.turns_since_artifact_creation > 7):
            return True
        
        return False

    def _update_plan_progression_stage(self, tool_name: str, success: bool) -> None:
        """Update the plan progression stage based on tool usage"""
        if not success:
            return
            
        base_tool = self._get_base_function_name(tool_name)
        
        # Update turns since artifact creation
        if base_tool == "record_artifact":
            self.state.turns_since_artifact_creation = 0
            self.state.search_attempts_count = 0  # Reset search count after artifact creation
            self.state.plan_progression_stage = "creation"
        else:
            self.state.turns_since_artifact_creation += 1
        
        # Determine current stage based on tool usage
        search_tools = {"hybrid_search_memories", "search_semantic_memories", "web_search", "smart_browser"}
        analysis_tools = {"store_memory", "create_memory_link", "record_thought"}
        creation_tools = {"record_artifact", "write_file", "create_file"}
        
        if base_tool in search_tools and self.state.plan_progression_stage in ["initial", "research"]:
            self.state.plan_progression_stage = "research"
        elif base_tool in analysis_tools and self.state.plan_progression_stage in ["research", "analysis"]:
            self.state.plan_progression_stage = "analysis"  
        elif base_tool in creation_tools:
            self.state.plan_progression_stage = "creation"
        elif base_tool in {"update_goal_status", "update_workflow_status"}:
            self.state.plan_progression_stage = "completion"

    def _generate_progression_aware_plan(self, current_goal_desc: str) -> List[PlanStep]:
        """Generate a plan that moves toward completion based on current progression stage"""
        
        # Check if we should force progression to prevent infinite loops
        if self._should_force_progression_to_creation():
            self.logger.warning("🚨 INFINITE LOOP PREVENTION: Forcing progression to deliverable creation")
            return self._get_forced_progression_plan(current_goal_desc)
        
        # Analyze what stage we should be in based on goal and current state
        goal_lower = current_goal_desc.lower()
        
        # Determine if we need to move to the next stage
        force_progression = (
            self.state.turns_since_artifact_creation > 8 or
            self.state.search_attempts_count >= 4 or
            self.state.consecutive_same_tool_count >= 2
        )
        
        plan_steps = []
        
        if self.state.plan_progression_stage == "initial":
            # ACTION-FIRST APPROACH: Bias toward concrete action over endless information gathering
            # Use semantic goal classification instead of primitive keyword matching
            goal_type = self._classify_goal_type(current_goal_desc)
            
            if goal_type in ["creation", "communication"]:
                plan_steps.append(PlanStep(
                    description="Take immediate action to fulfill the goal using available knowledge"
                ))
            elif goal_type in ["analysis", "evaluation"]:
                # Only allow research if explicitly requested AND with limits
                plan_steps.append(PlanStep(
                    description="Conduct ONE focused search for the most critical information needed",
                    assigned_tool=self._get_ums_tool_mcp_name("hybrid_search_memories")
                ))
                plan_steps.append(PlanStep(
                    description="Take concrete action based on findings (NO MORE SEARCHING)"
                ))
            else:
                plan_steps.append(PlanStep(
                    description="Take concrete action toward the goal using available information"
                ))
                
        elif self.state.plan_progression_stage == "research":
            if force_progression or self.state.search_attempts_count >= 2:  # Reduced from 3 to 2
                # Force move to action phase
                self.logger.info("🎯 PROGRESSION: Forcing move from research to action phase")
                plan_steps.append(PlanStep(
                    description="Take concrete action based on gathered information"
                ))
            else:
                # ONE more search ONLY, then MUST act
                plan_steps.append(PlanStep(
                    description="Final search for any missing critical information (LAST SEARCH)",
                    assigned_tool=self._get_ums_tool_mcp_name("hybrid_search_memories")
                ))
                plan_steps.append(PlanStep(
                    description="Take concrete action immediately after this search"
                ))
                
        elif self.state.plan_progression_stage == "analysis":
            # Move to action
            plan_steps.append(PlanStep(
                description="Take concrete action based on analysis"
            ))
            
        elif self.state.plan_progression_stage == "creation":
            # Check if goal is complete or if more actions needed
            if "and" in goal_lower or "multiple" in goal_lower:
                plan_steps.append(PlanStep(
                    description="Check if all required actions have been completed",
                    assigned_tool=self._get_ums_tool_mcp_name("get_goal_details")
                ))
            else:
                plan_steps.append(PlanStep(
                    description="Verify goal completion and update goal status",
                    assigned_tool=self._get_ums_tool_mcp_name("update_goal_status")
                ))
        
        return plan_steps if plan_steps else [PlanStep(description="Create the primary deliverable for this goal")]

    def _generate_concrete_action_plan(self, goal_description: str) -> List[PlanStep]:
        """
        Generate concrete, action-oriented plans that prevent infinite loops through GENERIC principles:
        
        1. BIAS TOWARD ACTION over endless information gathering
        2. CONCRETE STEPS with clear success criteria  
        3. LIMITED RESEARCH with forced progression to action
        4. MEASURABLE OUTCOMES that can be evaluated
        """
        # GENERIC ACTION-FIRST PLANNING PRINCIPLES:
        
        # Use semantic goal classification instead of primitive keyword matching  
        goal_type = self._classify_goal_type(goal_description)
        
        # If goal explicitly requires research/investigation, allow ONE focused search then ACT
        if goal_type in ["analysis", "evaluation"]:
            return [
                PlanStep(
                    description="Gather essential information for the task (ONE focused search)",
                    assigned_tool=self._get_ums_tool_mcp_name(UMS_FUNC_HYBRID_SEARCH),
                    status="planned"
                ),
                PlanStep(
                    description="Take concrete action based on gathered information",
                    status="planned"
                )
            ]
        
        # If goal is action-oriented, proceed directly to action
        elif goal_type in ["creation", "communication"]:
            return [
                PlanStep(
                    description=f"Execute the requested action: {goal_description[:120]}",
                    status="planned"
                )
            ]
        
        # Default: favor immediate action with available information
        else:
            return [
                PlanStep(
                    description="Take immediate concrete action toward the goal using available information",
                    status="planned"
                ),
                PlanStep(
                    description="Evaluate results and determine if goal is complete",
                    status="planned"
                )
            ]

    def _should_force_progression_to_creation(self) -> bool:
        """
        Determine if agent should be forced to stop researching and start creating deliverables.
        
        This is a key infinite loop prevention mechanism.
        """
        # Force progression after 2 search attempts (reduced from 3)
        if self.state.search_attempts_count >= 2:
            self.logger.warning("🚨 FORCING PROGRESSION: 2+ search attempts detected")
            return True
        
        # Force progression after 5 turns without artifact creation (reduced from 6)
        if self.state.turns_since_artifact_creation >= 5:
            self.logger.warning("🚨 FORCING PROGRESSION: 5+ turns without creating artifacts")
            return True
        
        # Force progression if stuck in research stage too long
        if (self.state.plan_progression_stage == "research" and 
            self.state.current_loop >= 6):  # Reduced from 8
            self.logger.warning("🚨 FORCING PROGRESSION: Stuck in research stage too long")
            return True
        
        # Force progression if same tool used 2+ times consecutively (reduced from 3)
        if self.state.consecutive_same_tool_count >= 2:
            self.logger.warning("🚨 FORCING PROGRESSION: Same tool repeated 2+ times")
            return True
        
        return False

    def _get_forced_progression_plan(self, current_goal_desc: str) -> List[PlanStep]:
        """
        Generate a plan that FORCES progression to concrete action.
        Used when infinite loop prevention triggers.
        """
        # Generic action-oriented forced progression (no task-specific hardcoding)
        action_desc = f"Take immediate concrete action toward the goal using available information"
        
        return [
            PlanStep(
                description=f"FORCED PROGRESSION: {action_desc}",
                status="planned"
            ),
            PlanStep(
                description="Evaluate action results and determine if goal requirements are met",
                status="planned"
            )
        ]

    def _handle_tool_error(self, error: Dict) -> Dict:
        """Handle tool errors clearly and simply - don't try to be clever"""
        # Log the error clearly
        error_msg = error.get("error_message", "Unknown error")
        tool_name = error.get("tool", "unknown_tool")
        
        self.logger.error(f"Tool {tool_name} failed: {error_msg}")
        
        # Return the error to the caller without trying to fix it
        return error
        

                
    def classify_error(self, error_msg: str) -> str:
        """Use simple pattern matching to classify errors"""
        error_lower = error_msg.lower()
        
        if "workflow" in error_lower:
            return "workflow_error"
        elif "file" in error_lower or "access" in error_lower:
            return "file_error"
        elif "memory" in error_lower:
            return "memory_error"
        elif "artifact" in error_lower:
            return "artifact_error"
        elif "missing" in error_lower or "required" in error_lower:
            return "validation_error"
        else:
            return "unknown_error"

    # Removed complex LLM error classification - now using simple pattern matching
    
    async def _improve_error_classification_with_llm(self, error_msg: str, context: str, error_category: str, context_hash: int):
        """Use LLM to improve error classification"""
        
        error_classification_prompt = f"""
Analyze this error message and rate how well it matches the specified error category.

Error Message: "{error_msg}"
Context: "{context}"
Category to check: {error_category}

Error Categories:
- memory_type_error: Errors related to invalid memory types or memory classification issues
- artifact_type_error: Errors related to invalid artifact types or artifact classification issues  
- missing_fields_error: Errors about missing required fields, parameters, or validation failures
- workflow_id_error: Errors related to workflow ID issues, not found, or invalid workflow references
- file_access_error: Errors related to file permissions, access denied, file not found, or path issues
- planning_error: Errors related to planning, sequencing, dependencies, or complex operations

Rate the confidence that this error matches the category on a scale of 0-100:
- 0-20: Definitely not this error type
- 21-40: Probably not this error type
- 41-60: Uncertain, could be this type
- 61-80: Likely this error type
- 81-100: Definitely this error type

Consider:
1. Semantic alignment with error category
2. Context supporting the classification
3. Specific terminology and patterns

Respond with just the numerical confidence score (0-100).
"""
        error_classification_improvement_schema = {
            "type": "object",
            "properties": {
                "confidence_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Confidence score 0-100 for error category match"
                }
            },
            "required": ["confidence_score"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": error_classification_prompt}],
                response_schema=error_classification_improvement_schema,
                schema_name="error_classification_improvement",
                use_cheap_model=True,
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                confidence_score = response.get("confidence_score", 0)
                if 0 <= confidence_score <= 100:
                    # Use threshold of 70+ for high confidence error classification  
                    ERROR_CLASSIFICATION_THRESHOLD = 70
                    matches_category = confidence_score >= ERROR_CLASSIFICATION_THRESHOLD
                    # Update cache with LLM result
                    self._caches['error_classification'][context_hash] = matches_category
                    self.logger.debug(f"🧠 LLM error classification: {error_category} confidence={confidence_score}, matches={matches_category} (threshold={ERROR_CLASSIFICATION_THRESHOLD})")
                    
        except Exception as e:
            self.logger.debug(f"Background LLM error classification failed: {e}")



    async def _analyze_tool_type_with_llm(self, tool_name: str, tool_type: str) -> bool:
        """Comprehensive LLM-based tool type analysis with intelligent caching and fallback"""
        
        # Cache to avoid repeated LLM calls
        context_hash = hash(f"{tool_type}:{tool_name}")
        if context_hash in self._tool_analysis_cache:
            return self._caches['tool_analysis'][context_hash]

        # LLM analysis with structured output
        prompt = f"""
    Analyze this tool and determine if it matches the specified type.

    Tool Name: "{tool_name}"
    Tool Type: {tool_type}

    Tool Type Definitions:
    - search_tool: Tools for searching, querying, finding information, retrieving data from databases/memory
    - artifact_tool: Tools for creating, recording, writing, generating artifacts/files/documents/deliverables  
    - research_tool: Tools for researching, browsing, fetching, gathering information from web/APIs

    Consider:
    1. Tool name semantic meaning and functionality implications
    2. Expected operations and use cases
    3. Primary purpose alignment with tool type definition

    Does this tool match the specified type?
    """
        
        tool_analysis_schema = {
            "type": "object",
            "properties": {
                "matches_type": {
                    "type": "boolean",
                    "description": "Whether the tool matches the specified type"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in the classification (0.0-1.0)"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation of the classification decision"
                }
            },
            "required": ["matches_type"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": prompt}],
                response_schema=tool_analysis_schema,
                schema_name="tool_type_analysis",
                use_cheap_model=True,
                max_tokens=100,
                temperature=0.1
            )
            
            if response:
                result = response.get("matches_type", False)
                confidence = response.get("confidence", 0.5)
                reasoning = response.get("reasoning", "")
                
                # Cache the result
                self._caches['tool_analysis'][context_hash] = result
                
                self.logger.debug(f"🧠 LLM tool analysis: {tool_name} → {tool_type} = {result} (confidence: {confidence:.2f}) - {reasoning}")
                return result
                
        except Exception as e:
            self.logger.debug(f"LLM tool type analysis failed: {e}")
        
        # Intelligent fallback using semantic patterns
        tool_lower = tool_name.lower()
        
        if tool_type == "search_tool":
            result = any(pattern in tool_lower for pattern in [
                "search", "query", "find", "lookup", "retrieve", "fetch", "get", "hybrid"
            ])
        elif tool_type == "artifact_tool":
            result = any(pattern in tool_lower for pattern in [
                "record", "create", "artifact", "write", "save", "generate", "build", "file"
            ])
        elif tool_type == "research_tool":
            result = any(pattern in tool_lower for pattern in [
                "research", "browse", "web", "browser", "smart", "investigate", "study"
            ])
        else:
            result = False
        
        # Cache fallback result
        self._caches['tool_analysis'][context_hash] = result
        self.logger.debug(f"🔄 Fallback tool analysis: {tool_name} → {tool_type} = {result}")
        return result
    
    async def _analyze_content_format_semantically(self, content: str) -> Dict[str, Any]:
        """Analyze content format using LLM with intelligent fallback"""
        
        if not content or not isinstance(content, str):
            return {"primary_format": "text", "confidence": 0.0}
        
        # Cache to avoid repeated LLM calls
        content_hash = hash(content[:500])
        if content_hash in self._content_format_cache:
            return self._caches['content_format'][content_hash]
        
        # Intelligent fallback instead of primitive scoring
        fallback_analysis = await self._get_content_format_intelligent_fallback(content)
        self._caches['content_format'][content_hash] = fallback_analysis
        
        # Schedule LLM analysis for improvement
        if hasattr(self, 'mcp_client') and self.mcp_client:
            self._start_background_task(
                self._improve_content_format_analysis_with_llm,
                content, content_hash
            )
        
        return fallback_analysis
    
    async def _get_content_format_intelligent_fallback(self, content: str) -> Dict[str, Any]:
        """Use LLM for content format analysis"""
        
        format_prompt = f"""
Analyze this content and determine its format.

Content: {content[:500]}

Available formats: html, json, xml, python, javascript, c_cpp, csv, markdown, text

Respond with just the format name (e.g., "html", "json", "text").
"""
        content_format_schema = {
            "type": "object",
            "properties": {
                "format_name": {
                    "type": "string",
                    "enum": ["html", "json", "xml", "python", "javascript", "c_cpp", "csv", "markdown", "text"],
                    "description": "Detected content format"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in format detection"
                }
            },
            "required": ["format_name"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": format_prompt}],
                response_schema=content_format_schema,
                schema_name="content_format_analysis",
                use_cheap_model=True,
                max_tokens=30,
                temperature=0.1
            )
            
            if response:
                format_name = response.get("format_name", "text")
                confidence = response.get("confidence", 0.9)
                return {"primary_format": format_name, "confidence": confidence}
        except Exception as e:
            self.logger.error(f"LLM content format analysis failed: {e}")
            
        return {"primary_format": "text", "confidence": 0.3}
    
    async def _improve_content_format_analysis_with_llm(self, content: str, content_hash: int):
        """Use LLM to improve content format analysis"""
        
        format_analysis_prompt = f"""
Analyze this content and determine its format/type. 

Content sample: {content[:800]}

Choose the most appropriate format:
- html: HTML web content with tags and structure
- json: JSON data format with key-value pairs
- xml: XML markup with opening/closing tags
- python: Python programming code
- javascript: JavaScript code
- c_cpp: C/C++ programming code
- csv: Comma-separated values data
- markdown: Markdown formatted text
- text: Plain text content

Consider:
1. Syntax patterns and structure
2. Opening/closing patterns
3. Programming language keywords
4. Data organization patterns

Respond with just the format name and confidence (0.0-1.0) in this format: "format_name,confidence"
For example: "html,0.9" or "text,0.3"
"""
        content_format_improvement_schema = {
            "type": "object",
            "properties": {
                "format_name": {
                    "type": "string",
                    "enum": ["html", "json", "xml", "python", "javascript", "c_cpp", "csv", "markdown", "text"],
                    "description": "Detected content format"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Confidence in format detection"
                }
            },
            "required": ["format_name", "confidence"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": format_analysis_prompt}],
                response_schema=content_format_improvement_schema,
                schema_name="content_format_improvement",
                use_cheap_model=True,
                max_tokens=50,
                temperature=0.1
            )
            
            if response:
                format_name = response.get("format_name", "text")
                confidence = response.get("confidence", 0.5)
                
                # Update cache with LLM result
                improved_analysis = {
                    "primary_format": format_name,
                    "confidence": confidence,
                    "llm_enhanced": True
                }
                self._caches['content_format'][content_hash] = improved_analysis
                self.logger.debug(f"🧠 LLM improved content format: {format_name} ({confidence:.1f})")
                    
        except Exception as e:
            self.logger.debug(f"Background LLM content format analysis failed: {e}")
    
    async def _analyze_feedback_for_replan_semantically(self, feedback: str) -> bool:
        """Analyze feedback using LLM to detect if replanning is needed"""
        
        if not feedback or not isinstance(feedback, str):
            return False
        
        # Cache to avoid repeated LLM calls
        feedback_hash = hash(feedback[:500])
        if feedback_hash in self._replan_feedback_cache:
            return self._caches['replan_feedback'][feedback_hash]
        # No LLM result = assume no replanning needed
        self._caches['replan_feedback'][feedback_hash] = False
        
        # Schedule LLM analysis for improvement
        if hasattr(self, 'mcp_client') and self.mcp_client:
            self._start_background_task(
                self._improve_replan_analysis_with_llm,
                feedback, feedback_hash
            )
        
        return False
    
    async def _improve_replan_analysis_with_llm(self, feedback: str, feedback_hash: int):
        """Use LLM to improve replan analysis"""
        
        replan_analysis_prompt = f"""
Analyze this feedback and rate how strongly it suggests the agent should replan or change approach.

Feedback: "{feedback}"

Rate the replanning necessity on a scale of 0-100:
- 0-20: Feedback is positive, continue current approach
- 21-40: Minor issues, current approach mostly fine
- 41-60: Moderate concerns, some adjustments may be needed
- 61-80: Significant problems, replanning strongly recommended  
- 81-100: Critical issues, replanning absolutely necessary

Consider:
1. Severity of criticism or ineffectiveness indicated
2. Explicit recommendations to change strategy
3. Signs of being stuck or making poor progress
4. Urgency of suggested approach changes

Respond with just the numerical replanning necessity score (0-100).
"""
        replan_analysis_schema = {
            "type": "object",
            "properties": {
                "replan_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Replanning necessity score 0-100"
                }
            },
            "required": ["replan_score"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": replan_analysis_prompt}],
                response_schema=replan_analysis_schema,
                schema_name="replan_analysis",
                use_cheap_model=True,
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                replan_score = response.get("replan_score", 0)
                if 0 <= replan_score <= 100:
                    # Use configurable threshold for triggering replanning
                    needs_replan = replan_score >= LLM_REPLAN_NECESSITY_THRESHOLD
                    # Update cache with LLM result
                    self._caches['replan_feedback'][feedback_hash] = needs_replan
                    self.logger.debug(f"🧠 LLM replan analysis: score={replan_score}, needs_replan={needs_replan} (threshold={LLM_REPLAN_NECESSITY_THRESHOLD})")
                    
        except Exception as e:
            self.logger.debug(f"Background LLM replan analysis failed: {e}")

    async def _attempt_file_path_auto_fix(self, tool_name: str, args: Dict, error_context: Dict) -> bool:
        """Attempt to fix file path/permission issues using diagnose_file_access_issues tool"""
        try:
            # Find the diagnose tool
            diagnose_tool_name = self._get_ums_tool_mcp_name(UMS_FUNC_DIAGNOSE_FILE_ACCESS)
            if not self._find_tool_server(diagnose_tool_name):
                self.logger.warning("🔧 diagnose_file_access_issues tool not available for auto-fix")
                return False
            
            # Look for file path arguments that might be causing issues
            path_args = ["path", "file_path", "output_path", "filename", "filepath", "dir", "directory"]
            file_path = None
            path_arg_name = None
            
            for arg_name in path_args:
                if arg_name in args and args[arg_name]:
                    file_path = str(args[arg_name])
                    path_arg_name = arg_name
                    break
            
            if not file_path:
                # Try to extract path from content or other args
                content = args.get("content", "")
                if isinstance(content, str) and ("save to" in content.lower() or "write to" in content.lower()):
                    # Use a safe default path for content creation
                    file_path = "/home/ubuntu/ultimate_mcp_server/storage/agent_output.txt"
                    path_arg_name = "content_path"  # We'll handle this specially
                else:
                    return False
            
            self.logger.info(f"🔧 Diagnosing file access issue for path: {file_path}")
            
            # Call diagnose tool
            diagnose_args = {
                "path_to_check": file_path,
                "operation_type": "artifacts"
            }
            
            diagnose_result = await self._execute_tool_call_internal(
                diagnose_tool_name, diagnose_args, record_action=False
            )
            
            if not diagnose_result.get("success"):
                self.logger.warning(f"🔧 File diagnosis failed: {diagnose_result.get('error_message')}")
                return False
            
            diagnosis = diagnose_result.get("data", {})
            safe_alternatives = diagnosis.get("safe_alternatives", [])
            
            if not safe_alternatives:
                # Provide fallback safe locations
                import os
                safe_alternatives = [
                    "/home/ubuntu/ultimate_mcp_server/storage/",
                    os.path.expanduser("~/.ultimate_mcp_server/artifacts/"),
                    "/tmp/ultimate_mcp_server_artifacts/"
                ]
            
            # Use the first safe alternative
            safe_path = safe_alternatives[0]
            
            # If it's a directory, append a filename
            if safe_path.endswith('/'):
                original_name = file_path.split('/')[-1] if '/' in file_path else "agent_artifact.txt"
                safe_path = safe_path + original_name
            
            # Ensure the directory exists
            import os
            os.makedirs(os.path.dirname(safe_path), exist_ok=True)
            
            # Update the arguments
            if path_arg_name == "content_path":
                # Special case: we need to modify the content or add a path argument
                if "artifact_type" not in args:
                    args["artifact_type"] = "file"
                # For record_artifact, we might need to update the content
                if tool_name.endswith("record_artifact"):
                    args["file_path"] = safe_path
            else:
                args[path_arg_name] = safe_path
            
            self.logger.info(f"🔧 Auto-fixing file path '{file_path}' → '{safe_path}'")
            return True
            
        except Exception as e:
            self.logger.error(f"🔧 Error in file path auto-fix: {e}")
            return False

    def _get_safe_file_path(self, filename: str = "agent_output.txt", subdir: str = "") -> str:
        """Get a safe file path for creating artifacts"""
        import os
        
        # Preferred locations in order
        safe_bases = [
            "/home/ubuntu/ultimate_mcp_server/storage",
            os.path.expanduser("~/.ultimate_mcp_server/artifacts"),
            "/tmp/ultimate_mcp_server_artifacts"
        ]
        
        for base_path in safe_bases:
            try:
                full_path = os.path.join(base_path, subdir, filename) if subdir else os.path.join(base_path, filename)
                # Ensure directory exists
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                # Test write permission
                test_file = full_path + ".test"
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                return full_path
            except Exception:
                continue
        
        # Fallback to current directory
        return os.path.join(os.getcwd(), filename)

    def _is_likely_file_creation_task(self) -> bool:
        """Semantically detect if the current task is likely to involve creating files"""
        
        # Cache file creation detection
        context_text = ""
        if self.state.current_plan:
            context_text += " ".join([step.description for step in self.state.current_plan])
        if self.state.goal_stack:
            current_goal = self.state.goal_stack[-1] if self.state.goal_stack else {}
            context_text += " " + str(current_goal.get("description", ""))
        context_text += " " + self.state.last_action_summary
        
        context_hash = hash(context_text[:500])
        if context_hash in self._file_creation_cache:
            return self._caches['file_creation'][context_hash]
        
        # Use simple heuristics for file creation detection
        is_file_creation = any(keyword in context_text.lower() for keyword in [
            "create", "generate", "write", "build", "make", "save", "file", "document", 
            "artifact", "deliverable", "output", "export", "download", "report"
        ])
        self._caches['file_creation'][context_hash] = is_file_creation
        
        return is_file_creation
    

    

    

    


    async def _should_suggest_multi_tool_guidance(self) -> bool:
        """Semantically determine if the agent would benefit from multi-tool guidance"""
        
        # Suggest if there are consecutive errors (might be due to poor tool planning)
        if self.state.consecutive_error_count >= 2:
            return True
        
        # Suggest for complex plans with many steps
        if len(self.state.current_plan) > 4:
            return True
        
        # Use semantic understanding for error analysis
        if self.state.last_error_details:
            error_context = f"{self.state.last_error_details.get('type', '')} {self.state.last_error_details.get('error', '')}"
            # Use LLM analysis for planning-related errors
            is_planning_related = await self._analyze_error_context_with_llm(error_context, "planning_related")
            if is_planning_related:
                return True
        
        # Use semantic understanding for goal complexity analysis
        if self.state.goal_stack:
            current_goal = self.state.goal_stack[-1] if self.state.goal_stack else {}
            goal_desc = str(current_goal.get("description", ""))
            # Use LLM analysis for goal complexity
            is_complex = await self._is_complex_goal_semantically(goal_desc)
            if is_complex:
                return True
        
        # Suggest if the agent is struggling with efficiency (too many turns for simple tasks)
        if self.state.current_loop > 8 and not self.state.artifact_focus_mode:
            return True
        
        return False


    
    async def _analyze_error_context_with_llm(self, error_context: str, analysis_type: str) -> bool:
        """Use LLM to analyze error context"""
        
        # Cache to avoid repeated LLM calls
        context_hash = hash(f"{analysis_type}:{error_context[:200]}")
        if context_hash in self._error_context_cache:
            return self._caches['error_context'][context_hash]
        
        if analysis_type == "planning_related":
            analysis_prompt = f"""
Analyze this error context and rate how strongly it relates to planning, sequencing, or coordination issues.

Error Context: "{error_context}"

Rate the confidence that this is planning-related on a scale of 0-100:
- 0-20: Definitely not planning-related (technical, data, network errors)
- 21-40: Probably not planning-related but some workflow impact
- 41-60: Uncertain, could involve some coordination issues
- 61-80: Likely planning-related (sequencing, dependencies, strategy)
- 81-100: Definitely planning-related (plan validation, coordination)

Consider:
1. Plan validation or execution issues
2. Step sequencing or dependency problems
3. Workflow coordination or strategy failures
4. Prerequisite or ordering problems

Respond with just the numerical confidence score (0-100).
"""
        else:
            # Unknown analysis type - cache false and return
            self._caches['error_context'][context_hash] = False
            return False
            
        error_context_schema = {
            "type": "object",
            "properties": {
                "confidence_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Confidence score 0-100 for error context analysis"
                }
            },
            "required": ["confidence_score"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": analysis_prompt}],
                response_schema=error_context_schema,
                schema_name="error_context_analysis",
                use_cheap_model=True,
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                confidence_score = response.get("confidence_score", 0)
                if 0 <= confidence_score <= 100:
                    result = confidence_score >= 60  # 60+ threshold for planning-related
                    self._caches['error_context'][context_hash] = result
                    return result

        except Exception as e:
            self.logger.debug(f"LLM error context analysis failed: {e}")
        
        # Cache and return False on failure
        self._caches['error_context'][context_hash] = False
        return False
    

    

    
    async def _is_complex_goal_semantically(self, goal_description: str) -> bool:
        """Use LLM analysis to detect if a goal is complex and might benefit from guidance"""
        
        # Cache to avoid repeated LLM calls
        goal_hash = hash(f"complex_goal:{goal_description[:300]}")
        if goal_hash in self._complex_goal_cache:
            return self._caches['complex_goal'][goal_hash]
        
        complexity_prompt = f"""
Rate how complex this goal is for strategic guidance needs on a scale of 0-100.

Goal: "{goal_description}"

Complexity indicators:
- Multiple phases or distinct steps
- Coordination between different types of work  
- Research combined with creation tasks
- Comprehensive scope or depth requirements
- Dependencies and sequencing challenges

Rate complexity (0-100):
- 0-59: Simple to moderate, doesn't need strategic guidance
- 60-100: Complex, would benefit from multi-tool guidance and coordination

Respond with just the numerical score (0-100).
"""
        
        complexity_schema = {
            "type": "object",
            "properties": {
                "complexity_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Goal complexity score 0-100"
                }
            },
            "required": ["complexity_score"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": complexity_prompt}],
                response_schema=complexity_schema,
                schema_name="goal_complexity_analysis",
                use_cheap_model=True,
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                complexity_score = response.get("complexity_score", 0)
                if 0 <= complexity_score <= 100:
                    result = complexity_score >= 60  # 60+ threshold for complex
                    self._caches['complex_goal'][goal_hash] = result
                    return result
                    
        except Exception as e:
            self.logger.debug(f"LLM goal complexity analysis failed: {e}")
        
        # Cache and return False on failure
        self._caches['complex_goal'][goal_hash] = False
        return False
    

    

    
    async def _is_vague_plan_description_semantically(self, description: str) -> bool:
        """Use LLM analysis to detect if a plan description is too vague/short"""
        
        # Cache to avoid repeated LLM calls
        desc_hash = hash(f"vague_plan:{description}")
        if desc_hash in self._vague_plan_cache:
            return self._caches['vague_plan'][desc_hash]
        
        vagueness_prompt = f"""
Rate how vague and non-actionable this plan step is on a scale of 0-100.

Plan step: "{description}"

Vagueness indicators:
- Generic actions without specifics
- Unclear what exactly to do  
- No clear deliverable or outcome
- Ambiguous guidance that's hard to execute
- Non-specific verbs without details

Rate vagueness (0-100):
- 0-59: Specific enough to execute confidently
- 60-100: Too vague, needs clarification

Respond with just the numerical score (0-100).
"""
        
        vague_plan_schema = {
            "type": "object",
            "properties": {
                "vagueness_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Plan vagueness score 0-100"
                }
            },
            "required": ["vagueness_score"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": vagueness_prompt}],
                response_schema=vague_plan_schema,
                schema_name="vague_plan_analysis",
                use_cheap_model=True,
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                vagueness_score = response.get("vagueness_score", 0)
                if 0 <= vagueness_score <= 100:
                    result = vagueness_score >= 60  # 60+ threshold for vague
                    self._caches['vague_plan'][desc_hash] = result
                    return result
                    
        except Exception as e:
            self.logger.debug(f"LLM vague plan analysis failed: {e}")
        
        # Cache and return False on failure
        self._caches['vague_plan'][desc_hash] = False
        return False
    

    

    









    
    async def _analyze_workflow_type_with_llm(self, description: str, workflow_type: str) -> bool:
        """Use LLM to analyze workflow types"""
        
        # Cache to avoid repeated LLM calls
        context_hash = hash(f"{workflow_type}:{description[:300]}")
        if context_hash in self._workflow_analysis_cache:
            return self._caches['workflow_analysis'][context_hash]
        
        workflow_prompt = f"""
Rate how well this description matches the "{workflow_type}" workflow type on a scale of 0-100.

Description: "{description}"
Target workflow type: {workflow_type}

Workflow definitions:
- research_workflow: Primarily researching, investigating, studying, gathering information
- artifact_work: Creating, writing, generating, building, producing tangible outputs/deliverables  
- research_work: Searching, finding, investigating, gathering information, browsing for data

Rate match confidence (0-100):
- 0-69: Doesn't match this workflow type
- 70-100: Matches this workflow type well

Respond with just the numerical score (0-100).
"""
        
        workflow_schema = {
            "type": "object",
            "properties": {
                "confidence_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Workflow type match confidence 0-100"
                }
            },
            "required": ["confidence_score"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": workflow_prompt}],
                response_schema=workflow_schema,
                schema_name="workflow_type_analysis",
                use_cheap_model=True,
                max_tokens=20,
                temperature=0.1
            )
            
            if response:
                confidence_score = response.get("confidence_score", 0)
                if 0 <= confidence_score <= 100:
                    result = confidence_score >= 70  # 70+ threshold for match
                    self._caches['workflow_analysis'][context_hash] = result
                    return result
                    
        except Exception as e:
            self.logger.debug(f"LLM workflow analysis failed: {e}")
        
        # Cache and return False on failure
        self._caches['workflow_analysis'][context_hash] = False
        return False
    

    

    
    async def _execute_tool_call_internal(
        self,
        tool_name_mcp: str,
        arguments: Dict[str, Any],
        record_action: bool = True,
        planned_dependencies: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute a tool (UMS or agent-internal) and return the *standard envelope*:

            {
                "success": bool,
                "data": … | None,
                "error_type": str | None,
                "error_message": str | None,
                "status_code": int | None,
                "details": … | None,
            }

        The method

        • performs dependency checks, retry logic, action-start/finish bookkeeping  
        • routes the call through `MCPClient._execute_tool_and_parse_for_agent`  
        • updates agent-state counters, error tracking, `needs_replan`, etc.  
        • schedules background tasks for auto-linking / promotion when relevant  
        • calls `_handle_workflow_and_goal_side_effects` after completion.
        """

        # ──────────────────────────── helpers ─────────────────────────────
        def _force_print(*a, **kw):  # retained for live debugging (unchanged)
            print(*a, file=sys.stderr, flush=True, **kw)

        def _mk_envelope(
            *,
            success: bool = False,
            data: Any = None,
            err_type: Optional[str] = None,
            err_msg: Optional[str] = None,
            status: Optional[int] = None,
            details: Any = None,
        ) -> Dict[str, Any]:
            """Create a fully-populated envelope with sensible defaults."""
            return {
                "success": success,
                "data": data,
                "error_type": err_type,
                "error_message": err_msg,
                "status_code": status,
                "details": details,
            }

        async def _bail(
            envelope: Dict[str, Any],
            *,
            set_state_error: bool = True,
            mark_replan: bool = False,
            summary_prefix: str = "Failed",
        ) -> Dict[str, Any]:
            """
            Early-exit helper:
            • records `last_error_details` / `needs_replan` / `last_action_summary`
            • returns the envelope
            """
            if set_state_error:
                self.state.last_error_details = {
                    "tool": tool_name_mcp,
                    "args": arguments,
                    "error": envelope.get("error_message"),
                    "type": envelope.get("error_type"),
                    "status_code": envelope.get("status_code"),
                    "details": envelope.get("details"),
                }
            if mark_replan and not self.state.needs_replan:
                self.state.needs_replan = True

            # NOTE: summary recording kept identical to previous behaviour
            self.state.last_action_summary = (
                f"{tool_name_mcp} -> {summary_prefix} "
                f"({envelope.get('error_type')}): {envelope.get('error_message')}"
            )
            self.logger.error(f"AML_EXEC_TOOL_INTERNAL: returning envelope for {tool_name_mcp}: success={envelope.get('success')}, data_keys={str(envelope.get('data', {}).keys())[:200] if isinstance(envelope.get('data'), dict) else 'not_dict'}")
            return envelope



        current_base = self._get_base_function_name(tool_name_mcp)
        
        # Pre-execution loop detection check
        if self._is_in_loop(tool_name_mcp):
            raise InfiniteLoopError(f"Tool {tool_name_mcp} called too many times")
        
        self.logger.info(
            f"🔧 EXECUTING TOOL: {tool_name_mcp} | Base: {current_base} | "
            f"Args: {str(arguments)[:150]}{'...' if len(str(arguments)) > 150 else ''}"
        )

        envelope: Dict[str, Any] = _mk_envelope(
            err_type="UnknownInternalError_AML",
            err_msg="AML: Initial error before tool call attempt.",
        )

        # ─────────────────────── server lookup / availability ──────────────────────
        target_server = self._find_tool_server(tool_name_mcp)
        if not target_server and tool_name_mcp != AGENT_TOOL_UPDATE_PLAN:
            envelope.update(
                err_type="ServerUnavailable_AML",
                err_msg=f"Tool server unavailable for {tool_name_mcp}",
                status=503,
            )
            return await _bail(envelope, mark_replan=True)

        # ───────────────────────── special-case: update_plan ───────────────────────
        if tool_name_mcp == AGENT_TOOL_UPDATE_PLAN:
            # validate input
            plan_data = arguments.get("plan")
            if not isinstance(plan_data, list):
                envelope.update(
                    err_type="PlanUpdateError_AML",
                    err_msg="Invalid 'plan' argument (must be list of steps).",
                )
                return await _bail(envelope, mark_replan=True)

            try:
                validated_plan = [PlanStep(**step) for step in plan_data]
            except (ValidationError, TypeError) as exc:
                envelope.update(
                    err_type="PlanValidationError_AML",
                    err_msg=f"Plan validation error: {exc}",
                )
                return await _bail(envelope, mark_replan=True)

            if self._detect_plan_cycle(validated_plan):
                envelope.update(
                    err_type="PlanValidationError_AML",
                    err_msg="Plan cycle detected in proposed update.",
                )
                return await _bail(envelope, mark_replan=True)

            # happy-path: commit plan
            self.state.current_plan = validated_plan
            self.state.needs_replan = False
            self.state.last_error_details = None
            self.state.consecutive_error_count = 0
            msg = f"Plan updated with {len(validated_plan)} steps."
            self.logger.info("AML EXEC_TOOL_INTERNAL: %s", msg)

            envelope = _mk_envelope(success=True, data={"message": msg})
            self.state.last_action_summary = f"{tool_name_mcp} -> Success: {msg}"
            return envelope

        # ───────────────────────── arg auto-injection / coercion ────────────────────
        final_args = arguments.copy()
        wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        ctx_id = self.state.context_id
        goal_id = self.state.current_goal_id

        # Proactive auto-fixing for common argument issues
        if self._is_ums_tool(tool_name_mcp):
            # Fix memory_type if invalid
            if current_base == UMS_FUNC_STORE_MEMORY and "memory_type" in final_args:
                memory_type = final_args.get("memory_type", "").lower()
                valid_memory_types = {
                    "observation", "action_log", "tool_output", "artifact_creation", 
                    "reasoning_step", "fact", "insight", "plan", "question", "summary", 
                    "reflection", "skill", "procedure", "pattern", "code", "json", 
                    "url", "user_input", "text"
                }
                if memory_type not in valid_memory_types:
                    # Use semantic understanding to classify memory type
                    corrected_type = await self._classify_memory_type_semantically(
                        memory_type,
                        final_args.get("content", ""),
                        final_args.get("description", "")
                    )
                    if corrected_type:
                        final_args["memory_type"] = corrected_type
                        self.logger.info(f"🔧 Proactive semantic fix: memory_type '{memory_type}' → '{corrected_type}'")

            # Fix artifact_type if invalid
            if current_base == UMS_FUNC_RECORD_ARTIFACT and "artifact_type" in final_args:
                artifact_type = final_args.get("artifact_type", "").lower()
                valid_artifact_types = {"file", "text", "image", "table", "chart", "code", "data", "json", "url"}
                if artifact_type not in valid_artifact_types:
                    # Use semantic understanding to classify artifact type
                    corrected_type = await self._classify_artifact_type_semantically(
                        artifact_type,
                        final_args.get("name", ""),
                        final_args.get("description", ""),
                        final_args.get("content", "")
                    )
                    if corrected_type:
                        final_args["artifact_type"] = corrected_type
                        self.logger.info(f"🔧 Proactive semantic fix: artifact_type '{artifact_type}' → '{corrected_type}'")

        # Standard auto-injection logic
        if (
            wf_id
            and "workflow_id" not in final_args
            and current_base not in [UMS_FUNC_CREATE_WORKFLOW, UMS_FUNC_LIST_WORKFLOWS]
        ):
            final_args["workflow_id"] = wf_id

        # Ensure workflow_id is consistent if already present
        if "workflow_id" in final_args and wf_id and final_args["workflow_id"] != wf_id:
            self.logger.warning(f"🔧 Workflow ID mismatch corrected: '{final_args['workflow_id']}' → '{wf_id}'")
            final_args["workflow_id"] = wf_id

        if (
            ctx_id
            and current_base
            in {
                UMS_FUNC_GET_WORKING_MEMORY,
                UMS_FUNC_OPTIMIZE_WM,
                UMS_FUNC_AUTO_FOCUS,
                UMS_FUNC_FOCUS_MEMORY,
                UMS_FUNC_SAVE_COGNITIVE_STATE,
                UMS_FUNC_LOAD_COGNITIVE_STATE,
                UMS_FUNC_GET_RICH_CONTEXT_PACKAGE,
            }
            and "context_id" not in final_args
        ):
            final_args["context_id"] = ctx_id
            if current_base == UMS_FUNC_LOAD_COGNITIVE_STATE and "state_id" not in final_args:
                final_args["state_id"] = ctx_id

        if (
            self.state.current_thought_chain_id
            and current_base == UMS_FUNC_RECORD_THOUGHT
            and "thought_chain_id" not in final_args
        ):
            final_args["thought_chain_id"] = self.state.current_thought_chain_id

        if (
            goal_id
            and current_base == UMS_FUNC_CREATE_GOAL
            and "parent_goal_id" not in final_args
        ):
            final_args["parent_goal_id"] = goal_id

        _force_print(
            f"AML_EXEC_TOOL_INTERNAL: final args for '{tool_name_mcp}': "
            f"{str(final_args)[:200]}…"
        )

        # ───────────────────────────── pre-validate UMS tool arguments ─────────────────────
        if self._is_ums_tool(tool_name_mcp):
            validation_result = self._validate_ums_tool_arguments(tool_name_mcp, final_args)
            if not validation_result["valid"]:
                envelope = _mk_envelope(
                    success=False,
                    err_type="ToolInputValidationError_AML",
                    err_msg=f"Schema validation failed for {tool_name_mcp}: {validation_result['error']}",
                    details={"validation_errors": validation_result.get("details", {})}
                )
                return await _bail(envelope, mark_replan=True, summary_prefix="Schema validation failed")

        # ───────────────────────────── prevent duplicate artifacts ─────────────────────
        if current_base == UMS_FUNC_RECORD_ARTIFACT and record_action:
            artifact_name = final_args.get("name", "")
            artifact_description = final_args.get("description", "")
            
            if artifact_name:
                similar_artifacts = await self._check_for_similar_artifacts(artifact_name, artifact_description)
                
                if similar_artifacts:
                    # Found similar artifacts - actually prevent creation
                    similar_names = [mem.get("content", "")[:100] for mem in similar_artifacts[:3]]
                    error_msg = (f"Duplicate artifact blocked: Found {len(similar_artifacts)} similar artifacts. "
                               f"Similar artifacts: {similar_names}")
                    self.logger.warning(error_msg)
                    
                    envelope = _mk_envelope(
                        success=False,
                        err_type="DuplicateArtifactPrevention",
                        err_msg=error_msg,
                        details={
                            "similar_count": len(similar_artifacts),
                            "similar_artifacts": similar_names
                        }
                    )
                    return await _bail(envelope, set_state_error=True, summary_prefix="Duplicate blocked")

        # ───────────────────────────── intelligent memory duplication prevention ─────────────────────
        if current_base == UMS_FUNC_STORE_MEMORY and record_action:
            memory_content = final_args.get("content", "")
            memory_type = final_args.get("memory_type", "")
            memory_description = final_args.get("description", "")
            
            if memory_content:
                # Check if we should store this memory using intelligent analysis
                storage_decision = await self.memory_manager.should_store_memory(
                    memory_content, memory_type, memory_description
                )
                
                if not storage_decision["should_store"]:
                    # Block duplicate memory storage
                    reason = storage_decision["reason"]
                    recommendation = storage_decision.get("recommendation", "")
                    similar_count = storage_decision.get("similar_count", 0)
                    
                    self.logger.warning(f"🚫 MEMORY DUPLICATION PREVENTED: {reason}")
                    
                    envelope = _mk_envelope(
                        success=False,
                        err_type="DuplicateMemoryPrevention",
                        err_msg=f"Memory storage blocked to prevent duplication: {reason}. {recommendation}",
                        details={
                            "similar_count": similar_count,
                            "action": storage_decision.get("action", "skip_duplicate"),
                            "recommendation": recommendation,
                            "content_preview": memory_content[:100] + "..." if len(memory_content) > 100 else memory_content
                        }
                    )
                    
                    return await _bail(
                        envelope,
                        set_state_error=True,
                        summary_prefix="Duplicate memory blocked"
                    )
                
                elif storage_decision.get("action") == "store_with_note":
                    # Enhance content with reference to similar memory
                    similar_count = storage_decision.get("similar_count", 0)
                    final_args["content"] = f"{memory_content}\n\n[Note: {similar_count} similar memory exists - this adds new information]"
                    self.logger.info(f"📝 ENHANCED MEMORY: Added context note about {similar_count} similar memory")

        # ───────────────────── prerequisite dependency check ───────────────────────
        if planned_dependencies:
            ok, reason = await self._check_prerequisites(planned_dependencies)
            if not ok:
                envelope.update(
                    err_type="DependencyNotMetError_AML",
                    err_msg=f"Prerequisites not met: {reason}",
                    status=412,
                    details={"dependencies": planned_dependencies},
                )
                return await _bail(envelope, mark_replan=True)

        # ───────────────────────────── action start log ────────────────────────────
        should_record = (
            record_action
            and current_base not in self._INTERNAL_OR_META_TOOLS_BASE_NAMES
        )
        action_id: Optional[str] = None
        if should_record:
            action_id = await self._record_action_start_internal(
                tool_name_mcp, final_args, planned_dependencies
            )
            if not action_id:
                envelope.update(
                    err_type="UMSError_AML",
                    err_msg=f"Failed to record action_start for {tool_name_mcp}.",
                    status=500,
                    details={"reason": "ActionStartFailed"},
                )
                return await _bail(envelope, mark_replan=True)

        # ────────────────────────── retry-wrapped tool call ─────────────────────────
        idempotent = current_base in {
            UMS_FUNC_GET_MEMORY_BY_ID,
            UMS_FUNC_SEARCH_SEMANTIC_MEMORIES,
            UMS_FUNC_HYBRID_SEARCH,
            UMS_FUNC_QUERY_MEMORIES,
            UMS_FUNC_GET_ACTION_DETAILS,
            UMS_FUNC_LIST_WORKFLOWS,
            UMS_FUNC_COMPUTE_STATS,
            UMS_FUNC_GET_WORKING_MEMORY,
            UMS_FUNC_GET_LINKED_MEMORIES,
            UMS_FUNC_GET_ARTIFACTS,
            UMS_FUNC_GET_ARTIFACT_BY_ID,
            UMS_FUNC_GET_ACTION_DEPENDENCIES,
            UMS_FUNC_GET_THOUGHT_CHAIN,
            UMS_FUNC_GET_WORKFLOW_DETAILS,
            UMS_FUNC_GET_GOAL_DETAILS,
            UMS_FUNC_SUMMARIZE_TEXT,
            UMS_FUNC_SUMMARIZE_CONTEXT_BLOCK,
            UMS_FUNC_GET_RICH_CONTEXT_PACKAGE,
            UMS_FUNC_GET_RECENT_ACTIONS,
            UMS_FUNC_VISUALIZE_REASONING_CHAIN,
            UMS_FUNC_VISUALIZE_MEMORY_NETWORK,
            UMS_FUNC_GENERATE_WORKFLOW_REPORT,
            UMS_FUNC_LOAD_COGNITIVE_STATE,
            UMS_FUNC_GET_MULTI_TOOL_GUIDANCE,
            UMS_FUNC_DIAGNOSE_FILE_ACCESS,
        }

        stats = self.state.tool_usage_stats.setdefault(
            tool_name_mcp, {"success": 0, "failure": 0, "latency_ms_total": 0.0}
        )
        start_ts = time.time()

        async def _run_call():
            clean_args = {k: v for k, v in final_args.items() if v is not None}
            return await self.mcp_client._execute_tool_and_parse_for_agent(
                target_server, tool_name_mcp, clean_args
            )

        try:
            result = await self._with_retries(
                _run_call, max_retries=3 if idempotent else 1
            )
        except asyncio.CancelledError:
            envelope.update(
                err_type="CancelledError_AML",
                err_msg="Tool execution cancelled by AML.",
            )
            if action_id and should_record:
                await self._record_action_completion_internal(
                    action_id,
                    _mk_envelope(
                        success=False,
                        err_type="CancelledError_AML",
                        err_msg="Operation cancelled by AML",
                        details={"reason": "AML cancellation"},
                    ),
                )
            raise
        except Exception as exc:
            envelope.update(
                err_type="AMLRetryOrWrapperError",
                err_msg=f"Unexpected error during tool call: {exc}",
            )
            result = envelope  # ensure downstream code has dict

        # result is guaranteed dict at this point
        if not isinstance(result, dict):
            result = _mk_envelope(
                err_type="MCPClientContractError_AML",
                err_msg="MCPClient wrapper returned non-dict result.",
            )

        envelope = result  # definitive envelope

        # ─────────────────────────── handle errors clearly ──────────────────────────────
        if not envelope.get("success"):
            # Don't try to auto-fix - just handle the error clearly
            envelope = self._handle_tool_error(envelope)

        # ─────────────────────────── stats + counters ──────────────────────────────
        stats["latency_ms_total"] += (time.time() - start_ts) * 1000.0
        if envelope.get("success"):
            stats["success"] += 1
            # Update focus mode detection based on tool execution
            await self._detect_and_update_focus_mode(tool_name_mcp, True)
            if (
                self.state.last_error_details
                and self.state.last_error_details.get("tool") == tool_name_mcp
            ):
                self.state.last_error_details = None
            # Track tool usage for pattern learning
            base_tool = self._get_base_function_name(tool_name_mcp)
            if (base_tool not in self._INTERNAL_OR_META_TOOLS_BASE_NAMES and 
                tool_name_mcp != AGENT_TOOL_UPDATE_PLAN):
                self.state.last_workflow_tools.append(tool_name_mcp)
                # Keep last 10 tools only
                if len(self.state.last_workflow_tools) > 10:
                    self.state.last_workflow_tools = self.state.last_workflow_tools[-10:]
                                    
        else:
            stats["failure"] += 1
            # Update focus mode detection for failed tools
            await self._detect_and_update_focus_mode(tool_name_mcp, False)

        # ─────────────────────── minimal background processing ────────────────────────
        if envelope.get("success"):
            # Only do essential background tasks - no complex proliferation
            data = envelope.get("data") or {}
            
            # Simple logging of successful operations
            if current_base == UMS_FUNC_RECORD_ARTIFACT:
                self.logger.info(f"✅ Artifact recorded: {data.get('artifact_id', 'unknown')}")
            elif current_base == UMS_FUNC_STORE_MEMORY:
                self.logger.info(f"✅ Memory stored: {data.get('memory_id', 'unknown')}")

        # ─────────────── propagate failure to agent-level error state ──────────────
        if not envelope.get("success"):
            mapped_type = envelope.get("error_type") or "ToolExecutionError_AML"
            # canonicalise for downstream heuristics
            if mapped_type in {
                "UMSToolReportedFailure",
                "UMSError_AML",
                "UMSError",
                "ContentParsingError",
                "MCPClientContractError_AML",
                "AMLRetryOrWrapperError",
            }:
                mapped_type = "UMSError"
            elif mapped_type == "ToolMaxRetriesOrServerError":
                mapped_type = "ServerUnavailable"
            elif mapped_type.startswith("AML"):
                mapped_type = "AgentError"

            self.state.last_error_details = {
                "tool": tool_name_mcp,
                "args": arguments,
                "error": envelope.get("error_message"),
                "status_code": envelope.get("status_code"),
                "type": mapped_type,
                "details": envelope.get("details"),
            }
            if not idempotent and not self.state.needs_replan:
                self.state.needs_replan = True
                self.logger.info(
                    "AML EXEC_TOOL_INTERNAL: needs_replan=True (non-idempotent failure %s)",
                    tool_name_mcp,
                )

        # ───────────────────────- human-readable action summary ────────────────────
        if envelope.get("success"):
            data = envelope.get("data")
            summary = "Success"
            if isinstance(data, dict):
                for k in (
                    "summary",
                    "message",
                    "memory_id",
                    "action_id",
                    "artifact_id",
                    "link_id",
                    "thought_chain_id",
                    "thought_id",
                    "state_id",
                    "report",
                    "visualization",
                    "goal_id",
                    "workflow_id",
                    "title",
                ):
                    if k in data and data[k] is not None:
                        v = str(data[k])
                        summary = f"{k}: {_fmt_id(v) if 'id' in k.lower() else v}"
                        break
            self.state.last_action_summary = f"{tool_name_mcp} -> {summary}"
        else:
            msg = f"Failed ({envelope.get('error_type')}): {envelope.get('error_message')}"
            if envelope.get("status_code") is not None:
                msg += f" (Code: {envelope['status_code']})"
            self.state.last_action_summary = f"{tool_name_mcp} -> {msg}"

        execution_time_ms = (time.time() - start_ts) * 1000
        
        # Extract meaningful result snippet for debugging
        result_preview = "No result"
        if envelope.get("success"):
            data = envelope.get("data", {})
            if isinstance(data, dict):
                # Show key identifiers or messages
                for key in ["memory_id", "artifact_id", "goal_id", "action_id", "thought_id", "message", "summary"]:
                    if data.get(key):
                        result_preview = f"{key}: {str(data[key])[:100]}"
                        break
                else:
                    # Show truncated data if no key identifiers
                    result_preview = str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
            else:
                result_preview = str(data)[:100] + "..." if len(str(data)) > 100 else str(data)
        else:
            error_msg = envelope.get("error_message", envelope.get("error", "Unknown error"))
            result_preview = f"Error: {str(error_msg)[:100]}"
        
        success_indicator = "✅" if envelope.get("success") else "❌"
        self.logger.info(
            f"{success_indicator} TOOL COMPLETED: {tool_name_mcp} | "
            f"Time: {execution_time_ms:.1f}ms | "
            f"Result: {result_preview}"
        )

        # ────────────────────────── record action completion ───────────────────────
        if action_id and should_record:
            try:
                await self._record_action_completion_internal(action_id, envelope)
            except Exception as exc:
                self.logger.error(
                    "Failed to record completion for action %s: %s", action_id, exc, exc_info=True
                )

        # ───────────────────────── workflow / goal side-effects ────────────────────
        await self._handle_workflow_and_goal_side_effects(current_base, final_args, envelope)

        _force_print(
            f"AML_EXEC_TOOL_INTERNAL: returning envelope "
            f"success={envelope.get('success')} err={envelope.get('error_type')}"
        )
        return envelope


    async def _run_periodic_tasks(self) -> None:  # noqa: C901 – complex but contained
        """Schedule and execute housekeeping / meta-cognitive UMS calls.
        """
        # ------------------------------------------------------------------ guard
        if (
            not self.state.workflow_id
            or not self.state.context_id  # context_id might be None if just workflow created
            or self._shutdown_event.is_set()
        ):
            self.logger.debug(
                "Skipping periodic tasks: inactive workflow/context or shutdown."
            )
            return

        log_prefix = f"AML_PERIODIC (WF:{_fmt_id(self.state.workflow_id)}, Loop:{self.state.current_loop})"
        self.logger.debug("%s: Evaluating periodic tasks.", log_prefix)

        # ----------------------------------------------------- helper: add_task()
        tasks_to_run: List[Tuple[str, Dict[str, any]]] = []
        trigger_reasons: List[str] = []

        def add_task(tool_name: str, args: Dict[str, Any], reason: str) -> None:
            """Idempotently queue a task and record its trigger reason."""
            if any(t[0] == tool_name for t in tasks_to_run):
                return  # already queued
            tasks_to_run.append((tool_name, args))
            trigger_reasons.append(reason)

        # ---------------------------------------------------- tool registry table
        # Each entry: (base_ums_function, should_check_workflow_id, should_check_ctx_id)
        # For periodic tasks, workflow_id is almost always needed for context,
        # context_id is needed for context-specific operations like optimize_wm.
        tool_registry: Dict[str, Tuple[str, bool, bool]] = {
            "reflection"      : (UMS_FUNC_REFLECTION,       True,  False),
            "consolidation"   : (UMS_FUNC_CONSOLIDATION,    True,  False),
            "opt_wm"          : (UMS_FUNC_OPTIMIZE_WM,      False, True ), # Requires context_id
            "auto_focus"      : (UMS_FUNC_AUTO_FOCUS,       False, True ), # Requires context_id
            "query_mem"       : (UMS_FUNC_QUERY_MEMORIES,   True,  False), # For promotion checks
            "stats"           : (UMS_FUNC_COMPUTE_STATS,    True,  False),
            "maintenance"     : (UMS_FUNC_DELETE_EXPIRED_MEMORIES, False, False), # Global or UMS handles scope
        }

        # Map base-name → full MCP name and availability
        tool_info: Dict[str, Tuple[str, bool]] = {}
        for alias, (base_fn, *_rest) in tool_registry.items():
            full_mcp_name = self._get_ums_tool_mcp_name(base_fn)
            tool_info[alias] = (full_mcp_name, self._find_tool_server(full_mcp_name) is not None)

        # ---------------------------------------------------------------- stats +
        self.state.loops_since_stats_adaptation += 1
        try:
            if self.state.loops_since_stats_adaptation >= STATS_ADAPTATION_INTERVAL:
                full_stats_tool_name, stats_tool_available = tool_info["stats"]
                if not stats_tool_available:
                    self.logger.warning("%s: Stats tool '%s' unavailable.", log_prefix, full_stats_tool_name)
                elif not self.state.workflow_id: # Guard against stats call without workflow_id
                    self.logger.warning("%s: Skipping stats collection - no active workflow_id.", log_prefix)
                else:
                    self.logger.debug("%s: Triggering compute_memory_statistics for workflow %s.", log_prefix, self.state.workflow_id)
                    stats_result_envelope = {} # Default to empty dict
                    with contextlib.suppress(asyncio.CancelledError): # Allow cancellation
                        stats_result_envelope = await asyncio.wait_for(
                            self._execute_tool_call_internal(
                                full_stats_tool_name,
                                {"workflow_id": self.state.workflow_id}, # Explicitly pass workflow_id
                                record_action=False,
                            ),
                            timeout=_TASK_TIMEOUT_SEC,
                        )
                    
                    # UMS tools return payload directly in envelope if not nested under 'data'
                    # For compute_memory_statistics, the stats are the direct payload.
                    if stats_result_envelope.get("success"):
                        # The stats_result_envelope *is* the UMS payload here.
                        await self._adapt_thresholds(stats_result_envelope)
                        episodic_cnt = stats_result_envelope.get("by_level", {}).get(
                            MemoryLevel.EPISODIC.value, 0
                        )
                        # Queue consolidation if episodic backlog is huge
                        full_consolidation_tool_name, consolidation_tool_available = tool_info["consolidation"]
                        if (
                            episodic_cnt
                            > (self.state.current_consolidation_threshold * 2.0)
                            and consolidation_tool_available
                            and self.state.workflow_id # Ensure workflow_id for consolidation
                        ):
                            add_task(
                                full_consolidation_tool_name,
                                {
                                    "workflow_id": self.state.workflow_id, # Pass workflow_id
                                    "consolidation_type": "summary",
                                    "query_filter": {"memory_level": MemoryLevel.EPISODIC.value},
                                    "max_source_memories": self.consolidation_max_sources,
                                },
                                f"HighEpisodic({episodic_cnt})",
                            )
                    else:
                        self.logger.warning(
                            "%s: compute_memory_statistics call failed: %s",
                            log_prefix,
                            stats_result_envelope.get("error_message") or stats_result_envelope.get("error"),
                        )
        except Exception as e_stats:
             self.logger.error(f"{log_prefix}: Error during stats adaptation block: {e_stats}", exc_info=True)
        finally:
            # guarantee counter reset even if an exception bubbles
            self.state.loops_since_stats_adaptation = 0

        # ------------------------------------------------------- reflection need
        # Use effective thresholds that account for focus mode
        effective_reflection_threshold, effective_consolidation_threshold = await self._get_effective_thresholds()
        
        needs_reflection = (
            self.state.needs_replan # Agent explicitly flagged for replan
            or self.state.successful_actions_since_reflection >= effective_reflection_threshold
        )
        
        # Skip reflection during focus mode unless explicitly needed for replan
        if self.state.artifact_focus_mode and not self.state.needs_replan:
            needs_reflection = False
            self.logger.debug(f"{log_prefix}: Skipping reflection during focus mode (threshold: {effective_reflection_threshold})")
        full_reflection_tool_name, reflection_tool_available = tool_info["reflection"]
        if needs_reflection and reflection_tool_available:
            if not self.state.workflow_id:
                self.logger.warning("%s: Skipping reflection - no active workflow_id.", log_prefix)
            else:
                reflection_type = self.reflection_type_sequence[
                    self.state.reflection_cycle_index % len(self.reflection_type_sequence)
                ]
                add_task(
                    full_reflection_tool_name,
                    {"workflow_id": self.state.workflow_id, "reflection_type": reflection_type},
                    "Replan" if self.state.needs_replan else "ReflectionThreshold",
                )
                self.state.successful_actions_since_reflection = 0 # Reset counter after scheduling
                self.state.reflection_cycle_index += 1
        elif needs_reflection and not reflection_tool_available:
            self.logger.warning("%s: Reflection tool '%s' unavailable. Resetting reflection counter.", log_prefix, full_reflection_tool_name)
            self.state.successful_actions_since_reflection = 0 # Still reset counter

        # ---------------------------------------------------- consolidation need
        full_consolidation_tool_name, consolidation_tool_available = tool_info["consolidation"]
        needs_consolidation = (
            self.state.successful_actions_since_consolidation >= effective_consolidation_threshold
        )
        
        # Skip consolidation during focus mode
        if self.state.artifact_focus_mode:
            needs_consolidation = False
            self.logger.debug(f"{log_prefix}: Skipping consolidation during focus mode (threshold: {effective_consolidation_threshold})")
        
        if needs_consolidation:
            if consolidation_tool_available:
                if not self.state.workflow_id:
                     self.logger.warning("%s: Skipping consolidation - no active workflow_id.", log_prefix)
                else:
                    add_task(
                        full_consolidation_tool_name,
                        {
                            "workflow_id": self.state.workflow_id, # Pass workflow_id
                            "consolidation_type": "summary",
                            "query_filter": {"memory_level": self.consolidation_memory_level},
                            "max_source_memories": self.consolidation_max_sources,
                        },
                        "ConsolidationThreshold",
                    )
                self.state.successful_actions_since_consolidation = 0 # Reset counter after scheduling
            else:
                self.logger.warning("%s: Consolidation tool '%s' unavailable. Resetting consolidation counter.", log_prefix, full_consolidation_tool_name)
                self.state.successful_actions_since_consolidation = 0 # Still reset

        # --------------------------------------------------- optimisation / focus
        self.state.loops_since_optimization += 1
        full_opt_wm_tool_name, opt_wm_tool_available = tool_info["opt_wm"]
        full_auto_focus_tool_name, auto_focus_tool_available = tool_info["auto_focus"]

        # Skip optimization during focus mode to avoid interrupting productive work
        should_optimize = (
            self.state.loops_since_optimization >= OPTIMIZATION_LOOP_INTERVAL 
            and not self.state.artifact_focus_mode
        )
        
        if should_optimize:
            self.state.loops_since_optimization = 0 # Reset counter
            if not self.state.context_id: # Guard for context_id specific tools
                self.logger.warning("%s: Skipping optimize_wm and auto_focus - no active context_id.", log_prefix)
            else:
                if opt_wm_tool_available:
                    add_task(
                        full_opt_wm_tool_name,
                        {"context_id": self.state.context_id}, # Pass context_id
                        "OptimizeInterval",
                    )
                else:
                    self.logger.warning("%s: optimize_working_memory tool '%s' unavailable.", log_prefix, full_opt_wm_tool_name)

                if auto_focus_tool_available:
                    add_task(
                        full_auto_focus_tool_name,
                        {"context_id": self.state.context_id}, # Pass context_id
                        "FocusUpdateInterval",
                    )
                else:
                    self.logger.warning("%s: auto_focus tool '%s' unavailable.", log_prefix, full_auto_focus_tool_name)

        # -------------------------------------------------------- promotion check
        self.state.loops_since_promotion_check += 1
        full_query_mem_tool_name, query_mem_tool_available = tool_info["query_mem"]
        
        # Skip promotion checks during focus mode
        should_check_promotions = (
            self.state.loops_since_promotion_check >= MEMORY_PROMOTION_LOOP_INTERVAL
            and not self.state.artifact_focus_mode
        )
        
        if should_check_promotions:
            self.state.loops_since_promotion_check = 0 # Reset counter
            if query_mem_tool_available: # query_memories is used by the internal helper
                if not self.state.workflow_id:
                    self.logger.warning("%s: Skipping promotion check - no active workflow_id for query_memories.", log_prefix)
                else:
                    add_task(
                        _INTERNAL_PROMO_SENTINEL, # Internal sentinel for _trigger_promotion_checks
                        {"workflow_id": self.state.workflow_id}, # Pass workflow_id
                        "PromotionInterval",
                    )
            else:
                self.logger.warning("%s: query_memories tool '%s' (needed for promotion) unavailable.", log_prefix, full_query_mem_tool_name)

        # ----------------------------------------------------------- maintenance
        self.state.loops_since_maintenance += 1
        full_maintenance_tool_name, maintenance_tool_available = tool_info["maintenance"]
        if self.state.loops_since_maintenance >= MAINTENANCE_INTERVAL:
            self.state.loops_since_maintenance = 0 # Reset counter
            if maintenance_tool_available:
                add_task(
                    full_maintenance_tool_name,
                    {"db_path": None}, # UMS uses its own default db_path for this
                    "MaintenanceInterval",
                )
            else:
                self.logger.warning(
                    "%s: delete_expired_memories tool '%s' unavailable.", log_prefix, full_maintenance_tool_name
                )

        # --------------------------------------------------- nothing to execute?
        if not tasks_to_run:
            self.logger.debug("%s: No periodic tasks scheduled this cycle.", log_prefix)
            return

        # -------------------------------------------------------- execute queue
        # stable ordering by priority
        tasks_to_run.sort(
            key=lambda t: _TASK_PRIORITY.get(self._get_base_function_name(t[0]), 99)
        )

        unique_reasons = ", ".join(sorted(set(trigger_reasons)))
        self.logger.info(
            "🧠 %s: Running %d periodic tasks (Triggers: %s).",
            log_prefix,
            len(tasks_to_run),
            unique_reasons,
        )

        # sequential execution keeps side-effects predictable
        for tool_full_name_mcp, tool_args_for_call in tasks_to_run:
            if self._shutdown_event.is_set():
                self.logger.info(
                    "%s: Shutdown signaled during periodic execution. Stopping further tasks.",
                    log_prefix,
                )
                break

            current_base_fn = self._get_base_function_name(tool_full_name_mcp)

            if tool_full_name_mcp == _INTERNAL_PROMO_SENTINEL:
                self.logger.debug("%s: Running promotion-check helper with args %s.", log_prefix, tool_args_for_call)
                # _trigger_promotion_checks expects workflow_id directly.
                wf_id_for_promo = tool_args_for_call.get("workflow_id", self.state.workflow_id)
                if wf_id_for_promo: # Ensure workflow_id is present
                    await self._trigger_promotion_checks() # This helper uses self.state.workflow_id internally.
                else:
                    self.logger.warning(f"{log_prefix}: Skipping promotion check as workflow_id is missing for _INTERNAL_PROMO_SENTINEL.")
                continue

            # auto-inject workflow / context ids where required by tool_registry mapping
            # This is a simplified check; specific tools manage their own required args more robustly.
            # The primary purpose here is to ensure workflow_id and context_id are passed IF NEEDED for THIS CALL.
            _tool_alias_for_registry = next((alias for alias, (base, _, _) in tool_registry.items() if base == current_base_fn), None)
            if _tool_alias_for_registry:
                _, needs_wf, needs_ctx = tool_registry[_tool_alias_for_registry]
                if needs_wf and "workflow_id" not in tool_args_for_call and self.state.workflow_id:
                    tool_args_for_call["workflow_id"] = self.state.workflow_id
                if needs_ctx and "context_id" not in tool_args_for_call and self.state.context_id:
                    tool_args_for_call["context_id"] = self.state.context_id
            
            # Final check for critical IDs for tools that absolutely need them
            if current_base_fn in [UMS_FUNC_REFLECTION, UMS_FUNC_CONSOLIDATION, UMS_FUNC_QUERY_MEMORIES] and not tool_args_for_call.get("workflow_id"):
                if self.state.workflow_id:
                    tool_args_for_call["workflow_id"] = self.state.workflow_id
                else:
                    self.logger.warning(f"{log_prefix}: Skipping {current_base_fn} as workflow_id is missing and couldn't be auto-injected.")
                    continue
            if current_base_fn in [UMS_FUNC_OPTIMIZE_WM, UMS_FUNC_AUTO_FOCUS] and not tool_args_for_call.get("context_id"):
                if self.state.context_id:
                    tool_args_for_call["context_id"] = self.state.context_id
                else:
                    self.logger.warning(f"{log_prefix}: Skipping {current_base_fn} as context_id is missing and couldn't be auto-injected.")
                    continue


            self.logger.debug(
                "%s: Executing periodic tool %s with args %s",
                log_prefix,
                tool_full_name_mcp,
                tool_args_for_call,
            )
            
            result_envelope = {} # Default to empty dict
            try:
                with contextlib.suppress(asyncio.CancelledError): # Allow cancellation
                    result_envelope = await asyncio.wait_for(
                        self._execute_tool_call_internal(
                            tool_full_name_mcp, tool_args_for_call, record_action=False
                        ),
                        timeout=_TASK_TIMEOUT_SEC,
                    )
            except asyncio.TimeoutError:
                self.logger.error(
                    "%s: Tool %s timed out after %ss.",
                    log_prefix,
                    tool_full_name_mcp,
                    _TASK_TIMEOUT_SEC,
                )
                # treat timeout as failure
                result_envelope = {"success": False, "error_message": "Timeout", "error_type": "PeriodicTaskTimeout"}
            except Exception as e_task_exec:
                self.logger.error(f"{log_prefix}: Unhandled error executing periodic tool {tool_full_name_mcp}: {e_task_exec}", exc_info=True)
                result_envelope = {"success": False, "error_message": str(e_task_exec), "error_type": "PeriodicTaskException"}


            # --------------- result handling
            if not result_envelope.get("success"):
                self.logger.warning(
                    "%s: Periodic tool %s failed: %s",
                    log_prefix,
                    tool_full_name_mcp,
                    result_envelope.get("error_message") or result_envelope.get("error"),
                )
                # For critical context-dependent tools, set error if they fail, so LLM can see it.
                if current_base_fn in [UMS_FUNC_OPTIMIZE_WM, UMS_FUNC_AUTO_FOCUS] and not self.state.last_error_details:
                    self.state.last_error_details = {
                        "tool": tool_full_name_mcp,
                        "error": result_envelope.get("error_message") or result_envelope.get("error", "Periodic task failed"),
                        "type": result_envelope.get("error_type", "PeriodicTaskToolError"),
                        "details": {"args_used": tool_args_for_call}
                    }
                continue  # next task in queue

            # reflection / consolidation meta-feedback
            if current_base_fn in [UMS_FUNC_REFLECTION, UMS_FUNC_CONSOLIDATION]:
                # UMS tools return payload directly, not nested under 'data' if this wrapper is used.
                feedback = (
                    result_envelope.get("content") # UMS generate_reflection returns content directly
                    if current_base_fn == UMS_FUNC_REFLECTION
                    else result_envelope.get("consolidated_content") # UMS consolidate_memories returns consolidated_content directly
                ) or ""
                if feedback:
                    headline = feedback.split("\n", 1)[0][:150]
                    self.state.last_meta_feedback = (
                        f"Feedback from UMS {current_base_fn}: {headline}..."
                    )
                    self.logger.info(
                        "%s: Meta-feedback captured: %s", log_prefix, self.state.last_meta_feedback
                    )
                    
                    # Revised logic for setting needs_replan
                    reflection_type_arg = tool_args_for_call.get("reflection_type", "summary") # Default if not specified
                    
                    # Case 1: Reflection type is explicitly 'plan'
                    if current_base_fn == UMS_FUNC_REFLECTION and reflection_type_arg == "plan":
                        if not self.state.needs_replan: # Only log if it's a change
                            self.state.needs_replan = True
                            self.logger.info(f"{log_prefix}: Setting needs_replan=True due to 'plan' reflection type.")
                    # Case 2: Use semantic analysis to detect replan needs
                    elif self._analyze_feedback_for_replan_semantically(feedback):
                        if not self.state.needs_replan: # Only log if it's a change
                            self.state.needs_replan = True
                            self.logger.info(f"{log_prefix}: Setting needs_replan=True due to semantic feedback analysis.")
                    # Case 3: Consolidation might implicitly suggest replan if it fundamentally changes understanding
                    elif current_base_fn == UMS_FUNC_CONSOLIDATION:
                        # For now, let's not automatically set needs_replan from consolidation unless feedback explicitly states it
                        # or unless a specific consolidation_type implies it (e.g., a 'planning_consolidation' type if you add one)
                        self.logger.debug(f"{log_prefix}: Consolidation occurred. Needs_replan remains: {self.state.needs_replan}. Feedback captured for LLM review.")
                    # else: self.state.needs_replan remains unchanged by this feedback.

                else:
                    self.logger.debug(
                        "%s: %s succeeded but returned no feedback content.",
                        log_prefix,
                        current_base_fn,
                    )

            # small breather to avoid thundering-herd on the tool server
            await asyncio.sleep(0.05)


    async def _check_and_trigger_promotion(
        self,
        memory_id: str,
        *,
        workflow_id: Optional[str],
        context_id: Optional[str],
    ) -> None:
        """
        Background coroutine that asks UMS to promote `memory_id` if eligible.

        Defensive improvements
        ----------------------
        • Guards against workflow switch & shutdown early.
        • Validates `memory_id` eagerly.
        • Adds jitter + explicit timeout to avoid thundering-herd and hung awaits.
        • Catches `CancelledError` separately for cooperative cancellation.
        • Structured debug paths to keep noisy logs down.
        """
        # ------------------------------------------------------------------ early guards
        if self._shutdown_event.is_set():
            return

        if not (isinstance(memory_id, str) and memory_id.strip()):
            self.logger.debug("Promo-check aborted: empty/invalid memory_id.")
            return

        if workflow_id != self.state.workflow_id:
            # Workflow changed while task was queued.
            self.logger.debug(
                f"Promo-check skipped for {_fmt_id(memory_id)}: "
                f"workflow switched from {_fmt_id(workflow_id)} ➜ {_fmt_id(self.state.workflow_id)}."
            )
            return

        promote_mcp_name = self._get_ums_tool_mcp_name(_PROMO_TOOL_NAME)
        if not self._find_tool_server(promote_mcp_name):
            self.logger.debug(
                f"Promo-check skipped for {_fmt_id(memory_id)}: "
                f"UMS tool '{_PROMO_TOOL_NAME}' unavailable."
            )
            return

        # ------------------------------------------------------------------ jittered start
        await asyncio.sleep(random.uniform(*_PROMO_JITTER_RANGE))
        if self._shutdown_event.is_set():
            return

        self.logger.debug(
            f"Checking promotion eligibility for memory {_fmt_id(memory_id)} "
            f"(WF {_fmt_id(workflow_id)}, CTX {_fmt_id(context_id)})..."
        )

        promo_args = {"memory_id": memory_id}

        # ------------------------------------------------------------------ guarded tool call
        try:
            with contextlib.suppress(CancelledError):  # allow cooperative cancellation
                promo_res = await asyncio.wait_for(
                    self._execute_tool_call_internal(promote_mcp_name, promo_args, record_action=False),
                    timeout=_PROMO_CALL_TIMEOUT,
                )
        except CancelledError:
            # Task was cancelled by shutdown or higher-level timeouts.
            self.logger.debug(f"Promo-check task for {_fmt_id(memory_id)} was cancelled.")
            return
        except asyncio.TimeoutError:
            self.logger.warning(
                f"Promo-check for {_fmt_id(memory_id)} timed out after {_PROMO_CALL_TIMEOUT}s."
            )
            return
        except Exception as exc:
            self.logger.warning(
                f"Unhandled error during promo-check for {_fmt_id(memory_id)}: {exc}", exc_info=False
            )
            return

        # ------------------------------------------------------------------ result handling
        if promo_res.get("success"):
            if promo_res.get("promoted"):
                self.logger.info(
                    f"⬆️  Memory {_fmt_id(memory_id)} promoted "
                    f"{promo_res.get('previous_level')} ➜ {promo_res.get('new_level')}."
                )
            else:
                # Only log at DEBUG if promotion did not happen – avoids log spam.
                self.logger.debug(
                    f"No promotion for {_fmt_id(memory_id)}: {promo_res.get('reason', 'no reason given')}."
                )
        else:
            self.logger.warning(
                f"Promo-check failed for {_fmt_id(memory_id)}: {promo_res.get('error', 'unknown error')}."
            )

    async def _handle_workflow_and_goal_side_effects(
        self,
        base_tool_func_name: str,
        arguments: Dict[str, Any],
        result_content_envelope: Dict[str, Any],
    ) -> None:
        """
        Centralised post-processing for UMS tool calls that mutate agent state
        (workflows, goals, statuses, etc.).  **Must** be called with the *canonical*
        base function name (e.g. ``"create_workflow"`` rather than the full MCP
        name) and the *standardised* envelope returned by
        ``_execute_tool_call_internal``.

        Parameters
        ----------
        base_tool_func_name : str
            Canonical base name of the UMS function just executed.
        arguments : Dict[str, Any]
            Arguments originally supplied to the tool.
        result_content_envelope : Dict[str, Any]
            Envelope with keys ``success`` / ``error_type`` / ``error_message`` /
            ``data`` (payload from UMS) / etc.
        """

        # ------------------------------------------------------------------ #
        # 0.  Helpers & constants
        # ------------------------------------------------------------------ #
        TERMINAL_GOAL_STATES: Tuple[GoalStatus, ...] = (
            GoalStatus.COMPLETED,
            GoalStatus.FAILED,
            GoalStatus.ABANDONED,
        )
        TERMINAL_WF_STATES: Tuple[WorkflowStatus, ...] = (
            WorkflowStatus.COMPLETED,
            WorkflowStatus.FAILED,
            WorkflowStatus.ABANDONED,
        )
        EXPECT_DICT_PAYLOAD_TOOLS: Tuple[str, ...] = (
            UMS_FUNC_CREATE_WORKFLOW,
            UMS_FUNC_UPDATE_WORKFLOW_STATUS,
            UMS_FUNC_CREATE_GOAL,
            UMS_FUNC_UPDATE_GOAL_STATUS,
        )

        def _env_success(env: Dict[str, Any]) -> bool:
            """True if the envelope signals overall success."""
            return bool(env.get("success", False))

        def _payload(env: Dict[str, Any]) -> Any:
            """Return envelope['data'] (may be None/str/dict/etc.)."""
            return env.get("data", {})

        def _payload_success(payload_obj: Any) -> bool:
            """True if payload is dict and payload['success'] is truthy."""
            return isinstance(payload_obj, dict) and payload_obj.get("success", False)

        def _set_error(tool: str, msg: str, err_type: str = "UMSError") -> None:
            """Utility to set last_error_details & flip needs_replan."""
            self.state.last_error_details = {
                "tool": self._get_ums_tool_mcp_name(tool),
                "error": msg,
                "type": err_type,
            }
            self.state.needs_replan = True

        envelope_ok = _env_success(result_content_envelope)
        payload_obj = _payload(result_content_envelope)

        # ------------------------------------------------------------------ #
        # 1.  Payload sanity-check for critical tools
        # ------------------------------------------------------------------ #
        if (
            base_tool_func_name in EXPECT_DICT_PAYLOAD_TOOLS
            and envelope_ok
            and not isinstance(payload_obj, dict)
        ):
            msg = (
                f"UMS tool '{base_tool_func_name}' reported success, but its "
                f"'data' payload is not a dict (type={type(payload_obj)})."
            )
            self.logger.warning(f"AML_SIDE_EFFECTS: {msg}  Envelope={str(result_content_envelope)[:300]}")
            # Don't treat as critical error - tool may have failed for legitimate reasons
            # Just log and continue processing with empty dict fallback
            self.logger.info(f"AML_SIDE_EFFECTS: Continuing with empty payload fallback for {base_tool_func_name}")
            payload_obj = {}

        # For uniform downstream handling, ensure dict for expected tools
        if base_tool_func_name in EXPECT_DICT_PAYLOAD_TOOLS and not isinstance(payload_obj, dict):
            payload_obj = {}

        # Snapshot state before mutation
        wf_id_before: Optional[str] = self.state.workflow_id
        goal_id_before: Optional[str] = self.state.current_goal_id
        needs_replan_before: bool = self.state.needs_replan

        log_prefix = (
            f"AML_SIDE_EFFECTS({base_tool_func_name}, "
            f"EnvSuccess={envelope_ok})"
        )
        self.logger.info(
            f"{log_prefix}: WF(before)={_fmt_id(wf_id_before)}, "
            f"Goal(before)={_fmt_id(goal_id_before)}, "
            f"needs_replan(before)={needs_replan_before}. "
            f"PayloadPreview={str(payload_obj)[:200]}"
        )

        # ------------------------------------------------------------------ #
        # 2.  Dispatch per-tool
        # ------------------------------------------------------------------ #
        if base_tool_func_name == UMS_FUNC_CREATE_WORKFLOW:
            # ---------------------------------------------------------- #
            # create_workflow
            # ---------------------------------------------------------- #
            if not (envelope_ok and _payload_success(payload_obj)):
                # Envelope said fail – nothing created.
                self.logger.error(
                    f"{log_prefix}: create_workflow failed. "
                    f"Error={result_content_envelope.get('error_message')}"
                )
                # _execute_tool_call_internal should already have set last_error_details,
                # but belt-and-suspenders:
                if not self.state.last_error_details:
                    _set_error(
                        UMS_FUNC_CREATE_WORKFLOW,
                        result_content_envelope.get(
                            "error_message",
                            "UMS create_workflow failed (envelope level).",
                        ),
                        result_content_envelope.get("error_type", "UMSError"),
                    )
                return

            # ---- Happy-path ------------------------------------------------ #
            wf_id: Optional[str] = payload_obj.get("workflow_id")
            chain_id: Optional[str] = payload_obj.get("primary_thought_chain_id")

            if not (wf_id and isinstance(wf_id, str)):
                _set_error(
                    UMS_FUNC_CREATE_WORKFLOW,
                    "create_workflow payload missing 'workflow_id'.",
                )
                return
            if not (chain_id and isinstance(chain_id, str)):
                _set_error(
                    UMS_FUNC_CREATE_WORKFLOW,
                    "create_workflow payload missing 'primary_thought_chain_id'.",
                )
                return

            # Double-check existence (DB eventual consistency)
            await asyncio.sleep(0.2)
            if not await self._check_workflow_exists(wf_id):
                _set_error(
                    UMS_FUNC_CREATE_WORKFLOW,
                    f"Workflow '{_fmt_id(wf_id)}' not visible after creation.",
                )
                return

            # ----- State updates ------------------------------------------- #
            self.state.workflow_id = wf_id
            self.state.context_id = wf_id          # convention: ctx == wf_id
            self.state.current_thought_chain_id = chain_id

            # Manage workflow stack (supporting sub-workflows)
            parent_wf_id = arguments.get("parent_workflow_id")
            if parent_wf_id and parent_wf_id == wf_id_before:
                self.state.workflow_stack.append(wf_id)          # sub-workflow
            else:
                self.state.workflow_stack = [wf_id]              # new root
                
            # Force immediate persistence to avoid losing the workflow ID
            await self._save_agent_state()

            # Reset goal info – will create a root goal below
            self.state.goal_stack = []
            self.state.current_goal_id = None
            self.state.needs_replan = False
            self.state.last_error_details = None
            self.state.consecutive_error_count = 0

            self.logger.info(
                f"{log_prefix}: New workflow {_fmt_id(wf_id)} verified. "
                f"Chain={_fmt_id(chain_id)}. StackDepth={len(self.state.workflow_stack)}"
            )

            # ---- Create ROOT goal in UMS 'goals' table -------------------- #
            root_goal_desc = (
                payload_obj.get("goal")
                or arguments.get("goal")
                or f"Overall objectives for workflow {_fmt_id(wf_id)}"
            )
            root_goal_title = (
                payload_obj.get("title")
                or arguments.get("title")
                or f"Primary Goal for WF-{wf_id[:8]}"
            )
            create_goal_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL)
            goal_created_ok = False
            if self._find_tool_server(create_goal_mcp):
                goal_args = {
                    "workflow_id": wf_id,
                    "description": root_goal_desc,
                    "title": root_goal_title,
                    "parent_goal_id": None,
                    "initial_status": GoalStatus.ACTIVE.value,
                }
                goal_env = await self._execute_tool_call_internal(
                    create_goal_mcp,
                    goal_args,
                    record_action=False,
                )
                goal_payload = goal_env.get("data", {})
                if _env_success(goal_env) and _payload_success(goal_payload):
                    goal_obj = goal_payload.get("goal", {})
                    goal_id = goal_obj.get("goal_id")
                    if goal_id:
                        self.state.goal_stack = [goal_obj]
                        self.state.current_goal_id = goal_id
                        goal_created_ok = True
                        self.logger.info(
                            f"{log_prefix}: Root goal {_fmt_id(goal_id)} created."
                        )

            if goal_created_ok:
                assess_desc = (
                    f"Initial assessment of root UMS goal: "
                    f"'{root_goal_desc[:70]}...' (Goal ID: {_fmt_id(self.state.current_goal_id)}). "
                    f"Understand goal and outline first steps."
                )
                self.state.current_plan = [
                    PlanStep(description=assess_desc, status="planned")
                ]
            else:
                retry_desc = (
                    f"CRITICAL: Workflow '{_fmt_id(wf_id)}' was created but "
                    f"root goal creation failed.  Retry using '{UMS_FUNC_CREATE_GOAL}'."
                )
                self.state.current_plan = [
                    PlanStep(
                        description=retry_desc,
                        status="planned",
                        assigned_tool=create_goal_mcp,
                    )
                ]
            # Done → fall-through to common summary / diff logs at bottom

        # ------------------------------------------------------------------ #
        elif base_tool_func_name == UMS_FUNC_CREATE_GOAL:
            # LLM called create_goal directly
            
            # Check if we already have a goal for this workflow AND it's not a sub-goal
            if (self.state.current_goal_id and 
                self.state.workflow_id and 
                arguments.get("workflow_id") == self.state.workflow_id and
                not arguments.get("parent_goal_id")):  # Allow sub-goals
                self.logger.warning(
                    f"{log_prefix}: Ignoring duplicate create_goal attempt - "
                    f"goal {self.state.current_goal_id} already exists for workflow"
                )
                # Clear error state and continue with existing goal
                self.state.needs_replan = False
                self.state.last_error_details = None
                self.state.consecutive_error_count = 0
                return
            if not envelope_ok:
                self.logger.error(
                    f"{log_prefix}: LLM-initiated create_goal failed at envelope level. "
                    f"Err={result_content_envelope.get('error_message')}"
                )
                
                # Check if the error is because a goal already exists for this workflow
                error_msg = result_content_envelope.get('error_message', '').lower()
                if (('already' in error_msg and 'goal' in error_msg) or 
                    ('exists' in error_msg and 'goal' in error_msg) or
                    ('duplicate' in error_msg)) and self.state.current_goal_id:
                    self.logger.warning(f"{log_prefix}: Goal already exists for workflow (error: {error_msg}), using existing goal {self.state.current_goal_id}")
                    # Don't replan if goal already exists - just continue with current goal
                    self.state.needs_replan = False
                    self.state.last_error_details = None
                    self.state.consecutive_error_count = 0
                    return
                
                self.state.needs_replan = True
                return

            # Log the payload structure for debugging
            self.logger.info(f"{log_prefix}: Create_goal result_envelope keys: {list(result_content_envelope.keys()) if isinstance(result_content_envelope, dict) else 'not_dict'}")
            self.logger.debug(f"{log_prefix}: Create_goal result_envelope: {str(result_content_envelope)[:800]}")
            
            # Enhanced goal object extraction with comprehensive location checking
            goal_obj = None
            extraction_location = "not_found"
            
            # Priority order for goal extraction:
            locations_to_check = [
                ("result_content_envelope.data.goal", lambda: result_content_envelope.get("data", {}).get("goal")),
                ("payload_obj.goal", lambda: payload_obj.get("goal") if isinstance(payload_obj, dict) else None),
                ("result_content_envelope.goal", lambda: result_content_envelope.get("goal")),
                ("payload_obj_direct", lambda: payload_obj if isinstance(payload_obj, dict) and "goal_id" in payload_obj else None),
                ("result_content_envelope_direct", lambda: result_content_envelope if "goal_id" in result_content_envelope else None)
            ]
            
            for location_name, extractor in locations_to_check:
                try:
                    candidate = extractor()
                    if isinstance(candidate, dict) and candidate.get("goal_id"):
                        goal_obj = candidate
                        extraction_location = location_name  # noqa: F841
                        self.logger.info(f"{log_prefix}: Successfully extracted goal from {location_name}: goal_id={goal_obj.get('goal_id', 'missing')}")
                        break
                except Exception as e:
                    self.logger.debug(f"{log_prefix}: Failed to extract from {location_name}: {e}")
                    continue
                    
            if not goal_obj:
                self.logger.error(f"{log_prefix}: Goal extraction failed from all locations. Available keys in envelope: {list(result_content_envelope.keys()) if isinstance(result_content_envelope, dict) else 'not_dict'}")
                self.logger.debug(f"{log_prefix}: Envelope data keys: {list(result_content_envelope.get('data', {}).keys()) if isinstance(result_content_envelope.get('data'), dict) else 'not_dict'}")
                self.logger.debug(f"{log_prefix}: Payload obj type: {type(payload_obj)}, keys: {list(payload_obj.keys()) if isinstance(payload_obj, dict) else 'not_dict'}")
            
            goal_id = None
            if isinstance(goal_obj, dict):
                goal_id = goal_obj.get("goal_id")
            if not goal_id:
                _set_error(
                    UMS_FUNC_CREATE_GOAL,
                    "create_goal success payload missing 'goal_id'.",
                    "GoalManagementError",
                )
                return

            # Reject if wrong workflow id
            if goal_obj.get("workflow_id") != self.state.workflow_id:
                _set_error(
                    UMS_FUNC_CREATE_GOAL,
                    "LLM created goal for a different workflow.",
                    "GoalManagementError",
                )
                return

            # Push onto stack & mark as current
            self.state.goal_stack.append(goal_obj)
            self.state.current_goal_id = goal_id
            
            # CRITICAL: Immediately synchronize with UMS to ensure consistency
            try:
                fresh_stack = await self._fetch_goal_stack_from_ums(goal_id)
                if fresh_stack:
                    self.state.goal_stack = fresh_stack
                    self.logger.info(f"{log_prefix}: Successfully synchronized goal stack with UMS. Stack depth: {len(self.state.goal_stack)}")
                    # Verify the current goal is actually in the stack
                    current_goal_found = any(
                        isinstance(g, dict) and g.get("goal_id") == goal_id 
                        for g in self.state.goal_stack
                    )
                    if not current_goal_found:
                        self.logger.warning(f"{log_prefix}: Current goal {goal_id} not found in fresh stack - keeping local version")
                        self.state.goal_stack.append(goal_obj)
                else:
                    self.logger.warning(f"{log_prefix}: Failed to fetch fresh goal stack from UMS - using local stack")
            except Exception as sync_err:
                self.logger.error(f"{log_prefix}: Goal stack sync failed: {sync_err}")
                # Keep local stack as fallback
            
            # Force immediate persistence to avoid losing the goal ID
            await self._save_agent_state()
            
            # --- ATOMIC STATE MANAGEMENT WITH RECOVERY ---
            is_root_goal_setup = (len(self.state.goal_stack) == 1 and goal_obj.get("parent_goal_id") is None)
            
            # Check if we need to update the plan
            if not is_root_goal_setup or \
               not self.state.current_plan or \
               self.state.current_plan[0].description == DEFAULT_PLAN_STEP or \
               "Establish root UMS goal" in self.state.current_plan[0].description:
                
                # Try state recovery first before falling back to replanning
                recovery_successful = await self.state_validator.validate_and_recover_state("create_goal_side_effects")
                if not recovery_successful:
                    self.state.needs_replan = True
                    self.state.current_plan = [
                        PlanStep(
                            description=f"New UMS goal established: '{goal_obj.get('description','')[:50]}...'. Formulate plan.",
                            status="planned",
                        )
                    ]
                    self.logger.info(f"{log_prefix}: State recovery failed after create_goal - triggering replan (Goal ID={goal_id})")
                else:
                    self.logger.info(f"{log_prefix}: State recovery successful after create_goal (Goal ID={goal_id})")
            else:
                # If it's root goal setup and a specific plan is already there, retain existing plan
                self.state.needs_replan = False 
                self.logger.info(f"{log_prefix}: New goal {_fmt_id(goal_id)} established. Existing plan retained or will be assessed. needs_replan=False.")
            # --- END ATOMIC STATE MANAGEMENT ---

            # Clear error state only after successful goal creation/update
            if goal_obj and goal_id:
                self.state.last_error_details = None
                self.state.consecutive_error_count = 0
                self.logger.info(f"{log_prefix}: Goal {_fmt_id(goal_id)} successfully established, error state cleared")

                goal_desc = goal_obj.get("description", "")
                self.logger.info(f"🎯 Goal established: '{goal_desc[:80]}...'")
                
                # Automatically assess completion readiness and provide specific guidance
                goal_type = self._classify_goal_type(goal_desc)
                completion_status = await self._analyze_goal_completion_semantically(goal_desc, goal_type)
                goal_type = completion_status["goal_type"]
                criteria = completion_status["completion_criteria"]
                
                # Create a direct action plan based on goal type and completion criteria
                if not self.state.current_plan or self.state.current_plan[0].description == DEFAULT_PLAN_STEP:
                    plan_description = f"Work on {goal_type.replace('_', ' ')} goal: '{goal_desc[:50]}...'. Target: {criteria['expected_deliverable']}"
                    
                    if goal_type == "analysis":
                        plan_description = f"Analyze available information and create {criteria['expected_deliverable']} with key findings"
                    elif goal_type == "creation":
                        plan_description = f"Create {criteria['expected_deliverable']} as required deliverable"
                    elif goal_type == "research":
                        plan_description = f"Research topic and document findings in {criteria['expected_deliverable']}"
                    elif goal_type == "planning":
                        plan_description = f"Develop {criteria['expected_deliverable']} with structured approach"
                    elif goal_type == "communication":
                        plan_description = f"Prepare {criteria['expected_deliverable']} for communication needs"
                    else:
                        plan_description = f"Complete goal requirements and create {criteria['expected_deliverable']}"
                    
                    self.state.current_plan = [
                        PlanStep(description=plan_description)
                    ]
                    self.state.needs_replan = False
                    self.logger.info(f"🎯 Created completion-oriented plan for {goal_type} goal: {criteria['expected_deliverable']}")

            # Record a thought (best-effort)
            record_thought_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_THOUGHT)
            if self._find_tool_server(record_thought_mcp):
                await self._execute_tool_call_internal(
                    record_thought_mcp,
                    {
                        "workflow_id": self.state.workflow_id,
                        "content": f"Established new UMS sub-goal: {goal_obj.get('description','')}",
                        "thought_type": ThoughtType.GOAL.value,
                        "thought_chain_id": self.state.current_thought_chain_id,
                    },
                    record_action=False,
                )

        # ------------------------------------------------------------------ #
        elif base_tool_func_name == UMS_FUNC_UPDATE_GOAL_STATUS:
            if not (envelope_ok and _payload_success(payload_obj)):
                self.logger.error(
                    f"{log_prefix}: update_goal_status failed. "
                    f"Err={result_content_envelope.get('error_message')}"
                )
                # Try state recovery before emergency replanning
                recovery_successful = await self.state_validator.validate_and_recover_state("update_goal_status_failed")
                if not recovery_successful:
                    self.state.needs_replan = True
                    self.logger.warning(f"{log_prefix}: State recovery failed after update_goal_status error - triggering replan")
                return

            goal_details = payload_obj.get("updated_goal_details", {})
            goal_id = goal_details.get("goal_id")
            new_status_raw = goal_details.get("status")
            if not (goal_id and new_status_raw):
                _set_error(
                    UMS_FUNC_UPDATE_GOAL_STATUS,
                    "update_goal_status payload malformed.",
                    "GoalManagementError",
                )
                return

            try:
                new_status = GoalStatus(new_status_raw.lower())
            except ValueError:
                _set_error(
                    UMS_FUNC_UPDATE_GOAL_STATUS,
                    f"Invalid goal status '{new_status_raw}'.",
                    "GoalManagementError",
                )
                return

            # Replace local copy in stack if present
            for idx, g in enumerate(self.state.goal_stack):
                if isinstance(g, dict) and g.get("goal_id") == goal_id:
                    self.state.goal_stack[idx] = goal_details
                    break

            is_terminal = new_status in TERMINAL_GOAL_STATES
            if goal_id == self.state.current_goal_id and is_terminal:
                # Pop and shift focus
                if self.state.goal_stack and self.state.goal_stack[-1].get("goal_id") == goal_id:
                    self.state.goal_stack.pop()
                parent_goal_id = payload_obj.get("parent_goal_id")
                self.state.current_goal_id = parent_goal_id
                if parent_goal_id:
                    self.state.goal_stack = await self._fetch_goal_stack_from_ums(parent_goal_id)
                    # Try state recovery before replanning for goal transition
                    recovery_successful = await self.state_validator.validate_and_recover_state("goal_status_transition")
                    if not recovery_successful:
                        self.state.needs_replan = True
                        self.logger.info(f"{log_prefix}: State recovery failed after goal transition - triggering replan")
                else:
                    self.state.goal_stack = []
                    self.state.needs_replan = False

                # Root goal finished?
                if payload_obj.get("is_root_finished", False):
                    self.state.goal_achieved_flag = (new_status == GoalStatus.COMPLETED)
                    # Record successful pattern if workflow completed successfully
                    if self.state.goal_achieved_flag and self.state.last_workflow_tools:
                        # Get the original root goal from workflow creation
                        if self.state.goal_stack and self.state.goal_stack[0]:
                            root_goal_desc = self.state.goal_stack[0].get("description", "")
                            if root_goal_desc:
                                goal_type = self._classify_goal_type(root_goal_desc)
                                self._record_successful_pattern(goal_type, self.state.last_workflow_tools)
                    # Ensure workflow status updated
                    update_wf_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_WORKFLOW_STATUS)
                    if self.state.workflow_id and self._find_tool_server(update_wf_mcp):
                        await self._execute_tool_call_internal(
                            update_wf_mcp,
                            {
                                "workflow_id": self.state.workflow_id,
                                "status": (
                                    WorkflowStatus.COMPLETED.value
                                    if self.state.goal_achieved_flag
                                    else WorkflowStatus.FAILED.value
                                ),
                                "completion_message": f"Root goal {_fmt_id(goal_id)} reached '{new_status.value}'.",
                            },
                            record_action=False,
                        )
                    self.state.current_plan = [
                        PlanStep(description="Overall workflow goal reached. Finalising.", status="completed")
                    ]
            elif goal_id == self.state.current_goal_id and new_status == GoalStatus.PAUSED:
                # Try state recovery before replanning for paused goal
                recovery_successful = await self.state_validator.validate_and_recover_state("goal_paused")
                if not recovery_successful:
                    self.state.needs_replan = True
                    self.state.current_plan = [
                        PlanStep(description=f"Current goal '{_fmt_id(goal_id)}' PAUSED. Re-evaluate strategy.", status="planned")
                    ]
                    self.logger.info(f"{log_prefix}: State recovery failed for paused goal - triggering replan")

        # ------------------------------------------------------------------ #
        elif base_tool_func_name == UMS_FUNC_UPDATE_WORKFLOW_STATUS:
            if not (envelope_ok and _payload_success(payload_obj)):
                self.logger.error(
                    f"{log_prefix}: update_workflow_status failed. "
                    f"Err={result_content_envelope.get('error_message')}"
                )
                # Try state recovery before emergency replanning
                recovery_successful = await self.state_validator.validate_and_recover_state("update_workflow_status_failed")
                if not recovery_successful:
                    self.state.needs_replan = True
                    self.logger.warning(f"{log_prefix}: State recovery failed after workflow status error - triggering replan")
                return

            wf_id = payload_obj.get("workflow_id")
            status_raw = payload_obj.get("status")
            if not (wf_id and status_raw):
                _set_error(
                    UMS_FUNC_UPDATE_WORKFLOW_STATUS,
                    "update_workflow_status payload malformed.",
                )
                return
            try:
                new_status = WorkflowStatus(status_raw.lower())
            except ValueError:
                _set_error(
                    UMS_FUNC_UPDATE_WORKFLOW_STATUS,
                    f"Invalid workflow status '{status_raw}'.",
                )
                return

            is_terminal = new_status in TERMINAL_WF_STATES
            if self.state.workflow_stack and wf_id == self.state.workflow_stack[-1]:
                # Current top-of-stack workflow updated
                if is_terminal:
                    finished_wf = self.state.workflow_stack.pop()
                    parent_wf = self.state.workflow_stack[-1] if self.state.workflow_stack else None
                    if parent_wf:
                        # Return to parent context
                        self.state.workflow_id = parent_wf
                        self.state.context_id = parent_wf
                        await self._set_default_thought_chain_id()
                        self.state.goal_stack = []
                        self.state.current_goal_id = None
                        
                        # Try state recovery before replanning for workflow transition
                        recovery_successful = await self.state_validator.validate_and_recover_state("workflow_transition")
                        if not recovery_successful:
                            self.state.needs_replan = True
                            self.state.current_plan = [
                                PlanStep(
                                    description=f"Returned to parent workflow '{_fmt_id(parent_wf)}' after sub-workflow '{_fmt_id(finished_wf)}' finished ({new_status.value}).",
                                    status="planned",
                                )
                            ]
                            self.logger.info(f"{log_prefix}: State recovery failed for workflow transition - triggering replan")
                    else:
                        # Root workflow finished
                        self.state.workflow_id = None
                        self.state.context_id = None
                        self.state.current_thought_chain_id = None
                        self.state.current_plan = []
                        self.state.goal_stack = []
                        self.state.current_goal_id = None
                        self.state.goal_achieved_flag = (new_status == WorkflowStatus.COMPLETED)
                        self.state.needs_replan = False
            elif wf_id == self.state.workflow_id and is_terminal:
                # Single-workflow run finished
                self.state.workflow_id = None
                self.state.context_id = None
                self.state.current_thought_chain_id = None
                self.state.current_plan = []
                self.state.goal_stack = []
                self.state.current_goal_id = None
                self.state.goal_achieved_flag = (new_status == WorkflowStatus.COMPLETED)
                self.state.needs_replan = False
            elif wf_id == self.state.workflow_id and new_status == WorkflowStatus.PAUSED:
                # Try state recovery before replanning for paused workflow
                recovery_successful = await self.state_validator.validate_and_recover_state("workflow_paused")
                if not recovery_successful:
                    self.state.needs_replan = True
                    self.state.current_plan = [
                        PlanStep(description=f"Workflow '{_fmt_id(wf_id)}' PAUSED. Await resume.", status="planned")
                    ]
                    self.logger.info(f"{log_prefix}: State recovery failed for paused workflow - triggering replan")

        # ------------------------------------------------------------------ #
        # 3.  Unknown tool – no side-effects required
        # ------------------------------------------------------------------ #
        else:
            self.logger.debug(f"{log_prefix}: No side-effects registered for tool '{base_tool_func_name}'.")
            # Nothing to do…

        # ------------------------------------------------------------------ #
        # 4.  Summary logging & needs_replan diff
        # ------------------------------------------------------------------ #
        if not needs_replan_before and self.state.needs_replan:
            self.logger.info(f"{log_prefix}: needs_replan became *True* during side-effects.")
        elif needs_replan_before and not self.state.needs_replan:
            self.logger.info(f"{log_prefix}: needs_replan became *False* during side-effects.")

        if (wf_id_before != self.state.workflow_id) or (goal_id_before != self.state.current_goal_id):
            self.logger.info(
                f"{log_prefix}: STATE-CHANGE → WF {_fmt_id(wf_id_before)} → {_fmt_id(self.state.workflow_id)}, "
                f"Goal {_fmt_id(goal_id_before)} → {_fmt_id(self.state.current_goal_id)}, "
                f"StackDepth={len(self.state.goal_stack)}"
            )

        self.logger.info(
            f"{log_prefix}: Exit. WF={_fmt_id(self.state.workflow_id)}, "
            f"Context={_fmt_id(self.state.context_id)}, "
            f"Goal={_fmt_id(self.state.current_goal_id)}, "
            f"needs_replan={self.state.needs_replan}"
        )


    async def _fetch_goal_stack_from_ums(self, goal_id: str) -> list[dict[str, Any]]:
        """
        Return the parent → leaf stack for *goal_id* as stored in UMS.

        The UMS wrapper has evolved over time, so the returned envelope may look
        like any of the following:

        A.  {"success": true, "stack": [...]}
        B.  {"success": true, "data": {"stack": [...]} }
        C.  {"success": true, "goal_stack": [...]}
        D.  {"success": true, "data": {"goal_stack": [...]} }

        This helper normalises all four shapes and always yields a **list**.
        An empty list means "couldn't fetch / tool missing / schema invalid".
        """
        # ------------------------------------------------------------------ #
        # Resolve the fully-qualified MCP name and ensure the tool is live.
        # ------------------------------------------------------------------ #
        get_goal_stack_mcp = self._get_ums_tool_mcp_name("get_goal_stack")
        if self._find_tool_server(get_goal_stack_mcp) is None:
            self.logger.warning(
                f"_fetch_goal_stack_from_ums: UMS tool '{get_goal_stack_mcp}' unavailable."
            )
            return []

        # ------------------------------------------------------------------ #
        # Invoke tool through the internal execution wrapper.
        # ------------------------------------------------------------------ #
        try:
            envelope: dict[str, Any] = await self._execute_tool_call_internal(
                get_goal_stack_mcp, {"goal_id": goal_id}, record_action=False
            )
        except Exception as exc:
            self.logger.error(
                f"_fetch_goal_stack_from_ums: Exception while calling '{get_goal_stack_mcp}': {exc}",
                exc_info=True,
            )
            return []

        if not envelope.get("success"):
            self.logger.warning(
                "_fetch_goal_stack_from_ums: Tool returned failure – "
                f"type={envelope.get('error_type')}, msg={envelope.get('error_message')}"
            )
            return []

        # ------------------------------------------------------------------ #
        # Normalise schema variations.
        # ------------------------------------------------------------------ #
        def _extract_stack(payload: dict[str, Any]) -> list[dict[str, Any]] | None:
            """
            Locate a list under any of the accepted keys.  
            Returns *None* if nothing matches.
            """
            if not payload:
                return None

            # Top-level keys -------------------------------------------------
            if isinstance(payload.get("stack"), list):
                return payload["stack"]
            if isinstance(payload.get("goal_stack"), list):
                return payload["goal_stack"]
            if isinstance(payload.get("goal_tree"), list):  # Added support for goal_tree key
                self.logger.info("_fetch_goal_stack_from_ums: Found goals in 'goal_tree' key")
                return payload["goal_tree"]

            # Nested inside "data" ------------------------------------------
            data_block = payload.get("data")
            if isinstance(data_block, dict):
                if isinstance(data_block.get("stack"), list):
                    return data_block["stack"]
                if isinstance(data_block.get("goal_tree"), list):  # Added support for goal_tree key
                    self.logger.info("_fetch_goal_stack_from_ums: Found goals in data.goal_tree key")
                    return data_block["goal_tree"]
                if isinstance(data_block.get("goal_stack"), list):
                    return data_block["goal_stack"]

            return None

        stack = _extract_stack(envelope)
        if stack is None:
            self.logger.error(
                "_fetch_goal_stack_from_ums: Successful call but no stack found "
                "in any supported location. Check UMS wrapper for schema drift."
            )
            return []

        # Return a shallow copy so callers can mutate safely.
        return list(stack)


    async def _apply_heuristic_plan_update(  # noqa: C901  (cyclomatic complexity intentionally managed with dispatch)
        self,
        last_llm_decision_from_mcpc: Dict[str, Any],
        last_tool_result_envelope: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Conservative heuristic-based plan maintenance (v2).

        * Maintains original behaviour 1-for-1.
        * Swaps long if/elif chain for a dispatcher to aid comprehension.
        * Ensures internal flags/counters are *always* updated deterministically.
        * Never wipes ``last_error_details`` unless this turn ends in success.
        """

        # ------------------------------------------------------------------ setup --
        self.logger.info("📋 Applying heuristic plan update (v2)…")
        needs_replan_on_entry = self.state.needs_replan

        decision_raw: str = str(last_llm_decision_from_mcpc.get("decision", ""))
        try:
            decision: Optional[_Decision] = _Decision(decision_raw)
        except ValueError:  # unknown decision type
            decision = None

        tool_name_mcp: Optional[str] = last_llm_decision_from_mcpc.get("tool_name")
        current_step = self.state.current_plan[0] if self.state.current_plan else None
        current_step_desc_lc = (current_step.description or "").lower() if current_step else ""

        # ------------- guard: empty plan ------------------------------------------
        if current_step is None:
            self.logger.error("📋 HEURISTIC CRITICAL: Current plan is empty. Attempting state recovery.")
            
            # Try state recovery first before emergency replanning
            recovery_successful = await self.state_validator.validate_and_recover_state("empty_plan_critical")
            if not recovery_successful:
                self.state.current_plan = [PlanStep(description="CRITICAL FALLBACK: Plan was empty. Re-evaluate.")]
                self.state.needs_replan = True
                self.state.last_error_details = self.state.last_error_details or {
                    "error": "Plan empty.",
                    "type": "PlanManagementError",
                }
                self.state.consecutive_error_count += 1
                self.logger.warning("📋 State recovery failed for empty plan - triggering emergency replan")
            else:
                self.logger.info("📋 State recovery successful for empty plan issue")
                
            self.state.successful_actions_since_reflection = 0
            self.state.successful_actions_since_consolidation = 0
            return

        # ------------------------- aliases & helpers ------------------------------
        SUCCESS, FAILED = ActionStatus.COMPLETED.value, ActionStatus.FAILED.value
        INTERNAL_META = self._INTERNAL_OR_META_TOOLS_BASE_NAMES
        get_base = self._get_base_function_name

        def _mark_step(status: str, summary: str = "") -> None:
            """Set status & summary on the *current* plan step (trim summary to 150 chars)."""
            current_step.status = status
            if summary:
                current_step.result_summary = summary[:150]

        def _pop_if_first(step) -> None:
            """Remove step from head of plan if it is indeed the head."""
            if self.state.current_plan and self.state.current_plan[0].id == step.id:
                self.state.current_plan.pop(0)

        # ----------------------------- scenario handlers --------------------------
        #
        # each returns: (success_flag: bool, reason_tag: str)

        async def _handle_tool_executed() -> Tuple[bool, str]:
            succeeded = bool(isinstance(last_tool_result_envelope, dict) and last_tool_result_envelope.get("success"))
            if succeeded:
                summary = "Success."
                data = last_tool_result_envelope.get("data") if isinstance(last_tool_result_envelope, dict) else {}
                if isinstance(data, dict):
                    for k in (
                        "summary",
                        "message",
                        "memory_id",
                        "action_id",
                        "artifact_id",
                        "link_id",
                        "chain_id",
                        "state_id",
                        "report",
                        "visualization",
                        "goal_id",
                        "workflow_id",
                    ):
                        if data.get(k) is not None:
                            summary_val = str(data[k])
                            summary = f"{k}: {_fmt_id(summary_val) if 'id' in k else summary_val}"
                            break
                _mark_step(SUCCESS, summary)
                
                # **CRITICAL FIX**: Always advance plan for vague/default steps, even on success
                should_advance_vague_step = (
                    current_step.description == DEFAULT_PLAN_STEP or
                    "Execute immediate action" in current_step.description or
                    len(current_step.description) < 30 or  # Simple fallback for vague descriptions
                    any(keyword in current_step.description.lower() for keyword in ["vague", "general", "decide", "assess"])
                )
                
                if should_advance_vague_step:
                    self.logger.info(f"🔄 ADVANCING PLAN: Removing vague successful step - '{current_step.description[:60]}...'")
                
                _pop_if_first(current_step)

                # UPDATE LOOP DETECTION AND PROGRESSION TRACKING
                if tool_name_mcp:
                    # Update progression stage tracking
                    self._update_plan_progression_stage(tool_name_mcp, True)
                    
                    # Check for infinite loop patterns
                    if self._is_in_loop(tool_name_mcp):
                        raise InfiniteLoopError(f"Tool {tool_name_mcp} called too many times")

                if not self.state.current_plan:
                    # Use DELIVERABLE-FIRST planning approach
                    current_goal_desc = ""
                    if self.state.goal_stack and self.state.goal_stack[-1]:
                        current_goal_desc = self.state.goal_stack[-1].get("description", "")
                    
                    # Generate concrete action plan biased toward deliverable creation
                    if current_goal_desc:
                        new_plan_steps = self._generate_concrete_action_plan(current_goal_desc)
                        self.state.current_plan.extend(new_plan_steps)
                        self.logger.info(f"🎯 DELIVERABLE-FIRST: Generated {len(new_plan_steps)} concrete action steps")
                    else:
                        # Even without goal description, default to creating something
                        self.state.current_plan.append(
                            PlanStep(
                                description="Create deliverable for current goal using available information",
                                assigned_tool=self._get_ums_tool_mcp_name("record_artifact")
                            )
                        )

                # Clear error only if same tool now succeeded
                if (
                    self.state.last_error_details
                    and self.state.last_error_details.get("tool") == tool_name_mcp
                ):
                    self.state.last_error_details = None

                return True, "tool-exec"

            # --- failure path with state recovery
            err_msg = (
                last_tool_result_envelope.get("error_message", "Unknown failure")[:120]
                if isinstance(last_tool_result_envelope, dict)
                else "Unknown tool failure."
            )
            _mark_step(FAILED, f"Failure: {err_msg}")
            
            # **CRITICAL FIX**: Track failure count and advance plan if step fails repeatedly
            current_step.failure_count += 1
            
            # **CRITICAL FIX**: Remove failed step from plan if it has failed multiple times or meets criteria
            should_advance_plan = (
                current_step.failure_count >= 2 or  # Failed 2+ times
                current_step.description == DEFAULT_PLAN_STEP or  # Vague default step  
                "Execute immediate action" in current_step.description or  # Other vague steps
                self.state.consecutive_same_tool_count >= 2  # Same tool failing repeatedly
            )
            
            if should_advance_plan:
                self.logger.warning(f"🔄 ADVANCING PLAN: Removing failed step after {current_step.failure_count} failures - '{current_step.description[:60]}...'")
                _pop_if_first(current_step)
                
                # If plan is now empty, create concrete next step instead of vague default
                if not self.state.current_plan:
                    current_goal_desc = ""
                    if self.state.goal_stack and self.state.goal_stack[-1]:
                        current_goal_desc = self.state.goal_stack[-1].get("description", "")
                    
                    if current_goal_desc:
                        # Create specific next step based on goal type
                        goal_type = self._classify_goal_type(current_goal_desc)
                        criteria = self._get_goal_completion_criteria(goal_type, current_goal_desc)
                        
                        next_step = PlanStep(
                            description=f"Create {criteria['expected_deliverable']} for {goal_type} goal",
                            assigned_tool=self._get_ums_tool_mcp_name("record_artifact")
                        )
                        self.state.current_plan.append(next_step)
                        self.logger.info(f"🎯 PLAN ADVANCEMENT: Added concrete deliverable-focused step")
                    else:
                        # Fallback to generic deliverable creation
                        self.state.current_plan.append(
                            PlanStep(
                                description="Create concrete deliverable using available information",
                                assigned_tool=self._get_ums_tool_mcp_name("record_artifact")
                            )
                        )
            
            # UPDATE TRACKING FOR FAILED TOOLS TOO
            if tool_name_mcp:
                self._update_plan_progression_stage(tool_name_mcp, False)
                # Still track tool execution for loop detection even on failure
                if self._is_in_loop(tool_name_mcp):
                    raise InfiniteLoopError(f"Tool {tool_name_mcp} called too many times (failed repeatedly)")
            
            # Try state recovery before replanning for tool failure
            recovery_successful = await self.state_validator.validate_and_recover_state("tool_execution_failure")
            if not recovery_successful:
                self.state.needs_replan = True
                self.logger.info("📋 State recovery failed for tool execution failure - triggering replan")
            else:
                self.logger.info("📋 State recovery successful for tool execution failure")
                
            return False, "tool-exec-fail"

        async def _handle_thought() -> Tuple[bool, str]:
            succeeded = bool(isinstance(last_tool_result_envelope, dict) and last_tool_result_envelope.get("success"))
            if succeeded:
                content = last_llm_decision_from_mcpc.get("content", "")
                thought_id = (
                    last_tool_result_envelope.get("data", {}).get("thought_id", "UnkID")
                    if isinstance(last_tool_result_envelope, dict)
                    else "UnkID"
                )
                _mark_step(SUCCESS, f"Thought({_fmt_id(thought_id)}): {content[:50]}…")
                _pop_if_first(current_step)

                if last_llm_decision_from_mcpc.get("_mcp_client_force_replan_after_thought_"):
                    self.state.needs_replan = True
                    if not self.state.current_plan:
                        self.state.current_plan.append(
                            PlanStep(description=f"Replan forced. Orig thought: {content[:60]}…")
                        )
                else:
                    # Don't overthink - if no plan, add action-oriented step
                    if not self.state.current_plan:
                        # Get current goal description for context
                        current_goal_desc = ""
                        if self.state.goal_stack and self.state.goal_stack[-1]:
                            current_goal_desc = self.state.goal_stack[-1].get("description", "")
                        
                        # Suggest next tools based on context
                        suggested_tools = self._suggest_tool_chain(current_goal_desc, self.state.last_action_summary)
                        if suggested_tools:
                            # Create plan steps for suggested tools
                            for i, tool_name in enumerate(suggested_tools[:3]):  # Limit to 3 tools
                                step_desc = f"Use {tool_name.split('_')[-1]} tool to progress toward goal"
                                if tool_name == UMS_FUNC_HYBRID_SEARCH:
                                    step_desc = "Search for relevant information to understand the task"
                                elif tool_name == UMS_FUNC_STORE_MEMORY:
                                    step_desc = "Store important findings in memory"
                                elif tool_name == UMS_FUNC_RECORD_ARTIFACT:
                                    step_desc = "Create the required deliverable/artifact"
                                
                                self.state.current_plan.append(
                                    PlanStep(
                                        description=step_desc,
                                        assigned_tool=self._get_ums_tool_mcp_name(tool_name)
                                    )
                                )
                            self.logger.info(f"Smart chaining suggested {len(suggested_tools)} tools for empty plan")
                        else:
                            self.state.current_plan.append(
                                PlanStep(description="Take concrete action: use tools to make progress toward the goal.")
                            )
                return True, "thought"

            # --- failure path with state recovery
            err_msg = (
                last_tool_result_envelope.get("error_message", "Failed to record thought.")[:100]
                if isinstance(last_tool_result_envelope, dict)
                else "Failed thought recording."
            )
            _mark_step(FAILED, f"Failed Thought: {err_msg}")
            
            # Try state recovery before replanning for thought failure
            recovery_successful = await self.state_validator.validate_and_recover_state("thought_recording_failure")
            if not recovery_successful:
                self.state.needs_replan = True
                self.logger.info("📋 State recovery failed for thought recording failure - triggering replan")
            else:
                self.logger.info("📋 State recovery successful for thought recording failure")
                
            return False, "thought-fail"

        def _handle_agent_plan_update() -> Tuple[bool, str]:
            succeeded = bool(isinstance(last_tool_result_envelope, dict) and last_tool_result_envelope.get("success"))
            if succeeded:
                if current_step.assigned_tool == AGENT_TOOL_UPDATE_PLAN or "update plan" in current_step_desc_lc:
                    _mark_step(SUCCESS, "Plan updated by LLM.")
                    _pop_if_first(current_step)
                if not self.state.current_plan:
                    self.state.current_plan.append(PlanStep(description="New plan active. Proceed."))
                return True, "agent-plan-update"

            _mark_step(
                FAILED,
                f"Failed plan update: {last_tool_result_envelope.get('error_message', 'Unknown')[:100]}",
            )
            self.state.needs_replan = True
            return False, "agent-plan-update-fail"

        def _handle_complete() -> Tuple[bool, str]:
            self.state.current_plan = [
                PlanStep(description="Goal Achieved. Finalizing workflow.", status="completed")
            ]
            self.state.needs_replan = False
            return True, "complete"

        def _handle_textual_plan_update() -> Tuple[bool, str]:
            if not self.state.needs_replan:
                return True, "textual-plan-update"
            _mark_step(
                FAILED,
                f"Textual Plan Error: {self.state.last_error_details.get('error', 'Validation error')[:100]}"
                if self.state.last_error_details
                else "Failed textual plan.",
            )
            return False, "textual-plan-fail"

        async def _handle_other() -> Tuple[bool, str]:
            if self.state.last_error_details and not self.state.needs_replan:
                # Try state recovery before emergency replanning
                recovery_successful = await self.state_validator.validate_and_recover_state("general_error_detected")
                if not recovery_successful:
                    self.state.needs_replan = True
                    _mark_step(
                        FAILED,
                        f"Error state detected: {str(self.state.last_error_details.get('error'))[:50]}",
                    )
                    self.logger.info("📋 State recovery failed for general error - triggering replan")
                else:
                    self.logger.info("📋 State recovery successful for general error")
                    
            if not self.state.current_plan:
                self.state.current_plan.append(
                    PlanStep(description="Assess situation and decide next action.")
                )
            return False, "unhandled"

        # ----------------------------- dispatcher ----------------------------------
        dispatcher = {
            _Decision.TOOL_EXECUTED: _handle_tool_executed,
            _Decision.THOUGHT: _handle_thought,
            _Decision.CALL_TOOL: _handle_agent_plan_update
            if tool_name_mcp == AGENT_TOOL_UPDATE_PLAN
            else _handle_other,
            _Decision.COMPLETE: _handle_complete,
            _Decision.COMPLETE_ART: _handle_complete,
            _Decision.PLAN_UPDATE: _handle_textual_plan_update,
        }

        if decision in dispatcher:
            result = dispatcher[decision]()
            if asyncio.iscoroutine(result):
                action_successful, reason_tag = await result
            else:
                action_successful, reason_tag = result
        else:  # unknown / unsupported decision
            action_successful, reason_tag = await _handle_other()

        # ----------------------- counters / meta-cognition -------------------------
        if action_successful:
            self.state.consecutive_error_count = 0
            base_tool = get_base(tool_name_mcp) if tool_name_mcp else None
            substantive_tool = (
                decision == _Decision.TOOL_EXECUTED
                and tool_name_mcp
                and base_tool not in INTERNAL_META
                and tool_name_mcp != AGENT_TOOL_UPDATE_PLAN
            )
            successful_thought = decision == _Decision.THOUGHT
            if substantive_tool:
                self.state.successful_actions_since_reflection += 1.0
                self.state.successful_actions_since_consolidation += 1.0
            elif successful_thought:
                self.state.successful_actions_since_reflection += 0.5
                self.state.successful_actions_since_consolidation += 0.5
        else:
            if self.state.last_error_details:
                self.state.consecutive_error_count += 1
            if self.state.last_error_details or (self.state.needs_replan and not needs_replan_on_entry):
                self.state.successful_actions_since_reflection = 0
                self.state.successful_actions_since_consolidation = 0

        # -------------------------------- logging ----------------------------------
        if self.logger.isEnabledFor(logging.DEBUG):
            log_msg = (
                f"📋 Heuristic Update End ({reason_tag}). Steps: {len(self.state.current_plan)}; "
            )
            if self.state.current_plan:
                nxt = self.state.current_plan[0]
                depends = f"Depends: {[_fmt_id(d) for d in nxt.depends_on]}" if nxt.depends_on else "Depends: None"
                log_msg += (
                    f"Next: '{nxt.description[:60]}…' (ID: {_fmt_id(nxt.id)}, Status: {nxt.status}, {depends})"
                )
            else:
                log_msg += "Plan empty."
            log_msg += (
                f"; NeedsReplan={self.state.needs_replan}; "
                f"ConsecutiveErrors={self.state.consecutive_error_count}"
            )
            self.logger.debug(log_msg)

        # ------------------------------- sanity assert -----------------------------
        assert action_successful is not None, "Internal error: action_successful not set!"


    async def _adapt_thresholds(self, stats: Dict[str, Any]) -> None:
        """
        Intelligently adjust reflection/consolidation thresholds using LLM-based semantic analysis

        Args:
            stats: result object returned by `ums.compute_memory_statistics`
        Side-effects:
            - May mutate `self.state.current_consolidation_threshold`
            - May mutate `self.state.current_reflection_threshold`
        """

        # ─────────────────────────────────────────────── early validity check ──
        if not (stats and stats.get("success", False)):
            self.logger.warning("Cannot adapt thresholds: Invalid stats object.")
            return

        # Use LLM to analyze current performance and suggest threshold adjustments
        threshold_analysis = await self._analyze_threshold_performance_with_llm(stats)
        
        if threshold_analysis:
            # Apply LLM-suggested changes
            changed = self._apply_llm_threshold_adjustments(threshold_analysis)
            if not changed:
                self.logger.debug("LLM analysis suggests no threshold adjustments needed.")
        else:
            # Fallback to intelligent heuristic analysis if LLM fails
            changed = await self._adapt_thresholds_intelligent_fallback(stats)

    async def _analyze_threshold_performance_with_llm(self, stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Use LLM to analyze current threshold performance and suggest adjustments"""
        
        # Prepare performance context for LLM analysis
        total_memories = max(int(stats.get("total_memories", 0)), 1)
        episodic_count = int(stats.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0))
        episodic_ratio = episodic_count / total_memories

        # Calculate failure rate from tool usage stats
        totals = [v.get("success", 0) + v.get("failure", 0) for v in self.state.tool_usage_stats.values()]
        failures = [v.get("failure", 0) for v in self.state.tool_usage_stats.values()]
        total_calls, total_failures = sum(totals), sum(failures)
        failure_rate = (total_failures / total_calls) if total_calls >= 5 else 0.0
        
        performance_context = f"""
Current System Performance:
- Reflection Threshold: {self.state.current_reflection_threshold}
- Consolidation Threshold: {self.state.current_consolidation_threshold}
- Total Memories: {total_memories}
- Episodic Memory Ratio: {episodic_ratio:.1%}
- Tool Failure Rate: {failure_rate:.1%}
- Consecutive Errors: {self.state.consecutive_error_count}
- Recent Actions: {self.state.turns_since_artifact_creation}
- Current Loop: {self.state.current_loop}

Performance Goals:
- Episodic memories should be 10-30% of total (currently {episodic_ratio:.1%})
- Tool failure rate should be under 10% (currently {failure_rate:.1%})
- System should be responsive but not over-reactive
"""

        threshold_prompt = f"""
Analyze this cognitive system's performance and suggest threshold adjustments.

{performance_context}

Rate the need for adjustments on a 0-100 scale and suggest changes:

reflection_adjustment_score: [0-100] (How much to adjust reflection threshold)
- 0-20: Working well, no change needed
- 21-40: Minor adjustment needed 
- 41-60: Moderate adjustment needed
- 61-80: Significant adjustment needed
- 81-100: Major adjustment required

consolidation_adjustment_score: [0-100] (How much to adjust consolidation threshold)
- 0-20: Working well, no change needed
- 21-40: Minor adjustment needed
- 41-60: Moderate adjustment needed  
- 61-80: Significant adjustment needed
- 81-100: Major adjustment required

reflection_direction: [increase/decrease/maintain] (Direction to adjust reflection threshold)
consolidation_direction: [increase/decrease/maintain] (Direction to adjust consolidation threshold)

reasoning: [Brief explanation of recommendations]

Consider:
1. High failure rates suggest need for more reflection (lower threshold)
2. Too many episodic memories suggest need for more consolidation (lower threshold)
3. System stability and responsiveness balance
4. Current workload and performance trends

Respond in the exact format above.
"""
        threshold_analysis_schema = {
            "type": "object",
            "properties": {
                "reflection_adjustment_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Reflection threshold adjustment score 0-100"
                },
                "reflection_direction": {
                    "type": "string",
                    "enum": ["increase", "decrease", "maintain"],
                    "description": "Direction to adjust reflection threshold"
                },
                "consolidation_adjustment_score": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Consolidation threshold adjustment score 0-100"
                },
                "consolidation_direction": {
                    "type": "string",
                    "enum": ["increase", "decrease", "maintain"],
                    "description": "Direction to adjust consolidation threshold"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning for threshold adjustments"
                }
            },
            "required": ["reflection_adjustment_score", "reflection_direction", "consolidation_adjustment_score", "consolidation_direction"],
            "additionalProperties": False
        }

        try:
            response = await self.mcp_client.query_llm_structured(
                prompt_messages=[{"role": "user", "content": threshold_prompt}],
                response_schema=threshold_analysis_schema,
                schema_name="threshold_analysis",
                use_cheap_model=True,
                max_tokens=150,
                temperature=0.1
            )
            
            if response:
                # Convert to expected format for compatibility
                analysis = {}
                analysis["reflection_adjustment_score"] = response.get("reflection_adjustment_score", 0)
                analysis["reflection_direction"] = response.get("reflection_direction", "maintain")
                analysis["consolidation_adjustment_score"] = response.get("consolidation_adjustment_score", 0)
                analysis["consolidation_direction"] = response.get("consolidation_direction", "maintain")
                analysis["reasoning"] = response.get("reasoning", "")
                
                self.logger.debug(f"🧠 LLM threshold analysis: {analysis}")
                return analysis
                    
        except Exception as e:
            self.logger.debug(f"LLM threshold analysis failed: {e}")
        
        return None
    
    def _apply_llm_threshold_adjustments(self, analysis: Dict[str, Any]) -> bool:
        """Apply threshold adjustments based on LLM analysis"""
        
        changed = False
        
        # Apply reflection threshold adjustment
        reflection_score = analysis.get("reflection_adjustment_score", 0)
        reflection_direction = analysis.get("reflection_direction", "maintain")
        
        if reflection_score >= 40 and reflection_direction != "maintain":
            # Calculate adjustment magnitude based on score
            if reflection_score >= 80:
                magnitude = 5  # Major adjustment
            elif reflection_score >= 60:
                magnitude = 3  # Significant adjustment
            else:
                magnitude = 2  # Moderate adjustment
            
            if reflection_direction == "decrease":
                magnitude = -magnitude
            
            new_reflection = max(MIN_REFLECTION_THRESHOLD, 
                               min(MAX_REFLECTION_THRESHOLD, 
                                   self.state.current_reflection_threshold + magnitude))
            
            if new_reflection != self.state.current_reflection_threshold:
                direction = "Lowering" if new_reflection < self.state.current_reflection_threshold else "Raising"
                self.logger.info(f"🧠 LLM-guided {direction} reflection threshold: "
                               f"{self.state.current_reflection_threshold} → {new_reflection} "
                               f"(score={reflection_score}, reasoning={analysis.get('reasoning', 'N/A')})")
                self.state.current_reflection_threshold = new_reflection
                changed = True
        
        # Apply consolidation threshold adjustment
        consolidation_score = analysis.get("consolidation_adjustment_score", 0)
        consolidation_direction = analysis.get("consolidation_direction", "maintain")
        
        if consolidation_score >= 40 and consolidation_direction != "maintain":
            # Calculate adjustment magnitude based on score
            if consolidation_score >= 80:
                magnitude = 5  # Major adjustment
            elif consolidation_score >= 60:
                magnitude = 3  # Significant adjustment
            else:
                magnitude = 2  # Moderate adjustment
            
            if consolidation_direction == "decrease":
                magnitude = -magnitude
            
            new_consolidation = max(MIN_CONSOLIDATION_THRESHOLD,
                                  min(MAX_CONSOLIDATION_THRESHOLD,
                                      self.state.current_consolidation_threshold + magnitude))
            
            if new_consolidation != self.state.current_consolidation_threshold:
                direction = "Lowering" if new_consolidation < self.state.current_consolidation_threshold else "Raising"
                self.logger.info(f"🧠 LLM-guided {direction} consolidation threshold: "
                               f"{self.state.current_consolidation_threshold} → {new_consolidation} "
                               f"(score={consolidation_score})")
                self.state.current_consolidation_threshold = new_consolidation
                changed = True
        
        return changed
    
    async def _adapt_thresholds_intelligent_fallback(self, stats: Dict[str, Any]) -> bool:
        """Intelligent fallback using semantic understanding"""
        
        changed = False
        
        # Analyze memory distribution with semantic understanding
        total_memories = max(int(stats.get("total_memories", 0)), 1)
        episodic_count = int(stats.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0))
        episodic_ratio = episodic_count / total_memories
        
        # Semantic assessment of memory distribution health
        if episodic_ratio > 0.35:  # Too many episodic memories
            # Need more consolidation
            adjustment = min(3, int((episodic_ratio - 0.20) * 10))  # Semantic scaling
            new_consolidation = max(MIN_CONSOLIDATION_THRESHOLD, 
                                  self.state.current_consolidation_threshold - adjustment)
            if new_consolidation != self.state.current_consolidation_threshold:
                self.logger.info(f"📊 Semantic analysis: Too many episodic memories ({episodic_ratio:.1%}) - "
                               f"lowering consolidation threshold: {self.state.current_consolidation_threshold} → {new_consolidation}")
                self.state.current_consolidation_threshold = new_consolidation
                changed = True
        
        elif episodic_ratio < 0.05:  # Too few episodic memories
            # Reduce consolidation frequency
            adjustment = min(3, int((0.15 - episodic_ratio) * 10))  # Semantic scaling
            new_consolidation = min(MAX_CONSOLIDATION_THRESHOLD,
                                  self.state.current_consolidation_threshold + adjustment)
            if new_consolidation != self.state.current_consolidation_threshold:
                self.logger.info(f"📊 Semantic analysis: Too few episodic memories ({episodic_ratio:.1%}) - "
                               f"raising consolidation threshold: {self.state.current_consolidation_threshold} → {new_consolidation}")
                self.state.current_consolidation_threshold = new_consolidation
                changed = True
        
        # Analyze error patterns with semantic understanding
        if self.state.consecutive_error_count >= 2:
            # System appears to need more reflection
            adjustment = min(3, self.state.consecutive_error_count)
            new_reflection = max(MIN_REFLECTION_THRESHOLD,
                               self.state.current_reflection_threshold - adjustment)
            if new_reflection != self.state.current_reflection_threshold:
                self.logger.info(f"🚨 Semantic analysis: Error pattern detected ({self.state.consecutive_error_count} consecutive) - "
                               f"lowering reflection threshold: {self.state.current_reflection_threshold} → {new_reflection}")
                self.state.current_reflection_threshold = new_reflection
                changed = True
        
        # Assess productivity patterns
        elif (self.state.turns_since_artifact_creation == 0 and 
              self.state.consecutive_error_count == 0 and
              self.state.current_loop > 5):
            # System appears to be working well - can reduce reflection frequency
            new_reflection = min(MAX_REFLECTION_THRESHOLD,
                               self.state.current_reflection_threshold + 1)
            if new_reflection != self.state.current_reflection_threshold:
                self.logger.info(f"✅ Semantic analysis: System performing well - "
                               f"raising reflection threshold: {self.state.current_reflection_threshold} → {new_reflection}")
                self.state.current_reflection_threshold = new_reflection
                changed = True
        
        return changed

    async def _trigger_promotion_checks(self) -> None:
        """
        Scan the current workflow for memories that are good candidates for
        promotion (episodic → semantic, semantic → procedural/skill) and fire
        `_check_and_trigger_promotion` in the background for each.
        """
        # ---- fast pre-flight checks ------------------------------------------------
        if self._shutdown_event.is_set():
            self.logger.debug("Promotion check aborted: shutdown signalled.")
            return

        wf_id = self.state.workflow_id
        if not wf_id:
            self.logger.debug("Skipping promotion check: no active workflow.")
            return

        self.logger.debug("🔄 Running periodic promotion check…")

        # ---- tool discovery --------------------------------------------------------
        query_mcp  = self._get_ums_tool_mcp_name(UMS_FUNC_QUERY_MEMORIES)
        promote_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_PROMOTE_MEM)

        if not (self._find_tool_server(query_mcp) and self._find_tool_server(promote_mcp)):
            self.logger.warning(
                "Skipping promotion checks: required UMS tools "
                f"('{UMS_FUNC_QUERY_MEMORIES}' or '{UMS_FUNC_PROMOTE_MEM}') unavailable."
            )
            return

        # ---- helper: safe tool call with timeout -----------------------------------
        async def _safe_call(tool_name: str, args: dict, *, tool_timeout: float = 15.0) -> dict:
            """Call `_execute_tool_call_internal` with a timeout & error wrapping."""
            try:
                return await asyncio.wait_for(
                    self._execute_tool_call_internal(tool_name, args, record_action=False),
                    timeout=tool_timeout,
                )
            except asyncio.TimeoutError:
                self.logger.error("%s timed-out after %.1fs", tool_name, tool_timeout)
                return {"success": False, "error": "timeout"}
            except Exception as exc:  # noqa: BLE001
                # full traceback only on DEBUG to avoid log spam
                self.logger.error("Exception in %s: %s", tool_name, exc, exc_info=self.logger.isEnabledFor(logging.DEBUG))
                return {"success": False, "error": str(exc)}

        # ---- build query argument payloads ----------------------------------------
        EPISODIC_BATCH_LIMIT = 5
        SEMANTIC_BATCH_LIMIT = 10
        PROCEDURAL_TYPES = {MemoryType.PROCEDURE.value, MemoryType.SKILL.value}

        episodic_args = {
            "workflow_id": wf_id,
            "memory_level": MemoryLevel.EPISODIC.value,
            "sort_by": "last_accessed",
            "sort_order": "DESC",
            "limit": EPISODIC_BATCH_LIMIT,
            "include_content": False,
        }
        semantic_args = {
            "workflow_id": wf_id,
            "memory_level": MemoryLevel.SEMANTIC.value,
            # memory_type intentionally omitted → get all, we'll filter
            "sort_by": "last_accessed",
            "sort_order": "DESC",
            "limit": SEMANTIC_BATCH_LIMIT,
            "include_content": False,
        }

        # ---- execute both queries concurrently ------------------------------------
        epis_res, sem_res = await asyncio.gather(
            _safe_call(query_mcp, episodic_args),
            _safe_call(query_mcp, semantic_args),
        )

        if self._shutdown_event.is_set():
            self.logger.debug("Promotion check aborted mid-scan: shutdown signalled.")
            return

        # ---- collect candidate memory ids -----------------------------------------
        candidate_ids: set[str] = set()

        if epis_res.get("success"):
            for mem in epis_res.get("memories", []):
                if isinstance(mem, dict) and (mid := mem.get("memory_id")):
                    candidate_ids.add(mid)
        else:
            self.logger.warning("Episodic memory query failed: %s", epis_res.get("error"))

        if sem_res.get("success"):
            for mem in sem_res.get("memories", []):
                if (
                    isinstance(mem, dict)
                    and (mid := mem.get("memory_id"))
                    and mem.get("memory_type") in PROCEDURAL_TYPES
                ):
                    candidate_ids.add(mid)
        else:
            self.logger.warning("Semantic memory query failed: %s", sem_res.get("error"))

        # ---- schedule promotion tasks ---------------------------------------------
        if not candidate_ids:
            self.logger.debug("No eligible memories for promotion.")
            return

        self.logger.debug(
            "📈 Promotion candidates (%d): %s",
            len(candidate_ids),
            ", ".join(_fmt_id(cid) for cid in candidate_ids),
        )

        for mem_id in candidate_ids:
            if self._shutdown_event.is_set():
                break
            # snapshot current context so that background task has stable ids
            self._start_background_task(
                AgentMasterLoop._check_and_trigger_promotion,
                memory_id=mem_id,
                workflow_id=wf_id,
                context_id=self.state.context_id,
            )

    async def _gather_lightweight_context(self) -> Dict[str, Any]:
        """
        Build a minimal context package during focus mode to avoid expensive UMS calls
        that would interrupt productive artifact creation work.
        """
        t_start = time.time()
        self.logger.info("🎯 Gathering lightweight context for LLM during focus mode "
                        f"(Loop: {self.state.current_loop}).")
        
        agent_retrieval_ts = datetime.now(timezone.utc).isoformat()
        
        # Minimal context with just essential state
        context_payload: Dict[str, Any] = {
            "agent_name": AGENT_NAME,
            "current_loop": self.state.current_loop,
            "current_plan_snapshot": [
                p.model_dump(exclude_none=True) for p in self.state.current_plan
            ],
            "last_action_summary": self.state.last_action_summary,
            "consecutive_error_count": self.state.consecutive_error_count,
            "last_error_details": copy.deepcopy(self.state.last_error_details),
            "needs_replan": self.state.needs_replan,
            "workflow_id": self.state.workflow_id,
            "cognitive_context_id_agent": self.state.context_id,
            "current_thought_chain_id": self.state.current_thought_chain_id,
            "retrieval_timestamp_agent_state": agent_retrieval_ts,
            "status_message_from_agent": "Lightweight context during focus mode.",
            "artifact_focus_mode": True,
            "focus_mode_message": "⚡ FOCUS MODE: Minimal context to maintain productivity flow",
            "errors_in_context_gathering": [],
            
            # Minimal goal context without expensive UMS calls
            "agent_assembled_goal_context": {
                "retrieved_at": agent_retrieval_ts,
                "current_goal_details_from_ums": None,
                "goal_stack_summary_from_agent_state": [
                    {
                        "goal_id": _fmt_id(g.get("goal_id")),
                        "description": (g.get("description") or "")[:150] + "...",
                        "status": g.get("status"),
                    }
                    for g in self.state.goal_stack[-3:]  # Just last 3
                    if isinstance(g, dict)
                ] if self.state.goal_stack else [],
                "data_source_comment": "Cached goal context during focus mode - no UMS calls.",
                "synchronization_status": "skipped_focus_mode",
            },
            
            # Skip expensive UMS context package
            "ums_context_package": {
                "focus_mode_notice": "UMS context package skipped during artifact focus mode",
                "workflow_goal": f"Active workflow: {_fmt_id(self.state.workflow_id)}" if self.state.workflow_id else "No workflow",
                "current_step": self.state.current_plan[0].description if self.state.current_plan else "No current step",
            },
            "ums_package_retrieval_status": "skipped_focus_mode",
            
            "processing_time_sec": time.time() - t_start,
        }
        
        # Always clear meta-feedback
        self.state.last_meta_feedback = None
        
        self.logger.info("Agent Context: Lightweight gathering complete. "
                        f"Time: {context_payload['processing_time_sec']:.3f}s "
                        "(Focus mode - UMS calls skipped)")
        
        return context_payload

    async def _gather_context(self) -> Dict[str, Any]:
        # Check if we can reuse cached context
        current_plan_hash = hashlib.md5(str(self.state.current_plan).encode()).hexdigest()
        cache_age = time.time() - self.state.context_cache_timestamp
        
        # Determine if this is a research workflow for smarter caching
        is_research_workflow = False
        cache_duration = 30  # Default 30 seconds
        
        if self.state.goal_stack and self.state.goal_stack[-1]:
            goal_desc = self.state.goal_stack[-1].get("description", "")
            is_research_workflow = await self._analyze_workflow_type_with_llm(goal_desc, "research_workflow")
        
        # **CRITICAL FIX**: Disable caching when storing memories or doing critical actions to prevent infinite loops
        recent_critical_actions = (
            self.state.last_action_summary and 
            any(action in self.state.last_action_summary for action in [
                "store_memory", "memory_id:", "record_artifact", "artifact_id:",
                "search_semantic", "hybrid_search", "query_memories"
            ])
        )
        
        # **CRITICAL FIX**: Force fresh context during active work to prevent infinite loops
        if recent_critical_actions or self.state.current_loop <= 3:
            cache_duration = 0  # NO CACHE during critical actions or early loops
            self.logger.info("🚫 CACHE DISABLED: Recent critical actions detected - forcing fresh context")
        elif is_research_workflow or self.state.artifact_focus_mode:
            cache_duration = 60  # Reduced from 120 to 60 seconds for research workflows
            
        # For research workflows, be less strict about plan hash matching
        plan_hash_matches = current_plan_hash == self.state.context_cache_plan_hash
        if is_research_workflow and not plan_hash_matches:
            # Check if plan changes are just minor (e.g., status updates)
            if self.state.last_context_cache and len(self.state.current_plan) > 0:
                # If first step description is similar, consider it a match
                current_step = self.state.current_plan[0].description.lower()
                if ("search" in current_step or "research" in current_step or "create" in current_step):
                    plan_hash_matches = True  # Allow cache reuse for research steps
        
        if (self.state.last_context_cache and 
            cache_age < cache_duration and
            plan_hash_matches and
            not self.state.last_error_details):
            cache_type = "research" if is_research_workflow else "standard"
            self.logger.info("🏎️ Using cached context ({} - age: {:.1f}s)".format(cache_type, cache_age))
            return self.state.last_context_cache
        
        # Generate new context and cache it
        context = await self._gather_context_fresh()
        self.state.last_context_cache = context
        self.state.context_cache_timestamp = time.time()
        self.state.context_cache_plan_hash = current_plan_hash
        return context

    async def _gather_research_optimized_context(self) -> Dict[str, Any]:
        """
        Build a research-optimized context package for research workflows.
        
        This is a middle ground between lightweight and comprehensive context,
        optimized for research tasks that need some UMS context but not everything.
        """
        t_start = time.time()
        self.logger.info("🔬 Gathering research-optimized context for LLM "
                        f"(Loop: {self.state.current_loop}).")
        
        agent_retrieval_ts = datetime.now(timezone.utc).isoformat()
        
        # Core agent state (same as lightweight)
        context_payload: Dict[str, Any] = {
            "agent_name": AGENT_NAME,
            "current_loop": self.state.current_loop,
            "current_plan_snapshot": [
                p.model_dump(exclude_none=True) for p in self.state.current_plan
            ],
            "last_action_summary": self.state.last_action_summary,
            "consecutive_error_count": self.state.consecutive_error_count,
            "last_error_details": copy.deepcopy(self.state.last_error_details),
            "needs_replan": self.state.needs_replan,
            "workflow_id": self.state.workflow_id,
            "cognitive_context_id_agent": self.state.context_id,
            "current_thought_chain_id": self.state.current_thought_chain_id,
            "retrieval_timestamp_agent_state": agent_retrieval_ts,
            "status_message_from_agent": "Research-optimized context for workflow efficiency.",
            "artifact_focus_mode": True,
            "focus_mode_message": "🔬 RESEARCH MODE: Optimized context for research workflows",
            "errors_in_context_gathering": [],
        }
        
        # Include goal context (important for research direction)
        goal_ctx_block: Dict[str, Any] = {
            "retrieved_at": agent_retrieval_ts,
            "current_goal_details_from_ums": None,
            "goal_stack_summary_from_agent_state": [
                {
                    "goal_id": _fmt_id(g.get("goal_id")),
                    "description": (g.get("description") or "")[:200] + "...",  # Longer for research context
                    "status": g.get("status"),
                }
                for g in self.state.goal_stack[-2:]  # Just last 2 goals
                if isinstance(g, dict)
            ] if self.state.goal_stack else [],
            "data_source_comment": "Goal context for research workflow focus.",
            "synchronization_status": "research_mode_cached",
        }
        context_payload["agent_assembled_goal_context"] = goal_ctx_block
        
        # Get minimal UMS context - just working memory and core context
        ums_tool_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_GET_RICH_CONTEXT_PACKAGE)
        ums_pkg: Dict[str, Any] = {}
        ums_pkg_status = "research_optimized"
        
        if self._find_tool_server(ums_tool_mcp) and self.state.workflow_id and self.state.context_id:
            try:
                # Minimal UMS parameters for research mode
                ums_params = {
                    "workflow_id": self.state.workflow_id,
                    "context_id": self.state.context_id,
                    "current_plan_step_description": self.state.current_plan[0].description if self.state.current_plan else "",
                    "fetch_limits": {
                        "recent_actions": 3,         # Fewer recent actions
                        "important_memories": 5,     # Key memories only
                        "key_thoughts": 3,           # Minimal thoughts
                        "proactive_memories": 3,     # Reduce proactive
                        "procedural_memories": 1,    # Minimal procedures
                        "link_traversal": 2,         # Minimal link traversal
                    },
                    "show_limits": {
                        "working_memory": 5,         # Focus on working memory
                        "link_traversal": 2,
                    },
                    "include_core_context": True,        # Still need workflow goal
                    "include_working_memory": True,      # Important for research
                    "include_proactive_memories": False, # Skip proactive in research mode
                    "include_relevant_procedures": False, # Skip procedures in research mode  
                    "include_contextual_links": False,   # Skip links in research mode
                    "compression_token_threshold": 8000, # Lower threshold
                    "compression_target_tokens": 3000,   # Smaller target
                }
                
                ums_raw = await self._execute_tool_call_internal(
                    ums_tool_mcp, ums_params, record_action=False
                )
                
                if ums_raw.get("success"):
                    pk = ums_raw.get("context_package", {})
                    if isinstance(pk, dict):
                        ums_pkg = pk
                        ums_pkg_status = "research_optimized_success"
                        self.logger.info("🔬 Research-optimized UMS context retrieved")
                    else:
                        ums_pkg = {"error": "Invalid UMS package type in research mode"}
                        ums_pkg_status = "research_optimized_invalid"
                else:
                    ums_pkg = {"error": f"UMS call failed in research mode: {ums_raw.get('error')}"}
                    ums_pkg_status = "research_optimized_failed"
                    
            except Exception as exc:
                context_payload["errors_in_context_gathering"].append(f"Research mode UMS error: {exc}")
                ums_pkg = {"error_research_mode": str(exc)}
                ums_pkg_status = "research_optimized_exception"
        else:
            ums_pkg = {"error": "UMS tool unavailable or missing workflow/context IDs"}
            ums_pkg_status = "research_optimized_tool_unavailable"
        
        context_payload.update(
            ums_context_package=ums_pkg,
            ums_package_retrieval_status=ums_pkg_status,
        )
        
        # Always clear meta-feedback in research mode too
        self.state.last_meta_feedback = None
        
        context_payload["processing_time_sec"] = time.time() - t_start
        
        self.logger.info("Agent Context: Research-optimized gathering complete. "
                        f"Time: {context_payload['processing_time_sec']:.3f}s "
                        "(Research mode - optimized for efficiency)")
        
        return context_payload
        
    async def _gather_context_fresh(self) -> Dict[str, Any]:
        """
        Build the full "context package" that is fed to the LLM for the *next* turn.
        """
        
        # Use lightweight context during artifact focus mode
        if self.state.artifact_focus_mode:
            return await self._gather_lightweight_context()
        
        # Also use lightweight context for simple tasks (2 steps or fewer)
        if (len(self.state.current_plan) <= 2 and 
            self.state.current_loop > 5 and  # Give agent time to understand task first
            self.state.successful_actions_since_reflection > 2):  # Ensure some real progress made
            self.logger.info("🏃 Using lightweight context for simple task (≤2 plan steps)")
            return await self._gather_lightweight_context()
        
        # Use research-optimized context for research workflows in mid-execution
        is_research_workflow = False
        if self.state.goal_stack and self.state.goal_stack[-1]:
            goal_desc = self.state.goal_stack[-1].get("description", "")
            is_research_workflow = await self._analyze_workflow_type_with_llm(goal_desc, "research_workflow")
        
        if (is_research_workflow and 
            self.state.current_loop > 3 and  # After initial setup
            self.state.artifact_focus_mode):  # In focus mode
            self.logger.info("🔬 Using research-optimized context for research workflow")
            return await self._gather_research_optimized_context()
        
        t_start = time.time()
        self.logger.info("🛰️ Gathering comprehensive context for LLM "
                        f"(Loop: {self.state.current_loop}).")

        # ------------------------------------------------------------------ #
        #  Helper utilities (local—do NOT escape the function's namespace)   #
        # ------------------------------------------------------------------ #
        def _safe_fmt(obj: Any, default: str = "N/A") -> str:
            """Best-effort stringify that never throws."""
            try:
                return _fmt_id(obj) if "id" in str(obj).lower() else str(obj)
            except Exception:  # pragma: no cover
                return default

        def _append_err(payload: Dict[str, Any], msg: str) -> None:
            """Append an error message to the context payload in a single place."""
            payload.setdefault("errors_in_context_gathering", []).append(msg)

        async def _fetch_goal_stack(goal_id: str) -> List[Dict[str, Any]]:
            """Wrapper so we can trap & log errors from _fetch_goal_stack_from_ums."""
            try:
                return await self._fetch_goal_stack_from_ums(goal_id) or []
            except Exception as exc:  # pragma: no cover
                err = (f"Exception while fetching goal stack for "
                    f"{_safe_fmt(goal_id)}: {exc}")
                self.logger.error(err, exc_info=True)
                _append_err(context_payload, err)
                return []

        # ------------------------- SECTION 0 ------------------------------ #
        #  Core "static" agent-state snapshot                                #
        # ------------------------------------------------------------------ #
        agent_retrieval_ts = datetime.now(timezone.utc).isoformat()
        context_payload: Dict[str, Any] = {
            "agent_name": AGENT_NAME,
            "current_loop": self.state.current_loop,
            "current_plan_snapshot": [
                p.model_dump(exclude_none=True) for p in self.state.current_plan
            ],
            "last_action_summary": self.state.last_action_summary,
            "consecutive_error_count": self.state.consecutive_error_count,
            "last_error_details": copy.deepcopy(self.state.last_error_details),
            "needs_replan": self.state.needs_replan,
            "workflow_stack_summary": [
                _safe_fmt(wf) for wf in self.state.workflow_stack[-3:]
            ],
            "meta_feedback": self.state.last_meta_feedback,   # cleared later
            "current_thought_chain_id": self.state.current_thought_chain_id,
            "retrieval_timestamp_agent_state": agent_retrieval_ts,
            "status_message_from_agent": "Context assembly by agent.",
            "errors_in_context_gathering": [],
        }

        # ensure we *always* clear meta-feedback exactly once,
        # even if later parts of this function raise.
        try:
            # ------------------------------------------------------------------ #
            #  SECTION 1 — Figure out workflow / basic IDs                        #
            # ------------------------------------------------------------------ #
            workflow_id_ctx = (
                self.state.workflow_stack[-1]
                if self.state.workflow_stack else self.state.workflow_id
            )
            context_id_ctx = self.state.context_id
            plan_step_desc_ctx = (
                self.state.current_plan[0].description
                if self.state.current_plan else DEFAULT_PLAN_STEP
            )

            if not workflow_id_ctx:
                # === EARLY EXIT: no workflow at all ===
                msg = "No Active Workflow. Agent will be prompted to create one."
                self.logger.warning(msg)
                context_payload.update(
                    status_message_from_agent=msg,
                    ums_package_retrieval_status="skipped_no_workflow",
                    agent_assembled_goal_context={
                        "retrieved_at": agent_retrieval_ts,
                        "current_goal_details_from_ums": None,
                        "goal_stack_summary_from_agent_state": [],
                        "data_source_comment": "No active workflow, so no UMS goal context available.",
                    },
                    processing_time_sec=time.time() - t_start,
                )
                return context_payload

            context_payload.update(
                workflow_id=workflow_id_ctx,
                cognitive_context_id_agent=context_id_ctx,
            )

            # ------------------------------------------------------------------ #
            #  SECTION 2 — Assemble / validate Goal Stack                         #
            # ------------------------------------------------------------------ #
            goal_ctx_block: Dict[str, Any] = {
                "retrieved_at": agent_retrieval_ts,
                "current_goal_details_from_ums": None,
                "goal_stack_summary_from_agent_state": [],
                "data_source_comment": "Goal context assembly by agent.",
                "synchronization_status": "pending",
            }

            if self.state.goal_stack:
                goal_ctx_block["goal_stack_summary_from_agent_state"] = [
                    {
                        "goal_id": _safe_fmt(g.get("goal_id")),
                        "description": (g.get("description") or "")[:150] + "...",
                        "status": g.get("status"),
                    }
                    for g in self.state.goal_stack[-CONTEXT_GOAL_STACK_SHOW_LIMIT:]
                    if isinstance(g, dict)
                ]

            if self.state.current_goal_id:
                self.logger.info("Agent Context: Current local UMS goal "
                                f"ID {_safe_fmt(self.state.current_goal_id)}. Verifying…")
                ums_stack = await _fetch_goal_stack(self.state.current_goal_id)

                if ums_stack:
                    ums_leaf = ums_stack[-1]
                    goal_ctx_block["current_goal_details_from_ums"] = ums_leaf

                    if ums_leaf.get("goal_id") != self.state.current_goal_id:
                        # -------- Mismatch -------- #
                        msg = (
                            f"Goal Sync Mismatch: Agent's current_goal_id "
                            f"{_safe_fmt(self.state.current_goal_id)} ≠ "
                            f"UMS leaf {_safe_fmt(ums_leaf.get('goal_id'))}."
                        )
                        self.logger.error(msg)
                        _append_err(context_payload, msg)
                        goal_ctx_block.update(
                            synchronization_status="mismatch_forcing_replan",
                            data_source_comment=(
                                "UMS goal stack fetched but mismatch detected. Forcing replan."
                            ),
                        )
                        # repair local state
                        self.state.last_error_details = {
                            "type": "GoalSyncError",
                            "error": msg,
                            "agent_current_goal_id": self.state.current_goal_id,
                            "ums_leaf_goal_id": ums_leaf.get("goal_id"),
                            "ums_leaf_goal_description": ums_leaf.get("description"),
                        }
                        self.state.needs_replan = True
                        self.state.goal_stack = ums_stack
                        self.state.current_goal_id = ums_leaf.get("goal_id")
                    else:
                        # -------- Synchronized -------- #
                        self.state.goal_stack = ums_stack
                        goal_ctx_block.update(
                            synchronization_status="synchronized_with_ums",
                            data_source_comment=(
                                "Goal stack fetched successfully and synchronized."
                            ),
                        )
                else:
                    # -------- Fetch failed -------- #
                    msg = (
                        "Goal Sync Error: Unable to fetch UMS goal stack for "
                        f"{_safe_fmt(self.state.current_goal_id)}."
                    )
                    self.logger.error(msg)
                    _append_err(context_payload, msg)
                    goal_ctx_block.update(
                        synchronization_status="fetch_failed_forcing_replan",
                        current_goal_details_from_ums={
                            "error_fetching_details": msg,
                            "goal_id_attempted": self.state.current_goal_id,
                        },
                        data_source_comment=(
                            "Critical error: Could not fetch UMS goal stack."
                        ),
                    )
                    self.state.last_error_details = {
                        "type": "GoalSyncError",
                        "error": msg,
                        "agent_current_goal_id": self.state.current_goal_id,
                        "recommendation": "Try state recovery before clearing goal",
                    }
                    
                    # Don't immediately clear goal state - try recovery first
                    self.logger.warning(f"Goal verification failed for {_safe_fmt(self.state.current_goal_id)}, but preserving goal state for recovery attempt")
                    self.state.needs_replan = True

            else:
                # -------- No current goal set -------- #
                goal_ctx_block.update(
                    synchronization_status="no_current_goal_in_agent_state",
                    data_source_comment=(
                        "No current_goal_id set in agent state. LLM will be prompted."
                    ),
                )
                self.logger.info(
                    "Agent Context: No current_goal_id for active workflow."
                )

            # refresh summary after any corrections above
            goal_ctx_block["goal_stack_summary_from_agent_state"] = [
                {
                    "goal_id": _safe_fmt(g.get("goal_id")),
                    "description": (g.get("description") or "")[:150] + "...",
                    "status": g.get("status"),
                }
                for g in self.state.goal_stack[-CONTEXT_GOAL_STACK_SHOW_LIMIT:]
            ] if self.state.goal_stack else []

            context_payload["agent_assembled_goal_context"] = goal_ctx_block

            # ------------------------------------------------------------------ #
            #  SECTION 3 — Fetch rich context package from UMS                   #
            # ------------------------------------------------------------------ #
            ums_tool_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_GET_RICH_CONTEXT_PACKAGE)
            ums_pkg: Dict[str, Any] = {}
            ums_pkg_status = "pending"

            if self._find_tool_server(ums_tool_mcp):
                focal_memory_hint = None
                if (
                    isinstance(self.state.last_error_details, dict)
                    and self.state.last_error_details.get("focal_memory_id_from_last_wm")
                ):
                    focal_memory_hint = self.state.last_error_details[
                        "focal_memory_id_from_last_wm"
                    ]

                # NOTE: we let UMS decide final focal memory if our hint is None/irrelevant
                adaptive_limits = self._get_adaptive_context_limits()
                ums_params = {
                    "workflow_id": workflow_id_ctx,
                    "context_id": context_id_ctx,
                    "current_plan_step_description": plan_step_desc_ctx,
                    "focal_memory_id_hint": focal_memory_hint,
                    "fetch_limits": adaptive_limits,
                    "show_limits": {
                        "working_memory": CONTEXT_WORKING_MEMORY_SHOW_LIMIT,
                        "link_traversal": CONTEXT_LINK_TRAVERSAL_SHOW_LIMIT,
                    },
                    "include_core_context": True,
                    "include_working_memory": True,
                    "include_proactive_memories": True,
                    "include_relevant_procedures": True,
                    "include_contextual_links": True,
                    "compression_token_threshold": CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD,
                    "compression_target_tokens": CONTEXT_COMPRESSION_TARGET_TOKENS,
                }

                try:
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug("Agent Context: Calling UMS tool "
                                        f"'{ums_tool_mcp}' with params: {ums_params}")
                    ums_raw = await self._execute_tool_call_internal(
                        ums_tool_mcp, ums_params, record_action=False
                    )

                    if ums_raw.get("success"):
                        pk = ums_raw.get("context_package", {})
                        if isinstance(pk, dict):
                            ums_pkg = pk
                            ums_pkg_status = "success"
                            internal_errs = ums_raw.get("errors", [])
                            if internal_errs:
                                for e in internal_errs:
                                    _append_err(context_payload, f"UMS_PKG_ERR: {e}")
                            self.logger.info("Agent Context: Received rich context package.")
                        else:
                            msg = (f"UMS tool '{ums_tool_mcp}' returned invalid "
                                f"type {type(pk)} for 'context_package'.")
                            self.logger.error(msg)
                            _append_err(context_payload, msg)
                            ums_pkg = {"error_ums_pkg_invalid_type": msg}
                            ums_pkg_status = "invalid_package_type"
                    else:
                        msg = (f"UMS rich context pkg retrieval failed: "
                            f"{ums_raw.get('error', 'Unknown error')}")
                        self.logger.warning(msg)
                        _append_err(context_payload, msg)
                        ums_pkg = {"error_ums_pkg_retrieval": msg}
                        ums_pkg_status = "failed_ums_tool_call"
                except asyncio.CancelledError:  # propagate cancellations
                    raise
                except Exception as exc:  # pragma: no cover
                    msg = f"Exception calling UMS for rich context pkg: {exc}"
                    self.logger.error(msg, exc_info=True)
                    _append_err(context_payload, msg)
                    ums_pkg = {"error_ums_pkg_exception": msg}
                    ums_pkg_status = "exception_calling_ums_tool"
            else:
                msg = f"UMS tool for '{UMS_FUNC_GET_RICH_CONTEXT_PACKAGE}' unavailable."
                self.logger.error(msg)
                _append_err(context_payload, msg)
                ums_pkg = {"error_ums_package_tool_unavailable": msg}
                ums_pkg_status = "tool_unavailable"

            context_payload.update(
                ums_context_package=ums_pkg,
                ums_package_retrieval_status=ums_pkg_status,
            )

            if ums_pkg.get("ums_compression_details"):
                self.logger.info("Agent Context: UMS package includes compression "
                                "details: %s", ums_pkg["ums_compression_details"])

            # ------------------------------------------------------------------ #
            #  SECTION 4 — Final status / timing                                  #
            # ------------------------------------------------------------------ #
            n_errors = len(context_payload["errors_in_context_gathering"])
            if n_errors == 0:
                context_payload["status_message_from_agent"] = (
                    "Workflow active. Context ready."
                )
            else:
                context_payload["status_message_from_agent"] = (
                    f"Workflow active. Context ready with {n_errors} gathering errors."
                )

        finally:
            # always clear meta-feedback exactly once
            self.state.last_meta_feedback = None
            context_payload["processing_time_sec"] = time.time() - t_start

        # Add file path guidance if agent is likely to create files
        if self._is_likely_file_creation_task():
            context_payload["file_creation_guidance"] = {
                "safe_storage_paths": [
                    "/home/ubuntu/ultimate_mcp_server/storage/",
                    "~/.ultimate_mcp_server/artifacts/",
                    "./data/"
                ],
                "recommended_path": self._get_safe_file_path("agent_output.txt"),
                "avoid_paths": ["/usr/", "/etc/", "/var/", "/root/", "/sys/", "/proc/"],
                "tips": [
                    "Use safe storage locations to avoid permission errors",
                    "System will auto-fix many path issues, but start with safe paths",
                    f"Use {UMS_FUNC_DIAGNOSE_FILE_ACCESS} tool for permission diagnostics"
                ]
            }

        # Add multi-tool guidance if agent is dealing with complex operations
        if await self._should_suggest_multi_tool_guidance():
            context_payload["multi_tool_guidance_available"] = {
                "tool": UMS_FUNC_GET_MULTI_TOOL_GUIDANCE,
                "when_to_use": [
                    "Planning complex multi-step operations",
                    "When unsure about tool combinations",
                    "After multiple consecutive errors",
                    "For comprehensive analysis workflows"
                ],
                "benefits": [
                    "Get proven tool sequence patterns",
                    "Learn tool synergies and best practices", 
                    "Optimize multi-tool operations",
                    "Reduce planning errors"
                ]
            }

        self.logger.info("Agent Context: Gathering complete. Status: %s. "
                        "Time: %.3fs",
                        context_payload["status_message_from_agent"],
                        context_payload["processing_time_sec"])
        if context_payload["errors_in_context_gathering"]:
            self.logger.info("Agent Context: Errors encountered: %s",
                            context_payload["errors_in_context_gathering"])

        return context_payload


    async def execute_llm_decision(
        self,
        llm_decision: Dict[str, Any],        # direct output from MCPClient.process_agent_llm_turn
    ) -> bool:                               # True → continue loop, False → stop
        """
        Executes a single LLM decision returned by MCPClient, handling tool calls,
        thought recording, plan updates, etc.  All side-effects on `self.state`
        (errors, last_action_summary, needs_replan, counters) are preserved.

        Returns
        -------
        bool
            True  – continue main loop  
            False – stop (goal achieved, shutdown, workflow missing, max errors, …)
        """
        # --------------------------------------------------------------------- #
        # 0.  Logging & fast-path resets                                        #
        # --------------------------------------------------------------------- #
        self.logger.info(
            "AML EXEC_DECISION: Entered for Loop %s. Current WF: %s, Current UMS Goal: %s",
            self.state.current_loop, _fmt_id(self.state.workflow_id), _fmt_id(self.state.current_goal_id),
        )
        self.logger.debug("AML EXEC_DECISION: Raw LLM decision: %s", str(llm_decision)[:500])

        # Reset stale error info if we are on a clean slate
        if not self.state.needs_replan:
            self.state.last_error_details = None

        decision_type: str = llm_decision.get("decision", "") or ""
        tool_name_in_turn: Optional[str] = llm_decision.get("tool_name")

        # This will be filled by each handler so the heuristic updater has a payload.
        tool_call_envelope: Optional[Dict[str, Any]] = None

        # --------------------------------------------------------------------- #
        # 1.  Helper functions (inner scope so this remains drop-in)            #
        # --------------------------------------------------------------------- #
        def _construct_envelope(
            success: bool,
            data: Any = None,
            *,
            error_type: Optional[str] = None,
            error_message: Optional[str] = None,
            status_code: Optional[int] = None,
            details: Any = None,
        ) -> Dict[str, Any]:
            """Return the standardized envelope used across AML."""
            return {
                "success": success,
                "data": data,
                "error_type": error_type,
                "error_message": error_message,
                "status_code": status_code,
                "details": details,
            }

        def _build_success_summary(data_payload: Any) -> str:
            """Create a one-line human log based on UMS payload content."""
            if data_payload is None:
                return "Success (No data payload from UMS tool)."

            if not isinstance(data_payload, dict):
                return f"Success (Data: {str(data_payload)[:50]}...)"

            summary_keys = (
                "summary", "message", "memory_id", "action_id", "artifact_id",
                "link_id", "thought_chain_id", "thought_id", "state_id", "report",
                "visualization", "goal_id", "workflow_id", "title",
            )
            for k in summary_keys:
                if k in data_payload and data_payload[k] is not None:
                    val = str(data_payload[k])
                    return f"{k}: {_fmt_id(val) if 'id' in k.lower() else val}"

            generic_parts = [
                f"{k}={_fmt_id(str(v)) if 'id' in k.lower() else str(v)[:20]}"
                for k, v in data_payload.items()
                if v is not None and k not in ("success", "processing_time")
            ][:3]
            return "Success. Data: " + ", ".join(generic_parts) if generic_parts else \
                "Success (UMS payload has no distinct summary key)."

        async def _handle_mcp_executed_tool() -> None:
            """Handle decision_type == 'tool_executed_by_mcp'."""
            nonlocal tool_call_envelope, tool_name_in_turn
            tool_name_in_turn = llm_decision.get("tool_name")
            arguments_used = llm_decision.get("arguments", {})
            ums_payload = llm_decision.get("result")
            
            # Handle atomic tool call processing fields
            deferred_calls = llm_decision.get("deferred_tool_calls", [])
            total_tools = llm_decision.get("total_tools_this_turn", 1)
            agent_msg = llm_decision.get("agent_message", "")

            self.logger.info(
                "AML EXEC_DECISION: 'tool_executed_by_mcp' Tool='%s' Total tools this turn=%s Deferred=%s",
                tool_name_in_turn, total_tools, len(deferred_calls),
            )
            
            # Store deferred tool calls for next turn
            if deferred_calls:
                self.state.deferred_tool_calls = deferred_calls
                self.logger.info(f"AML EXEC_DECISION: Stored {len(deferred_calls)} deferred tool calls for next turn")
            else:
                self.state.deferred_tool_calls = []
            
            # Store atomic decision info for context
            self.state.last_atomic_decision_info = {
                "total_tools_requested": total_tools,
                "tools_processed": 1,
                "tools_deferred": len(deferred_calls),
                "agent_message": agent_msg,
                "decision_type": "tool_executed_by_mcp"
            }

            envelope = _construct_envelope(False, ums_payload)          # filled out below
            critical_state_tools = {
                UMS_FUNC_CREATE_WORKFLOW, UMS_FUNC_CREATE_GOAL, UMS_FUNC_UPDATE_GOAL_STATUS,
                UMS_FUNC_UPDATE_WORKFLOW_STATUS, UMS_FUNC_SAVE_COGNITIVE_STATE, UMS_FUNC_LOAD_COGNITIVE_STATE,
            }
            base_tool = self._get_base_function_name(tool_name_in_turn)

            if isinstance(ums_payload, dict):
                if ums_payload.get("success", False):
                    envelope["success"] = True
                else:
                    envelope.update(
                        success=False,
                        error_type=ums_payload.get("error_type", "UMSToolReportedFailureInPayload"),
                        error_message=ums_payload.get("error_message", ums_payload.get("error", "UMS tool reported failure.")),
                        status_code=ums_payload.get("status_code"),
                        details=ums_payload.get("details"),
                    )
            elif ums_payload is not None:                              # non-dict payload
                envelope["success"] = True
                if base_tool in critical_state_tools:
                    self.logger.error(
                        "AML EXEC_DECISION: Critical tool '%s' returned non-dict payload (%s). Marking as failure.",
                        tool_name_in_turn, type(ums_payload),
                    )
                    envelope.update(
                        success=False,
                        error_type="UMSMalformedPayload",
                        error_message=f"Critical UMS tool '{tool_name_in_turn}' returned non-dictionary payload.",
                    )
            else:                                                      # payload None
                envelope.update(
                    success=False,
                    error_type="MissingUMSPayloadFromMCP",
                    error_message=f"MCPClient reported tool '{tool_name_in_turn}' executed but payload missing/None.",
                )

            tool_call_envelope = envelope
            await self._handle_workflow_and_goal_side_effects(base_tool, arguments_used, envelope)

            # Build `last_action_summary`
            if envelope["success"]:
                summary = _build_success_summary(envelope["data"])
            else:
                summary = f"Failed ({envelope.get('error_type', 'ToolExecutionError')}): " \
                        f"{str(envelope.get('error_message', 'Unknown'))[:100]}"
            if envelope.get("status_code"):
                summary += f" (Code: {envelope['status_code']})"

            # Build action summary with atomic processing info
            base_summary = f"{tool_name_in_turn} (executed by LLM via MCP) -> {summary}"
            if deferred_calls:
                base_summary += f" | {len(deferred_calls)} tool calls deferred to next turn"
            if agent_msg:
                base_summary += f" | Note: {agent_msg}"
            
            self.state.last_action_summary = base_summary
            self.logger.info("🏁 LLM-executed tool summary: %s", self.state.last_action_summary)

            if not envelope["success"]:
                self.state.last_error_details = {
                    "tool": tool_name_in_turn,
                    "args": arguments_used,
                    "error": envelope.get("error_message"),
                    "status_code": envelope.get("status_code"),
                    "type": envelope.get("error_type", "ToolExecutionError"),
                    "details": envelope.get("details"),
                }
                self.state.needs_replan = self.state.needs_replan or True

        async def _handle_multiple_mcp_executed_tools() -> None:
            """Handle decision_type == 'multiple_tools_executed_by_mcp'."""
            nonlocal tool_call_envelope, tool_name_in_turn
            
            executed_tools = llm_decision.get("executed_tools", [])
            deferred_calls = llm_decision.get("deferred_tool_calls", [])
            total_tools = llm_decision.get("total_tools_this_turn", len(executed_tools))
            agent_msg = llm_decision.get("agent_message", "")

            self.logger.info(
                "AML EXEC_DECISION: 'multiple_tools_executed_by_mcp' Tools=%s Total tools this turn=%s Deferred=%s",
                len(executed_tools), total_tools, len(deferred_calls),
            )
            
            # Store deferred tool calls for next turn
            if deferred_calls:
                self.state.deferred_tool_calls = deferred_calls
                self.logger.info(f"AML EXEC_DECISION: Stored {len(deferred_calls)} deferred tool calls for next turn")
            else:
                self.state.deferred_tool_calls = []
            
            # Store atomic decision info for context
            self.state.last_atomic_decision_info = {
                "total_tools_requested": total_tools,
                "tools_processed": len(executed_tools),
                "tools_deferred": len(deferred_calls),
                "agent_message": agent_msg,
                "decision_type": "multiple_tools_executed_by_mcp"
            }

            if not executed_tools:
                self.logger.error("AML EXEC_DECISION: No executed tools provided for multiple_tools_executed_by_mcp")
                tool_call_envelope = _construct_envelope(False, error_type="InvalidMultipleToolsDecision", error_message="No executed tools provided")
                return

            # Process each tool result and handle side effects
            successful_tools = []
            failed_tools = []
            
            for tool_info in executed_tools:
                tool_name = tool_info.get("tool_name", "unknown")
                arguments_used = tool_info.get("arguments", {})
                ums_payload = tool_info.get("result", {})
                
                # Build envelope for this tool
                envelope = _construct_envelope(False, ums_payload)
                base_tool = self._get_base_function_name(tool_name)
                
                if isinstance(ums_payload, dict):
                    if ums_payload.get("success", False):
                        envelope["success"] = True
                        successful_tools.append(tool_name)
                    else:
                        envelope.update(
                            success=False,
                            error_type=ums_payload.get("error_type", "UMSToolReportedFailureInPayload"),
                            error_message=ums_payload.get("error_message", ums_payload.get("error", "UMS tool reported failure.")),
                            status_code=ums_payload.get("status_code"),
                            details=ums_payload.get("details"),
                        )
                        failed_tools.append(tool_name)
                elif ums_payload is not None:
                    envelope["success"] = True
                    successful_tools.append(tool_name)
                else:
                    envelope.update(
                        success=False,
                        error_type="MissingUMSPayloadFromMCP",
                        error_message=f"MCPClient reported tool '{tool_name}' executed but payload missing/None.",
                    )
                    failed_tools.append(tool_name)
                
                # Handle side effects for each tool
                await self._handle_workflow_and_goal_side_effects(base_tool, arguments_used, envelope)
            
            # Set the overall tool name for reporting
            tool_name_in_turn = f"MultiTool[{len(executed_tools)}]"
            
            # Create summary envelope
            if failed_tools:
                # Some tools failed
                summary = f"Executed {len(executed_tools)} tools: {len(successful_tools)} succeeded, {len(failed_tools)} failed"
                tool_call_envelope = _construct_envelope(
                    False,
                    error_type="PartialMultipleToolsFailure",
                    error_message=f"Some tools failed: {', '.join(failed_tools)}",
                    data={
                        "executed_tools": [t["tool_name"] for t in executed_tools],
                        "successful_tools": successful_tools,
                        "failed_tools": failed_tools,
                        "summary": summary
                    }
                )
            else:
                # All tools succeeded
                summary = f"Successfully executed {len(executed_tools)} tools: {', '.join(successful_tools)}"
                tool_call_envelope = _construct_envelope(
                    True,
                    data={
                        "executed_tools": [t["tool_name"] for t in executed_tools],
                        "successful_tools": successful_tools,
                        "summary": summary
                    }
                )
            
            # Build action summary with atomic processing info
            base_summary = f"{len(executed_tools)} tools executed by LLM via MCP -> {summary}"
            if deferred_calls:
                base_summary += f" | {len(deferred_calls)} tool calls deferred to next turn"
            if agent_msg:
                base_summary += f" | Note: {agent_msg}"
            
            self.state.last_action_summary = base_summary
            self.logger.info("🏁 LLM-executed multiple tools summary: %s", self.state.last_action_summary)

            if failed_tools:
                self.state.last_error_details = {
                    "tools": failed_tools,
                    "error": f"Some tools failed in multi-tool execution: {', '.join(failed_tools)}",
                    "type": "PartialMultipleToolsFailure",
                    "details": {"successful_tools": successful_tools, "failed_tools": failed_tools},
                }
                self.state.needs_replan = self.state.needs_replan or True

        async def _handle_call_tool() -> None:
            """Handle decision_type == 'call_tool' (AML executes)."""
            nonlocal tool_call_envelope, tool_name_in_turn
            tool_name_in_turn = llm_decision.get("tool_name")
            args = llm_decision.get("arguments", {})
            
            # Handle atomic tool call processing fields
            deferred_calls = llm_decision.get("deferred_tool_calls", [])
            total_tools = llm_decision.get("total_tools_this_turn", 1)
            agent_msg = llm_decision.get("agent_message", "")

            self.logger.info(
                "AML EXEC_DECISION: 'call_tool' → will execute '%s' Total tools this turn=%s Deferred=%s",
                tool_name_in_turn, total_tools, len(deferred_calls),
            )
            
            # Store deferred tool calls for next turn
            if deferred_calls:
                self.state.deferred_tool_calls = deferred_calls
                self.logger.info(f"AML EXEC_DECISION: Stored {len(deferred_calls)} deferred tool calls for next turn")
            else:
                self.state.deferred_tool_calls = []
            
            # Store atomic decision info for context
            self.state.last_atomic_decision_info = {
                "total_tools_requested": total_tools,
                "tools_processed": 1,
                "tools_deferred": len(deferred_calls),
                "agent_message": agent_msg,
                "decision_type": "call_tool"
            }

            if not self.state.current_plan or not self.state.current_plan[0].description:
                err_msg = "Plan empty before tool call." if not self.state.current_plan else \
                        "Current plan step invalid (no description)."
                self.logger.error("AML EXEC_DECISION: %s", err_msg)
                self.state.last_error_details = {"tool": tool_name_in_turn, "args": args,
                                                "error": err_msg, "type": "PlanValidationError"}
                self.state.needs_replan = True
                tool_call_envelope = _construct_envelope(False, error_type="PlanValidationError", error_message=err_msg)
                return

            if not tool_name_in_turn:
                err_msg = "Missing tool name from LLM decision."
                self.logger.error("AML EXEC_DECISION: %s", err_msg)
                self.state.last_error_details = {"decision": llm_decision, "error": err_msg, "type": "LLMOutputError"}
                self.state.needs_replan = True
                tool_call_envelope = _construct_envelope(False, error_type="LLMOutputError", error_message=err_msg)
                return

            deps = self.state.current_plan[0].depends_on or []
            tool_call_envelope = await self._execute_tool_call_internal(
                tool_name_in_turn,
                args,
                record_action=(tool_name_in_turn != AGENT_TOOL_UPDATE_PLAN),
                planned_dependencies=deps,
            )

        async def _handle_call_multiple_tools() -> None:
            """Handle decision_type == 'call_multiple_tools' (AML executes multiple tools in sequence)."""
            nonlocal tool_call_envelope, tool_name_in_turn
            
            tool_calls = llm_decision.get("tool_calls", [])
            total_tools = llm_decision.get("total_tools_this_turn", len(tool_calls))
            agent_msg = llm_decision.get("agent_message", "")

            self.logger.info(
                "AML EXEC_DECISION: 'call_multiple_tools' → will execute %s tools sequentially",
                len(tool_calls),
            )
            
            # Store atomic decision info for context
            self.state.last_atomic_decision_info = {
                "total_tools_requested": total_tools,
                "tools_processed": len(tool_calls),
                "tools_deferred": 0,
                "agent_message": agent_msg,
                "decision_type": "call_multiple_tools"
            }

            if not self.state.current_plan or not self.state.current_plan[0].description:
                err_msg = "Plan empty before multi-tool call."
                self.logger.error("AML EXEC_DECISION: %s", err_msg)
                self.state.last_error_details = {"error": err_msg, "type": "PlanValidationError"}
                self.state.needs_replan = True
                tool_call_envelope = _construct_envelope(False, error_type="PlanValidationError", error_message=err_msg)
                return

            if not tool_calls:
                err_msg = "Missing tool calls from LLM decision."
                self.logger.error("AML EXEC_DECISION: %s", err_msg)
                self.state.last_error_details = {"decision": llm_decision, "error": err_msg, "type": "LLMOutputError"}
                self.state.needs_replan = True
                tool_call_envelope = _construct_envelope(False, error_type="LLMOutputError", error_message=err_msg)
                return

            # Execute multiple tools sequentially
            deps = self.state.current_plan[0].depends_on or []
            results = []
            last_tool_name = ""
            
            for i, tool_call in enumerate(tool_calls):
                tool_name = tool_call.get("name")
                tool_args = tool_call.get("input", {})
                last_tool_name = tool_name  # noqa: F841
                
                self.logger.info(f"AML EXEC_DECISION: Executing tool {i+1}/{len(tool_calls)}: {tool_name}")
                
                try:
                    result = await self._execute_tool_call_internal(
                        tool_name,
                        tool_args,
                        record_action=True,
                        planned_dependencies=deps if i == 0 else [],  # Only first tool has dependencies
                    )
                    results.append(result)
                    
                    # If any tool fails, stop and return the failure
                    if not result.get("success", False):
                        self.logger.warning(f"AML EXEC_DECISION: Multi-tool execution stopped at tool {i+1} due to failure")
                        tool_call_envelope = result
                        tool_name_in_turn = tool_name
                        return
                        
                except Exception as e:
                    self.logger.error(f"AML EXEC_DECISION: Exception executing tool {i+1}/{len(tool_calls)}: {e}")
                    tool_call_envelope = _construct_envelope(
                        False, 
                        error_type="ToolExecutionException", 
                        error_message=f"Exception during multi-tool execution: {str(e)}"
                    )
                    tool_name_in_turn = tool_name
                    return
            
            # All tools succeeded - create summary result
            tool_name_in_turn = f"MultiTool[{len(tool_calls)}]"
            successful_tools = [tc.get("name") for tc in tool_calls]
            
            tool_call_envelope = _construct_envelope(
                True,
                data={
                    "executed_tools": successful_tools,
                    "total_executed": len(tool_calls),
                    "summary": f"Successfully executed {len(tool_calls)} tools: {', '.join(successful_tools)}"
                }
            )

        async def _handle_thought_process() -> None:
            """Handle decision_type == 'thought_process'."""
            nonlocal tool_call_envelope, tool_name_in_turn
            content = llm_decision.get("content", "")
            tool_name_in_turn = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_THOUGHT)

            self.logger.info("AML EXEC_DECISION: 'thought_process' Content preview=%s", str(content)[:100])

            if content:
                # If LLM returned a UMS payload dict, bypass record_thought and wrap directly
                if isinstance(content, dict):
                    tool_call_envelope = _construct_envelope(True, data=content)
                else:
                    # Ensure content is a string for UMS record_thought
                    if isinstance(content, list):
                        content_str = json.dumps(content)
                    elif not isinstance(content, str):
                        content_str = str(content)
                    else:
                        content_str = content
                    tool_call_envelope = await self._execute_tool_call_internal(
                        tool_name_in_turn,
                        {"content": content_str, "thought_type": ThoughtType.INFERENCE.value},
                        record_action=False,
                    )
            else:
                err_msg = "Missing thought content from LLM."
                self.logger.warning("AML EXEC_DECISION: %s", err_msg)
                self.state.last_action_summary = "LLM Thought: No content."
                self.state.last_error_details = {"decision": "thought_process", "error": err_msg, "type": "LLMOutputError"}
                self.state.needs_replan = True
                tool_call_envelope = _construct_envelope(False, error_type="LLMOutputError", error_message=err_msg)

        # --------------------------------------------------------------------- #
        # 2.  Dispatcher                                                        #
        # --------------------------------------------------------------------- #
        handlers: Dict[str, Callable[[], Awaitable[None]]] = {
            "tool_executed_by_mcp": _handle_mcp_executed_tool,
            "multiple_tools_executed_by_mcp": _handle_multiple_mcp_executed_tools,
            "call_tool":              _handle_call_tool,
            "call_multiple_tools":    _handle_call_multiple_tools,
            "thought_process":        _handle_thought_process,
            "complete":               lambda: asyncio.sleep(0),  # handled as simple envelope below
            "complete_with_artifact": lambda: asyncio.sleep(0),
            "plan_update":            lambda: asyncio.sleep(0),
            "error":                  lambda: asyncio.sleep(0),  # handled inline below
        }

        if decision_type in handlers:
            await handlers[decision_type]()
        else:
            self.logger.error(
                "AML EXEC_DECISION: Unexpected decision type '%s'. Full decision: %s",
                decision_type, str(llm_decision)[:200],
            )
            err_msg = f"Unexpected decision type '{decision_type}' from MCPClient."
            self.state.last_action_summary = f"Agent Error: {err_msg}"
            self.state.last_error_details = {"error": err_msg, "type": "AgentError", "llm_decision_payload": llm_decision}
            self.state.needs_replan = True
            tool_call_envelope = _construct_envelope(False, error_type="AgentError", error_message=err_msg)

        # Simple envelopes for "non-action" decisions
        if decision_type in {"complete", "complete_with_artifact"}:
            completion_summary = llm_decision.get("summary", "Goal completed")
            tool_call_envelope = _construct_envelope(True, data={"message": f"LLM signaled overall completion: {completion_summary}"})
            
            # CRITICAL: Trigger actual goal completion when LLM signals completion
            self.state.goal_achieved_flag = True
            self.state.last_action_summary = f"Goal Achieved: {completion_summary}"
            
            # Update current goal status to completed
            if self.state.current_goal_id:
                try:
                    update_goal_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_GOAL_STATUS)
                    if self._find_tool_server(update_goal_mcp):
                        await self._execute_tool_call_internal(
                            update_goal_mcp,
                            {
                                "goal_id": self.state.current_goal_id,
                                "status": GoalStatus.COMPLETED.value,
                                "completion_message": completion_summary
                            },
                            record_action=False
                        )
                        self.logger.info(f"✅ Updated goal {self.state.current_goal_id} status to completed: {completion_summary}")
                except Exception as e:
                    self.logger.error(f"Failed to update goal status on completion: {e}")
            
            # Update workflow status if this is the root goal
            if self.state.workflow_id and len(self.state.goal_stack) <= 1:
                try:
                    update_wf_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_WORKFLOW_STATUS)
                    if self._find_tool_server(update_wf_mcp):
                        await self._execute_tool_call_internal(
                            update_wf_mcp,
                            {
                                "workflow_id": self.state.workflow_id,
                                "status": WorkflowStatus.COMPLETED.value,
                                "completion_message": completion_summary
                            },
                            record_action=False
                        )
                        self.logger.info(f"✅ Updated workflow {self.state.workflow_id} status to completed")
                except Exception as e:
                    self.logger.error(f"Failed to update workflow status on completion: {e}")
            
            self.logger.info("🎉 AML EXEC_DECISION: LLM signaled '%s' - Goal achieved: %s", decision_type, completion_summary)
        elif decision_type == "plan_update":
            tool_call_envelope = _construct_envelope(True, data={"message": "LLM textual plan received for processing."})
            self.logger.info("AML EXEC_DECISION: LLM provided textual plan update.")
        elif decision_type == "error":
            err_msg = llm_decision.get("message", "Unknown error from MCPClient")
            self.logger.error("AML EXEC_DECISION: MCPClient error decision: %s", err_msg)
            self.state.last_action_summary = f"LLM Decision Error (MCPClient): {err_msg[:100]}"
            if not self.state.last_error_details:
                self.state.last_error_details = {"error": err_msg,
                                                "type": llm_decision.get("error_type_for_agent", "LLMError")}
            self.state.needs_replan = True
            tool_call_envelope = _construct_envelope(
                False, error_type=llm_decision.get("error_type_for_agent", "LLMError"), error_message=err_msg,
            )

        # --------------------------------------------------------------------- #
        # 3.  Heuristic plan update                                             #
        # --------------------------------------------------------------------- #
        self.logger.debug(
            "AML EXEC_DECISION: Running heuristic update… Decision='%s' Tool='%s' Envelope preview=%s",
            decision_type, tool_name_in_turn, str(tool_call_envelope)[:200],
        )
        await self._apply_heuristic_plan_update(llm_decision, tool_call_envelope)

        # --------------------------------------------------------------------- #
        # 4.  Stop-conditions: max errors / shutdown / goal achieved / no WF    #
        # --------------------------------------------------------------------- #
        if self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
            self.logger.critical(
                "AML EXEC_DECISION: Max consecutive errors reached (%s/%s). Aborting workflow.",
                self.state.consecutive_error_count, MAX_CONSECUTIVE_ERRORS,
            )
            update_wf_status_mcp = self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_WORKFLOW_STATUS)
            if self.state.workflow_id and self._find_tool_server(update_wf_status_mcp):
                await self._execute_tool_call_internal(
                    update_wf_status_mcp,
                    {
                        "workflow_id": self.state.workflow_id,
                        "status": WorkflowStatus.FAILED.value,
                        "completion_message": f"Aborted after {self.state.consecutive_error_count} consecutive errors.",
                    },
                    record_action=False,
                )
            else:
                if self.state.workflow_id:
                    self.logger.warning("Unable to update workflow status – tool unavailable.")
                self.state.workflow_id = None

            await self._save_agent_state()
            return False

        # Persist state after every turn
        await self._save_agent_state()
        self.logger.info(
            "AML EXEC_DECISION: Post-turn state → WF=%s Goal=%s needs_replan=%s errors=%s plan_steps=%s",
            _fmt_id(self.state.workflow_id), _fmt_id(self.state.current_goal_id),
            self.state.needs_replan, self.state.consecutive_error_count,
            len(self.state.current_plan) if self.state.current_plan else "N/A",
        )

        # Final stop checks
        if self.state.goal_achieved_flag:
            self.logger.info("AML EXEC_DECISION: Goal achieved. Stopping loop.")
            return False
        if self._shutdown_event.is_set():
            self.logger.info("AML EXEC_DECISION: Shutdown signaled. Stopping loop.")
            return False
        if not self.state.workflow_id:
            # Check if we just created a workflow (from tool_name_mcp) but it hasn't been fully synced
            if tool_name_in_turn and self._get_base_function_name(tool_name_in_turn) == UMS_FUNC_CREATE_WORKFLOW:
                if isinstance(tool_call_envelope, dict) and tool_call_envelope.get("success"):
                    workflow_data = tool_call_envelope.get("data", {})
                    new_wf_id = workflow_data.get("workflow_id")
                    if new_wf_id:
                        self.logger.info(f"AML EXEC_DECISION: Workflow just created (ID: {_fmt_id(new_wf_id)}). Continuing loop.")
                        self.state.workflow_id = new_wf_id
                        if not self.state.workflow_stack:
                            self.state.workflow_stack = [new_wf_id]
                        return True
            
            self.logger.info("AML EXEC_DECISION: No active workflow. Stopping loop.")
            return False

        self.logger.info("AML EXEC_DECISION: Continue loop. WF=%s", _fmt_id(self.state.workflow_id))
        return True

