"""
EideticEngine Agent Master Loop (AML) - v4.2 - Direct MCP Tool Names
====================================================================

This module implements the core orchestration logic for the EideticEngine
AI agent. It manages the primary think-act cycle, interacts with the
Unified Memory System (UMS) via MCPClient, leverages an LLM
for decision-making and planning.

** V4.2 focuses on simplifying tool name management by directly using
original MCP tool names when the agent needs to call UMS tools, and correctly
handling de-sanitized original MCP names from LLM decisions. This eliminates
internal AML-specific tool constants for UMS functions and their associated map. **

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
import copy
import dataclasses
import json
import logging
import math
import os
import random
import re
import sys
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Union

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
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DECISION = "decision"
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
    REFLECTION = "reflection"
    CRITIQUE = "critique"
    SUMMARY = "summary"
    USER_GUIDANCE = "user_guidance"
    INSIGHT = "insight"


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


# Logger Setup
log = logging.getLogger("AgentMasterLoop")
if not logging.root.handlers and not log.handlers:
    logging.basicConfig(
        level=os.environ.get("AGENT_LOOP_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    log.warning("AgentMasterLoop logger self-initialized. If run integrated, mcp_client_multi should configure logging.")

LOG_LEVEL_ENV = os.environ.get("AGENT_LOOP_LOG_LEVEL", "INFO").upper()
log.setLevel(getattr(logging, LOG_LEVEL_ENV, logging.INFO))
if log.level <= logging.DEBUG:
    log.info("Verbose logging enabled for Agent loop.")

# ==========================================================================
# CONSTANTS
# ==========================================================================
AGENT_STATE_FILE = "agent_loop_state_v4.2_direct_mcp_names.json"
AGENT_NAME = "EidenticEngine4.2-DirectMCPNames"

AGENT_LOOP_TEMP_DIR = Path(".") / ".agent_loop_tmp"
TEMP_WORKFLOW_ID_FILE = AGENT_LOOP_TEMP_DIR / "current_workflow_id.txt"

BASE_REFLECTION_THRESHOLD = int(os.environ.get("BASE_REFLECTION_THRESHOLD", "7"))
BASE_CONSOLIDATION_THRESHOLD = int(os.environ.get("BASE_CONSOLIDATION_THRESHOLD", "12"))
MIN_REFLECTION_THRESHOLD = 3
MAX_REFLECTION_THRESHOLD = 15
MIN_CONSOLIDATION_THRESHOLD = 5
MAX_CONSOLIDATION_THRESHOLD = 25
THRESHOLD_ADAPTATION_DAMPENING = float(os.environ.get("THRESHOLD_DAMPENING", "0.75"))
MOMENTUM_THRESHOLD_BIAS_FACTOR = 1.2

OPTIMIZATION_LOOP_INTERVAL = int(os.environ.get("OPTIMIZATION_INTERVAL", "8"))
MEMORY_PROMOTION_LOOP_INTERVAL = int(os.environ.get("PROMOTION_INTERVAL", "15"))
STATS_ADAPTATION_INTERVAL = int(os.environ.get("STATS_ADAPTATION_INTERVAL", "10"))
MAINTENANCE_INTERVAL = int(os.environ.get("MAINTENANCE_INTERVAL", "50"))

AUTO_LINKING_DELAY_SECS: Tuple[float, float] = (1.5, 3.0)
DEFAULT_PLAN_STEP = "Assess goal, gather context, formulate initial plan."
CONTEXT_RECENT_ACTIONS_FETCH_LIMIT = 10
CONTEXT_IMPORTANT_MEMORIES_FETCH_LIMIT = 7
CONTEXT_KEY_THOUGHTS_FETCH_LIMIT = 7
CONTEXT_PROCEDURAL_MEMORIES_FETCH_LIMIT = 3
CONTEXT_PROACTIVE_MEMORIES_FETCH_LIMIT = 5
CONTEXT_LINK_TRAVERSAL_FETCH_LIMIT = 5
CONTEXT_GOAL_DETAILS_FETCH_LIMIT = 3
CONTEXT_RECENT_ACTIONS_SHOW_LIMIT = 7
CONTEXT_IMPORTANT_MEMORIES_SHOW_LIMIT = 5
CONTEXT_KEY_THOUGHTS_SHOW_LIMIT = 5
CONTEXT_PROCEDURAL_MEMORIES_SHOW_LIMIT = 2
CONTEXT_PROACTIVE_MEMORIES_SHOW_LIMIT = 3
CONTEXT_WORKING_MEMORY_SHOW_LIMIT = 10
CONTEXT_LINK_TRAVERSAL_SHOW_LIMIT = 3
CONTEXT_GOAL_STACK_SHOW_LIMIT = 5
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
UMS_FUNC_CREATE_LINK = "create_memory_link"  # Corrected from create_memory_link
UMS_FUNC_SEARCH_SEMANTIC_MEMORIES = "search_semantic_memories"
UMS_FUNC_QUERY_MEMORIES = "query_memories"
UMS_FUNC_HYBRID_SEARCH = "hybrid_search_memories"  # Corrected from hybrid_search
UMS_FUNC_UPDATE_MEMORY = "update_memory"
UMS_FUNC_GET_LINKED_MEMORIES = "get_linked_memories"
UMS_FUNC_GET_WORKING_MEMORY = "get_working_memory"
UMS_FUNC_FOCUS_MEMORY = "focus_memory"
UMS_FUNC_OPTIMIZE_WM = "optimize_working_memory"  # Corrected from optimize_working_memory
UMS_FUNC_SAVE_COGNITIVE_STATE = "save_cognitive_state"
UMS_FUNC_LOAD_COGNITIVE_STATE = "load_cognitive_state"
UMS_FUNC_AUTO_FOCUS = "auto_update_focus"  # Corrected from auto_update_focus
UMS_FUNC_PROMOTE_MEM = "promote_memory_level"  # Corrected from promote_memory_level
UMS_FUNC_CONSOLIDATION = "consolidate_memories"  # Corrected from consolidate_memories
UMS_FUNC_REFLECTION = "generate_reflection"  # Corrected from generate_reflection
UMS_FUNC_SUMMARIZE_TEXT = "summarize_text"
UMS_FUNC_SUMMARIZE_CONTEXT_BLOCK = "summarize_context_block"
UMS_FUNC_DELETE_EXPIRED_MEMORIES = "delete_expired_memories"
UMS_FUNC_COMPUTE_STATS = "compute_memory_statistics"  # Corrected from compute_memory_statistics
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


BACKGROUND_TASK_TIMEOUT_SECONDS = 60.0
MAX_CONCURRENT_BG_TASKS = 10


# --- LOCAL CUSTOM EXCEPTIONS ---
class ToolError(Exception):
    pass


class ToolInputError(ToolError):
    pass


# ==========================================================================
# LOCAL UTILITY CLASSES & HELPERS
# ==========================================================================
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


def _truncate_context(context: Dict[str, Any], max_len: int = 25_000) -> str:
    try:
        full = json.dumps(context, indent=2, default=str, ensure_ascii=False)
    except TypeError:
        context_serializable = json.loads(json.dumps(context, default=str))
        full = json.dumps(context_serializable, indent=2, default=str, ensure_ascii=False)

    if len(full) <= max_len:
        return full

    log.debug(f"Context length {len(full)} exceeds max {max_len}. Applying structured truncation.")
    ctx_copy = copy.deepcopy(context)
    ctx_copy["_truncation_applied"] = "structure‑aware_agent_side"
    original_length = len(full)

    list_paths_to_truncate = [
        ("ums_context_package", "core_context", "recent_actions", CONTEXT_RECENT_ACTIONS_SHOW_LIMIT),
        ("ums_context_package", "core_context", "important_memories", CONTEXT_IMPORTANT_MEMORIES_SHOW_LIMIT),
        ("ums_context_package", "core_context", "key_thoughts", CONTEXT_KEY_THOUGHTS_SHOW_LIMIT),
        ("ums_context_package", "proactive_memories", "memories", CONTEXT_PROACTIVE_MEMORIES_SHOW_LIMIT),
        ("ums_context_package", "current_working_memory", "working_memories", CONTEXT_WORKING_MEMORY_SHOW_LIMIT),
        ("ums_context_package", "relevant_procedures", "procedures", CONTEXT_PROCEDURAL_MEMORIES_SHOW_LIMIT),
        ("agent_assembled_goal_context", None, "goal_stack_summary_from_agent_state", CONTEXT_GOAL_STACK_SHOW_LIMIT),
        (None, "current_plan_snapshot", None, 5),
    ]

    keys_to_remove_low_priority = [
        ("ums_context_package", "contextual_links"),
        ("ums_context_package", "relevant_procedures"),
        ("ums_context_package", "proactive_memories"),
        ("ums_context_package", "core_context", "key_thoughts"),
        ("ums_context_package", "core_context", "important_memories"),
        ("ums_context_package", "core_context", "recent_actions"),
        ("ums_context_package", "core_context"),
        ("ums_context_package", "current_working_memory"),
        ("agent_assembled_goal_context"),
        ("ums_context_package"),
    ]

    for path_parts in list_paths_to_truncate:
        container = ctx_copy
        key_to_truncate = path_parts[-2]
        limit_count = path_parts[-1]
        valid_path = True
        for part_idx in range(len(path_parts) - 2):
            part = path_parts[part_idx]
            if part is None:
                continue
            if part in container and isinstance(container[part], dict):
                container = container[part]
            else:
                valid_path = False
                break
        if not valid_path:
            continue

        if key_to_truncate in container and isinstance(container[key_to_truncate], list) and len(container[key_to_truncate]) > limit_count:
            original_count_val = len(container[key_to_truncate])
            note = {"truncated_note": f"{original_count_val - limit_count} items omitted from '{'/'.join(str(p) for p in path_parts if p)}'"}
            container[key_to_truncate] = container[key_to_truncate][:limit_count]
            if limit_count > 0 and isinstance(container[key_to_truncate], list):
                container[key_to_truncate].append(note)
            log.debug(f"Truncated list at '{'/'.join(str(p) for p in path_parts if p)}' to {limit_count} items.")
            serial_val = json.dumps(ctx_copy, indent=2, default=str, ensure_ascii=False)
            if len(serial_val) <= max_len:
                log.info(f"Context truncated (List reduction: {len(serial_val)} bytes).")
                return serial_val

    for key_info_tuple in keys_to_remove_low_priority:
        container = ctx_copy
        key_to_remove_final = key_info_tuple[-1]
        valid_path_pop = True
        for part_idx in range(len(key_info_tuple) - 1):
            part = key_info_tuple[part_idx]
            if part in container and isinstance(container[part], dict):
                container = container[part]
            else:
                valid_path_pop = False
                break
        if not valid_path_pop:
            continue

        if key_to_remove_final in container:
            container.pop(key_to_remove_final)
            log.debug(f"Removed low-priority key '{'/'.join(str(p) for p in key_info_tuple)}' for truncation.")
            serial_val = json.dumps(ctx_copy, indent=2, default=str, ensure_ascii=False)
            if len(serial_val) <= max_len:
                log.info(f"Context truncated (Key removal: {len(serial_val)} bytes).")
                return serial_val

    serial_val = json.dumps(ctx_copy, indent=2, default=str, ensure_ascii=False)
    log.warning(f"Structured truncation insufficient (Length still {len(serial_val)}). Applying final byte-slice.")
    clipped_json_str = _utf8_safe_slice(full, max_len - 50)
    try:
        last_brace = clipped_json_str.rfind("}")
        last_bracket = clipped_json_str.rfind("]")
        cutoff = max(last_brace, last_bracket)
        final_str = (
            clipped_json_str[: cutoff + 1] + "\n// ... (CONTEXT TRUNCATED BY BYTE LIMIT) ...\n}"
            if cutoff > 0
            else clipped_json_str + "... (CONTEXT TRUNCATED)"
        )
    except Exception:
        final_str = clipped_json_str + "... (CONTEXT TRUNCATED)"
    log.error(f"Context severely truncated from {original_length} to {len(final_str)} bytes using fallback.")
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


# =====================================================================
# Agent Master Loop
# =====================================================================
class AgentMasterLoop:
    # This set should contain the BASE FUNCTION NAMES of UMS tools considered meta/internal.
    # Example: "record_action_start", "get_workflow_details", etc.
    _INTERNAL_OR_META_TOOLS_BASE_NAMES: Set[str] = {
        UMS_FUNC_RECORD_ACTION_START,
        UMS_FUNC_RECORD_ACTION_COMPLETION,
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
        mcp_client_instance: MCPClient,
        default_llm_model_string: str,
        agent_state_file: str = AGENT_STATE_FILE,
    ):
        self.mcp_client = mcp_client_instance
        self.agent_llm_model = default_llm_model_string
        self.logger = log
        self.agent_state_file = Path(agent_state_file)

        if not hasattr(self.mcp_client, "anthropic") or not isinstance(self.mcp_client.anthropic, AsyncAnthropic):
            self.logger.critical("CRITICAL: MCPClient instance missing valid Anthropic client.")
            raise ValueError("MCPClient instance missing Anthropic client.")
        self.anthropic_client: AsyncAnthropic = self.mcp_client.anthropic

        self.consolidation_memory_level = MemoryLevel.EPISODIC.value
        self.consolidation_max_sources = 10
        self.auto_linking_threshold = 0.7
        self.auto_linking_max_links = 3
        self.reflection_type_sequence = ["summary", "progress", "gaps", "strengths", "plan"]
        self.state = AgentState()
        self._shutdown_event = asyncio.Event()
        self._bg_tasks_lock = asyncio.Lock()
        self._bg_task_semaphore = asyncio.Semaphore(MAX_CONCURRENT_BG_TASKS)
        self.tool_schemas: List[Dict[str, Any]] = []  # Populated during initialize by MCPClient
        _module_level_ums_func_constants = {
            k: v
            for k, v in globals().items()  # Get all globals in agent_master_loop.py's module scope
            if k.startswith("UMS_FUNC_") and isinstance(v, str)
        }
        self.all_ums_base_function_names: Set[str] = set(_module_level_ums_func_constants.values())
        self.logger.info(f"AgentMasterLoop initialized. LLM: {self.agent_llm_model}, UMS Server Name assumed: {UMS_SERVER_NAME}")
        self.logger.debug(
            f"Initialized {len(self.all_ums_base_function_names)} UMS base function names for internal reference. Sample: {list(self.all_ums_base_function_names)[:5]}"
        )

    async def shutdown(self) -> None:
        self.logger.info("Shutdown requested.")
        self._shutdown_event.set()
        await self._cleanup_background_tasks()
        # Determine if this shutdown implies the current workflow is truly done
        # For example, if goal_achieved_flag is true, or if consecutive_error_count is high.
        # This logic might need refinement based on how you want to handle "pauses" vs "full stops".
        is_workflow_terminal = (
            self.state.goal_achieved_flag
            or (self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS)
            or (self.state.current_loop >= getattr(self, "_current_max_loops_for_run", float("inf")))  # If tracking max loops for current run
        )
        # If a workflow was active and is now considered terminal, clear its state.
        # If workflow_id is already None, this won't do much.
        if self.state.workflow_id and is_workflow_terminal:
            self.logger.info(
                f"AML Shutdown: Workflow '{_fmt_id(self.state.workflow_id)}' considered terminal. Clearing active workflow state before saving."
            )
            self.state.workflow_id = None
            self.state.context_id = None  # Usually tied to workflow_id
            self.state.workflow_stack = []
            self.state.goal_stack = []
            self.state.current_goal_id = None
            self.state.current_thought_chain_id = None
            # Optionally reset plan, or leave it if you might want to inspect it post-run
            self.state.current_plan = [PlanStep(description=DEFAULT_PLAN_STEP)]
            self.state.last_action_summary = "Agent shut down; workflow state cleared."
            self.state.needs_replan = False
            self.state.last_error_details = None
            # Keep consecutive_error_count as is for logging, or reset it.
            # Keep loop counters as they are more general.
            # Keep tool_usage_stats.
        elif self.state.workflow_id:
            self.logger.info(
                f"AML Shutdown: Workflow '{_fmt_id(self.state.workflow_id)}' was active but not considered terminal. Persisting its state."
            )
        else:
            self.logger.info("AML Shutdown: No active workflow to clear state for.")
        await self._save_agent_state()
        self.logger.info("Agent loop shutdown complete.")

    def _get_ums_tool_mcp_name(self, base_function_name: str) -> str:
        """Constructs the full original MCP tool name for a UMS base function."""
        return f"{UMS_SERVER_NAME}:{base_function_name}"

    def _construct_agent_prompt(self, current_task_goal_desc: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Use self.state.current_loop directly for the current turn number being processed.
        current_turn_for_log_prompt = self.state.current_loop
        self.logger.info(f"AML CONSTRUCT_PROMPT: Building prompt for turn {current_turn_for_log_prompt}")
        self.logger.debug(f"AML CONSTRUCT_PROMPT: self.state.workflow_id = {_fmt_id(self.state.workflow_id)}")
        self.logger.debug(f"AML CONSTRUCT_PROMPT: self.state.current_goal_id = {_fmt_id(self.state.current_goal_id)}")
        self.logger.debug(
            f"AML CONSTRUCT_PROMPT: self.state.goal_stack (summary) = {[{'id': _fmt_id(g.get('goal_id')), 'desc': (g.get('description') or '')[:30] + '...', 'status': g.get('status')} for g in self.state.goal_stack if isinstance(g, dict)]}"
        )
        self.logger.debug(f"AML CONSTRUCT_PROMPT: current_task_goal_desc (param for this prompt) = {current_task_goal_desc[:100]}...")

        agent_status_message = context.get("status_message_from_agent", "Status unknown.")
        self.logger.debug(f"AML CONSTRUCT_PROMPT: context['status_message_from_agent'] = {agent_status_message}")

        context_goal_details = context.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
        if context_goal_details and isinstance(context_goal_details, dict):
            self.logger.debug(
                f"AML CONSTRUCT_PROMPT: context['agent_assembled_goal_context']['current_goal_details_from_ums'] = {{'id': '{_fmt_id(context_goal_details.get('goal_id'))}', 'desc': '{(context_goal_details.get('description') or '')[:50]}...'}}"
            )
        else:
            self.logger.debug(
                f"AML CONSTRUCT_PROMPT: context['agent_assembled_goal_context']['current_goal_details_from_ums'] = {context_goal_details}"
            )

        system_blocks: List[str] = [
            f"You are '{AGENT_NAME}', an AI agent orchestrator using a Unified Memory System (UMS) provided by the '{UMS_SERVER_NAME}' server.",
            "",
        ]

        # Determine the LLM-visible name for agent:update_plan
        llm_seen_agent_update_plan_name_for_instr = AGENT_TOOL_UPDATE_PLAN
        for schema_item_instr in self.tool_schemas:  # self.tool_schemas should be up-to-date
            schema_sanitized_name_instr = None
            if isinstance(schema_item_instr, dict):
                if schema_item_instr.get("type") == "function" and isinstance(schema_item_instr.get("function"), dict):
                    schema_sanitized_name_instr = schema_item_instr.get("function", {}).get("name")
                elif "name" in schema_item_instr:
                    schema_sanitized_name_instr = schema_item_instr.get("name")

            if schema_sanitized_name_instr:
                original_mcp_name_for_this_schema_instr = self.mcp_client.server_manager.sanitized_to_original.get(schema_sanitized_name_instr)
                if original_mcp_name_for_this_schema_instr == AGENT_TOOL_UPDATE_PLAN:
                    llm_seen_agent_update_plan_name_for_instr = schema_sanitized_name_instr
                    break
        if llm_seen_agent_update_plan_name_for_instr == AGENT_TOOL_UPDATE_PLAN:  # Fallback if not found in schemas (should not happen)
            llm_seen_agent_update_plan_name_for_instr = re.sub(r"[^a-zA-Z0-9_-]", "_", AGENT_TOOL_UPDATE_PLAN)[:64] or "agent_update_plan_fallback"

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
                system_blocks.append(
                    f"Current State: UMS Workflow '{_fmt_id(self.state.workflow_id)}' is ACTIVE, but NO specific UMS operational goal is currently set in agent's focus. (Agent Status: {agent_status_message})"
                )
                system_blocks.append(f"The Overall UMS Workflow Goal is: {ums_workflow_goal_from_context}")
                system_blocks.append(
                    f"**Action Required: Your next step should be to establish the primary UMS operational goal for this workflow.**"
                )
                system_blocks.append(
                    f"   - If the Overall UMS Workflow Goal ('{ums_workflow_goal_from_context[:50]}...') is suitable as the first operational UMS goal, use the UMS tool with base function '{UMS_FUNC_CREATE_GOAL}' to create it. Set `parent_goal_id` to `null` or omit it. Use the Overall UMS Workflow Goal as the description for this new UMS goal."
                )
                system_blocks.append(
                    f"   - Then, update your plan using the tool named `{llm_seen_agent_update_plan_name_for_instr}` to reflect steps towards this new UMS goal."
                )

        system_blocks.append("")
        system_blocks.append(
            f"Available Tools (Use ONLY these for UMS/Agent actions; format arguments per schema. Refer to tools by 'Name LLM Sees'):"
        )

        if not self.tool_schemas:
            system_blocks.append("- CRITICAL WARNING: No tools loaded into agent's schema list. Cannot function effectively.")
        else:
            # These are base function names. The tool list will show the full original MCP name for context.
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
                UMS_FUNC_CREATE_WORKFLOW,
                UMS_FUNC_GET_RICH_CONTEXT_PACKAGE,
                UMS_FUNC_SUMMARIZE_TEXT,
            }
            essential_agent_tool_mcp_names = {AGENT_TOOL_UPDATE_PLAN}

            for schema in self.tool_schemas:  # self.tool_schemas are the LLM-formatted ones
                llm_seen_name = None
                if isinstance(schema, dict):
                    if schema.get("type") == "function" and isinstance(schema.get("function"), dict):
                        llm_seen_name = schema.get("function", {}).get("name")
                    elif "name" in schema:
                        llm_seen_name = schema.get("name")

                if not llm_seen_name:
                    self.logger.warning(f"AML CONSTRUCT_PROMPT: Skipping tool schema with no discernible name: {str(schema)[:100]}")
                    continue

                original_mcp_name = self.mcp_client.server_manager.sanitized_to_original.get(llm_seen_name, llm_seen_name)
                base_function_name = self._get_base_function_name(original_mcp_name)

                is_essential = False
                if original_mcp_name in essential_agent_tool_mcp_names:
                    is_essential = True
                elif base_function_name in essential_tool_base_names:
                    is_essential = True

                prefix = "**" if is_essential else ""
                desc = schema.get("description", "No description.")
                # Get the input schema based on provider format (Anthropic vs OpenAI-like)
                input_schema_obj = schema.get("input_schema")  # Anthropic style
                if not input_schema_obj and schema.get("type") == "function":  # OpenAI style
                    input_schema_obj = schema.get("function", {}).get("parameters")

                # The `idempotency_key` should now appear here if UMS schemas are updated
                # and MCPClient correctly includes it in the formatted schemas.
                input_schema_str = json.dumps(input_schema_obj or {"type": "object", "properties": {}}, ensure_ascii=False)

                system_blocks.append(
                    f"\n- {prefix}Name LLM Sees: `{llm_seen_name}`{prefix}\n  (Base Function: `{base_function_name}`, Original MCP: `{original_mcp_name}`)\n  Desc: {desc}\n  Schema: {input_schema_str}"
                )
        system_blocks.append("")

        system_blocks.extend(
            [
                "Your Process at each step:",
                "1.  Context Analysis: Deeply analyze 'Current Context'. Note workflow status, errors (`last_error_details`), **goal stack (`agent_assembled_goal_context` -> `goal_stack_summary_from_agent_state`) and the `current_goal_details_from_ums`**, UMS package (`ums_context_package`), `current_plan`, `current_thought_chain_id`, and `meta_feedback`. Pay attention to `retrieved_at` timestamps for freshness.",
                "2.  Error Handling: If `last_error_details` exists, **FIRST** reason about the error `type` and `message`. Propose a recovery strategy. Refer to 'Recovery Strategies'.",
                "3.  Reasoning & Planning:",
                f"    a. State step-by-step reasoning towards the Current Operational UMS Goal (or the Initial Overall Task / Overall UMS Workflow Goal if no specific UMS operational goal is active). Record key thoughts using the UMS tool with base function '{UMS_FUNC_RECORD_THOUGHT}'.",
                "    b. Evaluate `current_plan`. Is it aligned? Valid? Addresses errors? Dependencies met?",
                f"    c. **Goal Management:** If Current Operational UMS Goal is too complex, use the UMS tool with base function '{UMS_FUNC_CREATE_GOAL}' (providing `parent_goal_id` as current UMS goal ID). When a UMS goal is met/fails, use UMS tool with base function '{UMS_FUNC_UPDATE_GOAL_STATUS}' with the UMS `goal_id` and status.",
                f"    d. Action Dependencies: For plan steps, use `depends_on` with step IDs. Then use UMS tool with base function '{UMS_FUNC_ADD_ACTION_DEPENDENCY}' (with UMS action IDs) if inter-action dependencies are needed.",
                f"    e. Artifacts: Plan to use UMS tool with base function '{UMS_FUNC_RECORD_ARTIFACT}' for creations (use `is_output=True` for final deliverables). Use UMS tools like base function '{UMS_FUNC_GET_ARTIFACTS}' or '{UMS_FUNC_GET_ARTIFACT_BY_ID}' for existing.",
                f"    f. Memory: Use UMS tool with base function '{UMS_FUNC_STORE_MEMORY}' for new facts/insights. Use UMS tool with base function '{UMS_FUNC_UPDATE_MEMORY}' for corrections.",
                f"    g. Thought Chains: Use UMS tool with base function '{UMS_FUNC_CREATE_THOUGHT_CHAIN}' for distinct sub-problems.",
                f"    h. Linking: Use UMS tool with base function '{UMS_FUNC_CREATE_LINK}' for relationships between UMS memories.",
                f"    i. Search: Prefer UMS tool with base function '{UMS_FUNC_HYBRID_SEARCH}'. Use UMS tool with base function '{UMS_FUNC_SEARCH_SEMANTIC_MEMORIES}' for pure conceptual similarity.",
                f"    j. Plan Update Tool: Use the tool named `{llm_seen_agent_update_plan_name_for_instr}` ONLY for significant changes, error recovery, or fixing validation issues. Do NOT use for simple step completion.",
                "4.  Action Decision:",
                f"    *   If NO ACTIVE UMS WORKFLOW: Your ONLY action MUST be to call the UMS tool whose base function is '{UMS_FUNC_CREATE_WORKFLOW}'. Use 'Initial Overall Task Description' from above as the 'goal' for this tool.",
                f"    *   If UMS Workflow IS ACTIVE BUT NO specific UMS operational goal is set: Your ONLY action MUST be to call the UMS tool with base function '{UMS_FUNC_CREATE_GOAL}' to establish the root UMS goal for the current workflow. Use the 'Overall UMS Workflow Goal' as its description.",
                "    *   If a workflow AND a specific UMS operational goal ARE active: Choose ONE action based on the *first step in Current Plan*:",
                f"        - Call a UMS Tool (e.g., one with base function '{UMS_FUNC_STORE_MEMORY}', '{UMS_FUNC_RECORD_ARTIFACT}'). Provide args per schema. The agent (not you, the LLM) will manage idempotency keys if needed for retries.",
                f"        - Record Thought: Use UMS tool with base function '{UMS_FUNC_RECORD_THOUGHT}'.",
                f"        - Update Plan Tool: Call `{llm_seen_agent_update_plan_name_for_instr}` with the **complete, repaired** plan if replanning is necessary.",
                f"        - Signal Current UMS Goal Completion: If the Current Operational UMS Goal is MET, first use the UMS tool with base function '{UMS_FUNC_UPDATE_GOAL_STATUS}' (with the UMS `goal_id` of the current operational goal and status='completed').",
                f"        - Signal Overall UMS Workflow Completion (ONLY when the entire multi-step workflow is done):",
                f"            - If the Overall UMS Workflow Goal is MET AND it resulted in a primary output artifact (e.g., a report, a file): ",
                f"              1. Ensure you have ALREADY called the UMS tool with base function '{UMS_FUNC_RECORD_ARTIFACT}' with `is_output=True` for that final artifact.",
                f"              2. Then, respond ONLY with the exact text: `Goal Achieved. Final output artifact ID: [THE_ARTIFACT_ID_RETURNED_BY_RECORD_ARTIFACT]` (replace bracketed part with the actual ID).",
                f"            - If the Overall UMS Workflow Goal is MET but there's NO single primary output artifact, respond ONLY with the text: `Goal Achieved: [Your concise summary of overall goal completion, max 100 words]`.",
                "5.  Output Format: Respond ONLY with the valid JSON for the chosen tool call OR one of the 'Goal Achieved...' text formats described above.",
            ]
        )
        system_blocks.extend(
            [
                "\nKey Considerations:",
                "*   Goal Focus: Always work towards the Current Operational UMS Goal. Use UMS goal tools by their base function names.",
                "*   Mental Momentum: Prioritize current plan steps if progress is steady and no errors/replans needed.",
                "*   Dependencies & Cycles: Ensure `depends_on` actions (in plan or UMS) are complete. Avoid cycles.",
                "*   UMS Context: Leverage the `ums_context_package` (core, working, proactive, procedural memories, links).",
                "*   Errors: Prioritize error analysis based on `last_error_details.type` and `last_action_summary`.",
                "*   User Guidance: Pay close attention to thoughts of type 'user_guidance' or memories of type 'user_input'. These are direct inputs from the operator and will likely require plan adjustments.",
                f"*   Final Output: If your task involves creating a deliverable (report, file, etc.), ensure it's saved as a UMS artifact using the UMS tool with base function '{UMS_FUNC_RECORD_ARTIFACT}' (passing `is_output=True`) *before* signaling overall workflow completion with its ID.",
                "*   Idempotency Keys: For UMS creation tools (like create_workflow, record_action_start, store_memory, create_goal, record_artifact, record_thought, create_thought_chain), you may see an optional `idempotency_key` parameter in their schemas. You (the LLM) generally do NOT need to provide this key. The agent system will manage idempotency for retries internally. Only provide it if explicitly instructed to reuse a specific key from a previous failed attempt.",
            ]
        )
        system_blocks.extend(
            [  # Recovery Strategies (existing logic, ensure it aligns with AML's error types)
                "\nRecovery Strategies based on `last_error_details.type`:",
                f"*   `InvalidInputError`: Review tool schema, args, context. Correct args and retry OR choose different tool/step.",
                f"*   `DependencyNotMetError`: Use UMS tool with base function '{UMS_FUNC_GET_ACTION_DETAILS}' on dependency IDs. Adjust plan order (`{llm_seen_agent_update_plan_name_for_instr}`) or wait.",
                f"*   `ServerUnavailable` / `NetworkError`: Tool's server might be down. Try different tool, wait, or adjust plan.",
                f"*   `APILimitError` / `RateLimitError`: External API busy. Plan to wait (record thought) before retry.",
                f"*   `ToolExecutionError` / `ToolInternalError` / `UMSError`: Tool failed. Analyze message. Try different args, alternative tool, or adjust plan.",  # Added UMSError
                f"*   `PlanUpdateError`: Proposed plan structure was invalid when agent tried to apply it. Re-examine plan and dependencies, try `{llm_seen_agent_update_plan_name_for_instr}` again with a corrected *complete* plan.",
                f"*   `PlanValidationError`: Proposed plan has logical issues (e.g., cycles, missing dependencies). Debug dependencies, propose corrected plan using `{llm_seen_agent_update_plan_name_for_instr}`.",
                f"*   `CancelledError`: Previous action cancelled. Re-evaluate current step.",
                f"*   `GoalManagementError` / `GoalSyncError`: Error managing UMS goals or mismatch between agent and UMS state. Review `agent_assembled_goal_context` and `last_error_details.recommendation`. Use UMS goal tools to correct or re-establish goals. May need to call UMS tool with base func `{UMS_FUNC_GET_GOAL_DETAILS}`.",
                f"*   `CognitiveStateError`: Error saving or loading agent's cognitive state. This is serious. Attempt to record key information as memories and then try to re-establish state or simplify the current task.",
                f"*   `InternalStateSetupError`: Critical internal error during agent/workflow setup. Analyze error. May require `{llm_seen_agent_update_plan_name_for_instr}` to fix plan or re-initiate a step.",
                f"*   `UnknownError` / `UnexpectedExecutionError` / `AgentError` / `MCPClientError` / `LLMError` / `LLMOutputError`: Analyze error message carefully. Simplify step, use different approach, or record_thought if stuck. If related to agent state, try to save essential info and restart a simpler sub-task.",
            ]
        )
        system_prompt_str = "\n".join(system_blocks)

        context_json_str = _truncate_context(context)

        user_prompt_blocks = [
            "Current Context:",
            "```json",
            context_json_str,
            "```",
            "",
            "Current Plan:",
            "```json",
            json.dumps([step.model_dump(exclude_none=True) for step in self.state.current_plan], indent=2, ensure_ascii=False),
            "```",
            "",
            f"Last Action Summary:\n{self.state.last_action_summary}\n",
        ]
        if self.state.last_error_details:
            user_prompt_blocks.extend(
                [
                    "**CRITICAL: Address Last Error Details (refer to Recovery Strategies in System Prompt)**:",
                    "```json",
                    json.dumps(self.state.last_error_details, indent=2, default=str, ensure_ascii=False),
                    "```",
                    "",
                ]
            )
        if self.state.last_meta_feedback:
            user_prompt_blocks.extend(
                ["**Meta-Cognitive Feedback (e.g., a suggested plan or insight from reflection/consolidation):**", self.state.last_meta_feedback, ""]
            )

        current_goal_desc_for_reminder = "Overall UMS Workflow Goal or Initial Task"
        if self.state.workflow_id:
            current_op_goal_details_reminder = context.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
            if (
                current_op_goal_details_reminder
                and isinstance(current_op_goal_details_reminder, dict)
                and current_op_goal_details_reminder.get("description")
            ):
                current_goal_desc_for_reminder = current_op_goal_details_reminder["description"]
            elif context.get("ums_context_package", {}).get("core_context", {}).get("workflow_goal"):
                current_goal_desc_for_reminder = context["ums_context_package"]["core_context"]["workflow_goal"]
        else:
            current_goal_desc_for_reminder = current_task_goal_desc

        user_prompt_blocks.append(f"Current Goal Reminder: {current_goal_desc_for_reminder}")
        user_prompt_blocks.append("")

        final_instruction_text = ""
        if not self.state.workflow_id:
            final_instruction_text = f"Instruction: NO ACTIVE UMS WORKFLOW. Your first action MUST be to call the UMS tool whose base function is '{UMS_FUNC_CREATE_WORKFLOW}'. Use the 'Initial Overall Task Description' from the system prompt as the 'goal' for this tool. Provide a suitable 'title'."
        elif self.state.workflow_id and not self.state.current_goal_id:
            final_instruction_text = f"Instruction: UMS WORKFLOW ACTIVE, BUT NO UMS OPERATIONAL GOAL SET. Your first action MUST be to call the UMS tool with base function '{UMS_FUNC_CREATE_GOAL}' to establish the root UMS goal for the current workflow. Use the 'Overall UMS Workflow Goal' as its description."
        elif self.state.needs_replan and self.state.last_meta_feedback:
            final_instruction_text = (
                f"Instruction: **REPLANNING REQUIRED.** Meta-cognitive feedback (see 'Meta-Cognitive Feedback' in context) is available. "
                f"Your primary action MUST be to use the tool named `{llm_seen_agent_update_plan_name_for_instr}` to set a new, detailed plan. "
                f"Carefully consider the meta-feedback when formulating the new plan. After updating the plan, the agent will proceed with the first step of the new plan in the *next* turn."
            )
        elif self.state.needs_replan:
            final_instruction_text = (
                f"Instruction: **REPLANNING REQUIRED.** An error occurred (`last_error_details`) or a significant state change necessitates a new plan. "
                f"Analyze `last_error_details` and other context. Your primary action MUST be to use the tool named `{llm_seen_agent_update_plan_name_for_instr}` to propose a new, complete, and valid plan to address the situation and achieve the Current Operational UMS Goal."
            )
        else:  # Normal operation
            final_instruction_text = (
                f"Instruction: Proceed with the first 'planned' step in 'Current Plan'. "
                f"If dependencies are met, call the assigned UMS tool or record a thought (UMS base func '{UMS_FUNC_RECORD_THOUGHT}'). "
                f"If the Current Operational UMS Goal is met, use UMS tool with base func '{UMS_FUNC_UPDATE_GOAL_STATUS}'. "
                f"If Overall UMS Workflow Goal met, respond according to the 'Signal Overall Workflow Completion' rules (provide artifact ID if applicable, otherwise a summary). "
                f"Only use `{llm_seen_agent_update_plan_name_for_instr}` if plan becomes invalid or needs strategic change."
            )

        self.logger.info(f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): Final instruction: {final_instruction_text}")
        user_prompt_blocks.append(final_instruction_text)
        user_prompt_str = "\n".join(user_prompt_blocks)

        constructed_prompt_messages = [{"role": "user", "content": system_prompt_str + "\n---\n" + user_prompt_str}]

        self.logger.debug(
            f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages type: {type(constructed_prompt_messages)}"
        )
        if isinstance(constructed_prompt_messages, list):
            self.logger.debug(
                f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages length: {len(constructed_prompt_messages)}"
            )
            if len(constructed_prompt_messages) > 0:
                self.logger.debug(
                    f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages[0] type: {type(constructed_prompt_messages[0])}"
                )
                if isinstance(constructed_prompt_messages[0], dict):
                    self.logger.debug(
                        f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages[0] keys: {list(constructed_prompt_messages[0].keys())}"
                    )
                    content_snippet = str(constructed_prompt_messages[0].get("content", ""))[:500] + "..."
                    self.logger.debug(
                        f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages[0]['content'] snippet: {content_snippet}"
                    )

        return constructed_prompt_messages

    def _background_task_done(self, task: asyncio.Task) -> None:
        asyncio.create_task(self._background_task_done_safe(task))

    async def _background_task_done_safe(self, task: asyncio.Task) -> None:
        was_present = False
        async with self._bg_tasks_lock:
            if task in self.state.background_tasks:
                self.state.background_tasks.discard(task)
                was_present = True
        if was_present:
            try:
                self._bg_task_semaphore.release()
                log.debug(f"Released semaphore. Count: {self._bg_task_semaphore._value}. Task: {task.get_name()}")
            except ValueError:
                log.warning(f"Semaphore release failed for task {task.get_name()}.")
            except Exception as sem_err:
                log.error(f"Error releasing semaphore for {task.get_name()}: {sem_err}")
        if task.cancelled():
            self.logger.debug(f"Background task {task.get_name()} cancelled.")
        exc = task.exception()
        if exc:
            self.logger.error(f"Background task {task.get_name()} failed: {type(exc).__name__}", exc_info=(type(exc), exc, exc.__traceback__))

    def _start_background_task(self, coro_fn, *args, **kwargs) -> asyncio.Task:
        # The `workflow_id` and `context_id` passed to the wrapper are for the wrapper's context,
        # not directly for coro_fn unless it's designed to accept them explicitly via **kwargs.
        # Instance methods like _run_auto_linking will use `self.state.workflow_id`.
        snapshot_wf_id = self.state.workflow_id
        snapshot_ctx_id = self.state.context_id

        async def _wrapper():
            log.debug(f"Waiting for semaphore... Task: {asyncio.current_task().get_name()}. Current count: {self._bg_task_semaphore._value}")
            await self._bg_task_semaphore.acquire()
            log.debug(f"Acquired semaphore. Task: {asyncio.current_task().get_name()}. New count: {self._bg_task_semaphore._value}")
            try:
                # If coro_fn needs specific context IDs different from self.state at execution time,
                # they should be passed explicitly in **kwargs when calling _start_background_task.
                # For example, if _run_auto_linking needed a specific `workflow_id` different from `self.state.workflow_id`
                # at the time of execution, you would call:
                # self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=..., workflow_id_for_coro=some_specific_id)
                # And _run_auto_linking would need to accept `workflow_id_for_coro`.
                # However, the current UMS tools accept `db_path` and often `workflow_id` directly.
                # The background tasks here are instance methods, so they have `self`.
                # The `workflow_id` and `context_id` passed to `_start_background_task`'s kwargs
                # are typically used by the UMS tool if it needs them.

                # We need to ensure the UMS tool gets the correct workflow_id for its operation.
                # The snapshot_wf_id is the one relevant when the task was *scheduled*.
                # The UMS tools like create_memory_link take workflow_id if it's not implicit.
                # Let's assume the coro_fn (like _run_auto_linking) will correctly pass the
                # necessary workflow_id to the UMS tools it calls, using the `snapshot_wf_id`
                # if appropriate for its logic.

                # The current structure has coro_fn taking `self` and then `*args`, `**kwargs`.
                # The `workflow_id` and `context_id` in `_start_background_task`'s kwargs are
                # implicitly for the UMS tools called *within* `coro_fn`.

                # Construct the kwargs for the UMS tool call inside the wrapper
                final_kwargs_for_coro = kwargs.copy()
                if "workflow_id" not in final_kwargs_for_coro and snapshot_wf_id:
                    final_kwargs_for_coro["workflow_id"] = snapshot_wf_id
                if "context_id" not in final_kwargs_for_coro and snapshot_ctx_id:
                    final_kwargs_for_coro["context_id"] = snapshot_ctx_id

                await asyncio.wait_for(coro_fn(self, *args, **final_kwargs_for_coro), timeout=BACKGROUND_TASK_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                self.logger.warning(f"Background task {asyncio.current_task().get_name()} timed out.")
            except Exception as e:
                self.logger.error(f"Exception in background task wrapper {asyncio.current_task().get_name()}: {e}", exc_info=True)

        task_name = f"bg_{coro_fn.__name__}_{_fmt_id(snapshot_wf_id)}_{random.randint(100, 999)}"
        task = asyncio.create_task(_wrapper(), name=task_name)
        asyncio.create_task(self._add_bg_task(task))  # Schedules _add_bg_task to run
        task.add_done_callback(self._background_task_done)
        self.logger.debug(f"Started background task: {task.get_name()} for WF {_fmt_id(snapshot_wf_id)}")
        return task

    async def _add_bg_task(self, task: asyncio.Task) -> None:
        async with self._bg_tasks_lock:
            self.state.background_tasks.add(task)

    async def _cleanup_background_tasks(self) -> None:
        tasks_to_cleanup: List[asyncio.Task] = []
        async with self._bg_tasks_lock:
            tasks_to_cleanup = list(self.state.background_tasks)
        if not tasks_to_cleanup:
            self.logger.debug("No background tasks to clean up.")
            return
        self.logger.info(f"Cleaning up {len(tasks_to_cleanup)} background tasks…")
        for t in tasks_to_cleanup:
            if not t.done():
                t.cancel()
        results = await asyncio.gather(*tasks_to_cleanup, return_exceptions=True)
        for i, res in enumerate(results):
            task = tasks_to_cleanup[i]
            task_name = task.get_name()
            if isinstance(res, asyncio.CancelledError):
                self.logger.debug(f"Task {task_name} cancelled during cleanup.")
            elif isinstance(res, Exception):
                self.logger.error(f"Task {task_name} error during cleanup: {res}", exc_info=isinstance(res, Exception))
            else:
                self.logger.debug(f"Task {task_name} finalized during cleanup.")
        async with self._bg_tasks_lock:
            self.state.background_tasks.clear()
        # Release all permits from semaphore if cleanup was forceful
        # This is a bit of a brute-force way to reset the semaphore if tasks were killed.
        if self._bg_task_semaphore._value < MAX_CONCURRENT_BG_TASKS:
            permits_to_release = MAX_CONCURRENT_BG_TASKS - self._bg_task_semaphore._value
            for _ in range(permits_to_release):
                try:
                    self._bg_task_semaphore.release()
                except ValueError:  # If trying to release too many
                    break
            self.logger.info(f"Reset semaphore count during cleanup. New count: {self._bg_task_semaphore._value}")

        self.logger.info("Background tasks cleanup finished.")

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

    async def _with_retries(
        self,
        coro_fun,
        *args,
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
        retry_backoff: float = 2.0,
        jitter: Tuple[float, float] = (0.1, 0.5),
        **kwargs,
    ):
        attempt = 0
        last_exception: Optional[BaseException] = None
        while True:
            try:
                return await coro_fun(*args, **kwargs)
            except retry_exceptions as e:
                attempt += 1
                last_exception = e
                if attempt >= max_retries:
                    self.logger.error(f"{coro_fun.__name__} failed after {max_retries} attempts: {type(e).__name__} - {e}")
                    raise  # Re-raise the last caught exception

                delay = (retry_backoff ** (attempt - 1)) + random.uniform(*jitter)
                self.logger.warning(
                    f"{coro_fun.__name__} failed ({type(e).__name__}: {str(e)[:100]}...); retry {attempt}/{max_retries} in {delay:.2f}s"
                )
                if self._shutdown_event.is_set():
                    self.logger.warning(f"Shutdown during retry for {coro_fun.__name__}.")
                    raise asyncio.CancelledError(f"Shutdown during retry for {coro_fun.__name__}") from last_exception
                await asyncio.sleep(delay)
            except asyncio.CancelledError:  # Explicitly handle CancelledError
                self.logger.info(f"{coro_fun.__name__} cancelled during retry wait or execution.")
                raise  # Re-raise to propagate cancellation

    async def _save_agent_state(self) -> None:
        self.logger.debug(f"Attempting to save agent state. WF ID: {_fmt_id(self.state.workflow_id)}")

        # Create a dictionary from the state, handling potential non-serializable fields manually
        state_dict_to_save = {}
        for fld in dataclasses.fields(AgentState):
            if fld.name == "background_tasks":  # Explicitly skip background_tasks
                continue

            value = getattr(self.state, fld.name)

            if fld.name == "current_plan":
                state_dict_to_save[fld.name] = [step.model_dump(exclude_none=True) for step in value] if value else []
            elif fld.name == "tool_usage_stats":
                # Convert defaultdict to regular dict for JSON
                state_dict_to_save[fld.name] = {k: dict(v) for k, v in value.items()} if value else {}
            elif fld.name == "goal_stack":
                # Ensure goal_stack is a list of dicts; UMS goal objects should be dicts
                state_dict_to_save[fld.name] = (
                    [
                        dict(g) if not isinstance(g, dict) else g  # Ensure it's a dict
                        for g in value
                        if isinstance(g, (dict, BaseModel))  # Allow Pydantic models too if they get in there
                    ]
                    if value
                    else []
                )
            else:
                # For other fields, directly assign. asdict handles most basic types.
                # If a field could contain complex non-serializable objects not handled above,
                # specific serialization logic would be needed here.
                state_dict_to_save[fld.name] = value

        state_dict_to_save["timestamp"] = datetime.now(timezone.utc).isoformat()

        try:
            self.agent_state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.agent_state_file.with_suffix(f".tmp_{os.getpid()}_{uuid.uuid4().hex[:4]}")  # More unique temp file

            json_string_to_write = json.dumps(state_dict_to_save, indent=2, ensure_ascii=False, default=str)  # Add default=str as a fallback

            async with aiofiles.open(tmp_file, "w", encoding="utf-8") as f:
                await f.write(json_string_to_write)
                await f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as e:
                    self.logger.warning(f"os.fsync failed during state save: {e}")

            os.replace(tmp_file, self.agent_state_file)
            self.logger.debug(f"State saved atomically -> {self.agent_state_file}")
        except TypeError as te:  # Catch specific TypeError during json.dumps
            self.logger.error(f"TypeError during JSON serialization of agent state: {te}", exc_info=True)
            self.logger.error(f"Problematic state_dict_to_save (keys): {list(state_dict_to_save.keys())}")
            # Try to log problematic field if possible (this is hard to do generically)
        except Exception as e:
            self.logger.error(f"Failed to save agent state: {e}", exc_info=True)
            if "tmp_file" in locals() and tmp_file.exists():
                try:
                    os.remove(tmp_file)
                except OSError as rm_err:
                    self.logger.error(f"Failed remove temporary state file {tmp_file}: {rm_err}")

    async def _load_agent_state(self) -> None:
        if not self.agent_state_file.exists():
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD, current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )
            self.logger.info("No prior state file. Starting fresh.")
            return
        try:
            async with aiofiles.open(self.agent_state_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())

            kwargs: Dict[str, Any] = {}
            processed_keys = set()

            for fld in dataclasses.fields(AgentState):
                if not fld.init:
                    continue  # Skip non-init fields like background_tasks
                name = fld.name
                processed_keys.add(name)
                if name in data:
                    value = data[name]
                    if name == "current_plan":
                        try:
                            kwargs["current_plan"] = (
                                [PlanStep(**d) for d in value] if isinstance(value, list) and value else [PlanStep(description=DEFAULT_PLAN_STEP)]
                            )
                        except (ValidationError, TypeError) as e:
                            self.logger.warning(f"Plan reload failed: {e}. Resetting to default plan.")
                            kwargs["current_plan"] = [PlanStep(description=DEFAULT_PLAN_STEP)]
                    elif name == "tool_usage_stats":
                        dd = _default_tool_stats()  # Use the factory for correct type
                        if isinstance(value, dict):
                            for k, v_dict in value.items():
                                if isinstance(v_dict, dict):  # Ensure inner value is a dict
                                    dd[k]["success"] = int(v_dict.get("success", 0))
                                    dd[k]["failure"] = int(v_dict.get("failure", 0))
                                    dd[k]["latency_ms_total"] = float(v_dict.get("latency_ms_total", 0.0))
                        kwargs["tool_usage_stats"] = dd
                    elif name == "goal_stack":
                        # Ensure it's a list of dicts, otherwise default to empty list
                        kwargs[name] = value if isinstance(value, list) and all(isinstance(item, dict) for item in value) else []
                    else:
                        # For other fields, assign directly if key exists
                        kwargs[name] = value
                else:  # Field missing from saved data, use dataclass default
                    if fld.default_factory is not dataclasses.MISSING:
                        kwargs[name] = fld.default_factory()
                    elif fld.default is not dataclasses.MISSING:
                        kwargs[name] = fld.default
                    # Explicit defaults for thresholds if not covered by above (though they should be)
                    elif name == "current_reflection_threshold":
                        kwargs[name] = BASE_REFLECTION_THRESHOLD
                    elif name == "current_consolidation_threshold":
                        kwargs[name] = BASE_CONSOLIDATION_THRESHOLD
                    elif name == "goal_stack":  # Ensure default for goal_stack if somehow missed
                        kwargs[name] = []
                    elif name == "current_goal_id":  # current_goal_id can be None
                        kwargs[name] = None

            extra_keys = set(data.keys()) - processed_keys - {"timestamp"}  # Exclude known non-field keys
            if extra_keys:
                self.logger.warning(f"Ignoring unknown keys in state file: {extra_keys}")

            temp_state = AgentState(**kwargs)  # Create state object

            # Post-load validation and correction
            if not isinstance(temp_state.current_reflection_threshold, int) or not (
                MIN_REFLECTION_THRESHOLD <= temp_state.current_reflection_threshold <= MAX_REFLECTION_THRESHOLD
            ):
                self.logger.warning(f"Invalid current_reflection_threshold ({temp_state.current_reflection_threshold}) loaded. Resetting to default.")
                temp_state.current_reflection_threshold = BASE_REFLECTION_THRESHOLD

            if not isinstance(temp_state.current_consolidation_threshold, int) or not (
                MIN_CONSOLIDATION_THRESHOLD <= temp_state.current_consolidation_threshold <= MAX_CONSOLIDATION_THRESHOLD
            ):
                self.logger.warning(
                    f"Invalid current_consolidation_threshold ({temp_state.current_consolidation_threshold}) loaded. Resetting to default."
                )
                temp_state.current_consolidation_threshold = BASE_CONSOLIDATION_THRESHOLD

            if not isinstance(temp_state.goal_stack, list):  # Should be handled by load, but double check
                self.logger.warning("goal_stack was not a list after load, resetting to empty list.")
                temp_state.goal_stack = []

            if temp_state.current_goal_id and not any(
                g.get("goal_id") == temp_state.current_goal_id for g in temp_state.goal_stack if isinstance(g, dict)
            ):
                self.logger.warning(
                    f"Loaded current_goal_id {_fmt_id(temp_state.current_goal_id)} not found in loaded goal_stack. Resetting current_goal_id."
                )
                temp_state.current_goal_id = (
                    temp_state.goal_stack[-1].get("goal_id") if temp_state.goal_stack and isinstance(temp_state.goal_stack[-1], dict) else None
                )

            self.state = temp_state  # Assign validated state
            self.logger.info(
                f"Loaded state from {self.agent_state_file}; loop {self.state.current_loop}, WF: {_fmt_id(self.state.workflow_id)}, Goal: {_fmt_id(self.state.current_goal_id)}"
            )

        except (json.JSONDecodeError, TypeError, FileNotFoundError) as e:
            self.logger.error(f"State load failed: {e}. Resetting to default state.", exc_info=True)
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD, current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )
        except Exception as e:  # Catch any other unexpected errors
            self.logger.critical(f"Unexpected error loading state: {e}. Resetting to default state.", exc_info=True)
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD, current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )

    def _get_ums_tool_mcp_name(self, base_function_name: str) -> str:
        """Constructs the full original MCP tool name for a UMS base function."""
        # Example: base_function_name = "create_workflow"
        # Returns "Ultimate MCP Server:create_workflow"
        return f"{UMS_SERVER_NAME}:{base_function_name}"

    def _get_base_function_name(self, tool_name_input: str) -> str:
        """
        Extracts the base function name from various tool name formats.
        e.g., "Ultimate MCP Server:create_workflow" -> "create_workflow"
              "create_workflow" -> "create_workflow"
              "agent:update_plan" -> "update_plan"
        """
        return tool_name_input.split(":")[-1]

    async def _check_workflow_exists(self, workflow_id: str) -> bool:
        """Robustly checks if a workflow_id exists in UMS."""
        get_details_mcp_tool_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_WORKFLOW_DETAILS)
        self.logger.debug(f"AML Check WF Exists: Checking workflow {_fmt_id(workflow_id)} using {get_details_mcp_tool_name}.")

        if not self._find_tool_server(get_details_mcp_tool_name):
            self.logger.error(f"AML Check WF Exists: Tool {get_details_mcp_tool_name} unavailable.")
            return False
        try:
            result_envelope = await self._execute_tool_call_internal(
                get_details_mcp_tool_name,
                {
                    "workflow_id": workflow_id,
                    "include_actions": False,
                    "include_artifacts": False,
                    "include_thoughts": False,
                    "include_memories": False,
                    "include_cognitive_states": False, # Don't need cognitive states just for WF existence check
                },
                record_action=False,
            )
            
            if isinstance(result_envelope, dict) and result_envelope.get("success"):
                ums_payload = result_envelope.get("data")
                if isinstance(ums_payload, dict) and ums_payload.get("success"):
                    if ums_payload.get("workflow_id") == workflow_id:
                        self.logger.debug(f"AML Check WF Exists: Workflow {_fmt_id(workflow_id)} confirmed to exist in UMS.")
                        return True
                    else:
                        self.logger.warning(f"AML Check WF Exists: UMS get_workflow_details success, but returned WF ID '{_fmt_id(ums_payload.get('workflow_id'))}' does not match query ID '{_fmt_id(workflow_id)}'. Assuming not found.")
                        return False
                elif isinstance(ums_payload, dict) and not ums_payload.get("success"):
                    self.logger.debug(f"AML Check WF Exists: Workflow {_fmt_id(workflow_id)} check - UMS tool reported failure: {ums_payload.get('error_message', ums_payload.get('error', 'Unknown UMS payload error'))}")
                    return False
                else: 
                    self.logger.warning(f"AML Check WF Exists: Envelope success for '{_fmt_id(workflow_id)}', but UMS payload inconclusive or malformed. Assuming exists. Payload preview: {str(ums_payload)[:150]}")
                    return True 
            else: 
                error_msg = result_envelope.get("error_message", "Unknown error from get_workflow_details envelope") if isinstance(result_envelope, dict) else "Malformed envelope from get_workflow_details"
                self.logger.debug(f"AML Check WF Exists: Workflow {_fmt_id(workflow_id)} not found or error during check (envelope failure): {error_msg}")
                return False
        except Exception as e: 
            self.logger.error(f"AML Check WF Exists: Unexpected error checking workflow '{_fmt_id(workflow_id)}' existence: {e}", exc_info=False)
            return False

    async def _validate_agent_workflow_and_context(self) -> bool:
        """
        Verifies that self.state.workflow_id exists in UMS.
        If self.state.context_id is set, verifies it exists, belongs to the workflow, and is the latest.
        If self.state.context_id is not set (or becomes invalid), attempts to load the latest 
        cognitive state for the workflow from UMS and sets self.state.context_id if successful.
        Then, re-verifies this established self.state.context_id.

        Returns:
            True if a valid workflow_id and a corresponding, loadable, and latest context_id 
            are established in self.state. False otherwise.
        """
        workflow_id_to_validate = self.state.workflow_id
        self.logger.debug(
            f"AML Validate WF/CTX: Starting validation. Agent State WF ID: '{_fmt_id(workflow_id_to_validate)}', "
            f"Agent State Context ID (CognitiveState.state_id): '{_fmt_id(self.state.context_id)}'."
        )

        if not workflow_id_to_validate:
            self.logger.info("AML Validate WF/CTX: No workflow_id in agent state. Validation cannot proceed meaningfully.")
            return False 

        # 1. Validate Workflow ID existence in UMS
        if not await self._check_workflow_exists(workflow_id_to_validate):
            self.logger.warning(
                f"AML Validate WF/CTX FAILED: Workflow ID '{_fmt_id(workflow_id_to_validate)}' from agent state not found in UMS."
            )
            return False
        self.logger.debug(f"AML Validate WF/CTX: Workflow ID '{_fmt_id(workflow_id_to_validate)}' confirmed in UMS.")

        # 2. Manage and Validate self.state.context_id (which is the UMS cognitive_state.state_id)
        context_id_to_check = self.state.context_id
        attempt_to_load_latest_context = False

        if context_id_to_check:
            # If a context_id exists in state, try to load it specifically to verify it.
            self.logger.debug(f"AML Validate WF/CTX: Agent state has context_id '{_fmt_id(context_id_to_check)}'. Verifying its validity...")
            load_specific_state_tool_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_LOAD_COGNITIVE_STATE)
            if not self._find_tool_server(load_specific_state_tool_mcp_name):
                self.logger.error(f"AML Validate WF/CTX FAILED: Tool '{UMS_FUNC_LOAD_COGNITIVE_STATE}' unavailable for specific context validation.")
                return False
            
            try:
                verify_specific_envelope = await self._execute_tool_call_internal(
                    load_specific_state_tool_mcp_name,
                    {"workflow_id": workflow_id_to_validate, "state_id": context_id_to_check},
                    record_action=False,
                )
                verify_ums_payload = verify_specific_envelope.get("data", {}) if isinstance(verify_specific_envelope, dict) else {}

                if (verify_specific_envelope.get("success") and verify_ums_payload.get("success") and
                    verify_ums_payload.get("state_id") == context_id_to_check and
                    verify_ums_payload.get("workflow_id") == workflow_id_to_validate and
                    verify_ums_payload.get("is_latest") is True):
                    self.logger.info(f"AML Validate WF/CTX: Agent's existing context_id '{_fmt_id(context_id_to_check)}' is VALID and LATEST for WF '{_fmt_id(workflow_id_to_validate)}'.")
                    return True # Existing context_id is good.
                else:
                    self.logger.warning(
                        f"AML Validate WF/CTX: Agent's existing context_id '{_fmt_id(context_id_to_check)}' is NOT valid/latest for WF '{_fmt_id(workflow_id_to_validate)}'. "
                        f"UMS Tool Error: {verify_specific_envelope.get('error_message', 'N/A')}. "
                        f"UMS Payload 'success': {verify_ums_payload.get('success', 'N/A')}, "
                        f"'state_id': {_fmt_id(verify_ums_payload.get('state_id'))}, "
                        f"'workflow_id': {_fmt_id(verify_ums_payload.get('workflow_id'))}, "
                        f"'is_latest': {verify_ums_payload.get('is_latest')}. "
                        f"Will attempt to load latest."
                    )
                    attempt_to_load_latest_context = True # Current one is bad, try to get a good one.
                    self.state.context_id = None # Clear stale context_id before attempting load
            except Exception as e_verify:
                self.logger.error(f"AML Validate WF/CTX: Exception verifying specific context_id '{_fmt_id(context_id_to_check)}': {e_verify}", exc_info=False)
                attempt_to_load_latest_context = True # Error verifying, try to load latest
                self.state.context_id = None # Clear stale context_id
        else: # No context_id in agent state initially
            self.logger.info(f"AML Validate WF/CTX: No context_id in agent state for WF '{_fmt_id(workflow_id_to_validate)}'. Will attempt to load latest.")
            attempt_to_load_latest_context = True

        if attempt_to_load_latest_context:
            self.logger.info(f"AML Validate WF/CTX: Attempting to load LATEST cognitive state from UMS for WF '{_fmt_id(workflow_id_to_validate)}'.")
            load_latest_state_tool_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_LOAD_COGNITIVE_STATE)
            if not self._find_tool_server(load_latest_state_tool_mcp_name):
                self.logger.error(f"AML Validate WF/CTX FAILED: Tool '{UMS_FUNC_LOAD_COGNITIVE_STATE}' unavailable. Cannot load/validate latest context.")
                return False

            try:
                load_latest_envelope = await self._execute_tool_call_internal(
                    load_latest_state_tool_mcp_name,
                    {"workflow_id": workflow_id_to_validate, "state_id": None}, # state_id=None means load latest
                    record_action=False,
                )
                load_latest_ums_payload = load_latest_envelope.get("data", {}) if isinstance(load_latest_envelope, dict) else {}

                if (load_latest_envelope.get("success") and load_latest_ums_payload.get("success")):
                    loaded_state_id_from_ums = load_latest_ums_payload.get("state_id")
                    loaded_workflow_id_from_ums_state = load_latest_ums_payload.get("workflow_id")
                    is_latest_flag_from_ums = load_latest_ums_payload.get("is_latest")

                    if (loaded_state_id_from_ums and isinstance(loaded_state_id_from_ums, str) and
                        loaded_workflow_id_from_ums_state == workflow_id_to_validate and
                        is_latest_flag_from_ums is True): # Must be the latest one
                        
                        self.state.context_id = loaded_state_id_from_ums
                        self.logger.info(
                            f"AML Validate WF/CTX SUCCESS: Successfully loaded and set LATEST context_id (CognitiveState.state_id) "
                            f"to '{_fmt_id(self.state.context_id)}' for WF '{_fmt_id(workflow_id_to_validate)}'."
                        )
                        return True
                    else:
                        self.logger.error(
                            f"AML Validate WF/CTX FAILED: UMS load_cognitive_state (latest) returned success, but payload was invalid "
                            f"(state_id: '{_fmt_id(loaded_state_id_from_ums)}', wf_id: '{_fmt_id(loaded_workflow_id_from_ums_state)}', is_latest: {is_latest_flag_from_ums}) "
                            f"for queried WF '{_fmt_id(workflow_id_to_validate)}'. UMS Payload: {str(load_latest_ums_payload)[:200]}"
                        )
                        self.state.context_id = None # Ensure it's cleared if load failed to produce valid one
                        return False
                else: 
                    error_msg_load_latest = load_latest_envelope.get("error_message", "Unknown error") if isinstance(load_latest_envelope, dict) else "Malformed envelope"
                    self.logger.error(
                        f"AML Validate WF/CTX FAILED: Could not load LATEST cognitive state for WF '{_fmt_id(workflow_id_to_validate)}'. "
                        f"UMS Tool Error: {error_msg_load_latest}. "
                        f"UMS Payload Preview (if any): {str(load_latest_ums_payload)[:200]}"
                    )
                    self.state.context_id = None # Ensure it's cleared
                    return False
            except Exception as e_load_latest_cog:
                self.logger.error(
                    f"AML Validate WF/CTX FAILED: Exception loading LATEST cognitive state for WF '{_fmt_id(workflow_id_to_validate)}': {e_load_latest_cog}",
                    exc_info=False
                )
                self.state.context_id = None # Ensure it's cleared
                return False
        
        # If we reached here, it means we had an initial context_id_to_check and it was already validated as good.
        # This path should ideally not be taken if the first check passed, but for safety:
        self.logger.debug(f"AML Validate WF/CTX: Reached end of validation. Context ID '{_fmt_id(self.state.context_id)}' should be valid.")
        return True # Should have returned True earlier if valid.


    async def initialize(self) -> bool:
        self.logger.info("🤖 AML: Initializing Agent Master Loop (v_initialize_no_temp_file)...")
        await self._load_agent_state()  # Loads self.state from AGENT_STATE_FILE

        loaded_wf_id = self.state.workflow_id
        loaded_ctx_id = self.state.context_id
        state_is_valid_and_ready = False

        if loaded_wf_id:
            self.logger.info(
                f"🤖 AML Initialize: Loaded state from '{self.agent_state_file.name}' has WF='{_fmt_id(loaded_wf_id)}', CTX='{_fmt_id(loaded_ctx_id)}'. "
                f"Attempting validation and synchronization with UMS..."
            )
            # _validate_agent_workflow_and_context will use self.state.workflow_id.
            # If self.state.context_id is None, it will attempt to load the latest for that workflow.
            # It returns True if a valid workflow_id AND a valid (latest) context_id are established.
            if await self._validate_agent_workflow_and_context():
                self.logger.info(
                    f"🤖 AML Initialize: Loaded/Synced state is VALID. "
                    f"Effective WF ID for session: '{_fmt_id(self.state.workflow_id)}', "
                    f"Effective Context ID (CognitiveState.state_id) for session: '{_fmt_id(self.state.context_id)}'."
                )
                state_is_valid_and_ready = True

                # Ensure workflow stack is consistent with the validated workflow_id.
                if not self.state.workflow_stack or self.state.workflow_stack[-1] != self.state.workflow_id:
                    self.state.workflow_stack = [self.state.workflow_id]
                    self.logger.info(f"🤖 AML Initialize: Workflow stack reset/set to: [{_fmt_id(self.state.workflow_id)}]")
                
                # Validate/sync goal stack for the current workflow
                # This ensures self.state.current_goal_id and self.state.goal_stack are consistent with UMS
                # for the now-confirmed-valid self.state.workflow_id.
                await self._validate_goal_stack_on_load()

                # Set default thought chain if none is current for this valid workflow.
                if not self.state.current_thought_chain_id:
                    self.logger.info(
                        f"🤖 AML Initialize: No current_thought_chain_id set for valid workflow '{_fmt_id(self.state.workflow_id)}'. "
                        f"Attempting to set default."
                    )
                    await self._set_default_thought_chain_id() # This tries to find the primary chain in UMS.

                # If, after all validation, there's a valid workflow but no current UMS operational goal,
                # the main loop's prompt construction will guide the LLM to establish one.
                if self.state.workflow_id and not self.state.current_goal_id:
                    self.logger.info(
                        f"🤖 AML Initialize: Valid workflow '{_fmt_id(self.state.workflow_id)}' has no current UMS operational goal "
                        f"(after validation). Agent will be prompted by main loop to establish one."
                    )
                    # Ensure plan reflects the need to establish a root UMS goal.
                    if not self.state.current_plan or self.state.current_plan[0].description == DEFAULT_PLAN_STEP:
                        self.state.current_plan = [
                            PlanStep(description=f"Establish root UMS goal for existing active workflow: {_fmt_id(self.state.workflow_id)}")
                        ]
                        self.state.needs_replan = False # This is the new plan for the agent to execute.
                        self.logger.info("🤖 AML Initialize: Plan updated to establish root UMS goal for existing active workflow.")
            else: # _validate_agent_workflow_and_context returned False
                self.logger.warning(
                    f"🤖 AML Initialize WARNING: Loaded state (WF='{_fmt_id(loaded_wf_id)}', CTX='{_fmt_id(loaded_ctx_id)}') "
                    f"FAILED UMS validation or synchronization. Resetting workflow-specific state."
                )
                state_is_valid_and_ready = False # Mark for reset
        
        else: # No workflow_id loaded from state file at all
            self.logger.info(
                f"🤖 AML Initialize: No workflow_id found in loaded state file '{self.agent_state_file.name}'. "
                f"Agent will start completely fresh."
            )
            state_is_valid_and_ready = False # Mark for reset

        if not state_is_valid_and_ready:
            self.logger.info("🤖 AML Initialize: Performing reset of workflow-specific agent state attributes.")
            # Preserve essential non-workflow state items
            preserved_tool_stats = self.state.tool_usage_stats
            pres_ref_thresh = self.state.current_reflection_threshold
            pres_con_thresh = self.state.current_consolidation_threshold
            
            # Reset only workflow-dependent parts of the state, or the whole state if it's simpler
            # For now, let's reset the whole AgentState object and re-apply preserved items
            self.state = AgentState() # Full reset to defaults as defined in AgentState dataclass
            
            # Re-apply preserved non-workflow-specific settings
            self.state.tool_usage_stats = preserved_tool_stats
            self.state.current_reflection_threshold = pres_ref_thresh
            self.state.current_consolidation_threshold = pres_con_thresh
            
            # Explicitly ensure key fields are at their "fresh start" values
            self.state.workflow_id = None
            self.state.context_id = None
            self.state.workflow_stack = []
            self.state.goal_stack = []
            self.state.current_goal_id = None
            self.state.current_thought_chain_id = None
            self.state.current_plan = [PlanStep(description=DEFAULT_PLAN_STEP)]
            self.state.last_action_summary = "Agent state reset: No valid prior workflow/context, or validation failed."
            self.state.needs_replan = False # Starting fresh, LLM will make first plan.
            self.state.last_error_details = None
            self.state.consecutive_error_count = 0
            self.state.goal_achieved_flag = False
            # Counters for periodic tasks are reset by new AgentState() init
            self.state.reflection_cycle_index = 0
            self.state.last_meta_feedback = None
            self.logger.info("🤖 AML Initialize: Agent state fully reset to defaults for a fresh start.")

        # Save the agent's state (whether it's a validated existing state or a freshly reset one)
        await self._save_agent_state()
        
        self.logger.info(
            f"🤖 AML Initialize: Agent state finalized. Effective state for session - "
            f"WF ID: '{_fmt_id(self.state.workflow_id)}', "
            f"Context ID (CognitiveState.state_id): '{_fmt_id(self.state.context_id)}', "
            f"UMS Goal ID: '{_fmt_id(self.state.current_goal_id)}', "
            f"Needs Replan: {self.state.needs_replan}"
        )

        # --- Tool Schema Setup ---
        self.logger.info("🤖 AML Initialize: Starting tool schema setup phase...")
        if not hasattr(self.mcp_client, "server_manager") or not self.mcp_client.server_manager:
            self.logger.critical("🤖 AML CRITICAL: MCPClient or its ServerManager is not initialized. Agent cannot set up tools.")
            return False
        agent_llm_provider_str = self.mcp_client.get_provider_from_model(self.agent_llm_model)
        if not agent_llm_provider_str:
            self.logger.critical(f"🤖 AML CRITICAL: Could not determine LLM provider for agent's model '{self.agent_llm_model}'.")
            return False

        all_llm_formatted_tools_from_mcpc: List[Dict[str, Any]] = self.mcp_client._format_tools_for_provider(agent_llm_provider_str) or []
        self.logger.info(
            f"🤖 AML: Received {len(all_llm_formatted_tools_from_mcpc)} LLM-formatted tools from MCPClient for provider '{agent_llm_provider_str}'."
        )

        self.tool_schemas = [] 
        agent_llm_final_used_sanitized_names: Set[str] = set()
        sanitized_to_original_map_shared_from_mcpc = self.mcp_client.server_manager.sanitized_to_original

        self.logger.info(f"🤖 AML (Tool Init): === STARTING AGENT TOOL SCHEMA PREPARATION (Provider: {agent_llm_provider_str}) ===")
        for idx, llm_tool_schema_from_mcpc in enumerate(all_llm_formatted_tools_from_mcpc):
            is_anthropic_format_for_agent = agent_llm_provider_str == Provider.ANTHROPIC.value
            sanitized_name_as_received_from_mcpc = ""
            if isinstance(llm_tool_schema_from_mcpc, dict):
                if is_anthropic_format_for_agent:
                    sanitized_name_as_received_from_mcpc = llm_tool_schema_from_mcpc.get("name", "")
                else: # OpenAI-like
                    sanitized_name_as_received_from_mcpc = (
                        llm_tool_schema_from_mcpc.get("function", {}).get("name")
                        if isinstance(llm_tool_schema_from_mcpc.get("function"), dict)
                        else ""
                    )
            if not sanitized_name_as_received_from_mcpc:
                self.logger.warning(f"🤖 AML (Tool Init - MCPC Tool {idx + 1}): Skipping schema with no discernible name: {str(llm_tool_schema_from_mcpc)[:150]}")
                continue
            
            original_mcp_name = sanitized_to_original_map_shared_from_mcpc.get(sanitized_name_as_received_from_mcpc)
            if not original_mcp_name:
                self.logger.error(
                    f"🤖 AML (Tool Init - MCPC Tool {idx + 1}): CRITICAL! No original MCP name in SHARED MAP for MCPC-sanitized name '{sanitized_name_as_received_from_mcpc}'. "
                    f"Desc Hint: {str(llm_tool_schema_from_mcpc.get('description', 'N/A'))[:70]}. Skipping."
                )
                continue

            final_sanitized_name_for_agent_llm = sanitized_name_as_received_from_mcpc
            counter = 1
            while final_sanitized_name_for_agent_llm in agent_llm_final_used_sanitized_names:
                suffix = f"_agent_v{counter}"
                base_name_for_suffix = sanitized_name_as_received_from_mcpc 
                if len(base_name_for_suffix) + len(suffix) > 64:
                    final_sanitized_name_for_agent_llm = base_name_for_suffix[: 64 - len(suffix)] + suffix
                else:
                    final_sanitized_name_for_agent_llm = base_name_for_suffix + suffix
                counter += 1
            
            agent_llm_final_used_sanitized_names.add(final_sanitized_name_for_agent_llm)
            updated_schema_for_agent_llm = copy.deepcopy(llm_tool_schema_from_mcpc)

            if final_sanitized_name_for_agent_llm != sanitized_name_as_received_from_mcpc:
                name_updated_in_schema = False
                if is_anthropic_format_for_agent:
                    if "name" in updated_schema_for_agent_llm:
                        updated_schema_for_agent_llm["name"] = final_sanitized_name_for_agent_llm
                        name_updated_in_schema = True
                else: 
                    if ("function" in updated_schema_for_agent_llm and 
                        isinstance(updated_schema_for_agent_llm.get("function"), dict) and 
                        "name" in updated_schema_for_agent_llm["function"]):
                        updated_schema_for_agent_llm["function"]["name"] = final_sanitized_name_for_agent_llm
                        name_updated_in_schema = True
                
                if not name_updated_in_schema:
                    self.logger.error(f"🤖 AML (Tool Init): Could not UPDATE SCHEMA with final unique name '{final_sanitized_name_for_agent_llm}' for '{original_mcp_name}'. Skipping.")
                    if final_sanitized_name_for_agent_llm in agent_llm_final_used_sanitized_names:
                        agent_llm_final_used_sanitized_names.remove(final_sanitized_name_for_agent_llm)
                    continue
                
                if (sanitized_name_as_received_from_mcpc in sanitized_to_original_map_shared_from_mcpc and
                    sanitized_to_original_map_shared_from_mcpc[sanitized_name_as_received_from_mcpc] == original_mcp_name):
                    del sanitized_to_original_map_shared_from_mcpc[sanitized_name_as_received_from_mcpc]
                sanitized_to_original_map_shared_from_mcpc[final_sanitized_name_for_agent_llm] = original_mcp_name
                self.logger.info(
                    f"🤖 AML (Tool Init): Re-sanitized & UPDATED SHARED MCPClient MAP: '{final_sanitized_name_for_agent_llm}' -> '{original_mcp_name}' (was '{sanitized_name_as_received_from_mcpc}')"
                )
            
            self.tool_schemas.append(updated_schema_for_agent_llm)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    f"🤖 AML (Tool Init - MCPC Tool {idx + 1}): ADDED. Original='{original_mcp_name}' (Agent LLM sees: '{final_sanitized_name_for_agent_llm}')"
                )

        plan_step_base_schema = PlanStep.model_json_schema(); plan_step_base_schema.pop("title", None)
        update_plan_input_schema = {"type": "object", "properties": {"plan": {"type": "array", "items": plan_step_base_schema, "description": "The new complete list of plan steps."}}, "required": ["plan"]}
        agent_update_plan_original_mcp_name = AGENT_TOOL_UPDATE_PLAN
        base_agent_tool_sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_update_plan_original_mcp_name)[:64] or f"internal_tool_{uuid.uuid4().hex[:8]}"
        final_agent_tool_sanitized_name_for_llm = base_agent_tool_sanitized_name; agent_tool_counter = 1
        while final_agent_tool_sanitized_name_for_llm in agent_llm_final_used_sanitized_names:
            suffix = f"_agent_v{agent_tool_counter}"; base_len_for_suffix = len(base_agent_tool_sanitized_name)
            if base_len_for_suffix + len(suffix) > 64: final_agent_tool_sanitized_name_for_llm = base_agent_tool_sanitized_name[:64 - len(suffix)] + suffix
            else: final_agent_tool_sanitized_name_for_llm = base_agent_tool_sanitized_name + suffix
            agent_tool_counter += 1
        agent_llm_final_used_sanitized_names.add(final_agent_tool_sanitized_name_for_llm)
        sanitized_to_original_map_shared_from_mcpc[final_agent_tool_sanitized_name_for_llm] = agent_update_plan_original_mcp_name
        self.logger.info(f"🤖 AML (Agent Tool Init): ENSURED MCPClient SHARED MAP for AGENT_TOOL_UPDATE_PLAN: '{final_agent_tool_sanitized_name_for_llm}' -> '{agent_update_plan_original_mcp_name}'")
        plan_tool_description = "Replace agent's current plan. Use for significant replanning, error recovery, or fixing validation issues. Submit the ENTIRE new plan."
        plan_tool_llm_schema_final = ({"name": final_agent_tool_sanitized_name_for_llm, "description": plan_tool_description, "input_schema": update_plan_input_schema}
                                      if agent_llm_provider_str == Provider.ANTHROPIC.value else
                                      {"type": "function", "function": {"name": final_agent_tool_sanitized_name_for_llm, "description": plan_tool_description, "parameters": update_plan_input_schema}})
        self.tool_schemas.append(plan_tool_llm_schema_final)
        self.logger.info(f"🤖 AML (Agent Tool Init): ADDED agent-internal tool to self.tool_schemas: '{agent_update_plan_original_mcp_name}' (LLM sees: '{final_agent_tool_sanitized_name_for_llm}')")
        
        self.logger.info(f"🤖 AML: Final {len(self.tool_schemas)} tool schemas prepared for agent's LLM.")
        self.logger.info(f"🤖 AML (Tool Init): === FINISHED AGENT TOOL SCHEMA PREPARATION ===")
        
        essential_mcp_tool_names_list_for_check = [
            self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_WORKFLOW), 
            self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL),
            self._get_ums_tool_mcp_name(UMS_FUNC_GET_RICH_CONTEXT_PACKAGE),
            AGENT_TOOL_UPDATE_PLAN,
        ]
        available_original_names_for_agent_llm = set(sanitized_to_original_map_shared_from_mcpc.values()) 
        missing_essential_tools = [orig_name for orig_name in essential_mcp_tool_names_list_for_check if orig_name not in available_original_names_for_agent_llm]
        
        if missing_essential_tools:
            self.logger.error(
                f"🤖 AML CRITICAL ERROR (Post-Tool-Setup): Missing essential tools (by original MCP name) from agent's final effective toolset: {missing_essential_tools}. "
                f"Agent functionality WILL BE severely impaired. Check MCPClient's tool formatting and ServerManager's sanitized_to_original map."
            )
            # Depending on severity, you might want to return False here or raise an error
            # return False 
        else:
            self.logger.info("🤖 AML: All specified essential tools (by original MCP name) confirmed available and mapped for agent's LLM.")

        self.logger.info(
            f"🤖 AML: Agent Master Loop initialization complete. Effective state for session - "
            f"WF ID: '{_fmt_id(self.state.workflow_id)}', "
            f"Context ID: '{_fmt_id(self.state.context_id)}', "
            f"UMS Goal ID: '{_fmt_id(self.state.current_goal_id)}'."
        )
        return True


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


    def _find_tool_server(self, tool_identifier_mcp_style: str) -> Optional[str]:
        """
        Finds an active server that provides the specified tool.
        The tool_identifier_mcp_style is expected to be an original MCP name
        (e.g., "ExpectedServerName:FunctionName") or AGENT_TOOL_UPDATE_PLAN.

        This method first extracts the target base function name. It then searches
        all tools from all *active* servers. If multiple active servers provide
        the same base function name, it uses the server part of
        tool_identifier_mcp_style as a hint to disambiguate. For known UMS functions,
        it explicitly prefers UMS_SERVER_NAME if that server is among the active candidates.
        """
        self.logger.debug(f"AML _find_tool_server: Locating server for tool identifier: '{tool_identifier_mcp_style}'")

        if not self.mcp_client or not self.mcp_client.server_manager:
            self.logger.warning("AML _find_tool_server: MCPClient or ServerManager is not available. Cannot find tool server.")
            return None

        sm = self.mcp_client.server_manager

        if tool_identifier_mcp_style == AGENT_TOOL_UPDATE_PLAN:
            self.logger.debug(f"AML _find_tool_server: Matched AGENT_TOOL_UPDATE_PLAN. Returning 'AGENT_INTERNAL'.")
            return "AGENT_INTERNAL"

        # Extract the expected server name part (hint) and the target base function name
        # from the provided MCP-style tool identifier.
        parts = tool_identifier_mcp_style.split(":", 1)
        expected_server_name_hint = parts[0] if len(parts) > 1 else None # e.g., "Ultimate MCP Server" or None
        target_base_function_name = parts[-1] # e.g., "create_goal" (self._get_base_function_name handles this)

        self.logger.debug(f"AML _find_tool_server: Parsed - Target Base Function: '{target_base_function_name}', Expected Server Hint: '{expected_server_name_hint}'")

        candidate_active_servers_providing_function: List[str] = []
        
        # Iterate through all tools known to the ServerManager.
        # sm.tools is keyed by the original MCP name used at registration time,
        # which is `ActualConnectedServerNameAsKnownToSM:BaseFunctionName`.
        for original_mcp_name_key_in_sm, mcp_tool_obj in sm.tools.items():
            # mcp_tool_obj.server_name is the ActualConnectedServerNameAsKnownToSM
            # mcp_tool_obj.name is original_mcp_name_key_in_sm
            
            actual_tool_server_name_in_sm = mcp_tool_obj.server_name 
            actual_tool_base_function_in_sm = self._get_base_function_name(mcp_tool_obj.name)

            if actual_tool_base_function_in_sm == target_base_function_name:
                # Found a tool with the matching base function name.
                # Now check if its server (actual_tool_server_name_in_sm) is currently active.
                if actual_tool_server_name_in_sm in sm.active_sessions:
                    candidate_active_servers_providing_function.append(actual_tool_server_name_in_sm)
                    self.logger.debug(
                        f"AML _find_tool_server: Active server '{actual_tool_server_name_in_sm}' provides base function '{target_base_function_name}' "
                        f"(via tool '{original_mcp_name_key_in_sm}'). Added to candidates."
                    )
                else:
                    self.logger.debug(
                        f"AML _find_tool_server: Server '{actual_tool_server_name_in_sm}' provides base function '{target_base_function_name}', "
                        f"but is NOT ACTIVE. Skipping."
                    )
        
        # Remove duplicates from candidate_active_servers_providing_function
        unique_candidate_servers = sorted(set(candidate_active_servers_providing_function))
        self.logger.debug(
            f"AML _find_tool_server: Unique active candidate servers for base_func '{target_base_function_name}': {unique_candidate_servers}"
        )

        if not unique_candidate_servers:
            self.logger.warning(f"AML _find_tool_server: No ACTIVE server found providing base function '{target_base_function_name}'. Input identifier was '{tool_identifier_mcp_style}'.")
            return None
        
        if len(unique_candidate_servers) == 1:
            # Only one active server provides this function, unambiguous.
            found_server = unique_candidate_servers[0]
            self.logger.info(
                f"AML _find_tool_server: Unambiguously found base function '{target_base_function_name}' "
                f"on active server '{found_server}'."
            )
            return found_server
        
        # Ambiguity: Multiple active servers provide this base function.
        self.logger.debug(
            f"AML _find_tool_server: Ambiguity - Multiple active servers for base_func '{target_base_function_name}': {unique_candidate_servers}. "
            f"Attempting to resolve using hint: '{expected_server_name_hint}'."
        )

        # Resolution Strategy:
        # 1. If the expected_server_name_hint (e.g., "Ultimate MCP Server" from the input identifier)
        #    is among the active candidates, prefer that.
        if expected_server_name_hint and expected_server_name_hint in unique_candidate_servers:
            self.logger.info(
                f"AML _find_tool_server: Resolving ambiguity for '{target_base_function_name}'. "
                f"Prioritizing server matching hint '{expected_server_name_hint}' as it is active and provides the function."
            )
            return expected_server_name_hint

        # 2. If it's a known UMS base function (from self.all_ums_base_function_names),
        #    and the UMS_SERVER_NAME constant (e.g. "Ultimate MCP Server") is among the active candidates,
        #    prefer UMS_SERVER_NAME. This covers cases where the hint might have been slightly different
        #    but the function is clearly a core UMS one.
        if target_base_function_name in self.all_ums_base_function_names:
            # Check if any of the candidate server names *exactly match* UMS_SERVER_NAME constant
            if UMS_SERVER_NAME in unique_candidate_servers:
                self.logger.info(
                    f"AML _find_tool_server: Resolving ambiguity for UMS base function '{target_base_function_name}'. "
                    f"Prioritizing the server named '{UMS_SERVER_NAME}' (matching constant) as it is an active candidate."
                )
                return UMS_SERVER_NAME
            else:
                # If UMS_SERVER_NAME constant is not an exact match, but there's only ONE candidate left,
                # and it's a UMS function, it's likely the UMS server (even if its connected name differs slightly).
                # This part is tricky. The current logic of unique_candidate_servers already handles if there's only one.
                # The above check is for when UMS_SERVER_NAME is explicitly one of the unique_candidate_servers.
                self.logger.debug(
                    f"AML _find_tool_server: UMS function '{target_base_function_name}' sought. "
                    f"Active candidates: {unique_candidate_servers}. UMS_SERVER_NAME constant ('{UMS_SERVER_NAME}') not found among them directly."
                )


        # 3. If still ambiguous after hints, we cannot safely choose.
        self.logger.warning(
            f"AML _find_tool_server: Could not unambiguously resolve server for base function '{target_base_function_name}'. "
            f"Active candidates: {unique_candidate_servers}. Original hint was '{expected_server_name_hint}'. "
            f"Input identifier: '{tool_identifier_mcp_style}'. Returning None."
        )
        return None

    async def _set_default_thought_chain_id(self):
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.debug("No active workflow for default thought chain.")
            return

        # Construct full MCP name for get_workflow_details
        get_details_mcp_tool_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_WORKFLOW_DETAILS)

        if self._find_tool_server(get_details_mcp_tool_name):  # Check availability using full MCP name
            try:
                details = await self._execute_tool_call_internal(
                    get_details_mcp_tool_name,  # Call using full MCP name
                    {
                        "workflow_id": current_wf_id,
                        "include_thoughts": True,
                        "include_actions": False,
                        "include_artifacts": False,
                        "include_memories": False,
                    },
                    record_action=False,
                )
                if details.get("success"):
                    thought_chains = details.get("thought_chains")
                    if isinstance(thought_chains, list) and thought_chains:
                        first_chain = thought_chains[0]
                        chain_id = first_chain.get("thought_chain_id")
                        if chain_id:
                            self.state.current_thought_chain_id = chain_id
                            self.logger.info(f"Set current_thought_chain_id: {_fmt_id(chain_id)} for WF {_fmt_id(current_wf_id)}")
                            return
                        else:
                            self.logger.warning(f"Primary thought chain lacks ID for WF {current_wf_id}.")
                    else:
                        self.logger.warning(f"No thought chains in details for WF {current_wf_id}.")
                else:
                    self.logger.error(f"Tool '{get_details_mcp_tool_name}' failed for default chain: {details.get('error')}")
            except Exception as e:
                self.logger.error(f"Error fetching WF details for default chain: {e}", exc_info=False)
        else:
            self.logger.warning(f"Tool '{get_details_mcp_tool_name}' unavailable for default chain.")
        self.logger.info(f"Could not set primary thought chain ID for WF {_fmt_id(current_wf_id)}.")

    async def _check_workflow_exists(self, workflow_id: str) -> bool:
        get_details_mcp_tool_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_WORKFLOW_DETAILS)
        self.logger.debug(f"Checking existence of workflow {_fmt_id(workflow_id)} using {get_details_mcp_tool_name}.")

        if not self._find_tool_server(get_details_mcp_tool_name):
            self.logger.error(f"Tool {get_details_mcp_tool_name} unavailable.")
            return False
        try:
            result = await self._execute_tool_call_internal(
                get_details_mcp_tool_name,
                {
                    "workflow_id": workflow_id,
                    "include_actions": False,
                    "include_artifacts": False,
                    "include_thoughts": False,
                    "include_memories": False,
                },
                record_action=False,
            )
            return isinstance(result, dict) and result.get("success", False)
        except ToolInputError as e:  # UMS tool itself raises ToolInputError if not found
            self.logger.debug(f"Workflow {_fmt_id(workflow_id)} not found (ToolInputError: {e}).")
            return False
        except Exception as e:
            self.logger.error(f"Error checking WF {_fmt_id(workflow_id)}: {e}", exc_info=False)
            return False

    async def _validate_goal_stack_on_load(self):
        if not self.state.workflow_id:
            self.logger.warning("Cannot validate goal stack on load: No active workflow ID in state.")
            self.state.goal_stack = []
            self.state.current_goal_id = None
            return

        if self.state.current_goal_id:
            self.logger.info(f"Validating loaded goal stack, current_goal_id: {_fmt_id(self.state.current_goal_id)}")
            ums_stack = await self._fetch_goal_stack_from_ums(self.state.current_goal_id)
            if ums_stack:
                if ums_stack[-1].get("goal_id") == self.state.current_goal_id:
                    self.state.goal_stack = ums_stack
                    self.logger.info(
                        f"Goal stack re-validated and updated from UMS. Current: {_fmt_id(self.state.current_goal_id)}, Stack depth: {len(self.state.goal_stack)}"
                    )
                else:
                    self.logger.warning(
                        f"Loaded current_goal_id {_fmt_id(self.state.current_goal_id)} "
                        f"could not be confirmed as leaf of a valid stack from UMS (UMS leaf: {_fmt_id(ums_stack[-1].get('goal_id'))}). "
                        f"Resetting current_goal_id and stack."
                    )
                    self.state.goal_stack = []
                    self.state.current_goal_id = None
            else:
                self.logger.warning(
                    f"Could not reconstruct goal stack from UMS for loaded current_goal_id {_fmt_id(self.state.current_goal_id)}. "
                    f"Resetting goal stack and current_goal_id."
                )
                self.state.goal_stack = []
                self.state.current_goal_id = None
        else:
            if self.state.goal_stack:
                self.logger.info("Loaded goal stack present but no current_goal_id. Clearing local stack.")
                self.state.goal_stack = []
            else:
                self.logger.debug("Goal stack and current_goal_id are empty on load, nothing to validate.")

    def _detect_plan_cycle(self, plan: List[PlanStep]) -> bool:
        if not plan:
            return False
        adj: Dict[str, Set[str]] = defaultdict(set)
        plan_step_ids = {step.id for step in plan}
        for step in plan:
            for dep_id in step.depends_on:
                if dep_id in plan_step_ids:
                    adj[step.id].add(dep_id)
                else:
                    self.logger.warning(f"Step {_fmt_id(step.id)} depends on non-existent step {_fmt_id(dep_id)}.")
        path: Set[str] = set()
        visited: Set[str] = set()

        def dfs(node_id: str) -> bool:
            path.add(node_id)
            visited.add(node_id)
            for neighbor_id in adj[node_id]:
                if neighbor_id in path:
                    self.logger.warning(f"Cycle detected: {_fmt_id(node_id)} -> {_fmt_id(neighbor_id)}")
                    return True
                if neighbor_id not in visited and dfs(neighbor_id):
                    return True
            path.remove(node_id)
            return False

        for step_id in plan_step_ids:
            if step_id not in visited and dfs(step_id):
                return True
        return False

    async def _check_prerequisites(self, ids: List[str]) -> Tuple[bool, str]:
        if not ids:
            return True, "No dependencies."

        get_action_details_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_ACTION_DETAILS)
        if not self._find_tool_server(get_action_details_mcp_name):
            self.logger.error(f"Tool for '{UMS_FUNC_GET_ACTION_DETAILS}' unavailable.")
            return False, f"Tool for '{UMS_FUNC_GET_ACTION_DETAILS}' unavailable."

        self.logger.debug(f"Checking prerequisites: {[_fmt_id(i) for i in ids]}")
        try:
            res = await self._execute_tool_call_internal(
                get_action_details_mcp_name, {"action_ids": ids, "include_dependencies": False}, record_action=False
            )
            if not res.get("success"):
                error_msg = res.get("error", "Unknown error.")
                self.logger.warning(f"Dep check failed: {error_msg}")
                return False, f"Failed: {error_msg}"
            actions_found = res.get("actions", [])
            found_ids = {a.get("action_id") for a in actions_found}
            missing_ids = list(set(ids) - found_ids)
            if missing_ids:
                self.logger.warning(f"Dep actions not found: {[_fmt_id(i) for i in missing_ids]}")
                return False, f"Not found: {[_fmt_id(i) for i in missing_ids]}"
            incomplete = [
                f"'{a.get('title', _fmt_id(a.get('action_id')))}' (Status: {a.get('status', 'UNK')})"
                for a in actions_found
                if a.get("status") != ActionStatus.COMPLETED.value
            ]
            if incomplete:
                reason = f"Not completed: {', '.join(incomplete)}"
                self.logger.warning(reason)
                return False, reason
            self.logger.debug("All deps completed.")
            return True, "All deps completed."
        except Exception as e:
            self.logger.error(f"Error checking prereqs: {e}", exc_info=True)
            return False, f"Exception: {str(e)}"

    async def _record_action_start_internal(
        self, tool_name_mcp: str, tool_args: Dict[str, Any], planned_dependencies: Optional[List[str]] = None
    ) -> Optional[str]:
        # tool_name_mcp is the original MCP name (e.g., "Ultimate MCP Server:some_tool")
        record_action_start_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_ACTION_START)

        if not self._find_tool_server(record_action_start_mcp_name):
            self.logger.error(f"Tool for '{UMS_FUNC_RECORD_ACTION_START}' unavailable.")
            return None
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning("No active WF ID for action start.")
            return None

        base_tool_name_for_title = tool_name_mcp.split(":")[-1]
        payload = {
            "workflow_id": current_wf_id,
            "title": f"Execute: {base_tool_name_for_title}",
            "action_type": ActionType.TOOL_USE.value,
            "tool_name": tool_name_mcp,  # Store the original MCP name of the tool being executed
            "tool_args": tool_args,
            "reasoning": f"Agent initiated call to tool: {tool_name_mcp}",
        }
        action_id: Optional[str] = None
        try:
            res = await self._execute_tool_call_internal(record_action_start_mcp_name, payload, record_action=False)
            if res.get("success"):
                action_id = res.get("action_id")
                if action_id:
                    self.logger.debug(f"Action started: {_fmt_id(action_id)} for tool {tool_name_mcp}")
                    if planned_dependencies:
                        await self._record_action_dependencies_internal(action_id, planned_dependencies)
                else:
                    self.logger.warning(f"{record_action_start_mcp_name} success but no action_id.")
            else:
                self.logger.error(f"Failed record action start for {tool_name_mcp}: {res.get('error')}")
        except Exception as e:
            self.logger.error(f"Exception recording action start for {tool_name_mcp}: {e}", exc_info=True)
        return action_id

    async def _record_action_dependencies_internal(self, source_id: str, target_ids: List[str]) -> None:
        if not source_id or not target_ids:
            return
        valid_targets = {tid for tid in target_ids if tid and tid != source_id}
        if not valid_targets:
            return

        add_dep_mcp_tool_name = self._get_ums_tool_mcp_name(UMS_FUNC_ADD_ACTION_DEPENDENCY)
        if not self._find_tool_server(add_dep_mcp_tool_name):
            self.logger.error(f"Tool for '{UMS_FUNC_ADD_ACTION_DEPENDENCY}' unavailable.")
            return
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning(f"No active WF ID for dep record for action {_fmt_id(source_id)}.")
            return

        self.logger.debug(
            f"Recording {len(valid_targets)} deps for action {_fmt_id(source_id)}: depends on {[_fmt_id(tid) for tid in valid_targets]}"
        )
        tasks = [
            asyncio.create_task(
                self._execute_tool_call_internal(
                    add_dep_mcp_tool_name,
                    {"source_action_id": source_id, "target_action_id": target_id, "dependency_type": "requires"},
                    record_action=False,
                )
            )
            for target_id in valid_targets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, res in enumerate(results):
            if isinstance(res, Exception):
                self.logger.error(f"Error recording dep {_fmt_id(source_id)} -> target_idx_{i}: {res}", exc_info=False)
            elif isinstance(res, dict) and not res.get("success"):
                self.logger.warning(f"Failed recording dep {_fmt_id(source_id)} -> target_idx_{i}: {res.get('error')}")

    async def _record_action_completion_internal(self, action_id: str, result: Dict[str, Any]) -> None:
        completion_mcp_tool_name = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_ACTION_COMPLETION)
        if not self._find_tool_server(completion_mcp_tool_name):
            self.logger.error(f"Tool for '{UMS_FUNC_RECORD_ACTION_COMPLETION}' unavailable.")
            return
        status = ActionStatus.COMPLETED.value if isinstance(result, dict) and result.get("success") else ActionStatus.FAILED.value
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.warning(f"No active WF ID for action completion {_fmt_id(action_id)}.")
            return

        payload = {"action_id": action_id, "status": status, "tool_result": result}
        try:
            completion_res = await self._execute_tool_call_internal(completion_mcp_tool_name, payload, record_action=False)
            if completion_res.get("success"):
                self.logger.debug(f"Action completion recorded for {_fmt_id(action_id)} (Status: {status})")
            else:
                self.logger.error(f"Failed record action completion for {_fmt_id(action_id)}: {completion_res.get('error')}")
        except Exception as e:
            self.logger.error(f"Exception recording action completion for {_fmt_id(action_id)}: {e}", exc_info=True)

    async def _run_auto_linking(self, memory_id: str, *, workflow_id: Optional[str], context_id: Optional[str]) -> None:
        # This method uses full MCP names for UMS tool calls.
        # workflow_id here is the snapshot taken when the task was scheduled.
        current_agent_workflow_id = self.state.workflow_id  # Get current agent state at execution time
        if workflow_id != current_agent_workflow_id or self._shutdown_event.is_set():
            self.logger.debug(
                f"Skipping auto-link for {_fmt_id(memory_id)}: WF changed ({_fmt_id(workflow_id)} vs {_fmt_id(current_agent_workflow_id)}) or shutdown."
            )
            return

        try:
            if not memory_id or not workflow_id:  # Use the snapshot workflow_id
                self.logger.debug(f"Skipping auto-link: Missing memory_id ({_fmt_id(memory_id)}) or snapshot workflow_id ({_fmt_id(workflow_id)}).")
                return
            await asyncio.sleep(random.uniform(*AUTO_LINKING_DELAY_SECS))
            if self._shutdown_event.is_set():
                return

            self.logger.debug(f"Attempting auto-link for memory {_fmt_id(memory_id)} in WF {_fmt_id(workflow_id)}...")

            get_mem_by_id_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_MEMORY_BY_ID)
            source_res = await self._execute_tool_call_internal(
                get_mem_by_id_mcp_name,
                {"memory_id": memory_id, "include_links": False, "workflow_id": workflow_id},  # Pass workflow_id
                record_action=False,
            )
            if not source_res.get("success") or source_res.get("workflow_id") != workflow_id:
                self.logger.warning(f"Auto-link failed for {_fmt_id(memory_id)}: Source mem error or WF mismatch. Resp: {source_res}")
                return
            source_mem = source_res
            query_text = source_mem.get("description", "") or source_mem.get("content", "")[:200]
            if not query_text:
                self.logger.debug(f"Skipping auto-link for {_fmt_id(memory_id)}: No query text.")
                return

            search_base_func = (
                UMS_FUNC_HYBRID_SEARCH
                if self._find_tool_server(self._get_ums_tool_mcp_name(UMS_FUNC_HYBRID_SEARCH))
                else UMS_FUNC_SEARCH_SEMANTIC_MEMORIES
            )
            search_mcp_tool_name = self._get_ums_tool_mcp_name(search_base_func)

            if not self._find_tool_server(search_mcp_tool_name):
                self.logger.warning(f"Skipping auto-link: Tool for {search_base_func} unavailable.")
                return

            search_args: Dict[str, Any] = {
                "workflow_id": workflow_id,
                "query": query_text,
                "limit": self.auto_linking_max_links + 1,
                "threshold": self.auto_linking_threshold,
                "include_content": False,
            }
            if search_base_func == UMS_FUNC_HYBRID_SEARCH:
                search_args.update({"semantic_weight": 0.8, "keyword_weight": 0.2})

            similar_res = await self._execute_tool_call_internal(search_mcp_tool_name, search_args, record_action=False)
            if not similar_res.get("success"):
                self.logger.warning(f"Auto-link search failed for {_fmt_id(memory_id)}: {similar_res.get('error')}")
                return

            link_count = 0
            score_key = "hybrid_score" if search_base_func == UMS_FUNC_HYBRID_SEARCH else "similarity"

            create_link_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_LINK)
            if not self._find_tool_server(create_link_mcp_name):
                self.logger.warning(f"Skipping link creation: Tool for '{UMS_FUNC_CREATE_LINK}' unavailable.")
                return

            for sim_mem_summary in similar_res.get("memories", []):
                if self._shutdown_event.is_set():
                    break
                target_id = sim_mem_summary.get("memory_id")
                sim_score = sim_mem_summary.get(score_key, 0.0)
                if not target_id or target_id == memory_id:
                    continue

                # No need to fetch target_mem again if we trust workflow_id from search_args
                # and sim_mem_summary usually contains enough for simple link typing.
                link_type = LinkType.RELATED.value  # Default
                # Simple link typing based on source and target types from summaries
                source_type_from_summary = source_mem.get("memory_type")
                target_type_from_summary = sim_mem_summary.get("memory_type")
                if source_type_from_summary == MemoryType.INSIGHT.value and target_type_from_summary == MemoryType.FACT.value:
                    link_type = LinkType.SUPPORTS.value

                link_args = {
                    "source_memory_id": memory_id,
                    "target_memory_id": target_id,
                    "link_type": link_type,
                    "strength": round(sim_score, 3),
                    "description": f"Auto-link ({link_type})",
                    "workflow_id": workflow_id,  # Pass workflow_id for the link operation itself
                }
                link_result = await self._execute_tool_call_internal(create_link_mcp_name, link_args, record_action=False)
                if link_result.get("success"):
                    link_count += 1
                    self.logger.debug(f"Auto-linked {_fmt_id(memory_id)} to {_fmt_id(target_id)} ({link_type}, {sim_score:.2f})")
                else:
                    self.logger.warning(f"Failed auto-create link {_fmt_id(memory_id)}->{_fmt_id(target_id)}: {link_result.get('error')}")
                if link_count >= self.auto_linking_max_links:
                    break
                await asyncio.sleep(0.1)  # Be nice
        except Exception as e:
            self.logger.warning(f"Error in auto-linking for {_fmt_id(memory_id)} in WF {_fmt_id(workflow_id)}: {e}", exc_info=False)


    async def _execute_tool_call_internal(
        self, tool_name_mcp: str, arguments: Dict[str, Any], record_action: bool = True, planned_dependencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:  # Returns the standard_envelope
        """
        Core internal method to execute a tool call (UMS or agent-internal),
        record it as a UMS action (if applicable), handle retries, parse results,
        and manage agent state updates related to the call.

        This method now calls `self.mcp_client._execute_tool_and_parse_for_agent`
        which returns a pre-parsed standardized dictionary envelope.
        """
        def _force_print(*args_print, **kwargs_print): # Keep for debugging
            print(*args_print, file=sys.stderr, flush=True, **kwargs_print)

        current_base_func_name = self._get_base_function_name(tool_name_mcp)
        _force_print(
            f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: Entered for tool '{tool_name_mcp}'. Base Func: '{current_base_func_name}'. Args: {str(arguments)[:200]}..."
        )

        # Initialize the standard envelope with error defaults
        standard_envelope: Dict[str, Any] = {
            "success": False,
            "data": None,
            "error_type": "UnknownInternalError_AML", # More specific initial error
            "error_message": "AML: Initial error before tool call attempt.",
            "status_code": None,
            "details": None,
        }

        target_server = self._find_tool_server(tool_name_mcp)
        if not target_server and tool_name_mcp != AGENT_TOOL_UPDATE_PLAN:
            err_msg = f"Tool server unavailable for {tool_name_mcp}"
            _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: ERROR - {err_msg}")
            standard_envelope.update({"error_message": err_msg, "error_type": "ServerUnavailable_AML", "status_code": 503})
            self.state.last_error_details = {"tool": tool_name_mcp, "error": err_msg, "type": "ServerUnavailable", "status_code": 503}
            if not self.state.needs_replan: self.state.needs_replan = True
            self.state.last_action_summary = f"{tool_name_mcp} -> Failed (ServerUnavailable_AML): {err_msg}"
            return standard_envelope

        if tool_name_mcp == AGENT_TOOL_UPDATE_PLAN:
            self.logger.info(f"AML EXEC_TOOL_INTERNAL: Handling AGENT_TOOL_UPDATE_PLAN directly.")
            new_plan_steps_data = arguments.get("plan")
            if not isinstance(new_plan_steps_data, list):
                err_msg = "Invalid plan: 'plan' argument must be a list of steps for agent:update_plan."
                self.logger.error(err_msg)
                standard_envelope.update({"success": False, "error_message": err_msg, "error_type": "PlanUpdateError_AML"})
                self.state.last_error_details = {"tool": tool_name_mcp, "error": err_msg, "type": "PlanUpdateError"}
                self.state.needs_replan = True
                self.state.last_action_summary = f"{tool_name_mcp} -> Failed: {err_msg}"
                return standard_envelope
            try:
                validated_plan = [PlanStep(**step_data) for step_data in new_plan_steps_data]
                if self._detect_plan_cycle(validated_plan):
                    err_msg = "Plan cycle detected in proposed update."
                    self.logger.error(err_msg)
                    standard_envelope.update({"success": False, "error_message": err_msg, "error_type": "PlanValidationError_AML"})
                    self.state.last_error_details = {"tool": tool_name_mcp, "error": err_msg, "type": "PlanValidationError"}
                    self.state.needs_replan = True
                    self.state.last_action_summary = f"{tool_name_mcp} -> Failed: {err_msg}"
                    return standard_envelope

                self.state.current_plan = validated_plan
                self.state.needs_replan = False
                self.state.last_error_details = None
                self.state.consecutive_error_count = 0
                msg = f"Plan updated with {len(validated_plan)} steps."
                self.logger.info(f"AML EXEC_TOOL_INTERNAL: {msg}")
                standard_envelope.update({"success": True, "data": {"message": msg}, "error_type": None, "error_message": None})
                self.state.last_action_summary = f"{tool_name_mcp} -> Success: {msg}"
                return standard_envelope
            except (ValidationError, TypeError) as e_plan_val:
                err_msg = f"Plan validation error: {e_plan_val}"
                self.logger.error(f"AML EXEC_TOOL_INTERNAL: {err_msg}")
                standard_envelope.update({"success": False, "error_message": err_msg, "error_type": "PlanValidationError_AML"})
                self.state.last_error_details = {"tool": tool_name_mcp, "error": err_msg, "type": "PlanValidationError"}
                self.state.needs_replan = True
                self.state.last_action_summary = f"{tool_name_mcp} -> Failed: {err_msg}"
                return standard_envelope

        final_arguments = arguments.copy()
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        current_ctx_id = self.state.context_id
        current_goal_id_for_tool = self.state.current_goal_id

        if (final_arguments.get("workflow_id") is None and current_wf_id and
            current_base_func_name not in [UMS_FUNC_CREATE_WORKFLOW, UMS_FUNC_LIST_WORKFLOWS]):
            final_arguments["workflow_id"] = current_wf_id
        if (final_arguments.get("context_id") is None and current_ctx_id and
            current_base_func_name in {UMS_FUNC_GET_WORKING_MEMORY, UMS_FUNC_OPTIMIZE_WM, UMS_FUNC_AUTO_FOCUS, UMS_FUNC_FOCUS_MEMORY,
                                       UMS_FUNC_SAVE_COGNITIVE_STATE, UMS_FUNC_LOAD_COGNITIVE_STATE, UMS_FUNC_GET_RICH_CONTEXT_PACKAGE}):
            final_arguments["context_id"] = current_ctx_id
            if current_base_func_name == UMS_FUNC_LOAD_COGNITIVE_STATE and "state_id" not in final_arguments:
                final_arguments["state_id"] = current_ctx_id
        if (final_arguments.get("thought_chain_id") is None and self.state.current_thought_chain_id and
            current_base_func_name == UMS_FUNC_RECORD_THOUGHT):
            final_arguments["thought_chain_id"] = self.state.current_thought_chain_id
        if (final_arguments.get("parent_goal_id") is None and current_goal_id_for_tool and
            current_base_func_name == UMS_FUNC_CREATE_GOAL):
            final_arguments["parent_goal_id"] = current_goal_id_for_tool

        _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: Final args for '{tool_name_mcp}': {str(final_arguments)[:200]}...")

        if planned_dependencies:
            _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: Checking prerequisites for '{tool_name_mcp}': {[_fmt_id(d) for d in planned_dependencies]}")
            ok, reason = await self._check_prerequisites(planned_dependencies)
            if not ok:
                err_msg = f"Prerequisites not met for {tool_name_mcp}: {reason}"
                _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: ERROR - {err_msg}")
                standard_envelope.update({
                    "error_message": err_msg, "error_type": "DependencyNotMetError_AML",
                    "status_code": 412, "details": {"dependencies": planned_dependencies}
                })
                self.state.last_error_details = {"tool": tool_name_mcp, "error": err_msg, "type": "DependencyNotMetError", "dependencies": planned_dependencies, "status_code": 412}
                if not self.state.needs_replan: self.state.needs_replan = True
                self.state.last_action_summary = f"{tool_name_mcp} -> Failed (DependencyNotMetError_AML): {err_msg}"
                return standard_envelope
            _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: Prerequisites met for {tool_name_mcp}.")

        action_id: Optional[str] = None
        should_record_this_action = record_action and current_base_func_name not in self._INTERNAL_OR_META_TOOLS_BASE_NAMES
        if should_record_this_action:
            _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: Recording UMS action start for '{tool_name_mcp}'.")
            action_id = await self._record_action_start_internal(tool_name_mcp, final_arguments, planned_dependencies)
            if not action_id:
                err_msg = f"Failed to record UMS action_start for tool {tool_name_mcp}."
                _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: ERROR - {err_msg}")
                standard_envelope.update({
                    "error_message": err_msg, "error_type": "UMSError_AML",
                    "details": {"reason": "ActionStartFailed"}, "status_code": 500
                })
                self.state.last_error_details = {"tool": tool_name_mcp, "error": err_msg, "type": "UMSError", "reason": "ActionStartFailed"}
                if not self.state.needs_replan: self.state.needs_replan = True
                self.state.last_action_summary = f"{tool_name_mcp} -> Failed (UMSError_AML): ActionStartFailed"
                return standard_envelope

        record_stats = self.state.tool_usage_stats.setdefault(tool_name_mcp, {"success": 0, "failure": 0, "latency_ms_total": 0.0})
        idempotent_tool_call = current_base_func_name in {
            UMS_FUNC_GET_MEMORY_BY_ID, UMS_FUNC_SEARCH_SEMANTIC_MEMORIES, UMS_FUNC_HYBRID_SEARCH,
            UMS_FUNC_QUERY_MEMORIES, UMS_FUNC_GET_ACTION_DETAILS, UMS_FUNC_LIST_WORKFLOWS,
            UMS_FUNC_COMPUTE_STATS, UMS_FUNC_GET_WORKING_MEMORY, UMS_FUNC_GET_LINKED_MEMORIES,
            UMS_FUNC_GET_ARTIFACTS, UMS_FUNC_GET_ARTIFACT_BY_ID, UMS_FUNC_GET_ACTION_DEPENDENCIES,
            UMS_FUNC_GET_THOUGHT_CHAIN, UMS_FUNC_GET_WORKFLOW_DETAILS, UMS_FUNC_GET_GOAL_DETAILS,
            UMS_FUNC_SUMMARIZE_TEXT, UMS_FUNC_SUMMARIZE_CONTEXT_BLOCK, UMS_FUNC_GET_RICH_CONTEXT_PACKAGE,
            UMS_FUNC_GET_RECENT_ACTIONS, UMS_FUNC_VISUALIZE_REASONING_CHAIN, UMS_FUNC_VISUALIZE_MEMORY_NETWORK,
            UMS_FUNC_GENERATE_WORKFLOW_REPORT, UMS_FUNC_LOAD_COGNITIVE_STATE,
        }
        start_ts = time.time()

        try:
            _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: Attempting MCPClient._execute_tool_and_parse_for_agent for '{tool_name_mcp}' on server '{target_server}'.")

            async def _do_mcp_agent_call():
                call_args = {k: v for k, v in final_arguments.items() if v is not None}
                # This now calls the new wrapper method in MCPClient
                agent_standard_envelope_from_wrapper: Dict[str, Any] = await self.mcp_client._execute_tool_and_parse_for_agent(
                    target_server, tool_name_mcp, call_args
                )
                return agent_standard_envelope_from_wrapper

            # `result_from_mcpc_wrapper` is the standardized dictionary from the MCPClient wrapper
            result_from_mcpc_wrapper: Dict[str, Any] = await self._with_retries(
                _do_mcp_agent_call, max_retries=3 if idempotent_tool_call else 1
            )
            
            _force_print(
                f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: MCPClient._execute_tool_and_parse_for_agent for '{tool_name_mcp}' RETURNED. "
                f"Type: {type(result_from_mcpc_wrapper)}. Value Preview: {str(result_from_mcpc_wrapper)[:500]}..."
            )

            latency_ms = (time.time() - start_ts) * 1000
            current_latency_total = record_stats.get("latency_ms_total", 0.0)
            if not isinstance(current_latency_total, (int, float)): current_latency_total = 0.0
            record_stats["latency_ms_total"] = current_latency_total + latency_ms

            # Assign the result from the wrapper directly to standard_envelope
            if isinstance(result_from_mcpc_wrapper, dict):
                standard_envelope = result_from_mcpc_wrapper 
            else: 
                _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: CRITICAL - MCPClient._execute_tool_and_parse_for_agent returned non-dict: {type(result_from_mcpc_wrapper)}")
                standard_envelope.update({
                    "success": False,
                    "error_message": "Internal MCPClient error: Agent wrapper did not return expected dictionary.",
                    "error_type": "MCPClientContractError_AML", 
                })
        
        except asyncio.CancelledError:
            _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: CAUGHT CancelledError for tool '{tool_name_mcp}' during _with_retries/agent_call.")
            standard_envelope.update({"success": False, "error_message": "Tool execution cancelled by AML.", "error_type": "CancelledError_AML"})
            if action_id and should_record_this_action:
                await self._record_action_completion_internal(
                    action_id, 
                    {"success": False, "data": {"error_details": "Operation cancelled by AML"}, 
                     "error_type": "CancelledError_AML", "error_message": "Operation Cancelled by AML", 
                     "status_code": None, "details": {"reason": "AML retry wrapper or agent call caught cancellation"}}
                )
            raise 
        except Exception as e_unhandled_retries: 
            err_str_unhandled = str(e_unhandled_retries)
            _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: CAUGHT UNEXPECTED EXCEPTION from _with_retries/agent_call for tool '{tool_name_mcp}': {type(e_unhandled_retries).__name__} - {e_unhandled_retries}")
            self.logger.error(f"AML EXEC_TOOL_INTERNAL: Unexpected Error from _with_retries/agent_call for {tool_name_mcp}: {err_str_unhandled}", exc_info=True)
            standard_envelope.update({
                "success": False,
                "error_message": f"Unexpected error during tool call (AML retries/wrapper): {err_str_unhandled}",
                "error_type": "AMLRetryOrWrapperError", 
            })

        _force_print(
            f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL: Standardized Envelope for '{tool_name_mcp}' (after MCPC call and parsing): "
            f"Success={standard_envelope.get('success')}, ErrorType={standard_envelope.get('error_type')}, "
            f"Data is dict: {isinstance(standard_envelope.get('data'), dict)}, "
            f"Data Keys (if dict): {list(standard_envelope.get('data', {}).keys()) if isinstance(standard_envelope.get('data'), dict) else 'N/A'}"
        )
        
        current_success_count = record_stats.get("success", 0)
        if not isinstance(current_success_count, int): current_success_count = 0
        current_failure_count = record_stats.get("failure", 0)
        if not isinstance(current_failure_count, int): current_failure_count = 0

        if standard_envelope.get("success"):
            record_stats["success"] = current_success_count + 1
            if self.state.last_error_details and self.state.last_error_details.get("tool") == tool_name_mcp:
                self.state.last_error_details = None 
            
            ums_data_for_bg_tasks = standard_envelope.get("data")
            if isinstance(ums_data_for_bg_tasks, dict):
                bg_task_workflow_id = ums_data_for_bg_tasks.get("workflow_id", current_wf_id) 
                bg_task_context_id = ums_data_for_bg_tasks.get("context_id", current_ctx_id)   

                if bg_task_workflow_id: 
                    if current_base_func_name in [UMS_FUNC_STORE_MEMORY, UMS_FUNC_UPDATE_MEMORY] and ums_data_for_bg_tasks.get("memory_id"):
                        self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=ums_data_for_bg_tasks["memory_id"], workflow_id=bg_task_workflow_id, context_id=bg_task_context_id)
                    elif current_base_func_name == UMS_FUNC_RECORD_ARTIFACT and ums_data_for_bg_tasks.get("linked_memory_id"):
                        self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=ums_data_for_bg_tasks["linked_memory_id"], workflow_id=bg_task_workflow_id, context_id=bg_task_context_id)
                    elif current_base_func_name == UMS_FUNC_RECORD_THOUGHT and ums_data_for_bg_tasks.get("linked_memory_id"):
                        self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=ums_data_for_bg_tasks["linked_memory_id"], workflow_id=bg_task_workflow_id, context_id=bg_task_context_id)
                    
                    mem_ids_to_check_for_promo = set()
                    if current_base_func_name == UMS_FUNC_GET_MEMORY_BY_ID and isinstance(ums_data_for_bg_tasks.get("memory_id"), str):
                        mem_ids_to_check_for_promo.add(ums_data_for_bg_tasks["memory_id"])
                    elif current_base_func_name == UMS_FUNC_GET_WORKING_MEMORY and isinstance(ums_data_for_bg_tasks.get("working_memories"), list):
                        mems_list = ums_data_for_bg_tasks["working_memories"]
                        mem_ids_to_check_for_promo.update(m.get("memory_id") for m in mems_list[:3] if isinstance(m, dict) and m.get("memory_id"))
                        if ums_data_for_bg_tasks.get("focal_memory_id"): mem_ids_to_check_for_promo.add(ums_data_for_bg_tasks["focal_memory_id"])
                    elif current_base_func_name in [UMS_FUNC_QUERY_MEMORIES, UMS_FUNC_HYBRID_SEARCH, UMS_FUNC_SEARCH_SEMANTIC_MEMORIES] and isinstance(ums_data_for_bg_tasks.get("memories"), list):
                        mems_list = ums_data_for_bg_tasks["memories"]
                        mem_ids_to_check_for_promo.update(m.get("memory_id") for m in mems_list[:3] if isinstance(m, dict) and m.get("memory_id"))
                    
                    for mem_id_chk_promo in filter(None, mem_ids_to_check_for_promo):
                        self._start_background_task(AgentMasterLoop._check_and_trigger_promotion, memory_id=mem_id_chk_promo, workflow_id=bg_task_workflow_id, context_id=bg_task_context_id)
                else:
                    self.logger.warning(f"AML_EXEC_TOOL_INTERNAL: Cannot schedule BG tasks for tool '{tool_name_mcp}' - no valid workflow_id (UMS payload or agent state). UMS data: {str(ums_data_for_bg_tasks)[:100]}")
        else: 
            record_stats["failure"] = current_failure_count + 1
            agent_facing_error_type = standard_envelope.get("error_type", "ToolExecutionError_AML") # Default to AML specific error
            
            # Map error types from the wrapper/UMS to types the LLM's recovery strategies expect
            if agent_facing_error_type in [
                "UMSToolReportedFailure", "ContentParsingError", "InternalParsingWrapperError", 
                "MCPClientContractError_AML", "AgentSideToolWrapperUnexpectedError", 
                "UMSUnparsableJSONError", "UMSError_AML", 
                "ToolInternalError_AML", "InvalidInputError_AML", # Suffix to indicate where it was set
                "UMSError" # Generic from wrapper
            ]:
                agent_facing_error_type = "UMSError" 
            elif agent_facing_error_type == "ToolMaxRetriesOrServerError":
                agent_facing_error_type = "ServerUnavailable"
            elif agent_facing_error_type in ["AMLToolCallProcessingError", "AMLRetryOrWrapperError", "UnknownInternalError_AML", "AMLUnexpectedExecutionError"]:
                agent_facing_error_type = "AgentError" 
            
            self.state.last_error_details = {
                "tool": tool_name_mcp, 
                "args": arguments, 
                "error": standard_envelope.get("error_message", "Unknown tool failure."),
                "status_code": standard_envelope.get("status_code"),
                "type": agent_facing_error_type, 
                "details": standard_envelope.get("details"),
            }
            if not idempotent_tool_call and not self.state.needs_replan: 
                self.state.needs_replan = True
                self.logger.info(f"AML EXEC_TOOL_INTERNAL: Setting needs_replan=True due to failure in non-idempotent tool '{tool_name_mcp}'. Error type: {agent_facing_error_type}")

        summary_text_for_log = ""
        if standard_envelope.get("success"):
            data_payload_for_summary = standard_envelope.get("data")
            if isinstance(data_payload_for_summary, dict):
                summary_keys_log = ["summary", "message", "memory_id", "action_id", "artifact_id", "link_id", "thought_chain_id", "thought_id", "state_id", "report", "visualization", "goal_id", "workflow_id", "title"]
                found_summary_key = False
                for k_log in summary_keys_log:
                    if k_log in data_payload_for_summary and data_payload_for_summary[k_log] is not None:
                        val_str_log = str(data_payload_for_summary[k_log])
                        summary_text_for_log = f"{k_log}: {_fmt_id(val_str_log) if 'id' in k_log.lower() else val_str_log}"
                        found_summary_key = True; break
                if not found_summary_key:
                    ums_payload_success = data_payload_for_summary.get("success") 
                    if ums_payload_success is True and len(data_payload_for_summary) == 1 and "success" in data_payload_for_summary:
                        summary_text_for_log = "Success (UMS tool returned minimal success payload)."
                    elif ums_payload_success is False: 
                        summary_text_for_log = f"UMS Tool Reported Failure: {data_payload_for_summary.get('error', data_payload_for_summary.get('error_message', 'Unknown UMS tool error'))[:70]}"
                    else: 
                        generic_parts_log = [f"{k_s}={_fmt_id(str(v_s)) if 'id' in k_s.lower() else str(v_s)[:20]}" for k_s, v_s in data_payload_for_summary.items() if v_s is not None and k_s != "success"][:3]
                        summary_text_for_log = f"Success. Data: {', '.join(generic_parts_log)}" if generic_parts_log else "Success (No specific data in UMS 'data' field)."
            elif data_payload_for_summary is not None: 
                summary_text_for_log = f"Success (Data: {str(data_payload_for_summary)[:50]}...)"
            else: 
                summary_text_for_log = "Success (No data payload returned by UMS tool)."
        else: 
            err_type_log = standard_envelope.get("error_type", "Err_AML")
            err_msg_log = str(standard_envelope.get("error_message", "Unknown_AML"))[:100]
            summary_text_for_log = f"Failed ({err_type_log}): {err_msg_log}"
        
        if standard_envelope.get("status_code"):
            summary_text_for_log += f" (Code: {standard_envelope['status_code']})"
        self.state.last_action_summary = f"{tool_name_mcp} -> {summary_text_for_log}"
        _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL (AML): Standardized Last Action Summary: {self.state.last_action_summary}")

        if action_id and should_record_this_action:
            _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL (AML): Recording UMS action completion for action ID {_fmt_id(action_id)} of tool '{tool_name_mcp}'.")
            # Pass the standard_envelope, which _record_action_completion_internal expects to find 'tool_result' (which is standard_envelope['data'])
            # and 'status' (derived from standard_envelope['success']) within its own logic.
            # To be absolutely clear for _record_action_completion_internal, we can construct what it expects:
            completion_payload = {
                "action_id": action_id,
                "status": ActionStatus.COMPLETED.value if standard_envelope.get("success") else ActionStatus.FAILED.value,
                "tool_result": standard_envelope # Store the whole envelope as the tool_result in UMS action record
            }
            await self._record_action_completion_internal(action_id, completion_payload) 

        _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL (AML): Calling side effects for base_func_name='{current_base_func_name}'.")
        await self._handle_workflow_and_goal_side_effects(current_base_func_name, final_arguments, standard_envelope) 
        _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL (AML): After calling side effects for base_func_name='{current_base_func_name}'.")
        
        _force_print(f"FORCE_PRINT AML_EXEC_TOOL_INTERNAL (AML): Final envelope for '{tool_name_mcp}': Success={standard_envelope.get('success')}, ErrorType={standard_envelope.get('error_type')}")
        return standard_envelope
    

    async def _check_and_trigger_promotion(self, memory_id: str, *, workflow_id: Optional[str], context_id: Optional[str]):
        # workflow_id here is the snapshot
        current_agent_workflow_id = self.state.workflow_id
        if workflow_id != current_agent_workflow_id or self._shutdown_event.is_set():
            self.logger.debug(f"Skipping promo check for {_fmt_id(memory_id)}: WF changed or shutdown.")
            return

        promote_mem_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_PROMOTE_MEM)
        if not memory_id or not self._find_tool_server(promote_mem_mcp_name):
            self.logger.debug(f"Skipping promo check for {_fmt_id(memory_id)}: Invalid ID or tool for '{UMS_FUNC_PROMOTE_MEM}' unavailable.")
            return
        try:
            await asyncio.sleep(random.uniform(0.1, 0.4))
            if self._shutdown_event.is_set():
                return

            self.logger.debug(f"Checking promo potential for memory {_fmt_id(memory_id)} in WF {_fmt_id(workflow_id)}...")
            # Pass workflow_id to the UMS tool if it requires it (some UMS tools might infer from memory_id)
            promo_args = {"memory_id": memory_id}
            # if workflow_id: promo_args["workflow_id"] = workflow_id # promote_memory_level doesn't take workflow_id

            promo_res = await self._execute_tool_call_internal(promote_mem_mcp_name, promo_args, record_action=False)
            if promo_res.get("success"):
                if promo_res.get("promoted"):
                    self.logger.info(
                        f"⬆️ Memory {_fmt_id(memory_id)} promoted from {promo_res.get('previous_level')} to {promo_res.get('new_level')}."
                    )
                else:
                    self.logger.debug(f"Memory {_fmt_id(memory_id)} not promoted: {promo_res.get('reason')}")
            else:
                self.logger.warning(f"Promo check tool failed for {_fmt_id(memory_id)}: {promo_res.get('error')}")
        except Exception as e:
            self.logger.warning(f"Error in promo check task for {_fmt_id(memory_id)}: {e}", exc_info=False)


    async def _handle_workflow_and_goal_side_effects(self, base_tool_func_name: str, arguments: Dict, result_content_envelope: Dict):
        """
        Handles agent state changes triggered by specific UMS tool outcomes,
        particularly workflow creation/status changes and goal creation/status changes.
        This method consumes the standardized envelope from _execute_tool_call_internal.

        Args:
            base_tool_func_name: The canonical base function name (e.g., "create_workflow").
            arguments: The arguments passed to the original UMS tool call.
            result_content_envelope: The standardized dictionary envelope from _execute_tool_call_internal.
        """
        if not isinstance(result_content_envelope, dict):
            self.logger.error(
                f"AML_SIDE_EFFECTS CRITICAL: result_content_envelope is not a dict (type: {type(result_content_envelope)}) for tool '{base_tool_func_name}'. This indicates a severe internal error in _execute_tool_call_internal."
            )
            self.state.last_error_details = {
                "tool": self._get_ums_tool_mcp_name(base_tool_func_name) if base_tool_func_name else "UnknownToolDueToError",
                "error": "Internal error: Tool result processing failed to produce standard envelope.",
                "type": "AgentError",
                "details": {"raw_result_type": str(type(result_content_envelope))},
            }
            self.state.needs_replan = True
            return

        is_ums_tool_call_successful_in_envelope = result_content_envelope.get("success", False)
        ums_data_payload = result_content_envelope.get("data", {}) # Default to empty dict if 'data' is missing

        # Ensure ums_data_payload is a dict if the envelope claimed success and the tool is expected to return a dict.
        # For tools that might return non-dict success (e.g., summarize_text returns string), this check is nuanced.
        # For the critical state-changing tools here, we expect a dict.
        expected_dict_payload_tools = [
            UMS_FUNC_CREATE_WORKFLOW, UMS_FUNC_UPDATE_WORKFLOW_STATUS,
            UMS_FUNC_CREATE_GOAL, UMS_FUNC_UPDATE_GOAL_STATUS,
            # UMS_FUNC_RECORD_THOUGHT, UMS_FUNC_SAVE_COGNITIVE_STATE are also expected to return dicts
        ]

        if base_tool_func_name in expected_dict_payload_tools and not isinstance(ums_data_payload, dict):
            self.logger.warning(
                f"AML_SIDE_EFFECTS: UMS Tool '{base_tool_func_name}' envelope success={is_ums_tool_call_successful_in_envelope}, "
                f"but its 'data' payload is not a dict (type: {type(ums_data_payload)}). Using empty dict for safety. "
                f"Full Envelope Preview: {str(result_content_envelope)[:300]}"
            )
            if is_ums_tool_call_successful_in_envelope: # If envelope said success but data is wrong type for these tools
                self.logger.error(
                    f"AML_SIDE_EFFECTS: UMS Tool '{base_tool_func_name}' reported success in envelope, "
                    f"but its 'data' payload is malformed for critical state update. Cannot reliably process side effects."
                )
                self.state.last_error_details = {
                    "tool": self._get_ums_tool_mcp_name(base_tool_func_name),
                    "error": f"UMS tool {base_tool_func_name} success payload malformed (data not a dict).",
                    "type": "UMSError", 
                    "envelope_preview": str(result_content_envelope)[:200],
                }
                self.state.needs_replan = True
                return 
            ums_data_payload = {} # Force to empty dict for safety in subsequent .get() calls if envelope was already failure

        log_prefix = f"AML_SIDE_EFFECTS ({base_tool_func_name}, EnvelopeSuccess: {is_ums_tool_call_successful_in_envelope})"
        current_wf_id_before_effect = self.state.workflow_id
        current_goal_id_before_effect = self.state.current_goal_id
        needs_replan_before_side_effects = self.state.needs_replan

        self.logger.info(
            f"{log_prefix}: Entered. WF (before): {_fmt_id(current_wf_id_before_effect)}, "
            f"UMS Goal (before): {_fmt_id(current_goal_id_before_effect)}, "
            f"NeedsReplan (before this func): {needs_replan_before_side_effects}. "
            f"UMS Data Payload Preview (from envelope.data): {str(ums_data_payload)[:200]}"
        )

        # --- Side effects for UMS_FUNC_CREATE_WORKFLOW ---
        if base_tool_func_name == UMS_FUNC_CREATE_WORKFLOW:
            if is_ums_tool_call_successful_in_envelope and ums_data_payload.get("success"): # Both envelope and UMS payload success
                self.logger.info(f"{log_prefix}: UMS 'create_workflow' fully successful. Payload: {str(ums_data_payload)[:300]}")

                new_wf_id_from_ums = ums_data_payload.get("workflow_id")
                primary_chain_id_from_ums = ums_data_payload.get("primary_thought_chain_id")
                # UMS create_workflow returns the 'goal' text it was given, and its 'title'.
                # It also implicitly creates the primary cognitive_state with state_id == workflow_id.
                workflow_goal_text_from_ums_payload = ums_data_payload.get("goal")
                workflow_title_from_ums_payload = ums_data_payload.get("title")

                if not (new_wf_id_from_ums and isinstance(new_wf_id_from_ums, str)):
                    self.logger.error(f"{log_prefix}: CRITICAL - UMS create_workflow success payload missing valid 'workflow_id'. UMS Data: {ums_data_payload}")
                    self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_WORKFLOW), "error": "UMS create_workflow success payload invalid 'workflow_id'.", "type": "UMSError"}
                    self.state.needs_replan = True; return

                if not (primary_chain_id_from_ums and isinstance(primary_chain_id_from_ums, str)):
                    self.logger.error(f"{log_prefix}: CRITICAL - UMS create_workflow success payload missing valid 'primary_thought_chain_id'. UMS Data: {ums_data_payload}")
                    self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_WORKFLOW), "error": "UMS create_workflow success payload invalid 'primary_thought_chain_id'.", "type": "UMSError"}
                    self.state.needs_replan = True; return

                await asyncio.sleep(0.2) # Increased delay slightly for DB visibility
                if not await self._check_workflow_exists(new_wf_id_from_ums): # Robust check
                    self.logger.error(f"{log_prefix}: CRITICAL - UMS create_workflow success, but WF ID '{_fmt_id(new_wf_id_from_ums)}' NOT FOUND on re-check.")
                    self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_WORKFLOW), "error": f"UMS created WF '{_fmt_id(new_wf_id_from_ums)}' not found on re-check.", "type": "UMSError", "details":"DB_VISIBILITY_ISSUE_POST_CREATE_WF"}
                    self.state.needs_replan = True; return
                
                self.logger.info(f"{log_prefix}: Successfully created AND VERIFIED new workflow_id in UMS: {_fmt_id(new_wf_id_from_ums)}.")

                # --- Update Agent State with NEW Workflow Info ---
                self.state.workflow_id = new_wf_id_from_ums
                self.state.context_id = new_wf_id_from_ums # UMS create_workflow sets up cognitive_state with state_id = workflow_id
                self.state.current_thought_chain_id = primary_chain_id_from_ums

                parent_wf_id_arg_from_call = arguments.get("parent_workflow_id")
                is_sub_workflow = parent_wf_id_arg_from_call and parent_wf_id_arg_from_call == current_wf_id_before_effect
                if is_sub_workflow:
                    self.state.workflow_stack.append(new_wf_id_from_ums)
                else:
                    self.state.workflow_stack = [new_wf_id_from_ums]

                self.state.goal_stack = []
                self.state.current_goal_id = None
                self.state.needs_replan = False 
                self.state.last_error_details = None
                self.state.consecutive_error_count = 0
                self.logger.info(f"{log_prefix}: Agent state updated for new WF. WF ID={_fmt_id(self.state.workflow_id)}, Context ID={_fmt_id(self.state.context_id)}, Chain={_fmt_id(self.state.current_thought_chain_id)}.")

                # --- Create the Root UMS GOAL (in 'goals' table) for this new workflow ---
                initial_setup_for_goal_succeeded = True
                created_ums_goal_object_for_state: Optional[Dict[str, Any]] = None
                
                # Determine description & title for the root UMS goal
                root_goal_description_for_ums_table = (
                    workflow_goal_text_from_ums_payload 
                    if workflow_goal_text_from_ums_payload is not None 
                    else arguments.get("goal", f"Overall objectives for workflow {_fmt_id(self.state.workflow_id)}")
                )
                root_goal_title_for_ums_table = (
                    workflow_title_from_ums_payload # Use workflow title as goal title for root
                    if workflow_title_from_ums_payload is not None
                    else arguments.get("title", f"Primary Goal for WF-{self.state.workflow_id[:8]}")
                )

                create_goal_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL)
                if self._find_tool_server(create_goal_mcp_name):
                    goal_args_for_ums = {
                        "workflow_id": self.state.workflow_id,
                        "description": root_goal_description_for_ums_table,
                        "title": root_goal_title_for_ums_table,
                        "parent_goal_id": None, # This IS the root UMS goal
                        "initial_status": GoalStatus.ACTIVE.value
                    }
                    self.logger.info(f"{log_prefix}: Creating root UMS 'goals' table entry for WF '{_fmt_id(self.state.workflow_id)}'. Desc: '{root_goal_description_for_ums_table[:50]}...'")
                    
                    goal_creation_envelope = await self._execute_tool_call_internal(create_goal_mcp_name, goal_args_for_ums, record_action=False)
                    goal_creation_ums_payload = goal_creation_envelope.get("data", {})

                    if goal_creation_envelope.get("success") and isinstance(goal_creation_ums_payload, dict) and goal_creation_ums_payload.get("success"):
                        created_ums_goal_object_for_state = goal_creation_ums_payload.get("goal")
                        if isinstance(created_ums_goal_object_for_state, dict) and created_ums_goal_object_for_state.get("goal_id"):
                            self.state.goal_stack = [created_ums_goal_object_for_state] # Initialize stack with this root UMS goal
                            self.state.current_goal_id = created_ums_goal_object_for_state["goal_id"]
                            self.logger.info(f"{log_prefix}: Successfully created UMS root goal (in 'goals' table): {_fmt_id(self.state.current_goal_id)}.")
                        else:
                            self.logger.error(f"{log_prefix}: UMS create_goal for root returned success but malformed 'goal' object. Payload: {goal_creation_ums_payload}")
                            initial_setup_for_goal_succeeded = False
                            if not self.state.last_error_details: self.state.last_error_details = {"tool": create_goal_mcp_name, "error": "UMS create_goal (root) success payload malformed.", "type": "GoalManagementError"}
                    else: # UMS create_goal call failed
                        self.logger.error(f"{log_prefix}: Failed to create UMS root goal. Error: {goal_creation_envelope.get('error_message', 'Unknown error')}")
                        initial_setup_for_goal_succeeded = False
                        if not self.state.last_error_details: self.state.last_error_details = {"tool": create_goal_mcp_name, "error": goal_creation_envelope.get('error_message', "Failed UMS root goal creation."), "type": goal_creation_envelope.get('error_type', "GoalManagementError")}
                else: # UMS_FUNC_CREATE_GOAL tool not found
                    self.logger.error(f"{log_prefix}: Tool '{UMS_FUNC_CREATE_GOAL}' unavailable. Cannot create root UMS goal in 'goals' table.")
                    initial_setup_for_goal_succeeded = False
                    if not self.state.last_error_details: self.state.last_error_details = {"tool": create_goal_mcp_name, "error": "Tool unavailable", "type": "ToolUnavailable"}

                # Set Initial Plan
                if initial_setup_for_goal_succeeded and self.state.current_goal_id:
                    current_ums_goal_obj_desc = created_ums_goal_object_for_state.get("description", "Newly established root UMS goal") if created_ums_goal_object_for_state else "Newly established root UMS goal"
                    plan_step_desc = (
                        f"Initial assessment of root UMS goal: '{current_ums_goal_obj_desc[:70]}...' "
                        f"(UMS Goal ID: {_fmt_id(self.state.current_goal_id)}). My task is to understand this goal and outline the first concrete steps."
                    )
                    self.state.current_plan = [PlanStep(description=plan_step_desc, assigned_tool=None)]
                    self.state.needs_replan = False # LLM should execute this assessment step by recording thoughts
                    self.logger.info(f"{log_prefix}: Initial plan set: Assess new root UMS goal. NeedsReplan is False.")
                else: # Root UMS goal setup failed
                    plan_desc_error = (
                        f"ERROR: Failed to establish initial UMS root goal (in 'goals' table) for workflow '{_fmt_id(self.state.workflow_id)}'. "
                        f"Agent cannot proceed effectively. Review last error details and UMS state."
                    )
                    self.state.current_plan = [PlanStep(description=plan_desc_error, status="failed")]
                    self.state.needs_replan = True 
                    self.logger.error(f"{log_prefix}: Plan set to error state due to failed root UMS goal setup. Last error: {self.state.last_error_details}")

            elif not is_ums_tool_call_successful_in_envelope: # UMS create_workflow tool call itself FAILED (envelope error)
                self.logger.error(
                    f"{log_prefix}: UMS Tool '{UMS_FUNC_CREATE_WORKFLOW}' call failed as per standardized envelope. "
                    f"Error: {result_content_envelope.get('error_message', 'Unknown UMS error')}. Cannot proceed."
                )
                # self.state.last_error_details should have been set by _execute_tool_call_internal
                self.state.needs_replan = True 

            self.logger.info(
                f"{log_prefix}: END of create_workflow side effects. Current State: "
                f"WF ID='{_fmt_id(self.state.workflow_id)}', "
                f"Context ID='{_fmt_id(self.state.context_id)}', "
                f"UMS Goal ID='{_fmt_id(self.state.current_goal_id)}', "
                f"Plan Step 0 Desc='{self.state.current_plan[0].description[:50] if self.state.current_plan else 'N/A'}', "
                f"needs_replan={self.state.needs_replan}"
            )

        # --- Side effects for UMS_FUNC_CREATE_GOAL (when called by LLM directly) ---
        elif base_tool_func_name == UMS_FUNC_CREATE_GOAL:
            if is_ums_tool_call_successful_in_envelope and isinstance(ums_data_payload, dict) and ums_data_payload.get("success"):
                self.logger.info(
                    f"{log_prefix} (LLM invoked UMS_FUNC_CREATE_GOAL): START (UMS Success). Current UMS Goal (before): {_fmt_id(self.state.current_goal_id)}"
                )
                created_ums_goal_obj_from_llm_req = ums_data_payload.get("goal") 

                if isinstance(created_ums_goal_obj_from_llm_req, dict) and created_ums_goal_obj_from_llm_req.get("goal_id"):
                    if created_ums_goal_obj_from_llm_req.get("workflow_id") != self.state.workflow_id:
                        self.logger.error(
                            f"{log_prefix}: LLM created UMS goal {_fmt_id(created_ums_goal_obj_from_llm_req.get('goal_id'))} for incorrect workflow "
                            f"'{_fmt_id(created_ums_goal_obj_from_llm_req.get('workflow_id'))}' (Agent WF: '{_fmt_id(self.state.workflow_id)}'). Ignoring."
                        )
                        self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL), "error": "LLM created UMS goal for an incorrect workflow.", "type": "GoalManagementError"}
                        self.state.needs_replan = True
                    else: 
                        self.state.goal_stack.append(created_ums_goal_obj_from_llm_req)
                        self.state.current_goal_id = created_ums_goal_obj_from_llm_req["goal_id"]
                        self.logger.info(
                            f"📌 {log_prefix}: Pushed new UMS goal {_fmt_id(self.state.current_goal_id)} to local stack: '{created_ums_goal_obj_from_llm_req.get('description', '')[:50]}...'. Stack depth: {len(self.state.goal_stack)}"
                        )
                        
                        new_goal_desc_llm = created_ums_goal_obj_from_llm_req.get("description", f"Goal {self.state.current_goal_id[:8]}")
                        record_thought_mcp_name_llm = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_THOUGHT)
                        if self._find_tool_server(record_thought_mcp_name_llm):
                            thought_args = {"workflow_id": self.state.workflow_id, "content": f"Established new UMS sub-goal: {new_goal_desc_llm}", "thought_type": ThoughtType.GOAL.value, "thought_chain_id": self.state.current_thought_chain_id}
                            thought_res_env = await self._execute_tool_call_internal(record_thought_mcp_name_llm, thought_args, record_action=False)
                            if thought_res_env.get("success") and isinstance(thought_res_env.get("data"), dict) and thought_res_env["data"].get("success"):
                                self.logger.info(f"{log_prefix}: Recorded UMS thought {_fmt_id(thought_res_env['data'].get('thought_id'))} for LLM-created UMS goal.")
                            else:
                                self.logger.warning(f"{log_prefix}: Failed to record UMS thought for LLM-created UMS goal. Error: {thought_res_env.get('error_message')}")
                        else:
                            self.logger.warning(f"{log_prefix}: Tool '{UMS_FUNC_RECORD_THOUGHT}' unavailable for LLM-created UMS goal thought.")

                        self.state.needs_replan = True 
                        plan_desc = f"New UMS goal established by LLM: '{new_goal_desc_llm[:50]}...' ({_fmt_id(self.state.current_goal_id)}). Formulate plan for this sub-goal."
                        self.state.current_plan = [PlanStep(description=plan_desc)]
                        self.state.last_error_details = None 
                        self.state.consecutive_error_count = 0
                else: 
                    self.logger.warning(f"{log_prefix}: UMS Tool '{UMS_FUNC_CREATE_GOAL}' (LLM) success payload malformed. Data: {str(ums_data_payload)[:300]}")
                    self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL), "error": "UMS create_goal (LLM) success payload malformed.", "type": "GoalManagementError"}
                    self.state.needs_replan = True
            elif not is_ums_tool_call_successful_in_envelope:
                self.logger.error(f"{log_prefix}: UMS Tool '{UMS_FUNC_CREATE_GOAL}' (LLM) failed. Error: {result_content_envelope.get('error_message')}")
                self.state.needs_replan = True
            
            self.logger.info(f"{log_prefix} (LLM invoked UMS_FUNC_CREATE_GOAL): END. Current UMS Goal: {_fmt_id(self.state.current_goal_id)}, Stack Depth: {len(self.state.goal_stack)}, needs_replan={self.state.needs_replan}")

        # --- Side effects for UMS_FUNC_UPDATE_GOAL_STATUS ---
        elif base_tool_func_name == UMS_FUNC_UPDATE_GOAL_STATUS:
            if is_ums_tool_call_successful_in_envelope and isinstance(ums_data_payload, dict) and ums_data_payload.get("success"):
                self.logger.info(
                    f"{log_prefix}: START (UMS Success). UMS Goal marked in arguments: {_fmt_id(arguments.get('goal_id'))}, "
                    f"New Status in arguments: {arguments.get('status')}"
                )
                updated_goal_details_from_ums = ums_data_payload.get("updated_goal_details")
                parent_goal_id_of_updated_from_ums = ums_data_payload.get("parent_goal_id") # Parent of the goal that was updated
                is_root_finished_according_to_ums = ums_data_payload.get("is_root_finished", False)

                goal_id_that_was_updated_in_ums = updated_goal_details_from_ums.get("goal_id") if isinstance(updated_goal_details_from_ums, dict) else None
                new_status_str_from_ums = updated_goal_details_from_ums.get("status") if isinstance(updated_goal_details_from_ums, dict) else None

                if not goal_id_that_was_updated_in_ums or not new_status_str_from_ums:
                    self.logger.error(f"{log_prefix}: UMS update_goal_status success payload malformed. Data: {str(ums_data_payload)[:300]}")
                    self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_GOAL_STATUS), "error": "UMS update_goal_status success payload malformed.", "type": "GoalManagementError"}
                    self.state.needs_replan = True; return
                try:
                    new_status_enum_from_ums = GoalStatus(new_status_str_from_ums.lower())
                except ValueError:
                    self.logger.error(f"{log_prefix}: Invalid status '{new_status_str_from_ums}' in UMS response.")
                    self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_GOAL_STATUS), "error": f"UMS returned invalid goal status '{new_status_str_from_ums}'.", "type": "GoalManagementError"}
                    self.state.needs_replan = True; return

                # Update local goal stack view if the updated goal is in it
                if isinstance(updated_goal_details_from_ums, dict):
                    for i, local_goal_dict in enumerate(self.state.goal_stack):
                        if isinstance(local_goal_dict, dict) and local_goal_dict.get("goal_id") == goal_id_that_was_updated_in_ums:
                            self.state.goal_stack[i] = updated_goal_details_from_ums # Replace with UMS truth
                            self.logger.debug(f"{log_prefix}: Updated goal {_fmt_id(goal_id_that_was_updated_in_ums)} in local stack to status '{new_status_enum_from_ums.value}'.")
                            break
                
                is_terminal_status_update = new_status_enum_from_ums in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED]
                if goal_id_that_was_updated_in_ums == self.state.current_goal_id and is_terminal_status_update:
                    self.logger.info(f"{log_prefix}: Current UMS operational goal {_fmt_id(self.state.current_goal_id)} reached terminal state '{new_status_enum_from_ums.value}'.")
                    
                    # Pop current goal and shift focus to its parent (which might be None)
                    if self.state.goal_stack and self.state.goal_stack[-1].get("goal_id") == self.state.current_goal_id:
                        self.state.goal_stack.pop() 
                    
                    self.state.current_goal_id = parent_goal_id_of_updated_from_ums 
                    # If current_goal_id is now None, it means we are back to the overall workflow goal.
                    # If it's not None, rebuild the stack view from this new current parent.
                    if self.state.current_goal_id:
                        self.state.goal_stack = await self._fetch_goal_stack_from_ums(self.state.current_goal_id)
                    else: # Popped to root of goal hierarchy (no parent_goal_id for the one just completed)
                        self.state.goal_stack = [] # Stack is empty if no current operational UMS goal

                    self.logger.info(f"{log_prefix}: Agent focus shifted. New current UMS goal: '{_fmt_id(self.state.current_goal_id) if self.state.current_goal_id else 'Overall Workflow Goal'}'. Local stack depth: {len(self.state.goal_stack)}")

                    if is_root_finished_according_to_ums: # UMS confirms this was the root goal of the workflow
                        self.logger.info(f"{log_prefix}: UMS indicated root UMS goal {_fmt_id(goal_id_that_was_updated_in_ums)} finished. Workflow presumed finished with status: {new_status_enum_from_ums.value}.")
                        self.state.goal_achieved_flag = (new_status_enum_from_ums == GoalStatus.COMPLETED) # This flag stops the main agent loop if true
                        
                        update_wf_status_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_WORKFLOW_STATUS)
                        if self.state.workflow_id and self._find_tool_server(update_wf_status_mcp_name):
                            final_wf_status_str = WorkflowStatus.COMPLETED.value if self.state.goal_achieved_flag else WorkflowStatus.FAILED.value
                            # This internal call will have its own side effects handled by this same function.
                            await self._execute_tool_call_internal(update_wf_status_mcp_name, {"workflow_id": self.state.workflow_id, "status": final_wf_status_str, "completion_message": f"Root UMS goal {_fmt_id(goal_id_that_was_updated_in_ums)} marked '{new_status_enum_from_ums.value}'. Workflow concluded."}, record_action=False)
                        
                        # Set plan to finalization, no more replanning needed from this state.
                        self.state.current_plan = [PlanStep(description="Overall workflow goal achieved or failed. Finalizing.")]
                        self.state.needs_replan = False
                    elif self.state.current_goal_id: # Switched to a parent UMS goal
                        self.state.needs_replan = True
                        current_parent_goal_obj = next((g for g in self.state.goal_stack if isinstance(g, dict) and g.get("goal_id") == self.state.current_goal_id), None)
                        current_parent_goal_desc = current_parent_goal_obj.get("description", "Unknown Parent UMS Goal") if current_parent_goal_obj else "Unknown Parent UMS Goal"
                        plan_desc = f"Returned from UMS sub-goal {_fmt_id(goal_id_that_was_updated_in_ums)} (status: {new_status_enum_from_ums.value}). Re-assess current UMS parent goal: '{current_parent_goal_desc[:50]}...' ({_fmt_id(self.state.current_goal_id)})."
                        self.state.current_plan = [PlanStep(description=plan_desc)]
                    else: # No parent_goal_id, but UMS didn't flag as root_finished (should be rare if logic is correct)
                        self.logger.warning(f"{log_prefix}: Completed UMS goal {_fmt_id(goal_id_that_was_updated_in_ums)} with no parent. UMS did not flag as 'root_finished'. Re-evaluating overall workflow goal for safety.")
                        self.state.goal_achieved_flag = (new_status_enum_from_ums == GoalStatus.COMPLETED) # Tentative, may be overridden by workflow status update
                        self.state.needs_replan = True
                        self.state.current_plan = [PlanStep(description=f"Completed UMS goal {_fmt_id(goal_id_that_was_updated_in_ums)}. Re-evaluating overall workflow objectives.")]
                    
                    self.state.last_error_details = None; self.state.consecutive_error_count = 0
                else: # A non-current goal was updated, or current goal was updated to non-terminal status
                    self.logger.info(f"{log_prefix}: UMS Goal '{_fmt_id(goal_id_that_was_updated_in_ums)}' updated to '{new_status_enum_from_ums.value}'. Agent focus on '{_fmt_id(self.state.current_goal_id)}' (if set) remains.")
            
            elif not is_ums_tool_call_successful_in_envelope:
                self.logger.error(f"{log_prefix}: UMS Tool '{UMS_FUNC_UPDATE_GOAL_STATUS}' failed. Error: {result_content_envelope.get('error_message')}")
                self.state.needs_replan = True
                
            self.logger.info(f"{log_prefix}: END. Current UMS Goal: {_fmt_id(self.state.current_goal_id)}, Stack Depth: {len(self.state.goal_stack)}, needs_replan={self.state.needs_replan}, achieved_flag={self.state.goal_achieved_flag}")

        # --- Side effects for UMS_FUNC_UPDATE_WORKFLOW_STATUS ---
        elif base_tool_func_name == UMS_FUNC_UPDATE_WORKFLOW_STATUS:
            if is_ums_tool_call_successful_in_envelope and isinstance(ums_data_payload, dict) and ums_data_payload.get("success"):
                self.logger.info(f"{log_prefix}: START (UMS Success). WF ID from UMS payload: {_fmt_id(ums_data_payload.get('workflow_id'))}, New Status from UMS payload: {ums_data_payload.get('status')}")
                new_status_str_from_ums_wf = ums_data_payload.get("status")
                wf_id_updated_in_ums_wf = ums_data_payload.get("workflow_id")

                if not wf_id_updated_in_ums_wf or not new_status_str_from_ums_wf:
                    self.logger.error(f"{log_prefix}: UMS update_workflow_status success payload malformed. Data: {str(ums_data_payload)[:300]}")
                    self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_WORKFLOW_STATUS), "error": "UMS update_workflow_status success payload malformed.", "type": "UMSError"}
                    self.state.needs_replan = True; return
                try:
                    new_status_enum_from_ums_wf = WorkflowStatus(new_status_str_from_ums_wf.lower())
                except ValueError:
                    self.logger.error(f"{log_prefix}: Invalid workflow status '{new_status_str_from_ums_wf}' in UMS payload.")
                    self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_WORKFLOW_STATUS), "error": f"UMS returned invalid workflow status '{new_status_str_from_ums_wf}'.", "type": "UMSError"}
                    self.state.needs_replan = True; return

                is_terminal_wf_status_update = new_status_enum_from_ums_wf in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.ABANDONED]
                
                # Check if the updated workflow is the one on top of the agent's stack
                if self.state.workflow_stack and wf_id_updated_in_ums_wf == self.state.workflow_stack[-1]:
                    if is_terminal_wf_status_update:
                        finished_wf_id = self.state.workflow_stack.pop()
                        parent_wf_id_after_pop = self.state.workflow_stack[-1] if self.state.workflow_stack else None
                        
                        if parent_wf_id_after_pop: 
                            self.state.workflow_id = parent_wf_id_after_pop
                            self.state.context_id = self.state.workflow_id # Reset context to parent
                            await self._set_default_thought_chain_id() # For the parent
                            self.state.goal_stack = [] # Reset local goal stack view for parent
                            self.state.current_goal_id = None # Parent workflow starts without a specific UMS goal focus
                            self.logger.info(f"{log_prefix}: Sub-workflow '{_fmt_id(finished_wf_id)}' ({new_status_enum_from_ums_wf.value}) finished. Returning to parent '{_fmt_id(parent_wf_id_after_pop)}'.")
                            self.state.needs_replan = True # Must replan for parent context
                            self.state.current_plan = [PlanStep(description=f"Resumed parent workflow '{_fmt_id(self.state.workflow_id)}' after sub-workflow '{_fmt_id(finished_wf_id)}' ({new_status_enum_from_ums_wf.value}). Establish root UMS goal for parent or re-assess context.")]
                        else: # Root workflow (or last on stack) finished
                            self.logger.info(f"{log_prefix}: Root/Final workflow '{_fmt_id(finished_wf_id)}' finished ({new_status_enum_from_ums_wf.value}). No parent. Agent run concluding.")
                            self.state.workflow_id = None # This will stop the main agent loop
                            self.state.context_id = None; self.state.current_thought_chain_id = None
                            self.state.current_plan = []; self.state.goal_stack = []; self.state.current_goal_id = None
                            self.state.goal_achieved_flag = (new_status_enum_from_ums_wf == WorkflowStatus.COMPLETED)
                            self.state.needs_replan = False # Final state
                        
                        self.state.last_error_details = None 
                        self.state.consecutive_error_count = 0
                    else: # Non-terminal status update for current workflow (e.g., paused)
                        self.logger.info(f"{log_prefix}: Current workflow '{_fmt_id(wf_id_updated_in_ums_wf)}' status updated to '{new_status_enum_from_ums_wf.value}'. Agent focus remains.")
                        if new_status_enum_from_ums_wf == WorkflowStatus.PAUSED and not self.state.needs_replan:
                            self.state.needs_replan = True
                            self.state.current_plan = [PlanStep(description=f"Workflow '{_fmt_id(wf_id_updated_in_ums_wf)}' PAUSED. Await resume or new directive.")]
                
                elif wf_id_updated_in_ums_wf == self.state.workflow_id: # Current primary workflow was updated (and stack might be empty or mismatched)
                    self.logger.info(f"{log_prefix}: Current primary workflow '{_fmt_id(wf_id_updated_in_ums_wf)}' status changed to '{new_status_enum_from_ums_wf.value}'.")
                    if is_terminal_wf_status_update:
                        self.logger.info(f"{log_prefix}: Current primary workflow '{_fmt_id(wf_id_updated_in_ums_wf)}' reached terminal state. Agent run concluding.")
                        self.state.workflow_id = None
                        self.state.context_id = None; self.state.current_thought_chain_id = None
                        self.state.current_plan = []; self.state.goal_stack = []; self.state.current_goal_id = None
                        self.state.goal_achieved_flag = (new_status_enum_from_ums_wf == WorkflowStatus.COMPLETED)
                        self.state.needs_replan = False
                else: 
                    self.logger.info(f"{log_prefix}: UMS Workflow '{_fmt_id(wf_id_updated_in_ums_wf)}' status changed to '{new_status_enum_from_ums_wf.value}'. This was not agent's current focus. No change to agent's primary focus.")
            
            elif not is_ums_tool_call_successful_in_envelope:
                self.logger.error(f"{log_prefix}: UMS Tool '{UMS_FUNC_UPDATE_WORKFLOW_STATUS}' failed. Error: {result_content_envelope.get('error_message')}")
                self.state.needs_replan = True
                
            self.logger.info(f"{log_prefix}: END. WF: {_fmt_id(self.state.workflow_id)}, Current UMS Goal: {_fmt_id(self.state.current_goal_id)}, goal_achieved_flag: {self.state.goal_achieved_flag}, needs_replan: {self.state.needs_replan}")

        # --- No changes needed for UMS_FUNC_SAVE_COGNITIVE_STATE side effects in this function ---
        # save_cognitive_state is called internally for checkpoints; its direct success/failure
        # doesn't usually alter the agent's primary workflow/goal state in the same way
        # as create_workflow or goal updates. Its results are primarily for UMS record-keeping.
        # If its failure is critical, _execute_tool_call_internal would set last_error_details.

        # --- General Logging for State Changes After ALL Side Effects ---
        if not needs_replan_before_side_effects and self.state.needs_replan:
            self.logger.info(f"{log_prefix}: `needs_replan` became True due to overall side effects of '{base_tool_func_name}'.")
        elif needs_replan_before_side_effects and not self.state.needs_replan:
            self.logger.info(f"{log_prefix}: `needs_replan` became False due to overall side effects of '{base_tool_func_name}'.")

        new_current_wf_id_after_effect = self.state.workflow_id
        if current_wf_id_before_effect != new_current_wf_id_after_effect or current_goal_id_before_effect != self.state.current_goal_id:
            self.logger.info(
                f"AML_SIDE_EFFECTS Summary (Overall State Change by '{base_tool_func_name}'): "
                f"WF: {_fmt_id(current_wf_id_before_effect)} -> {_fmt_id(new_current_wf_id_after_effect)}, "
                f"ContextID: {_fmt_id(self.state.context_id)}, "
                f"UMS Goal: {_fmt_id(current_goal_id_before_effect)} -> {_fmt_id(self.state.current_goal_id)}, "
                f"Local Goal Stack Depth: {len(self.state.goal_stack)}"
            )
        else:
            self.logger.info(
                f"AML_SIDE_EFFECTS Summary (Agent WF/Goal/Context focus NOT significantly changed overall by tool '{base_tool_func_name}')"
            )

        self.logger.info(
            f"{log_prefix}: Exiting. Final agent state after side effects: WF ID='{_fmt_id(self.state.workflow_id)}', "
            f"Context ID='{_fmt_id(self.state.context_id)}', UMS Goal ID='{_fmt_id(self.state.current_goal_id)}', needs_replan={self.state.needs_replan}"
        )


    async def _fetch_goal_stack_from_ums(self, leaf_goal_id: Optional[str]) -> List[Dict[str, Any]]:
        if not leaf_goal_id:
            self.logger.debug("_fetch_goal_stack_from_ums: No leaf_goal_id provided, returning empty stack.")
            return []

        get_goal_details_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_GOAL_DETAILS)
        if not self._find_tool_server(get_goal_details_mcp_name):
            self.logger.warning(f"Tool for '{UMS_FUNC_GET_GOAL_DETAILS}' unavailable. Cannot reconstruct goal stack from UMS.")
            return []

        reconstructed_stack: List[Dict[str, Any]] = []
        current_id_to_fetch: Optional[str] = leaf_goal_id
        fetch_depth = 0
        max_fetch_depth = CONTEXT_GOAL_DETAILS_FETCH_LIMIT

        self.logger.debug(f"Reconstructing goal stack from UMS, starting with leaf: {_fmt_id(leaf_goal_id)}")

        while current_id_to_fetch and fetch_depth < max_fetch_depth:
            try:
                res = await self._execute_tool_call_internal(
                    get_goal_details_mcp_name,
                    {"goal_id": current_id_to_fetch},
                    record_action=False,
                )
                goal_data = res.get("goal") if isinstance(res, dict) and res.get("success") else None

                if isinstance(goal_data, dict):
                    reconstructed_stack.append(goal_data)
                    parent_id = goal_data.get("parent_goal_id")
                    self.logger.debug(f"Fetched goal {_fmt_id(current_id_to_fetch)}. Parent: {_fmt_id(parent_id)}. Depth: {fetch_depth}")
                    current_id_to_fetch = parent_id
                    fetch_depth += 1
                else:
                    self.logger.warning(
                        f"Failed to fetch details for goal {_fmt_id(current_id_to_fetch)} "
                        f"or invalid data received from UMS. Stopping stack reconstruction. Response: {res}"
                    )
                    break
            except Exception as e:
                self.logger.error(f"Exception while fetching goal {_fmt_id(current_id_to_fetch)} for stack reconstruction: {e}", exc_info=True)
                break

        reconstructed_stack.reverse()
        self.logger.info(
            f"Reconstructed goal stack from UMS: {[_fmt_id(g.get('goal_id')) for g in reconstructed_stack]} (Depth: {len(reconstructed_stack)})"
        )
        return reconstructed_stack

    async def _apply_heuristic_plan_update(
        self, last_llm_decision_from_mcpc: Dict[str, Any], last_tool_result_envelope: Optional[Dict[str, Any]] = None
    ):
        """
        Applies a heuristic update to the agent's plan based on the last action's outcome.
        This conservative version primarily focuses on marking steps complete/failed
        and advancing the plan, setting needs_replan=True on failures or if the
        plan becomes empty/too generic after an assessment.

        Args:
            last_llm_decision_from_mcpc: The decision object from MCPClient.process_agent_llm_turn.
                                         Keys: "decision", "tool_name" (original MCP name), "arguments", "content", etc.
            last_tool_result_envelope: The standardized envelope returned by _execute_tool_call_internal
                                       for the primary action of the turn.
                                       Structure: {"success": bool, "data": ..., "error_type": ..., ...}
        """
        self.logger.info("📋 Applying heuristic plan update (conservative)...")
        needs_replan_on_entry = self.state.needs_replan

        if not self.state.current_plan:
            self.logger.error("📋 HEURISTIC CRITICAL: Current plan is empty. Forcing replan.")
            self.state.current_plan = [PlanStep(description="CRITICAL FALLBACK: Plan was empty. Re-evaluate.")]
            self.state.needs_replan = True
            if not self.state.last_error_details:
                self.state.last_error_details = {"error": "Plan empty.", "type": "PlanManagementError"}
            self.state.consecutive_error_count += 1
            if self.state.successful_actions_since_reflection > 0:
                self.state.successful_actions_since_reflection = 0
            if self.state.successful_actions_since_consolidation > 0:
                self.state.successful_actions_since_consolidation = 0
            return

        current_step_obj = self.state.current_plan[0]
        original_step_description_lc = current_step_obj.description.lower()

        decision_type = last_llm_decision_from_mcpc.get("decision")
        tool_name_involved_in_llm_decision_mcp = last_llm_decision_from_mcpc.get("tool_name")

        action_was_successful_this_turn = False

        self.logger.debug(
            f"📋 Heuristic: Decision='{decision_type}', Tool='{tool_name_involved_in_llm_decision_mcp}', "
            f"StepID='{current_step_obj.id}', StepDesc='{current_step_obj.description[:50]}...', "
            f"AssignedTool='{current_step_obj.assigned_tool}', NeedsReplanOnEntry='{needs_replan_on_entry}'"
        )

        # current_step_was_agent_update_plan_tool_call_by_llm was unused, removed for now.
        # If needed later, can be re-added:
        # current_step_was_agent_update_plan_tool_call_by_llm = (
        #     decision_type == "call_tool" and
        #     tool_name_involved_in_llm_decision_mcp == AGENT_TOOL_UPDATE_PLAN and
        #     current_step_obj.assigned_tool and
        #     current_step_obj.assigned_tool == AGENT_TOOL_UPDATE_PLAN
        # )

        current_step_was_attempted_tool_execution = (
            decision_type == "tool_executed_by_mcp"
            and current_step_obj.assigned_tool
            and current_step_obj.assigned_tool == tool_name_involved_in_llm_decision_mcp
        )

        current_step_was_generic_thinking_intent_by_llm = (
            decision_type == "thought_process"
            and not current_step_obj.assigned_tool
            and tool_name_involved_in_llm_decision_mcp == self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_THOUGHT)
        )

        # --- Scenario 1: UMS/Server tool (current plan step's assigned_tool) EXECUTED BY MCPCLIENT ---
        if current_step_was_attempted_tool_execution:
            self.logger.debug(f"📋 Heuristic: Scenario 1 - UMS/Server tool '{tool_name_involved_in_llm_decision_mcp}' executed by MCPClient.")
            tool_call_succeeded = isinstance(last_tool_result_envelope, dict) and last_tool_result_envelope.get("success", False)
            action_was_successful_this_turn = tool_call_succeeded

            if tool_call_succeeded:
                current_step_obj.status = ActionStatus.COMPLETED.value
                # ... (summary generation logic as before) ...
                summary = "Success."
                ums_data_for_summary = last_tool_result_envelope.get("data", {}) if isinstance(last_tool_result_envelope, dict) else {}
                if isinstance(ums_data_for_summary, dict):
                    summary_keys = [
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
                    ]
                    for k_sum_h in summary_keys:
                        if k_sum_h in ums_data_for_summary and ums_data_for_summary[k_sum_h] is not None:
                            summary_value_str = str(ums_data_for_summary[k_sum_h])
                            summary = f"{k_sum_h}: {_fmt_id(summary_value_str) if 'id' in k_sum_h.lower() else summary_value_str}"
                            break
                    else:
                        generic_summary_parts = [
                            f"{k_s_h}={_fmt_id(str(v_s_h)) if 'id' in k_s_h.lower() else str(v_s_h)[:20]}"
                            for k_s_h, v_s_h in ums_data_for_summary.items()
                            if v_s_h is not None and k_s_h not in ["success", "processing_time"]
                        ]
                        if generic_summary_parts:
                            summary = f"Success. Data: {', '.join(generic_summary_parts)}"
                        else:
                            summary = "Success (No specific summary data from UMS tool)."
                current_step_obj.result_summary = summary[:150]
                self.state.current_plan.pop(0)
                if not self.state.needs_replan:
                    if not self.state.current_plan:
                        self.state.current_plan.append(
                            PlanStep(description="Plan finished. Analyze overall result and decide if overall UMS workflow goal is met.")
                        )
                else:
                    if not self.state.current_plan:
                        self.state.current_plan.append(
                            PlanStep(description="Plan cleared after successful UMS tool step due to side-effect requiring replan. Re-evaluate.")
                        )
                if self.state.last_error_details and self.state.last_error_details.get("tool") == tool_name_involved_in_llm_decision_mcp:
                    self.state.last_error_details = None
                self.logger.debug(f"📋 Heuristic: Step COMPLETED (UMS Tool). NeedsReplan={self.state.needs_replan}")
            else:
                current_step_obj.status = ActionStatus.FAILED.value
                error_msg = (
                    f"Type: {last_tool_result_envelope.get('error_type', 'Unk')}, Msg: {last_tool_result_envelope.get('error_message', 'Unk')}"
                    if isinstance(last_tool_result_envelope, dict)
                    else "Unknown tool failure."
                )
                current_step_obj.result_summary = f"Failure: {error_msg[:150]}"
                if not self.state.needs_replan:
                    self.state.needs_replan = True
                self.logger.warning(f"📋 Heuristic: Step FAILED (UMS Tool). NeedsReplan=True. Error: {error_msg}")

        # --- Scenario 2: Generic thinking step -> Thought Recorded ---
        elif current_step_was_generic_thinking_intent_by_llm:
            self.logger.debug(f"📋 Heuristic: Scenario 2 - LLM recorded thought for generic plan step.")
            thought_record_succeeded = isinstance(last_tool_result_envelope, dict) and last_tool_result_envelope.get("success", False)
            action_was_successful_this_turn = thought_record_succeeded
            original_thought_content_by_llm = last_llm_decision_from_mcpc.get("content", "")

            if thought_record_succeeded:
                current_step_obj.status = ActionStatus.COMPLETED.value
                ums_data = last_tool_result_envelope.get("data", {}) if isinstance(last_tool_result_envelope, dict) else {}
                thought_id = ums_data.get("thought_id", "UnkID") if isinstance(ums_data, dict) else "UnkID"
                current_step_obj.result_summary = f"Thought Recorded (ID: {_fmt_id(thought_id)}): {original_thought_content_by_llm[:50]}..."
                self.state.current_plan.pop(0)

                if last_llm_decision_from_mcpc.get("_mcp_client_force_replan_after_thought_"):
                    if not self.state.needs_replan:
                        self.state.needs_replan = True
                    if not self.state.current_plan:
                        self.state.current_plan.append(
                            PlanStep(description=f"Replan forced. Orig thought: {original_thought_content_by_llm[:60]}...")
                        )
                else:
                    if not self.state.needs_replan:
                        is_assessment_like = any(
                            kw in original_step_description_lc
                            for kw in [
                                "assess",
                                "evaluate",
                                "decide next",
                                "formulate initial plan",
                                "re-evaluate",
                                "analyze goal",
                                "understand task",
                                "initial step",
                                "root ums goal",
                                "determine initial research strategy",
                            ]
                        )
                        plan_empty_or_generic_next = not self.state.current_plan or (
                            self.state.current_plan
                            and any(
                                kw in self.state.current_plan[0].description.lower()
                                for kw in ["decide next", "re-evaluate", "formulate plan", "assess goal"]
                            )
                        )
                        if is_assessment_like and plan_empty_or_generic_next:
                            self.state.needs_replan = True
                            new_plan_desc = "Replan: Formulate detailed steps after assessment."
                            if not self.state.current_plan:
                                self.state.current_plan.append(PlanStep(description=new_plan_desc))
                            else:
                                self.state.current_plan[0].description = new_plan_desc
                                self.state.current_plan[0].status = "planned"
                                self.state.current_plan[0].assigned_tool = None
                                self.state.current_plan[0].depends_on = []
                        else:
                            self.state.needs_replan = False
                    if not self.state.current_plan and not self.state.needs_replan:
                        self.state.current_plan.append(PlanStep(description="Decide next action after thought."))
                self.logger.debug(f"📋 Heuristic: Step COMPLETED (Thought). NeedsReplan={self.state.needs_replan}")
            else:
                current_step_obj.status = ActionStatus.FAILED.value
                error_msg = (
                    str(last_tool_result_envelope.get("error_message", "Failed to record thought."))[:100]
                    if isinstance(last_tool_result_envelope, dict)
                    else "Failed thought recording."
                )
                current_step_obj.result_summary = f"Failed Thought: {error_msg}"
                if not self.state.needs_replan:
                    self.state.needs_replan = True
                self.logger.warning(f"📋 Heuristic: Step FAILED (Thought). NeedsReplan=True. Error: {error_msg}")

        # --- Scenario 3: LLM called AGENT_TOOL_UPDATE_PLAN ---
        elif decision_type == "call_tool" and tool_name_involved_in_llm_decision_mcp == AGENT_TOOL_UPDATE_PLAN:
            self.logger.debug(f"📋 Heuristic: Scenario 3 - LLM called AGENT_TOOL_UPDATE_PLAN.")
            plan_update_succeeded = isinstance(last_tool_result_envelope, dict) and last_tool_result_envelope.get("success", False)
            action_was_successful_this_turn = plan_update_succeeded

            if plan_update_succeeded:  # Plan already updated by _execute_tool_call_internal
                self.logger.info(f"📋 Heuristic: Plan successfully updated by '{AGENT_TOOL_UPDATE_PLAN}'.")
                if current_step_obj.assigned_tool == AGENT_TOOL_UPDATE_PLAN or "update plan" in original_step_description_lc:
                    current_step_obj.status = ActionStatus.COMPLETED.value
                    current_step_obj.result_summary = "Plan updated by LLM."
                    if self.state.current_plan and self.state.current_plan[0].id == current_step_obj.id:
                        self.state.current_plan.pop(0)
                if not self.state.current_plan:
                    self.state.current_plan.append(PlanStep(description="New plan active. Proceed."))
                # needs_replan should be False from successful _execute_tool_call_internal
            else:  # AGENT_TOOL_UPDATE_PLAN failed validation in _execute_tool_call_internal
                current_step_obj.status = ActionStatus.FAILED.value
                error_msg = (
                    f"Type: {last_tool_result_envelope.get('error_type', 'PlanUpdateErr')}, Msg: {last_tool_result_envelope.get('error_message', 'Unk')}"
                    if isinstance(last_tool_result_envelope, dict)
                    else "Plan update failed."
                )
                current_step_obj.result_summary = f"Failed plan update: {error_msg[:100]}"
                if not self.state.needs_replan:
                    self.state.needs_replan = True  # Error in plan update logic
                self.logger.warning(f"📋 Heuristic: Call to '{AGENT_TOOL_UPDATE_PLAN}' FAILED. Step FAILED. NeedsReplan=True. Error: {error_msg}")

        # --- Scenario 4: LLM signaled overall goal completion ---
        elif decision_type == "complete" or decision_type == "complete_with_artifact":
            self.logger.debug(f"📋 Heuristic: Scenario 4 - LLM signaled overall goal completion.")
            action_was_successful_this_turn = True
            self.state.current_plan = [PlanStep(description="Goal Achieved. Finalizing workflow.", status="completed")]
            self.state.needs_replan = False
            self.logger.info(f"📋 Heuristic: LLM signaled 'complete'. Plan set to finalization. NeedsReplan=False")

        # --- Scenario 5: LLM provided a textual plan update (parsed by MCPClient) ---
        elif decision_type == "plan_update":
            self.logger.debug(f"📋 Heuristic: Scenario 5 - LLM textual plan update processed by MCPClient.")
            if not self.state.needs_replan:  # Plan was successfully applied by MCPClient/execute_llm_decision
                action_was_successful_this_turn = True
                self.logger.info("📋 Heuristic: Plan successfully updated from LLM text (handled by MCPClient).")
                if "assess" in original_step_description_lc or "plan" in original_step_description_lc:
                    # Don't pop, self.state.current_plan is already the *new* plan.
                    # If we wanted to mark the *previous* step that led to this, it's harder without history here.
                    # For now, just acknowledge the new plan is active.
                    pass
            else:  # Textual plan update failed validation (needs_replan is True from MCPClient)
                action_was_successful_this_turn = False
                current_step_obj.status = ActionStatus.FAILED.value
                error_msg = (
                    f"LLM textual plan validation error: {self.state.last_error_details.get('error', 'Unk')}"
                    if self.state.last_error_details and self.state.last_error_details.get("type") == "PlanValidationError"
                    else "Failed to apply LLM textual plan."
                )
                current_step_obj.result_summary = f"Textual Plan Error: {error_msg[:100]}"
                self.logger.warning(f"📋 Heuristic: LLM textual plan NOT applied (needs_replan is True). Step FAILED. Error: {error_msg}")

        # --- Scenario 6: Other decision types or no direct action on current plan step ---
        else:
            self.logger.debug(
                f"📋 Heuristic: Scenario 6 - Decision type '{decision_type}' or tool '{tool_name_involved_in_llm_decision_mcp}' unmatched primary paths."
            )
            action_was_successful_this_turn = False
            if self.state.last_error_details and not self.state.needs_replan:
                self.logger.warning("📋 Heuristic: `last_error_details` set, `needs_replan` was false. Forcing replan due to error.")
                self.state.needs_replan = True
                current_step_obj.status = ActionStatus.FAILED.value
                current_step_obj.result_summary = f"Error state detected: {str(self.state.last_error_details.get('error'))[:50]}"

            if not self.state.current_plan and self.state.needs_replan:
                self.state.current_plan.append(PlanStep(description="Plan empty and replan needed. Re-evaluate."))
            elif not self.state.current_plan and not self.state.needs_replan:
                self.logger.warning("📋 Heuristic: Plan empty (Scenario 6) without needs_replan. Adding assessment step.")
                self.state.current_plan.append(PlanStep(description="Assess situation and decide next action."))

        # --- Update Meta-Cognitive Counters ---
        if action_was_successful_this_turn:
            self.state.consecutive_error_count = 0
            base_tool_name_for_counter = (
                self._get_base_function_name(tool_name_involved_in_llm_decision_mcp) if tool_name_involved_in_llm_decision_mcp else None
            )

            # Define conditions for substantive actions
            is_substantive_ums_tool_by_mcpc = (
                decision_type == "tool_executed_by_mcp"
                and tool_name_involved_in_llm_decision_mcp
                and base_tool_name_for_counter not in self._INTERNAL_OR_META_TOOLS_BASE_NAMES
                and tool_name_involved_in_llm_decision_mcp != AGENT_TOOL_UPDATE_PLAN
            )
            is_successful_thought = decision_type == "thought_process" and action_was_successful_this_turn

            if is_substantive_ums_tool_by_mcpc:
                self.state.successful_actions_since_reflection += 1.0
                self.state.successful_actions_since_consolidation += 1.0
            elif is_successful_thought:
                self.state.successful_actions_since_reflection += 0.5
                self.state.successful_actions_since_consolidation += 0.5
            # No increment for plan updates or goal completions for these counters
        else:
            if self.state.last_error_details:
                self.state.consecutive_error_count += 1
                self.logger.warning(
                    f"📋 Heuristic: Consecutive error count incremented to: {self.state.consecutive_error_count} due to active error."
                )

            if self.state.last_error_details or (self.state.needs_replan and not needs_replan_on_entry):
                if self.state.successful_actions_since_reflection > 0:
                    self.logger.info(f"📋 Heuristic: Reset reflection counter from {self.state.successful_actions_since_reflection:.1f}.")
                    self.state.successful_actions_since_reflection = 0
                if self.state.successful_actions_since_consolidation > 0:
                    self.logger.info(f"📋 Heuristic: Reset consolidation counter from {self.state.successful_actions_since_consolidation:.1f}.")
                    self.state.successful_actions_since_consolidation = 0

        log_plan_msg = f"Plan after heuristic. Steps: {len(self.state.current_plan)}. "
        if self.state.current_plan:
            next_step_log = self.state.current_plan[0]
            depends_str_log = f"Depends: {[_fmt_id(d) for d in next_step_log.depends_on]}" if next_step_log.depends_on else "Depends: None"
            log_plan_msg += (
                f"Next: '{next_step_log.description[:60]}...' (ID: {_fmt_id(next_step_log.id)}, Status: {next_step_log.status}, {depends_str_log})"
            )
        else:
            log_plan_msg += "Plan is CRITICALLY empty."
        self.logger.info(
            f"📋 Heuristic Update End: {log_plan_msg}. NeedsReplan={self.state.needs_replan}. ConsecutiveErrors={self.state.consecutive_error_count}"
        )

    def _adapt_thresholds(self, stats: Dict[str, Any]) -> None:
        if not stats or not stats.get("success"):
            self.logger.warning("Cannot adapt thresholds: Invalid stats.")
            return
        self.logger.debug(f"Adapting thresholds based on stats: {stats}")
        adjustment_dampening = THRESHOLD_ADAPTATION_DAMPENING
        changed = False
        episodic_count = stats.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0)
        total_memories = stats.get("total_memories", 1)
        episodic_ratio = episodic_count / total_memories if total_memories > 0 else 0
        target_episodic_ratio_upper = 0.30
        target_episodic_ratio_lower = 0.10
        mid_target_ratio = (target_episodic_ratio_upper + target_episodic_ratio_lower) / 2
        ratio_deviation = episodic_ratio - mid_target_ratio
        consolidation_adjustment = -math.ceil(ratio_deviation * self.state.current_consolidation_threshold * 2.0)
        dampened_adjustment = int(consolidation_adjustment * adjustment_dampening)
        if dampened_adjustment != 0:
            old_threshold = self.state.current_consolidation_threshold
            potential_new = max(MIN_CONSOLIDATION_THRESHOLD, min(MAX_CONSOLIDATION_THRESHOLD, old_threshold + dampened_adjustment))
            if potential_new != old_threshold:
                change_direction = "Lowering" if dampened_adjustment < 0 else "Raising"
                self.logger.info(
                    f"{change_direction} consolidation threshold: {old_threshold} -> {potential_new} (Episodic Ratio: {episodic_ratio:.1%}, Dev: {ratio_deviation:+.1%}, Adj: {dampened_adjustment})"
                )
                self.state.current_consolidation_threshold = potential_new
                changed = True
        total_calls = sum(v.get("success", 0) + v.get("failure", 0) for v in self.state.tool_usage_stats.values())
        total_failures = sum(v.get("failure", 0) for v in self.state.tool_usage_stats.values())
        min_calls_for_rate = 5
        failure_rate = (total_failures / total_calls) if total_calls >= min_calls_for_rate else 0.0
        target_failure_rate = 0.10
        failure_deviation = failure_rate - target_failure_rate
        reflection_adjustment = -math.ceil(failure_deviation * self.state.current_reflection_threshold * 3.0)
        is_stable_progress = (failure_rate < target_failure_rate * 0.5) and (self.state.consecutive_error_count == 0)
        momentum_bias = 0
        if is_stable_progress and reflection_adjustment >= 0:
            momentum_bias = math.ceil(reflection_adjustment * (MOMENTUM_THRESHOLD_BIAS_FACTOR - 1.0))
            reflection_adjustment += momentum_bias
            self.logger.debug(f"Mental Momentum: Adding +{momentum_bias} to reflection adj (Stable).")
        dampened_adjustment = int(reflection_adjustment * adjustment_dampening)
        if dampened_adjustment != 0 and total_calls >= min_calls_for_rate:
            old_threshold = self.state.current_reflection_threshold
            potential_new = max(MIN_REFLECTION_THRESHOLD, min(MAX_REFLECTION_THRESHOLD, old_threshold + dampened_adjustment))
            if potential_new != old_threshold:
                change_direction = "Lowering" if dampened_adjustment < 0 else "Raising"
                momentum_tag = " (+Momentum)" if is_stable_progress and momentum_bias > 0 else ""
                self.logger.info(
                    f"{change_direction} reflection threshold: {old_threshold} -> {potential_new} (Fail Rate: {failure_rate:.1%}, Dev: {failure_deviation:+.1%}, Adj: {dampened_adjustment}{momentum_tag})"
                )
                self.state.current_reflection_threshold = potential_new
                changed = True
        if not changed:
            self.logger.debug("No threshold adjustments triggered.")

    async def _run_periodic_tasks(self):
        # This method uses _get_ums_tool_mcp_name for UMS tool calls
        if not self.state.workflow_id or not self.state.context_id or self._shutdown_event.is_set():
            self.logger.debug("Skipping periodic tasks: No active workflow/context or shutdown signaled.")
            return

        # Store (full_mcp_tool_name, args_dict)
        tasks_to_run: List[Tuple[str, Dict[str, Any]]] = []
        trigger_reasons: List[str] = []

        log_prefix = f"AML_PERIODIC (WF:{_fmt_id(self.state.workflow_id)}, Loop:{self.state.current_loop})"
        self.logger.debug(f"{log_prefix}: Evaluating periodic tasks.")

        # Construct full MCP names for UMS tools
        ums_reflection_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_REFLECTION)
        ums_consolidation_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_CONSOLIDATION)
        ums_optimize_wm_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_OPTIMIZE_WM)
        ums_auto_focus_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_AUTO_FOCUS)
        ums_query_memories_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_QUERY_MEMORIES)
        ums_compute_stats_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_COMPUTE_STATS)
        ums_delete_expired_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_DELETE_EXPIRED_MEMORIES)

        reflection_tool_available = self._find_tool_server(ums_reflection_mcp_name) is not None
        consolidation_tool_available = self._find_tool_server(ums_consolidation_mcp_name) is not None
        optimize_wm_tool_available = self._find_tool_server(ums_optimize_wm_mcp_name) is not None
        auto_focus_tool_available = self._find_tool_server(ums_auto_focus_mcp_name) is not None
        promotion_query_tool_available = self._find_tool_server(ums_query_memories_mcp_name) is not None
        stats_tool_available = self._find_tool_server(ums_compute_stats_mcp_name) is not None
        maintenance_tool_available = self._find_tool_server(ums_delete_expired_mcp_name) is not None

        # 1. Statistics and Threshold Adaptation
        self.state.loops_since_stats_adaptation += 1
        if self.state.loops_since_stats_adaptation >= STATS_ADAPTATION_INTERVAL:
            if stats_tool_available:
                trigger_reasons.append("StatsInterval")
                try:
                    self.logger.debug(f"{log_prefix}: Triggering UMS compute_memory_statistics.")
                    stats_result = await self._execute_tool_call_internal(
                        ums_compute_stats_mcp_name, {"workflow_id": self.state.workflow_id}, record_action=False
                    )
                    if stats_result.get("success"):
                        self._adapt_thresholds(stats_result)
                        episodic_count = stats_result.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0)
                        # Check if consolidation is already scheduled to avoid duplication
                        if (
                            episodic_count > (self.state.current_consolidation_threshold * 2.0)
                            and consolidation_tool_available
                            and not any(task[0] == ums_consolidation_mcp_name for task in tasks_to_run)
                        ):
                            self.logger.info(f"{log_prefix}: High episodic count ({episodic_count}), scheduling consolidation.")
                            tasks_to_run.append(
                                (
                                    ums_consolidation_mcp_name,
                                    {
                                        "workflow_id": self.state.workflow_id,
                                        "consolidation_type": "summary",
                                        "query_filter": {"memory_level": MemoryLevel.EPISODIC.value},
                                        "max_source_memories": self.consolidation_max_sources,
                                    },
                                )
                            )
                            trigger_reasons.append(f"HighEpisodic({episodic_count})")
                            self.state.successful_actions_since_consolidation = 0  # Reset as it's being scheduled
                    else:
                        self.logger.warning(f"{log_prefix}: Failed to compute stats for adaptation: {stats_result.get('error')}")
                except Exception as e:
                    self.logger.error(f"{log_prefix}: Error during stats/adaptation: {e}", exc_info=True)  # Log full traceback for unexpected
                finally:
                    self.state.loops_since_stats_adaptation = 0
            else:
                self.logger.warning(f"{log_prefix}: Skipping stats/adaptation: Tool '{UMS_FUNC_COMPUTE_STATS}' not available.")
                self.state.loops_since_stats_adaptation = 0  # Still reset counter

        # 2. Reflection
        needs_reflection = self.state.needs_replan or (self.state.successful_actions_since_reflection >= self.state.current_reflection_threshold)
        if needs_reflection:
            if reflection_tool_available and not any(task[0] == ums_reflection_mcp_name for task in tasks_to_run):
                reflection_type = self.reflection_type_sequence[self.state.reflection_cycle_index % len(self.reflection_type_sequence)]
                tasks_to_run.append((ums_reflection_mcp_name, {"workflow_id": self.state.workflow_id, "reflection_type": reflection_type}))
                reason_str = (
                    f"Replan({self.state.needs_replan})"
                    if self.state.needs_replan
                    else f"SuccessCount({self.state.successful_actions_since_reflection:.1f}>={self.state.current_reflection_threshold})"
                )
                trigger_reasons.append(f"Reflect({reason_str})")
                self.state.successful_actions_since_reflection = 0  # Reset counter as it's being scheduled
                self.state.reflection_cycle_index += 1
            else:
                if not reflection_tool_available:
                    self.logger.warning(f"{log_prefix}: Skipping reflection: Tool '{UMS_FUNC_REFLECTION}' unavailable.")
                    self.state.successful_actions_since_reflection = 0  # Reset if tool missing, to avoid re-triggering immediately
                elif any(task[0] == ums_reflection_mcp_name for task in tasks_to_run):
                    self.logger.debug(f"{log_prefix}: Reflection already scheduled this cycle.")

        # 3. Consolidation
        needs_consolidation = self.state.successful_actions_since_consolidation >= self.state.current_consolidation_threshold
        if needs_consolidation:
            if consolidation_tool_available and not any(task[0] == ums_consolidation_mcp_name for task in tasks_to_run):
                tasks_to_run.append(
                    (
                        ums_consolidation_mcp_name,
                        {
                            "workflow_id": self.state.workflow_id,
                            "consolidation_type": "summary",
                            "query_filter": {"memory_level": self.consolidation_memory_level},  # Use configured level
                            "max_source_memories": self.consolidation_max_sources,
                        },
                    )
                )
                trigger_reasons.append(
                    f"ConsolidateThreshold({self.state.successful_actions_since_consolidation:.1f}>={self.state.current_consolidation_threshold})"
                )
                self.state.successful_actions_since_consolidation = 0  # Reset counter as it's being scheduled
            else:
                if not consolidation_tool_available:
                    self.logger.warning(f"{log_prefix}: Skipping consolidation: Tool '{UMS_FUNC_CONSOLIDATION}' unavailable.")
                    self.state.successful_actions_since_consolidation = 0  # Reset if tool missing
                elif any(task[0] == ums_consolidation_mcp_name for task in tasks_to_run):
                    self.logger.debug(f"{log_prefix}: Consolidation already scheduled this cycle.")

        # 4. Working Memory Optimization & Focus Update
        self.state.loops_since_optimization += 1
        if self.state.loops_since_optimization >= OPTIMIZATION_LOOP_INTERVAL:
            if optimize_wm_tool_available:
                # optimize_working_memory is agent-internal, its result is used to update cognitive_states.
                # It does not directly return a result to be used as meta-feedback for LLM.
                # The task here is to *calculate* the optimization. Applying it happens in main loop if needed.
                # For simplicity, let's assume the UMS tool does the calculation.
                # If the tool *also* applies it, then no further action here.
                # If it *only* calculates, Agent might need another step to apply.
                # Current UMS `optimize_working_memory` returns lists to retain/remove. Agent applies it.
                # So, this is more like an "initiate calculation" step.
                # For now, let's assume the UMS tool might update state or the agent will handle its output.
                # It seems `optimize_working_memory` itself logs its actions.
                tasks_to_run.append((ums_optimize_wm_mcp_name, {"context_id": self.state.context_id}))
                trigger_reasons.append("OptimizeInterval")
            else:
                self.logger.warning(f"{log_prefix}: Skipping WM optimization: Tool '{UMS_FUNC_OPTIMIZE_WM}' unavailable.")

            if auto_focus_tool_available:
                tasks_to_run.append((ums_auto_focus_mcp_name, {"context_id": self.state.context_id}))
                trigger_reasons.append("FocusUpdateInterval")
            else:
                self.logger.warning(f"{log_prefix}: Skipping auto-focus: Tool '{UMS_FUNC_AUTO_FOCUS}' unavailable.")
            self.state.loops_since_optimization = 0

        # 5. Memory Promotion Check
        self.state.loops_since_promotion_check += 1
        if self.state.loops_since_promotion_check >= MEMORY_PROMOTION_LOOP_INTERVAL:
            if promotion_query_tool_available:  # query_memories is needed to find candidates
                # This internal task will then call _check_and_trigger_promotion for each candidate
                tasks_to_run.append(("CHECK_PROMOTIONS_INTERNAL_TASK", {"workflow_id": self.state.workflow_id}))
                trigger_reasons.append("PromotionInterval")
            else:
                self.logger.warning(f"{log_prefix}: Skipping promotion check: Tool '{UMS_FUNC_QUERY_MEMORIES}' (for candidate search) unavailable.")
            self.state.loops_since_promotion_check = 0

        # 6. Maintenance (Delete Expired Memories)
        self.state.loops_since_maintenance += 1
        if self.state.loops_since_maintenance >= MAINTENANCE_INTERVAL:
            if maintenance_tool_available:
                tasks_to_run.append((ums_delete_expired_mcp_name, {"db_path": None}))  # db_path will be handled by UMS tool
                trigger_reasons.append("MaintenanceInterval")
                self.state.loops_since_maintenance = 0
            else:
                self.logger.warning(f"{log_prefix}: Skipping maintenance: Tool '{UMS_FUNC_DELETE_EXPIRED_MEMORIES}' unavailable.")
                self.state.loops_since_maintenance = 0  # Reset counter even if tool is missing

        # Execute scheduled tasks
        if tasks_to_run:
            unique_reasons_str = ", ".join(sorted(set(trigger_reasons)))  # Ensure unique reasons
            self.logger.info(f"🧠 {log_prefix}: Running {len(tasks_to_run)} periodic tasks (Triggers: {unique_reasons_str}).")

            # Sort for consistent execution order (e.g., maintenance first, then stats, then cognitive)
            def sort_key_periodic(task_tuple: Tuple[str, Dict[str, Any]]) -> int:
                tool_mcp_name = task_tuple[0]
                if tool_mcp_name == ums_delete_expired_mcp_name:
                    return 0
                if tool_mcp_name == ums_compute_stats_mcp_name:
                    return 1
                if tool_mcp_name == "CHECK_PROMOTIONS_INTERNAL_TASK":
                    return 2
                if tool_mcp_name == ums_optimize_wm_mcp_name:
                    return 3
                if tool_mcp_name == ums_auto_focus_mcp_name:
                    return 4
                if tool_mcp_name == ums_consolidation_mcp_name:
                    return 5
                if tool_mcp_name == ums_reflection_mcp_name:
                    return 6
                return 7  # Default for any other

            tasks_to_run.sort(key=sort_key_periodic)

            for mcp_tool_name_to_call, args_for_tool in tasks_to_run:
                if self._shutdown_event.is_set():
                    self.logger.info(f"{log_prefix}: Shutdown signaled during periodic tasks execution. Stopping further tasks.")
                    break
                try:
                    if mcp_tool_name_to_call == "CHECK_PROMOTIONS_INTERNAL_TASK":
                        self.logger.debug(
                            f"{log_prefix}: Triggering internal promotion checks based on workflow_id '{args_for_tool.get('workflow_id')}'..."
                        )
                        await self._trigger_promotion_checks()  # This internally uses self.state.workflow_id
                        continue  # Skip normal tool call processing

                    self.logger.debug(f"{log_prefix}: Executing periodic MCP Tool: {mcp_tool_name_to_call} with args: {args_for_tool}")

                    # Ensure workflow_id is in args if not already and tool is not create_workflow/list_workflows
                    if (
                        "workflow_id" not in args_for_tool
                        and self.state.workflow_id
                        and self._get_base_function_name(mcp_tool_name_to_call)
                        not in [UMS_FUNC_CREATE_WORKFLOW, UMS_FUNC_LIST_WORKFLOWS, UMS_FUNC_DELETE_EXPIRED_MEMORIES]
                    ):
                        args_for_tool["workflow_id"] = self.state.workflow_id
                    if (
                        "context_id" not in args_for_tool
                        and self.state.context_id
                        and self._get_base_function_name(mcp_tool_name_to_call) in [UMS_FUNC_OPTIMIZE_WM, UMS_FUNC_AUTO_FOCUS]
                    ):
                        args_for_tool["context_id"] = self.state.context_id

                    result_content_periodic = await self._execute_tool_call_internal(mcp_tool_name_to_call, args_for_tool, record_action=False)

                    base_func_name_periodic = self._get_base_function_name(mcp_tool_name_to_call)

                    if not result_content_periodic.get("success"):
                        self.logger.warning(
                            f"{log_prefix}: Periodic task UMS tool '{mcp_tool_name_to_call}' call failed: {result_content_periodic.get('error')}"
                        )
                        # If a critical periodic task (like optimize_wm or auto_focus) fails, it might warrant setting last_error_details
                        if base_func_name_periodic in [UMS_FUNC_OPTIMIZE_WM, UMS_FUNC_AUTO_FOCUS]:
                            if not self.state.last_error_details:  # Don't overwrite existing primary error
                                self.state.last_error_details = {
                                    "tool": mcp_tool_name_to_call,
                                    "error": f"Periodic {base_func_name_periodic} failed: {result_content_periodic.get('error')}",
                                    "type": "PeriodicTaskToolError",
                                }
                        continue  # Move to next periodic task

                    # Process successful periodic task result for reflection/consolidation
                    if base_func_name_periodic in [UMS_FUNC_REFLECTION, UMS_FUNC_CONSOLIDATION]:
                        feedback = ""
                        if base_func_name_periodic == UMS_FUNC_REFLECTION:
                            feedback = result_content_periodic.get("content", "")
                        elif base_func_name_periodic == UMS_FUNC_CONSOLIDATION:
                            feedback = result_content_periodic.get("consolidated_content", "")

                        if feedback:
                            feedback_summary = str(feedback).split("\n", 1)[0][:150]  # Get first line
                            self.state.last_meta_feedback = f"Feedback from UMS {base_func_name_periodic}: {feedback_summary}..."
                            self.logger.info(
                                f"{log_prefix}: Meta-feedback received from UMS {base_func_name_periodic}: {self.state.last_meta_feedback}"
                            )
                            self.state.needs_replan = True  # Meta-cognition often leads to replanning
                        else:
                            self.logger.debug(
                                f"{log_prefix}: Periodic UMS task {mcp_tool_name_to_call} succeeded but returned no feedback content. Result: {str(result_content_periodic)[:200]}"
                            )

                except Exception as e_periodic_task_exec:
                    self.logger.error(
                        f"{log_prefix}: Exception during execution of periodic task '{mcp_tool_name_to_call}': {e_periodic_task_exec}", exc_info=True
                    )
                    # Optionally set last_error_details if a periodic task itself has an unhandled exception
                    if not self.state.last_error_details:
                        self.state.last_error_details = {
                            "tool": mcp_tool_name_to_call,
                            "error": f"Unhandled exception in periodic task execution: {str(e_periodic_task_exec)[:100]}",
                            "type": "PeriodicTaskExecutionError",
                        }
                        self.state.needs_replan = True

                await asyncio.sleep(0.05)  # Small delay between tasks if many are queued
        else:
            self.logger.debug(f"{log_prefix}: No periodic tasks triggered this cycle.")

    async def _trigger_promotion_checks(self):
        # This method uses full MCP names for UMS tool calls
        if not self.state.workflow_id:
            self.logger.debug("Skipping promo check: No active WF.")
            return
        self.logger.debug("Running periodic promotion check...")

        query_memories_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_QUERY_MEMORIES)
        promote_level_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_PROMOTE_MEM)

        if not self._find_tool_server(query_memories_mcp_name) or not self._find_tool_server(promote_level_mcp_name):
            self.logger.warning(
                f"Skipping promotion checks: Required UMS tools for '{UMS_FUNC_QUERY_MEMORIES}' or '{UMS_FUNC_PROMOTE_MEM}' unavailable."
            )
            return

        candidate_ids = set()
        try:
            # Fetch episodic memories
            episodic_args = {
                "workflow_id": self.state.workflow_id,
                "memory_level": MemoryLevel.EPISODIC.value,
                "sort_by": "last_accessed",
                "sort_order": "DESC",
                "limit": 5,
                "include_content": False,
            }
            episodic_res = await self._execute_tool_call_internal(query_memories_mcp_name, episodic_args, record_action=False)
            if episodic_res.get("success"):
                mems = episodic_res.get("memories", [])
                candidate_ids.update(m.get("memory_id") for m in mems if isinstance(m, dict) and m.get("memory_id"))

            # Fetch semantic memories that are procedures or skills
            semantic_args = {
                "workflow_id": self.state.workflow_id,
                "memory_level": MemoryLevel.SEMANTIC.value,
                "memory_type": None,  # Will be filtered below
                "sort_by": "last_accessed",
                "sort_order": "DESC",
                "limit": 10,
                "include_content": False,  # Fetch more semantic to filter
            }
            semantic_res = await self._execute_tool_call_internal(query_memories_mcp_name, semantic_args, record_action=False)
            if semantic_res.get("success"):
                mems = semantic_res.get("memories", [])
                candidate_ids.update(
                    m.get("memory_id")
                    for m in mems
                    if isinstance(m, dict) and m.get("memory_id") and m.get("memory_type") in [MemoryType.PROCEDURE.value, MemoryType.SKILL.value]
                )

            if candidate_ids:
                self.logger.debug(f"Checking {len(candidate_ids)} memories for promotion: {[_fmt_id(i) for i in candidate_ids]}")
                for mem_id in candidate_ids:
                    if self._shutdown_event.is_set():
                        break
                    # Pass the snapshot workflow_id for the background task's context
                    self._start_background_task(
                        AgentMasterLoop._check_and_trigger_promotion,
                        memory_id=mem_id,
                        workflow_id=self.state.workflow_id,  # Pass current WF ID for context
                        context_id=self.state.context_id,
                    )  # Pass current context ID
            else:
                self.logger.debug("No eligible memories for promotion check.")
        except Exception as e:
            self.logger.error(f"Error during promotion check query: {e}", exc_info=False)

    async def _gather_context(self) -> Dict[str, Any]:
        self.logger.info(f"🛰️ Gathering comprehensive context for LLM (Loop: {self.state.current_loop}).")
        start_time = time.time()
        agent_retrieval_timestamp = datetime.now(timezone.utc).isoformat()

        context_payload: Dict[str, Any] = {
            "agent_name": AGENT_NAME,
            "current_loop": self.state.current_loop,
            "current_plan_snapshot": [p.model_dump(exclude_none=True) for p in self.state.current_plan],
            "last_action_summary": self.state.last_action_summary,
            "consecutive_error_count": self.state.consecutive_error_count,
            # Make a deep copy of last_error_details for the context, so modifications
            # during prompt construction don't affect the agent's true error state.
            "last_error_details": copy.deepcopy(self.state.last_error_details),
            "needs_replan": self.state.needs_replan,
            "workflow_stack_summary": [_fmt_id(wf_id) for wf_id in self.state.workflow_stack[-3:]],
            "meta_feedback": self.state.last_meta_feedback,  # This is cleared after being read
            "current_thought_chain_id": self.state.current_thought_chain_id,
            "retrieval_timestamp_agent_state": agent_retrieval_timestamp,
            "status_message_from_agent": "Context assembly by agent.",
            "errors_in_context_gathering": [],
        }
        # Clear meta_feedback after including it in the context for this turn
        self.state.last_meta_feedback = None

        current_workflow_id_for_context = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        current_cognitive_context_id_for_context = self.state.context_id
        current_plan_step_desc_for_context = self.state.current_plan[0].description if self.state.current_plan else DEFAULT_PLAN_STEP

        if not current_workflow_id_for_context:
            context_payload["status_message_from_agent"] = "No Active Workflow. Agent will be prompted to create one."
            self.logger.warning(context_payload["status_message_from_agent"])
            context_payload["ums_package_retrieval_status"] = "skipped_no_workflow"
            context_payload["agent_assembled_goal_context"] = {
                "retrieved_at": agent_retrieval_timestamp,
                "current_goal_details_from_ums": None,
                "goal_stack_summary_from_agent_state": [],
                "data_source_comment": "No active workflow, so no UMS goal context available.",
            }
            context_payload["processing_time_sec"] = time.time() - start_time
            return context_payload

        context_payload["workflow_id"] = current_workflow_id_for_context
        context_payload["cognitive_context_id_agent"] = current_cognitive_context_id_for_context

        # --- 1. Agent Assembles Its Goal Context (with Enhanced Sync Check) ---
        agent_goal_context_block: Dict[str, Any] = {
            "retrieved_at": agent_retrieval_timestamp,
            "current_goal_details_from_ums": None,  # Will be populated by UMS fetch
            "goal_stack_summary_from_agent_state": [],  # Represents agent's *local* view for comparison/debug
            "data_source_comment": "Goal context assembly by agent.",
            "synchronization_status": "pending",
        }

        # For context display, show the agent's *current local* stack summary before UMS validation
        if self.state.goal_stack:
            agent_goal_context_block["goal_stack_summary_from_agent_state"] = [
                {"goal_id": _fmt_id(g.get("goal_id")), "description": (g.get("description") or "")[:150] + "...", "status": g.get("status")}
                for g in self.state.goal_stack[-CONTEXT_GOAL_STACK_SHOW_LIMIT:]
                if isinstance(g, dict)
            ]

        if self.state.current_goal_id:
            self.logger.info(f"Agent Context: Current local UMS goal ID is {_fmt_id(self.state.current_goal_id)}. Fetching UMS stack to verify...")
            # Fetch the goal stack from UMS based on the agent's current_goal_id
            ums_fetched_stack = await self._fetch_goal_stack_from_ums(self.state.current_goal_id)

            if ums_fetched_stack:
                ums_leaf_goal = ums_fetched_stack[-1]
                agent_goal_context_block["current_goal_details_from_ums"] = ums_leaf_goal  # This is the UMS ground truth for the current leaf

                # *** Synchronization Check ***
                if ums_leaf_goal.get("goal_id") != self.state.current_goal_id:
                    mismatch_error_msg = (
                        f"Goal Sync Mismatch: Agent's current_goal_id '{_fmt_id(self.state.current_goal_id)}' "
                        f"does not match the leaf of the stack fetched from UMS '{_fmt_id(ums_leaf_goal.get('goal_id'))}'. "
                        f"Agent state might be stale. UMS leaf description: '{ums_leaf_goal.get('description', 'N/A')[:70]}...'."
                    )
                    self.logger.error(mismatch_error_msg)
                    context_payload["errors_in_context_gathering"].append(mismatch_error_msg)
                    agent_goal_context_block["synchronization_status"] = "mismatch_forcing_replan"
                    agent_goal_context_block["data_source_comment"] = (
                        "UMS goal stack fetched, but mismatch detected with agent's current goal ID. Forcing replan."
                    )

                    # Set error details and trigger replan
                    self.state.last_error_details = {
                        "type": "GoalSyncError",
                        "error": mismatch_error_msg,
                        "agent_current_goal_id": self.state.current_goal_id,
                        "ums_leaf_goal_id": ums_leaf_goal.get("goal_id"),
                        "ums_leaf_goal_description": ums_leaf_goal.get("description"),
                        "recommendation": f"Re-evaluate operational goal. Agent thought it was working on '{_fmt_id(self.state.current_goal_id)}', but UMS indicates current leaf goal might be '{_fmt_id(ums_leaf_goal.get('goal_id'))}'. Consider calling get_goal_details for both if confusion persists.",
                    }
                    self.state.needs_replan = True
                    # Update the agent's local stack and current_goal_id to reflect UMS reality to aid replan
                    self.state.goal_stack = ums_fetched_stack
                    self.state.current_goal_id = ums_leaf_goal.get("goal_id")
                    self.logger.info(
                        f"Agent Context: Corrected local goal_stack and current_goal_id to UMS truth: New current_goal_id is {_fmt_id(self.state.current_goal_id)}"
                    )
                else:
                    # Agent's current_goal_id matches UMS leaf, now sync the local stack content.
                    self.state.goal_stack = ums_fetched_stack  # Update local stack to exactly match UMS
                    agent_goal_context_block["synchronization_status"] = "synchronized_with_ums"
                    agent_goal_context_block["data_source_comment"] = (
                        "Goal stack and current goal details fetched successfully from UMS and synchronized with agent state."
                    )
                    self.logger.info(f"Agent Context: Goal stack synchronized with UMS for current goal {_fmt_id(self.state.current_goal_id)}.")

            else:  # _fetch_goal_stack_from_ums returned empty list for a non-None self.state.current_goal_id
                fetch_fail_error_msg = (
                    f"Goal Sync Error: Failed to fetch UMS goal stack for agent's current_goal_id '{_fmt_id(self.state.current_goal_id)}'. "
                    f"The goal might have been deleted or become orphaned in UMS. Agent state is likely stale."
                )
                self.logger.error(fetch_fail_error_msg)
                context_payload["errors_in_context_gathering"].append(fetch_fail_error_msg)
                agent_goal_context_block["synchronization_status"] = "fetch_failed_forcing_replan"
                agent_goal_context_block["current_goal_details_from_ums"] = {
                    "error_fetching_details": fetch_fail_error_msg,
                    "goal_id_attempted": self.state.current_goal_id,
                }
                agent_goal_context_block["data_source_comment"] = (
                    "Critical error: Could not fetch UMS goal stack for current agent goal ID. Forcing replan."
                )

                self.state.last_error_details = {
                    "type": "GoalSyncError",
                    "error": fetch_fail_error_msg,
                    "agent_current_goal_id": self.state.current_goal_id,
                    "recommendation": "Current UMS goal seems invalid in UMS. Agent needs to re-evaluate its objectives, possibly by listing top-level goals for the workflow or setting a new root goal if necessary.",
                }
                self.state.needs_replan = True
                # Since the current UMS goal is problematic, clear it and the local stack to force re-establishment.
                self.state.current_goal_id = None
                self.state.goal_stack = []
                self.logger.info("Agent Context: Cleared problematic current_goal_id and local stack due to UMS fetch failure.")

        else:  # self.state.current_goal_id is None (but workflow is active)
            agent_goal_context_block["synchronization_status"] = "no_current_goal_in_agent_state"
            agent_goal_context_block["data_source_comment"] = (
                "No current_goal_id set in agent state for this active workflow. UMS goal context is empty. Agent will be prompted to set one."
            )
            self.logger.info("Agent Context: No current_goal_id set for active workflow. LLM will be prompted to establish a root UMS goal.")
            # needs_replan might already be True if this state was reached via an error.
            # If not, this state implies a need to establish a goal, which the main prompt handles.
            # No specific error here, but the plan should reflect the need to set a goal.

        # Re-populate goal_stack_summary_from_agent_state *after* potential UMS sync
        # This ensures the LLM sees the most up-to-date local stack view
        if self.state.goal_stack:  # If stack exists after sync/error handling
            agent_goal_context_block["goal_stack_summary_from_agent_state"] = [
                {"goal_id": _fmt_id(g.get("goal_id")), "description": (g.get("description") or "")[:150] + "...", "status": g.get("status")}
                for g in self.state.goal_stack[-CONTEXT_GOAL_STACK_SHOW_LIMIT:]
                if isinstance(g, dict)
            ]
        else:  # Stack is empty
            agent_goal_context_block["goal_stack_summary_from_agent_state"] = []

        context_payload["agent_assembled_goal_context"] = agent_goal_context_block

        # --- 2. Call UMS Tool for Rich Context Package ---
        ums_get_rich_context_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_RICH_CONTEXT_PACKAGE)
        ums_package_data: Dict[str, Any] = {}  # Initialize to empty dict
        ums_package_retrieval_status_msg = "pending"

        if self._find_tool_server(ums_get_rich_context_mcp_name):
            focal_id_hint_for_ums = None
            # If last_error_details suggests a focal memory from a previous working memory optimization
            if isinstance(self.state.last_error_details, dict) and self.state.last_error_details.get("focal_memory_id_from_last_wm"):
                focal_id_hint_for_ums = self.state.last_error_details["focal_memory_id_from_last_wm"]

            # If current_goal_details_from_ums (synced from UMS) exists and has an associated thought->memory, prefer that.
            current_ums_goal_details_for_focal_hint = agent_goal_context_block.get("current_goal_details_from_ums")
            if isinstance(current_ums_goal_details_for_focal_hint, dict):
                # Assuming a UMS goal object might have a link to a thought ID, and that thought to a memory ID
                # This part is speculative based on UMS structure. If UMS goals are directly linked to focal memories, use that.
                # For now, let's assume the UMS get_rich_context_package handles focal selection well if no strong hint.
                pass  # Placeholder for more sophisticated focal hint derivation from goal

            ums_package_params = {
                "workflow_id": current_workflow_id_for_context,
                "context_id": current_cognitive_context_id_for_context,  # Agent's view of cognitive context ID
                "current_plan_step_description": current_plan_step_desc_for_context,
                "focal_memory_id_hint": focal_id_hint_for_ums,  # Pass hint
                "fetch_limits": {
                    "recent_actions": CONTEXT_RECENT_ACTIONS_FETCH_LIMIT,
                    "important_memories": CONTEXT_IMPORTANT_MEMORIES_FETCH_LIMIT,
                    "key_thoughts": CONTEXT_KEY_THOUGHTS_FETCH_LIMIT,
                    "proactive_memories": CONTEXT_PROACTIVE_MEMORIES_FETCH_LIMIT,
                    "procedural_memories": CONTEXT_PROCEDURAL_MEMORIES_FETCH_LIMIT,
                    "link_traversal": CONTEXT_LINK_TRAVERSAL_FETCH_LIMIT,
                },
                "show_limits": {  # These are for UMS-side summarization/truncation if it implements it
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
                self.logger.debug(f"Agent Context: Calling UMS tool '{ums_get_rich_context_mcp_name}' with params: {ums_package_params}")
                raw_ums_response = await self._execute_tool_call_internal(ums_get_rich_context_mcp_name, ums_package_params, record_action=False)

                if raw_ums_response.get("success"):
                    ums_package_content = raw_ums_response.get("context_package", {})
                    if not isinstance(ums_package_content, dict):
                        err_msg = f"Agent Context: UMS tool {ums_get_rich_context_mcp_name} returned invalid 'context_package' (type: {type(ums_package_content)})."
                        self.logger.error(err_msg)
                        context_payload["errors_in_context_gathering"].append(err_msg)
                        ums_package_data = {"error_ums_pkg_invalid_type": err_msg}
                        ums_package_retrieval_status_msg = "invalid_package_type"
                    else:
                        self.logger.info("Agent Context: Successfully retrieved rich context package from UMS.")
                        ums_internal_errors = raw_ums_response.get("errors")  # UMS tool's own error list
                        if ums_internal_errors and isinstance(ums_internal_errors, list):
                            context_payload["errors_in_context_gathering"].extend([f"UMS_PKG_ERR: {e}" for e in ums_internal_errors])
                        # Store the actual package, don't pop errors from it if UMS included them
                        ums_package_data = ums_package_content
                        ums_package_retrieval_status_msg = "success"
                else:
                    err_msg = f"Agent Context: UMS rich context pkg retrieval failed: {raw_ums_response.get('error', 'Unknown UMS tool error')}"
                    context_payload["errors_in_context_gathering"].append(err_msg)
                    self.logger.warning(err_msg)
                    ums_package_data = {"error_ums_pkg_retrieval": err_msg}
                    ums_package_retrieval_status_msg = "failed_ums_tool_call"
            except Exception as e:
                err_msg = f"Agent Context: Exception calling UMS for rich context pkg: {e}"
                self.logger.error(err_msg, exc_info=True)
                context_payload["errors_in_context_gathering"].append(err_msg)
                ums_package_data = {"error_ums_pkg_exception": err_msg}
                ums_package_retrieval_status_msg = "exception_calling_ums_tool"
        else:
            err_msg = f"Agent Context: UMS tool for '{UMS_FUNC_GET_RICH_CONTEXT_PACKAGE}' unavailable."
            self.logger.error(err_msg)
            context_payload["errors_in_context_gathering"].append(err_msg)
            ums_package_data = {"error_ums_package_tool_unavailable": err_msg}
            ums_package_retrieval_status_msg = "tool_unavailable"

        context_payload["ums_context_package"] = ums_package_data  # Add the retrieved (or error) package
        context_payload["ums_package_retrieval_status"] = ums_package_retrieval_status_msg

        if ums_package_data.get("ums_compression_details"):  # Check within the actual package data
            self.logger.info(f"Agent Context: UMS package includes compression details: {ums_package_data['ums_compression_details']}")

        final_errors_count = len(context_payload.get("errors_in_context_gathering", []))
        if current_workflow_id_for_context and not final_errors_count:
            context_payload["status_message_from_agent"] = "Workflow active. Context ready."
        elif current_workflow_id_for_context and final_errors_count:
            context_payload["status_message_from_agent"] = f"Workflow active. Context ready with {final_errors_count} gathering errors."

        self.logger.info(
            f"Agent Context: Gathering complete. Status: {context_payload['status_message_from_agent']}. Time: {(time.time() - start_time):.3f}s"
        )
        if final_errors_count > 0:
            self.logger.info(f"Agent Context: Errors during context gathering: {context_payload.get('errors_in_context_gathering')}")

        context_payload["processing_time_sec"] = time.time() - start_time
        return context_payload


    async def execute_llm_decision(
        self,
        llm_decision: Dict[str, Any],  # This is the direct output from MCPClient.process_agent_llm_turn
    ) -> bool:  # Returns True to continue loop, False to stop
        self.logger.info(
            f"AML EXEC_DECISION: Entered for Loop {self.state.current_loop}. Current WF: {_fmt_id(self.state.workflow_id)}, Current UMS Goal: {_fmt_id(self.state.current_goal_id)}"
        )
        self.logger.debug(f"AML EXEC_DECISION: Received LLM Decision from MCPClient: {str(llm_decision)[:500]}")

        tool_call_result_payload_for_heuristic: Optional[Dict[str, Any]] = None
        decision_type = llm_decision.get("decision")
        tool_name_involved_in_turn: Optional[str] = llm_decision.get("tool_name") # Original MCP name

        if not self.state.needs_replan:
            self.state.last_error_details = None
            # Note: consecutive_error_count is reset by _apply_heuristic_plan_update on successful actions

        # --- Process MCPClient's Decision ---

        if decision_type == "tool_executed_by_mcp":
            tool_name_original_mcp = llm_decision.get("tool_name")
            arguments_used = llm_decision.get("arguments", {})
            # ums_payload_direct_from_mcp is the direct dictionary result from the UMS tool
            # (or potentially a string if the tool, like summarize_text, returns raw string and parsing handled it that way)
            ums_payload_direct_from_mcp = llm_decision.get("result")

            self.logger.info(
                f"AML EXEC_DECISION: Processing 'tool_executed_by_mcp'. Tool: '{tool_name_original_mcp}'. "
                f"Direct UMS Payload Preview from MCPClient: {str(ums_payload_direct_from_mcp)[:200]}"
            )

            # Construct the standard envelope that _handle_workflow_and_goal_side_effects
            # and _apply_heuristic_plan_update expect.
            constructed_envelope_for_aml: Dict[str, Any] = {
                "success": False,  # Default, will be updated
                "data": ums_payload_direct_from_mcp,  # IMPORTANT: Nest the UMS payload here
                "error_type": None,
                "error_message": None,
                "status_code": None,
                "details": None,
            }

            base_tool_func_name_for_check = self._get_base_function_name(tool_name_original_mcp)
            critical_state_tools = [
                UMS_FUNC_CREATE_WORKFLOW, UMS_FUNC_CREATE_GOAL, UMS_FUNC_UPDATE_GOAL_STATUS,
                UMS_FUNC_UPDATE_WORKFLOW_STATUS, UMS_FUNC_SAVE_COGNITIVE_STATE, UMS_FUNC_LOAD_COGNITIVE_STATE
            ]

            if isinstance(ums_payload_direct_from_mcp, dict):
                if ums_payload_direct_from_mcp.get("success", False): # Default to False if 'success' key missing
                    constructed_envelope_for_aml["success"] = True
                else: # UMS payload indicates failure or 'success' key is missing/false
                    constructed_envelope_for_aml["success"] = False
                    constructed_envelope_for_aml["error_type"] = ums_payload_direct_from_mcp.get(
                        "error_type", "UMSToolReportedFailureInPayload"
                    )
                    error_msg_from_payload = ums_payload_direct_from_mcp.get(
                        "error_message", ums_payload_direct_from_mcp.get("error")
                    )
                    if error_msg_from_payload is None:
                         error_msg_from_payload = "UMS tool failed as per its payload (or 'success' key missing/false)."
                    constructed_envelope_for_aml["error_message"] = error_msg_from_payload
                    constructed_envelope_for_aml["status_code"] = ums_payload_direct_from_mcp.get("status_code")
                    constructed_envelope_for_aml["details"] = ums_payload_direct_from_mcp.get("details")
            elif ums_payload_direct_from_mcp is not None: # Non-dict payload from UMS (e.g., summarize_text)
                constructed_envelope_for_aml["success"] = True # Assume success if MCPClient passed it
                if base_tool_func_name_for_check in critical_state_tools:
                    self.logger.error(
                        f"AML EXEC_DECISION ('tool_executed_by_mcp'): UMS tool '{tool_name_original_mcp}' "
                        f"(critical for state) returned non-dict payload: {type(ums_payload_direct_from_mcp)}. "
                        f"This is a contract violation. Marking as failure."
                    )
                    constructed_envelope_for_aml["success"] = False
                    constructed_envelope_for_aml["error_type"] = "UMSMalformedPayload"
                    constructed_envelope_for_aml["error_message"] = f"Critical UMS tool '{tool_name_original_mcp}' returned non-dictionary payload."
            else: # ums_payload_direct_from_mcp is None
                constructed_envelope_for_aml["success"] = False
                constructed_envelope_for_aml["error_type"] = "MissingUMSPayloadFromMCP"
                constructed_envelope_for_aml["error_message"] = (
                    f"MCPClient reported tool '{tool_name_original_mcp}' executed by LLM, but UMS payload was missing/None."
                )

            # This is now the correctly structured envelope for heuristic update and side effects
            tool_call_result_payload_for_heuristic = constructed_envelope_for_aml

            base_tool_func_name = self._get_base_function_name(tool_name_original_mcp)
            await self._handle_workflow_and_goal_side_effects(base_tool_func_name, arguments_used, constructed_envelope_for_aml)

            # Update last_action_summary and last_error_details based on the *constructed envelope*
            summary_text_for_log_decision = ""
            if constructed_envelope_for_aml.get("success"):
                data_payload_for_summary_decision = constructed_envelope_for_aml.get("data", {})
                if isinstance(data_payload_for_summary_decision, dict):
                    summary_keys_log_decision = [
                        "summary", "message", "memory_id", "action_id", "artifact_id",
                        "link_id", "thought_chain_id", "thought_id", "state_id", "report",
                        "visualization", "goal_id", "workflow_id", "title"
                    ]
                    found_summary_key_decision = False
                    for k_log_decision in summary_keys_log_decision:
                        if k_log_decision in data_payload_for_summary_decision and data_payload_for_summary_decision[k_log_decision] is not None:
                            val_str_log_decision = str(data_payload_for_summary_decision[k_log_decision])
                            summary_text_for_log_decision = f"{k_log_decision}: {_fmt_id(val_str_log_decision) if 'id' in k_log_decision.lower() else val_str_log_decision}"
                            found_summary_key_decision = True
                            break
                    if not found_summary_key_decision:
                        generic_parts_log_decision = [
                            f"{k_s_d}={_fmt_id(str(v_s_d)) if 'id' in k_s_d.lower() else str(v_s_d)[:20]}"
                            for k_s_d, v_s_d in data_payload_for_summary_decision.items()
                            if v_s_d is not None and k_s_d not in ["success", "processing_time"] # Exclude common UMS envelope keys
                        ][:3]
                        summary_text_for_log_decision = (
                            f"Success. Data: {', '.join(generic_parts_log_decision)}"
                            if generic_parts_log_decision
                            else "Success (UMS payload has no distinct summary key)."
                        )
                elif data_payload_for_summary_decision is not None: # Non-dict but not None
                    summary_text_for_log_decision = f"Success (Data: {str(data_payload_for_summary_decision)[:50]}...)"
                else: # Data is None
                    summary_text_for_log_decision = "Success (No data payload from UMS tool)."
            else: # Envelope indicates failure
                err_type_log_decision = constructed_envelope_for_aml.get("error_type", "ToolExecutionError")
                err_msg_log_decision = str(constructed_envelope_for_aml.get("error_message", "Unknown tool error"))[:100]
                summary_text_for_log_decision = f"Failed ({err_type_log_decision}): {err_msg_log_decision}"

            if constructed_envelope_for_aml.get("status_code"):
                summary_text_for_log_decision += f" (Code: {constructed_envelope_for_aml['status_code']})"

            self.state.last_action_summary = f"{tool_name_original_mcp} (executed by LLM via MCP) -> {summary_text_for_log_decision}"
            self.logger.info(f"🏁 LLM-Executed UMS Tool (via MCP): {self.state.last_action_summary}")

            if not constructed_envelope_for_aml.get("success"):
                self.state.last_error_details = {
                    "tool": tool_name_original_mcp,
                    "args": arguments_used,
                    "error": constructed_envelope_for_aml.get("error_message", "UMS tool failed as reported by MCPClient."),
                    "status_code": constructed_envelope_for_aml.get("status_code"),
                    "type": constructed_envelope_for_aml.get("error_type", "ToolExecutionError"),
                    "details": constructed_envelope_for_aml.get("details")
                }
                if not self.state.needs_replan:
                    self.state.needs_replan = True
                    self.logger.info(f"AML EXEC_DECISION: Setting needs_replan=True due to LLM-executed UMS tool failure for '{tool_name_original_mcp}'.")

        elif decision_type == "call_tool":
            tool_name_to_execute_by_aml = llm_decision.get("tool_name") # Original MCP Name
            arguments_for_aml_tool = llm_decision.get("arguments", {})
            tool_name_involved_in_turn = tool_name_to_execute_by_aml

            self.logger.info(
                f"AML EXEC_DECISION: Processing 'call_tool' (to be executed by AML). Tool: '{tool_name_to_execute_by_aml}', Args: {str(arguments_for_aml_tool)[:100]}..."
            )

            if not self.state.current_plan:
                self.logger.error("AML EXEC_DECISION: Plan empty before AML tool call! Forcing replan.")
                err_msg_plan = "Plan empty before tool call."
                self.state.last_error_details = {"tool": tool_name_to_execute_by_aml, "args": arguments_for_aml_tool, "error": err_msg_plan, "type": "PlanValidationError"}
                tool_call_result_payload_for_heuristic = {"success": False, "error_message": err_msg_plan, "error_type": "PlanValidationError"}
                self.state.needs_replan = True
            elif not self.state.current_plan[0].description:
                self.logger.error(f"AML EXEC_DECISION: Current plan step invalid (no description)! Step ID: {self.state.current_plan[0].id}. Forcing replan.")
                err_msg_step = "Current plan step invalid (no description)."
                self.state.last_error_details = {"tool": tool_name_to_execute_by_aml, "args": arguments_for_aml_tool, "error": err_msg_step, "type": "PlanValidationError", "step_id": self.state.current_plan[0].id}
                tool_call_result_payload_for_heuristic = {"success": False, "error_message": err_msg_step, "error_type": "PlanValidationError"}
                self.state.needs_replan = True
            elif tool_name_to_execute_by_aml:
                current_step_deps = self.state.current_plan[0].depends_on if self.state.current_plan else []

                # _execute_tool_call_internal ALREADY returns the standardized envelope
                tool_call_result_payload_for_heuristic = await self._execute_tool_call_internal(
                    tool_name_to_execute_by_aml,
                    arguments_for_aml_tool,
                    record_action=(tool_name_to_execute_by_aml != AGENT_TOOL_UPDATE_PLAN),
                    planned_dependencies=current_step_deps,
                )
                # self.state.last_action_summary, self.state.last_error_details, and self.state.needs_replan
                # are set by _execute_tool_call_internal and its call to _handle_workflow_and_goal_side_effects.
            else: # tool_name_to_execute_by_aml is None
                self.logger.error("AML EXEC_DECISION: LLM 'call_tool' (for AML exec) decision missing 'tool_name'.")
                err_msg_missing_tool = "Missing tool name from LLM decision for AML exec."
                self.state.last_error_details = {"decision_data": llm_decision, "error": err_msg_missing_tool, "type": "LLMOutputError"}
                tool_call_result_payload_for_heuristic = {"success": False, "error_message": err_msg_missing_tool, "error_type": "LLMOutputError"}
                self.state.needs_replan = True

        elif decision_type == "thought_process":
            thought_content_text = llm_decision.get("content")
            tool_name_involved_in_turn = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_THOUGHT)
            self.logger.info(f"AML EXEC_DECISION: Processing 'thought_process'. Content: {str(thought_content_text)[:100]}...")

            if thought_content_text:
                # _execute_tool_call_internal returns the standardized envelope
                tool_call_result_payload_for_heuristic = await self._execute_tool_call_internal(
                    tool_name_involved_in_turn,
                    {"content": str(thought_content_text), "thought_type": ThoughtType.INFERENCE.value},
                    record_action=False,
                )
            else:
                self.logger.warning("AML EXEC_DECISION: LLM 'thought_process' decision, but no content provided.")
                err_msg_no_thought_content = "Missing thought content from LLM for 'thought_process' decision."
                tool_call_result_payload_for_heuristic = {"success": False, "error_message": err_msg_no_thought_content, "error_type": "LLMOutputError"}
                self.state.last_action_summary = "LLM Thought: No content."
                self.state.last_error_details = {"decision": "thought_process", "error": err_msg_no_thought_content, "type": "LLMOutputError"}
                self.state.needs_replan = True

        elif decision_type == "complete" or decision_type == "complete_with_artifact":
            tool_call_result_payload_for_heuristic = {"success": True, "data": {"message": "LLM signaled overall completion."}}
            self.logger.info(f"AML EXEC_DECISION: LLM signaled '{decision_type}'. Heuristic plan update will handle state changes.")

        elif decision_type == "plan_update": # Textual plan from LLM, parsed by MCPClient
            tool_call_result_payload_for_heuristic = {"success": True, "data": {"message": "LLM textual plan received for processing."}}
            self.logger.info("AML EXEC_DECISION: LLM provided textual plan update. Heuristic will apply.")

        elif decision_type == "error":
            error_message_from_mcpc = llm_decision.get("message", "Unknown error from LLM decision processing in MCPClient")
            tool_name_involved_in_turn = "agent:llm_decision_error"
            self.logger.error(f"AML EXEC_DECISION: Received 'error' decision from MCPClient: {error_message_from_mcpc}")
            self.state.last_action_summary = f"LLM Decision Error (MCPClient): {error_message_from_mcpc[:100]}"
            if not self.state.last_error_details:
                self.state.last_error_details = {"error": error_message_from_mcpc, "type": llm_decision.get("error_type_for_agent", "LLMError")}
            self.state.needs_replan = True
            tool_call_result_payload_for_heuristic = {"success": False, "error_message": error_message_from_mcpc, "error_type": llm_decision.get("error_type_for_agent", "LLMError")}

        else:
            tool_name_involved_in_turn = "agent:unknown_decision"
            self.logger.error(
                f"AML EXEC_DECISION: Unexpected decision type from MCPClient: '{decision_type}'. Full decision: {str(llm_decision)[:200]}"
            )
            err_msg_unknown_decision = f"Unexpected decision type '{decision_type}' from MCPClient."
            self.state.last_action_summary = f"Agent Error: {err_msg_unknown_decision}"
            self.state.last_error_details = {"error": err_msg_unknown_decision, "type": "AgentError", "llm_decision_payload": llm_decision}
            self.state.needs_replan = True
            tool_call_result_payload_for_heuristic = {"success": False, "error_message": err_msg_unknown_decision, "error_type": "AgentError"}

        # --- Apply Heuristic Plan Update ---
        self.logger.debug(
            f"AML EXEC_DECISION: Calling _apply_heuristic_plan_update. "
            f"LLM Decision Type: '{decision_type}', Tool Involved: '{tool_name_involved_in_turn}', "
            f"Result Payload for Heuristic (Preview): {str(tool_call_result_payload_for_heuristic)[:200]}..."
        )
        # Pass the llm_decision (from MCPClient) and the standardized envelope (tool_call_result_payload_for_heuristic)
        await self._apply_heuristic_plan_update(llm_decision, tool_call_result_payload_for_heuristic)

        # --- Max consecutive error check ---
        if self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
            self.logger.critical(
                f"AML EXEC_DECISION: Max consecutive errors ({self.state.consecutive_error_count}/{MAX_CONSECUTIVE_ERRORS}) reached. Signaling stop."
            )
            update_wf_status_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_WORKFLOW_STATUS)
            if self.state.workflow_id and self._find_tool_server(update_wf_status_mcp_name):
                # This internal call will trigger its own side effects via _execute_tool_call_internal
                await self._execute_tool_call_internal(
                    update_wf_status_mcp_name,
                    {
                        "workflow_id": self.state.workflow_id,
                        "status": WorkflowStatus.FAILED.value,
                        "completion_message": f"Aborted after {self.state.consecutive_error_count} consecutive errors.",
                    },
                    record_action=False,
                )
            else:
                self.state.goal_achieved_flag = False
                if self.state.workflow_id:
                    self.logger.warning(f"Could not update UMS workflow {self.state.workflow_id} to FAILED due to tool unavailability.")
                self.state.workflow_id = None

            await self._save_agent_state()
            self.logger.info(
                f"AML EXEC_DECISION: Returning False (max errors). WF: {_fmt_id(self.state.workflow_id)}, Goal: {_fmt_id(self.state.current_goal_id)}"
            )
            return False

        await self._save_agent_state()
        self.logger.info(
            f"AML EXEC_DECISION: State after turn processing: WF ID='{_fmt_id(self.state.workflow_id)}', Goal ID='{_fmt_id(self.state.current_goal_id)}', "
            f"needs_replan={self.state.needs_replan}, errors={self.state.consecutive_error_count}, "
            f"plan steps={len(self.state.current_plan) if self.state.current_plan else 'N/A'}"
        )

        if self.state.goal_achieved_flag:
            self.logger.info(f"AML EXEC_DECISION: Overall goal achieved. Signaling stop.")
            return False
        if self._shutdown_event.is_set():
            self.logger.info(f"AML EXEC_DECISION: Shutdown event set. Signaling stop.")
            return False
        if not self.state.workflow_id:
            self.logger.info(f"AML EXEC_DECISION: No active workflow_id. Signaling stop.")
            return False

        self.logger.info(f"AML EXEC_DECISION: Returning True (continue). WF ID='{_fmt_id(self.state.workflow_id)}'")
        return True
    

    async def run_main_loop(self, initial_goal_for_this_run: str, max_loops_from_mcpc: int = 100):
        # 1. Increment agent's internal loop counter for THIS turn being processed
        # self.state.current_loop tracks completed turns. This increments for the turn about to be processed.
        # If current_loop is 0 (initial), this turn becomes 1.
        self.state.current_loop += 1
        current_turn_for_log = self.state.current_loop  # For logging this specific turn
        self.logger.info(f"AgentMasterLoop.run_main_loop: Starting TURN {current_turn_for_log}. Max loops (MCPC): {max_loops_from_mcpc}")

        # 2. Check stopping conditions BEFORE any processing for this turn
        if self.state.current_loop > max_loops_from_mcpc:  # Check if *this turn* would exceed max
            self.logger.warning(f"AML: Agent loop ({current_turn_for_log}) would exceed max_loops_from_mcpc ({max_loops_from_mcpc}). Signaling stop.")
            # Decrement loop counter as this turn is not actually processed
            self.state.current_loop -= 1
            return None

        if self.state.goal_achieved_flag:
            self.logger.info(f"AML: Goal previously achieved. Loop {current_turn_for_log} will not run further. Signaling stop.")
            self.state.current_loop -= 1  # This turn wasn't processed
            return None

        if self._shutdown_event.is_set():
            self.logger.info(f"AML: Shutdown signaled. Loop {current_turn_for_log} will not run. Signaling stop.")
            self.state.current_loop -= 1  # This turn wasn't processed
            return None

        # 3. Initial Workflow/Goal Setup Logic (if it's the very first turn *for a new task* or needs re-establishment)
        #    This happens *after* loop counter increment and stop checks.
        #    The `initial_goal_for_this_run` is crucial here.
        if not self.state.workflow_id:
            self.logger.info(
                f"AML (Turn {current_turn_for_log}): No active workflow ID in state. Agent will be prompted to create one using goal: '{initial_goal_for_this_run[:70]}...'"
            )
            self.state.current_plan = [PlanStep(description=f"Establish UMS workflow for task: {initial_goal_for_this_run[:70]}...")]
            self.state.goal_stack = []
            self.state.current_goal_id = None
            self.state.current_thought_chain_id = None
            self.state.needs_replan = False
            self.logger.info(f"AML (Turn {current_turn_for_log}): Initial plan set for new task.")
        elif not self.state.current_thought_chain_id:  # Workflow exists, but no thought chain
            self.logger.info(
                f"AML (Turn {current_turn_for_log}): Active workflow {_fmt_id(self.state.workflow_id)}, but no thought chain. Setting default."
            )
            await self._set_default_thought_chain_id()
        elif not self.state.current_goal_id and self.state.workflow_id:  # Workflow exists, but no current UMS goal
            # This state implies the LLM needs to establish the root UMS goal for the existing workflow.
            # This is handled by the prompt logic in _construct_agent_prompt.
            self.logger.info(
                f"AML (Turn {current_turn_for_log}): Workflow {_fmt_id(self.state.workflow_id)} active, but no current UMS goal. LLM will be prompted to establish root UMS goal using overall workflow goal."
            )
            if not self.state.goal_stack:  # If stack is also empty, ensure plan reflects this.
                self.state.current_plan = [PlanStep(description=f"Establish root UMS goal for existing workflow: {_fmt_id(self.state.workflow_id)}")]
                self.state.needs_replan = False  # New plan set
        # Note: If self.state.goal_stack exists but self.state.current_goal_id is None,
        # _validate_goal_stack_on_load (during init) or logic in _handle_workflow_and_goal_side_effects
        # should ideally resolve self.state.current_goal_id. If it's still None here with a non-empty stack,
        # it implies a potential state inconsistency the LLM might need to address or that needs a specific recovery path.
        # The current prompt guides LLM if current_goal_id is None.

        # 4. Run Periodic Tasks (can modify state, e.g., needs_replan, last_meta_feedback)
        try:
            self.logger.info(f"AML (Turn {current_turn_for_log}): Running periodic tasks...")
            await self._run_periodic_tasks()
        except Exception as e_periodic:
            self.logger.error(f"AML (Turn {current_turn_for_log}): Error during periodic tasks: {e_periodic}", exc_info=True)
            # Setting last_error_details here might be useful if periodic task failures
            # should directly influence the LLM's next decision.
            self.state.last_error_details = {"error": f"Periodic task failure: {str(e_periodic)[:100]}", "type": "PeriodicTaskError"}
            self.state.needs_replan = True  # Assume replan needed if periodic tasks fail critically.
            # Continue to context gathering so LLM sees the error.

        if self._shutdown_event.is_set():  # Re-check after periodic tasks
            self.logger.warning(f"AML (Turn {current_turn_for_log}): Shutdown signaled during/after periodic tasks. Signaling stop.")
            self.state.current_loop -= 1  # This turn wasn't fully processed
            return None

        # 5. Gather Context
        self.logger.info(f"AML (Turn {current_turn_for_log}): Calling _gather_context...")
        agent_context_snapshot = await self._gather_context()
        self.logger.info(
            f"AML (Turn {current_turn_for_log}): _gather_context returned. Status from context: {agent_context_snapshot.get('status_message_from_agent')}"
        )

        # 6. Handle Error State for Prompt
        # Clear last_error_details ONLY if no replan is needed AND workflow is active.
        # If periodic tasks set needs_replan=True, this won't clear it.
        if not self.state.needs_replan and self.state.workflow_id:
            self.state.last_error_details = None  # Cleared if no replan AND workflow active
            self.logger.debug(f"AML (Turn {current_turn_for_log}): Cleared last_error_details (no replan needed, WF active).")
        elif self.state.needs_replan:
            self.logger.info(f"AML (Turn {current_turn_for_log}): needs_replan is True. Preserving last_error_details for LLM.")
        elif not self.state.workflow_id:
            self.logger.debug(f"AML (Turn {current_turn_for_log}): No active workflow, last_error_details preserved if any (likely None here).")

        # 7. Determine Goal for Prompt Construction
        prompt_goal_to_use = initial_goal_for_this_run  # Default to the overall goal for this run
        if self.state.current_goal_id:  # If a specific UMS operational goal is active
            current_op_goal_details_from_ctx = agent_context_snapshot.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
            if (
                current_op_goal_details_from_ctx
                and isinstance(current_op_goal_details_from_ctx, dict)
                and current_op_goal_details_from_ctx.get("goal_id") == self.state.current_goal_id
            ):
                prompt_goal_to_use = current_op_goal_details_from_ctx.get("description", initial_goal_for_this_run)
                self.logger.info(
                    f"AML (Turn {current_turn_for_log}): Using current operational UMS goal for prompt: '{prompt_goal_to_use[:50]}...' (ID: {_fmt_id(self.state.current_goal_id)})"
                )
            else:
                self.logger.warning(
                    f"AML (Turn {current_turn_for_log}): Current UMS goal ID {_fmt_id(self.state.current_goal_id)} set, but details not in gathered context. Using overall run goal for prompt."
                )
        elif self.state.workflow_id:  # Workflow active, but no specific current_goal_id (e.g. root goal needs to be set)
            ums_workflow_goal_from_ctx = agent_context_snapshot.get("ums_context_package", {}).get("core_context", {}).get("workflow_goal")
            if ums_workflow_goal_from_ctx:
                prompt_goal_to_use = ums_workflow_goal_from_ctx
                self.logger.info(
                    f"AML (Turn {current_turn_for_log}): Workflow active, no specific UMS goal. Using UMS Workflow Goal from context for prompt: '{prompt_goal_to_use[:50]}...'"
                )
            else:  # Fallback if UMS workflow goal also not in context (should be rare if workflow exists)
                self.logger.info(
                    f"AML (Turn {current_turn_for_log}): Workflow active, no specific UMS goal and UMS workflow goal not in context. Using initial_goal_for_this_run for prompt."
                )
        else:  # No workflow ID yet (first turn for this task)
            self.logger.info(
                f"AML (Turn {current_turn_for_log}): No workflow. Using initial_goal_for_this_run for prompt: '{initial_goal_for_this_run[:50]}...'"
            )

        # 8. Construct Prompt
        # _construct_agent_prompt now uses self.state.current_loop directly for logging current turn
        self.logger.info(
            f"AML (Turn {current_turn_for_log}): Calling _construct_agent_prompt with prompt_goal_to_use: '{prompt_goal_to_use[:70]}...'"
        )
        prompt_messages_for_llm = self._construct_agent_prompt(prompt_goal_to_use, agent_context_snapshot)

        # 9. Final check before returning data to MCPClient
        if self.state.goal_achieved_flag or self._shutdown_event.is_set():  # Re-check after all processing this turn
            self.logger.info(
                f"AML (Turn {current_turn_for_log}): Signaling stop to MCPClient post-prepare. GoalAchieved={self.state.goal_achieved_flag}, Shutdown={self._shutdown_event.is_set()}"
            )
            # Decrement loop counter as this turn's data won't be used by LLM
            self.state.current_loop -= 1
            return None

        self.logger.info(f"AML (Turn {current_turn_for_log}): Data prepared for MCPClient. Prompt messages count: {len(prompt_messages_for_llm)}")
        return {"prompt_messages": prompt_messages_for_llm, "tool_schemas": self.tool_schemas, "agent_context": agent_context_snapshot}
