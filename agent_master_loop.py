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
        CallToolResult = "CallToolResult" # Fallback to string for runtime if mcp not in sys.path properly

if TYPE_CHECKING:
    from mcp_client_multi import MCPClient  # This will be the actual class at runtime
else:
    MCPClient = "MCPClient" # Placeholder for static analysis

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
UMS_SERVER_NAME = "Ultimate MCP Server" # ASSUMPTION: Matches your MCP Server's registered name

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
UMS_FUNC_CREATE_LINK = "create_memory_link" # Corrected from create_memory_link
UMS_FUNC_SEARCH_SEMANTIC_MEMORIES = "search_semantic_memories"
UMS_FUNC_QUERY_MEMORIES = "query_memories"
UMS_FUNC_HYBRID_SEARCH = "hybrid_search_memories" # Corrected from hybrid_search
UMS_FUNC_UPDATE_MEMORY = "update_memory"
UMS_FUNC_GET_LINKED_MEMORIES = "get_linked_memories"
UMS_FUNC_GET_WORKING_MEMORY = "get_working_memory"
UMS_FUNC_FOCUS_MEMORY = "focus_memory"
UMS_FUNC_OPTIMIZE_WM = "optimize_working_memory" # Corrected from optimize_working_memory
UMS_FUNC_SAVE_COGNITIVE_STATE = "save_cognitive_state"
UMS_FUNC_LOAD_COGNITIVE_STATE = "load_cognitive_state"
UMS_FUNC_AUTO_FOCUS = "auto_update_focus" # Corrected from auto_update_focus
UMS_FUNC_PROMOTE_MEM = "promote_memory_level" # Corrected from promote_memory_level
UMS_FUNC_CONSOLIDATION = "consolidate_memories" # Corrected from consolidate_memories
UMS_FUNC_REFLECTION = "generate_reflection" # Corrected from generate_reflection
UMS_FUNC_SUMMARIZE_TEXT = "summarize_text"
UMS_FUNC_SUMMARIZE_CONTEXT_BLOCK = "summarize_context_block"
UMS_FUNC_DELETE_EXPIRED_MEMORIES = "delete_expired_memories"
UMS_FUNC_COMPUTE_STATS = "compute_memory_statistics" # Corrected from compute_memory_statistics
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
            if part is None: continue
            if part in container and isinstance(container[part], dict):
                container = container[part]
            else:
                valid_path = False
                break
        if not valid_path: continue

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
        if not valid_path_pop: continue

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
    assigned_tool: Optional[str] = None # This will be the original MCP Name (e.g., "UMS_SERVER_NAME:function_name")
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
    goal_stack: List[Dict[str, Any]] = field(default_factory=list) # Stores UMS Goal objects
    current_goal_id: Optional[str] = None # ID of the current UMS goal
    current_plan: List[PlanStep] = field(default_factory=lambda: [PlanStep(description=DEFAULT_PLAN_STEP)])
    current_thought_chain_id: Optional[str] = None
    last_action_summary: str = "Loop initialized."
    current_loop: int = 0
    goal_achieved_flag: bool = False # For overall workflow goal achievement
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
    tool_usage_stats: Dict[str, Dict[str, Union[int, float]]] = field(default_factory=_default_tool_stats) # Key is original MCP Name
    background_tasks: Set[asyncio.Task] = field(default_factory=set, init=False, repr=False)

# =====================================================================
# Agent Master Loop
# =====================================================================
class AgentMasterLoop:
    # This set should contain the BASE FUNCTION NAMES of UMS tools considered meta/internal.
    # Example: "record_action_start", "get_workflow_details", etc.
    _INTERNAL_OR_META_TOOLS_BASE_NAMES: Set[str] = { 
        UMS_FUNC_RECORD_ACTION_START, UMS_FUNC_RECORD_ACTION_COMPLETION,
        "get_workflow_context", UMS_FUNC_GET_RICH_CONTEXT_PACKAGE, UMS_FUNC_GET_WORKING_MEMORY,
        UMS_FUNC_SEARCH_SEMANTIC_MEMORIES, UMS_FUNC_HYBRID_SEARCH, UMS_FUNC_QUERY_MEMORIES,
        UMS_FUNC_GET_MEMORY_BY_ID, UMS_FUNC_GET_LINKED_MEMORIES, UMS_FUNC_GET_ACTION_DETAILS,
        UMS_FUNC_GET_ARTIFACTS, UMS_FUNC_GET_ARTIFACT_BY_ID, UMS_FUNC_GET_ACTION_DEPENDENCIES,
        UMS_FUNC_GET_THOUGHT_CHAIN, UMS_FUNC_GET_WORKFLOW_DETAILS, UMS_FUNC_GET_GOAL_DETAILS,
        UMS_FUNC_LIST_WORKFLOWS, UMS_FUNC_COMPUTE_STATS, UMS_FUNC_SUMMARIZE_TEXT,
        UMS_FUNC_SUMMARIZE_CONTEXT_BLOCK, UMS_FUNC_OPTIMIZE_WM, UMS_FUNC_AUTO_FOCUS,
        UMS_FUNC_PROMOTE_MEM, UMS_FUNC_REFLECTION, UMS_FUNC_CONSOLIDATION,
        UMS_FUNC_DELETE_EXPIRED_MEMORIES, UMS_FUNC_GET_RECENT_ACTIONS,
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
        self.tool_schemas: List[Dict[str, Any]] = [] # Populated during initialize by MCPClient
        _module_level_ums_func_constants = {
            k: v for k, v in globals().items() # Get all globals in agent_master_loop.py's module scope
            if k.startswith("UMS_FUNC_") and isinstance(v, str)
        }
        self.all_ums_base_function_names: Set[str] = set(_module_level_ums_func_constants.values())
        self.logger.info(f"AgentMasterLoop initialized. LLM: {self.agent_llm_model}, UMS Server Name assumed: {UMS_SERVER_NAME}")
        self.logger.debug(f"Initialized {len(self.all_ums_base_function_names)} UMS base function names for internal reference. Sample: {list(self.all_ums_base_function_names)[:5]}")

    async def shutdown(self) -> None:
        self.logger.info("Shutdown requested.")
        self._shutdown_event.set()
        await self._cleanup_background_tasks()
        # Determine if this shutdown implies the current workflow is truly done
        # For example, if goal_achieved_flag is true, or if consecutive_error_count is high.
        # This logic might need refinement based on how you want to handle "pauses" vs "full stops".
        is_workflow_terminal = (
            self.state.goal_achieved_flag or
            (self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS) or
            (self.state.current_loop >= getattr(self, '_current_max_loops_for_run', float('inf'))) # If tracking max loops for current run
        )
        # If a workflow was active and is now considered terminal, clear its state.
        # If workflow_id is already None, this won't do much.
        if self.state.workflow_id and is_workflow_terminal:
            self.logger.info(f"AML Shutdown: Workflow '{_fmt_id(self.state.workflow_id)}' considered terminal. Clearing active workflow state before saving.")
            self.state.workflow_id = None
            self.state.context_id = None # Usually tied to workflow_id
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
             self.logger.info(f"AML Shutdown: Workflow '{_fmt_id(self.state.workflow_id)}' was active but not considered terminal. Persisting its state.")
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
        self.logger.debug(f"AML CONSTRUCT_PROMPT: self.state.goal_stack (summary) = {[{'id': _fmt_id(g.get('goal_id')), 'desc': (g.get('description') or '')[:30]+'...', 'status': g.get('status')} for g in self.state.goal_stack]}")
        self.logger.debug(f"AML CONSTRUCT_PROMPT: current_task_goal_desc (param for this prompt) = {current_task_goal_desc[:100]}...")
        
        agent_status_message = context.get("status_message_from_agent", "Status unknown.")
        self.logger.debug(f"AML CONSTRUCT_PROMPT: context['status_message_from_agent'] = {agent_status_message}")

        context_goal_details = context.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
        if context_goal_details and isinstance(context_goal_details, dict):
            self.logger.debug(f"AML CONSTRUCT_PROMPT: context['agent_assembled_goal_context']['current_goal_details_from_ums'] = {{'id': '{_fmt_id(context_goal_details.get('goal_id'))}', 'desc': '{(context_goal_details.get('description') or '')[:50]}...'}}")
        else:
            self.logger.debug(f"AML CONSTRUCT_PROMPT: context['agent_assembled_goal_context']['current_goal_details_from_ums'] = {context_goal_details}")
                
        system_blocks: List[str] = [
            f"You are '{AGENT_NAME}', an AI agent orchestrator using a Unified Memory System (UMS) provided by the '{UMS_SERVER_NAME}' server.",
            "", 
        ]

        # Determine the LLM-visible name for agent:update_plan
        llm_seen_agent_update_plan_name_for_instr = AGENT_TOOL_UPDATE_PLAN # Default to original
        for schema_item_instr in self.tool_schemas:
            # Correctly extract sanitized name based on Anthropic or OpenAI-like format
            schema_sanitized_name_instr = None
            if isinstance(schema_item_instr, dict):
                if schema_item_instr.get("type") == "function" and isinstance(schema_item_instr.get("function"), dict): # OpenAI-like
                    schema_sanitized_name_instr = schema_item_instr.get("function", {}).get("name")
                elif "name" in schema_item_instr: # Anthropic-like
                     schema_sanitized_name_instr = schema_item_instr.get("name")

            if schema_sanitized_name_instr:
                original_mcp_name_for_this_schema_instr = self.mcp_client.server_manager.sanitized_to_original.get(schema_sanitized_name_instr)
                if original_mcp_name_for_this_schema_instr == AGENT_TOOL_UPDATE_PLAN:
                    llm_seen_agent_update_plan_name_for_instr = schema_sanitized_name_instr
                    break
        # Fallback sanitization if not found in schemas (should ideally not happen if init is correct)
        if llm_seen_agent_update_plan_name_for_instr == AGENT_TOOL_UPDATE_PLAN:
            llm_seen_agent_update_plan_name_for_instr = re.sub(r"[^a-zA-Z0-9_-]", "_", AGENT_TOOL_UPDATE_PLAN)[:64] or "agent_update_plan_fallback"


        if not self.state.workflow_id: 
            system_blocks.append(f"Current State: NO ACTIVE UMS WORKFLOW. (Agent Status: {agent_status_message}) Your immediate primary objective is to establish one using the task description below.")
            system_blocks.append(f"Initial Overall Task Description (from user/MCPClient): {current_task_goal_desc}")
            system_blocks.append(f"**Action Required: You MUST first call the tool whose base function is '{UMS_FUNC_CREATE_WORKFLOW}'.** Use the 'Initial Overall Task Description' above as the 'goal' parameter for this tool. Provide a suitable 'title'.")
        else:
            ums_workflow_goal_from_context = context.get('ums_context_package', {}).get('core_context', {}).get('workflow_goal', 'N/A')
            system_blocks.append(f"Active UMS Workflow ID: {_fmt_id(self.state.workflow_id)} (on server '{UMS_SERVER_NAME}')")
            system_blocks.append(f"Overall UMS Workflow Goal: {ums_workflow_goal_from_context}")

            current_operational_goal_details = context.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
            if current_operational_goal_details and isinstance(current_operational_goal_details, dict) and current_operational_goal_details.get("goal_id"):
                desc = current_operational_goal_details.get("description", "N/A")
                gid = _fmt_id(current_operational_goal_details.get("goal_id"))
                status = current_operational_goal_details.get("status", "N/A")
                system_blocks.append(f"Current Operational UMS Goal: {desc} (ID: {gid}, Status: {status})")
            else: 
                system_blocks.append(f"Current State: UMS Workflow '{_fmt_id(self.state.workflow_id)}' is ACTIVE, but NO specific UMS operational goal is currently set in agent's focus. (Agent Status: {agent_status_message})")
                system_blocks.append(f"The Overall UMS Workflow Goal is: {ums_workflow_goal_from_context}")
                system_blocks.append(f"**Action Required: Your next step should be to establish the primary UMS operational goal for this workflow.**")
                system_blocks.append(f"   - If the Overall UMS Workflow Goal ('{ums_workflow_goal_from_context[:50]}...') is suitable as the first operational UMS goal, use the tool with base function '{UMS_FUNC_CREATE_GOAL}' to create it. Set `parent_goal_id` to `null` or omit it. Use the Overall UMS Workflow Goal as the description for this new UMS goal.")
                system_blocks.append(f"   - Then, update your plan using the tool named `{llm_seen_agent_update_plan_name_for_instr}` to reflect steps towards this new UMS goal.")

        system_blocks.append("")
        system_blocks.append(f"Available Tools (Use ONLY these for UMS/Agent actions; format arguments per schema. Refer to tools by 'Name LLM Sees'):")

        if not self.tool_schemas:
            system_blocks.append("- CRITICAL WARNING: No tools loaded into agent's schema list. Cannot function effectively.")
        else:
            essential_tool_base_names = { 
                UMS_FUNC_ADD_ACTION_DEPENDENCY, UMS_FUNC_RECORD_ARTIFACT, UMS_FUNC_HYBRID_SEARCH,
                UMS_FUNC_STORE_MEMORY, UMS_FUNC_UPDATE_MEMORY, UMS_FUNC_CREATE_LINK,
                UMS_FUNC_CREATE_THOUGHT_CHAIN, UMS_FUNC_GET_THOUGHT_CHAIN, UMS_FUNC_RECORD_THOUGHT,
                UMS_FUNC_REFLECTION, UMS_FUNC_CONSOLIDATION, 
                UMS_FUNC_CREATE_GOAL, UMS_FUNC_UPDATE_GOAL_STATUS, UMS_FUNC_GET_GOAL_DETAILS, 
                UMS_FUNC_CREATE_WORKFLOW, 
                UMS_FUNC_GET_RICH_CONTEXT_PACKAGE, # Added as essential for context
            }
            essential_agent_tool_mcp_names = { AGENT_TOOL_UPDATE_PLAN }

            for schema in self.tool_schemas: 
                llm_seen_name = None
                if isinstance(schema, dict):
                    if schema.get("type") == "function" and isinstance(schema.get("function"), dict): # OpenAI-like
                        llm_seen_name = schema.get("function", {}).get("name")
                    elif "name" in schema: # Anthropic-like
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
                input_schema_obj = schema.get("input_schema") # Anthropic
                if not input_schema_obj and schema.get("type") == "function": # OpenAI
                    input_schema_obj = schema.get("function", {}).get("parameters")
                
                input_schema_str = json.dumps(input_schema_obj or {"type": "object", "properties": {}}, ensure_ascii=False) # Ensure valid JSON for empty

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
                f"    a. State step-by-step reasoning towards the Current Operational UMS Goal (or the Initial Overall Task / Overall UMS Workflow Goal if no specific UMS operational goal is active). Record key thoughts using the tool with base function '{UMS_FUNC_RECORD_THOUGHT}'.",
                "    b. Evaluate `current_plan`. Is it aligned? Valid? Addresses errors? Dependencies met?",
                f"    c. **Goal Management:** If Current Operational UMS Goal is too complex, use the tool with base function '{UMS_FUNC_CREATE_GOAL}' (providing `parent_goal_id` as current UMS goal ID). When a goal is met/fails, use tool with base function '{UMS_FUNC_UPDATE_GOAL_STATUS}' with the UMS `goal_id` and status.",
                f"    d. Action Dependencies: For plan steps, use `depends_on` with step IDs. Then use UMS tool with base function '{UMS_FUNC_ADD_ACTION_DEPENDENCY}' (with UMS action IDs) if inter-action dependencies are needed.",
                f"    e. Artifacts: Plan to use tool with base function '{UMS_FUNC_RECORD_ARTIFACT}' for creations. Use tools with base functions like '{UMS_FUNC_GET_ARTIFACTS}' or '{UMS_FUNC_GET_ARTIFACT_BY_ID}' for existing.",
                f"    f. Memory: Use tool with base function '{UMS_FUNC_STORE_MEMORY}' for new facts/insights. Use tool with base function '{UMS_FUNC_UPDATE_MEMORY}' for corrections.",
                f"    g. Thought Chains: Use tool with base function '{UMS_FUNC_CREATE_THOUGHT_CHAIN}' for distinct sub-problems.",
                f"    h. Linking: Use tool with base function '{UMS_FUNC_CREATE_LINK}' for relationships between UMS memories.",
                f"    i. Search: Prefer tool with base function '{UMS_FUNC_HYBRID_SEARCH}'. Use tool with base function '{UMS_FUNC_SEARCH_SEMANTIC_MEMORIES}' for pure conceptual similarity.",
                f"    j. Plan Update Tool: Use the tool named `{llm_seen_agent_update_plan_name_for_instr}` ONLY for significant changes, error recovery, or fixing validation issues. Do NOT use for simple step completion.",
                "4.  Action Decision:",
                f"    *   If NO ACTIVE UMS WORKFLOW: Your ONLY action MUST be to call the tool whose base function is '{UMS_FUNC_CREATE_WORKFLOW}'. Use 'Initial Overall Task Description' from above as the 'goal' for this tool.",
                f"    *   If UMS Workflow IS ACTIVE BUT NO specific UMS operational goal is set: Your ONLY action MUST be to call the tool with base function '{UMS_FUNC_CREATE_GOAL}' to establish the root UMS goal for the current workflow. Use the 'Overall UMS Workflow Goal' as its description.",
                "    *   If a workflow AND a specific UMS operational goal ARE active: Choose ONE action based on the *first step in Current Plan*:",
                f"        - Call a UMS Tool (e.g., one with base function '{UMS_FUNC_STORE_MEMORY}', '{UMS_FUNC_RECORD_ARTIFACT}'). Provide args per schema.",
                f"        - Record Thought: Use tool with base function '{UMS_FUNC_RECORD_THOUGHT}'.",
                f"        - Update Plan Tool: Call `{llm_seen_agent_update_plan_name_for_instr}` with the **complete, repaired** plan if replanning is necessary.",
                f"        - Signal Completion: If Current Operational UMS Goal is MET (use tool with base function '{UMS_FUNC_UPDATE_GOAL_STATUS}') OR if the Overall UMS Workflow Goal is MET (respond ONLY with 'Goal Achieved: ...summary...').",
                "5.  Output Format: Respond ONLY with the valid JSON for the chosen tool call OR the 'Goal Achieved: ...summary...' text.",
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
            ]
        )
        system_blocks.extend(
            [
                 "\nRecovery Strategies based on `last_error_details.type`:",
                f"*   `InvalidInputError`: Review tool schema, args, context. Correct args and retry OR choose different tool/step.",
                f"*   `DependencyNotMetError`: Use tool with base function '{UMS_FUNC_GET_ACTION_DETAILS}' on dependency IDs. Adjust plan order (`{llm_seen_agent_update_plan_name_for_instr}`) or wait.",
                f"*   `ServerUnavailable` / `NetworkError`: Tool's server might be down. Try different tool, wait, or adjust plan.",
                f"*   `APILimitError` / `RateLimitError`: External API busy. Plan to wait (record thought) before retry.",
                f"*   `ToolExecutionError` / `ToolInternalError`: Tool failed. Analyze message. Try different args, alternative tool, or adjust plan.",
                f"*   `PlanUpdateError`: Proposed plan structure was invalid. Re-examine plan and dependencies, try `{llm_seen_agent_update_plan_name_for_instr}` again.",
                f"*   `PlanValidationError`: Proposed plan has logical issues (e.g., cycles). Debug dependencies, propose corrected plan using `{llm_seen_agent_update_plan_name_for_instr}`.",
                f"*   `CancelledError`: Previous action cancelled. Re-evaluate current step.",
                f"*   `GoalManagementError`: Error managing UMS goal stack (e.g., marking non-existent goal). Review `agent_assembled_goal_context` and goal logic.",
                f"*   `CognitiveStateError`: Error saving or loading agent's cognitive state. This is serious. Attempt to record key information as memories and then try to re-establish state or simplify the current task.",
                f"*   `InternalStateSetupError`: Critical internal error during agent/workflow setup. Analyze error. May require `{llm_seen_agent_update_plan_name_for_instr}` to fix plan or re-initiate a step.",
                f"*   `UnknownError` / `UnexpectedExecutionError` / `AgentError` / `MCPClientError` / `LLMError` / `LLMOutputError`: Analyze error message carefully. Simplify step, use different approach, or record_thought if stuck. If related to agent state, try to save essential info and restart a simpler sub-task.",
            ]
        )
        system_prompt_str = "\n".join(system_blocks)

        # Truncate context for the prompt if necessary
        context_json_str = _truncate_context(context) # _truncate_context uses agent's constants
        
        user_prompt_blocks = [
            "Current Context:", "```json", context_json_str, "```", "",
            "Current Plan:", "```json", json.dumps([step.model_dump(exclude_none=True) for step in self.state.current_plan], indent=2, ensure_ascii=False), "```", "",
            f"Last Action Summary:\n{self.state.last_action_summary}\n",
        ]
        if self.state.last_error_details:
            user_prompt_blocks.extend(["**CRITICAL: Address Last Error Details**:", "```json", json.dumps(self.state.last_error_details, indent=2, default=str, ensure_ascii=False), "```", ""])
        if self.state.last_meta_feedback: # This comes from reflections/consolidations
            user_prompt_blocks.extend(["**Meta-Cognitive Feedback (e.g., a suggested plan or insight from reflection/consolidation):**", self.state.last_meta_feedback, ""])

        current_goal_desc_for_reminder = "Overall UMS Workflow Goal or Initial Task"
        if self.state.workflow_id:
            current_op_goal_details_reminder = context.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
            if current_op_goal_details_reminder and isinstance(current_op_goal_details_reminder, dict) and current_op_goal_details_reminder.get("description"):
                current_goal_desc_for_reminder = current_op_goal_details_reminder["description"]
            elif context.get('ums_context_package', {}).get('core_context', {}).get('workflow_goal'):
                current_goal_desc_for_reminder = context['ums_context_package']['core_context']['workflow_goal']
        else: 
            current_goal_desc_for_reminder = current_task_goal_desc
        
        user_prompt_blocks.append(f"Current Goal Reminder: {current_goal_desc_for_reminder}")
        user_prompt_blocks.append("")

        # Construct the final instruction based on agent state
        final_instruction_text = ""
        if not self.state.workflow_id:
            final_instruction_text = f"Instruction: NO ACTIVE UMS WORKFLOW. Your first action MUST be to call the tool whose base function is '{UMS_FUNC_CREATE_WORKFLOW}'. Use the 'Initial Overall Task Description' from the system prompt as the 'goal' for this tool. Provide a suitable 'title'."
        elif self.state.workflow_id and not self.state.current_goal_id: 
            final_instruction_text = f"Instruction: UMS WORKFLOW ACTIVE, BUT NO UMS OPERATIONAL GOAL SET. Your first action MUST be to call the tool with base function '{UMS_FUNC_CREATE_GOAL}' to establish the root UMS goal for the current workflow. Use the 'Overall UMS Workflow Goal' as its description."
        elif self.state.needs_replan and self.state.last_meta_feedback: # Check for this specific condition first
            final_instruction_text = (
                f"Instruction: **REPLANNING REQUIRED.** Meta-cognitive feedback (see 'Meta-Cognitive Feedback' in context, often a plan from reflection) is available. "
                f"Your primary action MUST be to use the tool named `{llm_seen_agent_update_plan_name_for_instr}` to set a new, detailed plan. "
                f"Carefully consider the meta-feedback when formulating the new plan. After updating the plan, the agent will proceed with the first step of the new plan in the *next* turn."
            )
        elif self.state.needs_replan: # General replan needed, possibly due to error
             final_instruction_text = (
                f"Instruction: **REPLANNING REQUIRED.** An error occurred or a significant state change necessitates a new plan. "
                f"Analyze `last_error_details` and other context. Your primary action MUST be to use the tool named `{llm_seen_agent_update_plan_name_for_instr}` to propose a new, complete, and valid plan to address the situation and achieve the Current Operational UMS Goal."
            )
        else: # Normal operation, workflow and goal active, no replan needed
            final_instruction_text = (
                f"Instruction: Proceed with the first 'planned' step in 'Current Plan'. "
                f"If dependencies are met, call the assigned UMS tool or record a thought (base func '{UMS_FUNC_RECORD_THOUGHT}'). "
                f"If the Current Operational UMS Goal is met, use tool with base func '{UMS_FUNC_UPDATE_GOAL_STATUS}'. "
                f"If Overall UMS Workflow Goal met, respond 'Goal Achieved: ...summary...'. "
                f"Only use `{llm_seen_agent_update_plan_name_for_instr}` if plan becomes invalid or needs strategic change."
            )
        
        self.logger.info(f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): Final instruction: {final_instruction_text}")
        user_prompt_blocks.append(final_instruction_text)
        user_prompt_str = "\n".join(user_prompt_blocks)

        constructed_prompt_messages = [{"role": "user", "content": system_prompt_str + "\n---\n" + user_prompt_str}]
        
        self.logger.debug(f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages type: {type(constructed_prompt_messages)}")
        if isinstance(constructed_prompt_messages, list):
            self.logger.debug(f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages length: {len(constructed_prompt_messages)}")
            if len(constructed_prompt_messages) > 0:
                self.logger.debug(f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages[0] type: {type(constructed_prompt_messages[0])}")
                if isinstance(constructed_prompt_messages[0], dict):
                    self.logger.debug(f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages[0] keys: {list(constructed_prompt_messages[0].keys())}")
                    content_snippet = str(constructed_prompt_messages[0].get('content', ''))[:500] + "..." # Increased snippet length
                    self.logger.debug(f"AML CONSTRUCT_PROMPT (Turn {current_turn_for_log_prompt}): FINAL CONSTRUCTED prompt_messages[0]['content'] snippet: {content_snippet}")
        
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
                if 'workflow_id' not in final_kwargs_for_coro and snapshot_wf_id:
                    final_kwargs_for_coro['workflow_id'] = snapshot_wf_id
                if 'context_id' not in final_kwargs_for_coro and snapshot_ctx_id:
                     final_kwargs_for_coro['context_id'] = snapshot_ctx_id

                await asyncio.wait_for(
                    coro_fn(self, *args, **final_kwargs_for_coro), 
                    timeout=BACKGROUND_TASK_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                self.logger.warning(f"Background task {asyncio.current_task().get_name()} timed out.")
            except Exception as e:
                self.logger.error(f"Exception in background task wrapper {asyncio.current_task().get_name()}: {e}", exc_info=True)


        task_name = f"bg_{coro_fn.__name__}_{_fmt_id(snapshot_wf_id)}_{random.randint(100, 999)}"
        task = asyncio.create_task(_wrapper(), name=task_name)
        asyncio.create_task(self._add_bg_task(task)) # Schedules _add_bg_task to run
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
                except ValueError: # If trying to release too many
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
            else: # Synchronous version
                token_count = self.anthropic_client.count_tokens(text_to_count)
            return int(token_count)
        except Exception as e:
            self.logger.warning(f"Token estimation via Anthropic API failed: {e}. Using fallback.")
            text_representation = data if isinstance(data, str) else json.dumps(data, default=str, ensure_ascii=False)
            return len(text_representation) // 4 # Rough fallback

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
                    raise # Re-raise the last caught exception
                
                delay = (retry_backoff ** (attempt - 1)) + random.uniform(*jitter)
                self.logger.warning(
                    f"{coro_fun.__name__} failed ({type(e).__name__}: {str(e)[:100]}...); retry {attempt}/{max_retries} in {delay:.2f}s"
                )
                if self._shutdown_event.is_set():
                    self.logger.warning(f"Shutdown during retry for {coro_fun.__name__}.")
                    raise asyncio.CancelledError(f"Shutdown during retry for {coro_fun.__name__}") from last_exception
                await asyncio.sleep(delay)
            except asyncio.CancelledError: # Explicitly handle CancelledError
                self.logger.info(f"{coro_fun.__name__} cancelled during retry wait or execution.")
                raise # Re-raise to propagate cancellation

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
                state_dict_to_save[fld.name] = [
                    dict(g) if not isinstance(g, dict) else g # Ensure it's a dict
                    for g in value if isinstance(g, (dict, BaseModel)) # Allow Pydantic models too if they get in there
                ] if value else []
            else:
                # For other fields, directly assign. asdict handles most basic types.
                # If a field could contain complex non-serializable objects not handled above,
                # specific serialization logic would be needed here.
                state_dict_to_save[fld.name] = value

        state_dict_to_save["timestamp"] = datetime.now(timezone.utc).isoformat()
        
        try:
            self.agent_state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_file = self.agent_state_file.with_suffix(f".tmp_{os.getpid()}_{uuid.uuid4().hex[:4]}") # More unique temp file
            
            json_string_to_write = json.dumps(state_dict_to_save, indent=2, ensure_ascii=False, default=str) # Add default=str as a fallback

            async with aiofiles.open(tmp_file, "w", encoding="utf-8") as f:
                await f.write(json_string_to_write)
                await f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError as e:
                    self.logger.warning(f"os.fsync failed during state save: {e}")
            
            os.replace(tmp_file, self.agent_state_file)
            self.logger.debug(f"State saved atomically -> {self.agent_state_file}")
        except TypeError as te: # Catch specific TypeError during json.dumps
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
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD, 
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )
            self.logger.info("No prior state file. Starting fresh.")
            return
        try:
            async with aiofiles.open(self.agent_state_file, "r", encoding="utf-8") as f:
                data = json.loads(await f.read())
            
            kwargs: Dict[str, Any] = {}
            processed_keys = set()

            for fld in dataclasses.fields(AgentState):
                if not fld.init: continue # Skip non-init fields like background_tasks
                name = fld.name
                processed_keys.add(name)
                if name in data:
                    value = data[name]
                    if name == "current_plan":
                        try:
                            kwargs["current_plan"] = ([PlanStep(**d) for d in value] 
                                                     if isinstance(value, list) and value 
                                                     else [PlanStep(description=DEFAULT_PLAN_STEP)])
                        except (ValidationError, TypeError) as e:
                            self.logger.warning(f"Plan reload failed: {e}. Resetting to default plan.")
                            kwargs["current_plan"] = [PlanStep(description=DEFAULT_PLAN_STEP)]
                    elif name == "tool_usage_stats":
                        dd = _default_tool_stats() # Use the factory for correct type
                        if isinstance(value, dict):
                            for k, v_dict in value.items():
                                if isinstance(v_dict, dict): # Ensure inner value is a dict
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
                else: # Field missing from saved data, use dataclass default
                    if fld.default_factory is not dataclasses.MISSING:
                        kwargs[name] = fld.default_factory()
                    elif fld.default is not dataclasses.MISSING:
                        kwargs[name] = fld.default
                    # Explicit defaults for thresholds if not covered by above (though they should be)
                    elif name == "current_reflection_threshold":
                        kwargs[name] = BASE_REFLECTION_THRESHOLD
                    elif name == "current_consolidation_threshold":
                        kwargs[name] = BASE_CONSOLIDATION_THRESHOLD
                    elif name == "goal_stack": # Ensure default for goal_stack if somehow missed
                        kwargs[name] = []
                    elif name == "current_goal_id": # current_goal_id can be None
                        kwargs[name] = None


            extra_keys = set(data.keys()) - processed_keys - {"timestamp"} # Exclude known non-field keys
            if extra_keys:
                self.logger.warning(f"Ignoring unknown keys in state file: {extra_keys}")

            temp_state = AgentState(**kwargs) # Create state object

            # Post-load validation and correction
            if not isinstance(temp_state.current_reflection_threshold, int) or \
               not (MIN_REFLECTION_THRESHOLD <= temp_state.current_reflection_threshold <= MAX_REFLECTION_THRESHOLD):
                self.logger.warning(f"Invalid current_reflection_threshold ({temp_state.current_reflection_threshold}) loaded. Resetting to default.")
                temp_state.current_reflection_threshold = BASE_REFLECTION_THRESHOLD
            
            if not isinstance(temp_state.current_consolidation_threshold, int) or \
               not (MIN_CONSOLIDATION_THRESHOLD <= temp_state.current_consolidation_threshold <= MAX_CONSOLIDATION_THRESHOLD):
                self.logger.warning(f"Invalid current_consolidation_threshold ({temp_state.current_consolidation_threshold}) loaded. Resetting to default.")
                temp_state.current_consolidation_threshold = BASE_CONSOLIDATION_THRESHOLD

            if not isinstance(temp_state.goal_stack, list): # Should be handled by load, but double check
                self.logger.warning("goal_stack was not a list after load, resetting to empty list.")
                temp_state.goal_stack = []
            
            if temp_state.current_goal_id and not any(g.get("goal_id") == temp_state.current_goal_id for g in temp_state.goal_stack if isinstance(g, dict)):
                self.logger.warning(f"Loaded current_goal_id {_fmt_id(temp_state.current_goal_id)} not found in loaded goal_stack. Resetting current_goal_id.")
                temp_state.current_goal_id = temp_state.goal_stack[-1].get("goal_id") if temp_state.goal_stack and isinstance(temp_state.goal_stack[-1], dict) else None

            self.state = temp_state # Assign validated state
            self.logger.info(f"Loaded state from {self.agent_state_file}; loop {self.state.current_loop}, WF: {_fmt_id(self.state.workflow_id)}, Goal: {_fmt_id(self.state.current_goal_id)}")

        except (json.JSONDecodeError, TypeError, FileNotFoundError) as e:
            self.logger.error(f"State load failed: {e}. Resetting to default state.", exc_info=True)
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD, 
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
            )
        except Exception as e: # Catch any other unexpected errors
            self.logger.critical(f"Unexpected error loading state: {e}. Resetting to default state.", exc_info=True)
            self.state = AgentState(
                current_reflection_threshold=BASE_REFLECTION_THRESHOLD, 
                current_consolidation_threshold=BASE_CONSOLIDATION_THRESHOLD
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
        return tool_name_input.split(':')[-1]


    async def initialize(self) -> bool:
        self.logger.info("🤖 AML: Initializing Agent Master Loop...")
        await self._load_agent_state()

        if not self.state.workflow_id:
            self.logger.info("🤖 AML Initialize: No workflow_id in loaded state. Ensuring temp workflow file is clear for a truly fresh start.")
            await self._write_temp_workflow_id(None)

        if self.state.workflow_id and not self.state.context_id:
            self.state.context_id = self.state.workflow_id
            self.logger.info(f"🤖 AML Initialize: Set context_id from loaded workflow_id: {_fmt_id(self.state.workflow_id)}")

        try:
            if not self.mcp_client or not hasattr(self.mcp_client, 'server_manager') or not self.mcp_client.server_manager:
                self.logger.critical("🤖 AML CRITICAL: MCPClient or its ServerManager is not initialized. Agent cannot function.")
                return False

            agent_llm_provider_str = self.mcp_client.get_provider_from_model(self.agent_llm_model)
            if not agent_llm_provider_str:
                self.logger.critical(f"🤖 AML CRITICAL: Could not determine LLM provider for agent's model '{self.agent_llm_model}'.")
                return False
            self.logger.info(f"🤖 AML: Agent's LLM is '{self.agent_llm_model}', determined provider: '{agent_llm_provider_str}'.")

            # 1. Get ALL tools formatted by MCPClient for the agent's LLM provider.
            # MCPClient._format_tools_for_provider handles initial sanitization and uniqueness
            # for the set of tools it knows about from ALL connected MCP servers.
            all_llm_formatted_tools_from_mcpc: List[Dict[str, Any]] = self.mcp_client._format_tools_for_provider(agent_llm_provider_str) or []
            self.logger.info(f"🤖 AML: Received {len(all_llm_formatted_tools_from_mcpc)} LLM-formatted tools from MCPClient for provider '{agent_llm_provider_str}'.")
            
            # This map is crucial: MCPClient populated this when it called _format_tools_for_provider.
            # It maps the sanitized names *it generated* back to original MCP names.
            # Create a working copy for this initialization run if modifications are needed below that shouldn't persist globally yet
            # For now, directly reference, assuming MCPClient's map is the source of truth that AML *can* update for its specific needs.
            sanitized_to_original_map_shared_from_mcpc = self.mcp_client.server_manager.sanitized_to_original
            if not sanitized_to_original_map_shared_from_mcpc and all_llm_formatted_tools_from_mcpc:
                self.logger.warning("🤖 AML WARNING: MCPClient's sanitized_to_original map is empty (after _format_tools_for_provider) despite tools being present. This is highly unexpected and will likely lead to errors.")
            else:
                 self.logger.debug(f"🤖 AML: MCPClient's sanitized_to_original map (for provider '{agent_llm_provider_str}') initial size {len(sanitized_to_original_map_shared_from_mcpc)}. Sample: {dict(list(sanitized_to_original_map_shared_from_mcpc.items())[:5])}")

            # 2. Prepare the list of tools for the Agent's LLM, ensuring final uniqueness.
            self.tool_schemas = [] # This will be the final list of schemas for the agent's LLM
            agent_llm_final_used_sanitized_names: Set[str] = set() # Tracks sanitized names *as they will be presented to the agent's LLM*.

            self.logger.info(f"🤖 AML (Tool Init): === STARTING AGENT TOOL SCHEMA PREPARATION (Provider: {agent_llm_provider_str}) ===")

            # Process tools received from MCPClient
            for idx, llm_tool_schema_from_mcpc in enumerate(all_llm_formatted_tools_from_mcpc):
                is_anthropic_format_for_agent = agent_llm_provider_str == Provider.ANTHROPIC.value
                
                # Extract the sanitized name AS IT WAS PREPARED BY MCPCLIENT
                sanitized_name_as_received_from_mcpc = ""
                if isinstance(llm_tool_schema_from_mcpc, dict):
                    if is_anthropic_format_for_agent:
                        sanitized_name_as_received_from_mcpc = llm_tool_schema_from_mcpc.get("name", "")
                    else: # OpenAI-like
                        sanitized_name_as_received_from_mcpc = (llm_tool_schema_from_mcpc.get("function", {}).get("name")
                                                               if isinstance(llm_tool_schema_from_mcpc.get("function"), dict) else "")
                
                if not sanitized_name_as_received_from_mcpc:
                    self.logger.warning(f"🤖 AML (Tool Init - MCPC Tool {idx+1}): Could not extract MCPC-sanitized name from schema. Skipping. Schema: {str(llm_tool_schema_from_mcpc)[:150]}")
                    continue

                original_mcp_name = sanitized_to_original_map_shared_from_mcpc.get(sanitized_name_as_received_from_mcpc)
                if not original_mcp_name:
                    self.logger.error(f"🤖 AML (Tool Init - MCPC Tool {idx+1}): CRITICAL! No original MCP name found in SHARED MAP for MCPC-sanitized name '{sanitized_name_as_received_from_mcpc}'. Original Schema Desc Hint: {str(llm_tool_schema_from_mcpc.get('description', 'N/A'))[:70]}. This tool CANNOT be used by the agent. Skipping.")
                    continue
                
                self.logger.debug(f"🤖 AML (Tool Init - MCPC Tool {idx+1}): Processing Original='{original_mcp_name}', MCPC_Sanitized='{sanitized_name_as_received_from_mcpc}'")

                # Ensure this MCPC-sanitized name is unique in the agent's *final* list.
                # If it clashes, the agent will re-suffix it.
                final_sanitized_name_for_agent_llm = sanitized_name_as_received_from_mcpc
                counter = 1
                while final_sanitized_name_for_agent_llm in agent_llm_final_used_sanitized_names:
                    self.logger.warning(f"🤖 AML (Tool Init - MCPC Tool {idx+1}): COLLISION for agent! Name '{final_sanitized_name_for_agent_llm}' (from MCPC's sanitized name for '{original_mcp_name}') already in agent_llm_final_used_sanitized_names. Re-suffixing for agent.")
                    suffix = f"_agent_v{counter}"
                    potential_name_base = sanitized_name_as_received_from_mcpc # Re-base on the name from MCPC
                    if len(potential_name_base) + len(suffix) > 64:
                        final_sanitized_name_for_agent_llm = potential_name_base[:64 - len(suffix)] + suffix
                    else:
                        final_sanitized_name_for_agent_llm = potential_name_base + suffix
                    counter += 1
                    self.logger.info(f"🤖 AML (Tool Init - MCPC Tool {idx+1}):   New unique name attempt for agent: '{final_sanitized_name_for_agent_llm}'")
                
                agent_llm_final_used_sanitized_names.add(final_sanitized_name_for_agent_llm)
                
                # The schema we are working with is a copy to avoid modifying the list from MCPClient
                updated_schema_for_agent_llm = copy.deepcopy(llm_tool_schema_from_mcpc)
                
                if final_sanitized_name_for_agent_llm != sanitized_name_as_received_from_mcpc:
                    self.logger.info(f"🤖 AML (Tool Init - MCPC Tool {idx+1}): Name re-sanitized FOR AGENT. OriginalMCP='{original_mcp_name}', MCPCSanitized='{sanitized_name_as_received_from_mcpc}', AgentLLMWillSee='{final_sanitized_name_for_agent_llm}'.")
                    
                    name_updated_in_schema = False
                    if is_anthropic_format_for_agent:
                        if "name" in updated_schema_for_agent_llm:
                            updated_schema_for_agent_llm["name"] = final_sanitized_name_for_agent_llm
                            name_updated_in_schema = True
                    else: # OpenAI-like
                        if "function" in updated_schema_for_agent_llm and isinstance(updated_schema_for_agent_llm.get("function"), dict) and "name" in updated_schema_for_agent_llm["function"]:
                            updated_schema_for_agent_llm["function"]["name"] = final_sanitized_name_for_agent_llm
                            name_updated_in_schema = True
                    
                    if not name_updated_in_schema:
                         self.logger.error(f"🤖 AML (Tool Init - MCPC Tool {idx+1}): Could not UPDATE SCHEMA with final unique name '{final_sanitized_name_for_agent_llm}' for '{original_mcp_name}'. Schema structure might be unexpected. Skipping tool.")
                         agent_llm_final_used_sanitized_names.remove(final_sanitized_name_for_agent_llm) # Backtrack: remove from used set
                         continue # Skip this tool

                    # CRITICAL: Update the SHARED sanitized_to_original map in MCPClient.ServerManager
                    # If the agent re-sanitized a name that MCPC had *already* sanitized,
                    # MCPC needs to know this new agent-LLM-visible name maps to the original.
                    # 1. Remove the old mapping from sanitized_name_as_received_from_mcpc -> original_mcp_name
                    #    ONLY IF it actually existed and pointed to THIS original_mcp_name.
                    #    This is to prevent accidentally deleting a mapping for a different original tool
                    #    that happened to sanitize to sanitized_name_as_received_from_mcpc before MCPC's _vX suffixing.
                    if sanitized_name_as_received_from_mcpc in sanitized_to_original_map_shared_from_mcpc and \
                       sanitized_to_original_map_shared_from_mcpc[sanitized_name_as_received_from_mcpc] == original_mcp_name:
                        del sanitized_to_original_map_shared_from_mcpc[sanitized_name_as_received_from_mcpc]
                        self.logger.debug(f"🤖 AML (Tool Init - MCPC Tool {idx+1}): Removed MCPClient's old mapping for '{sanitized_name_as_received_from_mcpc}' because agent re-sanitized it to '{final_sanitized_name_for_agent_llm}' for the same original tool '{original_mcp_name}'.")
                    
                    # 2. Add the new mapping: final_sanitized_name_for_agent_llm -> original_mcp_name
                    sanitized_to_original_map_shared_from_mcpc[final_sanitized_name_for_agent_llm] = original_mcp_name
                    self.logger.info(f"🤖 AML (Tool Init - MCPC Tool {idx+1}): ENSURED/UPDATED MCPClient's SHARED MAPPING due to agent re-sanitize: '{final_sanitized_name_for_agent_llm}' -> '{original_mcp_name}'")

                self.tool_schemas.append(updated_schema_for_agent_llm)
                self.logger.info(f"🤖 AML (Tool Init - MCPC Tool {idx+1}): ADDED MCP Tool to agent's tool_schemas: Original='{original_mcp_name}' (Agent LLM sees: '{final_sanitized_name_for_agent_llm}')")

            # --- Add AGENT_TOOL_UPDATE_PLAN (Agent's internal tool) ---
            plan_step_base_schema = PlanStep.model_json_schema()
            if "title" in plan_step_base_schema: del plan_step_base_schema["title"]
            update_plan_input_schema = {
                "type": "object",
                "properties": {"plan": {"type": "array", "items": plan_step_base_schema, "description": "The new complete list of plan steps for the agent."}},
                "required": ["plan"],
            }
            agent_update_plan_original_mcp_name = AGENT_TOOL_UPDATE_PLAN # e.g., "agent:update_plan"
            
            # Sanitize its name based on the agent's LLM provider rules.
            base_agent_tool_sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", agent_update_plan_original_mcp_name)[:64]
            if not base_agent_tool_sanitized_name: # Should not happen for "agent:update_plan"
                base_agent_tool_sanitized_name = f"internal_agent_tool_{str(uuid.uuid4())[:8]}" # Robust fallback
                self.logger.warning(f"🤖 AML (Agent Tool Init): Agent-internal tool '{agent_update_plan_original_mcp_name}' sanitized to empty, using fallback '{base_agent_tool_sanitized_name}'.")

            final_agent_tool_sanitized_name_for_llm = base_agent_tool_sanitized_name
            agent_tool_counter = 1
            while final_agent_tool_sanitized_name_for_llm in agent_llm_final_used_sanitized_names: # Check against names already added for agent
                self.logger.warning(f"🤖 AML (Agent Tool Init): COLLISION for agent tool '{agent_update_plan_original_mcp_name}' (initial sanitized: '{base_agent_tool_sanitized_name}', current attempt: '{final_agent_tool_sanitized_name_for_llm}'). Re-suffixing.")
                suffix = f"_agent_v{agent_tool_counter}"
                if len(base_agent_tool_sanitized_name) + len(suffix) > 64:
                    final_agent_tool_sanitized_name_for_llm = base_agent_tool_sanitized_name[:64 - len(suffix)] + suffix
                else:
                    final_agent_tool_sanitized_name_for_llm = base_agent_tool_sanitized_name + suffix
                agent_tool_counter += 1
                self.logger.info(f"🤖 AML (Agent Tool Init):   New unique name attempt for agent tool: '{final_agent_tool_sanitized_name_for_llm}'")
            
            agent_llm_final_used_sanitized_names.add(final_agent_tool_sanitized_name_for_llm)

            # Ensure MCPClient can map this agent-specific tool name back to its original form
            sanitized_to_original_map_shared_from_mcpc[final_agent_tool_sanitized_name_for_llm] = agent_update_plan_original_mcp_name
            self.logger.info(f"🤖 AML (Agent Tool Init): ENSURED MCPClient SHARED MAPPING for AGENT_TOOL_UPDATE_PLAN: '{final_agent_tool_sanitized_name_for_llm}' -> '{agent_update_plan_original_mcp_name}'")

            plan_tool_llm_schema_final = {}
            plan_tool_description = "Replace agent's current plan. Use for significant replanning, error recovery, or fixing validation issues. Submit the ENTIRE new plan."
            if agent_llm_provider_str == Provider.ANTHROPIC.value:
                plan_tool_llm_schema_final = {"name": final_agent_tool_sanitized_name_for_llm, "description": plan_tool_description, "input_schema": update_plan_input_schema}
            else: # OpenAI-like
                plan_tool_llm_schema_final = {"type": "function", "function": {"name": final_agent_tool_sanitized_name_for_llm, "description": plan_tool_description, "parameters": update_plan_input_schema}}
            
            self.tool_schemas.append(plan_tool_llm_schema_final)
            self.logger.info(f"🤖 AML (Agent Tool Init): ADDED agent-internal tool to self.tool_schemas for LLM: '{agent_update_plan_original_mcp_name}' (LLM sees: '{final_agent_tool_sanitized_name_for_llm}')")

            self.logger.info(f"🤖 AML: Final {len(self.tool_schemas)} tool schemas prepared for agent's LLM. Sample names agent LLM will see: {[t.get('name') or t.get('function',{}).get('name') for t in self.tool_schemas[:10]]}")
            self.logger.info(f"🤖 AML (Tool Init): === FINISHED AGENT TOOL SCHEMA PREPARATION ===")
            self.logger.debug(f"🤖 AML: MCPClient's sanitized_to_original map (AFTER agent tool processing, size {len(sanitized_to_original_map_shared_from_mcpc)}): {str(sanitized_to_original_map_shared_from_mcpc)[:1000]}...")

            # 4. Essential Tools Check
            essential_mcp_tool_names_list_for_check = [ # Original MCP Names
                self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_WORKFLOW),
                self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_THOUGHT),
                self._get_ums_tool_mcp_name(UMS_FUNC_STORE_MEMORY),
                self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL),
                # Add other UMS tools that are absolutely essential for the agent to function, by their original MCP name.
                # For example, if `get_rich_context_package` is essential for the agent to build its prompt:
                self._get_ums_tool_mcp_name(UMS_FUNC_GET_RICH_CONTEXT_PACKAGE),
                AGENT_TOOL_UPDATE_PLAN,
            ]
            
            # Filter this list to only include UMS tools the agent *knows about* via its UMS_FUNC constants.
            # This prevents errors if an "essential" tool listed above isn't actually defined in agent_master_loop's constants.
            effective_essential_tools_to_check_for = []
            for essential_orig_name in essential_mcp_tool_names_list_for_check:
                if essential_orig_name == AGENT_TOOL_UPDATE_PLAN: # Agent internal tools are always "known"
                    effective_essential_tools_to_check_for.append(essential_orig_name)
                elif essential_orig_name.startswith(UMS_SERVER_NAME + ":"): # It's a UMS tool
                    base_func = self._get_base_function_name(essential_orig_name)
                    if base_func in self.all_ums_base_function_names: # Check if agent has a constant for it
                        effective_essential_tools_to_check_for.append(essential_orig_name)
                    else:
                        self.logger.warning(f"🤖 AML (Essential Check): Tool '{essential_orig_name}' listed as essential, but its base function '{base_func}' is not in agent's `all_ums_base_function_names`. Skipping from essential check.")
                else: # Non-UMS essential tool, assume it should be present
                    effective_essential_tools_to_check_for.append(essential_orig_name)
            
            self.logger.debug(f"🤖 AML (Essential Check): Effective essential original MCP names to check for: {effective_essential_tools_to_check_for}")

            # Check if these essential original names are actually available to the agent's LLM
            # by seeing if they can be mapped back from the sanitized names in self.tool_schemas
            available_original_names_for_agent_llm: Set[str] = set()
            for schema_for_agent_llm_prompt in self.tool_schemas:
                sanitized_name_agent_sees = ""
                if agent_llm_provider_str == Provider.ANTHROPIC.value:
                    sanitized_name_agent_sees = schema_for_agent_llm_prompt.get("name", "")
                else: # OpenAI-like
                    sanitized_name_agent_sees = (schema_for_agent_llm_prompt.get("function", {}).get("name")
                                               if isinstance(schema_for_agent_llm_prompt.get("function"), dict) else "")
                
                if sanitized_name_agent_sees:
                    original_name_mapped = sanitized_to_original_map_shared_from_mcpc.get(sanitized_name_agent_sees)
                    if original_name_mapped:
                        available_original_names_for_agent_llm.add(original_name_mapped)
                    else:
                        self.logger.error(f"🤖 AML (Essential Check): CRITICAL MAPPING ERROR! Agent's final tool schema has sanitized name '{sanitized_name_agent_sees}' but it's NOT in MCPClient's SHARED sanitized_to_original map. This WILL break execution if LLM chooses it.")
            
            missing_essential_tools = [
                orig_name for orig_name in effective_essential_tools_to_check_for
                if orig_name not in available_original_names_for_agent_llm
            ]

            if missing_essential_tools:
                self.logger.error(f"🤖 AML CRITICAL ERROR (Post-Processing): Missing essential tools from agent's FINAL LLM tool list (checked by original MCP name): {missing_essential_tools}. Agent functionality WILL BE severely impaired.")
                self.logger.debug(f"    Effectively available original names for agent's LLM: {available_original_names_for_agent_llm}")
                self.logger.debug(f"    Sanitized names agent LLM will see: {[s.get('name') or s.get('function',{}).get('name') for s in self.tool_schemas]}")
            else:
                self.logger.info("🤖 AML: All specified essential tools (by original MCP name) are confirmed available and correctly mapped for the agent's LLM prompt.")


            # --- Rest of the initialize method ---
            # (Workflow and Goal Stack Validation/Setup - this part was okay from previous version)
            loaded_workflow_id_from_state = (self.state.workflow_stack[-1] if self.state.workflow_stack else None) or self.state.workflow_id
            
            if not loaded_workflow_id_from_state:
                self.logger.info(f"🤖 AML Initialize: No workflow_id in agent state. Checking temp file...")
                recovered_wf_id_from_temp = await self._read_temp_workflow_id()
                if recovered_wf_id_from_temp:
                    self.logger.info(f"🤖 AML Initialize: Found workflow_id '{_fmt_id(recovered_wf_id_from_temp)}' in temp file.")
                    if await self._check_workflow_exists(recovered_wf_id_from_temp):
                        self.logger.info(f"🤖 AML Initialize: Temp file workflow_id '{_fmt_id(recovered_wf_id_from_temp)}' is VALID in UMS. Using it.")
                        self.state.workflow_id = recovered_wf_id_from_temp
                        if not self.state.workflow_stack or self.state.workflow_stack[-1] != recovered_wf_id_from_temp:
                            self.state.workflow_stack = [recovered_wf_id_from_temp]
                        loaded_workflow_id_from_state = recovered_wf_id_from_temp
                    else:
                        self.logger.warning(f"🤖 AML Initialize: Temp file workflow_id '{_fmt_id(recovered_wf_id_from_temp)}' is NOT VALID in UMS. Discarding and clearing temp file.")
                        await self._write_temp_workflow_id(None) 
                else:
                    self.logger.info(f"🤖 AML Initialize: No workflow_id found in temp file either.")
            
            if loaded_workflow_id_from_state:
                self.logger.info(f"🤖 AML Initialize: Processing with effective loaded workflow ID: {_fmt_id(loaded_workflow_id_from_state)}.")
                if not await self._check_workflow_exists(loaded_workflow_id_from_state): 
                    self.logger.warning(f"🤖 AML WARNING: Loaded workflow '{_fmt_id(loaded_workflow_id_from_state)}' not found in UMS. Resetting workflow state.")
                    preserved_tool_stats = self.state.tool_usage_stats
                    pres_ref_thresh = self.state.current_reflection_threshold
                    pres_con_thresh = self.state.current_consolidation_threshold
                    self.state = AgentState( 
                        tool_usage_stats=preserved_tool_stats,
                        current_reflection_threshold=pres_ref_thresh,
                        current_consolidation_threshold=pres_con_thresh
                    )
                    await self._write_temp_workflow_id(None) 
                    await self._save_agent_state() 
                else: 
                    self.state.workflow_id = loaded_workflow_id_from_state 
                    if not self.state.context_id: self.state.context_id = loaded_workflow_id_from_state
                    self.logger.info(f"🤖 AML Initialize: Loaded workflow '{_fmt_id(self.state.workflow_id)}' confirmed in UMS. Validating UMS goal stack.")
                    await self._validate_goal_stack_on_load() 

                    if self.state.workflow_id and not self.state.current_goal_id:
                        self.logger.info(f"AML Initialize: Resuming existing UMS workflow {_fmt_id(self.state.workflow_id)} but no current UMS goal active. Attempting to establish its root UMS goal.")
                        
                        ums_get_wf_details_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_WORKFLOW_DETAILS)
                        workflow_main_goal_desc = f"Fulfill objectives of existing workflow '{_fmt_id(self.state.workflow_id)}'."
                        workflow_title_for_goal = f"Resume Workflow {_fmt_id(self.state.workflow_id)}"

                        if self._find_tool_server(ums_get_wf_details_mcp_name):
                            wf_details_res = await self._execute_tool_call_internal(
                                ums_get_wf_details_mcp_name, 
                                {"workflow_id": self.state.workflow_id, "include_thoughts": False, "include_actions": False, "include_artifacts": False},
                                record_action=False
                            )
                            if wf_details_res.get("success"):
                                workflow_main_goal_desc = wf_details_res.get("goal") or workflow_main_goal_desc
                                workflow_title_for_goal = wf_details_res.get("title") or workflow_title_for_goal
                            else:
                                self.logger.warning(f"AML Initialize: Could not get UMS details for existing workflow {_fmt_id(self.state.workflow_id)}. Error: {wf_details_res.get('error')}")
                        else:
                            self.logger.warning(f"AML Initialize: Tool for '{UMS_FUNC_GET_WORKFLOW_DETAILS}' unavailable.")
                        
                        ums_create_goal_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL)
                        if self._find_tool_server(ums_create_goal_mcp_name):
                            goal_creation_args = {
                                "workflow_id": self.state.workflow_id,
                                "description": workflow_main_goal_desc,
                                "title": f"Root Goal (on resume): {workflow_title_for_goal}",
                                "parent_goal_id": None, 
                                "initial_status": GoalStatus.ACTIVE.value
                            }
                            goal_res = await self._execute_tool_call_internal(ums_create_goal_mcp_name, goal_creation_args, record_action=False)
                            created_ums_goal_obj = goal_res.get("goal") if isinstance(goal_res, dict) and goal_res.get("success") else None

                            if isinstance(created_ums_goal_obj, dict) and created_ums_goal_obj.get("goal_id"):
                                self.state.goal_stack = [created_ums_goal_obj]
                                self.state.current_goal_id = created_ums_goal_obj.get("goal_id")
                                self.logger.info(f"AML Initialize: Successfully set NEW root UMS goal {_fmt_id(self.state.current_goal_id)} for existing workflow.")
                                self.state.current_plan = [PlanStep(description=f"Assess newly established root UMS goal '{workflow_main_goal_desc[:50]}...'")]
                                self.state.needs_replan = False 
                            else:
                                self.logger.error(f"AML Initialize: Failed to create new root UMS goal. Error: {goal_res.get('error', 'Unknown')}")
                                self.state.last_error_details = {"tool": ums_create_goal_mcp_name, "error": f"Failed to create root goal: {goal_res.get('error', 'Unknown')}", "type": "GoalManagementError"}
                                self.state.needs_replan = True 
                        else:
                             self.logger.error(f"AML Initialize: Tool for '{UMS_FUNC_CREATE_GOAL}' unavailable.")
                             self.state.last_error_details = {"tool": ums_create_goal_mcp_name, "error": "Tool unavailable", "type": "ToolUnavailable"}
                             self.state.needs_replan = True
                    
                    if self.state.workflow_id and not self.state.current_thought_chain_id:
                        await self._set_default_thought_chain_id() 
            else: 
                self.logger.info("🤖 AML Initialize: No prior workflow ID in state or temp file. Agent starts completely fresh.")

            self.logger.info("🤖 AML: Agent Master Loop initialization complete.")
            return True

        except Exception as e:
            self.logger.critical(f"🤖 AML CRITICAL: Agent loop initialization failed with an unhandled exception: {e}", exc_info=True)
            return False

    def _find_tool_server(self, tool_identifier_from_agent_or_llm: str) -> Optional[str]:
        # Use direct print to stderr for this debugging phase for this specific function
        print(f"FORCE_PRINT _find_tool_server: Entered for tool_identifier: '{tool_identifier_from_agent_or_llm}'", file=sys.stderr, flush=True)

        if not self.mcp_client or not self.mcp_client.server_manager:
            print("FORCE_PRINT _find_tool_server: MCPClient or ServerManager not available. Returning None.", file=sys.stderr, flush=True)
            self.logger.warning("AML _find_tool_server: MCPClient or ServerManager not available.") # Keep logger for general warnings
            return None
        
        sm = self.mcp_client.server_manager

        # Special check and logging when looking for create_goal
        if tool_identifier_from_agent_or_llm == self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL):
            print(f"FORCE_PRINT _find_tool_server (Specifically for '{self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL)}'):", file=sys.stderr, flush=True)
            print(f"  ServerManager sm.tools keys count: {len(sm.tools)}", file=sys.stderr, flush=True)
            # Printing all keys might be too verbose, let's check for the specific tool
            # print(f"  ServerManager sm.tools keys (first 10): {list(sm.tools.keys())[:10]}...", file=sys.stderr, flush=True)
            print(f"  ServerManager sm.active_sessions keys ({len(sm.active_sessions)}): {list(sm.active_sessions.keys())}", file=sys.stderr, flush=True)
            print(f"  Is '{tool_identifier_from_agent_or_llm}' in sm.tools? {tool_identifier_from_agent_or_llm in sm.tools}", file=sys.stderr, flush=True)
            if tool_identifier_from_agent_or_llm in sm.tools:
                tool_obj_dbg = sm.tools[tool_identifier_from_agent_or_llm]
                print(f"  Tool object server_name: {tool_obj_dbg.server_name}", file=sys.stderr, flush=True)
                print(f"  Is tool_obj_dbg.server_name ('{tool_obj_dbg.server_name}') in sm.active_sessions? {tool_obj_dbg.server_name in sm.active_sessions}", file=sys.stderr, flush=True)
            else:
                print(f"  '{tool_identifier_from_agent_or_llm}' was NOT found in sm.tools when checking for create_goal.", file=sys.stderr, flush=True)


        if tool_identifier_from_agent_or_llm == AGENT_TOOL_UPDATE_PLAN:
            print("FORCE_PRINT _find_tool_server: Recognized AGENT_TOOL_UPDATE_PLAN. Returning 'AGENT_INTERNAL'.", file=sys.stderr, flush=True)
            return "AGENT_INTERNAL"

        if tool_identifier_from_agent_or_llm in sm.tools:
            mcp_tool_obj = sm.tools[tool_identifier_from_agent_or_llm]
            server_name_direct = mcp_tool_obj.server_name
            if server_name_direct in sm.active_sessions:
                print(f"FORCE_PRINT _find_tool_server: Direct match for '{tool_identifier_from_agent_or_llm}' on active server '{server_name_direct}'. Returning server name.", file=sys.stderr, flush=True)
                return server_name_direct
            else:
                print(f"FORCE_PRINT _find_tool_server: Direct match for '{tool_identifier_from_agent_or_llm}', but server '{server_name_direct}' is NOT ACTIVE (Active: {list(sm.active_sessions.keys())}). Will attempt suffix match.", file=sys.stderr, flush=True)
        else:
            print(f"FORCE_PRINT _find_tool_server: Tool '{tool_identifier_from_agent_or_llm}' NOT FOUND via direct full name match in ServerManager.tools.", file=sys.stderr, flush=True)
        
        target_base_function_name = self._get_base_function_name(tool_identifier_from_agent_or_llm)
        print(f"FORCE_PRINT _find_tool_server: Fallback - searching for base function '{target_base_function_name}' on active servers.", file=sys.stderr, flush=True)

        candidate_servers_for_base_name: List[str] = []
        for full_mcp_name_from_server, mcp_tool_obj_iter in sm.tools.items():
            base_func_from_server = self._get_base_function_name(full_mcp_name_from_server)
            if base_func_from_server == target_base_function_name:
                server_name_iter = mcp_tool_obj_iter.server_name
                if server_name_iter in sm.active_sessions:
                    candidate_servers_for_base_name.append(server_name_iter)
        
        if len(candidate_servers_for_base_name) == 1:
            found_server = candidate_servers_for_base_name[0]
            print(f"FORCE_PRINT _find_tool_server: Fallback suffix match found base function '{target_base_function_name}' on active server '{found_server}'. Returning server name.", file=sys.stderr, flush=True)
            return found_server
        elif len(candidate_servers_for_base_name) > 1:
            if UMS_SERVER_NAME in candidate_servers_for_base_name and target_base_function_name in self.all_ums_base_function_names:
                 print(f"FORCE_PRINT _find_tool_server: Ambiguous suffix match for UMS function '{target_base_function_name}' on {candidate_servers_for_base_name}. Prioritizing '{UMS_SERVER_NAME}'. Returning '{UMS_SERVER_NAME}'.", file=sys.stderr, flush=True)
                 return UMS_SERVER_NAME
            print(f"FORCE_PRINT _find_tool_server: Fallback suffix match for '{target_base_function_name}' is AMBIGUOUS on {candidate_servers_for_base_name}. Returning None.", file=sys.stderr, flush=True)
            return None
        else: 
            print(f"FORCE_PRINT _find_tool_server: Fallback suffix match for base function '{target_base_function_name}' FAILED. Returning None.", file=sys.stderr, flush=True)
            return None

    async def _set_default_thought_chain_id(self):
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if not current_wf_id:
            self.logger.debug("No active workflow for default thought chain.")
            return
        
        # Construct full MCP name for get_workflow_details
        get_details_mcp_tool_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_WORKFLOW_DETAILS)

        if self._find_tool_server(get_details_mcp_tool_name): # Check availability using full MCP name
            try:
                details = await self._execute_tool_call_internal(
                    get_details_mcp_tool_name, # Call using full MCP name
                    {
                        "workflow_id": current_wf_id,
                        "include_thoughts": True, "include_actions": False,
                        "include_artifacts": False, "include_memories": False,
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
                    "include_actions": False, "include_artifacts": False,
                    "include_thoughts": False, "include_memories": False,
                },
                record_action=False,
            )
            return isinstance(result, dict) and result.get("success", False)
        except ToolInputError as e: # UMS tool itself raises ToolInputError if not found
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
        if not plan: return False
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
        if not ids: return True, "No dependencies."
        
        get_action_details_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_ACTION_DETAILS)
        if not self._find_tool_server(get_action_details_mcp_name):
            self.logger.error(f"Tool for '{UMS_FUNC_GET_ACTION_DETAILS}' unavailable.")
            return False, f"Tool for '{UMS_FUNC_GET_ACTION_DETAILS}' unavailable."
        
        self.logger.debug(f"Checking prerequisites: {[_fmt_id(i) for i in ids]}")
        try:
            res = await self._execute_tool_call_internal(get_action_details_mcp_name, {"action_ids": ids, "include_dependencies": False}, record_action=False)
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
        
        base_tool_name_for_title = tool_name_mcp.split(':')[-1]
        payload = {
            "workflow_id": current_wf_id,
            "title": f"Execute: {base_tool_name_for_title}",
            "action_type": ActionType.TOOL_USE.value,
            "tool_name": tool_name_mcp, # Store the original MCP name of the tool being executed
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
        if not source_id or not target_ids: return
        valid_targets = {tid for tid in target_ids if tid and tid != source_id}
        if not valid_targets: return
        
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
                    record_action=False
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
        current_agent_workflow_id = self.state.workflow_id # Get current agent state at execution time
        if workflow_id != current_agent_workflow_id or self._shutdown_event.is_set():
            self.logger.debug(f"Skipping auto-link for {_fmt_id(memory_id)}: WF changed ({_fmt_id(workflow_id)} vs {_fmt_id(current_agent_workflow_id)}) or shutdown.")
            return
        
        try:
            if not memory_id or not workflow_id: # Use the snapshot workflow_id
                self.logger.debug(f"Skipping auto-link: Missing memory_id ({_fmt_id(memory_id)}) or snapshot workflow_id ({_fmt_id(workflow_id)}).")
                return
            await asyncio.sleep(random.uniform(*AUTO_LINKING_DELAY_SECS))
            if self._shutdown_event.is_set(): return

            self.logger.debug(f"Attempting auto-link for memory {_fmt_id(memory_id)} in WF {_fmt_id(workflow_id)}...")
            
            get_mem_by_id_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_MEMORY_BY_ID)
            source_res = await self._execute_tool_call_internal(
                get_mem_by_id_mcp_name, 
                {"memory_id": memory_id, "include_links": False, "workflow_id": workflow_id}, # Pass workflow_id
                record_action=False
            )
            if not source_res.get("success") or source_res.get("workflow_id") != workflow_id:
                self.logger.warning(f"Auto-link failed for {_fmt_id(memory_id)}: Source mem error or WF mismatch. Resp: {source_res}")
                return
            source_mem = source_res
            query_text = source_mem.get("description", "") or source_mem.get("content", "")[:200]
            if not query_text:
                self.logger.debug(f"Skipping auto-link for {_fmt_id(memory_id)}: No query text.")
                return

            search_base_func = UMS_FUNC_HYBRID_SEARCH if self._find_tool_server(self._get_ums_tool_mcp_name(UMS_FUNC_HYBRID_SEARCH)) else UMS_FUNC_SEARCH_SEMANTIC_MEMORIES
            search_mcp_tool_name = self._get_ums_tool_mcp_name(search_base_func)

            if not self._find_tool_server(search_mcp_tool_name):
                self.logger.warning(f"Skipping auto-link: Tool for {search_base_func} unavailable.")
                return
            
            search_args: Dict[str, Any] = {
                "workflow_id": workflow_id, "query": query_text,
                "limit": self.auto_linking_max_links + 1,
                "threshold": self.auto_linking_threshold, "include_content": False,
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
                if self._shutdown_event.is_set(): break
                target_id = sim_mem_summary.get("memory_id")
                sim_score = sim_mem_summary.get(score_key, 0.0)
                if not target_id or target_id == memory_id: continue

                # No need to fetch target_mem again if we trust workflow_id from search_args
                # and sim_mem_summary usually contains enough for simple link typing.
                link_type = LinkType.RELATED.value # Default
                # Simple link typing based on source and target types from summaries
                source_type_from_summary = source_mem.get("memory_type")
                target_type_from_summary = sim_mem_summary.get("memory_type")
                if source_type_from_summary == MemoryType.INSIGHT.value and target_type_from_summary == MemoryType.FACT.value:
                    link_type = LinkType.SUPPORTS.value
                
                link_args = {
                    "source_memory_id": memory_id, "target_memory_id": target_id,
                    "link_type": link_type, "strength": round(sim_score, 3),
                    "description": f"Auto-link ({link_type})",
                    "workflow_id": workflow_id # Pass workflow_id for the link operation itself
                }
                link_result = await self._execute_tool_call_internal(create_link_mcp_name, link_args, record_action=False)
                if link_result.get("success"):
                    link_count += 1
                    self.logger.debug(f"Auto-linked {_fmt_id(memory_id)} to {_fmt_id(target_id)} ({link_type}, {sim_score:.2f})")
                else:
                    self.logger.warning(f"Failed auto-create link {_fmt_id(memory_id)}->{_fmt_id(target_id)}: {link_result.get('error')}")
                if link_count >= self.auto_linking_max_links: break
                await asyncio.sleep(0.1) # Be nice
        except Exception as e:
            self.logger.warning(f"Error in auto-linking for {_fmt_id(memory_id)} in WF {_fmt_id(workflow_id)}: {e}", exc_info=False)


    async def _execute_tool_call_internal(
        self, tool_name_mcp: str, arguments: Dict[str, Any], record_action: bool = True, planned_dependencies: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        # Force print for this entire function for debugging this specific issue
        def _force_print(*args_print, **kwargs_print):
            print(*args_print, file=sys.stderr, flush=True, **kwargs_print)

        current_base_func_name = self._get_base_function_name(tool_name_mcp)

        _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Entered for tool '{tool_name_mcp}'. Base Func: '{current_base_func_name}'. Args: {str(arguments)[:200]}...")
        
        target_server = self._find_tool_server(tool_name_mcp)
        if not target_server and tool_name_mcp != AGENT_TOOL_UPDATE_PLAN:
            err = f"Tool server unavailable for {tool_name_mcp}"
            self.logger.error(f"AML EXEC_TOOL_INTERNAL: {err}")
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: ERROR - Tool server unavailable for {tool_name_mcp}")
            self.state.last_error_details = {"tool": tool_name_mcp, "error": err, "type": "ServerUnavailable", "status_code": 503}
            return {"success": False, "error": err, "status_code": 503}
        
        final_arguments = arguments.copy()
        current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        current_ctx_id = self.state.context_id
        current_goal_id_for_tool = self.state.current_goal_id

        if (final_arguments.get("workflow_id") is None and current_wf_id and
            current_base_func_name != UMS_FUNC_CREATE_WORKFLOW and 
            current_base_func_name != UMS_FUNC_LIST_WORKFLOWS and
            tool_name_mcp != AGENT_TOOL_UPDATE_PLAN):
            final_arguments["workflow_id"] = current_wf_id
        if final_arguments.get("context_id") is None and current_ctx_id and current_base_func_name in {
            UMS_FUNC_GET_WORKING_MEMORY, UMS_FUNC_OPTIMIZE_WM, UMS_FUNC_AUTO_FOCUS, UMS_FUNC_FOCUS_MEMORY
        }:
            final_arguments["context_id"] = current_ctx_id
        if final_arguments.get("thought_chain_id") is None and self.state.current_thought_chain_id and current_base_func_name == UMS_FUNC_RECORD_THOUGHT:
            final_arguments["thought_chain_id"] = self.state.current_thought_chain_id
        if final_arguments.get("parent_goal_id") is None and current_goal_id_for_tool and current_base_func_name == UMS_FUNC_CREATE_GOAL:
             final_arguments["parent_goal_id"] = current_goal_id_for_tool

        _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Final args for '{tool_name_mcp}': {str(final_arguments)[:200]}...")

        if planned_dependencies:
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Checking prerequisites for '{tool_name_mcp}': {[_fmt_id(d) for d in planned_dependencies]}")
            ok, reason = await self._check_prerequisites(planned_dependencies)
            if not ok:
                err_msg = f"Prerequisites not met for {tool_name_mcp}: {reason}"
                _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: ERROR - {err_msg}")
                self.state.last_error_details = {"tool": tool_name_mcp, "error": err_msg, "type": "DependencyNotMetError", "dependencies": planned_dependencies, "status_code": 412}
                self.state.needs_replan = True
                return {"success": False, "error": err_msg, "status_code": 412}
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Prerequisites met for {tool_name_mcp}.")
        
        if tool_name_mcp == AGENT_TOOL_UPDATE_PLAN:
            try:
                new_plan_data = final_arguments.get("plan", [])
                if not isinstance(new_plan_data, list): raise ValueError("`plan` must be a list.")
                validated_plan = [PlanStep(**p) for p in new_plan_data]
                if self._detect_plan_cycle(validated_plan):
                    err_msg = "Proposed plan contains a dependency cycle."
                    _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (update_plan): ERROR - {err_msg}")
                    self.state.last_error_details = {"tool": tool_name_mcp, "error": err_msg, "type": "PlanValidationError", "proposed_plan": new_plan_data}
                    self.state.needs_replan = True
                    return {"success": False, "error": err_msg}
                self.state.current_plan = validated_plan
                self.state.needs_replan = False
                _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: AGENT_TOOL_UPDATE_PLAN successful.")
                self.state.last_error_details = None # Clear error after successful plan update
                self.state.consecutive_error_count = 0 # Reset error count
                return {"success": True, "message": f"Plan updated with {len(validated_plan)} steps."}
            except (ValidationError, TypeError, ValueError) as e:
                err_msg = f"Failed to validate/apply new plan: {e}"
                _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (update_plan): ERROR - {err_msg}")
                self.state.last_error_details = {"tool": tool_name_mcp, "error": err_msg, "type": "PlanUpdateError", "proposed_plan": final_arguments.get("plan")}
                self.state.consecutive_error_count += 1
                self.state.needs_replan = True
                return {"success": False, "error": err_msg}

        action_id: Optional[str] = None
        should_record_this_action = record_action and current_base_func_name not in self._INTERNAL_OR_META_TOOLS_BASE_NAMES

        if should_record_this_action:
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Recording UMS action start for '{tool_name_mcp}'.")
            action_id = await self._record_action_start_internal(tool_name_mcp, final_arguments, planned_dependencies)
            if not action_id:
                err_msg = f"Failed to record UMS action_start for tool {tool_name_mcp}."
                _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: ERROR - {err_msg}")
                self.state.last_error_details = {"tool": tool_name_mcp, "error": err_msg, "type": "UMSError", "reason": "ActionStartFailed"}
                # Do not set consecutive_error_count here, as last_error_details will trigger heuristic plan update later
                self.state.needs_replan = True # If action start fails, we need to replan
                return {"success": False, "error": err_msg, "status_code": 500}
        
        record_stats = self.state.tool_usage_stats[tool_name_mcp]
        idempotent_base_names = { 
            UMS_FUNC_GET_MEMORY_BY_ID, UMS_FUNC_SEARCH_SEMANTIC_MEMORIES, UMS_FUNC_HYBRID_SEARCH,
            UMS_FUNC_QUERY_MEMORIES, UMS_FUNC_GET_ACTION_DETAILS, UMS_FUNC_LIST_WORKFLOWS,
            UMS_FUNC_COMPUTE_STATS, UMS_FUNC_GET_WORKING_MEMORY, UMS_FUNC_GET_LINKED_MEMORIES,
            UMS_FUNC_GET_ARTIFACTS, UMS_FUNC_GET_ARTIFACT_BY_ID, UMS_FUNC_GET_ACTION_DEPENDENCIES,
            UMS_FUNC_GET_THOUGHT_CHAIN, UMS_FUNC_GET_WORKFLOW_DETAILS, UMS_FUNC_GET_GOAL_DETAILS,
            UMS_FUNC_SUMMARIZE_TEXT, UMS_FUNC_GET_RICH_CONTEXT_PACKAGE, UMS_FUNC_GET_RECENT_ACTIONS
        }
        idempotent = current_base_func_name in idempotent_base_names

        start_ts = time.time()
        ums_direct_payload_dict: Dict[str, Any] = {} 

        try:
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Attempting MCPClient.execute_tool for '{tool_name_mcp}' on server '{target_server}'.")
            async def _do_mcp_call():
                call_args = {k: v for k, v in final_arguments.items() if v is not None}
                mcp_call_result: CallToolResult = await self.mcp_client.execute_tool(target_server, tool_name_mcp, call_args)
                return mcp_call_result

            raw_mcp_tool_result: CallToolResult = await self._with_retries(_do_mcp_call, max_retries=3 if idempotent else 1)
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: MCPClient.execute_tool for '{tool_name_mcp}' RETURNED. Type: {type(raw_mcp_tool_result)}. Value: {str(raw_mcp_tool_result)[:500]}...")
            
            latency_ms = (time.time() - start_ts) * 1000
            record_stats["latency_ms_total"] += latency_ms
            
            if isinstance(raw_mcp_tool_result, CallToolResult):
                if raw_mcp_tool_result.isError:
                    error_content_str = str(raw_mcp_tool_result.content)
                    ums_direct_payload_dict = {"success": False, "error": error_content_str, "status_code": getattr(raw_mcp_tool_result, 'status_code', None)}
                    _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (CallToolResult Error): ums_direct_payload_dict set to error structure.")
                else: 
                    content_from_call = raw_mcp_tool_result.content
                    if isinstance(content_from_call, list) and len(content_from_call) == 1:
                        first_element = content_from_call[0]
                        is_text_content_like = False
                        text_attribute_value = None
                        if hasattr(first_element, 'get') and callable(first_element.get): 
                            if first_element.get("type") == "text" and isinstance(first_element.get("text"), str):
                                is_text_content_like = True; text_attribute_value = first_element.get("text")
                        elif isinstance(first_element, object) and not isinstance(first_element, dict): 
                            if getattr(first_element, 'type', None) == "text" and isinstance(getattr(first_element, 'text', None), str):
                                is_text_content_like = True; text_attribute_value = getattr(first_element, 'text') # noqa: B009
                        
                        if is_text_content_like and text_attribute_value is not None:
                            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (TextContent Parse): Attempting to parse JSON from TextContent: {text_attribute_value[:100]}...")
                            try:
                                parsed_json = json.loads(text_attribute_value)
                                if isinstance(parsed_json, dict):
                                    ums_direct_payload_dict = parsed_json
                                    _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (TextContent Parse): SUCCESS. ums_direct_payload_dict is UMS dict. Keys: {list(ums_direct_payload_dict.keys())}")
                                else: 
                                    ums_direct_payload_dict = {"success": True, "data": parsed_json, "_note": "UMS tool returned non-dict JSON in TextContent."}
                                    _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (TextContent Parse): Parsed non-dict JSON, wrapped in 'data'. Type: {type(parsed_json)}")
                            except json.JSONDecodeError as e:
                                ums_direct_payload_dict = {"success": False, "error": f"Failed to parse UMS result JSON from TextContent: {e}", "_original_raw_result_text": text_attribute_value}
                                _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (TextContent Parse): JSONDecodeError. ums_direct_payload_dict set to error. Error: {e}")
                        else: 
                            if isinstance(content_from_call, dict) and "success" in content_from_call: 
                                ums_direct_payload_dict = content_from_call
                                _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (CallToolResult Direct Dict): Assuming content is UMS payload. Keys: {list(ums_direct_payload_dict.keys())}")
                            else: 
                                ums_direct_payload_dict = {"success": True, "data": content_from_call, "_note": "Unrecognized structure in CallToolResult.content, wrapped in 'data'."}
                                _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (CallToolResult Unhandled Structure): Wrapped content in 'data'. Type: {type(content_from_call)}")
                    elif isinstance(content_from_call, dict) and "success" in content_from_call: 
                        ums_direct_payload_dict = content_from_call
                        _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (CallToolResult is UMS Dict): Content IS UMS payload. Keys: {list(ums_direct_payload_dict.keys())}")
                    else: 
                        ums_direct_payload_dict = {"success": True, "data": content_from_call, "_note": "CallToolResult.content was not List[TextContent] or UMS dict, wrapped in 'data'."}
                        _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL (CallToolResult Unexpected Content): Wrapped content in 'data'. Type: {type(content_from_call)}")
            elif isinstance(raw_mcp_tool_result, dict): 
                 _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: WARNING - MCPClient.execute_tool returned a dict, expected CallToolResult. Processing as raw dict.")
                 ums_direct_payload_dict = raw_mcp_tool_result 
            else: 
                _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: ERROR - Unexpected return type from MCPClient.execute_tool: {type(raw_mcp_tool_result)}. Treating as error.")
                ums_direct_payload_dict = {"success": False, "error": f"Unexpected data type from tool: {type(raw_mcp_tool_result)}", "data_received": str(raw_mcp_tool_result)}
            
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Final 'ums_direct_payload_dict' for '{tool_name_mcp}': Success={ums_direct_payload_dict.get('success')}, Keys: {list(ums_direct_payload_dict.keys())}, Value: {str(ums_direct_payload_dict)[:200]}...")
            
            if ums_direct_payload_dict.get("success"):
                record_stats["success"] += 1
                if current_base_func_name in [UMS_FUNC_STORE_MEMORY, UMS_FUNC_UPDATE_MEMORY] and ums_direct_payload_dict.get("memory_id"):
                    self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=ums_direct_payload_dict["memory_id"])
                elif current_base_func_name == UMS_FUNC_RECORD_ARTIFACT and ums_direct_payload_dict.get("linked_memory_id"):
                    self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=ums_direct_payload_dict["linked_memory_id"])
                elif current_base_func_name == UMS_FUNC_RECORD_THOUGHT and ums_direct_payload_dict.get("linked_memory_id"): 
                    self._start_background_task(AgentMasterLoop._run_auto_linking, memory_id=ums_direct_payload_dict["linked_memory_id"])
                
                mem_ids_to_check_for_promo = set()
                data_for_promo_check = ums_direct_payload_dict
                if isinstance(data_for_promo_check, dict):
                    if current_base_func_name == UMS_FUNC_GET_MEMORY_BY_ID and isinstance(data_for_promo_check.get("memory_id"), str):
                        mem_ids_to_check_for_promo.add(data_for_promo_check["memory_id"])
                    elif current_base_func_name == UMS_FUNC_GET_WORKING_MEMORY and isinstance(data_for_promo_check.get("working_memories"), list):
                        mems_list = data_for_promo_check["working_memories"]
                        mem_ids_to_check_for_promo.update(m.get("memory_id") for m in mems_list[:3] if isinstance(m, dict) and m.get("memory_id"))
                        if data_for_promo_check.get("focal_memory_id"): mem_ids_to_check_for_promo.add(data_for_promo_check["focal_memory_id"])
                    elif current_base_func_name in [UMS_FUNC_QUERY_MEMORIES, UMS_FUNC_HYBRID_SEARCH, UMS_FUNC_SEARCH_SEMANTIC_MEMORIES] and isinstance(data_for_promo_check.get("memories"), list):
                        mems_list = data_for_promo_check["memories"]
                        mem_ids_to_check_for_promo.update(m.get("memory_id") for m in mems_list[:3] if isinstance(m, dict) and m.get("memory_id"))
                for mem_id_chk_promo in filter(None, mem_ids_to_check_for_promo):
                     self._start_background_task(AgentMasterLoop._check_and_trigger_promotion, memory_id=mem_id_chk_promo)
            else: 
                record_stats["failure"] += 1
                error_type = "ToolExecutionError" 
                status_code = ums_direct_payload_dict.get("status_code")
                error_message_from_payload = ums_direct_payload_dict.get("error", "Unknown failure from tool execution.")
                
                if status_code == 412: error_type = "DependencyNotMetError"
                elif status_code == 503: error_type = "ServerUnavailable"
                elif ums_direct_payload_dict.get("error_type") == "ToolInputError": error_type = "InvalidInputError"
                elif ums_direct_payload_dict.get("error_type") == "ToolError": error_type = "ToolInternalError"
                elif "input" in str(error_message_from_payload).lower() or "validation" in str(error_message_from_payload).lower(): error_type = "InvalidInputError"
                elif "timeout" in str(error_message_from_payload).lower(): error_type = "NetworkError"
                elif current_base_func_name in [UMS_FUNC_CREATE_GOAL, UMS_FUNC_UPDATE_GOAL_STATUS] and \
                     ("not found" in str(error_message_from_payload).lower() or "invalid" in str(error_message_from_payload).lower()):
                    error_type = "GoalManagementError"
                
                self.state.last_error_details = {
                    "tool": tool_name_mcp, "args": arguments, "error": error_message_from_payload,
                    "status_code": status_code, "type": error_type,
                    "ums_error_type": ums_direct_payload_dict.get("error_type"),
                }
            
            summary_text_for_log = ""
            if ums_direct_payload_dict.get("success"):
                summary_keys_log = ["summary", "message", "memory_id", "action_id", "artifact_id", "link_id", "chain_id", "state_id", "report", "visualization", "goal_id", "workflow_id"]
                payload_for_summary_log = ums_direct_payload_dict
                
                if isinstance(payload_for_summary_log, dict):
                    for k_log in summary_keys_log:
                        if k_log in payload_for_summary_log and payload_for_summary_log[k_log] is not None:
                            val_str_log = str(payload_for_summary_log[k_log])
                            summary_text_for_log = f"{k_log}: {_fmt_id(val_str_log) if 'id' in k_log.lower() else val_str_log}"
                            break
                    if not summary_text_for_log: 
                        generic_parts_log = [f"{k_s}={_fmt_id(str(v_s)) if 'id' in k_s.lower() else str(v_s)[:20]}"
                                             for k_s, v_s in payload_for_summary_log.items()
                                             if k_s not in ['success', 'processing_time', '_note', 'data_received', 'error_type', '_original_raw_result_text'] and v_s is not None]
                        summary_text_for_log = f"Success. Data: {', '.join(generic_parts_log)}" if generic_parts_log else "Success (No further summary data)."
                else: 
                    summary_text_for_log = f"Success (Unexpected payload type: {type(payload_for_summary_log)})."
            else: 
                err_type_log = self.state.last_error_details.get("type", "Unknown") if self.state.last_error_details else "Unknown"
                err_msg_log = str(ums_direct_payload_dict.get("error", "Unknown Error"))[:100]
                summary_text_for_log = f"Failed ({err_type_log}): {err_msg_log}"
            
            if ums_direct_payload_dict.get("status_code"): summary_text_for_log += f" (Code: {ums_direct_payload_dict['status_code']})"
            self.state.last_action_summary = f"{tool_name_mcp} -> {summary_text_for_log}"
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Last Action Summary: {self.state.last_action_summary}")

        except (ToolError, ToolInputError) as e_tool_err:
            err_str_tool = str(e_tool_err)
            status_code_tool = getattr(e_tool_err, "status_code", None)
            error_type_tool = "InvalidInputError" if isinstance(e_tool_err, ToolInputError) else "ToolInternalError"
            if status_code_tool == 412: error_type_tool = "DependencyNotMetError"
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: CAUGHT ToolError/InputError for '{tool_name_mcp}': {error_type_tool} - {err_str_tool}")
            ums_direct_payload_dict = {"success": False, "error": err_str_tool, "status_code": status_code_tool, "error_type": error_type_tool}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name_mcp, "args": arguments, "error": err_str_tool, "type": error_type_tool, "status_code": status_code_tool}
            self.state.last_action_summary = f"{tool_name_mcp} -> Failed ({error_type_tool}): {err_str_tool[:100]}"
        except APIConnectionError as e_api_conn:
            err_str_api_conn = f"LLM API Connection Error: {e_api_conn}"
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: CAUGHT APIConnectionError for '{tool_name_mcp}': {err_str_api_conn}")
            ums_direct_payload_dict = {"success": False, "error": err_str_api_conn, "error_type": "NetworkError"}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name_mcp, "args": arguments, "error": err_str_api_conn, "type": "NetworkError"}
            self.state.last_action_summary = f"{tool_name_mcp} -> Failed: NetworkError (LLM related)"
        except RateLimitError as e_rl:
            err_str_rl = f"LLM Rate Limit Error: {e_rl}"
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: CAUGHT RateLimitError for '{tool_name_mcp}': {err_str_rl}")
            ums_direct_payload_dict = {"success": False, "error": err_str_rl, "error_type": "APILimitError"}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name_mcp, "args": arguments, "error": err_str_rl, "type": "APILimitError"}
            self.state.last_action_summary = f"{tool_name_mcp} -> Failed: APILimitError"
        except APIStatusError as e_stat:
            err_str_stat = f"LLM API Error {e_stat.status_code}: {e_stat.message}"
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: CAUGHT APIStatusError for '{tool_name_mcp}': {err_str_stat}")
            ums_direct_payload_dict = {"success": False, "error": err_str_stat, "status_code": e_stat.status_code, "error_type": "APIError"}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name_mcp, "args": arguments, "error": err_str_stat, "type": "APIError", "status_code": e_stat.status_code}
            self.state.last_action_summary = f"{tool_name_mcp} -> Failed: APIError ({e_stat.status_code})"
        except httpx.RequestError as e_httpx:
            err_str_httpx = f"HTTPX Request Error: {e_httpx}"
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: CAUGHT HTTPX RequestError for '{tool_name_mcp}': {err_str_httpx}")
            ums_direct_payload_dict = {"success": False, "error": err_str_httpx, "error_type": "NetworkError"}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name_mcp, "args": arguments, "error": err_str_httpx, "type": "NetworkError"}
            self.state.last_action_summary = f"{tool_name_mcp} -> Failed: HTTPX RequestError"
        except asyncio.TimeoutError as e_timeout:
            err_str_timeout = f"Operation timed out: {e_timeout}"
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: CAUGHT TimeoutError for '{tool_name_mcp}': {err_str_timeout}")
            ums_direct_payload_dict = {"success": False, "error": err_str_timeout, "error_type": "TimeoutError"}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name_mcp, "args": arguments, "error": err_str_timeout, "type": "TimeoutError"}
            self.state.last_action_summary = f"{tool_name_mcp} -> Failed: Timeout"
        except asyncio.CancelledError:
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: CAUGHT CancelledError for tool '{tool_name_mcp}'")
            self.state.last_action_summary = f"{tool_name_mcp} -> Cancelled"
            raise 
        except Exception as e_unhandled:
            err_str_unhandled = str(e_unhandled)
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: CAUGHT UNEXPECTED EXCEPTION for tool '{tool_name_mcp}': {type(e_unhandled).__name__} - {e_unhandled}")
            self.logger.error(f"AML EXEC_TOOL_INTERNAL: Unexpected Error executing {tool_name_mcp}: {err_str_unhandled}", exc_info=True)
            ums_direct_payload_dict = {"success": False, "error": f"Unexpected error during tool execution: {err_str_unhandled}", "error_type": "UnexpectedExecutionError"}
            record_stats["failure"] += 1
            self.state.last_error_details = {"tool": tool_name_mcp, "args": arguments, "error": err_str_unhandled, "type": "UnexpectedExecutionError"}
            self.state.last_action_summary = f"{tool_name_mcp} -> Failed: Unexpected error."
        
        # This block is now outside the try...except for MCP tool call
        if action_id and should_record_this_action:
            _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Recording UMS action completion for action ID {_fmt_id(action_id)} of tool '{tool_name_mcp}'.")
            # ums_direct_payload_dict here will be the result from the try block or one of the except blocks
            await self._record_action_completion_internal(action_id, ums_direct_payload_dict) 
        
        _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Calling side effects for base_func_name='{current_base_func_name}'. Current WF ID (before side effects): {_fmt_id(self.state.workflow_id)}")
        await self._handle_workflow_and_goal_side_effects(current_base_func_name, final_arguments, ums_direct_payload_dict)
        _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: After calling side effects for base_func_name='{current_base_func_name}'. Current WF ID (after side effects): {_fmt_id(self.state.workflow_id)}")
        
        _force_print(f"FORCE_PRINT EXEC_TOOL_INTERNAL: Final 'ums_direct_payload_dict' for '{tool_name_mcp}' (after side-effect error check): Success={ums_direct_payload_dict.get('success') if isinstance(ums_direct_payload_dict, dict) else 'N/A'}")
        return ums_direct_payload_dict

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
            if self._shutdown_event.is_set(): return

            self.logger.debug(f"Checking promo potential for memory {_fmt_id(memory_id)} in WF {_fmt_id(workflow_id)}...")
            # Pass workflow_id to the UMS tool if it requires it (some UMS tools might infer from memory_id)
            promo_args = {"memory_id": memory_id}
            # if workflow_id: promo_args["workflow_id"] = workflow_id # promote_memory_level doesn't take workflow_id
            
            promo_res = await self._execute_tool_call_internal(promote_mem_mcp_name, promo_args, record_action=False)
            if promo_res.get("success"):
                if promo_res.get("promoted"):
                    self.logger.info(f"⬆️ Memory {_fmt_id(memory_id)} promoted from {promo_res.get('previous_level')} to {promo_res.get('new_level')}.")
                else:
                    self.logger.debug(f"Memory {_fmt_id(memory_id)} not promoted: {promo_res.get('reason')}")
            else:
                self.logger.warning(f"Promo check tool failed for {_fmt_id(memory_id)}: {promo_res.get('error')}")
        except Exception as e:
            self.logger.warning(f"Error in promo check task for {_fmt_id(memory_id)}: {e}", exc_info=False)


    async def _write_temp_workflow_id(self, workflow_id: Optional[str]):
            try:
                AGENT_LOOP_TEMP_DIR.mkdir(parents=True, exist_ok=True)
                if workflow_id:
                    async with aiofiles.open(TEMP_WORKFLOW_ID_FILE, "w", encoding="utf-8") as f:
                        await f.write(workflow_id)
                    self.logger.info(f"AML TEMP_WF_ID: Wrote active workflow ID {_fmt_id(workflow_id)} to {TEMP_WORKFLOW_ID_FILE}")
                elif TEMP_WORKFLOW_ID_FILE.exists(): 
                    TEMP_WORKFLOW_ID_FILE.unlink()
                    self.logger.info(f"AML TEMP_WF_ID: Cleared temp workflow ID file {TEMP_WORKFLOW_ID_FILE} as workflow ID is None.")
            except Exception as e:
                self.logger.error(f"AML TEMP_WF_ID: Error writing/deleting temp workflow ID file: {e}", exc_info=True)

    async def _read_temp_workflow_id(self) -> Optional[str]:
        try:
            if TEMP_WORKFLOW_ID_FILE.exists():
                async with aiofiles.open(TEMP_WORKFLOW_ID_FILE, "r", encoding="utf-8") as f:
                    wf_id = (await f.read()).strip()
                    if wf_id:
                        self.logger.info(f"AML TEMP_WF_ID: Read workflow ID {_fmt_id(wf_id)} from {TEMP_WORKFLOW_ID_FILE}")
                        return wf_id
                    else: 
                        self.logger.warning(f"AML TEMP_WF_ID: Temp file {TEMP_WORKFLOW_ID_FILE} was empty.")
                        TEMP_WORKFLOW_ID_FILE.unlink() 
                        return None
            return None
        except Exception as e:
            self.logger.error(f"AML TEMP_WF_ID: Error reading temp workflow ID file: {e}", exc_info=True)
            return None


    async def _handle_workflow_and_goal_side_effects(self, base_tool_func_name: str, arguments: Dict, result_content: Dict):
        """
        Handles agent state changes triggered by specific tool outcomes, particularly
        workflow creation and goal management. Ensures UMS thoughts are created for UMS goals
        to facilitate correct cognitive state saving.

        Args:
            base_tool_func_name: The canonical base function name (e.g., "create_workflow").
            arguments: The arguments passed to the original tool call.
            result_content: The direct dictionary returned by the UMS tool (or an error dict).
        """
        is_successful_call = isinstance(result_content, dict) and result_content.get('success')

        # Use a consistent logging prefix for this method
        log_prefix = f"AML_SIDE_EFFECTS ({base_tool_func_name}, Success: {is_successful_call})"
        self.logger.info(
            f"{log_prefix}: Entered. Current WF (before): {_fmt_id(self.state.workflow_id)}, "
            f"Current UMS Goal (before): {_fmt_id(self.state.current_goal_id)}, "
            f"NeedsReplan (before): {self.state.needs_replan}"
        )
        
        current_wf_id_before_effect = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        current_goal_id_before_effect = self.state.current_goal_id

        if base_tool_func_name == UMS_FUNC_CREATE_WORKFLOW and is_successful_call:
            self.logger.debug(f"{log_prefix}: Matched create_workflow success block.")
            
            new_wf_id = result_content.get("workflow_id")
            primary_chain_id = result_content.get("primary_thought_chain_id")
            parent_wf_id_arg = arguments.get("parent_workflow_id")

            if not (new_wf_id and isinstance(new_wf_id, str)):
                self.logger.error(f"{log_prefix}: CRITICAL - UMS Tool '{base_tool_func_name}' succeeded but result did NOT contain a valid 'workflow_id' string. Result: {str(result_content)[:500]}")
                self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_WORKFLOW), "error": "Internal state error: workflow_id not set after successful UMS create_workflow.", "type": "AgentError"}
                self.state.needs_replan = True
                return # Critical failure, cannot proceed with side effects

            self.logger.info(f"{log_prefix}: Successfully received new workflow_id: {_fmt_id(new_wf_id)}")
            
            self.state.workflow_id = new_wf_id
            self.state.context_id = new_wf_id # Assume context_id is same as workflow_id initially
            await self._write_temp_workflow_id(new_wf_id)

            is_sub_workflow = parent_wf_id_arg and parent_wf_id_arg in self.state.workflow_stack
            if is_sub_workflow:
                self.state.workflow_stack.append(new_wf_id)
                self.logger.info(f"{log_prefix}: Pushed sub-workflow {_fmt_id(new_wf_id)} to stack. Depth: {len(self.state.workflow_stack)}")
            else:
                self.state.workflow_stack = [new_wf_id]
                self.logger.info(f"{log_prefix}: Set new root workflow {_fmt_id(new_wf_id)} on stack.")
            
            self.state.current_thought_chain_id = primary_chain_id
            self.state.goal_stack = [] 
            self.state.current_goal_id = None 
            
            self.logger.info(
                f"{log_prefix}: Agent state updated: WF ID={_fmt_id(self.state.workflow_id)}, "
                f"Context ID={_fmt_id(self.state.context_id)}, Chain={_fmt_id(self.state.current_thought_chain_id)}, "
                f"Goal stack reset."
            )

            # Establish the root UMS Goal for this new workflow
            wf_goal_desc_from_result = result_content.get("goal")
            wf_goal_desc_from_args = arguments.get("goal", f"Achieve objectives for workflow {new_wf_id[:8]}")
            final_wf_goal_desc = wf_goal_desc_from_result if wf_goal_desc_from_result is not None else wf_goal_desc_from_args
            wf_title = result_content.get("title", f"Workflow {new_wf_id[:8]}")
            root_goal_description_for_ums = final_wf_goal_desc if final_wf_goal_desc else f"Fulfill workflow: {wf_title}"
            
            self.logger.info(f"{log_prefix}: Preparing to create UMS root goal for new WF '{_fmt_id(self.state.workflow_id)}' with desc: '{root_goal_description_for_ums[:50]}...'")
            
            created_ums_goal_obj: Optional[Dict[str, Any]] = None
            root_goal_thought_id_for_state: Optional[str] = None
            root_goal_linked_memory_id_for_state: Optional[str] = None

            create_goal_mcp_tool_name = self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL)
            if self._find_tool_server(create_goal_mcp_tool_name):
                try:
                    goal_creation_args = {
                        "workflow_id": self.state.workflow_id,
                        "description": root_goal_description_for_ums,
                        "title": f"Root Goal for workflow: {wf_title}",
                        "parent_goal_id": None,
                        "initial_status": GoalStatus.ACTIVE.value
                    }
                    goal_res_payload = await self._execute_tool_call_internal(
                        create_goal_mcp_tool_name, goal_creation_args, record_action=False
                    )
                    
                    if isinstance(goal_res_payload, dict) and goal_res_payload.get("success"):
                        temp_goal_obj = goal_res_payload.get("goal")
                        if isinstance(temp_goal_obj, dict) and temp_goal_obj.get("goal_id"):
                            created_ums_goal_obj = temp_goal_obj # Valid UMS goal object
                            self.state.goal_stack = [created_ums_goal_obj]
                            self.state.current_goal_id = created_ums_goal_obj.get("goal_id")
                            self.logger.info(f"{log_prefix}: Successfully created UMS root goal {_fmt_id(self.state.current_goal_id)}.")
                        else:
                            self.logger.error(f"{log_prefix}: UMS create_goal tool returned success, but 'goal' object or 'goal_id' is missing/invalid. Payload: {str(goal_res_payload)[:300]}")
                    
                    if not created_ums_goal_obj: # If UMS goal creation failed or payload was bad
                        error_msg_goal_create = goal_res_payload.get('error', 'UMS create_goal did not return valid goal data') if isinstance(goal_res_payload, dict) else "Unknown error from create_goal internal call"
                        self.logger.error(f"{log_prefix} CRITICAL: Failed to create UMS root goal. Error: {error_msg_goal_create}. Full Response: {str(goal_res_payload)[:300]}")
                        self.state.last_error_details = {"tool": create_goal_mcp_tool_name, "args_sent": goal_creation_args, "error": f"Failed to establish root UMS goal: {error_msg_goal_create}", "type": "GoalManagementError"}
                        self.state.needs_replan = True

                except Exception as goal_err:
                   self.logger.error(f"{log_prefix} CRITICAL: Exception creating UMS root goal: {goal_err}", exc_info=True)
                   self.state.last_error_details = {"tool": create_goal_mcp_tool_name, "error": f"Exception establishing root UMS goal: {goal_err}", "type": "GoalManagementError"}
                   self.state.needs_replan = True
            else:
                self.logger.error(f"{log_prefix} CRITICAL: Tool for '{UMS_FUNC_CREATE_GOAL}' unavailable. Cannot create root UMS goal.")
                self.state.last_error_details = {"tool": create_goal_mcp_tool_name, "error": "Tool unavailable", "type": "ToolUnavailable"}
                self.state.needs_replan = True

            # If UMS Goal was successfully created, now create its UMS Thought representation
            if created_ums_goal_obj and self.state.current_goal_id:
                root_goal_thought_id_for_state: Optional[str] = None
                root_goal_linked_memory_id_for_state: Optional[str] = None # This is for focus_area_ids
                
                ums_record_thought_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_THOUGHT)
                if self._find_tool_server(ums_record_thought_mcp_name):
                    thought_for_root_goal_args = {
                            "workflow_id": self.state.workflow_id,
                            "content": created_ums_goal_obj.get("description", f"Root operational goal for workflow {self.state.workflow_id[:8]}"),
                            "thought_type": ThoughtType.GOAL.value,
                        }
                    self.logger.info(f"{log_prefix}: Recording UMS thought for new root UMS goal {_fmt_id(self.state.current_goal_id)}.")
                    thought_res = await self._execute_tool_call_internal(
                            ums_record_thought_mcp_name, thought_for_root_goal_args, record_action=False
                    )
                    if thought_res.get("success"):
                        root_goal_thought_id_for_state = thought_res.get("thought_id")
                        root_goal_linked_memory_id_for_state = thought_res.get("linked_memory_id")
                        self.logger.info(f"{log_prefix}: Recorded UMS thought {_fmt_id(root_goal_thought_id_for_state)} (LinkedMem: {_fmt_id(root_goal_linked_memory_id_for_state)}) for root UMS goal {_fmt_id(self.state.current_goal_id)}.")
                    else:
                        self.logger.warning(f"{log_prefix}: Failed to record UMS thought for root UMS goal {_fmt_id(self.state.current_goal_id)}. Error: {thought_res.get('error')}")
                        if not self.state.last_error_details: # Don't overwrite a more primary error
                            self.state.last_error_details = {"tool": ums_record_thought_mcp_name, "error": f"Failed to record UMS thought for root goal: {thought_res.get('error')}", "type": "InternalStateSetupError"}
                        self.state.needs_replan = True # Critical for state saving
                else:
                    self.logger.warning(f"{log_prefix}: Tool for '{UMS_FUNC_RECORD_THOUGHT}' unavailable. Cannot record UMS thought for root UMS goal.")
                    if not self.state.last_error_details:
                        self.state.last_error_details = {"tool": ums_record_thought_mcp_name, "error": "Tool unavailable", "type": "ToolUnavailable"}
                    self.state.needs_replan = True

            # Now, attempt to save the initial cognitive state
            if self.state.current_goal_id and root_goal_thought_id_for_state and not self.state.needs_replan: # Only if prior steps were okay
                ums_save_state_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_SAVE_COGNITIVE_STATE)
                if self._find_tool_server(ums_save_state_mcp_name):
                    self.logger.info(f"{log_prefix}: Saving initial cognitive state for new workflow {_fmt_id(self.state.workflow_id)}...")
                    save_state_args = {
                        "workflow_id": self.state.workflow_id,
                        "title": f"Initial cognitive state for workflow '{self.state.workflow_id[:8]}'",
                        "working_memory_ids": [], 
                        "focus_area_ids": [root_goal_linked_memory_id_for_state] if root_goal_linked_memory_id_for_state else [],
                        "current_goal_thought_ids": [root_goal_thought_id_for_state] # Must be present if we reached here
                    }
                    save_res = await self._execute_tool_call_internal(ums_save_state_mcp_name, save_state_args, record_action=False)
                    if save_res.get("success"):
                        self.logger.info(f"{log_prefix}: Initial cognitive state saved (State ID: {_fmt_id(save_res.get('state_id'))}).")
                    else:
                        self.logger.warning(f"{log_prefix}: Failed to save initial cognitive state: {save_res.get('error')}")
                        self.state.last_error_details = {"tool": ums_save_state_mcp_name, "args_sent": save_state_args, "error": f"Failed to save initial UMS cognitive state: {save_res.get('error')}", "type": "CognitiveStateError"}
                        self.state.needs_replan = True
                else:
                    self.logger.warning(f"{log_prefix}: Tool '{UMS_FUNC_SAVE_COGNITIVE_STATE}' not available. Cannot save initial cognitive state.")
                    self.state.last_error_details = {"tool": ums_save_state_mcp_name, "error": "Tool unavailable", "type": "ToolUnavailable"}
                    self.state.needs_replan = True
            elif self.state.current_goal_id and not root_goal_thought_id_for_state:
                self.logger.error(f"{log_prefix}: Cannot save initial state because UMS thought for root UMS goal was not created.")
                self.state.needs_replan = True # Implicitly, as setup is incomplete.

            # Final plan and error state update based on overall success of this block
            if not self.state.needs_replan and self.state.current_goal_id and root_goal_thought_id_for_state:
               current_goal_desc_for_plan = created_ums_goal_obj.get("description", wf_title) if created_ums_goal_obj else wf_title
               plan_desc = f"Assess current UMS goal '{current_goal_desc_for_plan[:50]}...' ({_fmt_id(self.state.current_goal_id)}) and associated thought ({_fmt_id(root_goal_thought_id_for_state)}) for workflow '{wf_title}' and formulate next steps."
               self.state.current_plan = [PlanStep(description=plan_desc)]
               self.state.last_error_details = None 
               self.state.consecutive_error_count = 0
               self.logger.info(f"{log_prefix}: Initial plan set for new workflow. NeedsReplan: {self.state.needs_replan}")
            else: # Some part of the setup failed (UMS goal, UMS thought, or state save)
               plan_desc = f"ERROR: Failed to fully establish initial state for workflow '{wf_title}'. Review errors and address."
               self.state.current_plan = [PlanStep(description=plan_desc, status="failed")]
               self.state.needs_replan = True # Ensure replan is set if any step failed.
               self.logger.error(f"{log_prefix}: Plan set to error state due to failed initial setup. Last error: {self.state.last_error_details}")
           
            self.logger.info(f"{log_prefix}: END of new_wf_id processing. WF State: id={_fmt_id(self.state.workflow_id)}, goal_id={_fmt_id(self.state.current_goal_id)}, plan_step_desc='{self.state.current_plan[0].description[:50] if self.state.current_plan else 'N/A'}', needs_replan={self.state.needs_replan}")

        elif base_tool_func_name == UMS_FUNC_CREATE_GOAL and is_successful_call:
            self.logger.info(f"{log_prefix} (LLM invoked): START. Current UMS Goal (before): {_fmt_id(self.state.current_goal_id)}")
            
            created_ums_goal_obj_from_llm_req = result_content.get("goal") 
            
            if isinstance(created_ums_goal_obj_from_llm_req, dict) and created_ums_goal_obj_from_llm_req.get("goal_id"):
                if created_ums_goal_obj_from_llm_req.get("workflow_id") != self.state.workflow_id:
                    self.logger.error(
                        f"{log_prefix}: LLM created UMS goal {_fmt_id(created_ums_goal_obj_from_llm_req.get('goal_id'))} for workflow "
                        f"{_fmt_id(created_ums_goal_obj_from_llm_req.get('workflow_id'))}, but current agent workflow is "
                        f"{_fmt_id(self.state.workflow_id)}. UMS Goal will not be added to local stack."
                    )
                    self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL), "error": "LLM created UMS goal for an incorrect workflow.", "type": "GoalManagementError"}
                    self.state.needs_replan = True
                else:
                    self.state.goal_stack.append(created_ums_goal_obj_from_llm_req) # Add the full UMS goal object
                    self.state.current_goal_id = created_ums_goal_obj_from_llm_req["goal_id"] # Set current UMS goal ID
                    self.logger.info(f"📌 {log_prefix}: Pushed new UMS goal {_fmt_id(self.state.current_goal_id)} to local stack: '{created_ums_goal_obj_from_llm_req.get('description', '')[:50]}...'. Stack depth: {len(self.state.goal_stack)}")

                    # Record a UMS Thought for this LLM-created UMS Goal
                    newly_created_ums_goal_id_llm = created_ums_goal_obj_from_llm_req["goal_id"]
                    newly_created_ums_goal_desc_llm = created_ums_goal_obj_from_llm_req.get("description", f"Goal {newly_created_ums_goal_id_llm[:8]}")
                    
                    ums_record_thought_mcp_name_llm = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_THOUGHT)
                    if self._find_tool_server(ums_record_thought_mcp_name_llm):
                        thought_for_llm_goal_args = {
                            "workflow_id": self.state.workflow_id,
                            "content": newly_created_ums_goal_desc_llm,
                            "thought_type": ThoughtType.GOAL.value,
                        }
                        self.logger.info(f"{log_prefix}: Recording UMS thought for LLM-created UMS goal {_fmt_id(newly_created_ums_goal_id_llm)}.")
                        thought_res_llm_goal = await self._execute_tool_call_internal(
                            ums_record_thought_mcp_name_llm, thought_for_llm_goal_args, record_action=False
                        )
                        if thought_res_llm_goal.get("success"):
                            self.logger.info(f"{log_prefix}: Recorded UMS thought {_fmt_id(thought_res_llm_goal.get('thought_id'))} (LinkedMem: {_fmt_id(thought_res_llm_goal.get('linked_memory_id'))}) for LLM-created UMS goal.")
                        else:
                            self.logger.warning(f"{log_prefix}: Failed to record UMS thought for LLM-created UMS goal {_fmt_id(newly_created_ums_goal_id_llm)}. Error: {thought_res_llm_goal.get('error')}")
                            # This is less critical than root goal setup, but still log.
                            # No self.state.last_error_details here, as the main UMS_CREATE_GOAL succeeded.
                    else:
                        self.logger.warning(f"{log_prefix}: Tool for '{UMS_FUNC_RECORD_THOUGHT}' unavailable. Cannot record UMS thought for LLM-created UMS goal.")

                    self.state.needs_replan = True 
                    plan_desc = f"Start new UMS sub-goal: '{created_ums_goal_obj_from_llm_req.get('description', '')[:50]}...' ({_fmt_id(self.state.current_goal_id)})"
                    self.state.current_plan = [PlanStep(description=plan_desc)]
                    self.state.last_error_details = None 
                    self.state.consecutive_error_count = 0
            else:
                self.logger.warning(f"{log_prefix}: UMS Tool '{UMS_FUNC_CREATE_GOAL}' called by LLM succeeded but did not return valid 'goal' object or 'goal_id' in its payload: {str(result_content)[:300]}")
                self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_CREATE_GOAL), "error": "UMS create_goal (called by LLM) returned invalid goal data.", "type": "GoalManagementError"}
                self.state.needs_replan = True
            self.logger.info(f"{log_prefix} (LLM invoked): END. Current UMS Goal: {_fmt_id(self.state.current_goal_id)}, Stack Depth: {len(self.state.goal_stack)}, needs_replan={self.state.needs_replan}")

        elif base_tool_func_name == UMS_FUNC_UPDATE_GOAL_STATUS and is_successful_call:
            self.logger.info(f"{log_prefix}: START. UMS Goal marked: {_fmt_id(arguments.get('goal_id'))}, New Status in UMS: {arguments.get('status')}")
            
            goal_id_marked_in_ums = result_content.get("updated_goal_details", {}).get("goal_id")
            new_status_in_ums_str = result_content.get("updated_goal_details", {}).get("status")
            
            if not goal_id_marked_in_ums or not new_status_in_ums_str:
                self.logger.error(f"{log_prefix}: UMS Tool for {UMS_FUNC_UPDATE_GOAL_STATUS} returned success, but 'updated_goal_details' or its fields are missing. Aborting. Response: {str(result_content)[:300]}")
                self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_GOAL_STATUS), "error": "UMS update_goal_status returned incomplete data.", "type": "GoalManagementError"}
                self.state.needs_replan = True
                return

            try:
                new_status_in_ums = GoalStatus(new_status_in_ums_str.lower()) 
            except ValueError:
                self.logger.error(f"{log_prefix}: Invalid status '{new_status_in_ums_str}' in UMS response for {UMS_FUNC_UPDATE_GOAL_STATUS}. Aborting.")
                self.state.last_error_details = {"tool": self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_GOAL_STATUS), "error": f"UMS returned invalid goal status '{new_status_in_ums_str}'.", "type": "GoalManagementError"}
                self.state.needs_replan = True
                return

            updated_goal_details_from_ums = result_content.get("updated_goal_details") 
            parent_goal_id_from_ums = result_content.get("parent_goal_id") 
            is_root_finished_from_ums = result_content.get("is_root_finished", False) 

            goal_found_in_local_stack_and_updated = False
            for i, local_goal_dict in enumerate(self.state.goal_stack):
                if isinstance(local_goal_dict, dict) and local_goal_dict.get("goal_id") == goal_id_marked_in_ums:
                    self.state.goal_stack[i] = updated_goal_details_from_ums 
                    goal_found_in_local_stack_and_updated = True
                    self.logger.debug(f"{log_prefix}: Updated goal {_fmt_id(goal_id_marked_in_ums)} in local stack. New status: {new_status_in_ums.value}")
                    break
            if not goal_found_in_local_stack_and_updated:
                self.logger.warning(f"{log_prefix}: UMS Goal {_fmt_id(goal_id_marked_in_ums)} (marked {new_status_in_ums.value} in UMS) not found in current agent goal stack ({len(self.state.goal_stack)} items). Agent stack might be out of sync.")
                if goal_id_marked_in_ums == self.state.current_goal_id:
                    # Attempt to recover the stack if the current goal was the one marked
                    recovered_stack = await self._fetch_goal_stack_from_ums(parent_goal_id_from_ums)
                    if recovered_stack: # If recovery successful
                         self.state.goal_stack = recovered_stack
                         self.logger.info(f"{log_prefix}: Recovered goal stack from UMS based on parent_goal_id {_fmt_id(parent_goal_id_from_ums)}.")
                    elif parent_goal_id_from_ums is None: # Current was root and stack recovery failed or parent was None
                        self.state.goal_stack = []
                        self.logger.info(f"{log_prefix}: Current root goal {_fmt_id(goal_id_marked_in_ums)} marked, stack now empty.")
                    else: # Current was not root, but stack recovery failed. This is a more problematic state.
                        self.state.goal_stack = [] # Clear stack to reflect uncertainty
                        self.logger.warning(f"{log_prefix}: Failed to recover stack for parent {_fmt_id(parent_goal_id_from_ums)}. Local stack cleared.")


            is_terminal_status = new_status_in_ums in [GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.ABANDONED]
            if goal_id_marked_in_ums == self.state.current_goal_id and is_terminal_status:
                self.logger.info(f"{log_prefix}: Current active UMS goal {_fmt_id(self.state.current_goal_id)} reached terminal state '{new_status_in_ums.value}'.")
                self.state.current_goal_id = parent_goal_id_from_ums # Shift focus to parent UMS goal
                
                # Rebuild local stack based on the new current_goal_id (the parent)
                if self.state.current_goal_id: 
                    self.state.goal_stack = await self._fetch_goal_stack_from_ums(self.state.current_goal_id)
                    if not self.state.goal_stack: 
                        self.logger.warning(f"{log_prefix}: Failed to fetch UMS stack for new current_goal_id '{_fmt_id(self.state.current_goal_id)}'. Clearing local stack.")
                        self.state.current_goal_id = None # No valid parent stack
                else: # No parent_goal_id from UMS, means we were at a root goal
                    self.state.goal_stack = [] 
                    self.logger.info(f"{log_prefix}: UMS root goal {_fmt_id(goal_id_marked_in_ums)} reached terminal state. Local goal stack cleared.")
                
                self.logger.info(f"{log_prefix}: Agent focus shifted. New current UMS goal: '{_fmt_id(self.state.current_goal_id) if self.state.current_goal_id else 'Overall Workflow Goal'}'. Local stack depth: {len(self.state.goal_stack)}")

                if is_root_finished_from_ums: 
                    self.logger.info(f"{log_prefix}: UMS indicated a root UMS goal {_fmt_id(goal_id_marked_in_ums)} was terminally finished. Overall workflow goal presumed finished with status: {new_status_in_ums.value}.")
                    self.state.goal_achieved_flag = (new_status_in_ums == GoalStatus.COMPLETED)                     
                    update_wf_status_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_WORKFLOW_STATUS)
                    if self.state.workflow_id and self._find_tool_server(update_wf_status_mcp_name):
                        final_wf_status = WorkflowStatus.COMPLETED.value if self.state.goal_achieved_flag else WorkflowStatus.FAILED.value
                        await self._execute_tool_call_internal(
                            update_wf_status_mcp_name,
                            {"workflow_id": self.state.workflow_id, "status": final_wf_status, "completion_message": f"Overall root UMS goal '{_fmt_id(goal_id_marked_in_ums)}' marked '{new_status_in_ums.value}'. Workflow concluded."},
                            record_action=False, 
                        )
                    self.state.current_plan = [] 
                elif self.state.current_goal_id: # Shifted to a parent UMS goal
                    self.state.needs_replan = True
                    current_goal_desc_for_plan = "Unknown UMS Goal"
                    for g_dict_plan in self.state.goal_stack: 
                        if isinstance(g_dict_plan, dict) and g_dict_plan.get("goal_id") == self.state.current_goal_id:
                            current_goal_desc_for_plan = g_dict_plan.get("description", "Unknown UMS Goal")[:50]
                            break
                    plan_desc = f"Returned from UMS sub-goal {_fmt_id(goal_id_marked_in_ums)} (status: {new_status_in_ums.value}). Re-assess current UMS goal: '{current_goal_desc_for_plan}...' ({_fmt_id(self.state.current_goal_id)})."
                    self.state.current_plan = [PlanStep(description=plan_desc)]
                elif not self.state.current_goal_id and not is_root_finished_from_ums: # No parent, but UMS didn't say root finished
                    self.logger.info(f"{log_prefix}: Completed UMS root goal {_fmt_id(goal_id_marked_in_ums)}. UMS tool gave no parent_id and did not flag as root_finished. Assuming overall workflow might be done or needs re-evaluation based on its main goal.")
                    self.state.goal_achieved_flag = (new_status_in_ums == GoalStatus.COMPLETED) 
                    self.state.needs_replan = True 
                    self.state.current_plan = [PlanStep(description=f"Completed UMS root goal {_fmt_id(goal_id_marked_in_ums)}. Re-evaluating overall workflow objectives.")]
                
                self.state.last_error_details = None 
                self.state.consecutive_error_count = 0
            elif goal_found_in_local_stack_and_updated: 
                 self.logger.info(f"{log_prefix}: UMS Goal '{_fmt_id(goal_id_marked_in_ums)}' (not current agent focus) updated to '{new_status_in_ums.value}'. Local stack view updated.")
            
            self.logger.info(f"{log_prefix}: END. Current UMS Goal: {_fmt_id(self.state.current_goal_id)}, Stack Depth: {len(self.state.goal_stack)}, needs_replan={self.state.needs_replan}, achieved_flag={self.state.goal_achieved_flag}")
        
        elif base_tool_func_name == UMS_FUNC_UPDATE_WORKFLOW_STATUS and is_successful_call:
            self.logger.info(f"{log_prefix}: START. WF ID marked in UMS: {_fmt_id(result_content.get('workflow_id'))}, New Status in UMS: {result_content.get('status')}")
            
            requested_status_str = result_content.get("status")
            wf_id_updated_in_ums = result_content.get("workflow_id")

            if not requested_status_str: 
                self.logger.error(f"{log_prefix}: 'status' missing from UMS tool response. Aborting side effect. Response: {str(result_content)[:200]}")
                return
            try:
                requested_status_enum = WorkflowStatus(requested_status_str.lower())
            except ValueError:
                self.logger.error(f"{log_prefix}: Invalid workflow status '{requested_status_str}' in UMS response. Aborting. Response: {str(result_content)[:200]}")
                return 

            if wf_id_updated_in_ums and self.state.workflow_stack and wf_id_updated_in_ums == self.state.workflow_stack[-1]:
                is_terminal_wf_status = requested_status_enum in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, WorkflowStatus.ABANDONED]
                
                if is_terminal_wf_status:
                    finished_wf_id = self.state.workflow_stack.pop() 
                    parent_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else None 
                    
                    if parent_wf_id: 
                        self.state.workflow_id = parent_wf_id 
                        self.state.context_id = self.state.workflow_id 
                        await self._set_default_thought_chain_id() # For the parent workflow
                        
                        self.logger.info(f"{log_prefix}: Sub-workflow '{_fmt_id(finished_wf_id)}' ({requested_status_enum.value}) finished. Returning to parent workflow '{_fmt_id(parent_wf_id)}'. Agent's UMS goal stack will be cleared.")
                        
                        self.state.goal_stack = [] 
                        self.state.current_goal_id = None 
                        
                        await self._write_temp_workflow_id(self.state.workflow_id) 

                        self.state.needs_replan = True
                        self.state.current_plan = [
                            PlanStep(description=f"Sub-workflow '{_fmt_id(finished_wf_id)}' ({requested_status_enum.value}) finished. Re-assess context and UMS goals for parent workflow '{_fmt_id(self.state.workflow_id)}'.")
                        ]
                        self.state.last_error_details = None 
                        self.state.consecutive_error_count = 0
                    else: # Root workflow finished
                        self.logger.info(f"{log_prefix}: Root workflow '{_fmt_id(finished_wf_id)}' finished ({requested_status_enum.value}). No parent workflow on stack. Agent run concluding.")
                        
                        self.state.workflow_id = None 
                        self.state.context_id = None
                        self.state.current_thought_chain_id = None
                        self.state.current_plan = [] 
                        self.state.goal_stack = [] 
                        self.state.current_goal_id = None
                        self.state.goal_achieved_flag = (requested_status_enum == WorkflowStatus.COMPLETED) 
                        await self._write_temp_workflow_id(None) 
            else: 
                self.logger.info(
                    f"{log_prefix}: UMS Workflow '{_fmt_id(wf_id_updated_in_ums)}' status changed to '{requested_status_enum.value}'. "
                    "This was not the agent's current active workflow on top of stack. No change to agent's primary focus or goal stack."
                )
            
            self.logger.info(f"{log_prefix}: END. WF: {_fmt_id(self.state.workflow_id)}, Current UMS Goal: {_fmt_id(self.state.current_goal_id)}, goal_achieved_flag: {self.state.goal_achieved_flag}")

        # Final summary log for focus changes
        new_current_wf_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        if (current_wf_id_before_effect != new_current_wf_id or
            current_goal_id_before_effect != self.state.current_goal_id):
            self.logger.info( 
                f"AML_SIDE_EFFECTS Summary (State Changed by '{base_tool_func_name}'): "
                f"WF: {_fmt_id(current_wf_id_before_effect)} -> {_fmt_id(new_current_wf_id)}, "
                f"UMS Goal: {_fmt_id(current_goal_id_before_effect)} -> {_fmt_id(self.state.current_goal_id)}, "
                f"Local Goal Stack Depth: {len(self.state.goal_stack)}"
            )
        else:
            self.logger.info(f"AML_SIDE_EFFECTS Summary (Agent WF/Goal focus NOT significantly changed by tool '{base_tool_func_name}')")
        
        self.logger.info(f"{log_prefix}: Exiting. Final state: WF ID='{_fmt_id(self.state.workflow_id)}', UMS Goal ID='{_fmt_id(self.state.current_goal_id)}', needs_replan={self.state.needs_replan}")


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


    async def _apply_heuristic_plan_update(self, last_decision: Dict[str, Any], last_tool_result_content: Optional[Dict[str, Any]] = None):
        self.logger.info("📋 Applying heuristic plan update (fallback)...")

        # Ensure there's always a plan, even if empty.
        if not self.state.current_plan:
            self.logger.warning("📋 Heuristic: Current plan was empty. Resetting to default re-evaluation step.")
            self.state.current_plan = [PlanStep(description="Fallback: Plan was empty. Re-evaluate current situation and objectives.")]
            self.state.needs_replan = True # Explicitly set replan as the plan was invalid
            # Log current state for debugging empty plan scenario
            self.logger.debug(f"📋 Heuristic: State after empty plan reset: WF={_fmt_id(self.state.workflow_id)}, Goal={_fmt_id(self.state.current_goal_id)}, NeedsReplan={self.state.needs_replan}")
            return # Exit early as there's no current step to process

        current_step = self.state.current_plan[0]
        decision_type = last_decision.get("decision")
        action_was_successful_this_turn = False # Tracks if the primary action/decision of this turn was successful

        # tool_name_executed is the original MCP name if from LLM, or AGENT_TOOL_UPDATE_PLAN
        # It's the tool that was *involved* in the LLM's decision, not necessarily what the agent just executed.
        tool_name_involved_in_llm_decision = last_decision.get("tool_name")

        self.logger.debug(
            f"📋 Heuristic: Decision Type='{decision_type}', Tool Involved in LLM Decision='{tool_name_involved_in_llm_decision}', Current Step='{current_step.description[:50]}...', NeedsReplan (Before)='{self.state.needs_replan}'"
        )

        # Case 1: A UMS/Server tool was executed by MCPClient (as per LLM's request)
        # OR an agent-internal tool (other than AGENT_TOOL_UPDATE_PLAN) was called by LLM (if that path existed)
        # AND its result is now being processed by heuristic.
        if decision_type == "tool_executed_by_mcp" or \
           (decision_type == "call_tool" and tool_name_involved_in_llm_decision != AGENT_TOOL_UPDATE_PLAN):
            
            # `last_tool_result_content` for "tool_executed_by_mcp" IS the direct UMS result payload.
            # For "call_tool" (if it were a UMS tool executed by AML), last_tool_result_content would also be UMS payload.
            tool_call_succeeded = isinstance(last_tool_result_content, dict) and last_tool_result_content.get("success", False)
            action_was_successful_this_turn = tool_call_succeeded

            if tool_call_succeeded:
                current_step.status = ActionStatus.COMPLETED.value
                summary = "Success."
                if isinstance(last_tool_result_content, dict):
                    summary_keys = ["summary", "message", "memory_id", "action_id", "artifact_id", "link_id", "chain_id", "state_id", "report", "visualization", "goal_id", "workflow_id"]
                    data_payload = last_tool_result_content
                    for k in summary_keys:
                        if k in data_payload and data_payload[k] is not None:
                            summary_value_str = str(data_payload[k])
                            summary = f"{k}: {_fmt_id(summary_value_str) if 'id' in k.lower() else summary_value_str}"
                            break
                    else:
                        generic_summary_parts = [
                            f"{k_sum}={_fmt_id(str(v_sum)) if 'id' in k_sum.lower() else str(v_sum)[:20]}"
                            for k_sum, v_sum in data_payload.items()
                            if k_sum not in ['success', 'processing_time', '_note', 'data_received', 'error_type_for_agent'] and v_sum is not None
                        ]
                        summary = f"Success. Data: {', '.join(generic_summary_parts)}" if generic_summary_parts else "Success (No further summary data)."
                current_step.result_summary = summary[:150]
                self.state.current_plan.pop(0)
                if not self.state.current_plan:
                    self.state.current_plan.append(PlanStep(description="Plan finished. Analyze overall result and decide if goal is met."))
                
                # `needs_replan` might have been set True by side-effects (e.g., goal change). Preserve it.
                if not self.state.needs_replan:
                    self.state.needs_replan = False # Explicitly set to false if tool was success and no side-effect replan.
                else:
                    self.logger.info(f"📋 Heuristic: Preserving needs_replan=True set by prior operation despite successful tool '{tool_name_involved_in_llm_decision}'.")
                self.logger.debug(f"📋 Heuristic: Tool '{tool_name_involved_in_llm_decision}' successful. Plan step completed. NeedsReplan={self.state.needs_replan}")

            else: # Tool call failed
                current_step.status = ActionStatus.FAILED.value
                error_msg = "Unknown failure from tool."
                # `self.state.last_error_details` should have been set by _execute_tool_call_internal or mcp_client
                if self.state.last_error_details and self.state.last_error_details.get("tool") == tool_name_involved_in_llm_decision:
                    error_msg = f"Type: {self.state.last_error_details.get('type', 'Unknown')}, Msg: {self.state.last_error_details.get('error', 'Unknown')}"
                elif isinstance(last_tool_result_content, dict): # Fallback to the content passed to heuristic
                    error_msg = str(last_tool_result_content.get("error", last_tool_result_content.get("message", "Unknown failure in result content")))
                
                current_step.result_summary = f"Failure: {error_msg[:150]}"
                # Insert "Analyze failure" step only if it's not already the next step or part of a similar error recovery
                if len(self.state.current_plan) < 2 or not self.state.current_plan[1].description.lower().startswith("analyze failure"):
                    self.state.current_plan.insert(
                        1, PlanStep(description=f"Analyze failure of step '{current_step.description[:30]}...' (Tool: {tool_name_involved_in_llm_decision or 'N/A'}) and replan.")
                    )
                self.state.needs_replan = True
                self.logger.warning(f"📋 Heuristic: Tool '{tool_name_involved_in_llm_decision}' failed. Plan step marked FAILED. NeedsReplan=True. Error: {error_msg}")

        # Case 2: LLM's decision was to record a thought
        elif decision_type == "thought_process":
            action_was_successful_this_turn = True # Recording a thought is considered a successful "action" for the turn
            current_step.status = ActionStatus.COMPLETED.value # Assume current plan step was "record a thought" or similar
            current_step.result_summary = f"Thought Recorded: {last_decision.get('content', '')[:50]}..."
            self.state.current_plan.pop(0)
            if not self.state.current_plan:
                self.state.current_plan.append(PlanStep(description="Decide next action based on recorded thought and overall goal."))
            self.state.needs_replan = False # Thought recorded, proceed with new plan head
            self.logger.debug(f"📋 Heuristic: Thought recorded. Plan step completed. NeedsReplan=False")


        # Case 3: LLM signaled overall goal completion
        elif decision_type == "complete":
            action_was_successful_this_turn = True
            self.state.current_plan = [PlanStep(description="Goal Achieved. Finalizing workflow.", status="completed")]
            self.state.needs_replan = False # Goal achieved, no replan needed from here.
            # self.state.goal_achieved_flag is set by _handle_workflow_and_goal_side_effects or by MCPClient if directly signaled.
            self.logger.info(f"📋 Heuristic: LLM signaled 'complete'. Plan set to finalization. NeedsReplan=False")

        # Case 4: LLM's decision was to update the plan via explicit tool call to AGENT_TOOL_UPDATE_PLAN
        elif decision_type == "call_tool" and tool_name_involved_in_llm_decision == AGENT_TOOL_UPDATE_PLAN:
            # The success of this is determined by last_tool_result_content from _execute_tool_call_internal
            plan_update_tool_succeeded = isinstance(last_tool_result_content, dict) and last_tool_result_content.get("success", False)
            action_was_successful_this_turn = plan_update_tool_succeeded

            if plan_update_tool_succeeded:
                # Plan was already updated by _execute_tool_call_internal. needs_replan is False.
                self.logger.info(f"📋 Heuristic: Plan successfully updated by explicit call to '{AGENT_TOOL_UPDATE_PLAN}'. No further plan changes here. NeedsReplan={self.state.needs_replan}")
            else:
                # Plan update tool call failed. _execute_tool_call_internal set last_error_details and needs_replan=True.
                # Mark current step as failed because the intended plan update failed.
                current_step.status = ActionStatus.FAILED.value
                error_msg = "Plan update tool call failed."
                if self.state.last_error_details:
                    error_msg = f"Type: {self.state.last_error_details.get('type', 'PlanUpdateError')}, Msg: {self.state.last_error_details.get('error', 'Unknown')}"
                current_step.result_summary = f"Failed plan update: {error_msg[:100]}"
                # Ensure needs_replan is true
                self.state.needs_replan = True
                self.logger.warning(f"📋 Heuristic: Explicit call to '{AGENT_TOOL_UPDATE_PLAN}' FAILED. Current step marked FAILED. NeedsReplan=True. Error: {error_msg}")

        # Case 5: LLM's decision was "plan_update" (parsed from text by MCPClient)
        elif decision_type == "plan_update":
            # The plan was already applied (or rejected due to cycle) by execute_llm_decision.
            # `action_was_successful_this_turn` depends on whether that application succeeded.
            # If `self.state.needs_replan` is False here, it means the textual plan was applied successfully.
            if not self.state.needs_replan:
                action_was_successful_this_turn = True
                self.logger.info("📋 Heuristic: Plan was successfully updated from LLM's textual proposal (MCPC parsed). No further plan changes here.")
            else: # Plan from text was not applied (e.g. cycle, or validation error in execute_llm_decision)
                action_was_successful_this_turn = False
                current_step.status = ActionStatus.FAILED.value
                error_msg = "Failed to apply LLM textual plan."
                if self.state.last_error_details and self.state.last_error_details.get("type") == "PlanValidationError":
                     error_msg = f"LLM textual plan validation error: {self.state.last_error_details.get('error', 'Unknown')}"
                current_step.result_summary = f"Textual Plan Error: {error_msg[:100]}"
                self.logger.warning(f"📋 Heuristic: LLM's textual plan proposal was NOT applied. Current step FAILED. NeedsReplan=True. Error: {error_msg}")

        # Case 6: Other error decisions from LLM/MCPClient or unhandled types
        else:
            action_was_successful_this_turn = False # Default to not successful
            current_step.status = ActionStatus.FAILED.value
            err_summary = self.state.last_action_summary or "Unknown agent error or unhandled decision"
            current_step.result_summary = f"Agent/Decision Error: {err_summary[:100]}..."
            if len(self.state.current_plan) < 2 or not self.state.current_plan[1].description.lower().startswith("re-evaluate due to"):
                self.state.current_plan.insert(1, PlanStep(description=f"Re-evaluate due to agent/decision error: {decision_type or 'UnknownDecision'}"))
            self.state.needs_replan = True
            self.logger.warning(f"📋 Heuristic: Unhandled decision type '{decision_type}' or error state. Step FAILED. NeedsReplan=True.")


        # --- Update Counters Based on `action_was_successful_this_turn` ---
        if action_was_successful_this_turn:
            self.state.consecutive_error_count = 0
            # Determine base function name for meta tool check
            # Use tool_name_involved_in_llm_decision as it's what LLM interacted with this turn
            base_tool_name_executed = self._get_base_function_name(tool_name_involved_in_llm_decision) if tool_name_involved_in_llm_decision else None
            
            # Increment progress counters if the action was substantive (not purely internal/meta)
            is_substantive_action = (
                tool_name_involved_in_llm_decision and 
                base_tool_name_executed not in self._INTERNAL_OR_META_TOOLS_BASE_NAMES and 
                tool_name_involved_in_llm_decision != AGENT_TOOL_UPDATE_PLAN
            )
            is_thought_process = decision_type == "thought_process"

            if is_substantive_action:
                self.state.successful_actions_since_reflection += 1.0
                self.state.successful_actions_since_consolidation += 1.0
                self.logger.debug(
                    f"📋 Heuristic: Incr success R:{self.state.successful_actions_since_reflection:.1f}, C:{self.state.successful_actions_since_consolidation:.1f} after substantive tool: {tool_name_involved_in_llm_decision}"
                )
            elif is_thought_process: # Also count thoughts as minor progress
                self.state.successful_actions_since_reflection += 0.5
                self.state.successful_actions_since_consolidation += 0.5
                self.logger.debug(
                    f"📋 Heuristic: Incr success R:{self.state.successful_actions_since_reflection:.1f}, C:{self.state.successful_actions_since_consolidation:.1f} after thought process."
                )
        else: # Action was not successful
            # Only increment consecutive_error_count if the failure wasn't an AGENT_TOOL_UPDATE_PLAN tool call
            # that itself succeeded (because a successful plan update *resolves* an error state).
            # `action_was_successful_this_turn` already covers this.
            self.state.consecutive_error_count += 1
            self.logger.warning(f"📋 Heuristic: Consecutive error count incremented to: {self.state.consecutive_error_count}")
            # Reset progress counters on any failure
            if self.state.successful_actions_since_reflection > 0:
                self.logger.info(f"📋 Heuristic: Reset reflection counter from {self.state.successful_actions_since_reflection:.1f} due to failed action/decision.")
                self.state.successful_actions_since_reflection = 0
            if self.state.successful_actions_since_consolidation > 0:
                self.logger.info(f"📋 Heuristic: Reset consolidation counter from {self.state.successful_actions_since_consolidation:.1f} due to failed action/decision.")
                self.state.successful_actions_since_consolidation = 0
        
        # Log final plan state
        log_plan_msg = f"Plan updated heuristically. Steps: {len(self.state.current_plan)}. "
        if self.state.current_plan:
            next_step_log = self.state.current_plan[0]
            depends_str_log = f"Depends: {[_fmt_id(d) for d in next_step_log.depends_on]}" if next_step_log.depends_on else "Depends: None"
            log_plan_msg += f"Next: '{next_step_log.description[:60]}...' (ID: {next_step_log.id[:8]}, Status: {next_step_log.status}, {depends_str_log})"
        else:
            log_plan_msg += "Plan is now empty."
        self.logger.info(f"📋 Heuristic Update End: {log_plan_msg}. NeedsReplan={self.state.needs_replan}. ConsecutiveErrors={self.state.consecutive_error_count}")

    def _adapt_thresholds(self, stats: Dict[str, Any]) -> None:
        # ... (This method's logic remains the same as provided previously)
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
                        ums_compute_stats_mcp_name, 
                        {"workflow_id": self.state.workflow_id}, 
                        record_action=False
                    )
                    if stats_result.get("success"):
                        self._adapt_thresholds(stats_result)
                        episodic_count = stats_result.get("by_level", {}).get(MemoryLevel.EPISODIC.value, 0)
                        # Check if consolidation is already scheduled to avoid duplication
                        if (episodic_count > (self.state.current_consolidation_threshold * 2.0) and 
                            consolidation_tool_available and 
                            not any(task[0] == ums_consolidation_mcp_name for task in tasks_to_run)):
                            self.logger.info(f"{log_prefix}: High episodic count ({episodic_count}), scheduling consolidation.")
                            tasks_to_run.append(
                                (ums_consolidation_mcp_name, {
                                    "workflow_id": self.state.workflow_id, 
                                    "consolidation_type": "summary",
                                    "query_filter": {"memory_level": MemoryLevel.EPISODIC.value},
                                    "max_source_memories": self.consolidation_max_sources,
                                })
                            )
                            trigger_reasons.append(f"HighEpisodic({episodic_count})")
                            self.state.successful_actions_since_consolidation = 0 # Reset as it's being scheduled
                    else:
                        self.logger.warning(f"{log_prefix}: Failed to compute stats for adaptation: {stats_result.get('error')}")
                except Exception as e:
                    self.logger.error(f"{log_prefix}: Error during stats/adaptation: {e}", exc_info=True) # Log full traceback for unexpected
                finally:
                    self.state.loops_since_stats_adaptation = 0
            else:
                self.logger.warning(f"{log_prefix}: Skipping stats/adaptation: Tool '{UMS_FUNC_COMPUTE_STATS}' not available.")
                self.state.loops_since_stats_adaptation = 0 # Still reset counter

        # 2. Reflection
        needs_reflection = self.state.needs_replan or \
                           (self.state.successful_actions_since_reflection >= self.state.current_reflection_threshold)
        if needs_reflection:
            if reflection_tool_available and not any(task[0] == ums_reflection_mcp_name for task in tasks_to_run):
                reflection_type = self.reflection_type_sequence[self.state.reflection_cycle_index % len(self.reflection_type_sequence)]
                tasks_to_run.append((ums_reflection_mcp_name, {"workflow_id": self.state.workflow_id, "reflection_type": reflection_type}))
                reason_str = (f"Replan({self.state.needs_replan})" if self.state.needs_replan else 
                              f"SuccessCount({self.state.successful_actions_since_reflection:.1f}>={self.state.current_reflection_threshold})")
                trigger_reasons.append(f"Reflect({reason_str})")
                self.state.successful_actions_since_reflection = 0 # Reset counter as it's being scheduled
                self.state.reflection_cycle_index += 1
            else:
                if not reflection_tool_available:
                    self.logger.warning(f"{log_prefix}: Skipping reflection: Tool '{UMS_FUNC_REFLECTION}' unavailable.")
                    self.state.successful_actions_since_reflection = 0 # Reset if tool missing, to avoid re-triggering immediately
                elif any(task[0] == ums_reflection_mcp_name for task in tasks_to_run):
                     self.logger.debug(f"{log_prefix}: Reflection already scheduled this cycle.")


        # 3. Consolidation
        needs_consolidation = self.state.successful_actions_since_consolidation >= self.state.current_consolidation_threshold
        if needs_consolidation:
            if consolidation_tool_available and not any(task[0] == ums_consolidation_mcp_name for task in tasks_to_run):
                tasks_to_run.append(
                    (ums_consolidation_mcp_name, {
                        "workflow_id": self.state.workflow_id, 
                        "consolidation_type": "summary",
                        "query_filter": {"memory_level": self.consolidation_memory_level}, # Use configured level
                        "max_source_memories": self.consolidation_max_sources,
                    })
                )
                trigger_reasons.append(f"ConsolidateThreshold({self.state.successful_actions_since_consolidation:.1f}>={self.state.current_consolidation_threshold})")
                self.state.successful_actions_since_consolidation = 0 # Reset counter as it's being scheduled
            else:
                if not consolidation_tool_available:
                    self.logger.warning(f"{log_prefix}: Skipping consolidation: Tool '{UMS_FUNC_CONSOLIDATION}' unavailable.")
                    self.state.successful_actions_since_consolidation = 0 # Reset if tool missing
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
            if promotion_query_tool_available: # query_memories is needed to find candidates
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
                tasks_to_run.append((ums_delete_expired_mcp_name, {"db_path": None})) # db_path will be handled by UMS tool
                trigger_reasons.append("MaintenanceInterval")
                self.state.loops_since_maintenance = 0
            else: 
                self.logger.warning(f"{log_prefix}: Skipping maintenance: Tool '{UMS_FUNC_DELETE_EXPIRED_MEMORIES}' unavailable.")
                self.state.loops_since_maintenance = 0 # Reset counter even if tool is missing

        # Execute scheduled tasks
        if tasks_to_run:
            unique_reasons_str = ", ".join(sorted(set(trigger_reasons))) # Ensure unique reasons
            self.logger.info(f"🧠 {log_prefix}: Running {len(tasks_to_run)} periodic tasks (Triggers: {unique_reasons_str}).")
            
            # Sort for consistent execution order (e.g., maintenance first, then stats, then cognitive)
            def sort_key_periodic(task_tuple: Tuple[str, Dict[str, Any]]) -> int:
                tool_mcp_name = task_tuple[0]
                if tool_mcp_name == ums_delete_expired_mcp_name: return 0
                if tool_mcp_name == ums_compute_stats_mcp_name: return 1
                if tool_mcp_name == "CHECK_PROMOTIONS_INTERNAL_TASK": return 2
                if tool_mcp_name == ums_optimize_wm_mcp_name: return 3
                if tool_mcp_name == ums_auto_focus_mcp_name: return 4
                if tool_mcp_name == ums_consolidation_mcp_name: return 5
                if tool_mcp_name == ums_reflection_mcp_name: return 6
                return 7 # Default for any other
            tasks_to_run.sort(key=sort_key_periodic)

            for mcp_tool_name_to_call, args_for_tool in tasks_to_run:
                if self._shutdown_event.is_set():
                    self.logger.info(f"{log_prefix}: Shutdown signaled during periodic tasks execution. Stopping further tasks.")
                    break
                try:
                    if mcp_tool_name_to_call == "CHECK_PROMOTIONS_INTERNAL_TASK":
                        self.logger.debug(f"{log_prefix}: Triggering internal promotion checks based on workflow_id '{args_for_tool.get('workflow_id')}'...")
                        await self._trigger_promotion_checks() # This internally uses self.state.workflow_id
                        continue # Skip normal tool call processing
                    
                    self.logger.debug(f"{log_prefix}: Executing periodic MCP Tool: {mcp_tool_name_to_call} with args: {args_for_tool}")
                    
                    # Ensure workflow_id is in args if not already and tool is not create_workflow/list_workflows
                    if 'workflow_id' not in args_for_tool and self.state.workflow_id and \
                       self._get_base_function_name(mcp_tool_name_to_call) not in [UMS_FUNC_CREATE_WORKFLOW, UMS_FUNC_LIST_WORKFLOWS, UMS_FUNC_DELETE_EXPIRED_MEMORIES]:
                        args_for_tool['workflow_id'] = self.state.workflow_id
                    if 'context_id' not in args_for_tool and self.state.context_id and \
                       self._get_base_function_name(mcp_tool_name_to_call) in [UMS_FUNC_OPTIMIZE_WM, UMS_FUNC_AUTO_FOCUS]:
                         args_for_tool['context_id'] = self.state.context_id

                    result_content_periodic = await self._execute_tool_call_internal(mcp_tool_name_to_call, args_for_tool, record_action=False)
                    
                    base_func_name_periodic = self._get_base_function_name(mcp_tool_name_to_call)
                    
                    if not result_content_periodic.get("success"):
                        self.logger.warning(f"{log_prefix}: Periodic task UMS tool '{mcp_tool_name_to_call}' call failed: {result_content_periodic.get('error')}")
                        # If a critical periodic task (like optimize_wm or auto_focus) fails, it might warrant setting last_error_details
                        if base_func_name_periodic in [UMS_FUNC_OPTIMIZE_WM, UMS_FUNC_AUTO_FOCUS]:
                            if not self.state.last_error_details: # Don't overwrite existing primary error
                                self.state.last_error_details = {
                                    "tool": mcp_tool_name_to_call, 
                                    "error": f"Periodic {base_func_name_periodic} failed: {result_content_periodic.get('error')}",
                                    "type": "PeriodicTaskToolError"
                                }
                        continue # Move to next periodic task

                    # Process successful periodic task result for reflection/consolidation
                    if base_func_name_periodic in [UMS_FUNC_REFLECTION, UMS_FUNC_CONSOLIDATION]:
                        feedback = ""
                        if base_func_name_periodic == UMS_FUNC_REFLECTION:
                            feedback = result_content_periodic.get("content", "")
                        elif base_func_name_periodic == UMS_FUNC_CONSOLIDATION:
                            feedback = result_content_periodic.get("consolidated_content", "")
                        
                        if feedback:
                            feedback_summary = str(feedback).split("\n", 1)[0][:150] # Get first line
                            self.state.last_meta_feedback = f"Feedback from UMS {base_func_name_periodic}: {feedback_summary}..."
                            self.logger.info(f"{log_prefix}: Meta-feedback received from UMS {base_func_name_periodic}: {self.state.last_meta_feedback}")
                            self.state.needs_replan = True # Meta-cognition often leads to replanning
                        else:
                            self.logger.debug(f"{log_prefix}: Periodic UMS task {mcp_tool_name_to_call} succeeded but returned no feedback content. Result: {str(result_content_periodic)[:200]}")
                
                except Exception as e_periodic_task_exec:
                    self.logger.error(f"{log_prefix}: Exception during execution of periodic task '{mcp_tool_name_to_call}': {e_periodic_task_exec}", exc_info=True)
                    # Optionally set last_error_details if a periodic task itself has an unhandled exception
                    if not self.state.last_error_details:
                        self.state.last_error_details = {
                            "tool": mcp_tool_name_to_call,
                            "error": f"Unhandled exception in periodic task execution: {str(e_periodic_task_exec)[:100]}",
                            "type": "PeriodicTaskExecutionError"
                        }
                        self.state.needs_replan = True

                await asyncio.sleep(0.05) # Small delay between tasks if many are queued
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

        if not self._find_tool_server(query_memories_mcp_name) or \
           not self._find_tool_server(promote_level_mcp_name):
            self.logger.warning(f"Skipping promotion checks: Required UMS tools for '{UMS_FUNC_QUERY_MEMORIES}' or '{UMS_FUNC_PROMOTE_MEM}' unavailable.")
            return

        candidate_ids = set()
        try:
            # Fetch episodic memories
            episodic_args = {
                "workflow_id": self.state.workflow_id,
                "memory_level": MemoryLevel.EPISODIC.value,
                "sort_by": "last_accessed", "sort_order": "DESC", "limit": 5, "include_content": False,
            }
            episodic_res = await self._execute_tool_call_internal(query_memories_mcp_name, episodic_args, record_action=False)
            if episodic_res.get("success"):
                mems = episodic_res.get("memories", [])
                candidate_ids.update(m.get("memory_id") for m in mems if isinstance(m, dict) and m.get("memory_id"))
            
            # Fetch semantic memories that are procedures or skills
            semantic_args = {
                "workflow_id": self.state.workflow_id,
                "memory_level": MemoryLevel.SEMANTIC.value,
                "memory_type": None, # Will be filtered below
                "sort_by": "last_accessed", "sort_order": "DESC", "limit": 10, "include_content": False, # Fetch more semantic to filter
            }
            semantic_res = await self._execute_tool_call_internal(query_memories_mcp_name, semantic_args, record_action=False)
            if semantic_res.get("success"):
                mems = semantic_res.get("memories", [])
                candidate_ids.update(
                    m.get("memory_id") for m in mems 
                    if isinstance(m, dict) and m.get("memory_id") and 
                       m.get("memory_type") in [MemoryType.PROCEDURE.value, MemoryType.SKILL.value]
                )
            
            if candidate_ids:
                self.logger.debug(f"Checking {len(candidate_ids)} memories for promotion: {[_fmt_id(i) for i in candidate_ids]}")
                for mem_id in candidate_ids:
                    if self._shutdown_event.is_set(): break
                    # Pass the snapshot workflow_id for the background task's context
                    self._start_background_task(AgentMasterLoop._check_and_trigger_promotion, 
                                                memory_id=mem_id, 
                                                workflow_id=self.state.workflow_id, # Pass current WF ID for context
                                                context_id=self.state.context_id)   # Pass current context ID
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
            "last_error_details": copy.deepcopy(self.state.last_error_details),
            "needs_replan": self.state.needs_replan,
            "workflow_stack_summary": [_fmt_id(wf_id) for wf_id in self.state.workflow_stack[-3:]],
            "meta_feedback": self.state.last_meta_feedback,
            "current_thought_chain_id": self.state.current_thought_chain_id,
            "retrieval_timestamp_agent_state": agent_retrieval_timestamp,
            "status_message_from_agent": "Context assembly by agent.",
            "errors_in_context_gathering": [],
        }
        self.state.last_meta_feedback = None # Clear after including

        current_workflow_id = self.state.workflow_stack[-1] if self.state.workflow_stack else self.state.workflow_id
        current_cognitive_context_id = self.state.context_id
        current_plan_step_desc = self.state.current_plan[0].description if self.state.current_plan else DEFAULT_PLAN_STEP

        if not current_workflow_id:
            context_payload["status_message_from_agent"] = "No Active Workflow."
            self.logger.warning(context_payload["status_message_from_agent"])
            context_payload["ums_package_retrieval_status"] = "skipped_no_workflow"
            context_payload["agent_assembled_goal_context"] = {
                "retrieved_at": agent_retrieval_timestamp,
                "current_goal_details_from_ums": None,
                "goal_stack_summary_from_agent_state": [],
                "data_source_comment": "No active workflow, so no UMS goal context.",
            }
            context_payload["processing_time_sec"] = time.time() - start_time
            return context_payload

        context_payload["workflow_id"] = current_workflow_id
        context_payload["cognitive_context_id_agent"] = current_cognitive_context_id

        # 1. Agent Assembles Its Goal Context
        agent_goal_context_block: Dict[str, Any] = {
            "retrieved_at": agent_retrieval_timestamp,
            "current_goal_details_from_ums": None,
            "goal_stack_summary_from_agent_state": [],
            "data_source_comment": "Goal context assembly by agent.",
        }

        if self.state.current_goal_id:
            ums_fetched_stack = await self._fetch_goal_stack_from_ums(self.state.current_goal_id)
            if ums_fetched_stack and ums_fetched_stack[-1].get("goal_id") == self.state.current_goal_id:
                agent_goal_context_block["current_goal_details_from_ums"] = ums_fetched_stack[-1]
                agent_goal_context_block["goal_stack_summary_from_agent_state"] = [
                    {"goal_id": _fmt_id(g.get("goal_id")), "description": (g.get("description") or "")[:150] + "...", "status": g.get("status")}
                    for g in ums_fetched_stack[-CONTEXT_GOAL_STACK_SHOW_LIMIT:]
                ]
                agent_goal_context_block["data_source_comment"] = "Goal stack and current goal details fetched successfully from UMS by agent."
                self.logger.info(f"Successfully fetched UMS goal stack for {_fmt_id(self.state.current_goal_id)} for context.")
            elif self.state.goal_stack and self.state.goal_stack[-1].get("goal_id") == self.state.current_goal_id:
                self.logger.warning(f"UMS fetch for goal stack {_fmt_id(self.state.current_goal_id)} failed or mismatched. Using agent's local goal_stack for context. UMS fetched: {[_fmt_id(g.get('goal_id')) for g in ums_fetched_stack]}")
                agent_goal_context_block["current_goal_details_from_ums"] = self.state.goal_stack[-1]
                agent_goal_context_block["goal_stack_summary_from_agent_state"] = [
                    {"goal_id": _fmt_id(g.get("goal_id")), "description": (g.get("description") or "")[:150] + "...", "status": g.get("status")}
                    for g in self.state.goal_stack[-CONTEXT_GOAL_STACK_SHOW_LIMIT:]
                ]
                agent_goal_context_block["data_source_comment"] = "Used agent's local goal stack state for context (UMS fetch failed/mismatched)."
                context_payload["errors_in_context_gathering"].append(f"Agent: UMS goal stack fetch for {_fmt_id(self.state.current_goal_id)} was inconsistent, used local state.")
            else:
                err_msg = f"Agent: Failed to fetch UMS goal stack for current goal {_fmt_id(self.state.current_goal_id)} and local agent stack is also inconsistent/empty."
                context_payload["errors_in_context_gathering"].append(err_msg)
                agent_goal_context_block["current_goal_details_from_ums"] = {"error_fetching_details": err_msg, "goal_id_attempted": self.state.current_goal_id}
                agent_goal_context_block["data_source_comment"] = "Critical error: Could not obtain current UMS goal details."
                self.logger.error(err_msg)
        else:
            agent_goal_context_block["data_source_comment"] = "No current_goal_id set in agent state for this active workflow. UMS goal context is empty."
            self.logger.info("Agent has no current_goal_id set for active workflow; UMS goal context will be empty for this turn.")
        context_payload["agent_assembled_goal_context"] = agent_goal_context_block

        # 2. Call UMS Tool for Rich Context Package
        ums_get_rich_context_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_GET_RICH_CONTEXT_PACKAGE)
        ums_package_data: Optional[Dict[str, Any]] = {"error_ums_package_tool_unavailable": f"Tool for '{UMS_FUNC_GET_RICH_CONTEXT_PACKAGE}' unavailable."}

        if self._find_tool_server(ums_get_rich_context_mcp_name):
            focal_id_hint_for_ums = None
            if isinstance(self.state.last_error_details, dict) and self.state.last_error_details.get("focal_memory_id_from_last_wm"):
                focal_id_hint_for_ums = self.state.last_error_details["focal_memory_id_from_last_wm"]

            ums_package_params = {
                "workflow_id": current_workflow_id,
                "context_id": current_cognitive_context_id,
                "current_plan_step_description": current_plan_step_desc,
                "focal_memory_id_hint": focal_id_hint_for_ums,
                "fetch_limits": {
                    "recent_actions": CONTEXT_RECENT_ACTIONS_FETCH_LIMIT,
                    "important_memories": CONTEXT_IMPORTANT_MEMORIES_FETCH_LIMIT,
                    "key_thoughts": CONTEXT_KEY_THOUGHTS_FETCH_LIMIT,
                    "proactive_memories": CONTEXT_PROACTIVE_MEMORIES_FETCH_LIMIT,
                    "procedural_memories": CONTEXT_PROCEDURAL_MEMORIES_FETCH_LIMIT,
                    "link_traversal": CONTEXT_LINK_TRAVERSAL_FETCH_LIMIT,
                },
                "show_limits": {
                    "working_memory": CONTEXT_WORKING_MEMORY_SHOW_LIMIT,
                    "link_traversal": CONTEXT_LINK_TRAVERSAL_SHOW_LIMIT,
                },
                "include_core_context": True, "include_working_memory": True,
                "include_proactive_memories": True, "include_relevant_procedures": True,
                "include_contextual_links": True,
                "compression_token_threshold": CONTEXT_MAX_TOKENS_COMPRESS_THRESHOLD,
                "compression_target_tokens": CONTEXT_COMPRESSION_TARGET_TOKENS,
            }
            try:
                self.logger.debug(f"Agent: Calling UMS tool '{ums_get_rich_context_mcp_name}' with params: {ums_package_params}")
                raw_ums_response = await self._execute_tool_call_internal(ums_get_rich_context_mcp_name, ums_package_params, record_action=False)

                if raw_ums_response.get("success"):
                    ums_package_data = raw_ums_response.get("context_package", {}) 
                    if not isinstance(ums_package_data, dict):
                        err_msg = f"Agent: UMS tool {ums_get_rich_context_mcp_name} returned invalid 'context_package' (type: {type(ums_package_data)})."
                        self.logger.error(err_msg)
                        context_payload["errors_in_context_gathering"].append(err_msg)
                        ums_package_data = {"error_ums_pkg_invalid_type": err_msg}
                    else:
                        self.logger.info("Agent: Successfully retrieved rich context package from UMS.")
                        ums_internal_errors = ums_package_data.get("errors") 
                        if ums_internal_errors and isinstance(ums_internal_errors, list):
                            context_payload["errors_in_context_gathering"].extend([f"UMS_PKG_ERR: {e}" for e in ums_internal_errors])
                        ums_package_data.pop("errors", None) 
                        context_payload["ums_package_retrieval_status"] = "success"
                else:
                    err_msg = f"Agent: UMS rich context pkg retrieval failed: {raw_ums_response.get('error', 'Unknown UMS tool error')}"
                    context_payload["errors_in_context_gathering"].append(err_msg)
                    self.logger.warning(err_msg)
                    ums_package_data = {"error_ums_pkg_retrieval": err_msg}
                    context_payload["ums_package_retrieval_status"] = "failed"
            except Exception as e:
                err_msg = f"Agent: Exception calling UMS for rich context pkg: {e}"
                self.logger.error(err_msg, exc_info=True)
                context_payload["errors_in_context_gathering"].append(err_msg)
                ums_package_data = {"error_ums_pkg_exception": err_msg}
                context_payload["ums_package_retrieval_status"] = "exception"
        else: 
            err_msg = f"Agent: UMS tool for '{UMS_FUNC_GET_RICH_CONTEXT_PACKAGE}' unavailable."
            self.logger.error(err_msg)
            context_payload["errors_in_context_gathering"].append(err_msg)
            context_payload["ums_package_retrieval_status"] = "tool_unavailable"

        context_payload["ums_context_package"] = ums_package_data

        if ums_package_data.get("ums_compression_details"):
            self.logger.info(f"Agent: UMS package includes compression details: {ums_package_data['ums_compression_details']}")

        final_errors_count = len(context_payload.get("errors_in_context_gathering", []))
        if current_workflow_id and not final_errors_count:
            context_payload["status_message_from_agent"] = "Workflow active. Context ready."
        elif current_workflow_id and final_errors_count:
            context_payload["status_message_from_agent"] = f"Workflow active. Context ready with {final_errors_count} errors."
        
        self.logger.info(f"Agent: Context gathering complete. Status: {context_payload['status_message_from_agent']}. Time: {(time.time() - start_time):.3f}s")
        if final_errors_count > 0:
            self.logger.info(f"Agent: Errors during context gathering: {context_payload.get('errors_in_context_gathering')}")
        
        context_payload["processing_time_sec"] = time.time() - start_time
        return context_payload

    async def execute_llm_decision(
        self,
        llm_decision: Dict[str, Any],
    ) -> bool:
        self.logger.info(
            f"AML EXEC_DECISION: Entered for Loop {self.state.current_loop}. Current WF: {_fmt_id(self.state.workflow_id)}, Current UMS Goal: {_fmt_id(self.state.current_goal_id)}"
        )
        self.logger.debug(f"AML EXEC_DECISION: Received LLM Decision from MCPClient: {str(llm_decision)[:500]}")

        tool_result_content: Optional[Dict[str, Any]] = None # This will store the *content* of the tool's result
        llm_proposed_plan_steps_data: Optional[List[Dict[str, Any]]] = llm_decision.get("updated_plan_steps")
        llm_proposed_plan_steps: Optional[List[PlanStep]] = None

        if llm_proposed_plan_steps_data:
            try:
                # Ensure PlanStep is imported or defined in this file
                llm_proposed_plan_steps = [PlanStep(**step_data) for step_data in llm_proposed_plan_steps_data]
            except (ValidationError, TypeError) as e:
                self.logger.error(f"AML EXEC_DECISION: Invalid data for 'updated_plan_steps': {e}. LLM proposed plan cannot be applied.", exc_info=True)
                self.state.last_error_details = {"tool": "agent:update_plan_implicit", "error": f"LLM proposed invalid plan structure: {e}", "type": "PlanUpdateError", "proposed_plan_data": llm_proposed_plan_steps_data}
                self.state.needs_replan = True
                llm_proposed_plan_steps = None # Ensure it's None if validation fails

        decision_type = llm_decision.get("decision")
        # tool_name_involved_in_turn will be the original MCP name of the tool the LLM interacted with or requested
        tool_name_involved_in_turn: Optional[str] = llm_decision.get("tool_name")

        if decision_type == "tool_executed_by_mcp":
            # This means MCPClient executed a UMS/Server tool based on LLM's request,
            # and llm_decision["result"] contains the *direct UMS payload dictionary*.
            tool_name_original_mcp = llm_decision.get("tool_name") # Original MCP name
            arguments_used = llm_decision.get("arguments", {})
            # ums_tool_result_payload IS THE DIRECT UMS RESULT DICTIONARY
            ums_tool_result_payload = llm_decision.get("result")

            tool_name_involved_in_turn = tool_name_original_mcp # For logging

            self.logger.info(f"AML EXEC_DECISION: Decision is 'tool_executed_by_mcp'. Tool: '{tool_name_original_mcp}'. Processing its side effects with provided result.")
            self.logger.debug(f"AML EXEC_DECISION (tool_executed_by_mcp): Received UMS payload type: {type(ums_tool_result_payload)}, Value: {str(ums_tool_result_payload)[:500]}...")

            actual_ums_payload: Optional[Dict[str, Any]]
            if isinstance(ums_tool_result_payload, dict):
                actual_ums_payload = ums_tool_result_payload
                self.logger.debug(f"AML EXEC_DECISION: 'tool_executed_by_mcp' - Using provided result directly as UMS payload. Keys: {list(actual_ums_payload.keys()) if actual_ums_payload else 'None'}")
            else:
                self.logger.error(f"AML EXEC_DECISION: CRITICAL - 'tool_executed_by_mcp' - Expected a dict for UMS result payload, got {type(ums_tool_result_payload)}. This indicates an issue in MCPClient's decision packaging.")
                actual_ums_payload = {"success": False, "error": f"Internal error: Unexpected UMS result payload format for 'tool_executed_by_mcp': {type(ums_tool_result_payload)}", "error_type_for_agent":"InternalProcessingError"}
            
            tool_result_content = actual_ums_payload # For heuristic update and summary logging

            base_tool_func_name = self._get_base_function_name(tool_name_original_mcp)
            await self._handle_workflow_and_goal_side_effects(base_tool_func_name, arguments_used, actual_ums_payload)

            # Generate last_action_summary based on the actual_ums_payload
            summary = ""
            if isinstance(actual_ums_payload, dict) and actual_ums_payload.get("success"):
                summary_keys = ["summary", "message", "memory_id", "action_id", "artifact_id", "link_id", "chain_id", "state_id", "report", "visualization", "goal_id", "workflow_id"]
                data_payload_for_summary = actual_ums_payload
                for k in summary_keys:
                    if k in data_payload_for_summary and data_payload_for_summary[k] is not None:
                        summary_value_str = str(data_payload_for_summary[k])
                        summary = f"{k}: {_fmt_id(summary_value_str) if 'id' in k.lower() else summary_value_str}"
                        break
                else: # No specific key found
                    generic_summary_parts = []
                    for k_sum, v_sum in data_payload_for_summary.items():
                        # Exclude internal/meta keys that MCPC might add, or UMS tool's own meta keys
                        if k_sum not in ['success', 'processing_time', '_original_raw_result', '_parsed_non_dict', '_note', 'error_type_for_agent'] and v_sum is not None:
                            generic_summary_parts.append(f"{k_sum}={_fmt_id(str(v_sum)) if 'id' in k_sum.lower() else str(v_sum)[:20]}")
                    if generic_summary_parts:
                        summary = f"Success. Data: {', '.join(generic_summary_parts)}"
                    else:
                        summary = "Success (No further summary data)."
            elif isinstance(actual_ums_payload, dict): # Error case from actual_ums_payload
                # Try to determine a more specific error type if `last_error_details` was set by the payload
                err_type_summary = actual_ums_payload.get("error_type_for_agent", actual_ums_payload.get("type", "ToolExecutionError"))
                err_msg_summary = str(actual_ums_payload.get("error", "Unknown Error from UMS tool execution"))[:100]
                summary = f"Failed ({err_type_summary}): {err_msg_summary}"
            else: # actual_ums_payload was not a dict (caught by the modified block above)
                 summary = f"Failed (Invalid UMS result structure for summarization: {type(actual_ums_payload)})"

            if isinstance(actual_ums_payload, dict) and actual_ums_payload.get("status_code"): summary += f" (Code: {actual_ums_payload['status_code']})"
            self.state.last_action_summary = f"{tool_name_original_mcp} (executed by MCP) -> {summary}"
            self.logger.info(f"🏁 Tool handled by MCP: {self.state.last_action_summary}")

            # Update agent's error state based on the UMS tool's success
            if isinstance(actual_ums_payload, dict) and actual_ums_payload.get("success"):
                # If UMS tool was successful, clear any prior error for *this specific tool*.
                # Global errors (like plan validation) might persist if needs_replan is true.
                if self.state.last_error_details and self.state.last_error_details.get("tool") == tool_name_original_mcp:
                    self.state.last_error_details = None
                # `needs_replan` is managed by `_handle_workflow_and_goal_side_effects`
            elif isinstance(actual_ums_payload, dict):
                # UMS tool itself failed. Set last_error_details for the agent.
                self.state.last_error_details = {
                    "tool": tool_name_original_mcp,
                    "args": arguments_used,
                    "error": actual_ums_payload.get("error", "UMS tool failed as reported by MCPClient"),
                    "status_code": actual_ums_payload.get("status_code"), # Propagate if available
                    "type": actual_ums_payload.get("error_type_for_agent", actual_ums_payload.get("type", "ToolExecutionError"))
                }
                self.state.needs_replan = True # If a UMS tool fails, agent usually needs to replan
            else: # Should not happen with the type check above
                self.state.last_error_details = { "tool": tool_name_original_mcp, "error": "Unexpected UMS result structure after MCP execution.", "type": "InternalProcessingError"}
                self.state.needs_replan = True

        elif decision_type == "call_tool":
            # This means MCPClient determined the LLM wants to call a tool, and this tool
            # is to be executed by AgentMasterLoop itself (e.g., agent:update_plan) OR
            # it's a UMS/Server tool that MCPClient didn't execute (e.g., due to config or if AML were to handle all).
            # In the V4.2 Direct MCP Names model, MCPClient executes UMS/Server tools, so this path
            # is PRIMARILY for AGENT_TOOL_UPDATE_PLAN.
            tool_name_to_execute_by_aml = llm_decision.get("tool_name") # Original MCP Name
            tool_name_involved_in_turn = tool_name_to_execute_by_aml # For logging
            arguments = llm_decision.get("arguments", {})

            self.logger.info(f"AML EXEC_DECISION: Decision is 'call_tool' (to be executed by AML). Tool: '{tool_name_to_execute_by_aml}', Args: {str(arguments)[:100]}...")

            if not self.state.current_plan:
                self.logger.error("AML EXEC_DECISION: Plan empty before AML tool call! Forcing replan.")
                self.state.last_error_details = {"tool": tool_name_to_execute_by_aml, "args": arguments, "error": "Plan empty before tool call.", "type": "PlanValidationError"}
                tool_result_content = {"success": False, "error": "Plan empty before tool call."} # Store for heuristic
            elif not self.state.current_plan[0].description:
                self.logger.error(f"AML EXEC_DECISION: Current plan step invalid (no description)! Step ID: {self.state.current_plan[0].id}. Forcing replan.")
                self.state.last_error_details = {"tool": tool_name_to_execute_by_aml, "args": arguments, "error": "Current plan step invalid (no description).", "type": "PlanValidationError", "step_id": self.state.current_plan[0].id}
                tool_result_content = {"success": False, "error": "Current plan step invalid."} # Store for heuristic
            elif tool_name_to_execute_by_aml:
                current_step_deps = self.state.current_plan[0].depends_on if self.state.current_plan else []
                # _execute_tool_call_internal handles AGENT_TOOL_UPDATE_PLAN internally,
                # and for other tools, it calls MCPClient (which shouldn't happen much here for V4.2 if MCPC executes UMS tools).
                tool_result_content = await self._execute_tool_call_internal(
                    tool_name_to_execute_by_aml, arguments,
                    record_action=(tool_name_to_execute_by_aml != AGENT_TOOL_UPDATE_PLAN), # Don't record UMS action for agent plan update
                    planned_dependencies=current_step_deps # Pass dependencies from current plan step
                )
            else: # Should not happen if MCPClient packaged decision correctly
                self.logger.error("AML EXEC_DECISION: LLM 'call_tool' (for AML exec) decision missing 'tool_name'.")
                self.state.last_error_details = {"decision_data": llm_decision, "error": "LLM 'call_tool' (for AML exec) decision missing 'tool_name'.", "type": "LLMOutputError"}
                tool_result_content = {"success": False, "error": "Missing tool name from LLM decision for AML exec."}

        elif decision_type == "thought_process":
            thought_content_text = llm_decision.get("content")
            tool_name_involved_in_turn = self._get_ums_tool_mcp_name(UMS_FUNC_RECORD_THOUGHT) # This is the UMS tool
            self.logger.info(f"AML EXEC_DECISION: Decision is 'thought_process'. Content: {str(thought_content_text)[:100]}...")
            if thought_content_text:
                # Call the UMS record_thought tool
                tool_result_content = await self._execute_tool_call_internal(
                    tool_name_involved_in_turn, # Full MCP name for UMS record_thought
                    {"content": str(thought_content_text), "thought_type": ThoughtType.INFERENCE.value}, # Default type
                    record_action=False # Recording thoughts is not a primary "action" towards goal in UMS terms
                )
            else:
                self.logger.warning("AML EXEC_DECISION: LLM 'thought_process' decision, but no content provided.")
                tool_result_content = {"success": False, "error": "Missing thought content from LLM for 'thought_process' decision."}

        elif decision_type == "complete":
            completion_summary = llm_decision.get('summary', 'Overall goal achieved based on LLM signal.')
            tool_name_involved_in_turn = "agent:signal_completion" # Agent-internal signal
            self.logger.info(f"AML EXEC_DECISION: LLM signaled OVERALL goal completion: {completion_summary}")
            
            # Find the UMS root goal for the current workflow to mark it as completed.
            root_goal_to_mark_id: Optional[str] = None
            if self.state.workflow_id and self.state.goal_stack: # Check if workflow and stack exist
                # The first goal in the agent's stack is assumed to be the root UMS goal for this workflow context.
                # Or iterate to find one with no parent_goal_id if structure is more complex.
                for g_dict_complete in self.state.goal_stack:
                    if isinstance(g_dict_complete, dict) and not g_dict_complete.get("parent_goal_id"):
                        root_goal_to_mark_id = g_dict_complete.get("goal_id")
                        break
                if not root_goal_to_mark_id and self.state.goal_stack: # Fallback if no explicit root, take current
                    root_goal_to_mark_id = self.state.current_goal_id
            
            update_goal_status_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_GOAL_STATUS)
            if root_goal_to_mark_id and self._find_tool_server(update_goal_status_mcp_name):
                self.logger.info(f"AML EXEC_DECISION: Marking UMS root goal {_fmt_id(root_goal_to_mark_id)} as completed based on LLM signal.")
                # This call will trigger side effects via _handle_workflow_and_goal_side_effects
                tool_result_content = await self._execute_tool_call_internal(
                    update_goal_status_mcp_name,
                    {"goal_id": root_goal_to_mark_id, "status": GoalStatus.COMPLETED.value, "reason": f"LLM signaled overall goal completion: {completion_summary}"},
                    record_action=False # Marking goal status is meta, not a direct UMS action for history
                )
                # _handle_workflow_and_goal_side_effects will set goal_achieved_flag if UMS confirms root finished.
            else:
                self.logger.warning(f"AML EXEC_DECISION: LLM signaled overall completion, but couldn't identify a root UMS goal on current stack, or UMS tool for '{UMS_FUNC_UPDATE_GOAL_STATUS}' unavailable. Manually setting goal_achieved_flag for agent.")
                self.state.goal_achieved_flag = True # Set agent's flag directly
                tool_result_content = {"success": True, "message": "Overall goal marked achieved by agent based on LLM signal."}
            
            # If goal_achieved_flag is true (either by UMS side effect or manually), agent will stop.
            if self.state.goal_achieved_flag:
                await self._save_agent_state() # Save before exiting
                self.logger.info(f"AML EXEC_DECISION: goal_achieved_flag is True. Returning False (stop).")
                return False # Signal stop

        elif decision_type == "plan_update":
            # This is for when MCPClient parsed a textual plan update from LLM
            tool_name_involved_in_turn = AGENT_TOOL_UPDATE_PLAN # Implies agent's own plan
            self.logger.info(f"AML EXEC_DECISION: Decision is 'plan_update' (from LLM text). Proposed steps count: {len(llm_proposed_plan_steps) if llm_proposed_plan_steps else 'None/Invalid'}")
            # Validation of llm_proposed_plan_steps (syntax, cycles) happens below in the heuristic section for this path.
            # No direct UMS tool call here, tool_result_content remains None for now.
            # The heuristic update section will handle applying this textual plan.

        elif decision_type == "error": # Error from MCPClient's processing of LLM response
            error_message_from_mcpc = llm_decision.get('message', 'Unknown error from LLM decision processing in MCPClient')
            tool_name_involved_in_turn = "agent:llm_decision_error" # Internal agent error
            self.logger.error(f"AML EXEC_DECISION: Received 'error' decision from MCPClient: {error_message_from_mcpc}")
            self.state.last_action_summary = f"LLM Decision Error (MCPClient): {error_message_from_mcpc[:100]}"
            # Set last_error_details if not already set by a more specific tool failure
            if not self.state.last_error_details:
                self.state.last_error_details = {"error": error_message_from_mcpc, "type": llm_decision.get("error_type_for_agent", "LLMError")}
            self.state.needs_replan = True
            tool_result_content = {"success": False, "error": error_message_from_mcpc} # For heuristic

        else: # Unhandled decision type from MCPClient
            tool_name_involved_in_turn = "agent:unknown_decision" # Internal agent error
            self.logger.error(f"AML EXEC_DECISION: Unexpected decision type from MCPClient: '{decision_type}'. Full decision: {str(llm_decision)[:200]}")
            self.state.last_action_summary = f"Agent Error: Unexpected decision type '{decision_type}' from MCPClient"
            self.state.last_error_details = {"error": f"Unexpected decision type '{decision_type}' from MCPClient.", "type": "AgentError", "llm_decision_payload": llm_decision}
            self.state.needs_replan = True
            tool_result_content = {"success": False, "error": f"Unknown decision type from MCPClient: {decision_type}"}


        # --- Apply Plan Updates (if LLM proposed one via text) & Heuristic Update ---
        # This logic block handles:
        # 1. Applying a plan that was extracted from LLM's text response by MCPClient (decision_type == "plan_update")
        # 2. Applying heuristic updates for other decision types.

        if decision_type == "plan_update" and llm_proposed_plan_steps:
            # This path is specifically for when MCPClient sends "plan_update" with parsed steps.
            self.logger.info(f"AML EXEC_DECISION: Attempting to apply LLM-proposed textual plan ({len(llm_proposed_plan_steps)} steps).")
            try:
                if self._detect_plan_cycle(llm_proposed_plan_steps):
                    err_msg = "AML EXEC_DECISION: LLM-proposed plan (via text) has dependency cycle. Plan not applied."
                    self.logger.error(err_msg)
                    self.state.last_error_details = {"error": err_msg, "type": "PlanValidationError", "proposed_plan": [p.model_dump(exclude_none=True) for p in llm_proposed_plan_steps]}
                    self.state.needs_replan = True
                    # Since plan application failed, tool_result_content for heuristic update should reflect this.
                    tool_result_content = {"success": False, "error": "Plan cycle in LLM text proposal (not applied)."}
                    await self._apply_heuristic_plan_update(llm_decision, tool_result_content) # Fallback to heuristic
                else:
                    self.state.current_plan = llm_proposed_plan_steps
                    self.state.needs_replan = False # Plan was successfully updated textually
                    self.logger.info(f"AML EXEC_DECISION: Applied LLM-proposed textual plan update ({len(llm_proposed_plan_steps)} steps). NeedsReplan set to False.")
                    self.state.last_error_details = None # Clear errors if plan was successfully updated
                    self.state.consecutive_error_count = 0
                    # No heuristic update needed if plan was directly set by LLM text.
                    # tool_result_content for plan_update decision is effectively the plan itself, which is now applied.
            except Exception as plan_apply_err:
                self.logger.error(f"AML EXEC_DECISION: Error validating/applying LLM proposed textual plan: {plan_apply_err}. Fallback to heuristic.", exc_info=True)
                self.state.last_error_details = {"error": f"Failed to apply LLM textual plan: {plan_apply_err}", "type": "PlanUpdateError", "proposed_plan_data": llm_proposed_plan_steps_data}
                self.state.needs_replan = True
                tool_result_content = {"success": False, "error": f"LLM textual plan application error: {plan_apply_err}"}
                await self._apply_heuristic_plan_update(llm_decision, tool_result_content) # Fallback
        
        # Apply heuristic if conditions met:
        # - Not a "plan_update" where the plan was successfully applied (needs_replan is False)
        # - Not a "call_tool" for AGENT_TOOL_UPDATE_PLAN where the tool call was successful (tool_result_content["success"] is True)
        elif not (decision_type == "plan_update" and not self.state.needs_replan) and \
             not (decision_type == "call_tool" and tool_name_involved_in_turn == AGENT_TOOL_UPDATE_PLAN and isinstance(tool_result_content, dict) and tool_result_content.get("success")):
            self.logger.debug(f"AML EXEC_DECISION: Conditions met for heuristic plan update. Decision: '{decision_type}', Tool: '{tool_name_involved_in_turn}', NeedsReplan: {self.state.needs_replan}, ToolResultSuccess: {tool_result_content.get('success') if isinstance(tool_result_content,dict) else 'N/A'}")
            await self._apply_heuristic_plan_update(llm_decision, tool_result_content)
        
        elif decision_type == "call_tool" and tool_name_involved_in_turn == AGENT_TOOL_UPDATE_PLAN:
            # This path is if LLM *explicitly called* the agent:update_plan tool.
            # _execute_tool_call_internal handles success/failure and setting needs_replan.
            self.logger.info(f"AML EXEC_DECISION: LLM called agent:update_plan tool. Heuristic update skipped. Needs replan state: {self.state.needs_replan}")
            if isinstance(tool_result_content, dict) and not tool_result_content.get("success", False):
                 self.logger.warning(f"AML EXEC_DECISION: agent:update_plan tool call itself failed. Error details should be set by _execute_tool_call_internal.")
        
        # Max consecutive error check
        if self.state.consecutive_error_count >= MAX_CONSECUTIVE_ERRORS:
            self.logger.critical(f"AML EXEC_DECISION: Max consecutive errors ({self.state.consecutive_error_count}/{MAX_CONSECUTIVE_ERRORS}) reached. Signaling stop.")
            update_wf_status_mcp_name = self._get_ums_tool_mcp_name(UMS_FUNC_UPDATE_WORKFLOW_STATUS)
            if self.state.workflow_id and self._find_tool_server(update_wf_status_mcp_name):
                # Call UMS to mark workflow as failed
                await self._execute_tool_call_internal(
                    update_wf_status_mcp_name,
                    {"workflow_id": self.state.workflow_id, "status": WorkflowStatus.FAILED.value, "completion_message": f"Aborted after {self.state.consecutive_error_count} consecutive errors."},
                    record_action=False # Not a primary workflow action
                )
            await self._save_agent_state()
            self.logger.info(f"AML EXEC_DECISION: Returning False (max errors). WF: {_fmt_id(self.state.workflow_id)}, Goal: {_fmt_id(self.state.current_goal_id)}")
            return False # Stop

        self.logger.info(f"AML EXEC_DECISION: State BEFORE _save_agent_state: WF ID='{_fmt_id(self.state.workflow_id)}', Goal ID='{_fmt_id(self.state.current_goal_id)}', needs_replan={self.state.needs_replan}, errors={self.state.consecutive_error_count}")
        await self._save_agent_state()
        self.logger.info(f"AML EXEC_DECISION: State AFTER _save_agent_state: WF ID='{_fmt_id(self.state.workflow_id)}', Goal ID='{_fmt_id(self.state.current_goal_id)}', needs_replan={self.state.needs_replan}, errors={self.state.consecutive_error_count}")

        # Final check for workflow validity before returning continue signal
        if not self.state.workflow_id and not self.state.goal_achieved_flag and not self._shutdown_event.is_set():
            self.logger.warning(f"AML EXEC_DECISION: self.state.workflow_id is None before final check. This can happen if create_workflow failed or side effects didn't set it. Agent will stop.")
            # Attempt recovery from temp file, although if create_workflow failed, this might not help.
            recovered_wf_id = await self._read_temp_workflow_id()
            if recovered_wf_id:
                if await self._check_workflow_exists(recovered_wf_id):
                    self.state.workflow_id = recovered_wf_id
                    if not self.state.context_id: self.state.context_id = recovered_wf_id
                    self.logger.info(f"AML EXEC_DECISION: RECOVERED workflow_id '{_fmt_id(self.state.workflow_id)}' from temp file and validated it against UMS. Agent might continue if other conditions allow.")
                    # If recovered, we might need to re-establish root UMS goal etc., but for now, let's see if this is enough.
                else:
                    self.logger.error(f"AML EXEC_DECISION: Recovered workflow_id '{_fmt_id(recovered_wf_id)}' from temp file, but it's NOT VALID in UMS. Clearing temp file. Agent will stop.")
                    await self._write_temp_workflow_id(None)
            else:
                self.logger.warning(f"AML EXEC_DECISION: self.state.workflow_id is None and could not recover from temp file. Agent will stop.")

        self.logger.info(
            f"AML EXEC_DECISION PRE-RETURN CHECK: "
            f"goal_achieved={self.state.goal_achieved_flag}, "
            f"shutdown_event={self._shutdown_event.is_set()}, "
            f"workflow_id='{_fmt_id(self.state.workflow_id)}' (is None: {self.state.workflow_id is None}), "
            f"consecutive_errors={self.state.consecutive_error_count}"
        )

        # Determine if the loop should continue
        if self.state.goal_achieved_flag or self._shutdown_event.is_set() or not self.state.workflow_id:
            stop_reason_log = f"goal_achieved={self.state.goal_achieved_flag}, shutdown={self._shutdown_event.is_set()}, no_workflow_id={not self.state.workflow_id}"
            self.logger.info(f"AML EXEC_DECISION: Returning False (stop condition met due to: {stop_reason_log}).")
            if not self.state.workflow_id and not self.state.goal_achieved_flag and not self._shutdown_event.is_set():
                 self.logger.warning(f"AML EXEC_DECISION: Stopping because workflow_id is None. Clearing temp workflow file as a precaution.")
                 await self._write_temp_workflow_id(None) # Ensure temp file is cleared if stopping due to no WF
            return False # Stop

        self.logger.info(f"AML EXEC_DECISION: Returning True (continue). WF ID='{_fmt_id(self.state.workflow_id)}'")
        return True # Continue

    async def run_main_loop(self, initial_goal_for_this_run: str, max_loops_from_mcpc: int = 100):
        # 1. Increment agent's internal loop counter for THIS turn being processed
        # self.state.current_loop tracks completed turns. This increments for the turn about to be processed.
        # If current_loop is 0 (initial), this turn becomes 1.
        self.state.current_loop += 1 
        current_turn_for_log = self.state.current_loop # For logging this specific turn
        self.logger.info(f"AgentMasterLoop.run_main_loop: Starting TURN {current_turn_for_log}. Max loops (MCPC): {max_loops_from_mcpc}")

        # 2. Check stopping conditions BEFORE any processing for this turn
        if self.state.current_loop > max_loops_from_mcpc: # Check if *this turn* would exceed max
            self.logger.warning(f"AML: Agent loop ({current_turn_for_log}) would exceed max_loops_from_mcpc ({max_loops_from_mcpc}). Signaling stop.")
            # Decrement loop counter as this turn is not actually processed
            self.state.current_loop -=1 
            return None 

        if self.state.goal_achieved_flag:
            self.logger.info(f"AML: Goal previously achieved. Loop {current_turn_for_log} will not run further. Signaling stop.")
            self.state.current_loop -=1 # This turn wasn't processed
            return None

        if self._shutdown_event.is_set():
            self.logger.info(f"AML: Shutdown signaled. Loop {current_turn_for_log} will not run. Signaling stop.")
            self.state.current_loop -=1 # This turn wasn't processed
            return None

        # 3. Initial Workflow/Goal Setup Logic (if it's the very first turn *for a new task* or needs re-establishment)
        #    This happens *after* loop counter increment and stop checks.
        #    The `initial_goal_for_this_run` is crucial here.
        if not self.state.workflow_id:
            self.logger.info(f"AML (Turn {current_turn_for_log}): No active workflow ID in state. Agent will be prompted to create one using goal: '{initial_goal_for_this_run[:70]}...'")
            self.state.current_plan = [PlanStep(description=f"Establish UMS workflow for task: {initial_goal_for_this_run[:70]}...")]
            self.state.goal_stack = []
            self.state.current_goal_id = None
            self.state.current_thought_chain_id = None
            self.state.needs_replan = False
            self.logger.info(f"AML (Turn {current_turn_for_log}): Initial plan set for new task.")
        elif not self.state.current_thought_chain_id: # Workflow exists, but no thought chain
            self.logger.info(f"AML (Turn {current_turn_for_log}): Active workflow {_fmt_id(self.state.workflow_id)}, but no thought chain. Setting default.")
            await self._set_default_thought_chain_id()
        elif not self.state.current_goal_id and self.state.workflow_id: # Workflow exists, but no current UMS goal
             # This state implies the LLM needs to establish the root UMS goal for the existing workflow.
             # This is handled by the prompt logic in _construct_agent_prompt.
             self.logger.info(f"AML (Turn {current_turn_for_log}): Workflow {_fmt_id(self.state.workflow_id)} active, but no current UMS goal. LLM will be prompted to establish root UMS goal using overall workflow goal.")
             if not self.state.goal_stack: # If stack is also empty, ensure plan reflects this.
                self.state.current_plan = [PlanStep(description=f"Establish root UMS goal for existing workflow: {_fmt_id(self.state.workflow_id)}")]
                self.state.needs_replan = False # New plan set
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
            self.state.last_error_details = {
                "error": f"Periodic task failure: {str(e_periodic)[:100]}",
                "type": "PeriodicTaskError"
            }
            self.state.needs_replan = True # Assume replan needed if periodic tasks fail critically.
            # Continue to context gathering so LLM sees the error.

        if self._shutdown_event.is_set(): # Re-check after periodic tasks
            self.logger.warning(f"AML (Turn {current_turn_for_log}): Shutdown signaled during/after periodic tasks. Signaling stop.")
            self.state.current_loop -=1 # This turn wasn't fully processed
            return None

        # 5. Gather Context
        self.logger.info(f"AML (Turn {current_turn_for_log}): Calling _gather_context...")
        agent_context_snapshot = await self._gather_context()
        self.logger.info(f"AML (Turn {current_turn_for_log}): _gather_context returned. Status from context: {agent_context_snapshot.get('status_message_from_agent')}")

        # 6. Handle Error State for Prompt
        # Clear last_error_details ONLY if no replan is needed AND workflow is active.
        # If periodic tasks set needs_replan=True, this won't clear it.
        if not self.state.needs_replan and self.state.workflow_id:
            self.state.last_error_details = None # Cleared if no replan AND workflow active
            self.logger.debug(f"AML (Turn {current_turn_for_log}): Cleared last_error_details (no replan needed, WF active).")
        elif self.state.needs_replan:
            self.logger.info(f"AML (Turn {current_turn_for_log}): needs_replan is True. Preserving last_error_details for LLM.")
        elif not self.state.workflow_id:
            self.logger.debug(f"AML (Turn {current_turn_for_log}): No active workflow, last_error_details preserved if any (likely None here).")


        # 7. Determine Goal for Prompt Construction
        prompt_goal_to_use = initial_goal_for_this_run # Default to the overall goal for this run
        if self.state.current_goal_id: # If a specific UMS operational goal is active
            current_op_goal_details_from_ctx = agent_context_snapshot.get("agent_assembled_goal_context", {}).get("current_goal_details_from_ums")
            if current_op_goal_details_from_ctx and isinstance(current_op_goal_details_from_ctx, dict) and current_op_goal_details_from_ctx.get("goal_id") == self.state.current_goal_id:
                prompt_goal_to_use = current_op_goal_details_from_ctx.get("description", initial_goal_for_this_run)
                self.logger.info(f"AML (Turn {current_turn_for_log}): Using current operational UMS goal for prompt: '{prompt_goal_to_use[:50]}...' (ID: {_fmt_id(self.state.current_goal_id)})")
            else:
                self.logger.warning(f"AML (Turn {current_turn_for_log}): Current UMS goal ID {_fmt_id(self.state.current_goal_id)} set, but details not in gathered context. Using overall run goal for prompt.")
        elif self.state.workflow_id: # Workflow active, but no specific current_goal_id (e.g. root goal needs to be set)
            ums_workflow_goal_from_ctx = agent_context_snapshot.get('ums_context_package', {}).get('core_context', {}).get('workflow_goal')
            if ums_workflow_goal_from_ctx:
                prompt_goal_to_use = ums_workflow_goal_from_ctx
                self.logger.info(f"AML (Turn {current_turn_for_log}): Workflow active, no specific UMS goal. Using UMS Workflow Goal from context for prompt: '{prompt_goal_to_use[:50]}...'")
            else: # Fallback if UMS workflow goal also not in context (should be rare if workflow exists)
                self.logger.info(f"AML (Turn {current_turn_for_log}): Workflow active, no specific UMS goal and UMS workflow goal not in context. Using initial_goal_for_this_run for prompt.")
        else: # No workflow ID yet (first turn for this task)
            self.logger.info(f"AML (Turn {current_turn_for_log}): No workflow. Using initial_goal_for_this_run for prompt: '{initial_goal_for_this_run[:50]}...'")


        # 8. Construct Prompt
        # _construct_agent_prompt now uses self.state.current_loop directly for logging current turn
        self.logger.info(f"AML (Turn {current_turn_for_log}): Calling _construct_agent_prompt with prompt_goal_to_use: '{prompt_goal_to_use[:70]}...'")
        prompt_messages_for_llm = self._construct_agent_prompt(prompt_goal_to_use, agent_context_snapshot)

        # 9. Final check before returning data to MCPClient
        if self.state.goal_achieved_flag or self._shutdown_event.is_set(): # Re-check after all processing this turn
            self.logger.info(f"AML (Turn {current_turn_for_log}): Signaling stop to MCPClient post-prepare. GoalAchieved={self.state.goal_achieved_flag}, Shutdown={self._shutdown_event.is_set()}")
            # Decrement loop counter as this turn's data won't be used by LLM
            self.state.current_loop -= 1 
            return None

        self.logger.info(f"AML (Turn {current_turn_for_log}): Data prepared for MCPClient. Prompt messages count: {len(prompt_messages_for_llm)}")
        return {"prompt_messages": prompt_messages_for_llm, "tool_schemas": self.tool_schemas, "agent_context": agent_context_snapshot}
